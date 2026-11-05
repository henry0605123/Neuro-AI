"""Source normalization: make both arms' transcripts look alike before any feature sees them.

Why this exists. The ASD and TD transcripts were produced by different pipelines, and the
2026-09-30 health check measured how much that matters on TD alone -- same audio, same
words, only the transcript's conventions changed:

* punctuation (0.61 marks per segment in TD, 0.04 in ASD) moved G1 by 0.89 SD;
* merging same-speaker segments separated by <= 1.5 s (the RuleStitch rule the ASD
  transcripts went through) moved utterance length and pause rate by 1.4-1.6 SD;
* stitching the timeline at speaker changes -- 36-92% of gaps are exactly 0.000 s in the
  later pipelines -- turned response latency into a measure of the stitching.

31 of 98 features moved by >= 0.5 SD, against a minimum detectable group effect of about
0.9. A classifier would not need to recognise autism, only which team made the file.

The remedy is not to correct one arm towards the other but to put **both** through the same
three steps, so that any convention either pipeline applied is either removed or applied to
everything:

1. **Strip punctuation.** Neither arm keeps it, so nothing downstream can read it.
2. **Merge same-speaker segments separated by <= 1.5 s.** This *is* the RuleStitch rule, so
   the ASD transcripts are unchanged by it and the TD transcripts become what RuleStitch
   would have produced. The step is idempotent: running it on its own output changes
   nothing, which is what makes the merge transform a no-op after normalization.
3. **Trim each segment's edges to where the audio is actually voiced.** Timeline stitching
   extends a turn back into the preceding silence; trimming removes that silence again.
   Trimming only ever shrinks a span, never extends it, so it cannot pull in a neighbouring
   speaker's audio.

What this cannot fix, and does not pretend to: a speaker's voice bleeding across a turn
boundary (the energy detector cannot tell speakers apart), and codec differences in the
audio itself (health check P5), which needs the original recordings.

The gate in ``configs/decision_rules.yaml: convention_invariance_gate`` measures whatever
convention sensitivity is left after this runs.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field, replace
from typing import Mapping, Sequence

import numpy as np

from .contracts import Corpus, Turn
from .features.common import frame_energy, noise_floor_energy

__all__ = ["NormalizationReport", "strip_punctuation", "merge_same_speaker", "trim_to_voice",
           "normalize_turns", "normalize_corpus", "NORMALIZE_VERSION"]

# Bumped whenever the output of normalize_turns can change for the same input. The
# extraction cache hashes the normalized turns, so this is belt and braces rather than the
# only guard -- but the alignment cache keys on turn boundaries and would otherwise serve
# alignments made for different spans.
NORMALIZE_VERSION = 2

_ENERGY_FRAME = 512                 # frame_energy's block size; read, not restated as ms
_EXTRA_PUNCT = frozenset("～~·•")    # math/other symbols used as punctuation in transcripts


@dataclass
class NormalizationReport:
    subject_id: str
    n_in: int = 0
    n_out: int = 0
    n_merged: int = 0
    n_trimmed: int = 0
    n_no_voice_found: int = 0
    seconds_trimmed: float = 0.0
    notes: list[str] = field(default_factory=list)


# ------------------------------------------------------------------------------ text
def _is_punct(ch: str) -> bool:
    cat = unicodedata.category(ch)
    return cat.startswith("P") or cat.startswith("Z") or ch.isspace() or ch in _EXTRA_PUNCT


def strip_punctuation(text: str) -> str:
    """Remove every punctuation and separator character, keep everything else."""
    return "".join(ch for ch in text if not _is_punct(ch))


# ------------------------------------------------------------------------------ structure
def merge_same_speaker(turns: Sequence[Turn], max_gap_s: float) -> list[Turn]:
    """Greedy left-to-right merge of adjacent same-speaker turns with gap <= max_gap_s.

    The merged turn takes the question flags of its **last** member: a reply responds to
    the last thing said, so a block ending "...那你今天做什麼" is an open question even if it
    began with a yes/no one.
    """
    out: list[Turn] = []
    for t in sorted(turns, key=lambda x: (x.t_start, x.t_end)):
        if out and out[-1].speaker is t.speaker and t.t_start - out[-1].t_end <= max_gap_s:
            prev = out[-1]
            out[-1] = replace(
                prev,
                t_end=max(prev.t_end, t.t_end),
                text=prev.text + t.text,
                is_question=t.is_question,
                is_yes_no=t.is_yes_no,
                unintelligible=prev.unintelligible and t.unintelligible,
                syllables=(),
            )
        else:
            out.append(t)
    return out


# ------------------------------------------------------------------------------ timing
def _voiced_runs(mask: np.ndarray, min_frames: int) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, v in enumerate(np.append(mask, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_frames:
                runs.append((start, i))
            start = None
    return runs


def trim_to_voice(turns: Sequence[Turn], audio: np.ndarray, sr: int, *,
                  over_floor_db: float, min_run_s: float, min_keep_s: float,
                  search_back_s: float, report: NormalizationReport | None = None
                  ) -> list[Turn]:
    """Place each turn's edges on the voice, identically whatever the annotator did.

    **End**: shrink to the offset of the last voiced run inside the turn. Never extended --
    no convention in use moves ends, and extending would reach into the next speaker.

    **Start**: the onset of the first voiced run found searching forward from
    ``max(previous turn's end, own start - search_back_s)`` -- not from the turn's own
    start. This is the step that makes timeline stitching disappear. Stitching only ever
    moves a start back to the previous turn's end, so both versions of a turn search from
    the same point and land on the same onset. A shrink-only trim cannot do that: an
    annotator who marked a start slightly late keeps that late start, while the stitched
    copy of the same turn gets trimmed to the true onset, and the two disagree (measured:
    a residual 0.128 s in the median speaker-change gap on TD).

    ``search_back_s`` must be at least the widest gap any pipeline stitched over, and it
    bounds how far a turn can grow into a long silence if a noise burst sits in it.

    Voicing is frame energy above the *session's* noise floor, sustained for ``min_run_s``
    so a click is not an onset. A turn with no such run keeps its span and is counted:
    a quiet child's few turns must not be discarded by a detector tuned on louder ones.
    """
    e = frame_energy(audio, _ENERGY_FRAME)
    hop_s = _ENERGY_FRAME / sr
    thresh = noise_floor_energy(audio) * (10.0 ** (over_floor_db / 10.0))
    min_frames = max(1, int(round(min_run_s / hop_s)))

    ordered = sorted(turns, key=lambda x: (x.t_start, x.t_end))
    out: list[Turn] = []
    prev_end: float | None = None
    for t in ordered:
        lo_s = t.t_start - search_back_s
        if prev_end is not None:
            lo_s = max(lo_s, min(prev_end, t.t_start))
        lo_s = max(0.0, lo_s)
        a = max(0, int(np.floor(lo_s / hop_s)))
        b = min(len(e), int(np.ceil(t.t_end / hop_s)))
        runs = _voiced_runs(e[a:b] > thresh, min_frames) if b > a else []
        own_end = t.t_end
        prev_end = own_end if prev_end is None else max(prev_end, own_end)
        if not runs:
            out.append(t)
            if report is not None:
                report.n_no_voice_found += 1
            continue
        new_start = max(lo_s, (a + runs[0][0]) * hop_s)
        new_end = min(own_end, (a + runs[-1][1]) * hop_s)
        if new_end - new_start < min_keep_s:
            out.append(t)
            if report is not None:
                report.n_no_voice_found += 1
            continue
        if abs(new_start - t.t_start) > 1e-9 or new_end < own_end:
            if report is not None:
                report.n_trimmed += 1
                report.seconds_trimmed += (new_start - t.t_start) + (own_end - new_end)
            t = replace(t, t_start=new_start, t_end=new_end, syllables=())
        out.append(t)
    return out


# ------------------------------------------------------------------------------ pipeline
def normalize_turns(turns: Sequence[Turn], audio: np.ndarray, sr: int, cfg: Mapping,
                    subject_id: str = "") -> tuple[list[Turn], NormalizationReport]:
    n = cfg["realdata"]["normalize"]
    rep = NormalizationReport(subject_id=subject_id, n_in=len(turns))
    stripped = [replace(t, text=strip_punctuation(t.text)) for t in turns]
    merged = merge_same_speaker(stripped, float(n["merge_gap_s"]))
    rep.n_merged = len(stripped) - len(merged)
    trimmed = trim_to_voice(merged, audio, sr,
                            over_floor_db=float(n["voice_over_floor_db"]),
                            min_run_s=float(n["voice_min_run_s"]),
                            min_keep_s=float(n["voice_min_keep_s"]),
                            search_back_s=float(n["voice_search_back_s"]),
                            report=rep)
    rep.n_out = len(trimmed)
    return trimmed, rep


def normalize_corpus(corpus: Corpus, cfg: Mapping
                     ) -> tuple[Corpus, list[NormalizationReport]]:
    """Apply `normalize_turns` to every subject. Identical steps for every cohort."""
    subjects, reports = [], []
    for s in corpus.subjects:
        turns, rep = normalize_turns(s.session.turns, s.audio, s.sample_rate, cfg,
                                     subject_id=s.subject_id)
        subjects.append(replace(s, session=replace(s.session, turns=tuple(turns))))
        reports.append(rep)
    return replace(corpus, subjects=tuple(subjects)), reports
