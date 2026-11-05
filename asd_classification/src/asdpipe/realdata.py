"""Adapter from the clinic's transcript export to the pipeline's data contract.

This is the only module that knows what the raw files look like. Everything downstream
sees `Session` / `Turn` exactly as it does for the synthetic corpus, so the extractors,
the CV machinery and the audit gates are shared code, not a parallel implementation.

Three things this module decides, all of them frozen in `configs/pipeline.yaml` and all of
them audit requirements rather than taste:

* **speaker role normalisation** (audit P0-2). The export uses four different vocabularies
  for the same two roles -- 評估者 / 測試者 / SPEAKER_00 for the examiner and
  兒童 / 小孩 / SPEAKER_01 for the child -- plus a handful of third parties. The mapping is
  a table, applied identically to every case; it is never adjusted per file.
* **overlap policy** (audit P0-2 requires one strategy applied to every group). Overlaps in
  this cohort are real speech, not rounding: the median is 367 ms and the 90th percentile
  is 2 s. The default drops *both* segments of an overlapping pair, because the audio in
  the overlap region contains two voices and would contaminate the acoustic features of
  either one.
* **third-party turns** (個管師 / 家長 / 廣播 / Unknown) are dropped.

Both policies are configurable and both report exactly what they cost, because "how much
data did the cleaning rule remove, and did it remove more from one group than the other"
is itself an audit question (audit P0-4: missing-data patterns can leak the label).

No file from the source directory is ever copied into the repository. Decoded audio is
cached under `cache/`, which is git-ignored.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .config import REPO_ROOT
from .contracts import Corpus, Label, Session, Speaker, SubjectRecord, Turn

__all__ = [
    "SPEAKER_ROLES", "THIRD_PARTY_HINTS", "SILENT_MARKER", "UNINTELLIGIBLE_MARKER",
    "EXCLUDED_ID_MAP", "IntakeReport", "CaseFiles",
    "resolve_excluded", "discover_cases", "load_transcript", "load_subject", "load_cohort",
    "is_question", "is_yes_no_question", "normalise_speaker",
]

# Frozen role table (audit P0-2). Anything not listed here is treated as a third party.
#
# The English labels came from the ASD arm (2026-09-12), whose transcripts were produced by
# a different tool and use Child/Tester where the TD transcripts use 兒童/測試者. Without
# them every ASD turn was classified as a third party and dropped, which surfaces as "0
# child turns" rather than as an error -- the same shape of silent failure as `is_question`
# being hard-coded to examiner-only. Matching stays exact and case-sensitive on purpose: an
# unrecognised label should show up in the intake report's unmapped count, not be guessed at.
SPEAKER_ROLES: dict[str, Speaker] = {
    "評估者": Speaker.EXAMINER,
    "測試者": Speaker.EXAMINER,
    "施測者": Speaker.EXAMINER,
    "SPEAKER_00": Speaker.EXAMINER,
    "Tester": Speaker.EXAMINER,
    "兒童": Speaker.CHILD,
    "小孩": Speaker.CHILD,
    "個案": Speaker.CHILD,
    "SPEAKER_01": Speaker.CHILD,
    "Child": Speaker.CHILD,
}

# Recorded for the intake report only; membership is decided by absence from SPEAKER_ROLES.
THIRD_PARTY_HINTS: frozenset[str] = frozenset({"個管師", "家長", "廣播", "Unknown", "老師"})

# Annotation markers introduced with the ASD arm (2026-09-12). The TD transcripts contain
# neither, which is exactly why they need explicit handling: left as ordinary text they are
# *invisible damage* rather than an error.
#
# [SILENT] marks a stretch with no speech. There are 375 of them across the four transcribed
# ASD cases, averaging 7.8 s -- 48.6 minutes of "speech" that is silence, 332 of it charged
# to the examiner. Treated as text it inflated examiner turn duration from 2.84 s to 4.26 s
# and made the two cohorts look like they had been segmented under different conventions.
# It is not a turn and is dropped.
#
# [UNINTELLIGIBLE] marks a vocalisation that could not be transcribed: 35 of the 37 are the
# child's. Dropping it would be wrong in the opposite direction -- the child *did* vocalise,
# and for a minimally verbal child that may be most of what there is. It is kept as a turn
# with no text: it counts towards vocalisation rate, and is excluded from everything that
# needs words (embeddings, word segmentation, forced alignment).
SILENT_MARKER = "[SILENT]"
UNINTELLIGIBLE_MARKER = "[UNINTELLIGIBLE]"

# Silence and unintelligibility are each marked through three independent channels in the
# ASD export, and only one of them is the text. Handling the text alone left 53 segments
# labelled with the speaker "Silence" to be counted as an unrecognised speaker, i.e. as a
# data problem rather than as silence.
SILENCE_SPEAKER_LABELS = frozenset({"Silence", "silence", "靜音"})
_ANNOTATION_SILENT = frozenset({"silent", "silence"})
_ANNOTATION_UNINTELLIGIBLE = frozenset({"unintelligible", "unclear"})

# Yes/no question patterns. Audit P0-F1 requires these to be excluded from the G1
# relevance pairing: a yes/no answer is naturally dissimilar to its prompt, so leaving them
# in makes every child look off-topic.
_YES_NO_PATTERNS = (
    "是不是", "要不要", "有沒有", "會不會", "好不好", "對不對", "能不能", "可不可以",
    "行不行", "想不想", "喜不喜歡", "知不知道", "記不記得", "敢不敢", "肯不肯",
)
_YES_NO_TAIL = re.compile(r"(嗎|吧)\s*[?？]?\s*$")
_QUESTION_MARK = re.compile(r"[?？]")
# Alternative ("A 還是 B") questions. They carry no 嗎/吧 and no wh-word, so before this the
# only thing marking them was the question mark -- which the ASD transcripts do not have.
_A_OR_B = re.compile(r"[一-鿿]{1,12}還是[一-鿿]{1,12}")
_QUESTION_WORDS = (
    "什麼", "甚麼", "怎麼", "為什麼", "哪裡", "哪一", "哪個", "誰", "幾", "多少",
    "如何", "怎樣", "為何", "幹嘛",
)
# Question forms added 2026-10-15 (DEVIATIONS D15). X不X / X沒X anywhere ("要不要",
# "吃不吃", "有沒有"), but not a run of negations ("不不不").
_A_NOT_A = re.compile(r"([一-鿿])[不沒]\1")
_FINAL_QUESTION_PARTICLE = re.compile(r"[呢勒咧]$")


def _bare(text: str) -> str:
    """Text without punctuation or spaces -- what the ASD transcripts look like."""
    return re.sub(r"[^\w]", "", text)


def _has_a_not_a(text: str) -> bool:
    return any(m.group(1) not in "不沒" for m in _A_NOT_A.finditer(text))


def normalise_speaker(raw: str | None, mapping: Mapping[str, str] | None = None) -> Speaker | None:
    """Map a raw speaker label to a role. ``None`` means "third party, drop it"."""
    if raw is None:
        return None
    label = str(raw).strip()
    if mapping and label in mapping:                      # metadata.speaker_mapping
        label = str(mapping[label]).strip()
    return SPEAKER_ROLES.get(label)


def is_question(text: str) -> bool:
    """Punctuation-blind on purpose (2026-09-12).

    The two cohorts are punctuated completely differently: the TD transcripts carry 0.61
    punctuation marks per segment and 1,352 question marks; the ASD transcripts carry 0.04
    and 9. Reading the question mark would therefore make question detection *systematically
    more sensitive for one group*, and every feature built on question detection -- QA
    pairing, G1's open-pair selection, the yes/no exclusion, initiation rate -- would inherit
    a cohort difference that is a property of the annotator.

    So the mark is ignored and only sentence form is used.

    Revised 2026-10-15 (DEVIATIONS D15) against the TD transcribers' own question marks.
    Those marks cover only about half of all questions -- 45% of TD examiner turns ending
    in 嗎 carry one -- so they can show what the rule misses, not what it over-calls. A form
    was therefore judged by how often its turns are marked compared with that 45%:

    * kept or added near it: 嗎 anywhere (46%; catches a question followed by more talk),
      X不X / X沒X (47%), 幹嘛 (44%), A還是B (43%), wh-words (40%), final 呢/勒/咧 (32%);
    * dropped far below it: final 吧 (2%) -- "我們來玩吧" is a suggestion, not a question;
      final 喔 (4%) and 啊 (17%) were never in.
    """
    t = _bare(text)
    if "嗎" in t or _has_a_not_a(t) or _FINAL_QUESTION_PARTICLE.search(t):
        return True
    if _A_OR_B.search(t):
        return True
    return any(w in t for w in _QUESTION_WORDS) and len(t) <= 40


def is_yes_no_question(text: str) -> bool:
    """A-not-A constructions and 嗎/吧 tags (audit P0-F1). Since 2026-10-15 any X不X /
    X沒X counts, not only the frozen list: is_question now detects them all, and one it
    detects but this function missed would be scored by G1 as an open question."""
    if any(p in text for p in _YES_NO_PATTERNS) or _has_a_not_a(_bare(text)):
        return True
    return bool(_YES_NO_TAIL.search(text))


@dataclass
class IntakeReport:
    """Per-case audit counters. Feeds the P0-0A / P0-1 / P0-2 gates."""

    case_id: str
    n_raw: int = 0
    n_kept: int = 0
    n_third_party: int = 0
    n_unmapped_labels: int = 0
    n_empty_text: int = 0
    n_silent_marker: int = 0
    silent_marker_s: float = 0.0
    n_unintelligible: int = 0
    n_out_of_order: int = 0
    n_overlap_pairs: int = 0
    n_dropped_overlap: int = 0
    overlap_seconds: float = 0.0
    n_examiner: int = 0
    n_child: int = 0
    child_speech_s: float = 0.0
    examiner_speech_s: float = 0.0
    session_s: float = 0.0
    n_qa_pairs: int = 0
    n_open_qa_pairs: int = 0
    n_examiner_questions: int = 0
    n_examiner_yes_no: int = 0
    n_question_marks: int = 0
    speaker_vocabulary: dict[str, int] = field(default_factory=dict)
    third_party_labels: dict[str, int] = field(default_factory=dict)
    audio_path: str = ""
    transcript_path: str = ""
    label_is_placeholder: bool = True
    errors: list[str] = field(default_factory=list)
    """Conditions that make the case unusable rather than merely imperfect. Kept separate
    from the counters so a caller can refuse a case without having to know which counter
    combination means 'broken'."""

    @property
    def usable(self) -> bool:
        return not self.errors
    notes: list[str] = field(default_factory=list)

    @property
    def kept_ratio(self) -> float:
        return self.n_kept / self.n_raw if self.n_raw else 0.0

    @property
    def child_talk_ratio(self) -> float:
        total = self.child_speech_s + self.examiner_speech_s
        return self.child_speech_s / total if total else 0.0


@dataclass(frozen=True)
class CaseFiles:
    case_id: str
    directory: Path
    audio: Path
    transcript: Path
    # Every part, in order, when one session was saved as several files (health check
    # P25). Empty for the usual single file; `audio` is then the whole session.
    audio_parts: tuple[Path, ...] = ()


# When a session exists in several containers, take the one every session shares. All 23
# camera MP4s on disk (21 ASD files, 2 TD) carry the same audio track -- AAC, 48 kHz,
# stereo, 127 kbps -- and both arms' MP3s were converted from them with the same tool (VLC)
# to the same spec: 44.1 kHz, stereo, 128 kbps (ffprobe, 2026-10-01). 19 of the 21 TD
# sessions exist only as MP3, so MP3 is the only container both arms share; preferring the
# MP4 would put two TD sessions through a different codec chain from the other nineteen
# and every ASD session through a different chain from almost all of TD (health check 0930
# P5). Until 2026-10-01 one TD MP3 had been made by another tool (ffmpeg, 16 kHz mono,
# 34 kbps); it was re-converted with VLC from its MP4.
_CONTAINER_RANK = {".mp3": 0, ".mp4": 1, ".mkv": 2, ".wav": 3, ".m4a": 4}


def cohort_root(root: str | Path, cohort: str | None = None) -> Path:
    """Resolve the directory that actually holds case folders.

    The intake tree gained a cohort level on 2026-09-12: what used to be
    ``Fine_tune/<case>/`` became ``Fine_tune/TD/<case>/`` alongside a new ``ASD/``. Nothing
    raised -- ``discover_cases`` simply found two directories with no audio in them and
    returned an empty list, and the extractor died several frames later on an empty corpus.
    Resolving the cohort level here keeps both layouts working and makes the failure
    impossible to mistake for "no data".
    """
    root = Path(root)
    if cohort:
        nested = root / cohort
        if nested.is_dir():
            return nested
    return root


# The TD sessions excluded for unresolved transcript defects are named in the committed config
# by an id (TDX01), because the case folders are named after the children. The id -> folder
# name table is the gitignored `_private_id_map_excluded.csv` at the git root, next to the other
# private maps (columns `xid,session`; `session` is the fragment of the folder name that the
# config used to hold). `resolve_excluded` is the only reader of that table.
EXCLUDED_ID_MAP = "_private_id_map_excluded.csv"
_EXCLUDED_ID = re.compile(r"TDX\d+")
_GIT_ROOT = REPO_ROOT.parent                 # where the private maps live; the CLI passes the same


def _read_excluded_map(id_root: Path) -> dict[str, str]:
    """xid -> folder-name fragment; empty if the file is not there. An error names the file, a
    column or a line, never a value: in a malformed row any cell may hold a name."""
    path = id_root / EXCLUDED_ID_MAP
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as fh:          # a BOM is not data
        reader = csv.DictReader(fh)
        if not {"xid", "session"} <= set(reader.fieldnames or ()):
            raise ValueError(f"{EXCLUDED_ID_MAP} needs the columns 'xid' and 'session' on its "
                             "first line")
        for line, row in enumerate(reader, start=2):
            xid = (row.get("xid") or "").strip()
            session = (row.get("session") or "").strip()
            if not _EXCLUDED_ID.fullmatch(xid):
                raise ValueError(f"{EXCLUDED_ID_MAP} line {line}: the 'xid' column does not hold "
                                 "an id")
            if not session:                          # an empty fragment would match every folder
                raise ValueError(f"{EXCLUDED_ID_MAP} line {line}: no session")
            if out.setdefault(xid, session) != session:
                raise ValueError(f"{EXCLUDED_ID_MAP} line {line}: an id listed earlier with "
                                 "another session")
    return out


def resolve_excluded(exclude: Iterable[str], id_root: str | Path | None = None) -> list[str]:
    """``realdata.exclude_cases`` as the case-folder name fragments to skip.

    An entry shaped like an id (``TDX01``) is looked up in ``EXCLUDED_ID_MAP`` under ``id_root``
    (default: the git root); any other entry is a fragment already and passes through, so a
    config that still holds fragments keeps working. Every consumer of the list goes through
    here -- ``discover_cases`` (so ``load_cohort``, the extractor, the checks, the MIL run and
    the CLI) and ``audit_cohort`` -- which is why a name is only ever in the list this returns,
    held by the caller's loop and never put back into a config or an output.

    An id the table does not hold raises ``ValueError`` naming that id and the file, nothing
    read from the table. The table is not opened when no entry is an id.
    """
    entries = [str(x) for x in exclude]
    wanted = [x for x in entries if _EXCLUDED_ID.fullmatch(x)]
    if not wanted:
        return entries
    table = _read_excluded_map(Path(id_root) if id_root is not None else _GIT_ROOT)
    absent = [x for x in dict.fromkeys(wanted) if x not in table]
    if absent:
        raise ValueError(f"exclude_cases lists {', '.join(absent)}, which {EXCLUDED_ID_MAP} "
                         "does not hold (or the file is missing)")
    return [table.get(x, x) for x in entries]


def discover_cases(root: str | Path, exclude: Iterable[str] = (),
                   cohort: str | None = None, id_root: str | Path | None = None
                   ) -> list[CaseFiles]:
    """Find case folders holding exactly one audio file and one transcript.

    ``exclude`` is ``realdata.exclude_cases``: ids or folder-name fragments (``resolve_excluded``).
    """
    drop = set(resolve_excluded(exclude, id_root))
    root = cohort_root(root, cohort)
    cases: list[CaseFiles] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if any(x in d.name for x in drop):
            continue
        # .mp4 included since 2026-09-12: the ASD arm was delivered as video. librosa
        # decodes the audio track fine, and a container extension is not a reason to make
        # a case invisible -- which is what happened, silently, for four transcribed cases.
        audio = sorted([*d.glob("*.mp3"), *d.glob("*.wav"), *d.glob("*.m4a"),
                        *d.glob("*.mp4"), *d.glob("*.mkv")],
                       key=lambda p: (_CONTAINER_RANK.get(p.suffix.lower(), 9), p.name))
        js = sorted(d.glob("*.json"))
        if not audio or not js:
            continue
        edited = [p for p in js if "_edited" in p.name]
        # Several files of the chosen container are parts of one session, read end to end
        # in name order. Taking only the first silently dropped 45 of 109 minutes for one
        # ASD session (health check P25).
        parts = tuple(sorted((p for p in audio if p.suffix.lower() == audio[0].suffix.lower()),
                             key=lambda p: p.name))
        cases.append(CaseFiles(d.name, d, parts[0], (edited or js)[0],
                               audio_parts=parts if len(parts) > 1 else ()))
    return cases


def _sorted_segments(segments: Sequence[Mapping]) -> tuple[list[Mapping], int]:
    ordered = sorted(segments, key=lambda s: (float(s.get("start", 0.0)),
                                              float(s.get("end", 0.0))))
    out_of_order = sum(
        1 for a, b in zip(segments[:-1], segments[1:])
        if float(b.get("start", 0.0)) < float(a.get("start", 0.0))
    )
    return ordered, out_of_order


def load_transcript(path: str | Path, cfg: Mapping,
                    case_id: str | None = None) -> tuple[list[Turn], IntakeReport]:
    """Parse one exported transcript into `Turn`s, applying the frozen cleaning rules.

    ``case_id`` defaults to the *directory* name, not ``metadata.case_name``. The two
    disagree in at least one case in this cohort (a folder dated 20251215 whose metadata
    says 2025012015), and the subject id is the join key for every subject-level split --
    it cannot come from a field that is known to drift.
    """
    path = Path(path)
    rd = cfg["realdata"]
    tol = float(rd["overlap_tolerance_s"])
    policy = str(rd["overlap_policy"])
    drop_third = bool(rd["drop_third_party"])
    # Default true, and the reason is comparability rather than tidiness. The TD annotators
    # marked a child's unintelligible vocalisations as silence, so in that cohort they are
    # simply absent; the ASD annotators mark them. Keeping them would add roughly 9% to the
    # ASD child turn count and nothing to the TD one -- a cohort difference produced entirely
    # by the annotation protocol. They are counted in the intake report either way, so the
    # size of what is being discarded stays visible.
    drop_unintelligible = bool(rd.get("drop_unintelligible", True))

    payload = json.loads(path.read_text(encoding="utf-8"))
    meta = payload.get("metadata", {}) or {}
    raw_segments = payload.get("segments", []) or []
    resolved = case_id or path.parent.name
    rep = IntakeReport(case_id=resolved, transcript_path=str(path), n_raw=len(raw_segments))
    declared = meta.get("case_name")
    if declared and str(declared) != resolved:
        rep.notes.append(
            f"metadata.case_name ({declared!r}) disagrees with the folder name "
            f"({resolved!r}); the folder name is authoritative"
        )

    mapping = meta.get("speaker_mapping") or {}
    ordered, rep.n_out_of_order = _sorted_segments(raw_segments)

    # ---- pass 1: role normalisation, third parties, empty text ----
    staged: list[dict] = []
    for s in ordered:
        raw_label = s.get("speaker")
        key = str(raw_label).strip() if raw_label is not None else "None"
        rep.speaker_vocabulary[key] = rep.speaker_vocabulary.get(key, 0) + 1
        role = normalise_speaker(raw_label, mapping)
        text = (s.get("text") or "").strip()
        note = str(s.get("manual_annotation") or "").strip().lower()
        if (SILENT_MARKER in text or key in SILENCE_SPEAKER_LABELS
                or note in _ANNOTATION_SILENT):
            rep.n_silent_marker += 1
            rep.silent_marker_s += max(0.0, float(s.get("end", 0.0)) - float(s.get("start", 0.0)))
            continue
        unintelligible = (UNINTELLIGIBLE_MARKER in text
                          or note in _ANNOTATION_UNINTELLIGIBLE)
        if unintelligible:
            rep.n_unintelligible += 1
            text = ""              # a vocalisation, but not words
            if drop_unintelligible:
                continue
        if role is None:
            rep.n_third_party += 1
            rep.third_party_labels[key] = rep.third_party_labels.get(key, 0) + 1
            if key not in THIRD_PARTY_HINTS:
                rep.n_unmapped_labels += 1
            if drop_third:
                continue
        if not text and not unintelligible:
            rep.n_empty_text += 1
            continue
        start, end = float(s.get("start", 0.0)), float(s.get("end", 0.0))
        if end <= start:
            rep.n_empty_text += 1
            continue
        staged.append({"start": start, "end": end, "text": text,
                       "role": role or Speaker.OVERLAP,
                       "unintelligible": unintelligible})

    # ---- pass 2: overlap handling ----
    drop_flags = [False] * len(staged)
    for i in range(len(staged) - 1):
        a, b = staged[i], staged[i + 1]
        gap = b["start"] - a["end"]
        if gap >= -tol:
            continue                                   # touching or a rounding-level nick
        rep.n_overlap_pairs += 1
        rep.overlap_seconds += min(a["end"], b["end"]) - b["start"]
        if policy == "drop_both":
            drop_flags[i] = drop_flags[i + 1] = True
        elif policy == "drop_later":
            drop_flags[i + 1] = True
        elif policy == "mark":
            a["role"] = b["role"] = Speaker.OVERLAP
        else:
            raise ValueError(f"unknown overlap_policy {policy!r}")
    rep.n_dropped_overlap = sum(drop_flags)

    turns: list[Turn] = []
    for keep, s in zip((not f for f in drop_flags), staged):
        if not keep:
            continue
        text = s["text"]
        turns.append(Turn(
            turn_id=len(turns), speaker=s["role"], t_start=s["start"], t_end=s["end"],
            text=text,
            # Marked for every speaker, not just the examiner. The restriction was
            # harmless while only the E stream read this flag -- F7/F8 are about examiner
            # probes and filter to examiner turns anyway -- but it silently zeroed the
            # question branch of assessment item B3 (句尾語調不對): all 4,860 child turns
            # in the real cohort came back as non-questions, so the feature was constant.
            is_question=is_question(text),
            is_yes_no=is_yes_no_question(text),
            unintelligible=bool(s.get("unintelligible", False)),
        ))

    rep.n_kept = len(turns)
    rep.n_examiner = sum(1 for t in turns if t.speaker is Speaker.EXAMINER)
    rep.n_child = sum(1 for t in turns if t.speaker is Speaker.CHILD)
    if turns and rep.n_child == 0:
        # Loud, because the quiet version of this is a table of zeros that reads as a
        # finding about the child. Every cause is a data or configuration problem: an
        # unrecognised speaker label, a transcript with only examiner speech, or an
        # overlap/third-party policy that removed everything.
        rep.errors.append(
            f"no child turns after role mapping ({rep.n_kept} turns kept, "
            f"{rep.n_unmapped_labels} unmapped labels: {sorted(rep.third_party_labels)})"
        )
    rep.child_speech_s = sum(t.duration for t in turns if t.speaker is Speaker.CHILD)
    rep.examiner_speech_s = sum(t.duration for t in turns if t.speaker is Speaker.EXAMINER)
    rep.session_s = max((t.t_end for t in turns), default=0.0)

    from .features.e_text import pair_turns

    pairs = pair_turns(turns)
    rep.n_qa_pairs = len(pairs)
    rep.n_open_qa_pairs = sum(1 for ex, _ in pairs if ex.is_question and not ex.is_yes_no)
    ex_turns = [t for t in turns if t.speaker is Speaker.EXAMINER]
    rep.n_examiner_questions = sum(1 for t in ex_turns if t.is_question)
    rep.n_examiner_yes_no = sum(1 for t in ex_turns if t.is_yes_no)
    rep.n_question_marks = sum(1 for t in ex_turns if _QUESTION_MARK.search(t.text))
    return turns, rep


def _audio_cache_path(audio: Path, sr: int, cache_dir: Path) -> Path:
    stat = audio.stat()
    key = hashlib.blake2s(
        f"{audio.resolve()}|{stat.st_size}|{int(stat.st_mtime)}|{sr}".encode("utf-8"),
        digest_size=10,
    ).hexdigest()
    return cache_dir / "audio" / f"{key}.npy"


def load_audio(audio: Path, sr: int, cache_dir: Path | None) -> np.ndarray:
    """Decode to mono at `sr`, cached as .npy (decoding a 30-minute MP3 costs ~9 s)."""
    if cache_dir is not None:
        cached = _audio_cache_path(audio, sr, Path(cache_dir))
        if cached.exists():
            return np.load(cached, mmap_mode=None)
    import librosa

    y, _ = librosa.load(str(audio), sr=sr, mono=True)
    y = y.astype(np.float32)
    if cache_dir is not None:
        cached.parent.mkdir(parents=True, exist_ok=True)
        np.save(cached, y)
    return y


def load_subject(case: CaseFiles, cfg: Mapping, label: Label | None = None,
                 cache_dir: Path | None = None,
                 with_audio: bool = True) -> tuple[SubjectRecord, IntakeReport]:
    sr = int(cfg["synth"]["sample_rate"])
    turns, rep = load_transcript(case.transcript, cfg, case_id=case.case_id)
    rep.audio_path = str(case.audio)
    rep.label_is_placeholder = label is None
    if with_audio and case.audio_parts:
        pieces = [load_audio(p, sr, cache_dir) for p in case.audio_parts]
        audio = np.concatenate(pieces).astype(np.float32, copy=False)
        bounds = np.cumsum([len(x) / sr for x in pieces])[:-1]
        rep.notes.append(f"{len(pieces)} audio parts joined end to end; part boundaries at "
                         + ", ".join(f"{b:.1f}" for b in bounds) + " s. The transcript must "
                         "use one continuous timeline across the parts.")
    elif with_audio:
        audio = load_audio(case.audio, sr, cache_dir)
    else:
        audio = np.zeros(int(rep.session_s * sr) + sr, dtype=np.float32)
    record = SubjectRecord(
        subject_id=case.case_id,
        label=label if label is not None else Label.CONTROL,
        session=Session(session_id=case.case_id, subject_id=case.case_id,
                        turns=tuple(turns), sample_rate=sr),
        audio=audio, sample_rate=sr,
        ados={}, latents={},
        cohort=str(cfg["realdata"].get("cohort", "")),
    )
    return record, rep


def load_cohort(root: str | Path, cfg: Mapping, labels: Mapping[str, Label] | None = None,
                cache_dir: Path | None = None, with_audio: bool = True,
                exclude: Iterable[str] | None = None,
                id_root: str | Path | None = None) -> tuple[Corpus, list[IntakeReport]]:
    """Load every usable case under `root`.

    ``labels`` is deliberately required for any modelling use. Without it every record
    carries a placeholder label and `IntakeReport.label_is_placeholder` is True, so a
    downstream classification run cannot silently treat one cohort as if it were two.
    ``exclude`` defaults to ``realdata.exclude_cases`` (ids, see ``resolve_excluded``).
    """
    rd = cfg["realdata"]
    exclude = list(exclude if exclude is not None else rd.get("exclude_cases", []))
    cases = discover_cases(root, exclude, cohort=cfg["realdata"].get("cohort"), id_root=id_root)
    records, reports = [], []
    for c in cases:
        lab = labels.get(c.case_id) if labels else None
        rec, rep = load_subject(c, cfg, lab, cache_dir, with_audio)
        records.append(rec)
        reports.append(rep)
    return Corpus(subjects=tuple(records), config_hash=f"real:{Path(root).name}"), reports
