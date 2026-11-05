"""Alignment quality measurement (audit P1-1).

**What this module cannot do.** P1-1 asks for "human spot-check of 30-50 segments, median
boundary error < 20 ms". That threshold is defined against *human-annotated* syllable
boundaries, and this cohort has none. No amount of code produces a ground truth that does
not exist, so this module does two separate things and never conflates them:

1. **Automatic checks that stand on their own.** Out-of-vocabulary rate, alignment yield,
   duration plausibility, and -- when two dictionaries are available -- the disagreement
   between two independent alignments of the same audio, which is a genuine upper bound on
   how well the boundary is determined.
2. **Materials for the human check that P1-1 actually mandates**: sampled utterances
   exported as wav + TextGrid so a person can open them in Praat and judge.

The energy-trough comparison is reported as a *proxy* and labelled as one. Mandarin
syllable boundaries often coincide with an amplitude minimum, but "often" is not
"ground truth", and a good agreement number here does not discharge P1-1.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..contracts import SyllableSpan, Turn

__all__ = ["AlignmentQuality", "read_dictionary_words", "oov_report", "duration_sanity",
           "boundary_energy_agreement", "cross_alignment_agreement", "implied_rate_profile",
           "write_textgrid", "export_spotcheck"]


@dataclass
class AlignmentQuality:
    case_id: str = ""
    n_turns_attempted: int = 0
    n_turns_aligned: int = 0
    n_syllables: int = 0
    n_expected_syllables: int = 0
    oov_types: int = 0
    vocab_types: int = 0
    oov_tokens: int = 0
    total_tokens: int = 0
    oov_examples: list[str] = field(default_factory=list)
    dur_median_ms: float = float("nan")
    dur_p10_ms: float = float("nan")
    dur_p90_ms: float = float("nan")
    n_implausible: int = 0
    energy_proxy_median_ms: float = float("nan")
    energy_proxy_p90_ms: float = float("nan")
    cross_dict_median_ms: float = float("nan")
    cross_dict_p90_ms: float = float("nan")
    notes: list[str] = field(default_factory=list)

    @property
    def turn_yield(self) -> float:
        return self.n_turns_aligned / self.n_turns_attempted if self.n_turns_attempted else 0.0

    @property
    def syllable_yield(self) -> float:
        return self.n_syllables / self.n_expected_syllables if self.n_expected_syllables else 0.0

    @property
    def oov_token_rate(self) -> float:
        return self.oov_tokens / self.total_tokens if self.total_tokens else 0.0


def read_dictionary_words(path: str | Path) -> set[str]:
    """Read the orthographic side of an MFA pronunciation dictionary."""
    words: set[str] = set()
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            words.add(line.split()[0])
    return words


def oov_report(words_by_turn: Mapping[int, Sequence[str]], lexicon: set[str],
               n_examples: int = 30) -> dict:
    """Which segmented words the dictionary does not know.

    This is the decisive check for a Traditional-Chinese corpus: MFA's Mandarin
    dictionaries are distributed in one orthography, and if the transcripts use the other
    one the OOV rate goes to nearly 100% and every alignment is meaningless.
    """
    tokens = [w for ws in words_by_turn.values() for w in ws]
    types = set(tokens)
    oov_types = sorted(t for t in types if t not in lexicon)
    oov_tokens = sum(1 for t in tokens if t not in lexicon)
    return {
        "total_tokens": len(tokens), "vocab_types": len(types),
        "oov_types": len(oov_types), "oov_tokens": oov_tokens,
        "oov_type_rate": len(oov_types) / len(types) if types else 0.0,
        "oov_token_rate": oov_tokens / len(tokens) if tokens else 0.0,
        "examples": oov_types[:n_examples],
    }


def duration_sanity(spans: Sequence[SyllableSpan], lo_s: float = 0.05,
                    hi_s: float = 1.0) -> dict:
    """Mandarin syllables in child speech run roughly 100-400 ms; flag the impossible ones."""
    d = np.array([s.duration for s in spans], dtype=float)
    if d.size == 0:
        return {"n": 0, "median_ms": float("nan"), "p10_ms": float("nan"),
                "p90_ms": float("nan"), "n_implausible": 0, "implausible_rate": 0.0}
    bad = int(((d < lo_s) | (d > hi_s)).sum())
    return {
        "n": int(d.size),
        "median_ms": float(np.median(d) * 1000),
        "p10_ms": float(np.percentile(d, 10) * 1000),
        "p90_ms": float(np.percentile(d, 90) * 1000),
        "n_implausible": bad,
        "implausible_rate": bad / d.size,
    }


def _short_time_energy(x: np.ndarray, sr: int, hop_s: float = 0.005,
                       win_s: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    hop, win = max(1, int(hop_s * sr)), max(2, int(win_s * sr))
    n = max(0, (len(x) - win) // hop + 1)
    if n <= 0:
        return np.zeros(0), np.zeros(0)
    idx = np.arange(n) * hop
    frames = np.lib.stride_tricks.sliding_window_view(x, win)[::hop][:n]
    return (idx + win / 2) / sr, np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))


def boundary_energy_agreement(audio: np.ndarray, sr: int, spans: Sequence[SyllableSpan],
                              search_ms: float = 100.0) -> dict:
    """PROXY ONLY. Distance from each internal boundary to the nearest energy trough.

    Not a substitute for P1-1: an amplitude minimum is where a syllable boundary *tends*
    to be, not where it *is*. Useful for spotting gross failure (a systematically offset
    alignment shows up as a large median), useless for certifying 20 ms accuracy.
    """
    if len(spans) < 2 or audio.size == 0:
        return {"n": 0, "median_ms": float("nan"), "p90_ms": float("nan")}
    t, e = _short_time_energy(audio, sr)
    if t.size < 3:
        return {"n": 0, "median_ms": float("nan"), "p90_ms": float("nan")}
    troughs = t[1:-1][(e[1:-1] <= e[:-2]) & (e[1:-1] <= e[2:])]
    if troughs.size == 0:
        return {"n": 0, "median_ms": float("nan"), "p90_ms": float("nan")}
    deltas = []
    for a, b in zip(spans[:-1], spans[1:]):
        if abs(b.t_start - a.t_end) > 1e-6:      # only boundaries shared by two syllables
            continue
        d = np.abs(troughs - a.t_end).min() * 1000
        if d <= search_ms:
            deltas.append(d)
    if not deltas:
        return {"n": 0, "median_ms": float("nan"), "p90_ms": float("nan")}
    arr = np.array(deltas)
    return {"n": int(arr.size), "median_ms": float(np.median(arr)),
            "p90_ms": float(np.percentile(arr, 90))}


def cross_alignment_agreement(a: Mapping[int, Sequence[SyllableSpan]],
                              b: Mapping[int, Sequence[SyllableSpan]]) -> dict:
    """Boundary disagreement between two alignments of the same audio.

    Two dictionaries (e.g. the mainland and Taiwan Mandarin variants) produce independent
    alignments; where they disagree, at least one is wrong. This is a real lower bound on
    the error and it directly quantifies how much the QUESTIONS Q14 choice matters.
    """
    deltas: list[float] = []
    compared_turns = 0
    for turn_id, sa in a.items():
        sb = b.get(turn_id)
        if not sb or len(sa) != len(sb):
            continue
        compared_turns += 1
        deltas += [abs(x.t_start - y.t_start) * 1000 for x, y in zip(sa, sb)]
        deltas += [abs(sa[-1].t_end - sb[-1].t_end) * 1000]
    if not deltas:
        return {"n": 0, "n_turns": 0, "median_ms": float("nan"), "p90_ms": float("nan"),
                "within_20ms": float("nan")}
    arr = np.array(deltas)
    return {
        "n": int(arr.size), "n_turns": compared_turns,
        "median_ms": float(np.median(arr)), "p90_ms": float(np.percentile(arr, 90)),
        "within_20ms": float(np.mean(arr <= 20.0)),
    }


def implied_rate_profile(spans_by_turn: Mapping[int, Sequence[SyllableSpan]],
                         bins: Sequence[tuple[int, int]] = ((2, 5), (5, 10), (10, 15), (15, 10**6)),
                         implausible_rate: float = 8.0) -> dict:
    """Syllables per second implied by the alignment, broken down by utterance length.

    This is the cheapest way to catch a compressed alignment: if the aligner squeezes an
    utterance's syllables into a fraction of the speech, the implied articulation rate goes
    far above what a 4-6 year old can produce. Published Mandarin articulation rates sit
    around 4-6 syllables/s for adults, so anything sustained above ~8/s is a flag.

    It has to be read **per utterance length**. Short backchannels ("對啊", "會啊") legitimately
    hit 10+/s, and they are excluded from the rhythm features anyway (audit P1-2 wants
    utterances of >= 10 syllables for nPVI). The number that matters is the rate in the long
    bins.
    """
    rows = []
    for spans in spans_by_turn.values():
        if len(spans) < 2:
            continue
        span = spans[-1].t_end - spans[0].t_start
        if span <= 0:
            continue
        durs = np.array([s.duration for s in spans], dtype=float)
        rows.append((len(spans), span, len(spans) / span, float(np.median(durs)),
                     float(durs.sum() / span)))
    if not rows:
        return {"n": 0, "bins": [], "long_utterances": {}}
    r = np.array(rows)
    out_bins = []
    for lo, hi in bins:
        m = (r[:, 0] >= lo) & (r[:, 0] < hi)
        if not m.any():
            continue
        out_bins.append({
            "min_syllables": lo, "max_syllables": None if hi > 10**5 else hi,
            "n": int(m.sum()),
            "median_rate": float(np.median(r[m, 2])),
            "median_syllable_ms": float(1000 * np.median(r[m, 3])),
            "syllable_coverage": float(np.median(r[m, 4])),
            "frac_implausible": float((r[m, 2] > implausible_rate).mean()),
        })
    long_m = r[:, 0] >= 10
    long_stats = {}
    if long_m.any():
        v = r[long_m, 2]
        long_stats = {
            "n": int(long_m.sum()),
            "p10": float(np.percentile(v, 10)), "median": float(np.median(v)),
            "p90": float(np.percentile(v, 90)),
            "frac_implausible": float((v > implausible_rate).mean()),
        }
    return {"n": len(rows), "bins": out_bins, "long_utterances": long_stats,
            "implausible_rate": implausible_rate}


def write_textgrid(path: Path, spans: Sequence[SyllableSpan], xmin: float, xmax: float,
                   tier_name: str = "syllables") -> Path:
    """Minimal long-form TextGrid so a human can open the sample in Praat."""
    lines = [
        'File type = "ooTextFile"', 'Object class = "TextGrid"', "",
        f"xmin = {xmin}", f"xmax = {xmax}", "tiers? <exists>", "size = 1", "item []:",
        "    item [1]:", '        class = "IntervalTier"', f'        name = "{tier_name}"',
        f"        xmin = {xmin}", f"        xmax = {xmax}",
        f"        intervals: size = {len(spans)}",
    ]
    for i, s in enumerate(spans, 1):
        lines += [f"        intervals [{i}]:", f"            xmin = {s.t_start}",
                  f"            xmax = {s.t_end}",
                  f'            text = "{s.text or ""}|T{s.tone}"']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def export_spotcheck(out_dir: Path, case_id: str, audio: np.ndarray, sr: int,
                     turns: Sequence[Turn], spans_by_turn: Mapping[int, Sequence[SyllableSpan]],
                     n: int = 40, seed: int = 0) -> list[Path]:
    """Export sampled utterances as wav + TextGrid for the human check P1-1 requires.

    Boundaries are rebased to the clip, so each pair opens directly in Praat.
    """
    import soundfile as sf

    rng = random.Random(seed)
    candidates = [t for t in turns if spans_by_turn.get(t.turn_id)]
    picked = rng.sample(candidates, min(n, len(candidates)))
    out_dir = Path(out_dir) / case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    manifest = ["turn_id\tstart_s\tend_s\tn_syllables\ttext"]
    for t in sorted(picked, key=lambda x: x.turn_id):
        a = max(0, int(round(t.t_start * sr)))
        b = min(len(audio), int(round(t.t_end * sr)))
        if b - a < sr // 20:
            continue
        clip = audio[a:b]
        stem = out_dir / f"turn{t.turn_id:05d}"
        sf.write(stem.with_suffix(".wav"), clip, sr)
        rebased = [SyllableSpan(s.t_start - t.t_start, s.t_end - t.t_start, s.tone, s.text)
                   for s in spans_by_turn[t.turn_id]
                   if s.t_end - t.t_start > 0 and s.t_start - t.t_start < len(clip) / sr]
        write_textgrid(stem.with_suffix(".TextGrid"), rebased, 0.0, len(clip) / sr)
        written += [stem.with_suffix(".wav"), stem.with_suffix(".TextGrid")]
        manifest.append(f"{t.turn_id}\t{t.t_start:.3f}\t{t.t_end:.3f}\t"
                        f"{len(rebased)}\t{t.text}")
    (out_dir / "manifest.tsv").write_text("\n".join(manifest), encoding="utf-8")
    return written
