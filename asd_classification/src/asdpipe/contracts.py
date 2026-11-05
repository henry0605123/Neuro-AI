"""Data contracts shared by every stage of the pipeline.

Design rules enforced here:

* R2  -- a subject is the atomic unit. `SubjectTable` stores exactly one row per subject
         and refuses to be built otherwise.
* R6  -- syllable boundaries are carried on the `Turn` itself, so the synthetic path
         (ground-truth boundaries) and the real path (MFA output) are interchangeable.

Nothing in this module imports a model or touches the filesystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "Speaker",
    "Label",
    "SyllableSpan",
    "Turn",
    "Session",
    "SubjectRecord",
    "Corpus",
    "SubjectRow",
    "SubjectTable",
    "Fold",
]


class Speaker(str, Enum):
    EXAMINER = "EXAMINER"
    CHILD = "CHILD"
    OVERLAP = "OVERLAP"


class Label(int, Enum):
    CONTROL = 0
    ASD = 1


@dataclass(frozen=True)
class SyllableSpan:
    """One Mandarin syllable with its tone category.

    ``tone`` uses the standard Mandarin inventory: 1-4 are the lexical tones, 0 is the
    neutral tone (轻声/輕聲). For the synthetic corpus these are ground truth; for real
    data they come from the forced aligner (see ``asdpipe.align``).
    """

    t_start: float
    t_end: float
    tone: int
    text: str = ""
    vocalic_s: float | None = None
    """Duration of the syllable's nucleus. Lets %V and dC use the real vocalic/consonantal
    split (Ramus et al.) instead of a voicing proxy; ``None`` falls back to the proxy."""
    nucleus_start: float | None = None
    nucleus_end: float | None = None
    """Time span of the tone-bearing rime, when the aligner supplies phone boundaries.

    Mandarin tone is carried by the rime, not the onset, and F0 over a pre-nuclear glide or
    a nasal onset is not the tone -- it is the transition into it. Measuring the contour
    across the whole syllable mixes the two, and on this corpus that shows up as a T4
    template that *rises* through its first half, which is phonetically impossible for a
    high-falling tone. These two fields let the C stream measure the tone where it actually
    lives. ``None`` means the aligner did not provide phone timings."""
    underlying_tone: int | None = None
    """Citation tone before sandhi. Comes from the generator on synthetic data and from the
    pronunciation dictionary on the real path; ``tone`` is what was actually produced."""

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    def __post_init__(self) -> None:
        if self.t_end <= self.t_start:
            raise ValueError(f"syllable must have positive duration: {self}")
        if self.tone not in (0, 1, 2, 3, 4):
            raise ValueError(f"tone must be one of 0..4, got {self.tone}")


@dataclass(frozen=True)
class Turn:
    """One conversational turn.

    ``syllables`` is empty until an aligner fills it in. ``intensity_db`` is the
    generator's ground-truth level and is deliberately **not** used by any feature
    extractor -- intensity is always re-measured from the waveform.
    """

    turn_id: int
    speaker: Speaker
    t_start: float
    t_end: float
    text: str
    syllables: tuple[SyllableSpan, ...] = ()
    intensity_db: float | None = None
    is_question: bool = False
    is_yes_no: bool = False
    topic: str = ""
    unintelligible: bool = False
    """The child vocalised but the annotator could not make out words ([UNINTELLIGIBLE]).

    ``text`` is empty for these, so anything needing words -- embeddings, segmentation,
    forced alignment -- skips them naturally. They still count as vocalisation, which
    matters most for exactly the children who produce the fewest words."""

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    def with_syllables(self, syls: Sequence[SyllableSpan]) -> "Turn":
        return replace(self, syllables=tuple(syls))

    def __post_init__(self) -> None:
        if self.t_end < self.t_start:
            raise ValueError(f"turn {self.turn_id} ends before it starts")


@dataclass(frozen=True)
class Session:
    session_id: str
    subject_id: str
    turns: tuple[Turn, ...]
    sample_rate: int

    def by_speaker(self, speaker: Speaker) -> tuple[Turn, ...]:
        return tuple(t for t in self.turns if t.speaker is speaker)

    @property
    def duration(self) -> float:
        return max((t.t_end for t in self.turns), default=0.0)


@dataclass(frozen=True)
class SubjectRecord:
    """Everything known about one subject.

    ``latents`` holds the generator's injected ground truth. It exists so tests can check
    that a feature recovers what was injected; it must never reach a model. `SubjectTable`
    keeps it in ``meta`` and the leakage tests assert it never enters ``X``.
    """

    subject_id: str
    label: Label
    session: Session
    audio: np.ndarray
    sample_rate: int
    ados: Mapping[str, int] = field(default_factory=dict)
    latents: Mapping[str, float] = field(default_factory=dict)
    cohort: str = ""
    """Which population this subject came from, e.g. "TD" or "referred-not-diagnosed".
    Kept separate from `label` on purpose: community TD is **not** the study's control
    group, and conflating the two would quietly change what a result means."""

    def __post_init__(self) -> None:
        if self.audio.ndim != 1:
            raise ValueError("audio must be mono 1-D")


@dataclass(frozen=True)
class Corpus:
    subjects: tuple[SubjectRecord, ...]
    config_hash: str = ""

    @property
    def labels(self) -> np.ndarray:
        return np.array([int(s.label) for s in self.subjects], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.subjects)


@dataclass(frozen=True)
class SubjectRow:
    """Pooled, subject-level representation produced by milestone M3."""

    subject_id: str
    label: Label
    b_pooled: Mapping[str, float]
    c_pooled: Mapping[str, float]
    e_pooled: Mapping[str, float]
    c_sequence: np.ndarray          # (n_syllables, syllable_feature_dim)
    egemaps_functionals: np.ndarray  # (88,) for baseline B2
    examiner_probe: Mapping[str, float]
    ados: Mapping[str, int]
    latents: Mapping[str, float]
    quality: Mapping[str, float]


class SubjectTable:
    """Model-facing table. One row per subject -- this is the R2 guarantee in code."""

    def __init__(self, rows: Sequence[SubjectRow], max_sequence_len: int | None = None):
        ids = [r.subject_id for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError("SubjectTable requires unique subject ids (R2: no segment expansion)")
        self.rows = list(rows)
        self.subject_ids = np.array(ids, dtype=object)
        self.y = np.array([int(r.label) for r in rows], dtype=np.int64)

        # Columns are the union over subjects, and a feature a subject lacks is NaN. Taking
        # the names from the first row crashed -- or silently mislabelled columns -- whenever
        # a stream returned a partial dict, which is exactly what happens for a child with no
        # syllables at all (health check P12).
        self.b_names = _union_keys(r.b_pooled for r in rows)
        self.c_names = _union_keys(r.c_pooled for r in rows)
        self.e_names = _union_keys(r.e_pooled for r in rows)
        self.X_b = _matrix([r.b_pooled for r in rows], self.b_names)
        self.X_c = _matrix([r.c_pooled for r in rows], self.c_names)
        self.X_e = _matrix([r.e_pooled for r in rows], self.e_names)
        self.X_egemaps = np.array([r.egemaps_functionals for r in rows], dtype=np.float64)

        self.probe_names = list(rows[0].examiner_probe.keys())
        self.X_probe = np.array(
            [[r.examiner_probe[k] for k in self.probe_names] for r in rows], dtype=np.float64
        )

        seq_lens = [len(r.c_sequence) for r in rows]
        T = max_sequence_len or max(seq_lens)
        D = rows[0].c_sequence.shape[1]
        self.seq = np.zeros((len(rows), T, D), dtype=np.float32)
        self.seq_mask = np.zeros((len(rows), T), dtype=np.float32)
        for i, r in enumerate(rows):
            n = min(len(r.c_sequence), T)
            if n:
                self.seq[i, :n] = r.c_sequence[:n]
                self.seq_mask[i, :n] = 1.0

        self.ados = {k: np.array([r.ados.get(k, np.nan) for r in rows], dtype=float)
                     for k in (rows[0].ados.keys() if rows[0].ados else [])}
        self.latents = {k: np.array([r.latents.get(k, np.nan) for r in rows], dtype=float)
                        for k in (rows[0].latents.keys() if rows[0].latents else [])}
        self.quality = [dict(r.quality) for r in rows]

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def n_subjects(self) -> int:
        return len(self.rows)

    def stream(self, name: str) -> np.ndarray:
        return {"b": self.X_b, "c": self.X_c, "e": self.X_e}[name]

    def feature_names(self, name: str) -> list[str]:
        return {"b": self.b_names, "c": self.c_names, "e": self.e_names}[name]

    def all_feature_names(self) -> list[str]:
        return (
            [f"b/{n}" for n in self.b_names]
            + [f"c/{n}" for n in self.c_names]
            + [f"e/{n}" for n in self.e_names]
        )

    def without_features(self, names: Iterable[str]) -> "SubjectTable":
        """A copy with the named feature columns removed from every stream."""
        import copy

        drop = set(names)
        out = copy.copy(self)
        for s in ("b", "c", "e"):
            keep = [j for j, n in enumerate(self.feature_names(s)) if n not in drop]
            setattr(out, f"{s}_names", [self.feature_names(s)[j] for j in keep])
            setattr(out, f"X_{s}", self.stream(s)[:, keep])
        return out

    def unmeasurable_counts(self) -> dict[str, np.ndarray]:
        """Per stream, how many features could not be measured for each subject (NaN).

        Disclosed by group in the report (audit P0-4); never a model input.
        """
        return {s: (~np.isfinite(self.stream(s))).sum(axis=1) for s in ("b", "c", "e")}


def _union_keys(dicts: Iterable[Mapping[str, float]]) -> list[str]:
    seen: dict[str, None] = {}
    for d in dicts:
        for k in d:
            seen.setdefault(k, None)
    return list(seen)


def _matrix(dicts: Sequence[Mapping[str, float]], names: Sequence[str]) -> np.ndarray:
    return np.array([[d.get(k, np.nan) for k in names] for d in dicts],
                    dtype=np.float64).reshape(len(dicts), len(names))


@dataclass(frozen=True)
class Fold:
    """One outer cross-validation fold. Indices are positions into `SubjectTable`."""

    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray

    def __post_init__(self) -> None:
        overlap = set(self.train_idx.tolist()) & set(self.test_idx.tolist())
        if overlap:
            raise ValueError(f"R2 violated: subjects {sorted(overlap)} are in both train and test")
