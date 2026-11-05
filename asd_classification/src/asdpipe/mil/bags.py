"""Bags: one child is one bag of segments (research plan 3.2 and 3.5).

The segment-attention model scores a child from the series of that child's segments, and the
label belongs to the child, never to a segment (R2). Cross-validation therefore splits
children, and everything the model sees is one ``Bags``: every child at once, padded to the
longest bag, so that many models can be trained in a single batch.

Real data (``bags_from_arrays``, fed with ``segment_measures`` output) and the synthetic cohort
the model and the cross-validation are tested on first (plan section 7, step 4: AUC about 0.5
with no signal, rising with signal) go through the same container, so a test on one is a test
of the other.

**Missing versus padding.** ``seg`` is NaN where a measure could not be computed inside a real
segment (the model imputes it from the training folds only, R3) and exactly 0 in padding rows;
``mask`` is what tells the two apart. Padding must be 0 and never NaN: the model multiplies
every position by its attention weight, and 0 * NaN is NaN even where the weight is 0.
Infinity is refused outright, because "could not be computed" has exactly one spelling.

``Bags`` checks itself when it is built, and ``select`` and ``dataclasses.replace`` build
through the same constructor, so a Bags that exists is consistent: the shapes agree, ids and
names are unique, labels are 0 (TD) or 1 (ASD), and every child holds at least one segment (an
empty bag would make the attention softmax 0 / 0). Unique ids are R2 in code, the way
``SubjectTable`` refuses a repeated subject.

The module needs numpy only, so the CPU-only checks and the report can import it without
loading torch.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .segments import CHILD_MEASURES, SEGMENT_MEASURES

__all__ = ["Bags", "bags_from_arrays", "synthetic_bags"]


def _repeated(items: Sequence[str]) -> list[str]:
    return sorted(name for name, count in Counter(items).items() if count > 1)


def _columns(have: tuple[str, ...], want: Sequence[str] | None, kind: str) -> np.ndarray:
    """Positions of the wanted names, in the original column order.

    A name this Bags does not have raises: a typo must not quietly select fewer columns than
    asked for, which is what a plain filter would do.
    """
    if want is None:
        return np.arange(len(have), dtype=np.intp)
    if isinstance(want, str):                       # one name, not a sequence of characters
        want = (want,)
    wanted = set(want)                              # once: ``want`` may be a one-shot generator
    unknown = sorted(wanted - set(have))
    if unknown:
        raise ValueError(f"unknown {kind} measure(s) {unknown}; this Bags has {list(have)}")
    return np.array([j for j, name in enumerate(have) if name in wanted], dtype=np.intp)


def _positions(rows: Sequence[int] | np.ndarray, n: int) -> np.ndarray:
    """Row positions as integers, whether ``rows`` holds positions or a boolean mask."""
    idx = np.asarray(rows)
    if idx.ndim != 1:
        raise ValueError(f"rows must be a 1-D array of positions or a boolean mask, "
                         f"got {idx.ndim}-D")
    if idx.dtype != bool and idx.size == 0:         # an empty list has no integer dtype
        return np.empty(0, dtype=np.intp)
    return np.arange(n)[idx]                        # negative and out-of-range raise as usual


# eq=False: every field is an array or holds one, so a field-wise == could only raise
# ("truth value of an array is ambiguous"). Compare the arrays you mean to compare.
@dataclass(frozen=True, eq=False)
class Bags:
    """Every child's bag of segments, padded to a common length.

    ``ids`` are the pseudonymous ids (never a name). ``y`` is 1 for ASD and 0 for TD.
    ``seg`` is ``(N, T, Ds)`` float32: row ``t`` of child ``i`` is its t-th segment, NaN marks
    a measure that could not be computed, and rows past the child's last segment are 0 (see the
    module docstring). ``mask`` is ``(N, T)`` and True on real segments. ``child`` is
    ``(N, Dc)`` float32, the whole-session measures. ``seg_names`` and ``child_names`` name the
    columns of ``seg`` and ``child``. ``exposed`` marks the children whose data had been looked
    at before the analysis was frozen (DEVIATIONS D10), so that a run can leave them out.
    """

    ids: tuple[str, ...]
    y: np.ndarray
    seg: np.ndarray
    mask: np.ndarray
    child: np.ndarray
    seg_names: tuple[str, ...]
    child_names: tuple[str, ...]
    exposed: np.ndarray

    def __post_init__(self) -> None:
        # Normalise first, so a list or a float64 array is accepted and every Bags has the
        # documented dtypes. The dataclass is frozen, hence object.__setattr__ at the end.
        ids = tuple(str(i) for i in self.ids)
        seg_names = tuple(str(n) for n in self.seg_names)
        child_names = tuple(str(n) for n in self.child_names)
        y = np.asarray(self.y)
        seg = np.asarray(self.seg, dtype=np.float32)
        mask = np.asarray(self.mask, dtype=bool)
        child = np.asarray(self.child, dtype=np.float32)
        exposed = np.asarray(self.exposed, dtype=bool)

        if seg.ndim != 3:
            raise ValueError(f"seg must be (children, segments, measures); got shape {seg.shape}")
        n = len(ids)
        expected = {"y": (n,), "seg": (n, seg.shape[1], len(seg_names)), "mask": seg.shape[:2],
                    "child": (n, len(child_names)), "exposed": (n,)}
        actual = {"y": y.shape, "seg": seg.shape, "mask": mask.shape, "child": child.shape,
                  "exposed": exposed.shape}
        for name, want in expected.items():
            if actual[name] != want:
                raise ValueError(f"{name} has shape {actual[name]}, expected {want}")

        for label, names in (("ids", ids), ("seg_names", seg_names),
                             ("child_names", child_names)):
            repeated = _repeated(names)
            if repeated:
                raise ValueError(f"{label} must be unique; repeated: {repeated}")
        both = sorted(set(seg_names) & set(child_names))
        if both:
            raise ValueError(f"a name cannot be both a segment and a child measure: {both}")

        if not np.isin(y, (0, 1)).all():
            raise ValueError("y must be 0 (TD) or 1 (ASD)")
        empty = np.flatnonzero(~mask.any(axis=1))
        if empty.size:
            who = [(int(i), ids[i]) for i in empty[:5]]
            raise ValueError(f"every child needs at least one segment; none for (row, id) {who}")
        if not (seg[~mask] == 0).all():
            raise ValueError("padding rows of seg must be exactly 0 (NaN is only for a measure "
                             "that could not be computed inside a real segment)")
        for name, values in (("seg", seg), ("child", child)):
            infinite = np.isinf(values)
            if infinite.any():
                rows = np.flatnonzero(infinite.any(axis=tuple(range(1, infinite.ndim))))
                raise ValueError(f"{name} holds an infinite value at row(s) {rows[:5].tolist()}; "
                                 "a measure that could not be computed is NaN")

        for name, value in (("ids", ids), ("seg_names", seg_names), ("child_names", child_names),
                            ("y", y.astype(np.int64)), ("seg", seg), ("mask", mask),
                            ("child", child), ("exposed", exposed)):
            object.__setattr__(self, name, value)

    def select(self, seg_names: Sequence[str] | None = None,
               child_names: Sequence[str] | None = None,
               rows: Sequence[int] | np.ndarray | None = None) -> "Bags":
        """A subset of the measures and of the children, as a new, independent ``Bags``.

        ``seg_names`` and ``child_names`` pick columns by name: ``None`` keeps them all and an
        empty tuple drops that whole layer. The columns keep their original order however the
        names are listed, so a variant is the same Bags whichever way the caller built its
        list. A name this Bags does not have raises.

        ``rows`` are positions (or a boolean mask) and the children come out in the order
        given. ``T`` is kept as it was: the padding of a shorter bag is already 0 and masked.
        A child cannot be selected twice, since one child is one unit (R2).
        """
        s_cols = _columns(self.seg_names, seg_names, "segment")
        c_cols = _columns(self.child_names, child_names, "child")
        pos = (np.arange(len(self.ids), dtype=np.intp) if rows is None
               else _positions(rows, len(self.ids)))
        # fancy indexing copies, so nothing here shares memory with this Bags
        return Bags(
            ids=tuple(self.ids[i] for i in pos),
            y=self.y[pos],
            seg=self.seg[pos][:, :, s_cols],
            mask=self.mask[pos],
            child=self.child[pos][:, c_cols],
            seg_names=tuple(self.seg_names[j] for j in s_cols),
            child_names=tuple(self.child_names[j] for j in c_cols),
            exposed=self.exposed[pos],
        )


def bags_from_arrays(ids: Sequence[str], y: Sequence[int] | np.ndarray,
                     seg_list: Sequence[np.ndarray], child: np.ndarray,
                     seg_names: Sequence[str], child_names: Sequence[str],
                     exposed: Sequence[bool] | np.ndarray) -> Bags:
    """Pad one ``(n_segments, Ds)`` array per child into a ``Bags``.

    ``seg_list[i]`` is what ``segment_measures`` returned for child ``i``. The bags are padded
    with 0 to the longest one and the mask marks the real rows. A child with no segments at
    all (an array with 0 rows) raises ValueError naming its row and id (ruling R10): a real
    session always has exchanges, so an empty one means something upstream is wrong.
    """
    n = len(ids)
    if not len(y) == len(seg_list) == len(child) == len(exposed) == n:
        raise ValueError(f"ids ({n}), y ({len(y)}), seg_list ({len(seg_list)}), child "
                         f"({len(child)}) and exposed ({len(exposed)}) must describe the same "
                         "children")
    d_seg = len(seg_names)
    arrays = []
    for i, a in enumerate(seg_list):
        a = np.asarray(a, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != d_seg:
            raise ValueError(f"seg_list row {i} has shape {a.shape}, expected (segments, {d_seg})")
        arrays.append(a)

    longest = max((len(a) for a in arrays), default=0)
    seg = np.zeros((n, longest, d_seg), dtype=np.float32)
    mask = np.zeros((n, longest), dtype=bool)
    for i, a in enumerate(arrays):
        seg[i, : len(a)] = a
        mask[i, : len(a)] = True
    # np.array copies, so editing the caller's child or exposed array later cannot change this
    # Bags (seg and mask are built here, and y is copied by the cast to int64)
    return Bags(ids=tuple(ids), y=y, seg=seg, mask=mask, child=np.array(child, dtype=np.float32),
                seg_names=tuple(seg_names), child_names=tuple(child_names),
                exposed=np.array(exposed, dtype=bool))


def _indices(values: Sequence[int], size: int, name: str) -> np.ndarray:
    idx = np.asarray(values, dtype=np.intp).reshape(-1)
    if ((idx < 0) | (idx >= size)).any():
        raise ValueError(f"{name} must lie in [0, {size}); got {idx.tolist()}")
    return idx


def synthetic_bags(n_pos: int, n_neg: int, *, signal: float, seed: int,
                   n_seg: tuple[int, int] = (20, 60), d_seg: int = 6, d_child: int = 4,
                   informative_seg: Sequence[int] = (0,), informative_child: Sequence[int] = (0,),
                   signal_fraction: float = 0.3, nan_rate: float = 0.05) -> Bags:
    """A planted-signal cohort for testing the model and the cross-validation (plan section 7).

    Every value is N(0, 1). Each positive (ASD) child has ``round(signal_fraction * n_segments)``
    of its segments picked at random, and ``signal`` is added to the ``informative_seg`` columns
    of those; ``signal`` is also added to its ``informative_child`` columns. Negatives get
    nothing. Last, every value is blanked out (NaN) with probability ``nan_rate``, in real
    segments only. A child has between ``n_seg[0]`` and ``n_seg[1]`` segments, both ends
    included. With ``signal=0`` both groups come from one distribution, so any honest procedure
    scores about 0.5.

    Deterministic for a ``seed``. The signal is added on top of noise that does not depend on
    it, so two calls that differ only in ``signal`` differ only by what was planted.

    Positives come first, ``exposed`` is all False, and ids read ``syn-asd-000`` and
    ``syn-td-000``. The columns carry the real measure names when the sizes match (6 segment
    measures, 4 child measures), so that a variant selected by name on real data can be
    selected the same way here; otherwise they are ``seg0, seg1, ...`` and ``child0, ...``.
    """
    lo, hi = (int(v) for v in n_seg)
    if not 1 <= lo <= hi:
        raise ValueError(f"n_seg must be (low, high) with 1 <= low <= high, got {tuple(n_seg)}")
    if n_pos < 0 or n_neg < 0:
        raise ValueError(f"n_pos and n_neg cannot be negative, got {n_pos} and {n_neg}")
    if d_seg < 0 or d_child < 0:
        raise ValueError(f"d_seg and d_child cannot be negative, got {d_seg} and {d_child}")
    if not 0.0 <= signal_fraction <= 1.0:
        raise ValueError(f"signal_fraction must lie in [0, 1], got {signal_fraction}")
    if not 0.0 <= nan_rate <= 1.0:
        raise ValueError(f"nan_rate must lie in [0, 1], got {nan_rate}")
    informative_seg = _indices(informative_seg, d_seg, "informative_seg")
    informative_child = _indices(informative_child, d_child, "informative_child")

    rng = np.random.default_rng(seed)
    n = n_pos + n_neg
    y = np.concatenate([np.ones(n_pos, dtype=np.int64), np.zeros(n_neg, dtype=np.int64)])
    lengths = rng.integers(lo, hi, size=n, endpoint=True)

    seg_list = []
    for i in range(n):
        x = rng.standard_normal((lengths[i], d_seg))
        if y[i] == 1:
            picked = rng.choice(lengths[i], size=int(round(signal_fraction * lengths[i])),
                                replace=False)
            x[np.ix_(picked, informative_seg)] += signal
        seg_list.append(x)
    child = rng.standard_normal((n, d_child))
    child[np.ix_(y == 1, informative_child)] += signal

    for x in seg_list:
        x[rng.random(x.shape) < nan_rate] = np.nan
    child[rng.random(child.shape) < nan_rate] = np.nan

    seg_names = (SEGMENT_MEASURES if d_seg == len(SEGMENT_MEASURES)
                 else tuple(f"seg{j}" for j in range(d_seg)))
    child_names = (CHILD_MEASURES if d_child == len(CHILD_MEASURES)
                   else tuple(f"child{j}" for j in range(d_child)))
    ids = (tuple(f"syn-asd-{i:03d}" for i in range(n_pos))
           + tuple(f"syn-td-{i:03d}" for i in range(n_neg)))
    return bags_from_arrays(ids, y, seg_list, child, seg_names, child_names,
                            np.zeros(n, dtype=bool))
