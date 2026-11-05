"""Bags: padding, selection and the synthetic cohort, hand-checked.

The first three tests are the ones the task brief specifies, and the fourth is controller ruling
R10 (a child with no segments is refused). The block after them pins decisions the brief leaves
open, so that a later edit cannot change them unnoticed: they are what the model, the
cross-validation and the analysis variants all lean on.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from asdpipe.contracts import Speaker
from asdpipe.mil.bags import Bags, bags_from_arrays, synthetic_bags
from asdpipe.mil.segments import CHILD_MEASURES, SEGMENT_MEASURES, segment_measures
from conftest import make_turn

SEG6, CH4 = SEGMENT_MEASURES, CHILD_MEASURES


def _two_children() -> Bags:
    """Child "a" holds 2 segments of ones, child "b" holds 5; every child measure is 0."""
    return bags_from_arrays(("a", "b"), [0, 1], [np.ones((2, 6)), np.ones((5, 6))],
                            np.zeros((2, 4)), SEG6, CH4, [False, False])


def _labelled_bags() -> Bags:
    """Three children whose every value says where it came from: 10 * child + column."""
    seg = [np.add.outer(np.zeros(k + 1), 10 * k + np.arange(6)) for k in range(3)]
    child = 10 * np.arange(3)[:, None] + np.arange(4)
    return bags_from_arrays(("a", "b", "c"), [0, 1, 1], seg, child, SEG6, CH4,
                            [False, False, True])


# ---------------------------------------------------------------- the brief's three tests
def test_padding_and_mask():
    b = bags_from_arrays(("a", "b"), [0, 1], [np.ones((2, 6)), np.ones((5, 6))], np.zeros((2, 4)), SEG6, CH4, [False, False])
    assert b.seg.shape == (2, 5, 6) and b.mask.sum(1).tolist() == [2, 5] and (b.seg[0, 2:] == 0).all()


def test_select_keeps_order():
    b = _two_children()
    s = b.select(seg_names=("e_e5_talk_share",), child_names=(), rows=np.array([1]))
    assert s.seg.shape[2] == 1 and s.child.shape == (1, 0) and s.ids == ("b",)


def test_synthetic_null_and_signal():
    null = synthetic_bags(200, 200, signal=0.0, seed=0)
    assert abs(np.nanmean(null.child[null.y == 1, 0]) - np.nanmean(null.child[null.y == 0, 0])) < 0.2
    sig = synthetic_bags(200, 200, signal=2.0, seed=0)
    assert np.nanmean(sig.child[sig.y == 1, 0]) - np.nanmean(sig.child[sig.y == 0, 0]) > 1.5


# ---------------------------------------------------------------- ruling R10
def test_a_child_with_no_segments_is_refused():
    # segment_measures returns shape (0, 6) for a session with no exchanges. Fail loudly and say
    # which child, rather than let an empty bag reach the attention softmax (0 / 0 = NaN).
    with pytest.raises(ValueError, match=r"\(1, 'b'\)"):
        bags_from_arrays(("a", "b"), [0, 1], [np.ones((2, 6)), np.empty((0, 6))],
                         np.zeros((2, 4)), SEG6, CH4, [False, False])


# ------------------------------------------- beyond the brief: decisions it leaves open
def test_dtypes_and_the_missing_versus_padding_contract():
    seg = [np.array([[np.nan, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]), np.ones((3, 6))]
    child = np.array([[np.nan, 0, 0, 0], [1, 2, 3, 4]])
    b = bags_from_arrays(("a", "b"), [1, 0], seg, child, SEG6, CH4, [True, False])
    assert b.seg.dtype == np.float32 and b.child.dtype == np.float32
    assert b.y.dtype == np.int64 and b.mask.dtype == np.bool_ and b.exposed.dtype == np.bool_
    assert np.isnan(b.seg[0, 0, 0]) and np.isnan(b.child[0, 0])      # unmeasurable stays NaN
    assert b.seg[0, 1].tolist() == [6, 7, 8, 9, 10, 11]
    assert (b.seg[0, 2:] == 0).all()                                 # padding is 0, never NaN
    assert b.y.tolist() == [1, 0] and b.exposed.tolist() == [True, False]
    assert b.ids == ("a", "b") and b.seg_names == SEG6 and b.child_names == CH4


def test_bags_from_arrays_owns_its_data():
    seg = [np.ones((2, 6)), np.ones((3, 6))]
    child, exposed = np.zeros((2, 4), np.float32), np.zeros(2, bool)    # already the right dtypes
    b = bags_from_arrays(("a", "b"), [0, 1], seg, child, SEG6, CH4, exposed)
    seg[0][:] = 5
    child[:] = 5
    exposed[:] = True
    assert (b.seg[0, :2] == 1).all() and (b.child == 0).all() and not b.exposed.any()


def test_segment_measures_output_feeds_straight_in(cfg):
    # the real path: one (n_segments, 6) array per child, NaN where a measure was not computable
    E, C = Speaker.EXAMINER, Speaker.CHILD
    talk = [make_turn(0, E, 0, 7, "你好"), make_turn(1, C, 7, 10, "你好嗎")]
    silent = [make_turn(0, E, 0, 2, "你好"), make_turn(1, E, 3, 5, "你叫什麼名字")]
    per_child = [segment_measures(talk, {1: ["你好", "嗎"]}, cfg, 5),
                 segment_measures(silent, {}, cfg, 5)]
    b = bags_from_arrays(("a", "b"), [0, 1], per_child, np.zeros((2, 4)), SEG6, CH4,
                         [False, False])
    assert b.seg.shape == (2, 1, 6) and b.mask.all()
    assert b.seg[0, 0, SEG6.index("e_e5_talk_share")] == pytest.approx(0.3)   # 3 s of 10 s
    assert np.isnan(b.seg[1, 0, SEG6.index("c_c9_chars_per_utt")])            # a silent child ...
    assert b.seg[1, 0, SEG6.index("e_e5_talk_share")] == 0.0                  # ... has a share of 0
    # a session with no exchanges has no segments, shape (0, 6), and that child is refused (R10)
    with pytest.raises(ValueError, match=r"\(0, 'a'\)"):
        bags_from_arrays(("a",), [0], [segment_measures([], {}, cfg, 5)], np.zeros((1, 4)),
                         SEG6, CH4, [False])


def test_bags_from_arrays_refuses_mismatched_inputs():
    ok = dict(child=np.zeros((2, 4)), seg_names=SEG6, child_names=CH4, exposed=[False, False])
    with pytest.raises(ValueError, match="same children"):
        bags_from_arrays(("a", "b"), [0], [np.ones((2, 6)), np.ones((2, 6))], **ok)
    with pytest.raises(ValueError, match="row 0"):                    # 5 columns, 6 names
        bags_from_arrays(("a", "b"), [0, 1], [np.ones((2, 5)), np.ones((2, 6))], **ok)
    with pytest.raises(ValueError, match="row 1"):                    # not (segments, measures)
        bags_from_arrays(("a", "b"), [0, 1], [np.ones((2, 6)), np.ones(6)], **ok)


def test_rows_come_out_in_the_order_given_and_names_in_the_original_order():
    b = _labelled_bags()
    s = b.select(rows=np.array([2, 0]))
    assert s.ids == ("c", "a") and s.y.tolist() == [1, 0] and s.exposed.tolist() == [True, False]
    assert s.mask.sum(1).tolist() == [3, 1]
    assert s.seg[0, 0].tolist() == [20, 21, 22, 23, 24, 25]
    assert s.seg[1, 0].tolist() == [0, 1, 2, 3, 4, 5]
    assert s.child.tolist() == [[20, 21, 22, 23], [0, 1, 2, 3]]
    # names pick columns. However they are listed, the columns keep their original order, so a
    # variant is the same Bags whatever order (or set iteration) the caller built the names in
    n = b.select(seg_names=(SEG6[4], SEG6[1]), child_names=(CH4[3], CH4[0]))
    assert n.seg_names == (SEG6[1], SEG6[4]) and n.child_names == (CH4[0], CH4[3])
    assert n.seg[2, 0].tolist() == [21, 24] and n.child[2].tolist() == [20, 23]
    assert n.ids == b.ids                                             # no rows given: all rows


def test_select_takes_a_boolean_mask_and_returns_independent_copies():
    b = _two_children()
    s = b.select(rows=np.array([False, True]))
    assert s.ids == ("b",) and s.mask.sum(1).tolist() == [5]
    whole = b.select()
    assert whole.ids == b.ids and whole.seg.shape == b.seg.shape
    whole.seg[:] = 7
    whole.child[:] = 7
    whole.mask[:] = False
    assert (b.seg[0, :2] == 1).all() and (b.child == 0).all() and b.mask.sum() == 7


def test_empty_name_tuples_drop_a_layer_and_nothing_else():
    b = _two_children()
    no_seg = b.select(seg_names=())
    assert no_seg.seg.shape == (2, 5, 0) and no_seg.seg_names == ()
    assert no_seg.mask.sum(1).tolist() == [2, 5] and no_seg.child.shape == (2, 4)
    no_child = b.select(child_names=())
    assert no_child.child.shape == (2, 0) and no_child.child_names == ()
    assert no_child.seg.shape == b.seg.shape
    none = b.select(rows=np.array([], dtype=int))                    # no rows is a valid subset
    assert none.ids == () and none.seg.shape[0] == 0 and none.child.shape == (0, 4)


def test_select_refuses_what_it_cannot_honour():
    b = _two_children()
    # a typo in a measure name must not quietly select fewer columns than asked for
    with pytest.raises(ValueError, match="e_e5_talk_shares"):
        b.select(seg_names=("e_e5_talk_shares",))
    with pytest.raises(ValueError, match="no_such"):
        b.select(child_names=("no_such",))
    assert b.select(seg_names="e_e5_talk_share").seg_names == ("e_e5_talk_share",)  # bare str
    # a one-shot generator must not be used up by the check and then select nothing
    rest = b.select(seg_names=(n for n in SEG6 if n != SEG6[0]), child_names=iter(CH4[2:]))
    assert rest.seg_names == SEG6[1:] and rest.child_names == CH4[2:]
    with pytest.raises(ValueError, match="unique"):                  # one child is one unit (R2)
        b.select(rows=np.array([1, 1]))
    with pytest.raises(IndexError):
        b.select(rows=np.array([2]))
    with pytest.raises(ValueError, match="1-D"):
        b.select(rows=np.array([[0, 1]]))


def test_bags_are_frozen():
    b = _two_children()
    with pytest.raises(dataclasses.FrozenInstanceError):
        b.y = np.array([1, 1])


def _parts(**override):
    """Valid constructor arguments for two children (rows 0 and 1 hold 2 and 3 segments)."""
    parts = dict(ids=("a", "b"), y=np.array([0, 1]), seg=np.zeros((2, 3, 6), np.float32),
                 mask=np.array([[True, True, False], [True, True, True]]),
                 child=np.zeros((2, 4), np.float32), seg_names=SEG6, child_names=CH4,
                 exposed=np.zeros(2, bool))
    parts.update(override)
    return parts


def _seg_with(row, col, value):
    seg = np.zeros((2, 3, 6), np.float32)
    seg[row, col, 0] = value
    return seg


def test_bags_accept_consistent_parts():
    b = Bags(**_parts())
    assert b.y.dtype == np.int64 and b.seg.dtype == np.float32 and b.mask.dtype == np.bool_
    # the arrays are normalised, so a float64 seg or a list of labels also works
    b = Bags(**_parts(seg=np.zeros((2, 3, 6)), y=[0, 1], exposed=[False, True]))
    assert b.seg.dtype == np.float32 and b.y.dtype == np.int64
    assert b.exposed.tolist() == [False, True]


@pytest.mark.parametrize("override, message", [
    (dict(ids=("a", "a")), "ids must be unique"),
    (dict(y=np.array([0, 2])), "y must be 0"),
    (dict(y=np.array([0.0, 0.5])), "y must be 0"),
    (dict(y=np.array([0, 1, 1])), "y has shape"),
    (dict(seg=np.zeros((2, 3, 5), np.float32)), "seg has shape"),
    (dict(seg=np.zeros((2, 3), np.float32)), "seg must be"),
    (dict(child=np.zeros((2, 3), np.float32)), "child has shape"),
    (dict(mask=np.ones((2, 2), bool)), "mask has shape"),
    (dict(exposed=np.zeros(3, bool)), "exposed has shape"),
    (dict(seg_names=(SEG6[0],) * 6), "seg_names must be unique"),
    (dict(child_names=(CH4[0],) * 4), "child_names must be unique"),
    (dict(child_names=(SEG6[0],) + CH4[1:]), "both"),
    (dict(mask=np.array([[True, True, False], [False, False, False]])), r"\(1, 'b'\)"),
    (dict(seg=_seg_with(0, 2, 1.0)), "padding"),                 # a value in a padding row
    (dict(seg=_seg_with(0, 2, np.nan)), "padding"),              # a NaN in a padding row
    (dict(seg=_seg_with(1, 0, np.inf)), "infinite"),
    (dict(child=np.array([[0, 0, 0, 0], [-np.inf, 0, 0, 0]], np.float32)), "infinite"),
])
def test_bags_refuse_inconsistent_parts(override, message):
    with pytest.raises(ValueError, match=message):
        Bags(**_parts(**override))


# --------------------------------------------------------------------- synthetic cohort
def test_synthetic_columns_take_the_measure_names_when_the_sizes_match():
    b = synthetic_bags(2, 2, signal=0.0, seed=0)                       # R1: 6 and 4 columns
    assert b.seg_names == SEG6 and b.child_names == CH4
    odd = synthetic_bags(2, 2, signal=0.0, seed=0, d_seg=3, d_child=2,
                         informative_seg=(2,), informative_child=(1,))
    assert odd.seg_names == ("seg0", "seg1", "seg2") and odd.child_names == ("child0", "child1")
    assert odd.seg.shape[2] == 3 and odd.child.shape == (4, 2)


def test_synthetic_labels_come_positives_first_and_nobody_is_exposed():
    b = synthetic_bags(3, 4, signal=1.0, seed=0)
    assert b.y.tolist() == [1, 1, 1, 0, 0, 0, 0] and b.y.dtype == np.int64
    assert len(set(b.ids)) == 7 and not b.exposed.any()


def test_synthetic_is_deterministic_for_a_seed():
    a = synthetic_bags(5, 6, signal=1.0, seed=3)
    b = synthetic_bags(5, 6, signal=1.0, seed=3)
    c = synthetic_bags(5, 6, signal=1.0, seed=4)
    assert a.ids == b.ids and a.seg_names == b.seg_names
    for field in ("y", "seg", "mask", "child", "exposed"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))   # NaNs included
    assert not np.array_equal(a.child, c.child, equal_nan=True)


def test_the_signal_is_added_on_top_of_the_same_noise():
    # the same seed with a different signal changes only what was planted: negatives, the
    # segment counts and the blanked-out cells are all identical
    null = synthetic_bags(8, 8, signal=0.0, seed=5)
    sig = synthetic_bags(8, 8, signal=2.0, seed=5)
    np.testing.assert_array_equal(null.mask, sig.mask)
    np.testing.assert_array_equal(np.isnan(null.seg), np.isnan(sig.seg))
    np.testing.assert_array_equal(np.isnan(null.child), np.isnan(sig.child))
    np.testing.assert_array_equal(null.seg[null.y == 0], sig.seg[sig.y == 0])
    np.testing.assert_array_equal(null.child[null.y == 0], sig.child[sig.y == 0])
    assert not np.array_equal(null.child[null.y == 1], sig.child[sig.y == 1], equal_nan=True)


def test_signal_lands_on_a_fraction_of_the_positive_segments_in_the_informative_columns():
    # no blanking and a shift of 10 make every planted value unmistakable against N(0, 1) noise
    b = synthetic_bags(30, 30, signal=10.0, seed=1, nan_rate=0.0,
                       informative_seg=(2, 4), informative_child=(1,))
    n_seg = b.mask.sum(1)
    for col in range(6):
        planted = ((b.seg[:, :, col] > 5) & b.mask).sum(1)
        if col in (2, 4):
            # a positive child gets round(0.3 * its segments) of them; a negative child none
            np.testing.assert_array_equal(planted, np.where(b.y == 1, np.round(0.3 * n_seg), 0))
        else:
            assert planted.sum() == 0
    high = b.child > 5
    assert high[:, 1].tolist() == (b.y == 1).tolist() and not high[:, [0, 2, 3]].any()
    # signal_fraction is a parameter: all of the segments when it is 1
    full = synthetic_bags(5, 5, signal=10.0, seed=1, nan_rate=0.0, signal_fraction=1.0)
    assert np.array_equal(((full.seg[:, :, 0] > 5) & full.mask).sum(1)[full.y == 1],
                          full.mask.sum(1)[full.y == 1])


def test_nan_rate_blanks_values_but_never_padding():
    b = synthetic_bags(40, 40, signal=0.0, seed=2, nan_rate=0.2)
    assert abs(np.isnan(b.seg[b.mask]).mean() - 0.2) < 0.02
    assert abs(np.isnan(b.child).mean() - 0.2) < 0.08                 # 320 values only
    assert (b.seg[~b.mask] == 0).all()                                # padding stays 0
    clean = synthetic_bags(5, 5, signal=0.0, seed=2, nan_rate=0.0)
    assert not np.isnan(clean.seg).any() and not np.isnan(clean.child).any()


def test_n_seg_is_an_inclusive_range_and_one_segment_bags_work():
    b = synthetic_bags(100, 100, signal=0.0, seed=0, n_seg=(3, 5))
    n = b.mask.sum(1)
    assert n.min() == 3 and n.max() == 5 and b.seg.shape[1] == 5      # both ends are drawn
    one = synthetic_bags(3, 3, signal=1.0, seed=0, n_seg=(1, 1))
    assert one.seg.shape[:2] == (6, 1) and one.mask.all()


@pytest.mark.parametrize("bad", [
    dict(informative_seg=(6,)), dict(informative_seg=(-1,)), dict(informative_child=(4,)),
    dict(signal_fraction=1.5), dict(nan_rate=-0.1), dict(nan_rate=1.5),
    dict(n_seg=(0, 5)), dict(n_seg=(5, 3)), dict(d_seg=-1),
])
def test_synthetic_refuses_arguments_that_make_no_sense(bad):
    with pytest.raises(ValueError):
        synthetic_bags(2, 2, signal=1.0, seed=0, **bad)


def test_the_module_needs_numpy_only():
    # later CPU-only code (the checks, the report) imports Bags and must not pull torch in
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import asdpipe.mil.bags; "
            "sys.exit(1 if 'torch' in sys.modules else 0)")
    src = str(Path(__file__).resolve().parents[1] / "src")
    assert subprocess.run([sys.executable, "-c", code, src]).returncode == 0
