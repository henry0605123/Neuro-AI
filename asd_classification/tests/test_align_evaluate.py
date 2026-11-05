"""Alignment-quality measurement. Hand-computed, no MFA required."""

from __future__ import annotations

import numpy as np
import pytest

from asdpipe.align.evaluate import (boundary_energy_agreement, cross_alignment_agreement,
                                    duration_sanity, implied_rate_profile, oov_report,
                                    read_dictionary_words, write_textgrid)
from asdpipe.align.mfa import parse_textgrid_intervals
from asdpipe.contracts import SyllableSpan


def test_read_dictionary_words_takes_the_orthographic_column(tmp_path):
    p = tmp_path / "d.dict"
    p.write_text("媽\t0.1\t0.2\t1.0\t1.0\tm a˥\n"
                 "媽\t0.9\t0.1\t1.0\t1.0\tm a˧˥\n"
                 "# comment\n"
                 "馬\t0.1\t0.2\t1.0\t1.0\tm a˨˩˦\n", encoding="utf-8")
    assert read_dictionary_words(p) == {"媽", "馬"}


def test_oov_report_hand_example():
    """Tokens 甲 乙 甲 丙 against a lexicon holding only 甲 and 乙.

    types = {甲, 乙, 丙} -> 1 of 3 unknown = 33.3%
    tokens = 4, of which 丙 appears once -> 25.0%
    """
    words = {0: ["甲", "乙"], 1: ["甲", "丙"]}
    r = oov_report(words, {"甲", "乙"})
    assert r["total_tokens"] == 4 and r["vocab_types"] == 3
    assert r["oov_types"] == 1 and r["oov_tokens"] == 1
    assert r["oov_type_rate"] == pytest.approx(1 / 3)
    assert r["oov_token_rate"] == pytest.approx(0.25)
    assert r["examples"] == ["丙"]


def test_duration_sanity_hand_example():
    """Durations 0.02, 0.20, 0.30, 2.0 s -> median 0.25 s, two outside [0.05, 1.0]."""
    spans = [SyllableSpan(0.0, 0.02, 1), SyllableSpan(1.0, 1.20, 1),
             SyllableSpan(2.0, 2.30, 1), SyllableSpan(3.0, 5.0, 1)]
    d = duration_sanity(spans)
    assert d["n"] == 4
    assert d["median_ms"] == pytest.approx(250.0)
    assert d["n_implausible"] == 2
    assert d["implausible_rate"] == pytest.approx(0.5)


def test_duration_sanity_handles_an_empty_alignment():
    d = duration_sanity([])
    assert d["n"] == 0 and d["n_implausible"] == 0


def test_cross_alignment_agreement_hand_example():
    """Two alignments of one turn; starts differ by 0 and 30 ms, ends by 10 ms.

    Compared values are [0, 30, 10] -> median 10 ms, 2 of 3 within 20 ms.
    """
    a = {7: [SyllableSpan(0.00, 0.20, 1), SyllableSpan(0.20, 0.40, 2)]}
    b = {7: [SyllableSpan(0.00, 0.20, 1), SyllableSpan(0.23, 0.41, 2)]}
    r = cross_alignment_agreement(a, b)
    assert r["n"] == 3 and r["n_turns"] == 1
    assert r["median_ms"] == pytest.approx(10.0)
    assert r["within_20ms"] == pytest.approx(2 / 3)


def test_cross_alignment_skips_turns_with_different_syllable_counts():
    a = {1: [SyllableSpan(0.0, 0.2, 1)], 2: [SyllableSpan(0.0, 0.2, 1)]}
    b = {1: [SyllableSpan(0.0, 0.2, 1), SyllableSpan(0.2, 0.4, 1)],
         2: [SyllableSpan(0.0, 0.2, 1)]}
    r = cross_alignment_agreement(a, b)
    assert r["n_turns"] == 1          # only turn 2 is comparable


def test_boundary_energy_agreement_finds_a_planted_trough():
    """Two 0.2 s tone bursts with a silent notch exactly at the shared boundary.

    The boundary is at 0.20 s and the energy minimum is there too, so the distance must be
    small. This is only a sanity check on the proxy itself, not on alignment accuracy.
    """
    sr = 16000
    t = np.arange(int(0.4 * sr)) / sr
    x = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    notch = slice(int(0.195 * sr), int(0.205 * sr))
    x[notch] = 0.0
    spans = [SyllableSpan(0.0, 0.2, 1), SyllableSpan(0.2, 0.4, 2)]
    r = boundary_energy_agreement(x, sr, spans)
    assert r["n"] == 1
    assert r["median_ms"] < 20.0


def test_boundary_energy_agreement_needs_touching_syllables():
    """A gap between syllables is a pause, not a syllable boundary; it is not scored."""
    sr = 16000
    x = np.random.default_rng(0).standard_normal(int(0.6 * sr)).astype(np.float32) * 0.1
    spans = [SyllableSpan(0.0, 0.2, 1), SyllableSpan(0.35, 0.55, 2)]
    assert boundary_energy_agreement(x, sr, spans)["n"] == 0


def test_textgrid_round_trip(tmp_path):
    """What we write for the human spot-check must be readable by our own parser."""
    spans = [SyllableSpan(0.00, 0.18, 1, text="ma"),
             SyllableSpan(0.18, 0.36, 3, text="li")]
    p = write_textgrid(tmp_path / "x.TextGrid", spans, 0.0, 0.36)
    intervals = parse_textgrid_intervals(p.read_text(encoding="utf-8"),
                                         tier_name="syllables")
    assert len(intervals) == 2
    assert intervals[0][0] == pytest.approx(0.0)
    assert intervals[1][1] == pytest.approx(0.36)
    assert "T1" in intervals[0][2] and "T3" in intervals[1][2]


# ---------------------------------------------------------------- implied articulation rate
def _even(n, syllable_s, gap_s=0.0, t0=0.0):
    spans, t = [], t0
    for _ in range(n):
        spans.append(SyllableSpan(t, t + syllable_s, 1))
        t += syllable_s + gap_s
    return spans


def test_implied_rate_hand_example():
    """10 syllables of 100 ms with no gaps span 1.0 s -> exactly 10 syllables/second."""
    prof = implied_rate_profile({0: _even(10, 0.1)})
    assert prof["n"] == 1
    bin10 = next(b for b in prof["bins"] if b["min_syllables"] == 10)
    assert bin10["median_rate"] == pytest.approx(10.0)
    assert bin10["median_syllable_ms"] == pytest.approx(100.0)
    assert bin10["syllable_coverage"] == pytest.approx(1.0)
    assert bin10["frac_implausible"] == pytest.approx(1.0)     # 10 > 8


def test_implied_rate_counts_gaps_against_the_span():
    """10 syllables of 100 ms separated by 100 ms cover 1.9 s -> 5.3/s, coverage 53%."""
    prof = implied_rate_profile({0: _even(10, 0.1, gap_s=0.1)})
    b = prof["bins"][0]
    assert b["median_rate"] == pytest.approx(10 / 1.9, abs=1e-6)
    assert b["syllable_coverage"] == pytest.approx(1.0 / 1.9, abs=1e-6)
    assert b["frac_implausible"] == pytest.approx(0.0)


def test_implied_rate_separates_short_from_long_utterances():
    """Short backchannels legitimately run fast; only the long bin constrains rhythm features."""
    spans = {0: _even(3, 0.09), 1: _even(3, 0.09), 2: _even(12, 0.2), 3: _even(12, 0.2)}
    prof = implied_rate_profile(spans)
    short = next(b for b in prof["bins"] if b["min_syllables"] == 2)
    long_ = next(b for b in prof["bins"] if b["min_syllables"] == 10)
    assert short["median_rate"] > 8 and short["frac_implausible"] == pytest.approx(1.0)
    assert long_["median_rate"] == pytest.approx(5.0) and long_["frac_implausible"] == 0.0
    # the summary the report quotes uses only utterances of >= 10 syllables
    assert prof["long_utterances"]["n"] == 2
    assert prof["long_utterances"]["median"] == pytest.approx(5.0)
    assert prof["long_utterances"]["frac_implausible"] == pytest.approx(0.0)


def test_implied_rate_ignores_single_syllable_turns():
    assert implied_rate_profile({0: _even(1, 0.2)})["n"] == 0
