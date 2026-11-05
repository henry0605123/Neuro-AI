"""Milestone M1: the data contract and the generator."""

from __future__ import annotations

import numpy as np
import pytest

from asdpipe.contracts import Label, Speaker, SubjectTable, SyllableSpan
from asdpipe.synth.generator import TRAIT_ORDER, SynthConfig, generate_corpus, generate_subject


def test_signal_zero_makes_the_label_irrelevant():
    """R-critical: at s=0 the two groups must be drawn from identical distributions.

    The strongest possible version of that claim: flipping one subject's label changes
    nothing at all -- not the audio, not the transcript, not the ADOS codes.
    """
    cfg = SynthConfig(n_subjects=4, turns_per_subject=12, signal_strength=0.0)
    asd = generate_subject(0, Label.ASD, cfg)
    ctrl = generate_subject(0, Label.CONTROL, cfg)
    assert np.array_equal(asd.audio, ctrl.audio)
    assert [t.text for t in asd.session.turns] == [t.text for t in ctrl.session.turns]
    assert asd.ados == ctrl.ados
    assert asd.latents == ctrl.latents


def test_signal_strength_only_shifts_the_asd_group():
    cfg0 = SynthConfig(n_subjects=4, turns_per_subject=12, signal_strength=0.0)
    cfg1 = cfg0.replace(signal_strength=1.5)
    assert np.array_equal(generate_subject(1, Label.CONTROL, cfg0).audio,
                          generate_subject(1, Label.CONTROL, cfg1).audio)
    assert not np.array_equal(generate_subject(1, Label.ASD, cfg0).audio,
                              generate_subject(1, Label.ASD, cfg1).audio)


def test_injection_directions_follow_the_spec():
    """rhythm/pause/latency/echolalia up, topic relevance down, at s>0."""
    cfg = SynthConfig(n_subjects=2, turns_per_subject=8, signal_strength=2.0)
    a = generate_subject(0, Label.ASD, cfg).latents
    c = generate_subject(0, Label.CONTROL, cfg).latents
    assert a["rhythm_variability"] > c["rhythm_variability"]
    assert a["pause_anomaly"] > c["pause_anomaly"]
    assert a["response_latency"] > c["response_latency"]
    assert a["echolalia_similarity"] > c["echolalia_similarity"]
    assert a["topic_relevance"] < c["topic_relevance"]


def test_generator_is_deterministic():
    cfg = SynthConfig(n_subjects=3, turns_per_subject=10, seed=7)
    a, b = generate_corpus(cfg), generate_corpus(cfg)
    for x, y in zip(a.subjects, b.subjects):
        assert np.array_equal(x.audio, y.audio)
        assert x.ados == y.ados


def test_corpus_shape_and_labels():
    corpus = generate_corpus(SynthConfig(n_subjects=12, asd_ratio=0.5, turns_per_subject=10))
    assert len(corpus) == 12
    assert int(corpus.labels.sum()) == 6


def test_ados_codes_are_ordinal_0_to_3():
    corpus = generate_corpus(SynthConfig(n_subjects=8, turns_per_subject=10))
    for s in corpus.subjects:
        for item in ("A2", "A3", "A4", "A8", "OQR"):
            assert s.ados[item] in (0, 1, 2, 3)


def test_every_turn_carries_ground_truth_syllables_and_tones():
    corpus = generate_corpus(SynthConfig(n_subjects=2, turns_per_subject=10))
    for s in corpus.subjects:
        for t in s.session.turns:
            assert t.syllables, "R6: the synthetic path supplies ground-truth boundaries"
            for a, b in zip(t.syllables[:-1], t.syllables[1:]):
                assert a.t_end <= b.t_start
            for syl in t.syllables:
                assert syl.tone in (0, 1, 2, 3, 4)
                assert syl.underlying_tone in (0, 1, 2, 3, 4)


def test_child_f0_stays_in_the_100_600_band():
    """Section 9.2.1: the child band is 100-600 Hz, so the audio must live there."""
    import librosa

    corpus = generate_corpus(SynthConfig(n_subjects=1, turns_per_subject=10))
    s = corpus.subjects[0]
    kid = [t for t in s.session.turns if t.speaker is Speaker.CHILD][0]
    a = int(kid.t_start * s.sample_rate)
    b = int(kid.t_end * s.sample_rate)
    f0, voiced, _ = librosa.pyin(s.audio[a:b].astype(float), fmin=100, fmax=600,
                                 sr=s.sample_rate)
    got = f0[np.isfinite(f0)]
    assert got.size > 0
    assert got.min() >= 100.0 and got.max() <= 600.0


def test_subject_table_refuses_duplicate_subjects():
    """R2 in the type system: one subject can never become two rows."""
    from asdpipe.contracts import SubjectRow

    def row(sid):
        return SubjectRow(subject_id=sid, label=Label.ASD, b_pooled={"x": 1.0},
                          c_pooled={"y": 1.0}, e_pooled={"z": 1.0},
                          c_sequence=np.zeros((2, 11)), egemaps_functionals=np.zeros(88),
                          examiner_probe={"p": 0.0}, ados={}, latents={}, quality={})

    with pytest.raises(ValueError):
        SubjectTable([row("A"), row("A")])


def test_syllable_span_rejects_impossible_values():
    with pytest.raises(ValueError):
        SyllableSpan(1.0, 1.0, 1)
    with pytest.raises(ValueError):
        SyllableSpan(0.0, 1.0, 9)


def test_trait_order_is_stable():
    """The RNG draws in this order; reordering silently changes every past corpus."""
    assert TRAIT_ORDER[:6] == ("rhythm_variability", "pause_anomaly", "response_latency",
                               "echolalia_similarity", "topic_relevance", "residual_f0_sd")
