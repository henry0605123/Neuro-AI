"""R6: the aligner is an injectable interface. MFA is verified separately and skipped when
its environment is absent -- the smoke test must never fail because MFA is missing."""

from __future__ import annotations

import numpy as np
import pytest

from asdpipe.align.base import AlignmentUnavailable
from asdpipe.contracts import Speaker, SyllableSpan, Turn
from asdpipe.align.gt import GroundTruthAligner
from asdpipe.align.mfa import (MFAAligner, mfa_binary, parse_textgrid_intervals,
                               syllabify_phones)
from asdpipe.synth.generator import SynthConfig, generate_corpus


def test_ground_truth_aligner_returns_the_generator_boundaries():
    corpus = generate_corpus(SynthConfig(n_subjects=1, turns_per_subject=8))
    s = corpus.subjects[0]
    out = GroundTruthAligner().align(s.audio, s.sample_rate, s.session.turns)
    assert set(out) == {t.turn_id for t in s.session.turns}
    for t in s.session.turns:
        assert out[t.turn_id] == t.syllables


def test_ground_truth_aligner_refuses_a_turn_without_boundaries():
    from asdpipe.contracts import Speaker, SyllableSpan, Turn

    turn = Turn(0, Speaker.CHILD, 0.0, 1.0, "沒有邊界")
    with pytest.raises(ValueError, match="no ground-truth syllables"):
        GroundTruthAligner().align(np.zeros(16000), 16000, [turn])


def test_mfa_adapter_refuses_to_construct_without_explicit_models():
    """QUESTIONS Q14: the acoustic model and dictionary must be named, never guessed."""
    with pytest.raises(AlignmentUnavailable, match="must be set"):
        MFAAligner(acoustic_model="", dictionary="")


def test_textgrid_parsing_hand_example():
    grid = """File type = "ooTextFile"
item [1]:
    name = "phones"
    intervals [1]:
        xmin = 0.00
        xmax = 0.10
        text = "n"
    intervals [2]:
        xmin = 0.10
        xmax = 0.30
        text = "i3"
    intervals [3]:
        xmin = 0.30
        xmax = 0.35
        text = "sil"
    intervals [4]:
        xmin = 0.35
        xmax = 0.55
        text = "hao3"
"""
    intervals = parse_textgrid_intervals(grid)
    assert [lbl for _, _, lbl in intervals] == ["n", "i3", "sil", "hao3"]
    syls = syllabify_phones(intervals)
    assert len(syls) == 2
    assert syls[0].t_start == pytest.approx(0.0) and syls[0].t_end == pytest.approx(0.30)
    assert syls[0].tone == 3
    assert syls[1].tone == 3


def test_syllabify_maps_tone_digit_five_to_the_neutral_tone():
    """Legacy pinyin-digit dictionaries (mandarin_pinyin v2.0.0)."""
    syls = syllabify_phones([(0.0, 0.1, "d"), (0.1, 0.2, "e5")])
    assert len(syls) == 1 and syls[0].tone == 0


# ---------------------------------------------------------------- IPA tone letters (MFA 3.x)
@pytest.mark.parametrize("nucleus,expected", [
    ("a˥", 1),          # 媽 m a˥
    ("a˧˥", 2),    # 麻 m a˧˥
    ("a˨˩˦", 3),  # 馬 m a˨˩˦
    ("a˥˩", 4),    # 罵 m a˥˩
    ("a˩", 0),          # reduced in non-initial position
    ("a", 0),                # no tone letter at all -> neutral
])
def test_ipa_tone_letters_map_to_the_right_category(nucleus, expected):
    """MFA 3.x writes Chao tone letters, not pinyin digits.

    Verified against words whose tone is not in doubt: 媽/麻/馬/罵 and 八/拔/把/爸.
    A digit-suffix rule returns neutral tone for every one of these.
    """
    syls = syllabify_phones([(0.0, 0.05, "m"), (0.05, 0.25, nucleus)])
    assert len(syls) == 1 and syls[0].tone == expected


def test_syllabify_attaches_the_nasal_coda_to_its_nucleus():
    """忙 m a˧˥ ŋ is one syllable, not two."""
    syls = syllabify_phones([(0.0, 0.05, "m"), (0.05, 0.20, "a˧˥"),
                             (0.20, 0.28, "ŋ")])
    assert len(syls) == 1
    assert syls[0].tone == 2
    assert syls[0].t_start == pytest.approx(0.0) and syls[0].t_end == pytest.approx(0.28)


def test_syllabify_splits_a_two_syllable_word():
    """東西 tʷ u˥ ŋ ɕ i˩ -> T1 then neutral; the onset ɕ starts the second syllable."""
    syls = syllabify_phones([(0.0, 0.05, "tʷ"), (0.05, 0.18, "u˥"),
                             (0.18, 0.24, "ŋ"), (0.24, 0.30, "ɕ"),
                             (0.30, 0.45, "i˩")])
    assert [s.tone for s in syls] == [1, 0]
    assert syls[0].t_end == pytest.approx(0.24)
    assert syls[1].t_start == pytest.approx(0.24)


def test_syllabify_splits_when_two_nuclei_are_adjacent():
    """公園 ... ɥ e˧˥ n: the glide ɥ is an onset, so 園 is its own syllable."""
    syls = syllabify_phones([(0.0, 0.05, "kʷ"), (0.05, 0.15, "u˥"),
                             (0.15, 0.20, "ŋ"), (0.20, 0.24, "ɥ"),
                             (0.24, 0.38, "e˧˥"), (0.38, 0.44, "n")])
    assert [s.tone for s in syls] == [1, 2]


def test_silence_flushes_the_current_syllable():
    syls = syllabify_phones([(0.0, 0.05, "m"), (0.05, 0.20, "a˥"),
                             (0.20, 0.40, "sil"),
                             (0.40, 0.45, "p"), (0.45, 0.60, "a˥˩")])
    assert [s.tone for s in syls] == [1, 4]
    assert syls[0].t_end == pytest.approx(0.20)


def test_beam_goes_into_a_config_file_not_the_command_line(tmp_path, monkeypatch):
    """MFA 3.x has no --beam flag; passing one shifts the positional arguments.

    The symptom is an unhelpful "Corpus directory does not exist", so this guards the
    construction of the command rather than waiting to be surprised by it again.
    """
    import subprocess

    from asdpipe.align import mfa as mfa_mod

    fake = tmp_path / "mfa.exe"
    fake.write_text("", encoding="utf-8")
    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = list(cmd)
        return subprocess.CompletedProcess(cmd, 1, "", "stopped on purpose")

    monkeypatch.setattr(mfa_mod.subprocess, "run", fake_run)
    aligner = mfa_mod.MFAAligner(acoustic_model="m", dictionary="d",
                                 binary=str(fake), beam=100, fine_tune=True)
    turn = Turn(0, Speaker.CHILD, 0.0, 1.0, "你好")
    with pytest.raises(AlignmentUnavailable):
        aligner.align(np.zeros(16000, dtype=np.float32), 16000, [turn], {0: ["你好"]})

    cmd = captured["cmd"]
    assert "--beam" not in cmd and "--retry_beam" not in cmd
    assert "--config_path" in cmd
    assert "--fine_tune" in cmd
    assert "--no_tokenization" in cmd, "MFA's own Chinese tokenizer is pkuseg (R4)"
    # the last four arguments must remain corpus / dictionary / acoustic model / output
    assert cmd[-3] == "d" and cmd[-2] == "m"


def test_subprocess_env_adds_the_library_bin_of_the_mfa_prefix(tmp_path):
    """Without <prefix>/Library/bin on PATH, MFA's soundfile cannot load libsndfile.dll."""
    from asdpipe.align.mfa import mfa_subprocess_env

    prefix = tmp_path / "envs" / "mfa"
    (prefix / "Scripts").mkdir(parents=True)
    (prefix / "Library" / "bin").mkdir(parents=True)
    exe = prefix / "Scripts" / "mfa.exe"
    exe.write_text("", encoding="utf-8")
    env = mfa_subprocess_env(str(exe))
    assert str(prefix / "Library" / "bin") in env["PATH"]
    assert str(prefix / "Scripts") in env["PATH"]


# ---------------------------------------------------------------- caching
class _CountingAligner:
    """Stand-in backend that records how many times it was actually asked to work."""

    name = "counting"

    def __init__(self, spans):
        self.spans = spans
        self.calls = 0

    def available(self):
        return True

    def align(self, audio, sample_rate, turns, words=None):
        self.calls += 1
        return dict(self.spans)


def _one_turn():
    from asdpipe.contracts import SyllableSpan

    turn = Turn(0, Speaker.CHILD, 0.0, 1.0, "你好")
    spans = {0: (SyllableSpan(0.0, 0.4, 3), SyllableSpan(0.4, 0.9, 3))}
    return turn, spans


def test_cached_aligner_only_calls_the_backend_once(tmp_path):
    """MFA costs about a minute per session; a re-run must not pay for it again."""
    from asdpipe.align.cache import CachedAligner

    turn, spans = _one_turn()
    inner = _CountingAligner(spans)
    cached = CachedAligner(inner, tmp_path)
    audio = np.zeros(16000, dtype=np.float32)
    a = cached.align(audio, 16000, [turn], {0: ["你好"]})
    b = cached.align(audio, 16000, [turn], {0: ["你好"]})
    assert a == b == spans
    assert inner.calls == 1
    assert (cached.hits, cached.misses) == (1, 1)


@pytest.mark.parametrize("mutate", ["audio", "words", "boundary", "identity"])
def test_cached_aligner_key_covers_everything_that_changes_the_answer(tmp_path, mutate):
    from asdpipe.align.cache import CachedAligner

    turn, spans = _one_turn()
    inner = _CountingAligner(spans)
    audio = np.zeros(16000, dtype=np.float32)
    cached = CachedAligner(inner, tmp_path, {"beam": 100})
    cached.align(audio, 16000, [turn], {0: ["你好"]})

    if mutate == "audio":
        cached.align(audio + 0.1, 16000, [turn], {0: ["你好"]})
    elif mutate == "words":
        cached.align(audio, 16000, [turn], {0: ["你", "好"]})
    elif mutate == "boundary":
        moved = Turn(0, Speaker.CHILD, 0.0, 1.5, "你好")
        cached.align(audio, 16000, [moved], {0: ["你好"]})
    else:
        CachedAligner(inner, tmp_path, {"beam": 10}).align(
            audio, 16000, [turn], {0: ["你好"]})
    assert inner.calls == 2, f"changing {mutate} must invalidate the cache"


def test_aligner_from_config_builds_a_cached_mfa(cfg, tmp_path):
    from asdpipe.align import CachedAligner, MFAAligner, aligner_from_config

    a = aligner_from_config(cfg, tmp_path)
    assert isinstance(a, CachedAligner)
    assert isinstance(a.inner, MFAAligner)
    assert a.inner.beam == cfg["models"]["mfa"]["beam"]
    assert a.inner.dictionary == cfg["models"]["mfa"]["dictionary"]
    plain = aligner_from_config(cfg, None, cached=False)
    assert isinstance(plain, MFAAligner)


def test_config_pins_the_q14_decision(cfg):
    """QUESTIONS Q14 was resolved with measurements; the config must still say so."""
    mfa = cfg["models"]["mfa"]
    assert mfa["acoustic_model"] == "mandarin_mfa"
    assert mfa["dictionary"] == "mandarin_taiwan_mfa"
    assert mfa["beam"] == 100, "the default beam lost 21% of turns, all of them long ones"
    assert mfa["g2p_model"], "OOV words need the G2P model; both dictionaries miss 裡/牠/蝙蝠車"


def test_speechbrain_placeholders_survive_a_hasattr_sweep():
    """Regression guard for a failure that surfaces far from its cause.

    SpeechBrain 1.1 leaves placeholder modules in sys.modules whose __getattr__ raises
    ImportError for a missing optional dependency. torchvision and librosa both walk
    sys.modules calling hasattr(mod, "__file__"), so the ImportError surfaces from an
    unrelated import and names k2_fsa. Reordering imports only moves it around.
    """
    import sys

    from asdpipe.features.backends import defuse_speechbrain_lazy_modules

    class Landmine:
        __name__ = "speechbrain.fake_placeholder"

        def __getattr__(self, item):
            raise ImportError("lazy import failed")

    sys.modules["speechbrain.fake_placeholder"] = Landmine()
    try:
        with pytest.raises(ImportError):
            hasattr(sys.modules["speechbrain.fake_placeholder"], "__file__")
        assert defuse_speechbrain_lazy_modules() >= 1
        assert hasattr(sys.modules["speechbrain.fake_placeholder"], "__file__") is True
    finally:
        sys.modules.pop("speechbrain.fake_placeholder", None)


@pytest.mark.integration
def test_mfa_end_to_end():
    if mfa_binary() is None:
        pytest.skip("the `mfa` executable is not on PATH -- integration test skipped by design")
    corpus = generate_corpus(SynthConfig(n_subjects=1, turns_per_subject=4))
    s = corpus.subjects[0]
    aligner = MFAAligner(acoustic_model="mandarin_mfa", dictionary="mandarin_china_mfa")
    out = aligner.align(s.audio, s.sample_rate, s.session.turns)
    assert out, "MFA produced no alignments"
    for spans in out.values():
        for a, b in zip(spans[:-1], spans[1:]):
            assert a.t_end <= b.t_start


@pytest.mark.integration
def test_d_arm_is_disabled_in_the_smoke_profile(cfg):
    """M8: the benchmark arm exists as an interface but must not download in the smoke run."""
    from asdpipe.features.d_ssl import DArmUnavailable, extract_d_embeddings

    assert cfg["models"]["d_arm"]["enabled"] is False
    with pytest.raises(DArmUnavailable):
        extract_d_embeddings([], cfg)


def test_d_arm_protocol_constants(cfg):
    d = cfg["models"]["d_arm"]
    assert d["backbone"] == "facebook/wav2vec2-xls-r-300m"
    assert d["pca_dim"] == 20
    assert isinstance(d["layer"], int)


def test_syllabify_marks_the_rime_and_excludes_onset_and_coda():
    """Mandarin tone is carried by the rime. Measuring F0 across the onset as well produced
    a T4 template that rose through its first half, which a high-falling tone cannot do."""
    # 公園  kʷ u˥ ŋ  ɥ e˧˥ n
    syls = syllabify_phones([
        (0.00, 0.05, "kʷ"), (0.05, 0.18, "u˥"), (0.18, 0.24, "ŋ"),
        (0.24, 0.29, "ɥ"), (0.29, 0.44, "e˧˥"), (0.44, 0.50, "n"),
    ])
    assert [(s.nucleus_start, s.nucleus_end) for s in syls] == [(0.05, 0.18), (0.29, 0.44)]
    for s in syls:
        assert s.t_start < s.nucleus_start and s.nucleus_end < s.t_end


def test_syllabify_treats_a_diphthong_as_one_nucleus():
    """mandarin_taiwan_mfa encodes diphthongs as single symbols (aj/aw/ej/ow), so a
    diphthong is one nuclear segment, not two. Written after asserting the opposite and
    watching it fail: the dictionary has no bare glide nuclei to string together."""
    syls = syllabify_phones([(0.0, 0.04, "x"), (0.04, 0.20, "aj˥˩")])
    assert len(syls) == 1 and syls[0].tone == 4
    assert (syls[0].nucleus_start, syls[0].nucleus_end) == (0.04, 0.20)


def test_offsetting_a_syllable_keeps_every_optional_field():
    """Guards the rebuild in MFAAligner.align, which has silently corrupted spans twice:
    once dropping the phone string, once shifting `underlying_tone` into `nucleus_start`
    when a field was inserted ahead of it. Both left the tones looking correct."""
    from dataclasses import replace

    s = SyllableSpan(1.0, 1.2, tone=4, text="ma", vocalic_s=0.15,
                     nucleus_start=1.05, nucleus_end=1.18, underlying_tone=3)
    moved = replace(s, t_start=s.t_start + 10, t_end=s.t_end + 10,
                    nucleus_start=s.nucleus_start + 10, nucleus_end=s.nucleus_end + 10)
    assert (moved.text, moved.vocalic_s, moved.underlying_tone) == ("ma", 0.15, 3)
    assert (moved.nucleus_start, moved.nucleus_end) == (11.05, 11.18)
    assert moved.t_start < moved.nucleus_start < moved.nucleus_end < moved.t_end
