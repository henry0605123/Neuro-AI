"""Montreal Forced Aligner adapter -- the real-data path (R6).

Audit P1-1 requires utterance-by-utterance alignment driven by the *human* transcript:
MFA is given ``audio + verbatim text + utterance-level timestamps`` and returns
phone/syllable boundaries. The human transcript is MFA's input, never its replacement.

This adapter is deliberately not exercised by ``make smoke``. ``tests/test_mfa_adapter.py``
skips when the binary or the models are missing.
"""

from __future__ import annotations

from dataclasses import replace

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..contracts import SyllableSpan, Turn
from .base import Aligner, AlignmentUnavailable

__all__ = ["MFAAligner", "mfa_binary", "mfa_subprocess_env",
           "parse_textgrid_intervals", "syllabify_phones"]

# ---------------------------------------------------------------------------------------
# Phone inventory of the MFA 3.x Mandarin dictionaries.
#
# These are not guesses: the sets below were read out of
# `mandarin_taiwan_mfa.dict` and `mandarin_china_mfa.dict` (v3.0.0), and the tone mapping
# was checked against words whose tone is not in doubt --
#     媽 m a˥ (T1)   麻 m a˧˥ (T2)   馬 m a˨˩˦ (T3)   罵 m a˥˩ (T4)
#     八 p a˥        拔 p a˧˥        把 p a˨˩˦        爸 p a˥˩
#
# MFA 3.x writes IPA with Chao tone letters, **not** pinyin tone digits. The previous
# digit-suffix rule silently returned neutral tone for every syllable.
# ---------------------------------------------------------------------------------------
_TONE_LETTERS = "˥˦˧˨˩"

_TONE_FROM_CONTOUR: dict[str, int] = {
    "˥": 1,       # high level, 55
    "˧˥": 2,      # rising, 35
    "˨˩˦": 3,     # dipping, 214
    "˥˩": 4,      # falling, 51
    "˩": 0,       # reduced / neutralised in non-initial position (謝謝 -> ɕ e˥˩ ɕ e˩)
    "˧": 0,       # ditto
    "": 0,        # no tone letter at all -> neutral tone (的 -> t ə)
}

# Phones that ever carry a tone letter, i.e. syllable nuclei.
_NUCLEI = frozenset({"a", "aj", "aw", "e", "ej", "i", "m̩", "n̩", "o", "ow", "u", "y",
                     "z̩", "ŋ̍", "ə", "ʐ̩"})
# The only Mandarin codas. `n`/`ŋ` here are the non-syllabic ones; the syllabic variants
# carry the combining diacritic and live in _NUCLEI above.
_CODAS = frozenset({"n", "ŋ", "ɻ"})

_SILENCE = frozenset({"sil", "sp", "spn", "", "<eps>"})

# Bumped whenever the phone->syllable logic changes. It goes into the alignment cache key,
# so a code change cannot be masked by a stale cache (the same trap the feature cache hit).
# 4 added rime timings; 5 fixes the offset rebuild in MFAAligner.align, which was dropping
# them -- version 4 therefore cached spans whose nucleus fields were all None.
SYLLABIFY_VERSION = 5

# Legacy pinyin dictionaries (mandarin_pinyin v2.0.0) still use a trailing digit.
_TONE_FROM_DIGIT = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 0, "0": 0}


def _split_tone(label: str) -> tuple[str, int | None]:
    """Return (base phone, tone) for one MFA phone label."""
    if label and label[-1] in _TONE_FROM_DIGIT and not label[-1].isalpha():
        return label[:-1], _TONE_FROM_DIGIT[label[-1]]
    contour = "".join(c for c in label if c in _TONE_LETTERS)
    base = "".join(c for c in label if c not in _TONE_LETTERS)
    if not contour:
        return base, None
    return base, _TONE_FROM_CONTOUR.get(contour, 0)


def mfa_binary(explicit: str | None = None) -> str | None:
    """Locate the `mfa` executable.

    MFA lives in its own conda environment on purpose -- it pins kalpy, postgresql and a
    CUDA stack that would wreck the analysis environment -- so it is normally *not* on the
    PATH of the interpreter running this pipeline. The path therefore comes from, in order:
    an explicit argument, the MFA_BIN environment variable, then PATH.
    """
    for candidate in (explicit, os.environ.get("MFA_BIN")):
        if candidate and Path(candidate).exists():
            return str(candidate)
    return shutil.which("mfa")


def mfa_subprocess_env(binary: str | None) -> dict[str, str]:
    """Environment for the MFA subprocess.

    Invoking `<env>/Scripts/mfa.exe` without activating the environment leaves
    `<env>/Library/bin` off the DLL search path, and MFA's `soundfile` then dies with
    ``OSError: cannot load library 'libsndfile.dll': error 0x7e``. Rather than asking a
    caller to remember to activate first, the prefix is derived from the binary path and
    prepended here.
    """
    env = dict(os.environ)
    exe = mfa_binary(binary)
    if not exe:
        return env
    scripts = Path(exe).parent
    prefix = scripts.parent if scripts.name.lower() in {"scripts", "bin"} else scripts
    extra = [str(prefix / "Library" / "bin"), str(prefix / "Library" / "mingw-w64" / "bin"),
             str(prefix / "Library" / "usr" / "bin"), str(scripts), str(prefix)]
    env["PATH"] = os.pathsep.join([p for p in extra if Path(p).exists()]
                                  + [env.get("PATH", "")])
    return env


def parse_textgrid_intervals(text: str, tier_name: str = "phones") -> list[tuple[float, float, str]]:
    """Minimal TextGrid reader -- enough for MFA's long-form output, no praat dependency."""
    lines = [ln.strip() for ln in text.splitlines()]
    out: list[tuple[float, float, str]] = []
    in_tier = False
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("name ="):
            in_tier = tier_name in ln
        elif in_tier and ln.startswith("xmin =") and i + 2 < len(lines):
            try:
                xmin = float(ln.split("=")[1])
                xmax = float(lines[i + 1].split("=")[1])
            except (ValueError, IndexError):
                i += 1
                continue
            label_line = lines[i + 2]
            if label_line.startswith("text ="):
                label = label_line.split("=", 1)[1].strip().strip('"')
                if label:
                    out.append((xmin, xmax, label))
                i += 2
        i += 1
    return out


def syllabify_phones(intervals: Sequence[tuple[float, float, str]]) -> list[SyllableSpan]:
    """Group an MFA phone tier into Mandarin syllables.

    One syllable is [onset consonants] + nucleus + [coda]. A new syllable therefore starts
    whenever a nucleus or an onset appears after the current syllable already has its
    nucleus; a coda attaches to the nucleus that precedes it.

    Worked examples from the dictionary:
        忙  m a˧˥ ŋ          -> one syllable, T2
        東西 tʷ u˥ ŋ ɕ i˩    -> two syllables, T1 then neutral
        公園 kʷ u˥ ŋ ɥ e˧˥ n -> two syllables, T1 then T2
    """
    syllables: list[SyllableSpan] = []
    start: float | None = None
    end = 0.0
    tone = 0
    phones: list[str] = []
    has_nucleus = False
    vocalic = 0.0
    nuc_start: float | None = None
    nuc_end: float | None = None

    def flush() -> None:
        nonlocal start, has_nucleus, phones, tone, vocalic, nuc_start, nuc_end
        if start is not None and end > start:
            syllables.append(SyllableSpan(start, end, tone, text="".join(phones),
                                          vocalic_s=vocalic,
                                          nucleus_start=nuc_start, nucleus_end=nuc_end))
        start, has_nucleus, phones, tone, vocalic = None, False, [], 0, 0.0
        nuc_start = nuc_end = None

    for xmin, xmax, label in intervals:
        if label in _SILENCE:
            flush()
            continue
        base, phone_tone = _split_tone(label)
        # Anything carrying a tone is by definition the tone-bearing unit. This also covers
        # legacy pinyin dictionaries, whose labels ("hao3") are not in the IPA nucleus set.
        if base in _NUCLEI or phone_tone is not None:
            kind = "nucleus"
        elif base in _CODAS:
            kind = "coda"
        else:
            kind = "onset"          # unknown phones are treated as onsets, conservatively

        if kind == "coda" and not has_nucleus:
            kind = "onset"          # a stray nasal with nothing to attach to
        if kind in ("nucleus", "onset") and has_nucleus:
            flush()
        if start is None:
            start = xmin
        phones.append(base)
        if kind == "nucleus":
            has_nucleus = True
            tone = phone_tone if phone_tone is not None else 0
            vocalic += xmax - xmin
            # The rime runs from the first nuclear segment to the last. In practice
            # mandarin_taiwan_mfa gives exactly one per syllable -- it encodes diphthongs as
            # single symbols (aj/aw/ej/ow), verified against the inventory -- but the loop
            # does not assume that. The coda is deliberately excluded: a nasal coda carries
            # F0 without being where the tone target is realised.
            if nuc_start is None:
                nuc_start = xmin
            nuc_end = xmax
        end = xmax
    flush()
    return syllables


class MFAAligner(Aligner):
    name = "mfa"

    def __init__(self, acoustic_model: str, dictionary: str, num_jobs: int = 8,
                 timeout_s: int = 3600, binary: str | None = None,
                 g2p_model: str | None = None, beam: int | None = None,
                 fine_tune: bool = False):
        if not acoustic_model or not dictionary:
            raise AlignmentUnavailable(
                "configs/pipeline.yaml models.mfa.acoustic_model / .dictionary must be set "
                "explicitly before running on real data (QUESTIONS Q14)"
            )
        self.acoustic_model = acoustic_model
        self.dictionary = dictionary
        self.num_jobs = num_jobs
        self.timeout_s = timeout_s
        self.binary = binary
        self.g2p_model = g2p_model
        self.beam = beam
        self.fine_tune = fine_tune

    def available(self) -> bool:
        return mfa_binary(self.binary) is not None

    def align(
        self, audio: np.ndarray, sample_rate: int, turns: Sequence[Turn],
        words: Mapping[int, Sequence[str]] | None = None,
    ) -> dict[int, tuple[SyllableSpan, ...]]:
        """Align each turn separately (audit P1-1: utterance-by-utterance, human text).

        ``words`` supplies the word segmentation. MFA looks Mandarin up **word by word**
        in a pronunciation dictionary, and Chinese orthography has no word boundaries, so
        an unsegmented transcript is almost entirely out-of-vocabulary. Pass the CKIP
        segmentation here; omitting it is allowed only for languages that are already
        whitespace-delimited.
        """
        if not self.available():
            raise AlignmentUnavailable(
                "the `mfa` executable was not found; set MFA_BIN or pass binary=..."
            )
        import soundfile as sf

        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus"
            out_dir = Path(tmp) / "aligned"
            corpus.mkdir(parents=True)
            # audit P1-1: one MFA utterance per human-transcribed turn
            for t in turns:
                a = int(round(t.t_start * sample_rate))
                b = min(len(audio), int(round(t.t_end * sample_rate)))
                if b - a < sample_rate // 20:
                    continue
                sf.write(corpus / f"turn{t.turn_id:05d}.wav", audio[a:b], sample_rate)
                seg = words.get(t.turn_id) if words else None
                line = " ".join(seg) if seg else t.text
                (corpus / f"turn{t.turn_id:05d}.lab").write_text(line, encoding="utf-8")

            # --no_tokenization keeps MFA from re-segmenting the Chinese text. Two reasons:
            # (1) MFA's Mandarin tokenizer is spacy-pkuseg (Peking University), a
            #     China-origin model, which R4 forbids;
            # (2) the words are already segmented by CKIP upstream, and letting a second
            #     segmenter re-split them would decouple the E stream's word units from the
            #     C stream's alignment units.
            cmd = [mfa_binary(self.binary), "align", "--clean", "--quiet", "--use_mp",
                   "--no_tokenization",
                   "--num_jobs", str(self.num_jobs), "--single_speaker",
                   "--output_format", "long_textgrid"]
            # Search-beam settings are NOT command-line options in MFA 3.x; they live in a
            # config file. Passing them as flags silently shifts the positional arguments
            # and MFA then complains that the corpus directory does not exist.
            if self.beam:
                conf = Path(tmp) / "align.yaml"
                conf.write_text(
                    "beam: {0}{2}retry_beam: {1}{2}".format(
                        self.beam, self.beam * 4, chr(10)),
                    encoding="utf-8")
                cmd += ["--config_path", str(conf)]
            if self.fine_tune:
                cmd += ["--fine_tune"]
            if self.g2p_model:
                cmd += ["--g2p_model_path", self.g2p_model]
            cmd += [str(corpus), self.dictionary, self.acoustic_model, str(out_dir)]
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=self.timeout_s,
                                  env=mfa_subprocess_env(self.binary))
            if proc.returncode != 0:
                raise AlignmentUnavailable(f"mfa align failed: {proc.stderr[-2000:]}")

            result: dict[int, tuple[SyllableSpan, ...]] = {}
            for grid in sorted(out_dir.rglob("turn*.TextGrid")):
                turn_id = int(grid.stem.replace("turn", ""))
                intervals = parse_textgrid_intervals(grid.read_text(encoding="utf-8"))
                base = next((t.t_start for t in turns if t.turn_id == turn_id), 0.0)
                # `replace` rather than a positional rebuild. This line has now broken
                # twice: first it dropped the phone string, then adding nucleus timings in
                # front of `underlying_tone` silently shifted the tone into `nucleus_start`.
                # Both failures kept the tones looking right, so nothing raised. Copying the
                # dataclass and overriding only the two time fields cannot go wrong when a
                # field is added.
                syls = [
                    replace(s, t_start=s.t_start + base, t_end=s.t_end + base,
                            nucleus_start=None if s.nucleus_start is None else s.nucleus_start + base,
                            nucleus_end=None if s.nucleus_end is None else s.nucleus_end + base)
                    for s in syllabify_phones(intervals)
                ]
                if syls:
                    result[turn_id] = tuple(syls)
            return result
