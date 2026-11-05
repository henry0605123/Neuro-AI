"""Synthetic ADOS-like corpus generator (milestone M1).

Contract that matters most: **at signal_strength = 0 the label has literally no effect on
the generated data.** Latents are drawn first, from a subject-indexed RNG stream, and the
label enters only through the additive term ``s * delta * label``. So at s = 0 you can
permute every label and the corpus is byte-identical -- which is exactly what
``tests/test_synth.py`` asserts.

Injected directions follow the task spec: rhythm variability up, pause anomaly up,
response latency up, echolalia similarity up, topic relevance down, residual F0 with no
preset direction. Three further traits (cartoon-voice embedding variance up, volume
following down, multi-turn adaptation down) follow the plan doc and are listed separately
in `configs/pipeline.yaml`. Bidirectional asymmetry (F8) is deliberately *not* injected --
it emerges because the examiner reacts to the child (see QUESTIONS Q11).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ..contracts import Corpus, Label, Session, Speaker, SubjectRecord, SyllableSpan, Turn
from . import lexicon as lex
from .audio import VoiceProfile, render_session, render_turn

__all__ = ["SynthConfig", "TRAIT_ORDER", "generate_corpus", "generate_subject"]

# Fixed order -- the RNG draws in exactly this sequence, so adding a trait at the end
# never changes previously generated corpora.
TRAIT_ORDER: tuple[str, ...] = (
    "rhythm_variability",
    "pause_anomaly",
    "response_latency",
    "echolalia_similarity",
    "topic_relevance",
    "residual_f0_sd",
    "stereotyped_rate",
    "embedding_variance",
    "volume_following",
    "multiturn_adaptation",
)

DEFAULT_INJECTIONS: dict[str, float] = {
    "rhythm_variability": 0.9,
    "pause_anomaly": 0.9,
    "response_latency": 1.0,
    "echolalia_similarity": 1.2,
    "topic_relevance": -1.0,
    "residual_f0_sd": 0.8,
    "stereotyped_rate": 1.1,
    "embedding_variance": 1.0,
    "volume_following": -1.0,
    "multiturn_adaptation": -0.9,
}


@dataclass(frozen=True)
class SynthConfig:
    n_subjects: int = 12
    asd_ratio: float = 0.5
    turns_per_subject: int = 60
    seed: int = 20260824
    signal_strength: float = 1.0
    sample_rate: int = 16000
    f0_child_range: tuple[float, float] = (180.0, 420.0)
    f0_examiner_range: tuple[float, float] = (100.0, 160.0)
    n_harmonics: int = 16
    noise_level: float = 0.02
    syllable_duration_mean: float = 0.22
    render_examiner_audio: bool = True
    injections: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_INJECTIONS))
    examiner_leak_strength: float = 0.0
    ados_items: Sequence[str] = ("A2", "A3", "A4", "A8", "OQR")
    ados_target_spearman: float = 0.7

    @classmethod
    def from_dict(cls, d: Mapping) -> "SynthConfig":
        a = d.get("audio", {})
        inj = dict(DEFAULT_INJECTIONS)
        inj.update(d.get("injections", {}))
        ados = d.get("ados", {})
        return cls(
            n_subjects=int(d.get("n_subjects", 12)),
            asd_ratio=float(d.get("asd_ratio", 0.5)),
            turns_per_subject=int(d.get("turns_per_subject", 60)),
            seed=int(d.get("seed", 20260824)),
            signal_strength=float(d.get("signal_strength", 1.0)),
            sample_rate=int(d.get("sample_rate", 16000)),
            f0_child_range=tuple(a.get("f0_child_range", (180.0, 420.0))),
            f0_examiner_range=tuple(a.get("f0_examiner_range", (100.0, 160.0))),
            n_harmonics=int(a.get("n_harmonics", 16)),
            noise_level=float(a.get("noise_level", 0.02)),
            syllable_duration_mean=float(a.get("syllable_duration_mean", 0.22)),
            render_examiner_audio=bool(a.get("render_examiner_audio", True)),
            injections=inj,
            examiner_leak_strength=float(d.get("examiner_leak_strength", 0.0)),
            ados_items=tuple(ados.get("items", ("A2", "A3", "A4", "A8", "OQR"))),
            ados_target_spearman=float(ados.get("target_spearman", 0.7)),
        )

    def replace(self, **kw) -> "SynthConfig":
        from dataclasses import replace as _r

        return _r(self, **kw)


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _child_voice(base_f0: float, mode: int, cfg: SynthConfig) -> VoiceProfile:
    """Three cartoon-voice modes. Different formants -> different ECAPA embeddings."""
    presets = (
        ((820.0, 1900.0, 3100.0), (140.0, 260.0, 380.0), (1.0, 0.62, 0.34), 1.35, 0.0),
        ((520.0, 2450.0, 3400.0), (110.0, 300.0, 420.0), (1.0, 0.85, 0.55), 1.05, 3.5),
        ((980.0, 1450.0, 2700.0), (180.0, 220.0, 340.0), (1.0, 0.48, 0.22), 1.65, -2.5),
    )
    formants, bws, gains, rolloff, shift = presets[mode % 3]
    return VoiceProfile(
        base_f0=base_f0 * 2.0 ** (shift / 12.0),
        formants=formants,
        bandwidths=bws,
        gains=gains,
        rolloff=rolloff,
        n_harmonics=cfg.n_harmonics,
        noise_level=cfg.noise_level,
    )


def _examiner_voice(base_f0: float, cfg: SynthConfig) -> VoiceProfile:
    return VoiceProfile(
        base_f0=base_f0,
        formants=(620.0, 1350.0, 2500.0),
        bandwidths=(130.0, 210.0, 330.0),
        gains=(1.0, 0.55, 0.30),
        rolloff=1.45,
        n_harmonics=cfg.n_harmonics,
        noise_level=cfg.noise_level,
    )


def _lay_out_syllables(
    text: str,
    t_start: float,
    mean_dur: float,
    dur_cv: float,
    pause_prob: float,
    pause_mean: float,
    rng: np.random.Generator,
) -> tuple[list[SyllableSpan], float]:
    """Turn a sentence into timed syllables, inserting within-utterance pauses."""
    chars = lex.strip_to_syllables(text)
    tones = [lex.tone_of(c) for c in chars]
    # Tone-3 sandhi: T3 before T3 is realised as T2. Constant across groups on purpose --
    # the feature exists, but carries no injected group signal (see QUESTIONS Q10 note).
    realised = list(tones)
    for i in range(len(tones) - 1):
        if tones[i] == 3 and tones[i + 1] == 3 and rng.random() < 0.85:
            realised[i] = 2

    spans: list[SyllableSpan] = []
    t = t_start
    sigma = math.sqrt(math.log1p(dur_cv ** 2))
    for i, (ch, tone) in enumerate(zip(chars, realised)):
        base = mean_dur * (0.55 if tone == 0 else 1.0)
        dur = float(base * math.exp(rng.normal(0.0, sigma) - 0.5 * sigma ** 2))
        dur = min(max(dur, 0.05), 1.2)
        spans.append(SyllableSpan(t_start=t, t_end=t + dur, tone=tone, text=ch,
                                  underlying_tone=tones[i]))
        t += dur
        if i < len(chars) - 1 and rng.random() < pause_prob:
            t += float(pause_mean * math.exp(rng.normal(0.0, 0.4)))
    return spans, t


def generate_subject(index: int, label: Label, cfg: SynthConfig) -> SubjectRecord:
    rng = np.random.default_rng(cfg.seed * 1_000_003 + index)

    # ---- latents: drawn BEFORE the label is used, so s=0 makes the label irrelevant ----
    raw = {name: float(rng.standard_normal()) for name in TRAIT_ORDER}
    s = cfg.signal_strength
    lab = float(int(label))
    latents = {
        name: raw[name] + s * float(cfg.injections.get(name, 0.0)) * lab for name in TRAIT_ORDER
    }
    resid_sign = 1.0 if rng.random() < 0.5 else -1.0     # no preset direction (QUESTIONS Q10)

    z = latents
    p_echo = _sig(-2.20 + 0.80 * z["echolalia_similarity"])
    p_stereo = _sig(-2.10 + 0.80 * z["stereotyped_rate"])
    p_ontopic0 = _sig(1.30 + 0.85 * z["topic_relevance"])
    adapt = 0.55 * z["multiturn_adaptation"]
    dur_cv = 0.13 * math.exp(0.34 * z["rhythm_variability"])
    pause_prob = _sig(-2.05 + 0.62 * z["pause_anomaly"])
    pause_mean = 0.22 * math.exp(0.30 * z["pause_anomaly"])
    latency_mean = 0.42 * math.exp(0.36 * z["response_latency"])
    resid_sd = 0.080 * math.exp(0.45 * z["residual_f0_sd"])
    resid_mu = resid_sign * 0.045 * z["residual_f0_sd"]
    vol_slope = 0.50 + 0.34 * z["volume_following"]
    p_switch = _sig(-2.15 + 0.85 * z["embedding_variance"])

    child_base_f0 = float(rng.uniform(*cfg.f0_child_range))
    exam_base_f0 = float(rng.uniform(*cfg.f0_examiner_range))
    exam_voice = _examiner_voice(exam_base_f0, cfg)
    child_base_db, exam_base_db = -22.0, -20.0

    own_phrases = [lex.STEREOTYPED_PHRASES[i] for i in rng.choice(
        len(lex.STEREOTYPED_PHRASES), size=2, replace=False)]

    # Examiner-side leakage is OFF by default: the examiner only reacts to the child.
    exam_leak = cfg.examiner_leak_strength * lab
    p_examiner_follow = 0.60
    exam_extra_words = int(round(2.0 * exam_leak))

    n_pairs = max(2, cfg.turns_per_subject // 2)
    topic = lex.TOPICS[int(rng.integers(len(lex.TOPICS)))]
    turns: list[Turn] = []
    rendered: list[tuple[np.ndarray, float]] = []
    t = 0.5
    voice_mode = 0
    turn_id = 0
    prev_child_topic = topic

    for k in range(n_pairs):
        # ---------------- examiner turn ----------------
        yes_no = rng.random() < 0.18
        if yes_no:
            q_text = lex.YES_NO_QUESTIONS[int(rng.integers(len(lex.YES_NO_QUESTIONS)))]
        else:
            pool = lex.QUESTIONS[topic]
            q_text = pool[int(rng.integers(len(pool)))]
        if exam_extra_words > 0:                      # positive-control leakage only
            q_text = q_text + "好不好" * exam_extra_words
        ex_db = exam_base_db + float(rng.normal(0.0, 4.0)) + 1.5 * exam_leak
        spans, t_end = _lay_out_syllables(
            q_text, t, cfg.syllable_duration_mean * 0.95, 0.13, 0.04, 0.18, rng
        )
        ex_turn = Turn(
            turn_id=turn_id, speaker=Speaker.EXAMINER, t_start=t, t_end=t_end, text=q_text,
            syllables=tuple(spans), intensity_db=ex_db, is_question=True,
            is_yes_no=yes_no, topic=topic,
        )
        turns.append(ex_turn)
        turn_id += 1
        if cfg.render_examiner_audio:
            rendered.append(render_turn(spans, exam_voice, cfg.sample_rate, ex_db,
                                        rng.normal(0.0, 0.010, len(spans)), rng))
        t = t_end

        # ---------------- child turn ----------------
        t += float(latency_mean * math.exp(rng.normal(0.0, 0.30)))
        p_ontopic = min(0.98, max(0.02, p_ontopic0 + adapt * (k / max(1, n_pairs - 1))))
        u = rng.random()
        if u < p_echo:
            a_text = q_text.rstrip("嗎呢吧")
            child_topic, kind = topic, "echo"
        elif u < p_echo + p_stereo:
            a_text = own_phrases[int(rng.integers(len(own_phrases)))]
            child_topic, kind = topic, "stereotyped"
        else:
            if rng.random() < p_ontopic:
                child_topic = topic
            else:
                others = [x for x in lex.TOPICS if x != topic]
                child_topic = others[int(rng.integers(len(others)))]
            pool = lex.ANSWERS[child_topic]
            a_text = pool[int(rng.integers(len(pool)))]
            kind = "answer"

        if rng.random() < p_switch:
            voice_mode = int(rng.integers(3))
        ch_voice = _child_voice(child_base_f0, voice_mode, cfg)
        ch_db = child_base_db + vol_slope * (ex_db - exam_base_db) + float(rng.normal(0.0, 1.5))

        spans, t_end = _lay_out_syllables(
            a_text, t, cfg.syllable_duration_mean, dur_cv, pause_prob, pause_mean, rng
        )
        resid = resid_mu + resid_sd * rng.standard_normal(len(spans))
        ch_turn = Turn(
            turn_id=turn_id, speaker=Speaker.CHILD, t_start=t, t_end=t_end, text=a_text,
            syllables=tuple(spans), intensity_db=ch_db, is_question=False,
            is_yes_no=False, topic=f"{child_topic}:{kind}",
        )
        turns.append(ch_turn)
        turn_id += 1
        rendered.append(render_turn(spans, ch_voice, cfg.sample_rate, ch_db, resid, rng))
        t = t_end + 0.35 + float(abs(rng.normal(0.0, 0.12)))

        # examiner reacts to the child (no label involved) -> F8 asymmetry emerges
        prev_child_topic = child_topic
        if rng.random() < p_examiner_follow:
            topic = prev_child_topic

    duration = max(x.t_end for x in turns)
    audio = render_session(rendered, cfg.sample_rate, duration)
    session = Session(session_id=f"S{index:03d}", subject_id=f"SUBJ{index:03d}",
                      turns=tuple(turns), sample_rate=cfg.sample_rate)
    ados = _synth_ados(latents, cfg, rng)
    return SubjectRecord(
        subject_id=session.subject_id, label=label, session=session, audio=audio,
        sample_rate=cfg.sample_rate, ados=ados, latents=latents,
    )


def _synth_ados(latents: Mapping[str, float], cfg: SynthConfig,
                rng: np.random.Generator) -> dict[str, int]:
    """Ordinal 0-3 ADOS codes correlated with the injected latents (QUESTIONS Q21).

    Note these are built from the *latents*, not from the label, so the criterion-validity
    analysis stays label-free exactly as section 4b.2 requires.
    """
    z = latents
    # (coefficients over the latents, which are unit-variance) -> value, normaliser
    weights: dict[str, dict[str, float]] = {
        "A2": {"rhythm_variability": 0.45, "volume_following": -0.35,
               "embedding_variance": 0.35, "residual_f0_sd": 0.20},
        "A3": {"echolalia_similarity": 1.0},
        "A4": {"stereotyped_rate": 1.0},
        "A8": {"topic_relevance": -0.65, "multiturn_adaptation": -0.40},
        "OQR": {"topic_relevance": -0.55, "multiturn_adaptation": -0.30},
    }
    rho = cfg.ados_target_spearman
    cuts = (-0.55, 0.25, 1.00)
    out: dict[str, int] = {}
    for item in cfg.ados_items:
        w = weights.get(item, {})
        norm = math.sqrt(sum(v * v for v in w.values())) or 1.0
        d = sum(v * z[k] for k, v in w.items()) / norm
        score = rho * d + math.sqrt(max(0.0, 1.0 - rho ** 2)) * float(rng.standard_normal())
        out[item] = int(sum(score > c for c in cuts))
    return out


def generate_corpus(cfg: SynthConfig) -> Corpus:
    n_asd = int(round(cfg.n_subjects * cfg.asd_ratio))
    labels = [Label.ASD] * n_asd + [Label.CONTROL] * (cfg.n_subjects - n_asd)
    subjects = tuple(generate_subject(i, labels[i], cfg) for i in range(cfg.n_subjects))
    tag = f"n{cfg.n_subjects}_t{cfg.turns_per_subject}_s{cfg.signal_strength}_seed{cfg.seed}"
    return Corpus(subjects=subjects, config_hash=tag)
