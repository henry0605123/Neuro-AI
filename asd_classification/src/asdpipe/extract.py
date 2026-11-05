"""Corpus-level feature extraction (milestones M2 + M3), tuned for this machine.

Where the work goes, and why it is split this way:

* **CPU-bound** -- openSMILE LLDs and Praat CPP. Praat dominates (~0.13 s per turn), so
  subjects are farmed out to a process pool across the 8-core / 16-thread Ryzen. Nothing
  in this path touches CUDA, so the workers stay cheap to spawn.
* **GPU-bound** -- SBERT, CKIP, ECAPA and torchcrepe. These stay in the parent process and
  are called **once per corpus** with everything batched, instead of once per subject.
  Loading four models per worker would cost far more than it saves.
* **RAM** -- with 63 GB available the whole corpus, all embeddings and all caches stay
  resident; nothing is recomputed between folds or between seeds.

Extraction results are cached under `cache/` keyed by the generator config plus the frozen
feature config, so re-running `make smoke` or sweeping seeds does not redo finished work.
"""

from __future__ import annotations

import hashlib
import json
import re
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from typing import Mapping, Sequence

import numpy as np

from .align.base import Aligner
from .align.gt import GroundTruthAligner
from .config import CACHE_DIR
from .contracts import Corpus, Speaker, SubjectRecord, SubjectRow, Turn
from .features import b_voice, c_prosody, e_clinical, e_text, intonation
from .features.backends import Backends
from .features.common import child_turns, slice_turn

_HAN = re.compile(r"[一-鿿]")

__all__ = ["extract_corpus", "cpu_stage", "child_alignment_inputs"]


def corpus_digest(corpus: Corpus) -> str:
    """Content hash of the corpus.

    Keying the cache on the generator *config* is not enough: editing the lexicon or a
    generator constant leaves the config string identical and silently serves stale
    features. Hashing the actual waveforms and transcripts costs ~0.15 s and cannot go
    stale.
    """
    h = hashlib.blake2s(digest_size=12)
    for s in corpus.subjects:
        h.update(s.subject_id.encode("utf-8"))
        h.update(bytes([int(s.label)]))
        h.update(np.ascontiguousarray(s.audio, dtype=np.float32).tobytes())
        for t in s.session.turns:
            # The question flags are derived from the text, but by a rule that has changed:
            # they were once hard-coded to False for every non-examiner turn, which zeroed
            # assessment item B3 without changing a single byte of the transcript. Hash the
            # derived flags, not just the source text, or that fix serves from stale cache.
            h.update(f"{t.turn_id}|{t.speaker.value}|{t.t_start:.6f}|{t.t_end:.6f}|{t.text}"
                     f"|{int(t.is_question)}{int(t.is_yes_no)}".encode("utf-8"))
        h.update(json.dumps(dict(s.ados), sort_keys=True).encode("utf-8"))
    return h.hexdigest()


# Bump whenever extraction *semantics* change without the config changing. The cache key
# hashes config and corpus content, neither of which notices an edit to extractor code --
# and a silently stale feature table is the worst possible failure here, because it looks
# like a successful run. History: 1 = original, 2 = CPP measured on voiced segments only,
# seeded F0 dither, utterance-level intonation features; 3 = B1/B3 intonation features
# dropped after failing split-half reliability; 4 = CPP, perturbation LLD, tone-contour and
# multi-turn-adaptation features removed, assessment-sheet E5/F2/F4/F7 substitutes added;
# 5-6 = D9/D11 measure changes; 7 = health check step 4: NaN instead of 0 for anything
# unmeasurable, one response per examiner turn, G1 on open questions only, duplicate and
# sensitivity-run columns out of the feature set.
EXTRACT_VERSION = 7


def _cfg_hash(cfg: Mapping, corpus: Corpus, aligner_name: str = "") -> str:
    payload = json.dumps(
        {"features": cfg["features"], "models": cfg["models"], "aligner": aligner_name,
         "extract_version": EXTRACT_VERSION,
         "corpus": corpus_digest(corpus)},
        sort_keys=True, default=str,
    )
    return hashlib.blake2s(payload.encode("utf-8"), digest_size=10).hexdigest()


def cpu_stage(args) -> tuple[b_voice.BStreamFeatures, np.ndarray]:
    """Runs in a worker process: openSMILE LLD + Praat CPP + eGeMAPS functionals (B2)."""
    audio, sr, turns, cfg = args
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    bf = b_voice.extract_b(audio, sr, turns, _worker_lld(cfg), cfg)
    kids = child_turns(turns)
    if kids:
        child_audio = np.concatenate([slice_turn(audio, sr, t) for t in kids])
    else:
        child_audio = np.zeros(sr, dtype=np.float32)
    smile = _worker_functionals(cfg)
    df = smile.process_signal(np.ascontiguousarray(child_audio, dtype=np.float64), sr)
    return bf, df.to_numpy(dtype=np.float64)[0]


_WORKER_SMILE = {}


def _worker_smile(level: str):
    """One openSMILE instance per worker process per level, created lazily."""
    if level not in _WORKER_SMILE:
        import opensmile

        _WORKER_SMILE[level] = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=getattr(opensmile.FeatureLevel, level),
            num_workers=1,
        )
    return _WORKER_SMILE[level]


def _worker_lld(cfg: Mapping):
    return _worker_smile("LowLevelDescriptors")


def _worker_functionals(cfg: Mapping):
    return _worker_smile("Functionals")


def child_alignment_inputs(turns: Sequence[Turn], words: Mapping[int, Sequence[str]]
                           ) -> tuple[list[Turn], dict[int, list[str]]]:
    """The child turns MFA aligns, and the Han-character words it looks up for each.

    Only the child's turns are aligned: the C stream never looks at examiner speech, and
    aligning it would double the MFA cost for nothing. The word segmentation goes with
    them -- MFA looks Mandarin up word by word, so without it almost every token is out
    of vocabulary. Shared with the review package (review/prepare.py), so the boundaries
    a person checks are the ones the features were computed from.
    """
    kid_turns = [t for t in turns if t.speaker is Speaker.CHILD]
    align_words = {t.turn_id: [w for w in words.get(t.turn_id, ()) if _HAN.search(w)]
                   for t in kid_turns}
    return [t for t in kid_turns if align_words.get(t.turn_id)], align_words


def extract_corpus(
    corpus: Corpus,
    cfg: Mapping,
    backends: Backends,
    aligner: Aligner | None = None,
    n_workers: int | None = None,
    use_cache: bool = True,
) -> list[SubjectRow]:
    aligner = aligner or GroundTruthAligner()
    key = _cfg_hash(cfg, corpus, aligner.name)
    cache_path = CACHE_DIR / f"features_{key}.pkl"
    if use_cache and cache_path.exists():
        with cache_path.open("rb") as fh:
            return pickle.load(fh)

    sr = corpus.subjects[0].sample_rate
    c_cfg = cfg["features"]["c_stream"]
    functionals = list(cfg["features"]["functionals"]["full"])
    min_windows = int(cfg["features"]["b_stream"]["ecapa_min_windows"])

    # ---------------------------------------------------------------- CPU pool
    payloads = [(s.audio, sr, list(s.session.turns), cfg) for s in corpus.subjects]
    workers = n_workers if n_workers is not None else min(8, (os.cpu_count() or 4))
    if workers > 1 and len(payloads) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            cpu_results = list(pool.map(cpu_stage, payloads, chunksize=1))
    else:
        cpu_results = [cpu_stage(p) for p in payloads]

    # ---------------------------------------------------------------- GPU, batched once
    all_texts: list[str] = []
    text_index: list[tuple[int, int]] = []          # (subject index, turn id)
    for i, s in enumerate(corpus.subjects):
        for t in s.session.turns:
            if t.text:
                all_texts.append(t.text)
                text_index.append((i, t.turn_id))
    embeddings_flat = backends.encode_sentences(all_texts)
    words_flat = backends.segment(all_texts)

    embeddings: list[dict[int, np.ndarray]] = [{} for _ in corpus.subjects]
    words: list[dict[int, list[str]]] = [{} for _ in corpus.subjects]
    for (si, tid), emb, ws in zip(text_index, embeddings_flat, words_flat):
        embeddings[si][tid] = emb
        words[si][tid] = list(ws)

    # ECAPA: one batched call for the whole corpus
    win_counts, win_blocks = [], []
    for s in corpus.subjects:
        w = b_voice.ecapa_windows(s.audio, sr, s.session.turns, cfg)
        win_counts.append(len(w))
        if len(w):
            win_blocks.append(w)
    all_windows = np.concatenate(win_blocks) if win_blocks else np.zeros((0, 1), dtype=np.float32)
    all_emb = backends.embed_windows(all_windows) if len(all_windows) else np.zeros((0, 192), np.float32)

    rows: list[SubjectRow] = []
    cursor = 0
    for i, subject in enumerate(corpus.subjects):
        bf, egemaps = cpu_results[i]
        n = win_counts[i]
        bf.attach(all_emb[cursor : cursor + n], min_windows)
        cursor += n

        # ---- C stream: F0 on the GPU, then alignment-driven prosody on the CPU ----
        f0, periodicity = backends.track_f0(
            subject.audio, sr, float(c_cfg["f0_min_hz"]), float(c_cfg["f0_max_hz"]),
            float(c_cfg["f0_hop_s"]),
        )
        # see child_alignment_inputs
        kid_turns = [t for t in subject.session.turns if t.speaker is Speaker.CHILD]
        alignable, align_words = child_alignment_inputs(subject.session.turns, words[i])
        syls = aligner.align(subject.audio, sr, alignable, align_words) if alignable else {}

        # audit P1-3: cross-algorithm F0 check on a sample of the child's speech
        kid_audio = (np.concatenate([slice_turn(subject.audio, sr, t) for t in kid_turns])
                     if kid_turns else np.zeros(sr, dtype=np.float32))
        f0_check = backends.f0_cross_check(
            kid_audio, sr, float(c_cfg["f0_min_hz"]), float(c_cfg["f0_max_hz"]),
            float(c_cfg["f0_hop_s"]),
        )
        cf = c_prosody.extract_c(
            subject.session.turns, syls, f0, periodicity, float(c_cfg["f0_hop_s"]), cfg
        )
        # Utterance-level intonation (A2/A3/B1/B2/B3/B5). Kept separate from extract_c
        # because it deliberately does not use the syllable alignment -- that is the whole
        # point of it; see features/intonation.py.
        hop_s = float(c_cfg["f0_hop_s"])
        inton = intonation.extract_intonation(
            subject.session.turns, f0, periodicity,
            intonation.frame_db(subject.audio, sr, hop_s), hop_s, cfg,
        )
        cf.scalars.update(inton.scalars)
        cf.quality.update(inton.quality)

        # ---- E stream ----
        turn_db = _measured_turn_db(subject, sr)
        ef = e_text.extract_e(subject.session.turns, embeddings[i], words[i], turn_db, cfg)
        probe = e_text.examiner_probe_features(
            subject.session.turns, embeddings[i], words[i], turn_db
        )
        # Assessment-sheet E5/F2/F4/F7 (features/e_clinical.py). Transcript and turn
        # structure only -- they replace features the recording conditions made impossible,
        # so they must not depend on those same conditions.
        clin = e_clinical.extract_clinical(
            subject.session.turns, embeddings[i], words[i], cfg
        )
        ef.scalars.update(clin.scalars)
        ef.quality.update(clin.quality)

        quality = {**bf.quality, **cf.quality, **ef.quality,
                   "c_f0_agreement": f0_check.get("agreement", float("nan")),
                   "c_f0_octave_error": f0_check.get("octave_error", float("nan"))}
        rows.append(SubjectRow(
            subject_id=subject.subject_id,
            label=subject.label,
            b_pooled=bf.pooled(functionals),
            c_pooled=cf.pooled(functionals),
            e_pooled=ef.pooled(functionals),
            c_sequence=cf.sequence,
            egemaps_functionals=egemaps,
            examiner_probe=probe,
            ados=dict(subject.ados),
            latents=dict(subject.latents),
            quality=quality,
        ))

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as fh:
            pickle.dump(rows, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return rows


def _measured_turn_db(subject: SubjectRecord, sr: int) -> dict[int, float]:
    """Level per turn, measured from the waveform.

    Never read from ``Turn.intensity_db``: that is the generator's ground truth, and using
    it would make volume following (A12) a tautology rather than a measurement.
    """
    from .features.common import rms_db

    out: dict[int, float] = {}
    for t in subject.session.turns:
        if t.speaker is Speaker.OVERLAP:
            continue
        x = slice_turn(subject.audio, sr, t)
        if x.size:
            out[t.turn_id] = rms_db(x)
    return out
