"""Lazy, process-wide singletons for the frozen extractor toolchain.

Two jobs:

1. **R4 compliance.** Every model id is checked against the banned-substring list before it
   is loaded, and the resolved revision hash is recorded so the report can satisfy audit
   P0-0F (version freeze).
2. **Efficiency.** The heavy models are loaded once and kept resident (the box has 63 GB of
   RAM and a 16 GB RTX 5070 Ti), and inference is batched on the GPU in fp16 where the
   model supports it. Nothing here is re-created per subject or per fold.
"""

from __future__ import annotations

import functools
import os
import time
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

__all__ = ["Backends", "ComplianceError", "CREPE_BATCH_SIZE", "assert_compliant",
           "defuse_speechbrain_lazy_modules"]

# Frames per CREPE forward pass. A fixed setting, not an argument: the labelled run records it
# with the model, the device and torchcrepe's version (ruling R33), so the recorded number has to
# be the one that was used.
CREPE_BATCH_SIZE = 1024


def defuse_speechbrain_lazy_modules() -> int:
    """Make SpeechBrain's lazy modules survive a ``hasattr`` sweep over ``sys.modules``.

    SpeechBrain 1.1 registers placeholder modules (``speechbrain.integrations.k2_fsa`` and
    friends) whose ``__getattr__`` raises **ImportError** when the optional dependency is
    absent. Anything that walks ``sys.modules`` and calls ``hasattr(mod, "__file__")`` then
    blows up instead of getting False -- and two very common libraries do exactly that:
    ``torchvision._meta_registrations`` and ``lazy_loader`` (used by librosa) both call
    ``inspect.stack()``/``inspect.getmodule()``.

    The symptom is an ImportError naming k2_fsa, raised from an unrelated import several
    layers away, which is why ordering imports "correctly" only moves the failure around.
    Giving each placeholder a real ``__file__`` entry makes normal attribute lookup succeed,
    so ``__getattr__`` is never reached.

    Returns the number of modules defused. Safe to call repeatedly.
    """
    import sys

    fixed = 0
    for name, mod in list(sys.modules.items()):
        if not name.startswith("speechbrain") or mod is None:
            continue
        # Do not guess the placeholder class -- SpeechBrain 1.1 uses
        # `DeprecatedModuleRedirect`, not `LazyModule`. Just probe the attribute the
        # callers actually touch and patch whatever raises.
        try:
            hasattr(mod, "__file__")
            continue
        except Exception:
            pass
        try:
            mod.__dict__["__file__"] = None
            fixed += 1
        except Exception:
            pass
    return fixed


class ComplianceError(RuntimeError):
    """Raised when a model id trips the R4 banned-source list."""


def assert_compliant(model_id: str, banned: Iterable[str]) -> None:
    low = model_id.lower()
    for bad in banned:
        if bad and bad.lower() in low:
            raise ComplianceError(
                f"R4 violation: model id {model_id!r} matches banned source {bad!r}"
            )


@dataclass
class Backends:
    """Holds every frozen extractor. Construct once, pass around."""

    sbert_id: str
    ecapa_id: str
    ckip_ws_id: str
    crepe_model: str = "tiny"
    f0_seed: int = 20260902
    banned: Sequence[str] = ()
    device: str = "auto"
    cache_dir: str | None = None
    f0_cache_dir: str | None = None        # None: <cache_dir>/f0; see use_fresh_f0_cache
    _versions: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_config(cls, cfg: Mapping) -> "Backends":
        models = cfg["models"]
        prof = models.get("profile", "smoke")
        p = models[prof]
        return cls(
            sbert_id=p["sbert"],
            ecapa_id=p["ecapa"],
            ckip_ws_id=p["ckip_ws"],
            crepe_model=p.get("crepe_model", "tiny"),
            banned=tuple(models.get("banned_substrings", ())),
        )

    def __post_init__(self) -> None:
        for mid in (self.sbert_id, self.ecapa_id, self.ckip_ws_id):
            assert_compliant(mid, self.banned)
        if self.device == "auto":
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------ openSMILE
    @functools.cached_property
    def smile_lld(self):
        import opensmile

        self._versions["opensmile"] = opensmile.__version__
        return opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
            num_workers=1,
        )

    @functools.cached_property
    def smile_functionals(self):
        import opensmile

        return opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
            num_workers=1,
        )

    # ------------------------------------------------------------------ ECAPA (B7)
    @functools.cached_property
    def ecapa(self):
        # Import order matters and the failure is obscure, so it is pinned here.
        # torchvision's _meta_registrations calls inspect.getmodule(), which does
        # hasattr(mod, "__file__") over every entry in sys.modules. SpeechBrain 1.1
        # installs LazyModule objects whose __getattr__ raises ImportError (k2 is not
        # installed) instead of AttributeError, so hasattr() propagates and every later
        # `import transformers.modeling_utils` dies. Importing it first side-steps that.
        import transformers.modeling_utils  # noqa: F401  -- must precede speechbrain

        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        savedir = os.path.join(self.cache_dir or ".cache", "ecapa")
        # LocalStrategy.COPY: Windows refuses symlinks without Developer Mode / admin,
        # and SpeechBrain's default SYMLINK raises WinError 1314.
        model = EncoderClassifier.from_hparams(
            source=self.ecapa_id, savedir=savedir, run_opts={"device": self.device},
            local_strategy=LocalStrategy.COPY,
        )
        model.eval()
        defuse_speechbrain_lazy_modules()
        self._versions[self.ecapa_id] = _hf_revision(self.ecapa_id)
        return model

    def embed_windows(self, windows: np.ndarray) -> np.ndarray:
        """(n, samples) float32 -> (n, 192) ECAPA embeddings. Batched on the GPU."""
        import torch

        if len(windows) == 0:
            return np.zeros((0, 192), dtype=np.float32)
        out = []
        # 16 three-second windows peak well under 1 GB, so a run can share the card with
        # someone else's training under ASDPIPE_GPU_MEM_FRACTION (64 did not fit in 2.4 GB).
        bs = 16
        with torch.inference_mode():
            for i in range(0, len(windows), bs):
                chunk = torch.from_numpy(np.ascontiguousarray(windows[i : i + bs])).to(self.device)
                emb = self.ecapa.encode_batch(chunk).squeeze(1)
                out.append(emb.float().cpu().numpy())
        return np.concatenate(out, axis=0)

    def embed_segments(self, segments: Sequence[np.ndarray]) -> np.ndarray:
        """Variable-length clips -> (n, 192) ECAPA embeddings.

        Batches clips of similar length together, zero-pads each batch and passes the
        relative lengths, so the pooling layer ignores the padding. Untranscribed
        vocalisations are mostly well under the 3 s B7 window and cannot be forced into it.
        """
        import torch

        if len(segments) == 0:
            return np.zeros((0, 192), dtype=np.float32)
        # A voiceprint needs a few seconds, not a whole minute: long clips keep their middle
        # 8 s. Batches are capped by total audio, not count -- 64 one-minute turns overflowed
        # the convolution's 32-bit indexing and would take gigabytes of VRAM besides.
        sr = 16000
        cap = 8 * sr
        segments = [x if len(x) <= cap else x[(len(x) - cap) // 2:(len(x) - cap) // 2 + cap]
                    for x in segments]
        budget = 4 * cap                  # 32 s of audio per batch, well under 1 GB
        order = np.argsort([len(x) for x in segments])
        out = np.zeros((len(segments), 192), dtype=np.float32)
        batches, cur = [], []
        for j in order:
            if cur and (len(cur) + 1) * len(segments[j]) > budget:
                batches.append(cur)
                cur = []
            cur.append(j)
        if cur:
            batches.append(cur)
        with torch.inference_mode():
            for idx in batches:
                n = max(len(segments[j]) for j in idx)
                batch = np.zeros((len(idx), n), dtype=np.float32)
                for r, j in enumerate(idx):
                    batch[r, : len(segments[j])] = segments[j]
                lens = torch.tensor([len(segments[j]) / n for j in idx], dtype=torch.float32)
                emb = self.ecapa.encode_batch(torch.from_numpy(batch).to(self.device),
                                              lens.to(self.device)).squeeze(1)
                out[idx] = emb.float().cpu().numpy()
        return out

    # ------------------------------------------------------------------ SBERT (E stream)
    @functools.cached_property
    def sbert(self):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.sbert_id, device=self.device)
        model.eval()
        self._versions[self.sbert_id] = _hf_revision(self.sbert_id)
        return model

    def encode_sentences(self, sentences: Sequence[str]) -> np.ndarray:
        if not sentences:
            return np.zeros((0, 384), dtype=np.float32)
        return np.asarray(
            self.sbert.encode(
                list(sentences), batch_size=32, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------ CKIP word seg
    @functools.cached_property
    def ckip_ws(self):
        from ckip_transformers.nlp import CkipWordSegmenter

        dev = 0 if self.device.startswith("cuda") else -1
        model = CkipWordSegmenter(model_name=self.ckip_ws_id, device=dev)
        self._versions[self.ckip_ws_id] = _hf_revision(self.ckip_ws_id)
        return model

    def segment(self, sentences: Sequence[str]) -> list[list[str]]:
        if not sentences:
            return []
        # 16, not 128: CKIP pads each batch to its longest sentence, and 128 long ones
        # took more than the 2.4 GB a shared-GPU run is allowed (ASDPIPE_GPU_MEM_FRACTION).
        return self.ckip_ws(list(sentences), batch_size=16, show_progress=False)

    # ------------------------------------------------------------------ F0 (C stream)
    def track_f0(self, audio: np.ndarray, sr: int, fmin: float, fmax: float,
                 hop_s: float) -> tuple[np.ndarray, np.ndarray]:
        """torchcrepe F0 + periodicity. Search range is caller-supplied and must be 100-600 Hz.

        Cached on disk when ``cache_dir`` is set (``f0_cache_dir`` if that is). That is only
        sound because the output is deterministic -- the dither seed below is what makes it so --
        and it matters because CREPE over a whole session is the single most expensive stage,
        while the audio it runs on never changes between the convention-gate transforms. The key
        holds the audio, the search range and hop, the CREPE model and the seed; it does not hold
        the device, the batch size or torchcrepe's version, so a track cached by another run was
        made under settings this key cannot vouch for (``use_fresh_f0_cache``).
        """
        path = None
        f0_dir = self.f0_cache_dir or (os.path.join(self.cache_dir, "f0") if self.cache_dir
                                       else None)
        if f0_dir:
            import hashlib

            h = hashlib.blake2s(digest_size=16)
            h.update(np.ascontiguousarray(audio, dtype=np.float32).tobytes())
            h.update(f"{sr}|{fmin}|{fmax}|{hop_s}|{self.crepe_model}|{self.f0_seed}".encode())
            path = os.path.join(f0_dir, f"{h.hexdigest()}.npz")
            if os.path.exists(path):
                z = np.load(path)
                return z["f0"], z["periodicity"]
        f0, per = self._track_f0_uncached(audio, sr, fmin, fmax, hop_s)
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez(path, f0=f0, periodicity=per)
        return f0, per

    def _track_f0_uncached(self, audio: np.ndarray, sr: int, fmin: float, fmax: float,
                           hop_s: float) -> tuple[np.ndarray, np.ndarray]:
        import torch
        import torchcrepe

        hop = max(1, int(round(hop_s * sr)))
        x = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None]
        # torchcrepe adds unseeded random dither to every pitch value:
        # `convert.bins_to_cents` calls `dither()`, which draws triangular noise of
        # +/-CENTS_PER_BIN (+/-20 cents = +/-0.2 semitones) from scipy's global RNG. Two
        # calls on identical audio therefore disagree on 100% of frames -- measured here at
        # a median of 0.079 and a maximum of 0.34 semitones, with 39% of voiced frames
        # differing by more than 0.1 semitones. That makes every F0-derived feature
        # irreproducible, which a research pipeline cannot have. The global RNG state is
        # saved and restored so seeding does not leak into anything else.
        state = np.random.get_state()
        np.random.seed(self.f0_seed)
        try:
            with torch.inference_mode():
                f0, periodicity = torchcrepe.predict(
                    x, sr, hop_length=hop, fmin=fmin, fmax=fmax,
                    model=self.crepe_model, batch_size=CREPE_BATCH_SIZE, device=self.device,
                    return_periodicity=True, pad=True,
                )
        finally:
            np.random.set_state(state)
        return f0[0].cpu().numpy().astype(np.float64), periodicity[0].cpu().numpy().astype(np.float64)

    def use_fresh_f0_cache(self, parent: str | os.PathLike) -> str:
        """Send F0 tracks to a cache directory made now, so that none is older than this run
        (ruling R33), and return its path.

        The usual cache is keyed on the audio, not on how the track was made, so a track from an
        earlier process (another device, batch size or torchcrepe version) would be served as if
        it were this run's. The labelled run must not process one group from such tracks and the
        other group afresh. The directory is named by time and process id and created with
        ``exist_ok=False``: one that exists already is an error, never a cache to reuse.
        """
        path = os.path.join(parent, f"f0_{time.strftime('%Y%m%d-%H%M%S')}_{os.getpid()}")
        os.makedirs(path, exist_ok=False)
        self.f0_cache_dir = path
        return path

    def crepe_settings(self) -> dict[str, object]:
        """What decides CREPE's output besides the audio: model, batch size, device and the
        torchcrepe version. Recorded with the labelled run's results (ruling R33)."""
        import importlib.metadata

        try:
            version = importlib.metadata.version("torchcrepe")     # the package has no __version__
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
        return {"model": self.crepe_model, "batch_size": CREPE_BATCH_SIZE, "device": self.device,
                "torchcrepe_version": version}

    def f0_cross_check(self, audio: np.ndarray, sr: int, fmin: float, fmax: float,
                       hop_s: float, max_seconds: float = 60.0) -> dict:
        """Audit P1-3: cross-check torchcrepe against pYIN on a sample of the audio.

        The audit asks for three algorithms agreeing above 90% with octave errors below 3%.
        Two are implemented here (CREPE and pYIN); Praat autocorrelation is the missing
        third and is recorded as such. pYIN is slow, so only the first `max_seconds` of the
        supplied audio are checked -- enough to detect a systematic octave problem, which is
        what the gate is for.
        """
        defuse_speechbrain_lazy_modules()   # librosa's lazy loader walks sys.modules
        import librosa

        x = np.ascontiguousarray(audio[: int(max_seconds * sr)], dtype=np.float32)
        if x.size < sr:
            return {"n_frames": 0, "agreement": float("nan"), "octave_error": float("nan")}
        f0_crepe, periodicity = self.track_f0(x, sr, fmin, fmax, hop_s)
        f0_pyin, voiced, _ = librosa.pyin(
            x.astype(np.float64), fmin=fmin, fmax=fmax, sr=sr,
            frame_length=1024, hop_length=max(1, int(hop_s * sr)),
        )
        n = min(len(f0_crepe), len(f0_pyin))
        ok = (periodicity[:n] >= 0.4) & np.isfinite(f0_pyin[:n]) & voiced[:n]
        if ok.sum() < 50:
            return {"n_frames": int(ok.sum()), "agreement": float("nan"),
                    "octave_error": float("nan")}
        ratio = f0_crepe[:n][ok] / f0_pyin[:n][ok]
        octave = ((ratio >= 1.9) & (ratio <= 2.1)) | ((ratio >= 0.45) & (ratio <= 0.55))
        return {
            "n_frames": int(ok.sum()),
            "agreement": float(np.mean((ratio >= 0.95) & (ratio <= 1.05))),
            "octave_error": float(np.mean(octave)),
            "median_ratio": float(np.median(ratio)),
            "algorithms": "torchcrepe + pYIN (Praat autocorrelation not implemented)",
        }

    # ------------------------------------------------------------------ bookkeeping
    def versions(self) -> dict[str, str]:
        """Everything loaded so far, for audit P0-0F."""
        import importlib

        v = dict(self._versions)
        for pkg in ("torch", "librosa", "opensmile", "parselmouth", "speechbrain",
                    "sentence_transformers", "ckip_transformers", "transformers",
                    "sklearn", "numpy", "scipy"):
            try:
                v[pkg] = getattr(importlib.import_module(pkg), "__version__", "?")
            except Exception:
                v[pkg] = "not-imported"
        try:
            import torchcrepe  # noqa: F401

            v["torchcrepe"] = f"model={self.crepe_model}"
        except Exception:
            pass
        v["device"] = self.device
        return v

    def warm_up(self) -> None:
        """Force every download up front so timing measurements are not polluted."""
        _ = self.smile_lld, self.smile_functionals, self.ecapa, self.sbert, self.ckip_ws


def _hf_revision(model_id: str) -> str:
    """Resolved commit hash of a Hugging Face repo (audit P0-0F). Best effort, never raises."""
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(model_id).sha or "unknown"
    except Exception:
        return "unknown"
