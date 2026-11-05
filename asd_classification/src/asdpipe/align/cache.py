"""Caching wrapper around an aligner.

MFA costs about a minute per 40-minute session, which is fine once and intolerable on
every re-run of the feature extraction. The key is a content hash of exactly the things
that change the answer: the audio, the turn boundaries, the word segmentation fed to the
aligner, and the aligner's own identity and settings.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..contracts import SyllableSpan, Turn
from .base import Aligner

__all__ = ["CachedAligner"]


class CachedAligner(Aligner):
    def __init__(self, inner: Aligner, cache_dir: Path, identity: Mapping | None = None):
        self.inner = inner
        self.name = f"cached:{inner.name}"
        self.cache_dir = Path(cache_dir) / "align"
        self.identity = dict(identity or {})
        self.hits = 0
        self.misses = 0

    def available(self) -> bool:
        return True                      # a cache hit does not need the backend

    def _key(self, audio: np.ndarray, sample_rate: int, turns: Sequence[Turn],
             words: Mapping[int, Sequence[str]] | None) -> str:
        h = hashlib.blake2s(digest_size=12)
        h.update(json.dumps({"aligner": self.inner.name, **self.identity},
                            sort_keys=True, default=str).encode("utf-8"))
        h.update(str(sample_rate).encode("utf-8"))
        h.update(np.ascontiguousarray(audio, dtype=np.float32).tobytes())
        for t in turns:
            h.update(f"{t.turn_id}|{t.t_start:.6f}|{t.t_end:.6f}".encode("utf-8"))
            if words is not None:
                h.update(("|".join(words.get(t.turn_id, ()))).encode("utf-8"))
        return h.hexdigest()

    def align(self, audio: np.ndarray, sample_rate: int, turns: Sequence[Turn],
              words: Mapping[int, Sequence[str]] | None = None
              ) -> dict[int, tuple[SyllableSpan, ...]]:
        path = self.cache_dir / f"{self._key(audio, sample_rate, turns, words)}.pkl"
        if path.exists():
            self.hits += 1
            with path.open("rb") as fh:
                return pickle.load(fh)
        result = self.inner.align(audio, sample_rate, turns, words)
        self.misses += 1
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return result
