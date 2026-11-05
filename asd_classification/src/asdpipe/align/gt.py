"""Ground-truth aligner for the synthetic corpus (R6).

The generator already knows every syllable boundary and tone, so this aligner just hands
them back. It exists so the C stream code path is identical to the real one.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..contracts import SyllableSpan, Turn
from .base import Aligner

__all__ = ["GroundTruthAligner"]


class GroundTruthAligner(Aligner):
    name = "ground_truth"

    def align(
        self, audio: np.ndarray, sample_rate: int, turns: Sequence[Turn],
        words: Mapping[int, Sequence[str]] | None = None,
    ) -> dict[int, tuple[SyllableSpan, ...]]:
        out: dict[int, tuple[SyllableSpan, ...]] = {}
        for t in turns:
            if not t.syllables:
                raise ValueError(
                    f"turn {t.turn_id} has no ground-truth syllables; use a real aligner"
                )
            out[t.turn_id] = t.syllables
        return out
