"""Injectable forced-alignment interface (R6).

The C stream never calls MFA directly. It asks an ``Aligner`` for syllable spans, so the
synthetic path (ground-truth boundaries carried on the Turn) and the real path (MFA) are
swappable and the smoke test cannot fail because MFA is absent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Sequence

import numpy as np

from ..contracts import SyllableSpan, Turn

__all__ = ["Aligner", "AlignmentUnavailable"]


class AlignmentUnavailable(RuntimeError):
    """Raised when a real aligner is asked to run without its environment."""


class Aligner(ABC):
    name: str = "abstract"

    @abstractmethod
    def align(
        self, audio: np.ndarray, sample_rate: int, turns: Sequence[Turn],
        words: Mapping[int, Sequence[str]] | None = None,
    ) -> dict[int, tuple[SyllableSpan, ...]]:
        """Map ``turn_id`` -> syllable spans with tone labels.

        ``words`` is the word segmentation, needed by aligners that look words up in a
        pronunciation dictionary. Implementations that do not need it ignore it.
        """

    def available(self) -> bool:
        return True
