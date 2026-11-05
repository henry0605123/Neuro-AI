"""Forced alignment (R6): one interface, two backends.

`GroundTruthAligner` serves the synthetic corpus, `MFAAligner` the real one, and
`CachedAligner` wraps either so a re-run of the feature extraction does not pay for MFA
again. The C stream only ever sees the interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .base import Aligner, AlignmentUnavailable
from .cache import CachedAligner
from .gt import GroundTruthAligner
from .mfa import SYLLABIFY_VERSION, MFAAligner

__all__ = ["Aligner", "AlignmentUnavailable", "CachedAligner", "GroundTruthAligner",
           "MFAAligner", "aligner_from_config"]


def aligner_from_config(cfg: Mapping, cache_dir: Path | None = None,
                        cached: bool = True) -> Aligner:
    """Build the real-data aligner from `configs/pipeline.yaml: models.mfa`."""
    mfa = cfg["models"]["mfa"]
    inner = MFAAligner(
        acoustic_model=mfa["acoustic_model"],
        dictionary=mfa["dictionary"],
        num_jobs=int(mfa.get("num_jobs", 8)),
        binary=mfa.get("binary") or None,
        g2p_model=mfa.get("g2p_model") or None,
        beam=int(mfa.get("beam") or 0) or None,
        fine_tune=bool(mfa.get("fine_tune", False)),
        timeout_s=int(mfa.get("timeout_s", 5400)),
    )
    if not cached or cache_dir is None:
        return inner
    identity = {k: mfa.get(k) for k in
                ("acoustic_model", "dictionary", "g2p_model", "beam", "fine_tune")}
    identity["syllabify_version"] = SYLLABIFY_VERSION
    return CachedAligner(inner, cache_dir, identity)
