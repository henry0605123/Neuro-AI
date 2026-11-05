"""M3 -- pooling and assembly into the subject-level table.

Two things are decided here and frozen (audit P0-4):

* which functionals are applied to which series (in `configs/pipeline.yaml`);
* which subset of them baseline B0 sees. R5 pins B0 at 36-66 dimensions, and the full
  feature set is far larger than that, so B0 gets a pre-specified reduced set. The rule is
  written below, not tuned against results -- see DEVIATIONS D5.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .contracts import SubjectRow, SubjectTable

__all__ = ["build_table", "b0_column_mask", "stream_blocks", "B0_E_FEATURES"]

# Frozen E-stream shortlist for B0: one indicator per construct, no redundant functionals.
B0_E_FEATURES: tuple[str, ...] = (
    "e_g1_similarity_median",
    "e_g1_low_ratio",
    "e_f1_embed_high_ratio",
    "e_f1_overlap_mean",
    "e_f3_repeated_type_ratio",
    "e_f3_repeated_token_ratio",
    "e_mlu_words",
    "e_ttr",
    "e_f8_asymmetry",
    "e_volume_following_slope",
    # e_latency_median removed 2026-10-01 (health check P4): on the raw transcripts it
    # measured how each pipeline stitched the timeline, and B0 must not carry that.
)


def build_table(rows: Sequence[SubjectRow]) -> SubjectTable:
    return SubjectTable(rows)


def _keep_b(name: str) -> bool:
    if name.startswith("b7_"):
        return name == "b7_mean_pairwise_cosdist"          # the frozen primary indicator
    return name.endswith("_mean") or name.endswith("_std")


def _keep_c(name: str) -> bool:
    if "_thr" in name:                                      # pause-threshold sensitivity runs
        return False
    if name.startswith("c_residual_f0"):
        return name.endswith("_mean") or name.endswith("_std")
    return True


def b0_column_mask(table: SubjectTable, cfg: Mapping) -> np.ndarray:
    """Boolean mask over the concatenated [B | C | E] feature vector for baseline B0."""
    b0_cfg = cfg["features"]["b0"]
    if b0_cfg.get("use_full_set", False):
        return np.ones(len(table.all_feature_names()), dtype=bool)

    mask = np.concatenate([
        np.array([_keep_b(n) for n in table.b_names], dtype=bool),
        np.array([_keep_c(n) for n in table.c_names], dtype=bool),
        np.array([n in B0_E_FEATURES for n in table.e_names], dtype=bool),
    ])
    lo, hi = b0_cfg["target_dim_range"]
    dim = int(mask.sum())
    if not (lo <= dim <= hi):
        raise ValueError(
            f"R5 pins baseline B0 at {lo}-{hi} dimensions but the frozen reduced set has "
            f"{dim}. Adjust the frozen lists in assemble.py, never the results."
        )
    return mask


def stream_blocks(table: SubjectTable) -> dict[str, tuple[int, int]]:
    """Column ranges of each stream inside the concatenated feature matrix."""
    nb, nc, ne = len(table.b_names), len(table.c_names), len(table.e_names)
    return {"b": (0, nb), "c": (nb, nb + nc), "e": (nb + nc, nb + nc + ne)}


def concat_features(table: SubjectTable) -> np.ndarray:
    return np.concatenate([table.X_b, table.X_c, table.X_e], axis=1)
