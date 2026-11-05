"""Configuration loading. All thresholds live in `configs/` and are frozen (audit P0-4)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml

__all__ = ["REPO_ROOT", "CONFIG_DIR", "OUTPUT_DIR", "CACHE_DIR",
           "load_pipeline_config", "load_decision_rules", "deep_update"]

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
OUTPUT_DIR = REPO_ROOT / "outputs"
CACHE_DIR = REPO_ROOT / "cache"


def deep_update(base: dict, other: Mapping) -> dict:
    out = copy.deepcopy(base)
    for k, v in other.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_pipeline_config(overrides: Mapping | None = None) -> dict[str, Any]:
    cfg = _load(CONFIG_DIR / "pipeline.yaml")
    return deep_update(cfg, overrides or {})


def load_decision_rules() -> dict[str, Any]:
    """R5: this file must exist before any training run."""
    path = CONFIG_DIR / "decision_rules.yaml"
    if not path.exists():
        raise FileNotFoundError(
            "R5 requires configs/decision_rules.yaml to be committed before training"
        )
    return _load(path)
