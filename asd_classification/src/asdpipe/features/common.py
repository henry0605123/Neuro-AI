"""Small shared helpers for the three feature streams."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..contracts import Speaker, Turn

__all__ = ["slice_turn", "rms_db", "snr_db", "noise_floor_energy", "frame_energy", "pool",
           "nan_safe", "npvi", "ols_slope", "child_turns", "examiner_turns", "sliding_windows"]

_EPS = 1e-12


def slice_turn(audio: np.ndarray, sr: int, turn: Turn) -> np.ndarray:
    a = max(0, int(round(turn.t_start * sr)))
    b = min(len(audio), int(round(turn.t_end * sr)))
    return audio[a:b] if b > a else np.zeros(0, dtype=audio.dtype)


def child_turns(turns: Sequence[Turn]) -> list[Turn]:
    return [t for t in turns if t.speaker is Speaker.CHILD]


def examiner_turns(turns: Sequence[Turn]) -> list[Turn]:
    return [t for t in turns if t.speaker is Speaker.EXAMINER]


def rms_db(x: np.ndarray) -> float:
    """Level of a waveform in dB. Always measured from the signal, never read off metadata."""
    if x.size == 0:
        return float("nan")
    return float(20.0 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + _EPS))


def frame_energy(x: np.ndarray, frame: int = 512) -> np.ndarray:
    n = x.size // frame
    if n == 0:
        return np.zeros(0)
    return np.square(x[: n * frame].reshape(n, frame), dtype=np.float64).mean(axis=1)


def noise_floor_energy(session: np.ndarray, quantile: float = 0.05) -> float:
    """Room-noise energy estimated from the whole session, i.e. from the silences.

    Estimating the floor inside a single turn is wrong: a short turn contains no silence,
    so the quietest frame is still speech and the SNR collapses. Audit P1-5 needs a
    segmental SNR, so the floor comes from the session and the signal from the turn.
    """
    e = frame_energy(session)
    return float(np.quantile(e, quantile)) if e.size else _EPS


def snr_db(x: np.ndarray, noise_energy: float) -> float:
    """Segmental SNR of one turn against a session-level noise floor (audit P1-5 gate)."""
    e = frame_energy(x)
    if e.size == 0:
        return float("nan")
    signal = float(np.quantile(e, 0.9))
    return float(10.0 * np.log10((signal + _EPS) / (max(noise_energy, _EPS))))


def nan_safe(value: float, default: float = 0.0) -> float:
    v = float(value)
    return default if (np.isnan(v) or np.isinf(v)) else v


def pool(values: np.ndarray, stats: Sequence[str], prefix: str,
         min_n: int = 1) -> dict[str, float]:
    """Apply the frozen functional set to one series (audit P0-4: the set never changes).

    Fewer than ``min_n`` finite values gives NaN, not 0 (health check P12).
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    out: dict[str, float] = {}
    for s in stats:
        if v.size < max(1, min_n):
            out[f"{prefix}_{s}"] = float("nan")
            continue
        if s == "mean":
            out[f"{prefix}_{s}"] = float(np.mean(v))
        elif s == "std":
            out[f"{prefix}_{s}"] = float(np.std(v))
        elif s == "median":
            out[f"{prefix}_{s}"] = float(np.median(v))
        elif s.startswith("p"):
            out[f"{prefix}_{s}"] = float(np.percentile(v, float(s[1:])))
        elif s == "iqr":
            out[f"{prefix}_{s}"] = float(np.percentile(v, 75) - np.percentile(v, 25))
        else:
            raise ValueError(f"unknown functional {s!r}")
    return out


def min_values(cfg: Mapping) -> int:
    """The P12 floor for a per-subject statistic (pipeline.yaml features.min_data)."""
    return int(cfg["features"].get("min_data", {}).get("values_per_statistic", 1))


def stat(values: Sequence[float], fn, min_n: int) -> float:
    """``fn(values)`` over the finite values, or NaN with fewer than ``min_n`` of them."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(fn(v)) if v.size >= max(1, min_n) else float("nan")


def npvi(durations: Sequence[float]) -> float:
    """Normalised Pairwise Variability Index (Grabe & Low 2002).

        nPVI = 100 / (m-1) * sum_k |d_k - d_{k+1}| / ((d_k + d_{k+1}) / 2)
    """
    d = np.asarray([x for x in durations if x > 0], dtype=np.float64)
    if d.size < 2:
        return float("nan")
    num = np.abs(d[:-1] - d[1:])
    den = (d[:-1] + d[1:]) / 2.0
    return float(100.0 * np.mean(num / np.maximum(den, _EPS)))


def ols_slope(x: Sequence[float], y: Sequence[float]) -> float:
    """Least-squares slope of y on x. NaN when there is no variation to regress on."""
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[ok], ya[ok]
    if xa.size < 3:
        return float("nan")
    vx = float(np.var(xa))
    if vx < _EPS:
        return float("nan")
    return float(np.cov(xa, ya, bias=True)[0, 1] / vx)


def sliding_windows(values: Sequence[float], size: int) -> list[np.ndarray]:
    v = np.asarray(values, dtype=np.float64)
    if v.size < size:
        return []
    return [v[i : i + size] for i in range(v.size - size + 1)]
