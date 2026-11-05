"""Gated segment-attention model, trained as a batch of independent small models (plan 3.5).

The model answers the plan's two questions with two of its own parts (plan section 1): a
**gate** per measure says which measures it uses, and **attention** over a child's segments
says which stretches of the conversation it weighed. For one model, with ``x`` a child's
standardised segments ``(T, Ds)`` and ``c`` its child-level measures ``(Dc,)``::

    z      = clamp(mu + sigma * eps, 0, 1)        # training; eps ~ N(0, 1), redrawn every step
    z      = clamp(mu, 0, 1)                      # prediction
    h      = tanh((x * z_seg) @ W1 + b1)          # (T, H)
    s      = h @ w2 + b2                          # how ASD-like each segment looks
    a      = softmax(h @ q) over real segments    # attention: 0 on padding, 1 in total
    logit  = sum_t a_t s_t + (c * z_child) @ wc + b0

The gates are the stochastic gates of Yamada et al. (ICML 2020). ``Phi(mu / sigma)`` is the
probability that a gate is open, so ``gate_lambda`` times their sum penalises the expected
number of open gates, and only the measures the data keeps paying for stay open. The loss of
a model is the class-weighted cross-entropy over its own training children, plus that penalty,
plus weight decay on the four weight arrays (not on the biases or the gates). Training is
full-batch Adam for a fixed number of epochs, with no early stopping (plan 3.5, step 3).

**Why a batch of models.** The permutation test retrains everything 1,000 times (plan 3.5,
step 7): 1,001 label sets x 5 repeats x 5 folds x 5 seeds is about 125,000 models of about
100 parameters. One at a time, Python overhead would dominate, so ``fit_predict`` stacks
``M`` models along a leading axis and trains them as one tensor program. They share nothing:
every parameter carries the model axis, the batch loss is the sum of the models' losses (so a
model's gradient is its own loss's gradient), and Adam updates every number on its own.
``mil.chunk_models`` (how many go into one batch) is therefore an execution setting, and a
model's result must not depend on which other models share its batch (Task 5 tests it). Two
choices make that hold in floating point too:

* each model draws its initial weights and all its gate noise from its own
  ``torch.Generator(seed)``, on the CPU, in one fixed order, so its numbers do not depend on
  the batch or on the device (a CUDA run moves the same draws to the GPU);
* the model computes in float64, on the CPU and on CUDA alike. Rounding follows the shape of
  the whole batch: on 250 synthetic models (masks, labels and seeds like Task 5's) float32
  moved a model's scores by up to 1e-5 between chunk sizes, the tolerance Task 5 checks, and
  CUDA float32 differed from the CPU by 6e-5. In float64 chunk sizes and devices agree to
  about 1e-14, for about a fifth more time on the CPU and twice the time on CUDA.

**R3.** ``prepare`` imputes (median) and standardises each model from that model's training
children alone. A column those children never measured, or never varied in (sd below
``SD_FLOOR``), carries no information in that fold and is 0 for every child. A test child's
logit is computed during training only to keep one shared tensor; its loss weight is exactly
0, so it adds nothing to any gradient.
"""

from __future__ import annotations

import functools
import math
import operator
import warnings
from dataclasses import dataclass, fields
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .bags import Bags

__all__ = ["MILConfig", "MILFit", "n_parameters", "prepare", "fit_predict"]

SD_FLOOR = 1e-6        # a training sd below this marks a column as constant in that fold
_DTYPE = torch.float64  # on every device; see the module docstring

# One model's initial weights are N(0, variance): W1 1/Ds (Ds = number of segment measures)
# and w2, q 1/H, as the brief writes them; wc starts near 0, so the child measures enter as a
# small correction until training says otherwise. Biases start at 0, the gates at gate_init.
_WC_INIT_SD = 0.1      # variance 0.01


@dataclass(frozen=True)
class MILConfig:
    """The fixed training settings (plan 3.5; ``mil.model`` in ``configs/pipeline.yaml``).

    The defaults are those frozen settings. gate_lambda was 0.05 in the brief and is 0.02 since
    the calibration on synthetic data recorded in DEVIATIONS D17.
    """

    hidden: int = 8
    epochs: int = 300
    lr: float = 0.01
    weight_decay: float = 1e-4
    gate_sigma: float = 0.5
    gate_init: float = 0.5
    gate_lambda: float = 0.02

    def __post_init__(self) -> None:
        if self.hidden < 1 or self.epochs < 1:
            raise ValueError(f"hidden and epochs must be at least 1, got {self.hidden} and "
                             f"{self.epochs}")
        if not (self.lr > 0 and self.gate_sigma > 0):
            raise ValueError(f"lr and gate_sigma must be positive, got {self.lr} and "
                             f"{self.gate_sigma}")
        if self.weight_decay < 0 or self.gate_lambda < 0:
            raise ValueError(f"weight_decay and gate_lambda cannot be negative, got "
                             f"{self.weight_decay} and {self.gate_lambda}")

    @classmethod
    def from_cfg(cls, cfg: Mapping) -> "MILConfig":
        """Read ``cfg["mil"]["model"]``. Every setting must be there and nothing else may be:
        a misspelt key would otherwise leave a default in force without anyone noticing."""
        given = cfg["mil"]["model"]
        names = {f.name for f in fields(cls)}
        unknown, missing = sorted(set(given) - names), sorted(names - set(given))
        if unknown or missing:
            raise ValueError(f"mil.model: unknown key(s) {unknown}, missing key(s) {missing}")
        return cls(**{f.name: type(f.default)(given[f.name]) for f in fields(cls)})


# eq=False for the reason Bags gives: an elementwise == of array fields can only raise.
@dataclass(frozen=True, eq=False)
class MILFit:
    """What ``fit_predict`` returns, one row per model, as float64 arrays.

    ``scores`` ``(M, N)``: the logit of every child, training and test alike, with every gate
    at its deterministic value ``clamp(mu, 0, 1)``. ``gates`` ``(M, Ds + Dc)``: those values,
    the segment measures first and then the child measures, in the Bags' column order.
    ``attention`` ``(M, N, T)``: exactly 0 on padding, summing to 1 over each child's real
    segments. A model whose gates are all shut sees no measure, and it gives every child
    exactly the same score (see ``_forward``), so an AUC counts those children as ties.
    """

    scores: np.ndarray
    gates: np.ndarray
    attention: np.ndarray


def _shapes(d_seg: int, d_child: int, hidden: int) -> dict[str, tuple[int, ...]]:
    """One model's parameters. ``fit_predict`` allocates from this table, so ``n_parameters``
    counts exactly what is trained."""
    return {"mu": (d_seg + d_child,), "W1": (d_seg, hidden), "b1": (hidden,),
            "w2": (hidden,), "b2": (), "q": (hidden,), "wc": (d_child,), "b0": ()}


def n_parameters(d_seg: int, d_child: int, hidden: int) -> int:
    """Trainable numbers in one model: 88 for 6 segment and 4 child measures at H = 8."""
    return sum(math.prod(shape) for shape in _shapes(d_seg, d_child, hidden).values())


def _train_mask(train_mask: np.ndarray, n_children: int) -> np.ndarray:
    train = np.asarray(train_mask)
    if train.dtype != bool or train.ndim != 2 or train.shape[1] != n_children:
        raise ValueError(f"train_mask must be a boolean (models, {n_children}) array; got "
                         f"{train.dtype} {train.shape}")
    if train.shape[0] == 0:
        raise ValueError("train_mask holds no model")
    empty = np.flatnonzero(~train.any(axis=1))
    if empty.size:
        raise ValueError(f"model(s) {empty[:5].tolist()} have no training child")
    return train


def _impute_standardise(fit_rows: np.ndarray, values: np.ndarray) -> np.ndarray:
    """``values`` ``(..., D)`` median-imputed, then z-scored, with statistics of ``fit_rows``
    ``(n, D)`` only (R3). The mean and sd are those of the imputed training values.

    A column with no value in ``fit_rows`` has no median, mean or sd, and a column whose
    training sd is below ``SD_FLOOR`` holds one value for every training child. Either way it
    carries no information in this fold, so it comes out 0 for everyone, the value of an
    average training child, and moves no score. A test child who differs there is not scaled
    by a missing or vanishing sd: divided by the floor, a tiny difference would enter at a
    million times its size.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # "All-NaN slice encountered"
        median = np.nanmedian(fit_rows, axis=0)
    filled = np.where(np.isnan(fit_rows), median, fit_rows)
    mean, sd = filled.mean(axis=0), filled.std(axis=0)
    dead = ~(sd >= SD_FLOOR)                                      # also True where sd is NaN
    out = (np.where(np.isnan(values), median, values) - mean) / np.where(dead, 1.0, sd)
    out[..., dead] = 0.0
    return out


def prepare(bags: Bags, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every model's inputs: ``(M, N, T, Ds)`` segments and ``(M, N, Dc)`` child measures.

    Model ``m`` fills a missing value with the median of its own training children
    (``train_mask[m]``) and standardises with their mean and sd, the segment measures over
    those children's real segments; a column they never measured or never varied in is 0.
    Padding rows stay exactly 0, as in ``Bags``. Models with the same training children (the
    seeds of one fold) share one computation.
    """
    train = _train_mask(train_mask, len(bags.ids))
    seg = bags.seg.astype(np.float64)
    child = bags.child.astype(np.float64)
    x = np.empty((train.shape[0],) + seg.shape)
    c = np.empty((train.shape[0],) + child.shape)
    first: dict[bytes, int] = {}
    for m, rows in enumerate(train):
        same = first.setdefault(rows.tobytes(), m)
        if same < m:
            x[m], c[m] = x[same], c[same]
            continue
        x[m] = _impute_standardise(seg[bags.mask & rows[:, None]], seg)
        x[m][~bags.mask] = 0.0
        c[m] = _impute_standardise(child[rows], child)
    return x, c


def _device(device: str | torch.device | None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _initial_state(seeds: Sequence[int], d_seg: int, d_child: int,
                   cfg: MILConfig) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Every model's starting parameters ``(M, ...)`` and all its gate noise
    ``(epochs, M, Ds + Dc)``, each model from its own generator.

    The draws are float64 on the CPU, in one order per model (W1, w2, q, wc, then the noise of
    every step), so a model's numbers do not depend on its batch or on the device.
    """
    shapes = _shapes(d_seg, d_child, cfg.hidden)
    params = {name: torch.zeros((len(seeds),) + shape, dtype=torch.float64)
              for name, shape in shapes.items()}
    params["mu"].fill_(cfg.gate_init)
    init_sd = {"W1": 1.0 / math.sqrt(max(d_seg, 1)), "w2": 1.0 / math.sqrt(cfg.hidden),
               "q": 1.0 / math.sqrt(cfg.hidden), "wc": _WC_INIT_SD}
    eps = torch.empty((cfg.epochs, len(seeds), d_seg + d_child), dtype=torch.float64)
    for m, seed in enumerate(seeds):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        for name, sd in init_sd.items():
            params[name][m] = sd * torch.randn(shapes[name], generator=gen, dtype=torch.float64)
        eps[:, m] = torch.randn((cfg.epochs, d_seg + d_child), generator=gen,
                                dtype=torch.float64)
    return params, eps


def _forward(p: Mapping[str, torch.Tensor], x: torch.Tensor, c: torch.Tensor,
             padding: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Logits ``(M, N)`` and attention ``(M, N, T)`` of every model for gates ``z``."""
    m, n, t, d_seg = x.shape
    z_seg, z_child = z[:, :d_seg], z[:, d_seg:]
    # (x * z_seg) @ W1 equals x @ (z_seg * W1), the gates scaling W1's rows; this way the
    # gated copy of x, the largest tensor here, is never formed
    w1 = z_seg.unsqueeze(-1) * p["W1"]                                     # (M, Ds, H)
    h = torch.tanh(torch.baddbmm(p["b1"].unsqueeze(1), x.reshape(m, n * t, d_seg), w1))
    s_e = torch.bmm(h, torch.stack((p["w2"], p["q"]), dim=-1)).reshape(m, n, t, 2)
    s = s_e[..., 0] + p["b2"][:, None, None]                               # (M, N, T)
    a = torch.softmax(s_e[..., 1].masked_fill(padding, float("-inf")), dim=-1)
    pooled = (a * s).sum(-1)                                               # (M, N)
    # With every segment gate shut, every segment is the empty one (gated input 0), so the
    # pooled score is exactly the empty segment's score. Summed back from attention weights it
    # would differ in the last bits between bags of different lengths, and an AUC would rank
    # the children by their number of segments; so that value is used as it is. The child
    # term needs no such care: a shut child gate multiplies wc by exactly 0.
    empty = (torch.tanh(p["b1"]) * p["w2"]).sum(-1) + p["b2"]              # (M,)
    pooled = torch.where((z_seg > 0).any(-1, keepdim=True), pooled, empty.unsqueeze(-1))
    child = torch.bmm(c, (z_child * p["wc"]).unsqueeze(-1)).squeeze(-1)    # (M, N)
    return pooled + child + p["b0"].unsqueeze(-1), a


def fit_predict(bags: Bags, train_mask: np.ndarray, labels: np.ndarray, seeds: Sequence[int],
                cfg: MILConfig, device: str | torch.device | None = "cpu") -> MILFit:
    """Train ``M`` independent models at once and score every child with each.

    Model ``m`` trains on the children where ``train_mask[m]`` is True, with ``labels[m]``
    (1 = ASD) and ``seeds[m]``; labels of the other children are never read. Its positive
    class weight is its own TD / ASD count among those children, so every model needs both.
    ``device`` is a torch device, or ``"auto"`` for CUDA when available.
    """
    n_children = len(bags.ids)
    train = _train_mask(train_mask, n_children)
    y = np.asarray(labels)
    if y.shape != train.shape or not np.isin(y, (0, 1)).all():
        raise ValueError(f"labels must be a {train.shape} array of 0 (TD) and 1 (ASD); got "
                         f"shape {y.shape}")
    seeds = [operator.index(s) for s in seeds]     # an integer, never a rounded float
    if len(seeds) != train.shape[0]:
        raise ValueError(f"{train.shape[0]} model(s) need as many seeds; got {len(seeds)}")
    n_asd = (train & (y == 1)).sum(axis=1)
    n_td = (train & (y == 0)).sum(axis=1)
    lacking = np.flatnonzero((n_asd == 0) | (n_td == 0))
    if lacking.size:
        raise ValueError(f"model(s) {lacking[:5].tolist()}: the training children must include "
                         "both ASD and TD")

    dev = _device(device)
    as_tensor = functools.partial(torch.as_tensor, dtype=_DTYPE, device=dev)
    x_np, c_np = prepare(bags, train)
    x, c = as_tensor(x_np), as_tensor(c_np)
    padding = torch.as_tensor(~bags.mask, device=dev)
    # the cross-entropy's weight on each child: 1 / n_train on a training child, times the
    # model's TD / ASD ratio on a positive one, and exactly 0 on a test child
    share = train / train.sum(axis=1, keepdims=True)
    w_pos = as_tensor(share * (y == 1) * (n_td / n_asd)[:, None])
    w_neg = as_tensor(share * (y == 0))

    state, eps = _initial_state(seeds, bags.seg.shape[2], bags.child.shape[1], cfg)
    p = {name: value.to(device=dev, dtype=_DTYPE).requires_grad_() for name, value in state.items()}
    eps = eps.to(device=dev, dtype=_DTYPE)
    decayed = ("W1", "w2", "q", "wc")
    opt = torch.optim.Adam(p.values(), lr=cfg.lr)
    for step in range(cfg.epochs):
        z = torch.clamp(p["mu"] + cfg.gate_sigma * eps[step], 0.0, 1.0)
        logit, _ = _forward(p, x, c, padding, z)
        fit = (w_pos * F.softplus(-logit) + w_neg * F.softplus(logit)).sum(-1)
        open_gates = torch.special.ndtr(p["mu"] / cfg.gate_sigma).sum(-1)
        decay = sum(p[name].square().flatten(1).sum(-1) for name in decayed)
        loss = (fit + cfg.gate_lambda * open_gates + cfg.weight_decay * decay).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        z = torch.clamp(p["mu"], 0.0, 1.0)
        logit, attention = _forward(p, x, c, padding, z)
    return MILFit(scores=_to_numpy(logit), gates=_to_numpy(z), attention=_to_numpy(attention))


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()
