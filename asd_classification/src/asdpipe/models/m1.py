"""Primary model M1 (R5, plan doc 5.0.2).

    C stream : TCN, 64 channels x 4 dilated conv layers, dilations 1/2/4/8
               -> masked temporal pooling -> embedding
    B, E     : statistical pooling (already pooled upstream), standardised in-fold
    fusion   : concatenate the three -> MLP with 64 hidden units -> 2 logits
    loss     : class-weighted cross-entropy. **Not** Focal -- section 5.7 demotes Focal to an
               ablation because gamma is an extra hyper-parameter and it distorts the output
               probabilities, which conflicts with shipping a calibrated probability.

Overfitting guards, all mandatory to report (section 5.7 / audit P1-11): dropout, weight
decay, early stopping **inside the training fold**, multiple seeds, cross-seed SD.

The architecture is deliberately not made configurable beyond what `configs/pipeline.yaml`
exposes, and `assert_architecture` re-checks the pinned numbers at construction time.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
import torch.nn as nn

from ..contracts import SubjectTable
from ..cv.guards import FoldScope
from ..cv.splits import inner_folds

__all__ = ["M1Net", "fit_predict_m1", "count_parameters"]


class TCN(nn.Module):
    """Dilated temporal convolution stack over the per-syllable sequence."""

    def __init__(self, in_dim: int, channels: int, dilations: tuple[int, ...],
                 kernel_size: int, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.project = nn.Conv1d(in_dim, channels, kernel_size=1)
        prev = in_dim
        for d in dilations:
            pad = (kernel_size - 1) * d // 2
            self.blocks.append(nn.Sequential(
                nn.Conv1d(prev, channels, kernel_size, dilation=d, padding=pad),
                nn.ReLU(),
                nn.Dropout(dropout),
            ))
            prev = channels
        self.channels = channels

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> (B, D, T)
        m = mask.unsqueeze(1)                       # (B, 1, T)
        # Re-apply the mask at every layer, not only at pooling. With dilations up to 8 and
        # kernel 3 the receptive field reaches ~17 steps, so anything left in the padded
        # region is convolved back into the valid region and silently changes the embedding.
        h = x.transpose(1, 2) * m
        residual = self.project(h) * m
        for i, block in enumerate(self.blocks):
            h = block(h) * m
            if i == 0:
                h = (h + residual) * m
        denom = m.sum(dim=2).clamp(min=1.0)
        return (h * m).sum(dim=2) / denom


class M1Net(nn.Module):
    def __init__(self, seq_dim: int, tab_dim: int, cfg: Mapping):
        super().__init__()
        self.tcn = TCN(
            in_dim=seq_dim,
            channels=int(cfg["tcn_channels"]),
            dilations=tuple(int(d) for d in cfg["tcn_dilations"]),
            kernel_size=int(cfg["tcn_kernel_size"]),
            dropout=float(cfg["dropout"]),
        )
        hidden = int(cfg["mlp_hidden"])
        self.head = nn.Sequential(
            nn.Linear(self.tcn.channels + tab_dim, hidden),
            nn.ReLU(),
            nn.Dropout(float(cfg["dropout"])),
            nn.Linear(hidden, 2),
        )

    def forward(self, seq: torch.Tensor, mask: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.tcn(seq, mask), tab], dim=1))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def assert_architecture(cfg: Mapping) -> None:
    """R5 pins these numbers; a silent edit to the config must fail loudly."""
    if int(cfg["tcn_channels"]) != 64:
        raise ValueError("R5: the C-stream TCN must have 64 channels")
    if int(cfg["tcn_layers"]) != 4 or list(cfg["tcn_dilations"]) != [1, 2, 4, 8]:
        raise ValueError("R5: the TCN must be 4 dilated layers with dilations 1/2/4/8")
    if int(cfg["mlp_hidden"]) != 64:
        raise ValueError("R5: the MLP classification head must have 64 hidden units")


def _finite(X: np.ndarray) -> np.ndarray:
    X = np.array(X, dtype=np.float64)
    X[~np.isfinite(X)] = np.nan
    return X


def _impute_median(train_rows: np.ndarray, X: np.ndarray) -> np.ndarray:
    """NaN -> the training rows' median of that column; 0 if the column is all NaN there."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN column
        med = np.nanmedian(X[train_rows], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return np.where(np.isfinite(X), X, med)


def _standardise(train_rows: np.ndarray, X: np.ndarray) -> np.ndarray:
    mu = X[train_rows].mean(axis=0)
    sd = X[train_rows].std(axis=0)
    return (X - mu) / np.maximum(sd, 1e-8)


def fit_predict_m1(scope: FoldScope, table: SubjectTable, cfg: Mapping,
                   torch_seed: int = 0, device: str | None = None,
                   drop_stream: str | None = None) -> dict:
    """``drop_stream`` supports the leave-one-stream-out post-hoc ablation (plan doc 5.6).

    Attribution is by ablation only. M1 never claims "learned stream weights": the v3.4
    design that did so was shown to be unidentifiable, and swapping in an MLP does not fix
    that, so the honest tool is the drop in AUC when a stream is removed.
    """
    m_cfg = cfg["cv"]["m1"]
    assert_architecture(m_cfg)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    y = table.y

    blocks = []
    if drop_stream != "b":
        blocks.append(table.X_b)
    if drop_stream != "e":
        blocks.append(table.X_e)
    if not blocks:
        blocks = [np.zeros((len(y), 1))]
    tab_raw = _finite(np.concatenate(blocks, axis=1))
    seq_raw = _finite(table.seq.astype(np.float64))
    if drop_stream == "c":
        seq_raw = np.zeros_like(seq_raw)

    # ---- imputation and standardisation are fitted on training rows only ----
    # An unmeasurable value (NaN, health check P12) takes the training rows' median in the
    # tabular block and the training mean in the sequence -- 0 after z-scoring. It used to
    # be 0 *before* scaling, an arbitrary and often extreme z.
    def _fit_scalers(idx):
        tab = _standardise(idx, _impute_median(idx, tab_raw))
        flat = seq_raw[idx][table.seq_mask[idx] > 0]
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(flat, axis=0) if len(flat) else np.zeros(seq_raw.shape[2])
            sd = np.nanstd(flat, axis=0) if len(flat) else np.ones(seq_raw.shape[2])
        mu, sd = np.nan_to_num(mu), np.nan_to_num(sd, nan=1.0)
        seq = np.nan_to_num((seq_raw - mu) / np.maximum(sd, 1e-8))
        seq = seq * table.seq_mask[..., None]
        return tab, seq

    tab_z, seq_z = scope.fit_callable(_fit_scalers, stage="m1:standardise")

    seq_t = torch.tensor(seq_z, dtype=torch.float32, device=device)
    mask_t = torch.tensor(table.seq_mask, dtype=torch.float32, device=device)
    tab_t = torch.tensor(tab_z, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)

    # ---- inner validation sub-fold for early stopping (NEVER the outer test fold) ----
    folds = inner_folds(scope.train_idx, y, int(cfg["cv"]["inner"]["n_folds"]), scope.seed)
    inner = folds[0]
    inner_scope = scope.inner(inner)
    tr, va = inner_scope.check(inner.train_idx, "m1:train"), np.asarray(inner.test_idx)

    counts = np.bincount(y[tr], minlength=2).astype(np.float64)
    weights = torch.tensor(len(tr) / (2.0 * np.maximum(counts, 1.0)),
                           dtype=torch.float32, device=device)

    torch.manual_seed(torch_seed)
    model = M1Net(seq_z.shape[2], tab_z.shape[1], m_cfg).to(device)
    n_params = count_parameters(model)
    opt = torch.optim.AdamW(model.parameters(), lr=float(m_cfg["lr"]),
                            weight_decay=float(m_cfg["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(m_cfg["max_epochs"]))
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    batch = int(m_cfg["batch_size"])
    patience = int(m_cfg["patience"])
    best_state, best_metric, bad = None, -np.inf, 0
    rng = np.random.default_rng(torch_seed)

    def _val_metric() -> float:
        model.eval()
        with torch.inference_mode():
            logits = model(seq_t[va], mask_t[va], tab_t[va])
            s = (logits[:, 1] - logits[:, 0]).cpu().numpy()
        if len(np.unique(y[va])) < 2:
            return -float(np.mean((s - y[va]) ** 2))
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y[va], s))

    for epoch in range(int(m_cfg["max_epochs"])):
        model.train()
        order = rng.permutation(tr)
        for i in range(0, len(order), batch):
            idx = torch.tensor(order[i : i + batch], dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(seq_t[idx], mask_t[idx], tab_t[idx]), y_t[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0, foreach=True)
            opt.step()
        sched.step()
        metric = _val_metric()
        if metric > best_metric + 1e-6:
            best_metric, bad = metric, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        logits = model(seq_t, mask_t, tab_t)
        scores_all = (logits[:, 1] - logits[:, 0]).cpu().numpy()

    return {
        "scores": scores_all[scope.test_idx],
        "scores_all": scores_all,
        "n_params": n_params,
        "best_inner_metric": float(best_metric),
        "epochs_run": epoch + 1,
        "dropped_stream": drop_stream,
    }
