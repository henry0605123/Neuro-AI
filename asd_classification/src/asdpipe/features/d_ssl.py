"""D stream -- SSL benchmark arm (milestone M8, plan doc 5.5).

Since v3.5 the D stream is **off to the side**: evaluated on its own, never fused. It
answers one question that has to be answered -- "does a general-purpose SSL representation
contain signal the clinical features missed?" -- and it answers the reviewer's inevitable
"why didn't you compare against SSL?".

Protocol, frozen (plan doc 5.5, audit P1-11):
  1. pre-select a single layer from probing priors, record the number, freeze it
     (learning layer weights would break both the parameter budget and the convexity claim);
  2. mean + standard-deviation statistical pooling (1024 -> 2048 dims);
  3. **PCA-20 fitted inside the training fold**, then logistic regression;
  4. identical splits, identical seeds, subject-level evaluation.

Nothing here downloads anything during `make smoke`: `enabled` is false in
`configs/pipeline.yaml`, and `tests/test_d_ssl.py` skips unless the backbone is present.
Re-inclusion into the fusion is future work, gated on "gap >= MDE and n > 150".
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..contracts import Fold, SubjectRecord
from ..cv.guards import FoldScope

__all__ = ["DArmUnavailable", "backbone_available", "extract_d_embeddings", "fit_predict_d"]


class DArmUnavailable(RuntimeError):
    """Raised when the benchmark arm is requested but the backbone is not present."""


def backbone_available(model_id: str) -> bool:
    try:
        from huggingface_hub import HfApi

        HfApi().model_info(model_id)
        return True
    except Exception:
        return False


def extract_d_embeddings(subjects: Sequence[SubjectRecord], cfg: Mapping,
                         device: str | None = None) -> np.ndarray:
    """Frozen backbone -> pre-selected layer -> mean+std pooling. (n_subjects, 2 * hidden)."""
    d_cfg = cfg["models"]["d_arm"]
    if not d_cfg.get("enabled", False):
        raise DArmUnavailable(
            "the D benchmark arm is disabled in configs/pipeline.yaml (models.d_arm.enabled)"
        )
    import torch
    from transformers import AutoModel

    from .backends import assert_compliant
    from .common import child_turns, slice_turn

    model_id = d_cfg["backbone"]
    assert_compliant(model_id, cfg["models"].get("banned_substrings", ()))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(model_id, output_hidden_states=True).to(device).eval()
    layer = int(d_cfg["layer"])

    out = []
    with torch.inference_mode():
        for s in subjects:
            kids = child_turns(s.session.turns)
            audio = np.concatenate([slice_turn(s.audio, s.sample_rate, t) for t in kids])
            chunks = [audio[i : i + s.sample_rate * 20]
                      for i in range(0, len(audio), s.sample_rate * 20)]
            feats = []
            for ch in chunks:
                if len(ch) < s.sample_rate:
                    continue
                x = torch.tensor(ch, dtype=torch.float32, device=device)[None]
                h = model(x).hidden_states[layer][0]
                feats.append(h.float().cpu().numpy())
            if not feats:
                out.append(np.zeros(2 * model.config.hidden_size))
                continue
            f = np.concatenate(feats, axis=0)
            out.append(np.concatenate([f.mean(axis=0), f.std(axis=0)]))
    return np.stack(out)


def fit_predict_d(scope: FoldScope, embeddings: np.ndarray, y: np.ndarray,
                  cfg: Mapping) -> dict:
    """In-fold PCA-20 then logistic regression. PCA outside the fold would be leakage."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    pca_dim = int(cfg["models"]["d_arm"]["pca_dim"])
    scaler = scope.fit(StandardScaler(), embeddings, None, stage="d:scale")
    Xz = scaler.transform(embeddings)
    pca = scope.fit(PCA(n_components=min(pca_dim, len(scope.train_idx) - 1)),
                    Xz, None, stage="d:pca")
    Xp = pca.transform(Xz)
    clf = LogisticRegression(C=1.0, penalty="l2", max_iter=5000, class_weight="balanced")
    scope.fit(clf, Xp, y, stage="d:lr")
    all_scores = clf.decision_function(Xp)
    return {"scores": all_scores[scope.test_idx], "scores_all": all_scores,
            "layer": int(cfg["models"]["d_arm"]["layer"]), "pca_dim": int(pca_dim)}
