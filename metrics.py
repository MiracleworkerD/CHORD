from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


def binary_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    probs = torch.sigmoid(logits.detach()).float().cpu().numpy().reshape(-1)
    y = targets.detach().float().cpu().numpy().reshape(-1)
    pred = (probs >= 0.5).astype(np.float32)
    out = {
        "acc": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }
    try:
        out["auc"] = float(roc_auc_score(y, probs))
    except ValueError:
        out["auc"] = 0.5
    try:
        out["ap"] = float(average_precision_score(y, probs))
    except ValueError:
        out["ap"] = float(y.mean()) if len(y) else 0.0
    return out
