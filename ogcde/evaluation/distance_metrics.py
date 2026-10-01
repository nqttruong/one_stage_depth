from __future__ import annotations

from typing import Dict, List

import numpy as np


def distance_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(gt) & (gt > 0)
    pred = pred[valid]
    gt = gt[valid]
    if pred.size == 0:
        return {"AbsRel": float("nan"), "SqRel": float("nan"), "RMSE": float("nan"), "delta1": float("nan"), "n": 0}
    abs_rel = np.mean(np.abs(pred - gt) / gt)
    sq_rel = np.mean(((pred - gt) ** 2) / gt)
    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    ratio = np.maximum(pred / gt, gt / pred)
    return {
        "AbsRel": float(abs_rel),
        "SqRel": float(sq_rel),
        "RMSE": float(rmse),
        "delta1": float(np.mean(ratio < 1.25)),
        "n": int(pred.size),
    }


def bearing_bins(theta_deg: np.ndarray) -> List[tuple]:
    bins = [
        (0.0, 5.0),
        (5.0, 10.0),
        (10.0, 15.0),
        (15.0, 20.0),
        (20.0, 90.0),
    ]
    out = []
    for lo, hi in bins:
        mask = (theta_deg >= lo) & (theta_deg < hi)
        if np.any(mask):
            out.append((lo, hi, mask))
    return out
