"""Evaluation metrics for OGCDE.

All functions accept numpy arrays or lists.

    (A) Depth:     AbsRel, RMSE, RMSE_log
    (B) δ:         δ1, δ2, δ3 (threshold 1.25, 1.25^2, 1.25^3)
    (C) DE:        object-level distance error on IoU>0.5 matches
    (D) CPE:       contact point pixel distance on IoU>0.5 matches
    (E) Stability: variance of distance per track over time
                   (requires tracking input — see evaluate_video.py)
"""

from __future__ import annotations

from typing import Dict, List
import numpy as np


# ---------------------------- depth metrics ---------------------------------

def depth_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    valid = (gt > 0) & (pred > 0) & np.isfinite(pred) & np.isfinite(gt)
    pred = pred[valid]
    gt = gt[valid]
    if pred.size == 0:
        return {"AbsRel": float("nan"), "SqRel": float("nan"),
                "RMSE": float("nan"), "RMSE_log": float("nan"),
                "delta1": float("nan"), "delta2": float("nan"), "delta3": float("nan"),
                "n": 0}

    abs_rel = np.mean(np.abs(pred - gt) / gt)
    sq_rel  = np.mean(((pred - gt) ** 2) / gt)
    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(pred) - np.log(gt)) ** 2))
    ratio = np.maximum(pred / gt, gt / pred)
    d1 = float(np.mean(ratio < 1.25))
    d2 = float(np.mean(ratio < 1.25 ** 2))
    d3 = float(np.mean(ratio < 1.25 ** 3))
    return {
        "AbsRel": float(abs_rel),
        "SqRel":  float(sq_rel),
        "RMSE": float(rmse),
        "RMSE_log": float(rmse_log),
        "delta1": d1, "delta2": d2, "delta3": d3,
        "n": int(pred.size),
    }


# ---------------------------- matching --------------------------------------

def bbox_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    lt_x = np.maximum(ax1[:, None], bx1[None, :])
    lt_y = np.maximum(ay1[:, None], by1[None, :])
    rb_x = np.minimum(ax2[:, None], bx2[None, :])
    rb_y = np.minimum(ay2[:, None], by2[None, :])
    w = (rb_x - lt_x).clip(min=0)
    h = (rb_y - lt_y).clip(min=0)
    inter = w * h
    union = area_a[:, None] + area_b[None, :] - inter + 1e-9
    return inter / union


def xywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=np.float64)
    xy = b[:, :2]
    wh = b[:, 2:4]
    return np.concatenate([xy - wh / 2, xy + wh / 2], -1)


def greedy_match(pred_xyxy, pred_score, gt_xyxy,
                 iou_thr=0.5, pred_cls=None, gt_cls=None):
    """Greedy score-sorted matching with IoU threshold.

    Returns (matched_pred_idx, matched_gt_idx, unmatched_pred_idx).
    If class arrays provided, only same-class pairs can match.
    """
    if len(pred_xyxy) == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, int)
    if len(gt_xyxy) == 0:
        return np.zeros(0, int), np.zeros(0, int), np.arange(len(pred_xyxy))

    order = np.argsort(-pred_score)
    ious = bbox_iou_xyxy(pred_xyxy, gt_xyxy)

    used = np.zeros(len(gt_xyxy), dtype=bool)
    matched_p, matched_g, unmatched_p = [], [], []
    for pi in order:
        row = ious[pi].copy()
        if pred_cls is not None and gt_cls is not None:
            bad = pred_cls[pi] != gt_cls
            row[bad] = -1
        row[used] = -1
        if row.size == 0:
            unmatched_p.append(pi)
            continue
        gi = int(row.argmax())
        if row[gi] < iou_thr:
            unmatched_p.append(pi)
            continue
        used[gi] = True
        matched_p.append(pi)
        matched_g.append(gi)
    return (
        np.asarray(matched_p, dtype=np.int64),
        np.asarray(matched_g, dtype=np.int64),
        np.asarray(unmatched_p, dtype=np.int64),
    )


# ---------------------------- object-level metrics --------------------------

def distance_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """MAE of distance in meters."""
    if len(pred) == 0:
        return float("nan")
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(gt))))


def contact_point_error(pred_uv: np.ndarray, gt_uv: np.ndarray) -> float:
    """Mean pixel Euclidean distance."""
    if len(pred_uv) == 0:
        return float("nan")
    d = np.asarray(pred_uv) - np.asarray(gt_uv)
    return float(np.mean(np.sqrt((d ** 2).sum(-1))))


# ---------------------------- stability (video) -----------------------------

def stability(tracks: Dict[int, List[float]]) -> float:
    """Mean per-track variance of predicted distance.

    tracks: dict {track_id: list of per-frame distance predictions}.
    Only tracks with >=2 frames contribute.
    """
    variances = []
    for _, dists in tracks.items():
        if len(dists) >= 2:
            variances.append(float(np.var(np.asarray(dists))))
    if not variances:
        return float("nan")
    return float(np.mean(variances))


# ---------------------------- mAP -------------------------------------------

def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """VOC 2010+ style AP: area under the precision-recall curve.

    Precision is interpolated as the max precision at any recall >= r
    (monotone envelope), then integrated via trapezoidal rule.
    """
    # append sentinel values
    r = np.concatenate([[0.0], recalls, [1.0]])
    p = np.concatenate([[1.0], precisions, [0.0]])
    # monotone envelope (right-to-left max)
    p = np.maximum.accumulate(p[::-1])[::-1]
    # area under curve
    idx = np.where(r[1:] != r[:-1])[0]
    return float(np.sum((r[idx + 1] - r[idx]) * p[idx + 1]))


class MAPEvaluator:
    """PASCAL VOC-style mAP@IoU evaluator.

    Also computes AP3D: TP requires IoU >= iou_thr AND
    max(dist_pred/dist_gt, dist_gt/dist_pred) < dist_ratio_thr (default 1.25).

    Usage::
        ev = MAPEvaluator(num_classes=3, iou_thr=0.5)
        for pred, gt in ...:
            ev.update(pred, gt, img_id)
        result = ev.compute()   # {"mAP": ..., "AP_per_class": [...],
                                #  "mAP3D": ..., "AP3D_per_class": [...]}
    """

    def __init__(self, num_classes: int = 3, iou_thr: float = 0.5,
                 dist_ratio_thr: float = 1.25):
        self.num_classes = num_classes
        self.iou_thr = iou_thr
        self.dist_ratio_thr = dist_ratio_thr
        self.reset()

    def reset(self):
        # per-class lists of (score, img_id, pred_box_xyxy, pred_dist)
        self._preds: list[list] = [[] for _ in range(self.num_classes)]
        self._gts: dict = {}       # (img_id, c) -> gt_boxes_xyxy (n,4)
        self._gt_dist: dict = {}   # (img_id, c) -> gt_distances  (n,)
        self._n_gt = np.zeros(self.num_classes, dtype=np.int64)

    def update(self, pred: dict, gt: dict, img_id):
        """
        pred: boxes (xywh), scores, classes, distance  (numpy)
        gt:   boxes (xywh), labels, dist               (numpy)
        img_id: any hashable identifier for this image.
        """
        g_xyxy   = xywh_to_xyxy(np.asarray(gt["boxes"]))
        g_labels = np.asarray(gt["labels"])
        g_dist   = np.asarray(gt.get("dist", np.zeros(len(g_labels))), dtype=np.float64)

        for c in range(self.num_classes):
            gt_mask = g_labels == c
            key = (img_id, c)
            self._gts[key]     = g_xyxy[gt_mask]
            self._gt_dist[key] = g_dist[gt_mask]
            self._n_gt[c] += int(gt_mask.sum())

        p_boxes  = xywh_to_xyxy(np.asarray(pred["boxes"]))
        p_scores = np.asarray(pred["scores"])
        p_cls    = np.asarray(pred["classes"])
        p_dist   = np.asarray(pred.get("distance", np.zeros(len(p_scores))), dtype=np.float64)

        for i in range(len(p_boxes)):
            c = int(p_cls[i])
            if 0 <= c < self.num_classes:
                self._preds[c].append((float(p_scores[i]), img_id, p_boxes[i], float(p_dist[i])))

    def _compute_ap_for_class(self, c: int, use_dist: bool) -> float:
        preds_c = self._preds[c]
        if self._n_gt[c] == 0:
            return float("nan")
        if len(preds_c) == 0:
            return 0.0

        preds_c = sorted(preds_c, key=lambda x: -x[0])
        matched: dict = {}
        tp = np.zeros(len(preds_c))
        fp = np.zeros(len(preds_c))

        for i, (score, img_id, pbox, p_dist_val) in enumerate(preds_c):
            key = (img_id, c)
            gt_boxes = self._gts.get(key, np.zeros((0, 4)))
            if len(gt_boxes) == 0:
                fp[i] = 1
                continue

            if key not in matched:
                matched[key] = np.zeros(len(gt_boxes), dtype=bool)

            ious = bbox_iou_xyxy(pbox[None], gt_boxes)[0]
            ious[matched[key]] = -1

            if use_dist:
                gt_dists = self._gt_dist.get(key, np.zeros(len(gt_boxes)))
                valid = gt_dists > 0
                ratio = np.where(
                    valid,
                    np.maximum(p_dist_val / np.where(gt_dists > 0, gt_dists, 1.0),
                               gt_dists / max(p_dist_val, 1e-6)),
                    np.inf,
                )
                ious[ratio >= self.dist_ratio_thr] = -1

            best = int(np.argmax(ious))
            if ious[best] >= self.iou_thr:
                tp[i] = 1
                matched[key][best] = True
            else:
                fp[i] = 1

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls    = cum_tp / (self._n_gt[c] + 1e-9)
        precisions = cum_tp / (cum_tp + cum_fp + 1e-9)
        return compute_ap(recalls, precisions)

    def compute(self) -> dict:
        aps    = [self._compute_ap_for_class(c, use_dist=False) for c in range(self.num_classes)]
        aps_3d = [self._compute_ap_for_class(c, use_dist=True)  for c in range(self.num_classes)]

        valid    = [a for a in aps    if not np.isnan(a)]
        valid_3d = [a for a in aps_3d if not np.isnan(a)]
        return {
            "mAP":           float(np.mean(valid))    if valid    else float("nan"),
            "AP_per_class":  aps,
            "mAP3D":         float(np.mean(valid_3d)) if valid_3d else float("nan"),
            "AP3D_per_class": aps_3d,
        }


# ---------------------------- aggregate over dataset -----------------------

class OGCDEEvaluator:
    """Collect predictions/GT across the dataset and compute all metrics."""

    def __init__(self, iou_thr=0.5, class_agnostic=False):
        self.iou_thr = iou_thr
        self.class_agnostic = class_agnostic
        self.reset()

    def reset(self):
        self.dist_pred = []
        self.dist_gt = []
        self.depth_pred = []
        self.depth_gt = []
        self.cp_pred = []
        self.cp_gt = []
        self.cls = []          # GT class label for each matched pair
        self.ale_errors = []
        self.ale_cls = []      # GT class label for each ALE entry

    # -------- one-image update ------------------------------------------
    def update(self, pred, gt, intrinsics=None):
        """
        pred: dict with numpy arrays: boxes (xywh), scores, classes,
              depth, distance, contact.
        gt:   dict with boxes (xywh), labels, dist, depth, contact.
              Optionally: loc3d (N,3) — 3D bottom-center in camera coords.
        intrinsics: dict with fx, fy, cx, cy (from P2). Required for ALE.
        """
        p_xyxy = xywh_to_xyxy(pred["boxes"])
        g_xyxy = xywh_to_xyxy(gt["boxes"])
        mp, mg, _ = greedy_match(
            p_xyxy, np.asarray(pred["scores"]), g_xyxy,
            iou_thr=self.iou_thr,
            pred_cls=None if self.class_agnostic else np.asarray(pred["classes"]),
            gt_cls=None if self.class_agnostic else np.asarray(gt["labels"]),
        )
        if len(mp) == 0:
            return
        gt_labels = np.asarray(gt["labels"])
        self.cls.extend(gt_labels[mg].tolist())
        self.dist_pred.extend(np.asarray(pred["distance"])[mp].tolist())
        self.dist_gt.extend(np.asarray(gt["dist"])[mg].tolist())
        self.depth_pred.extend(np.asarray(pred["depth"])[mp].tolist())
        self.depth_gt.extend(np.asarray(gt["depth"])[mg].tolist())
        self.cp_pred.extend(np.asarray(pred["contact"])[mp].tolist())
        self.cp_gt.extend(np.asarray(gt["contact"])[mg].tolist())

        if intrinsics is not None and "loc3d" in gt:
            fx, fy = intrinsics["fx"], intrinsics["fy"]
            cx, cy = intrinsics["cx"], intrinsics["cy"]
            cp_p   = np.asarray(pred["contact"])[mp]    # (K,2)
            dep_p  = np.asarray(pred["depth"])[mp]      # (K,)
            loc_g  = np.asarray(gt["loc3d"])[mg]        # (K,3)
            X_p = (cp_p[:, 0] - cx) * dep_p / fx
            Y_p = (cp_p[:, 1] - cy) * dep_p / fy
            err3d = np.sqrt((X_p - loc_g[:, 0])**2 +
                            (Y_p - loc_g[:, 1])**2 +
                            (dep_p - loc_g[:, 2])**2)
            self.ale_errors.extend(err3d.tolist())
            self.ale_cls.extend(gt_labels[mg].tolist())

    # -------- summary ---------------------------------------------------
    def compute(self, num_classes: int = 3):
        dist_pred  = np.array(self.dist_pred)
        dist_gt    = np.array(self.dist_gt)
        depth_pred = np.array(self.depth_pred)
        depth_gt   = np.array(self.depth_gt)
        cls_arr    = np.array(self.cls, dtype=np.int64)
        ale_arr    = np.array(self.ale_errors) if self.ale_errors else None
        ale_cls    = np.array(self.ale_cls, dtype=np.int64) if self.ale_cls else None

        depth_m = depth_metrics(depth_pred, depth_gt)
        dist_m  = depth_metrics(dist_pred,  dist_gt)
        de  = distance_error(dist_pred, dist_gt)
        cpe = contact_point_error(np.array(self.cp_pred), np.array(self.cp_gt))
        ale = float(np.mean(ale_arr)) if ale_arr is not None and len(ale_arr) else float("nan")

        per_class = []
        for c in range(num_classes):
            mask = cls_arr == c
            if ale_arr is not None and ale_cls is not None:
                amask = ale_cls == c
                ale_c = float(np.mean(ale_arr[amask])) if amask.sum() > 0 else float("nan")
            else:
                ale_c = float("nan")
            per_class.append({
                "distance": depth_metrics(dist_pred[mask],  dist_gt[mask]),
                "depth":    depth_metrics(depth_pred[mask], depth_gt[mask]),
                "DE":  distance_error(dist_pred[mask], dist_gt[mask]),
                "ALE": ale_c,
                "n":   int(mask.sum()),
            })

        return {
            "depth":     depth_m,
            "distance":  dist_m,
            "DE":        de,
            "CPE_px":    cpe,
            "ALE":       ale,
            "n_matched": len(self.dist_pred),
            "per_class": per_class,
        }
