"""Oracle-mode evaluation for OGCDE.

Instead of IoU-based matching, query the model's raw grid prediction at the
center cell of each GT bounding box.  This mirrors the setup used in papers
like DistFormer where GT boxes are given as input, making distance-estimation
metrics independent of detection quality.

Usage in evaluate_kitti.py via --mode oracle or --mode both.
"""

from __future__ import annotations

import numpy as np
import torch

from ogcde.model import decode_bbox, decode_contact, decode_depth_scale
from ogcde.metrics import depth_metrics, distance_error, contact_point_error


def oracle_query(
    preds: list,
    batch_idx: int,
    gt_boxes_xywh_orig: np.ndarray,
    meta: dict,
    nc: int,
    strides: tuple = (8, 16, 32),
) -> dict:
    """Query model prediction at the center cell of each GT bounding box.

    For each GT object, checks all three scales and picks the cell with the
    highest objectness score.

    Args:
        preds: raw model outputs — list of 3 tensors (B, 9+nc, H, W)
        batch_idx: image index within the batch
        gt_boxes_xywh_orig: (N, 4) GT boxes in original image coords (cx,cy,w,h)
        meta: dict with keys ratio, pad, fx, fy, cx, cy
        nc: number of classes
        strides: feature-map strides, must match len(preds)

    Returns dict (N entries, aligned to gt_boxes_xywh_orig):
        distance : (N,)   predicted Euclidean distance
        depth    : (N,)   predicted Z-depth
        contact  : (N,2)  predicted contact point in original image coords
        obj      : (N,)   objectness score at the queried cell
    """
    ratio = meta["ratio"]
    pad = meta["pad"]   # (pad_x, pad_y)

    N = len(gt_boxes_xywh_orig)
    if N == 0:
        return {
            "distance": np.zeros(0),
            "depth":    np.zeros(0),
            "contact":  np.zeros((0, 2)),
            "obj":      np.zeros(0),
        }

    out_dist    = np.full(N, float("nan"))
    out_depth   = np.full(N, float("nan"))
    out_contact = np.full((N, 2), float("nan"))
    out_obj     = np.full(N, -1.0)

    for i, box in enumerate(gt_boxes_xywh_orig):
        cx_orig, cy_orig = float(box[0]), float(box[1])

        # convert GT center to letterbox coords
        cx_lb = cx_orig * ratio + pad[0]
        cy_lb = cy_orig * ratio + pad[1]

        for stride, pred_scale in zip(strides, preds):
            p = pred_scale[batch_idx]          # (C, H, W)
            H, W = p.shape[1], p.shape[2]

            gx = min(int(cx_lb / stride), W - 1)
            gy = min(int(cy_lb / stride), H - 1)

            cell = p[:, gy, gx]               # (C,)
            obj_score = float(torch.sigmoid(cell[4]))

            if obj_score <= out_obj[i]:
                continue

            out_obj[i] = obj_score

            d_raw  = cell[5 + nc]
            s_raw  = cell[6 + nc]
            dxdy   = cell[7 + nc: 9 + nc]

            depth, _, dist = decode_depth_scale(
                d_raw.unsqueeze(0), s_raw.unsqueeze(0)
            )
            out_depth[i] = float(depth)
            out_dist[i]  = float(dist)

            # decode contact via predicted bbox at this cell
            grid     = torch.tensor([[gx, gy]], dtype=torch.float32, device=p.device)
            bbox_lb  = decode_bbox(cell[:4].unsqueeze(0), grid, stride)[0]   # (4,) xywh
            contact_lb = decode_contact(dxdy.unsqueeze(0), bbox_lb.unsqueeze(0), stride)[0]

            out_contact[i, 0] = (float(contact_lb[0]) - pad[0]) / ratio
            out_contact[i, 1] = (float(contact_lb[1]) - pad[1]) / ratio

    return {
        "distance": out_dist,
        "depth":    out_depth,
        "contact":  out_contact,
        "obj":      out_obj,
    }


class OracleEvaluator:
    """Accumulate oracle-mode metrics (no IoU matching, 1-to-1 GT alignment).

    Per-class breakdown is available because GT labels are known for every
    queried object.
    """

    def __init__(self, num_classes: int = 3):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.dist_pred  = []
        self.dist_gt    = []
        self.depth_pred = []
        self.depth_gt   = []
        self.cp_pred    = []
        self.cp_gt      = []
        self.cls        = []
        self.ale_errors = []
        self.ale_cls    = []

    def update(self, oracle_pred: dict, gt: dict, intrinsics: dict | None = None):
        """
        oracle_pred: output of oracle_query() — arrays aligned to GT objects
        gt: dict with labels, dist, depth, contact, optionally loc3d
        """
        labels = np.asarray(gt["labels"])
        N = len(labels)
        if N == 0:
            return

        self.cls.extend(labels.tolist())
        self.dist_pred.extend(oracle_pred["distance"].tolist())
        self.dist_gt.extend(np.asarray(gt["dist"]).tolist())
        self.depth_pred.extend(oracle_pred["depth"].tolist())
        self.depth_gt.extend(np.asarray(gt["depth"]).tolist())
        self.cp_pred.extend(oracle_pred["contact"].tolist())
        self.cp_gt.extend(np.asarray(gt["contact"]).tolist())

        if intrinsics is not None and "loc3d" in gt:
            fx, fy = intrinsics["fx"], intrinsics["fy"]
            cx, cy = intrinsics["cx"], intrinsics["cy"]
            cp_p  = oracle_pred["contact"]          # (N,2)
            dep_p = oracle_pred["depth"]            # (N,)
            loc_g = np.asarray(gt["loc3d"])         # (N,3)
            X_p   = (cp_p[:, 0] - cx) * dep_p / fx
            Y_p   = (cp_p[:, 1] - cy) * dep_p / fy
            err3d = np.sqrt((X_p - loc_g[:, 0])**2 +
                            (Y_p - loc_g[:, 1])**2 +
                            (dep_p - loc_g[:, 2])**2)
            self.ale_errors.extend(err3d.tolist())
            self.ale_cls.extend(labels.tolist())

    def compute(self):
        dist_pred  = np.array(self.dist_pred)
        dist_gt    = np.array(self.dist_gt)
        depth_pred = np.array(self.depth_pred)
        depth_gt   = np.array(self.depth_gt)
        cls_arr    = np.array(self.cls, dtype=np.int64)
        ale_arr    = np.array(self.ale_errors) if self.ale_errors else None
        ale_cls    = np.array(self.ale_cls, dtype=np.int64) if self.ale_cls else None

        dist_m  = depth_metrics(dist_pred,  dist_gt)
        depth_m = depth_metrics(depth_pred, depth_gt)
        de      = distance_error(dist_pred, dist_gt)
        cpe     = contact_point_error(np.array(self.cp_pred), np.array(self.cp_gt))
        ale     = float(np.mean(ale_arr)) if ale_arr is not None and len(ale_arr) else float("nan")

        per_class = []
        for c in range(self.num_classes):
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
            "distance":  dist_m,
            "depth":     depth_m,
            "DE":        de,
            "CPE_px":    cpe,
            "ALE":       ale,
            "n_objects": len(self.dist_pred),
            "per_class": per_class,
        }
