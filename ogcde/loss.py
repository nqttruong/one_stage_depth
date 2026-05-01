"""Multi-task loss for OGCDE one-stage model.

Total loss:
    L = w_box * L_box + w_obj * L_obj + w_cls * L_cls
      + l1 * L_depth + l2 * L_scale + l3 * L_contact + l4 * L_geo

Numerical choices (different from the raw paper spec, but more stable):
    - Depth loss in log-space (as specified).
    - Scale loss in log-space (spec has linear; log is more stable).
    - Geometry loss: linear L1 on distance (as specified). Because we decode
      distance = exp(s_raw + d_raw), the loss pulls the sum toward the true
      log-distance while still measuring error in meters.

Assignment (anchor-free, center-radius):
    A spatial cell at stride `r` is a positive sample for a GT box if:
      (a) the GT center falls inside the cell, OR
      (b) the cell center lies within the GT bbox AND within `center_radius`
          cells of the GT center.
    Additionally we use scale-aware filtering: a GT is only assigned to the
    scale whose stride matches its size (using ATSS-style size thresholds).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import split_pred, make_grid, decode_bbox, decode_contact, decode_depth_scale


# --------------------------- geometric helpers ------------------------------

def ciou_loss(pred_xywh: torch.Tensor, gt_xywh: torch.Tensor) -> torch.Tensor:
    """CIoU loss for matched pairs. Input shape (N, 4) in xywh pixels.

    Computed in fp32 regardless of autocast: squared pixel distances (up to
    640^2 = 409600) and iou→1 edge cases make fp16 gradients unstable.
    """
    eps = 1e-7
    pred_xywh = pred_xywh.float()
    gt_xywh   = gt_xywh.float()

    p_xy, p_wh = pred_xywh[..., :2], pred_xywh[..., 2:4]
    g_xy, g_wh = gt_xywh[..., :2], gt_xywh[..., 2:4]

    p_min = p_xy - p_wh / 2
    p_max = p_xy + p_wh / 2
    g_min = g_xy - g_wh / 2
    g_max = g_xy + g_wh / 2

    inter = (torch.min(p_max, g_max) - torch.max(p_min, g_min)).clamp(min=0)
    inter_area = inter[..., 0] * inter[..., 1]
    p_area = p_wh[..., 0] * p_wh[..., 1]
    g_area = g_wh[..., 0] * g_wh[..., 1]
    union = p_area + g_area - inter_area + eps
    iou = inter_area / union

    enc = torch.max(p_max, g_max) - torch.min(p_min, g_min)
    c2 = enc.pow(2).sum(-1) + eps
    rho2 = ((p_xy - g_xy) ** 2).sum(-1)

    v = (4 / math.pi ** 2) * (
        torch.atan(g_wh[..., 0] / (g_wh[..., 1] + eps)) -
        torch.atan(p_wh[..., 0] / (p_wh[..., 1] + eps))
    ).pow(2)
    alpha = v / (1 - iou + v + eps)

    ciou = iou - rho2 / c2 - alpha * v
    return 1 - ciou


# ------------------------------ assigner ------------------------------------

@torch.no_grad()
def assign_targets(gt_boxes, strides, feat_shapes, device,
                   size_ranges=((0, 64), (64, 128), (128, 1e6)),
                   center_radius=1.5):
    """Assign each GT to positive cells across the feature pyramid.

    Args:
        gt_boxes: (M, 4) in xywh absolute pixels for a single image.
        strides:  tuple of strides per scale.
        feat_shapes: list of (H, W) per scale.
        size_ranges: per-scale GT-size (max(w,h)) range that is assigned here.
        center_radius: how many stride units around GT center to also mark
                       positive (only cells whose center is inside both the
                       GT box and the radius ball count).

    Returns:
        List of (pos_idx, gt_idx) tensors per scale, where pos_idx is the
        flat HW-index of the positive cell and gt_idx is which GT it matched.
    """
    assignments = []
    for stride, (H, W), (s_lo, s_hi) in zip(strides, feat_shapes, size_ranges):
        if gt_boxes.numel() == 0:
            assignments.append((torch.zeros(0, dtype=torch.long, device=device),
                                torch.zeros(0, dtype=torch.long, device=device)))
            continue

        # scale filter
        sizes = gt_boxes[:, 2:].max(-1).values
        keep_gt = (sizes >= s_lo) & (sizes < s_hi)

        pos_list, gt_list = [], []

        grid = make_grid(H, W, device)            # (HW, 2) in cell units
        cell_center_px = (grid + 0.5) * stride    # pixel coords of cell centers

        for i in range(gt_boxes.size(0)):
            if not keep_gt[i]:
                continue
            gcx, gcy, gw, gh = gt_boxes[i]
            # inside-box check
            inside = (
                (cell_center_px[:, 0] > gcx - gw / 2) &
                (cell_center_px[:, 0] < gcx + gw / 2) &
                (cell_center_px[:, 1] > gcy - gh / 2) &
                (cell_center_px[:, 1] < gcy + gh / 2)
            )
            # center-radius check
            dx = (cell_center_px[:, 0] - gcx).abs() / stride
            dy = (cell_center_px[:, 1] - gcy).abs() / stride
            near = (dx < center_radius) & (dy < center_radius)

            mask = inside & near
            if mask.sum() == 0:
                # fall back to nearest cell
                dist_sq = (cell_center_px - torch.stack([gcx, gcy])).pow(2).sum(-1)
                idx = dist_sq.argmin(keepdim=True)
            else:
                idx = mask.nonzero(as_tuple=False).squeeze(1)

            pos_list.append(idx)
            gt_list.append(torch.full_like(idx, i))

        if pos_list:
            pos_idx = torch.cat(pos_list, 0)
            gt_idx = torch.cat(gt_list, 0)
            # dedupe: if a cell matches multiple GTs, keep the smallest-area GT
            # (classic center-sampling tie-break).
            if len(pos_idx) > 0:
                areas = gt_boxes[:, 2] * gt_boxes[:, 3]
                order = areas[gt_idx].argsort()
                pos_idx, gt_idx = pos_idx[order], gt_idx[order]
                # Dedup: if a cell matches multiple GTs, keep the first
                # (smallest-area GT after the sort above).
                seen = set()
                keep = []
                for k in range(len(pos_idx)):
                    p = pos_idx[k].item()
                    if p not in seen:
                        seen.add(p)
                        keep.append(k)
                keep = torch.tensor(keep, device=device, dtype=torch.long)
                pos_idx, gt_idx = pos_idx[keep], gt_idx[keep]
        else:
            pos_idx = torch.zeros(0, dtype=torch.long, device=device)
            gt_idx = torch.zeros(0, dtype=torch.long, device=device)
        assignments.append((pos_idx, gt_idx))

    return assignments


# ------------------------------- loss ---------------------------------------

class OGCDELoss(nn.Module):
    """Multi-task loss.

    Args:
        nc: number of classes.
        strides: per-scale stride.
        lambdas: (l_depth, l_scale, l_contact, l_geo).
        det_weights: (w_box, w_obj, w_cls).
        has_depth_gt: if False, L_depth and L_scale are skipped; L_geo alone
                      supervises the geometry branch (useful when you don't
                      have reliable per-object depth GT).
    """

    def __init__(self, nc=3, strides=(8, 16, 32),
                 lambdas=(1.0, 0.5, 1.0, 2.0),
                 det_weights=(7.5, 1.0, 0.5),
                 has_depth_gt=True,
                 focal_gamma: float = 1.5,
                 cls_weights=None):
        super().__init__()
        self.nc = nc
        self.strides = strides
        # Mutable so train loop can warmup λ_geo etc.
        self.lambdas = list(lambdas)
        self.w_box, self.w_obj, self.w_cls = det_weights
        self.has_depth_gt = has_depth_gt
        self.focal_gamma = focal_gamma  # 0 = plain BCE; >0 = focal
        # Per-class weight for cls loss: up-weight rare classes (e.g. Cyclist)
        if cls_weights is not None:
            w = torch.tensor(cls_weights, dtype=torch.float32)
            self.register_buffer("cls_weights", w / w.mean())  # normalise so total scale is preserved
        else:
            self.register_buffer("cls_weights", torch.ones(nc))

    @staticmethod
    def _focal_bce(pred: torch.Tensor, target: torch.Tensor,
                   gamma: float) -> torch.Tensor:
        """Focal BCE loss (element-wise).

        Down-weights easy negatives so the objectness head is forced to
        discriminate harder examples.  gamma=0 recovers plain BCE.
        """
        ce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        if gamma == 0.0:
            return ce
        p_t = torch.sigmoid(pred) * target + (1 - torch.sigmoid(pred)) * (1 - target)
        return ((1 - p_t) ** gamma) * ce

    def set_lambda_geo(self, value: float):
        """Update geometry-loss weight at runtime (used by warmup)."""
        self.lambdas[3] = float(value)

    # -----------------------------------------------------------------------
    def forward(self, preds, targets):
        """
        preds: list of (B, no, H, W) per scale.
        targets: dict with keys 'boxes', 'labels', 'dist', 'depth', 'contact',
                 'batch_idx'. All flat across the batch.
        """
        # Cast predictions to fp32 for loss computation.  The model forward
        # runs in fp16 (AMP), but fp16 backward through CIoU / focal-BCE /
        # contact-sqrt produces NaN gradients due to precision limits at
        # pixel scale (640^2=409600 >> fp16 max 65504) and near-zero denominators.
        # Casting here is zero-copy when preds are already fp32.
        preds = [p.float() for p in preds]

        device = preds[0].device
        B = preds[0].shape[0]
        feat_shapes = [p.shape[-2:] for p in preds]

        # Accumulators
        box_loss = torch.zeros((), device=device)
        obj_loss = torch.zeros((), device=device)
        cls_loss = torch.zeros((), device=device)
        depth_loss = torch.zeros((), device=device)
        scale_loss = torch.zeros((), device=device)
        contact_loss = torch.zeros((), device=device)
        geo_loss = torch.zeros((), device=device)

        n_pos_total = 0

        # Unpack per-scale predictions once
        decoded = []  # list of dict per scale
        for p in preds:
            d = split_pred(p, self.nc)
            decoded.append(d)

        # Per-image assignment (center-sampling is easiest done per image)
        for b in range(B):
            mask = targets["batch_idx"] == b
            if not mask.any():
                # negative-only image: still contribute obj loss
                for s_i, dec in enumerate(decoded):
                    obj_t = torch.zeros_like(dec["obj"][b])
                    obj_loss = obj_loss + self._focal_bce(dec["obj"][b], obj_t, self.focal_gamma).mean()
                continue

            gt_boxes = targets["boxes"][mask].to(device)
            gt_labels = targets["labels"][mask].to(device)
            gt_dist = targets["dist"][mask].to(device)
            gt_depth = targets["depth"][mask].to(device) if self.has_depth_gt else None
            gt_contact = targets["contact"][mask].to(device)

            assignments = assign_targets(
                gt_boxes, self.strides, feat_shapes, device,
            )

            for s_i, (dec, stride, (H, W), (pos_idx, gt_idx)) in enumerate(zip(
                decoded, self.strides, feat_shapes, assignments
            )):
                # --- objectness target (all cells of this scale) ---
                obj_t = torch.zeros(H * W, device=device)
                if len(pos_idx) > 0:
                    obj_t[pos_idx] = 1.0
                obj_loss = obj_loss + self._focal_bce(dec["obj"][b], obj_t, self.focal_gamma).mean()

                if len(pos_idx) == 0:
                    continue

                grid = make_grid(H, W, device)
                # decode predicted box at positive cells
                bbox_raw_pos = dec["bbox_raw"][b, pos_idx]
                grid_pos = grid[pos_idx]
                pred_xywh = decode_bbox(bbox_raw_pos, grid_pos, stride)
                gt_xywh = gt_boxes[gt_idx]

                # --- CIoU box loss ---
                box_loss = box_loss + ciou_loss(pred_xywh, gt_xywh).mean()

                # --- classification loss ---
                cls_t = F.one_hot(gt_labels[gt_idx].long(), self.nc).float()
                per_cell = self._focal_bce(dec["cls"][b, pos_idx], cls_t, 0.5).mean(-1)  # (N,)
                sample_w = self.cls_weights[gt_labels[gt_idx].long()]
                cls_loss = cls_loss + (per_cell * sample_w).mean()

                # --- geometry losses ---
                d_raw = dec["d_raw"][b, pos_idx]
                s_raw = dec["s_raw"][b, pos_idx]
                dxdy_raw = dec["dxdy"][b, pos_idx]

                d_pred, s_pred, dist_pred = decode_depth_scale(d_raw, s_raw)

                # depth: |log(d_pred) - log(d_gt)|
                if self.has_depth_gt and gt_depth is not None:
                    depth_loss = depth_loss + (
                        d_raw - gt_depth[gt_idx].clamp(min=1e-3).log()
                    ).abs().mean()

                    # scale GT: s_gt = dist / depth
                    s_gt = (gt_dist[gt_idx] / gt_depth[gt_idx].clamp(min=1e-3)).clamp(min=1e-3)
                    scale_loss = scale_loss + (s_raw - s_gt.log()).abs().mean()

                # contact point — fp32 for sqrt gradient stability
                pred_cp = decode_contact(dxdy_raw, pred_xywh.detach(), stride)
                contact_loss = contact_loss + (
                    (pred_cp.float() - gt_contact[gt_idx].float())
                    .pow(2).sum(-1).clamp(min=1e-4).sqrt()
                ).mean()

                # geometry consistency: |distance_pred - distance_gt|
                geo_loss = geo_loss + (dist_pred - gt_dist[gt_idx]).abs().mean()

                n_pos_total += len(pos_idx)

        # normalize
        nscales = len(preds)
        obj_loss = obj_loss / (B * nscales)

        # avoid divide-by-zero when a batch has no positives
        norm = max(1, B)
        box_loss = box_loss / norm
        cls_loss = cls_loss / norm
        depth_loss = depth_loss / norm
        scale_loss = scale_loss / norm
        contact_loss = contact_loss / norm
        geo_loss = geo_loss / norm

        l1, l2, l3, l4 = self.lambdas
        total = (
            self.w_box * box_loss +
            self.w_obj * obj_loss +
            self.w_cls * cls_loss +
            l1 * depth_loss +
            l2 * scale_loss +
            l3 * contact_loss +
            l4 * geo_loss
        )

        return total, {
            "total": float(total.detach()),
            "box": float(box_loss.detach()),
            "obj": float(obj_loss.detach()),
            "cls": float(cls_loss.detach()),
            "depth": float(depth_loss.detach()),
            "scale": float(scale_loss.detach()),
            "contact": float(contact_loss.detach()),
            "geo": float(geo_loss.detach()),
            "n_pos": n_pos_total,
        }
