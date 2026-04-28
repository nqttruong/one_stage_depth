"""Shared utilities: letterbox, NMS, inference decoding."""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torchvision

from .model import make_grid, decode_bbox, decode_contact, decode_depth_scale, split_pred


# ------------------------------ letterbox -----------------------------------

def letterbox(img, new_size, pad_value=114):
    """Resize (aspect-preserving) and pad to a square new_size.

    Returns padded image, scale ratio `r`, and (pad_x, pad_y) to map GT.
    """
    h, w = img.shape[:2]
    r = min(new_size / h, new_size / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = new_size - new_unpad[0]
    dh = new_size - new_unpad[1]
    left, top = dw // 2, dh // 2
    right, bottom = dw - left, dh - top
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(pad_value, pad_value, pad_value),
    )
    return img, r, (left, top)


# ------------------------------ NMS -----------------------------------------

def xywh_to_xyxy(b):
    x, y, w, h = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], -1)


def non_max_suppression(boxes_xywh, scores, cls_ids, iou_thr=0.5):
    """Per-class NMS. Inputs are torch tensors.

    Returns indices of kept detections.
    """
    if boxes_xywh.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes_xywh.device)
    xyxy = xywh_to_xyxy(boxes_xywh)
    # torchvision batched_nms handles per-class
    return torchvision.ops.batched_nms(xyxy, scores, cls_ids, iou_thr)


# ------------------------------ inference decode ----------------------------

@torch.no_grad()
def decode_predictions(preds, nc, strides=(8, 16, 32),
                       obj_thr=0.25, cls_thr=0.25, iou_thr=0.5,
                       max_det=300):
    """Convert raw multi-scale predictions to per-image detection lists.

    Returns list (len = batch size) of dicts:
        {
          'boxes'   : (N, 4)  xywh pixel coords,
          'scores'  : (N,),
          'classes' : (N,),
          'depth'   : (N,),
          'scale'   : (N,),
          'distance': (N,),
          'contact' : (N, 2) pixel coords,
        }
    """
    device = preds[0].device
    B = preds[0].shape[0]
    all_results = [[] for _ in range(B)]

    for p, stride in zip(preds, strides):
        dec = split_pred(p, nc)
        H, W = p.shape[-2:]
        grid = make_grid(H, W, device)

        obj = dec["obj"].sigmoid()
        cls = dec["cls"].sigmoid()
        # best class per cell
        cls_score, cls_id = cls.max(-1)
        score = obj * cls_score

        bbox = decode_bbox(dec["bbox_raw"], grid, stride)  # (B, HW, 4)
        d_pred, s_pred, dist_pred = decode_depth_scale(dec["d_raw"], dec["s_raw"])
        cp = decode_contact(dec["dxdy"], bbox, stride)

        mask = score > obj_thr
        for b in range(B):
            m = mask[b]
            if m.sum() == 0:
                continue
            all_results[b].append({
                "boxes": bbox[b][m],
                "scores": score[b][m],
                "classes": cls_id[b][m],
                "depth": d_pred[b][m],
                "scale": s_pred[b][m],
                "distance": dist_pred[b][m],
                "contact": cp[b][m],
            })

    final = []
    for b in range(B):
        if not all_results[b]:
            final.append({
                "boxes": torch.zeros(0, 4, device=device),
                "scores": torch.zeros(0, device=device),
                "classes": torch.zeros(0, dtype=torch.long, device=device),
                "depth": torch.zeros(0, device=device),
                "scale": torch.zeros(0, device=device),
                "distance": torch.zeros(0, device=device),
                "contact": torch.zeros(0, 2, device=device),
            })
            continue
        merged = {k: torch.cat([r[k] for r in all_results[b]], 0)
                  for k in all_results[b][0]}
        keep = non_max_suppression(
            merged["boxes"], merged["scores"], merged["classes"], iou_thr,
        )
        if len(keep) > max_det:
            keep = keep[:max_det]
        final.append({k: v[keep] for k, v in merged.items()})

    return final


# ------------------------------ coord mapping -------------------------------

def unletterbox_boxes(boxes_xywh, ratio, pad):
    """Map xywh boxes from letterboxed space back to original image."""
    boxes = boxes_xywh.clone() if isinstance(boxes_xywh, torch.Tensor) else np.copy(boxes_xywh)
    px, py = pad
    boxes[..., 0] = (boxes[..., 0] - px) / ratio
    boxes[..., 1] = (boxes[..., 1] - py) / ratio
    boxes[..., 2] = boxes[..., 2] / ratio
    boxes[..., 3] = boxes[..., 3] / ratio
    return boxes


def unletterbox_points(pts_uv, ratio, pad):
    pts = pts_uv.clone() if isinstance(pts_uv, torch.Tensor) else np.copy(pts_uv)
    px, py = pad
    pts[..., 0] = (pts[..., 0] - px) / ratio
    pts[..., 1] = (pts[..., 1] - py) / ratio
    return pts
