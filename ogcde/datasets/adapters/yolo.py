from __future__ import annotations

from typing import Dict

import torch


class YOLOTargetAdapter:
    """Convert the canonical physical target to YOLO-style per-image targets."""

    def __call__(self, target: Dict[str, torch.Tensor]):
        if target["boxes"].numel() == 0:
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "batch_idx": torch.zeros((0,), dtype=torch.long),
                "z_gt": torch.zeros((0,), dtype=torch.float32),
                "distance_gt": torch.zeros((0,), dtype=torch.float32),
            }
        boxes = target["boxes"]
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        cxcywh = torch.stack([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], dim=-1)
        return {
            "boxes": cxcywh,
            "labels": target["labels"],
            "batch_idx": torch.zeros_like(target["labels"], dtype=torch.long),
            "z_gt": target["z_gt"],
            "distance_gt": target["distance_gt"],
        }
