from __future__ import annotations

from typing import Dict

import torch


class YOLOTargetAdapter:
    """Convert the canonical physical target to YOLO-style per-image targets."""

    def __call__(self, targets):
        if isinstance(targets, dict):
            targets = [targets]

        converted = []
        for batch_idx, target in enumerate(targets):
            if target["boxes"].numel() == 0:
                converted.append({
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.int64),
                    "batch_idx": torch.zeros((0,), dtype=torch.long),
                    "z_gt": torch.zeros((0,), dtype=torch.float32),
                    "distance_gt": torch.zeros((0,), dtype=torch.float32),
                })
                continue
            boxes = target["boxes"]
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            cxcywh = torch.stack([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], dim=-1)
            converted.append({
                "boxes": cxcywh,
                "labels": target["labels"],
                "batch_idx": torch.full_like(target["labels"], batch_idx, dtype=torch.long),
                "z_gt": target["z_gt"],
                "distance_gt": target["distance_gt"],
            })
        if len(converted) == 1:
            return converted[0]
        return {
            "boxes": torch.cat([entry["boxes"] for entry in converted], dim=0),
            "labels": torch.cat([entry["labels"] for entry in converted], dim=0),
            "batch_idx": torch.cat([entry["batch_idx"] for entry in converted], dim=0),
            "z_gt": torch.cat([entry["z_gt"] for entry in converted], dim=0),
            "distance_gt": torch.cat([entry["distance_gt"] for entry in converted], dim=0),
        }
