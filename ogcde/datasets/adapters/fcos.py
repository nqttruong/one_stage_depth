from __future__ import annotations

from typing import Any, Dict, List

import torch


class FCOSTargetAdapter:
    """Convert the canonical physical target to an FCOS-style list of per-image dicts."""

    def __call__(self, target: Dict[str, torch.Tensor]):
        return [{
            "boxes": target["boxes"],
            "labels": target["labels"],
            "z_gt": target["z_gt"],
            "distance_gt": target["distance_gt"],
        }]
