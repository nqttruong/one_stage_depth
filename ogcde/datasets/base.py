from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class BaseDistanceDataset(Dataset):
    """Common physical-target dataset contract.

    Each sample must return (image, target, meta), where target contains only
    physical supervision and never detector-specific tensors.
    """

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]:
        raise NotImplementedError

    @staticmethod
    def validate_target(target: Dict[str, Any]):
        required = {"boxes", "labels", "z_gt", "distance_gt", "loc3d", "object_ids"}
        missing = required - set(target)
        if missing:
            raise KeyError(f"Missing required physical target fields: {sorted(missing)}")
        if target["boxes"].ndim != 2 or target["boxes"].shape[-1] != 4:
            raise ValueError("Common target boxes must be (N, 4) in xyxy format.")
        for key in ("labels", "z_gt", "distance_gt", "object_ids"):
            if target[key].ndim == 0:
                target[key] = target[key].reshape(1)
        return target
