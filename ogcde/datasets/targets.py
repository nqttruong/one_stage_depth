from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch


def make_common_target(objects: Iterable[Dict[str, Any]], class_map: Dict[str, int]) -> Dict[str, Any]:
    """Build the canonical per-image physical target for a dataset sample."""
    boxes: List[np.ndarray] = []
    labels: List[int] = []
    z_gt: List[float] = []
    distance_gt: List[float] = []
    loc3d: List[np.ndarray] = []
    object_ids: List[int] = []

    for obj_id, obj in enumerate(objects):
        obj_type = obj.get("type")
        if obj_type not in class_map:
            continue
        loc = np.asarray(obj.get("loc", [0.0, 0.0, 0.0]), dtype=np.float32)
        x, y, z = float(loc[0]), float(loc[1]), float(loc[2])
        dist = float(np.sqrt(x * x + y * y + z * z))
        bbox = obj.get("bbox", [0.0, 0.0, 0.0, 0.0])
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        boxes.append(np.asarray([x1, y1, x2, y2], dtype=np.float32))
        labels.append(class_map[obj_type])
        z_gt.append(z)
        distance_gt.append(dist)
        loc3d.append(loc)
        object_ids.append(obj_id)

    if not boxes:
        return {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "labels": np.zeros((0,), dtype=np.int64),
            "z_gt": np.zeros((0,), dtype=np.float32),
            "distance_gt": np.zeros((0,), dtype=np.float32),
            "loc3d": np.zeros((0, 3), dtype=np.float32),
            "object_ids": np.zeros((0,), dtype=np.int64),
        }

    return {
        "boxes": np.stack(boxes, axis=0),
        "labels": np.asarray(labels, dtype=np.int64),
        "z_gt": np.asarray(z_gt, dtype=np.float32),
        "distance_gt": np.asarray(distance_gt, dtype=np.float32),
        "loc3d": np.stack(loc3d, axis=0),
        "object_ids": np.asarray(object_ids, dtype=np.int64),
    }


def ensure_common_target(target: Dict[str, Any]) -> Dict[str, Any]:
    if "boxes" not in target:
        target["boxes"] = np.zeros((0, 4), dtype=np.float32)
    if "labels" not in target:
        target["labels"] = np.zeros((0,), dtype=np.int64)
    if "z_gt" not in target:
        target["z_gt"] = np.zeros((0,), dtype=np.float32)
    if "distance_gt" not in target:
        target["distance_gt"] = np.zeros((0,), dtype=np.float32)
    if "loc3d" not in target:
        target["loc3d"] = np.zeros((0, 3), dtype=np.float32)
    if "object_ids" not in target:
        target["object_ids"] = np.zeros((0,), dtype=np.int64)
    return target
