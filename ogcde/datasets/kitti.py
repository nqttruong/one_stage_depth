from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from PIL import Image

from ogcde.geometry.camera import CameraIntrinsics

from .base import BaseDistanceDataset
from .targets import ensure_common_target


def parse_kitti_calib(path: str | os.PathLike[str]) -> Dict[str, np.ndarray]:
    calib: Dict[str, List[float]] = {}
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            calib[key.strip()] = [float(v) for v in val.split()]
    p2 = np.asarray(calib["P2"], dtype=np.float64).reshape(3, 4)
    return {"P2": p2}


def parse_kitti_label(path: str | os.PathLike[str]) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    objs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fp:
        for raw in fp:
            parts = raw.strip().split()
            if len(parts) < 15:
                continue
            objs.append({
                "type": parts[0],
                "truncated": float(parts[1]),
                "occluded": int(parts[2]),
                "alpha": float(parts[3]),
                "bbox": [float(v) for v in parts[4:8]],
                "dims": [float(v) for v in parts[8:11]],
                "loc": [float(v) for v in parts[11:14]],
                "rot_y": float(parts[14]),
            })
    return objs


class KITTIDistanceDataset(BaseDistanceDataset):
    """KITTI dataset returning the common physical target contract.

    target keys are always:
      boxes, labels, z_gt, distance_gt, loc3d, object_ids
    with `xyxy` boxes and `annotation_bottom_center` GT semantics.
    """

    CLASS_MAP = {
        "Car": 0,
        "Van": 0,
        "Truck": 0,
        "Pedestrian": 1,
        "Person_sitting": 1,
        "Cyclist": 2,
    }

    def __init__(
        self,
        root: str,
        split_file: str,
        img_size: int = 640,
        augment: bool = False,
        hflip: bool = False,
        filter_difficult: bool = True,
        gt_source: str = "annotation_bottom_center",
        depth_source: Optional[str] = None,
        distance_source: Optional[str] = None,
    ):
        self.root = Path(root)
        self.image_dir = self.root / "image_2"
        self.label_dir = self.root / "label_2"
        self.calib_dir = self.root / "calib"
        self.img_size = int(img_size)
        self.augment = bool(augment)
        self.hflip = bool(hflip)
        self.filter_difficult = bool(filter_difficult)
        self.gt_source = gt_source
        if gt_source != "annotation_bottom_center":
            raise ValueError("Primary KITTI configuration must use annotation_bottom_center.")
        if depth_source is not None or distance_source is not None:
            raise ValueError("Primary experiment path forbids LiDAR override sources; use legacy path separately.")
        with open(split_file, "r", encoding="utf-8") as fp:
            self.ids = [line.strip() for line in fp if line.strip()]

    def __len__(self) -> int:
        return len(self.ids)

    def _build_target(self, objects: Iterable[Dict[str, Any]], calib: Dict[str, np.ndarray]):
        boxes: List[List[float]] = []
        labels: List[int] = []
        z_gt: List[float] = []
        distance_gt: List[float] = []
        loc3d: List[List[float]] = []
        object_ids: List[int] = []

        for obj_idx, obj in enumerate(objects):
            obj_type = obj.get("type", "")
            if obj_type not in self.CLASS_MAP:
                continue
            if self.filter_difficult and (obj.get("truncated", 0.0) > 0.5 or obj.get("occluded", 0) > 2):
                continue
            bbox = obj.get("bbox", [0.0, 0.0, 0.0, 0.0])
            x1, y1, x2, y2 = [float(v) for v in bbox]
            loc = np.asarray(obj.get("loc", [0.0, 0.0, 0.0]), dtype=np.float32)
            if loc.shape[0] < 3:
                continue
            x, y, z = float(loc[0]), float(loc[1]), float(loc[2])
            dist = float(math.sqrt(x * x + y * y + z * z))
            boxes.append([x1, y1, x2, y2])
            labels.append(int(self.CLASS_MAP[obj_type]))
            z_gt.append(z)
            distance_gt.append(dist)
            loc3d.append([x, y, z])
            object_ids.append(obj_idx)

        target = {
            "boxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4) if boxes else np.zeros((0, 4), dtype=np.float32),
            "labels": np.asarray(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64),
            "z_gt": np.asarray(z_gt, dtype=np.float32) if z_gt else np.zeros((0,), dtype=np.float32),
            "distance_gt": np.asarray(distance_gt, dtype=np.float32) if distance_gt else np.zeros((0,), dtype=np.float32),
            "loc3d": np.asarray(loc3d, dtype=np.float32).reshape(-1, 3) if loc3d else np.zeros((0, 3), dtype=np.float32),
            "object_ids": np.asarray(object_ids, dtype=np.int64) if object_ids else np.zeros((0,), dtype=np.int64),
        }
        return ensure_common_target(target)

    def __getitem__(self, index: int):
        image_id = self.ids[index]
        image_path = self.image_dir / f"{image_id}.png"
        if not image_path.exists():
            image_path = self.image_dir / f"{image_id}.jpg"
        label_path = self.label_dir / f"{image_id}.txt"
        calib_path = self.calib_dir / f"{image_id}.txt"

        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        objects = parse_kitti_label(label_path)
        calib = parse_kitti_calib(calib_path)
        p2 = calib["P2"]
        intrinsics_original = CameraIntrinsics(
            fx=float(p2[0, 0]),
            fy=float(p2[1, 1]),
            cx=float(p2[0, 2]),
            cy=float(p2[1, 2]),
        )
        target = self._build_target(objects, calib)
        boxes = target["boxes"].copy()
        intrinsics_network = intrinsics_original
        if self.augment and self.hflip and np.random.rand() < 0.5:
            width = image.shape[1]
            image = image[:, ::-1, :].copy()
            if boxes.size:
                x1 = boxes[:, 0].copy()
                x2 = boxes[:, 2].copy()
                boxes[:, 0] = width - x2
                boxes[:, 2] = width - x1
            intrinsics_network = intrinsics_network.hflip(float(width))
        target["boxes"] = boxes

        h, w = image.shape[:2]
        if self.img_size is not None and self.img_size > 0:
            scale = min(self.img_size / float(h), self.img_size / float(w))
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            resized = np.asarray(Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR), dtype=np.uint8)
            canvas = np.full((self.img_size, self.img_size, 3), 128, dtype=np.uint8)
            pad_x = (self.img_size - new_w) // 2
            pad_y = (self.img_size - new_h) // 2
            canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
            image = canvas
            if boxes.size:
                boxes = boxes.copy()
                boxes[:, 0] *= scale
                boxes[:, 2] *= scale
                boxes[:, 1] *= scale
                boxes[:, 3] *= scale
                boxes[:, 0] += pad_x
                boxes[:, 2] += pad_x
                boxes[:, 1] += pad_y
                boxes[:, 3] += pad_y
            intrinsics_network = intrinsics_original.resize(float(scale), float(scale)).letterbox(float(scale), (float(pad_x), float(pad_y)))
            target["boxes"] = boxes
        target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.float32)
        target["labels"] = torch.as_tensor(target["labels"], dtype=torch.int64)
        target["z_gt"] = torch.as_tensor(target["z_gt"], dtype=torch.float32)
        target["distance_gt"] = torch.as_tensor(target["distance_gt"], dtype=torch.float32)
        target["loc3d"] = torch.as_tensor(target["loc3d"], dtype=torch.float32)
        target["object_ids"] = torch.as_tensor(target["object_ids"], dtype=torch.int64)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        meta = {
            "image_id": image_id,
            "orig_shape": (h, w),
            "network_shape": (self.img_size, self.img_size),
            "intrinsics_original": intrinsics_original,
            "intrinsics_network": intrinsics_network,
            "ratio": float(scale if self.img_size is not None and self.img_size > 0 else 1.0),
            "pad": (float(pad_x) if self.img_size is not None and self.img_size > 0 else 0.0, float(pad_y) if self.img_size is not None and self.img_size > 0 else 0.0),
        }
        return image_tensor, target, meta
