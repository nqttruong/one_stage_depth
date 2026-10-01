"""Detector-agnostic dataset interfaces for KITTI distance supervision."""

from .base import BaseDistanceDataset
from .kitti import KITTIDistanceDataset, parse_kitti_calib, parse_kitti_label
from .collate import collate_distance_batch
from .adapters.yolo import YOLOTargetAdapter
from .adapters.fcos import FCOSTargetAdapter

__all__ = [
    "BaseDistanceDataset",
    "KITTIDistanceDataset",
    "parse_kitti_calib",
    "parse_kitti_label",
    "collate_distance_batch",
    "YOLOTargetAdapter",
    "FCOSTargetAdapter",
]
