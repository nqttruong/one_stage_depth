"""Detector adapters for the distance module."""

from .base import DetectorAdapter
from .yolov8_custom import YoloV8CustomDetector
from .fcos import FCOSDetectorAdapter


def build_detector(name: str, **kwargs):
    mapping = {
        "yolov8_custom": YoloV8CustomDetector,
        "fcos": FCOSDetectorAdapter,
    }
    if name not in mapping:
        raise ValueError(f"Unknown detector: {name}")
    return mapping[name](**kwargs)


__all__ = ["DetectorAdapter", "YoloV8CustomDetector", "FCOSDetectorAdapter", "build_detector"]
