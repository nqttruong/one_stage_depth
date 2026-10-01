"""Detector adapters for the distance module."""

from .base import DetectorAdapter
from .yolov8_custom import YoloV8CustomDetector
from .fcos import FCOSDetectorAdapter

__all__ = ["DetectorAdapter", "YoloV8CustomDetector", "FCOSDetectorAdapter"]
