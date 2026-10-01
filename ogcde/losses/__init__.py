"""Reusable detection and distance losses."""

from .detection import DetectionLoss
from .distance import DistanceLoss

__all__ = ["DetectionLoss", "DistanceLoss"]
