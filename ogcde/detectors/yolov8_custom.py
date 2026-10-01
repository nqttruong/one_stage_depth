from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

from ogcde.model import OGCDENet

from .base import DetectorAdapter


class YoloV8CustomDetector(DetectorAdapter):
    """Thin adapter around the current YOLOv8-style detector implementation."""

    name = "yolov8_custom"

    def __init__(self, nc=3, backbone_size="n", **kwargs):
        super().__init__()
        self.model = OGCDENet(nc=nc, backbone_size=backbone_size)

    @property
    def feature_channels(self):
        return self.model.neck.out_ch

    @property
    def feature_strides(self):
        return self.model.STRIDES

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        p3, p4, p5 = self.model.backbone(x)
        f3, f4, f5 = self.model.neck(p3, p4, p5)
        return [f3, f4, f5]

    def forward_detection_train(self, x: torch.Tensor, targets: Dict[str, Any]):
        return self.model(x)

    def forward_detection_inference(self, x: torch.Tensor):
        return self.model(x)

    def extra_state(self):
        return {"detector": "yolov8_custom"}
