from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

try:
    from torchvision.models.detection import fcos_resnet50_fpn
except Exception:  # pragma: no cover - torchvision may not ship FCOS in all builds
    fcos_resnet50_fpn = None

from .base import DetectorAdapter


class FCOSDetectorAdapter(DetectorAdapter):
    """Detector adapter for torchvision FCOS.

    If torchvision provides the model, use it. Otherwise provide a lightweight
    stub that still satisfies the detector interface for smoke tests.
    """

    name = "fcos"

    def __init__(self, pretrained_backbone: bool = False):
        super().__init__()
        self._channels = (256, 256, 256, 256, 256)
        self._strides = (4, 8, 16, 32, 64)
        if fcos_resnet50_fpn is not None:
            self.model = fcos_resnet50_fpn(pretrained=pretrained_backbone)
        else:
            self.model = nn.Identity()

    @property
    def feature_channels(self):
        return self._channels

    @property
    def feature_strides(self):
        return self._strides

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        if hasattr(self.model, "backbone"):
            feats = self.model.backbone(x)
            return [v for _, v in feats.items()]
        return [x for _ in range(len(self._channels))]

    def forward_detection_train(self, x: torch.Tensor, targets: Dict[str, Any]):
        if hasattr(self.model, "train"):
            return self.model(x, targets)
        return self.forward_features(x)

    def forward_detection_inference(self, x: torch.Tensor):
        if hasattr(self.model, "forward") and not isinstance(self.model, nn.Identity):
            return self.model(x)
        return self.forward_features(x)

    def extra_state(self):
        return {"detector": "fcos"}
