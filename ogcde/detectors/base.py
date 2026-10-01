from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn


class DetectorAdapter(nn.Module, ABC):
    """Common detector contract consumed by distance methods.

    A detector exposes pyramid features and a standardized feature metadata API.
    The distance module never depends on a particular detector class.
    """

    name: str = "base"

    @property
    @abstractmethod
    def feature_channels(self) -> Tuple[int, ...]:
        raise NotImplementedError

    @property
    @abstractmethod
    def feature_strides(self) -> Tuple[int, ...]:
        raise NotImplementedError

    @abstractmethod
    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError

    def forward_detection_train(self, x: torch.Tensor, targets: Dict[str, Any]):
        return self.forward_features(x)

    def forward_detection_inference(self, x: torch.Tensor):
        return self.forward_features(x)

    def extra_state(self) -> Dict[str, Any]:
        return {}


class NullDetectorAdapter(DetectorAdapter):
    name = "null"

    def __init__(self, channels=(64, 128, 256), strides=(8, 16, 32)):
        super().__init__()
        self._channels = tuple(channels)
        self._strides = tuple(strides)

    @property
    def feature_channels(self):
        return self._channels

    @property
    def feature_strides(self):
        return self._strides

    def forward_features(self, x: torch.Tensor):
        return [x for _ in self._channels]
