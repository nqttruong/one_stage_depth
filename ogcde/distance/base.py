from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn


class DistanceMethod(ABC):
    """A detector-agnostic distance formulation."""

    method_name: str = "base"
    output_dim: int = 1

    @abstractmethod
    def decode(self, logits: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError

    @abstractmethod
    def target_from_gt(self, z_gt: torch.Tensor, dist_gt: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def loss(self, logits: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class DistanceHead(nn.Module):
    """Reusable dense distance head for P3/P4/P5 feature pyramids."""

    def __init__(self, in_channels: Sequence[int], method: DistanceMethod, hidden_channels: int = 128):
        super().__init__()
        self.method = method
        self.heads = nn.ModuleList()
        for ch in in_channels:
            self.heads.append(
                nn.Sequential(
                    nn.Conv2d(ch, hidden_channels, kernel_size=3, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(hidden_channels, method.output_dim, kernel_size=1),
                )
            )

    def forward(self, feats: Sequence[torch.Tensor]):
        out = []
        for feat, h in zip(feats, self.heads):
            out.append(h(feat))
        return out
