from __future__ import annotations

from typing import Tuple

import torch

from .base import DistanceMethod


class DirectDistanceMethod(DistanceMethod):
    method_name = "direct_distance"
    output_dim = 1

    def decode(self, logits: torch.Tensor, **kwargs) -> Tuple[torch.Tensor]:
        return (torch.exp(logits.clamp(min=-5.0, max=6.0)),)

    def target_from_gt(self, z_gt: torch.Tensor, dist_gt: torch.Tensor, **kwargs) -> torch.Tensor:
        distance = dist_gt.clamp_min(1e-6)
        return torch.log(distance)

    def loss(self, logits: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.nn.functional.smooth_l1_loss(logits, target, reduction="mean")
