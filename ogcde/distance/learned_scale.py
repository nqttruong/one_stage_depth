from __future__ import annotations

from typing import Tuple

import torch

from .base import DistanceMethod


class LearnedScaleMethod(DistanceMethod):
    method_name = "learned_scale"
    output_dim = 2

    def decode(self, logits: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_z = logits[..., 0]
        log_s = logits[..., 1]
        z = torch.exp(log_z.clamp(min=-5.0, max=6.0))
        s = torch.exp(log_s.clamp(min=-3.0, max=3.0))
        d = torch.exp((log_z + log_s).clamp(min=-5.0, max=6.0))
        return z, s, d

    def target_from_gt(self, z_gt: torch.Tensor, dist_gt: torch.Tensor, **kwargs) -> torch.Tensor:
        z = z_gt.clamp(min=1e-3)
        s_gt = (dist_gt / z).clamp(min=1e-3)
        return torch.stack([torch.log(z), torch.log(s_gt)], dim=-1)

    def loss(self, logits: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.nn.functional.smooth_l1_loss(logits, target, reduction="mean")
