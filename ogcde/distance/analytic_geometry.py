from __future__ import annotations

from typing import Tuple

import torch

from .base import DistanceMethod


class AnalyticGeometryMethod(DistanceMethod):
    method_name = "analytic_geometry"
    output_dim = 1

    def decode(self, logits: torch.Tensor, intrinsics=None, box_center=None, **kwargs) -> Tuple[torch.Tensor]:
        log_z = logits
        z = torch.exp(log_z.clamp(min=-5.0, max=6.0))
        if intrinsics is None or box_center is None:
            return (z,)
        u = box_center[..., 0]
        v = box_center[..., 1]
        s_geo = torch.sqrt(1.0 + ((u - intrinsics.cx) / intrinsics.fx) ** 2 + ((v - intrinsics.cy) / intrinsics.fy) ** 2)
        d = z * s_geo
        return (z, s_geo, d)

    def target_from_gt(self, z_gt: torch.Tensor, dist_gt: torch.Tensor, **kwargs) -> torch.Tensor:
        return z_gt.clamp(min=1e-3).log()

    def loss(self, logits: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.nn.functional.smooth_l1_loss(logits, target, reduction="mean")
