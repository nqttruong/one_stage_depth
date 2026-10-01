from __future__ import annotations

from typing import Tuple

import torch

from .base import DistanceMethod


class GeometryResidualMethod(DistanceMethod):
    method_name = "geometry_residual"
    output_dim = 2

    def decode(self, logits: torch.Tensor, intrinsics=None, box_center=None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_z = logits[..., 0]
        delta = logits[..., 1]
        z = torch.exp(log_z.clamp(min=-5.0, max=6.0))
        if intrinsics is None or box_center is None:
            return z, torch.ones_like(z), z
        u = box_center[..., 0]
        v = box_center[..., 1]
        s_geo = torch.sqrt(1.0 + ((u - intrinsics.cx) / intrinsics.fx) ** 2 + ((v - intrinsics.cy) / intrinsics.fy) ** 2)
        d = z * s_geo * torch.exp(delta)
        return z, s_geo, d

    def target_from_gt(self, z_gt: torch.Tensor, dist_gt: torch.Tensor, intrinsics=None, box_center=None, **kwargs) -> torch.Tensor:
        z = z_gt.clamp(min=1e-3)
        if intrinsics is None or box_center is None:
            s_gt = dist_gt / z
            return torch.stack([torch.log(z), torch.log(s_gt).clamp(min=-10.0, max=10.0)], dim=-1)
        u = box_center[..., 0]
        v = box_center[..., 1]
        s_geo = torch.sqrt(1.0 + ((u - intrinsics.cx) / intrinsics.fx) ** 2 + ((v - intrinsics.cy) / intrinsics.fy) ** 2)
        s_gt = (dist_gt / z).clamp(min=1e-3)
        target_delta = torch.log(s_gt / s_geo.clamp(min=1e-6))
        return torch.stack([torch.log(z), target_delta], dim=-1)

    def loss(self, logits: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.nn.functional.smooth_l1_loss(logits, target, reduction="mean")
