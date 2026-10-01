from __future__ import annotations

import math
from typing import Optional

import torch


def _safe_log(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.log(x.clamp_min(eps))


def direct_distance_target(distance_gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Direct distance target uses the metric distance, not the camera depth Z."""
    return _safe_log(distance_gt, eps)


def learned_scale_target(z_gt: torch.Tensor, distance_gt: torch.Tensor, eps: float = 1e-6):
    z = z_gt.clamp_min(eps)
    s_gt = (distance_gt / z).clamp_min(eps)
    return torch.stack([_safe_log(z, eps), _safe_log(s_gt, eps)], dim=-1)


def analytic_geometry_target(z_gt: torch.Tensor, eps: float = 1e-6):
    return _safe_log(z_gt.clamp_min(eps))


def geometry_residual_target(
    z_gt: torch.Tensor,
    distance_gt: torch.Tensor,
    intrinsics=None,
    box_center: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
):
    """Target for residual geometry method.

    delta_log_s_gt = log(D/Z) - log(s_geo)
                  = log((D/Z) / s_geo)
    """
    z = z_gt.clamp_min(eps)
    s_gt = (distance_gt / z).clamp_min(eps)
    if intrinsics is None or box_center is None:
        return torch.stack([_safe_log(z, eps), _safe_log(s_gt, eps)], dim=-1)

    u = box_center[..., 0]
    v = box_center[..., 1]
    s_geo = torch.sqrt(1.0 + ((u - intrinsics.cx) / intrinsics.fx) ** 2 + ((v - intrinsics.cy) / intrinsics.fy) ** 2)
    delta_log_s = _safe_log(s_gt / s_geo.clamp_min(eps), eps)
    return torch.stack([_safe_log(z, eps), delta_log_s], dim=-1)


def build_distance_targets(z_gt: torch.Tensor, dist_gt: torch.Tensor, method: str, **kwargs):
    method = method.lower()
    if method == "direct_distance":
        return direct_distance_target(dist_gt)
    if method == "learned_scale":
        return learned_scale_target(z_gt, dist_gt)
    if method == "analytic_geometry":
        return analytic_geometry_target(z_gt)
    if method == "geometry_residual":
        return geometry_residual_target(z_gt, dist_gt, **kwargs)
    raise ValueError(f"Unknown distance method: {method}")
