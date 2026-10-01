from __future__ import annotations

import torch


def build_distance_targets(z_gt: torch.Tensor, dist_gt: torch.Tensor, method: str, **kwargs):
    z = z_gt.clamp(min=1e-3)
    if method == "direct_distance":
        return torch.log(z)
    if method == "learned_scale":
        s_gt = (dist_gt / z).clamp(min=1e-3)
        return torch.stack([torch.log(z), torch.log(s_gt)], dim=-1)
    if method == "analytic_geometry":
        return torch.log(z)
    if method == "geometry_residual":
        s_gt = (dist_gt / z).clamp(min=1e-3)
        target_delta = torch.log(s_gt)
        return torch.stack([torch.log(z), target_delta], dim=-1)
    raise ValueError(f"Unknown distance method: {method}")
