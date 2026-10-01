from __future__ import annotations

from typing import Tuple

import torch


def decode_distance_logits(logits: torch.Tensor, method: str, **kwargs) -> Tuple[torch.Tensor, ...]:
    if method == "direct_distance":
        return (torch.exp(logits.clamp(min=-5.0, max=6.0)),)
    if method == "learned_scale":
        log_z = logits[..., 0]
        log_s = logits[..., 1]
        z = torch.exp(log_z.clamp(min=-5.0, max=6.0))
        s = torch.exp(log_s.clamp(min=-3.0, max=3.0))
        d = torch.exp((log_z + log_s).clamp(min=-5.0, max=6.0))
        return z, s, d
    if method == "analytic_geometry":
        z = torch.exp(logits.clamp(min=-5.0, max=6.0))
        return (z,)
    if method == "geometry_residual":
        log_z = logits[..., 0]
        delta = logits[..., 1]
        z = torch.exp(log_z.clamp(min=-5.0, max=6.0))
        return z, torch.exp(delta), z * torch.exp(delta)
    raise ValueError(f"Unknown distance method: {method}")
