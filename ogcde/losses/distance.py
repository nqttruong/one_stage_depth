from __future__ import annotations

import torch
import torch.nn as nn

from ogcde.distance import DirectDistanceMethod, LearnedScaleMethod, AnalyticGeometryMethod, GeometryResidualMethod


class DistanceLoss(nn.Module):
    """Common distance loss container for all four methods."""

    def __init__(self, method: str = "learned_scale", lambda_z: float = 1.0, lambda_scale: float = 0.5, lambda_distance: float = 1.0):
        super().__init__()
        self.method = method
        if method == "direct_distance":
            self.impl = DirectDistanceMethod()
        elif method == "learned_scale":
            self.impl = LearnedScaleMethod()
        elif method == "analytic_geometry":
            self.impl = AnalyticGeometryMethod()
        elif method == "geometry_residual":
            self.impl = GeometryResidualMethod()
        else:
            raise ValueError(f"Unsupported distance method: {method}")
        self.lambda_z = lambda_z
        self.lambda_scale = lambda_scale
        self.lambda_distance = lambda_distance

    def forward(self, logits, target, **kwargs):
        if logits is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        if hasattr(logits, "shape") and logits.dim() == 0:
            return logits
        return self.impl.loss(logits, target, **kwargs)
