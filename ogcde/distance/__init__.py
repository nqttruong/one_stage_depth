"""Distance estimation methods decoupled from detector architecture."""

from .base import DistanceMethod, DistanceHead
from .direct_distance import DirectDistanceMethod
from .learned_scale import LearnedScaleMethod
from .analytic_geometry import AnalyticGeometryMethod
from .geometry_residual import GeometryResidualMethod


def build_distance_method(name: str, **kwargs):
    mapping = {
        "direct_distance": DirectDistanceMethod,
        "learned_scale": LearnedScaleMethod,
        "analytic_geometry": AnalyticGeometryMethod,
        "geometry_residual": GeometryResidualMethod,
    }
    if name not in mapping:
        raise ValueError(f"Unknown distance method: {name}")
    return mapping[name](**kwargs)


__all__ = [
    "DistanceMethod",
    "DistanceHead",
    "DirectDistanceMethod",
    "LearnedScaleMethod",
    "AnalyticGeometryMethod",
    "GeometryResidualMethod",
    "build_distance_method",
]
