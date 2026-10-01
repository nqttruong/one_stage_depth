"""Distance estimation methods decoupled from detector architecture."""

from .base import DistanceMethod, DistanceHead
from .direct_distance import DirectDistanceMethod
from .learned_scale import LearnedScaleMethod
from .analytic_geometry import AnalyticGeometryMethod
from .geometry_residual import GeometryResidualMethod

__all__ = [
    "DistanceMethod",
    "DistanceHead",
    "DirectDistanceMethod",
    "LearnedScaleMethod",
    "AnalyticGeometryMethod",
    "GeometryResidualMethod",
]
