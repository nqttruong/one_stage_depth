from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class CameraIntrinsics:
    """Camera intrinsics in the final preprocessed image space.

    The methods intentionally keep all transforms explicit: crop, resize,
    letterbox and hflip all update the intrinsics to match the image seen by
    the network. This is required for the analytic geometry and residual modes.
    """

    fx: float
    fy: float
    cx: float
    cy: float

    def scale_from_pixel(self, u: float, v: float) -> float:
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        return math.sqrt(1.0 + x * x + y * y)

    def crop(self, x0: float, y0: float) -> "CameraIntrinsics":
        return CameraIntrinsics(
            fx=self.fx,
            fy=self.fy,
            cx=self.cx - x0,
            cy=self.cy - y0,
        )

    def resize(self, sx: float, sy: float) -> "CameraIntrinsics":
        return CameraIntrinsics(
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
        )

    def letterbox(self, scale: float, pad: tuple[float, float]) -> "CameraIntrinsics":
        px, py = pad
        return CameraIntrinsics(
            fx=self.fx * scale,
            fy=self.fy * scale,
            cx=self.cx * scale + px,
            cy=self.cy * scale + py,
        )

    def with_scale(self, scale: float) -> "CameraIntrinsics":
        return CameraIntrinsics(
            fx=self.fx * scale,
            fy=self.fy * scale,
            cx=self.cx * scale,
            cy=self.cy * scale,
        )

    def hflip(self, width: float) -> "CameraIntrinsics":
        """Mirror the image horizontally while keeping the same convention.

        We use the same transform convention as the current repo: a pixel at x in
        the original image becomes x' = width - x. This implies cx' = width - cx.
        """
        return CameraIntrinsics(
            fx=self.fx,
            fy=self.fy,
            cx=width - self.cx,
            cy=self.cy,
        )

    def as_dict(self) -> dict:
        return {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy}

    @classmethod
    def from_dict(cls, payload: dict) -> "CameraIntrinsics":
        return cls(
            fx=float(payload["fx"]),
            fy=float(payload["fy"]),
            cx=float(payload["cx"]),
            cy=float(payload["cy"]),
        )
