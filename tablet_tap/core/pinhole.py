"""The pinhole camera model: intrinsics and lens distortion of one camera at one resolution."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class CameraModel:
    """Pixels are OpenCV's: x to the right, y down, integer values at pixel centres. ``dist`` holds OpenCV's
    distortion coefficients (k1, k2, p1, p2, k3)."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @property
    def distortion(self) -> np.ndarray:
        return np.asarray(self.dist, np.float64)

    def project(self, points_cam: np.ndarray) -> np.ndarray:
        """(..., 3) points in the camera frame (z forward) -> (..., 2) pixels, without lens distortion."""
        points = np.asarray(points_cam, np.float64)
        return np.stack(
            [self.fx * points[..., 0] / points[..., 2] + self.cx, self.fy * points[..., 1] / points[..., 2] + self.cy],
            axis=-1,
        )

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self) | {"dist": list(self.dist)}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> CameraModel:
        return cls(
            width=int(data["width"]),
            height=int(data["height"]),
            fx=float(data["fx"]),
            fy=float(data["fy"]),
            cx=float(data["cx"]),
            cy=float(data["cy"]),
            dist=tuple(float(v) for v in data.get("dist", (0.0,) * 5)),
        )
