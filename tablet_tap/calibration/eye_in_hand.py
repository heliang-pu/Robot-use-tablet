"""Where the wrist camera sits on the flange, from looks at the calibration page by the arm in different poses.

The tablet lies still and shows its markers; the arm looks at it from a set of poses. One least-squares problem over
all marker corners gives the camera's pose in the flange frame, the screen's pose in the arm's base frame and the
screen's size (as one scale), so nothing needs to be touched or measured by hand beforehand.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares

from tablet_tap.calibration.hand_eye import planar_pose
from tablet_tap.calibration.live_screen import mean_pose
from tablet_tap.core.kinematics import PenArm
from tablet_tap.core.kinematics import pose
from tablet_tap.core.look import flange_pose
from tablet_tap.core.pinhole import CameraModel
from tablet_tap.core.screen_map import ScreenMap
from tablet_tap.errors import CalibrationError
from tablet_tap.vision.layout import ScreenLayout

MIN_VIEWS = 8
MIN_MARKERS = 4
MIN_TURN_DEG = 25.0  # the views must turn the camera by at least this much about two different directions
OUTLIER_PX = 4.0  # times the median error: a view beyond it was blurred, or the arm had not arrived


@dataclasses.dataclass(frozen=True)
class View:
    q: np.ndarray  # (6,) measured joints at the moment of the picture
    markers: dict[int, np.ndarray]  # id -> (4, 2) corners in the picture


@dataclasses.dataclass(frozen=True)
class EyeInHand:
    camera_in_flange: np.ndarray  # (4, 4)
    screen_in_base: np.ndarray  # (4, 4): x right, y down, z into the glass, origin at the top-left corner
    screen_size_m: tuple[float, float]
    rms_px: float
    view_px: list[float]  # RMS of each view, the left-out ones included
    left_out: list[int]
    held_out_px: float  # RMS of views predicted by a fit that did not use them
    position_sigma_mm: float  # how far noise of this size moves the camera, and turns it
    angle_sigma_deg: float

    def to_json(self) -> dict[str, Any]:
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in dataclasses.asdict(self).items()}

    def screen_map(self) -> ScreenMap:
        width, height = self.screen_size_m
        rot, corner = self.screen_in_base[:3, :3], self.screen_in_base[:3, 3]
        centre = corner + 0.5 * width * rot[:, 0] + 0.5 * height * rot[:, 1]
        outward = np.array([centre[0], centre[1], 0.0])
        tool_x = outward - (outward @ rot[:, 2]) * rot[:, 2]
        return ScreenMap(corner, width * rot[:, 0], height * rot[:, 1], tool_x / np.linalg.norm(tool_x))


def _points(layout: ScreenLayout, markers: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(n, 2) normalized screen points and (n, 2) pixels of the layout's markers that were seen."""
    known = sorted(set(markers) & set(layout.markers))
    uv = [(u, v) for k in known for u0, v0, u1, v1 in (layout.markers[k],) for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))]
    return np.array(uv, np.float64).reshape(-1, 2), np.concatenate([np.asarray(markers[k], np.float64).reshape(4, 2) for k in known] or [np.zeros((0, 2))])


def _vector(t: np.ndarray) -> np.ndarray:
    return np.concatenate([cv2.Rodrigues(t[:3, :3])[0].ravel(), t[:3, 3]])


def _matrix(x: np.ndarray) -> np.ndarray:
    return pose(cv2.Rodrigues(np.asarray(x[:3], np.float64))[0], np.asarray(x[3:6], np.float64))


def _fit(
    flanges: list[np.ndarray], points: list[tuple[np.ndarray, np.ndarray]], size: np.ndarray, model: CameraModel, camera: np.ndarray, screen: np.ndarray,
    fit_size: bool = True,
) -> tuple[np.ndarray, np.ndarray, float, Any]:  # fmt: skip
    start = np.concatenate([np.zeros(6), np.zeros(6), [1.0]])

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        return camera @ _matrix(x[:6]), screen @ _matrix(x[6:12]), float(x[12]) if fit_size else 1.0

    def residual(x: np.ndarray) -> np.ndarray:
        cam, scr, scale = unpack(x)
        out = []
        for flange, (uv, px) in zip(flanges, points):
            on_glass = np.column_stack([uv * size * scale, np.zeros(len(uv))])
            to_camera = np.linalg.inv(flange @ cam) @ scr
            seen, _ = cv2.projectPoints(on_glass, cv2.Rodrigues(to_camera[:3, :3])[0], to_camera[:3, 3], model.matrix, model.distortion)
            out.append((seen.reshape(-1, 2) - px).ravel())
        return np.concatenate(out + [[(x[12] - 1.0) / 0.01 * 0.5]])  # the size is known to about a percent

    fit = least_squares(residual, start, x_scale=np.array([0.02] * 3 + [0.01] * 3 + [0.02] * 3 + [0.01] * 3 + [0.01]), loss="soft_l1", f_scale=3.0)
    cam, scr, scale = unpack(fit.x)
    return cam, scr, scale, fit


def _errors(
    flanges: list[np.ndarray], points: list[tuple[np.ndarray, np.ndarray]], size: np.ndarray, model: CameraModel, cam: np.ndarray, scr: np.ndarray, scale: float
) -> np.ndarray:
    """RMS pixel error of each view."""
    out = []
    for flange, (uv, px) in zip(flanges, points):
        to_camera = np.linalg.inv(flange @ cam) @ scr
        on_glass = np.column_stack([uv * size * scale, np.zeros(len(uv))])
        seen, _ = cv2.projectPoints(on_glass, cv2.Rodrigues(to_camera[:3, :3])[0], to_camera[:3, 3], model.matrix, model.distortion)
        out.append(float(np.sqrt(np.mean(np.sum((seen.reshape(-1, 2) - px) ** 2, axis=1)))))
    return np.array(out)


def solve_eye_in_hand(
    views: list[View],
    layout: ScreenLayout,
    screen_size_m: tuple[float, float],
    model: CameraModel,
    pen: PenArm,
    camera_guess: np.ndarray,
    *,
    fit_size: bool = True,
) -> EyeInHand:
    """Fit the camera on the flange to ``views``. ``camera_guess`` is where the mount's drawing puts it. With
    ``fit_size`` off the screen keeps the size it was given: an arm whose joint zeros are a few tenths of a degree off
    is fitted a little better by a screen that is larger and further away, by more than the size of a display is in
    doubt (1.8 % and 4 mm of height, in the bench's rehearsal)."""
    usable = [v for v in views if len(set(v.markers) & set(layout.markers)) >= MIN_MARKERS]
    if len(usable) < MIN_VIEWS:
        raise CalibrationError(f"看到至少 {MIN_MARKERS} 个标定码的位姿只有 {len(usable)} 个，至少要 {MIN_VIEWS} 个。")
    size = np.array(screen_size_m, np.float64)
    flanges = [flange_pose(pen, v.q) for v in usable]
    turns = np.array([cv2.Rodrigues(f[:3, :3] @ flanges[0][:3, :3].T)[0].ravel() for f in flanges])
    spread = np.degrees(np.linalg.svd(turns - turns.mean(0), compute_uv=False) / np.sqrt(len(turns)))
    if 2.0 * spread[1] < MIN_TURN_DEG / 2:
        raise CalibrationError(f"各位姿之间的转动太少（第二方向只有 {2 * spread[1]:.0f}°），定不出相机的位置。")
    points = [_points(layout, v.markers) for v in usable]
    guesses = []
    for flange, (uv, px) in zip(flanges, points):
        seen, _, _ = planar_pose(np.column_stack([uv * size, np.zeros(len(uv))]), px, model)
        guesses.append(flange @ camera_guess @ seen)
    screen_guess = mean_pose(guesses)

    keep = list(range(len(usable)))
    for _ in range(3):
        cam, scr, scale, fit = _fit([flanges[k] for k in keep], [points[k] for k in keep], size, model, camera_guess, screen_guess, fit_size)
        errors = _errors(flanges, points, size, model, cam, scr, scale)
        limit = max(OUTLIER_PX * float(np.median(errors)), 2.0)
        now = [k for k in range(len(usable)) if errors[k] <= limit]
        if now == keep or len(now) < MIN_VIEWS:
            break
        keep = now
    held = []
    for part in range(3):  # three folds: each view is predicted once by a fit that has not seen it
        train = [k for i, k in enumerate(keep) if i % 3 != part]
        test = [k for i, k in enumerate(keep) if i % 3 == part]
        if len(train) < 5 or not test:
            continue
        c, s, z, _ = _fit([flanges[k] for k in train], [points[k] for k in train], size, model, cam, scr, fit_size)
        held += _errors([flanges[k] for k in test], [points[k] for k in test], size, model, c, s, z).tolist()
    jac = fit.jac[:-1]
    rms = float(np.sqrt(np.mean(errors[keep] ** 2)))
    # corners of one view share the arm's error: the views, not the corners, are what is independent
    covariance = np.linalg.pinv(jac.T @ jac) * (rms**2 / 2) * (2 * sum(len(points[k][0]) for k in keep) / len(keep))
    sigma = np.sqrt(np.clip(np.diag(covariance), 0, None))
    return EyeInHand(
        camera_in_flange=cam,
        screen_in_base=scr,
        screen_size_m=(float(size[0] * scale), float(size[1] * scale)),
        rms_px=rms,
        view_px=errors.round(2).tolist(),
        left_out=[k for k in range(len(usable)) if k not in keep],
        held_out_px=float(np.sqrt(np.mean(np.square(held)))) if held else float("nan"),
        position_sigma_mm=float(np.linalg.norm(sigma[3:6]) * 1e3),
        angle_sigma_deg=float(np.degrees(np.linalg.norm(sigma[:3]))),
    )


def mount_error(found: np.ndarray, drawn: np.ndarray) -> tuple[float, float]:
    """How far the camera is from where the mount's drawing has it: (mm, degrees)."""
    turn = cv2.Rodrigues(drawn[:3, :3].T @ found[:3, :3])[0]
    return float(np.linalg.norm(found[:3, 3] - drawn[:3, 3]) * 1e3), float(np.degrees(np.linalg.norm(turn)))


def edge_width_px(image_bgr: np.ndarray, markers: dict[int, np.ndarray]) -> float | None:
    """How wide a sharp edge is in the picture: the 10-90 % distance (px) across the markers' outer edges, the median
    over their sides. About 1.5 in a well focused picture, 3 and more when the screen is out of focus."""
    gray = (cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr).astype(np.float32)
    widths = []
    for quad in markers.values():
        quad = np.asarray(quad, np.float64).reshape(4, 2)
        for k in range(4):
            a, b = quad[k], quad[(k + 1) % 4]
            along = b - a
            length = float(np.linalg.norm(along))
            if length < 30:
                continue
            normal = np.array([along[1], -along[0]]) / length
            reach = min(0.12 * length, 10.0)  # half the white border around the code
            t = np.linspace(-reach, reach, int(reach * 8) * 2 + 1)
            lines = []
            for s in np.linspace(0.15, 0.85, 25):
                at = a + s * along + t[:, None] * normal
                lines.append(cv2.remap(gray, at[:, 0].astype(np.float32)[None], at[:, 1].astype(np.float32)[None], cv2.INTER_LINEAR)[0])
            profile = np.median(np.array(lines), axis=0)
            if profile[0] > profile[-1]:
                profile = profile[::-1]
            low, high = float(np.mean(profile[:6])), float(np.mean(profile[-6:]))
            if high - low < 40:
                continue
            level = np.maximum.accumulate((profile - low) / (high - low))
            widths.append(float(np.interp(0.9, level, t) - np.interp(0.1, level, t)))
    return float(np.median(widths)) if widths else None
