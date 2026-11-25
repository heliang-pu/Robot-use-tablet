"""How far back from the screen the arm has to swing before a fixed camera sees all of it.

The way between "over the screen" and the park pose is a straight line in joint space that the arm goes all the time.
Looking at the screen does not need all of it: ``clear_share`` finds the point of that line nearest the screen at which
the arm, drawn fatter than it is, no longer comes between the camera and the screen.
"""

from __future__ import annotations

import cv2
import numpy as np
import pinocchio as pin

from tablet_tap.core.kinematics import PenArm
from tablet_tap.core.kinematics import _arm_model
from tablet_tap.core.pinhole import CameraModel

MAX_SHARE = 0.5  # never nearer the screen than half way, whatever the drawing says (half way back: seen free, 2026-11-24)
MARGIN_PX = 60.0  # at 1920 px wide; about 2 cm at the screen
LINK_RADIUS_M = 0.07
# (radius, from, to) along the flange's axis: the gripper with the wrist camera (its mount reaches 120 mm out), the pen
# holder with the wire's loop, the pen
TOOL_M = ((0.15, 0.0, 0.11), (0.08, 0.11, 0.17), (0.02, 0.17, None))


def _ring(radius: float, start: float, stop: float) -> np.ndarray:
    angle = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
    z = np.linspace(start, stop, 5)
    a, h = np.meshgrid(angle, z)
    return np.stack([radius * np.cos(a), radius * np.sin(a), h], axis=-1).reshape(-1, 3)


def _bodies(pen: PenArm, q: np.ndarray) -> list[np.ndarray]:
    """Point clouds around the arm's parts, in the base frame; each one's convex hull holds the part."""
    model = _arm_model(pen.urdf_path)
    data = model.createData()
    pin.forwardKinematics(model, data, np.asarray(q, np.float64))
    pin.updateFramePlacements(model, data)
    at = [data.oMf[model.getFrameId(name)] for name in ("link2", "link3", "link4", "link5", "link6")]
    ball = np.concatenate([_ring(LINK_RADIUS_M * np.cos(t), 0, 0)[:20] + [0, 0, LINK_RADIUS_M * np.sin(t)] for t in (-1.2, -0.6, 0.0, 0.6, 1.2)])
    clouds = [np.concatenate([ball + a.translation, ball + b.translation]) for a, b in zip(at[:-1], at[1:], strict=True)]
    tip = 0.143 + pen.tip_len
    flange = at[-1]
    for radius, start, stop in TOOL_M:
        clouds.append(_ring(radius, start, tip if stop is None else stop) @ flange.rotation.T + flange.translation)
    return clouds


def _in_front(points: np.ndarray, near: float = 0.02) -> np.ndarray:
    """The part of the points' convex hull in front of the camera, as points whose hull holds it: what lies behind the
    plane z = ``near`` is replaced by where the lines from it to the points in front go through that plane."""
    ahead = points[:, 2] >= near
    if ahead.all() or not ahead.any():
        return points[ahead]
    front, back = points[ahead][:, None], points[~ahead][None]
    t = (near - back[..., 2:]) / (front[..., 2:] - back[..., 2:])
    return np.concatenate([points[ahead], (back + t * (front - back)).reshape(-1, 3)])


def gap_px(pen: PenArm, q: np.ndarray, model: CameraModel, camera_pose: np.ndarray, corners: np.ndarray) -> float:
    """Pixels between the arm and the screen (``corners`` (4, 3), base frame) in the camera's picture; negative when
    they overlap."""
    to_camera = np.linalg.inv(np.asarray(camera_pose, np.float64))

    def see(points: np.ndarray) -> np.ndarray:
        return points @ to_camera[:3, :3].T + to_camera[:3, 3]

    quad = model.project(see(np.asarray(corners, np.float64))).astype(np.float32)
    edge = np.concatenate([quad + (np.roll(quad, -1, 0) - quad) * t for t in np.linspace(0, 1, 16)])
    nearest = np.inf
    for cloud in _bodies(pen, q):
        seen = see(cloud)
        seen = _in_front(seen)
        if len(seen) < 3:
            continue  # all of it behind the camera
        hull = cv2.convexHull(model.project(seen).astype(np.float32))
        outline = hull[:, 0]
        outline = np.concatenate([outline + (np.roll(outline, -1, 0) - outline) * t for t in np.linspace(0, 1, 16)])
        inside = max(cv2.pointPolygonTest(quad, (float(x), float(y)), True) for x, y in outline)
        around = max(cv2.pointPolygonTest(hull, (float(x), float(y)), True) for x, y in edge)
        nearest = min(nearest, -max(inside, around))
    return float(nearest)


def clear_share(
    pen: PenArm,
    park_q: np.ndarray,
    over_q: np.ndarray,
    model: CameraModel,
    camera_pose: np.ndarray,
    corners: np.ndarray,
    margin_px: float | None = None,
) -> float:
    """The largest share of the way from the park pose to over the screen (0: the park pose) that leaves the camera's
    view of the screen free, and from which every pose back to the park pose does too."""
    margin = MARGIN_PX * model.width / 1920.0 if margin_px is None else margin_px
    park_q, over_q = np.asarray(park_q, np.float64), np.asarray(over_q, np.float64)
    best = 0.0
    for share in np.arange(0.0, MAX_SHARE + 1e-9, 0.05):
        if gap_px(pen, park_q + share * (over_q - park_q), model, camera_pose, corners) < margin:
            break
        best = float(share)
    return best
