"""Where a fixed camera is, in the arm's base frame, from seeing the pen's tip at a set of arm poses.

The arm holds the pen at several poses in the air and the camera sees the tip in each of them. One least-squares problem
gives the camera's pose and the tip's position in the gripper at the same time, so the pen's length need not be known.
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np
from scipy.optimize import least_squares

from tablet_tap.core.kinematics import PenArm
from tablet_tap.core.kinematics import pose
from tablet_tap.core.pinhole import CameraModel
from tablet_tap.errors import CalibrationError

MIN_VIEWS = 6


@dataclasses.dataclass(frozen=True)
class TipFit:
    camera_in_base: np.ndarray  # (4, 4)
    tip_in_fingertip: np.ndarray  # (3,) tip in the fingertip TCP frame (m): z is the length beyond the fingertips
    rms_px: float
    errors_px: np.ndarray  # (n,) distance of each seen tip from where the fit puts it
    condition: float  # of the scaled problem: large means the poses do not pin everything down
    joint_offset_rad: float = 0.0  # the free joint's zero error: true angle = reading + this (0 when none was free)


def solve_tip_pnp(
    joints: np.ndarray,
    pixels: np.ndarray,
    model: CameraModel,
    camera_guess: np.ndarray,
    tip_guess_m: float,
    *,
    free_joint: int | None = None,
) -> TipFit:
    """Fit the camera's pose and the tip's offset to ``pixels`` (n, 2) of the tip seen at joints (n, 6).

    ``free_joint`` (0-5) also fits that joint's zero error, for an arm whose joint zero is off by a little; the poses
    must then move that joint a lot, and the fit should be checked on poses it has not seen.
    """
    joints, pixels = np.asarray(joints, np.float64).reshape(-1, 6), np.asarray(pixels, np.float64).reshape(-1, 2)
    if len(joints) != len(pixels) or len(joints) < MIN_VIEWS:
        raise CalibrationError(f"需要至少 {MIN_VIEWS} 个位置，现在有 {len(joints)} 个。")
    flange = PenArm(tip_len=0.0)
    fingertip = flange.fk(joints)  # (n, 4, 4) poses of the fingertip TCP
    n_free = 0 if free_joint is None else 1

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return cv2.Rodrigues(x[:3])[0], x[3:6]

    def residual(x: np.ndarray) -> np.ndarray:
        rot, shift = unpack(x)  # camera <- base
        reached = fingertip
        if n_free:
            moved = joints.copy()
            moved[:, free_joint] += x[9]
            reached = flange.fk(moved)
        tips = reached[:, :3, :3] @ x[6:9] + reached[:, :3, 3]
        return (model.project(tips @ rot.T + shift) - pixels).ravel()

    base_to_cam = np.linalg.inv(camera_guess)
    start = np.concatenate([cv2.Rodrigues(base_to_cam[:3, :3])[0].ravel(), base_to_cam[:3, 3], [0, 0, tip_guess_m], np.zeros(n_free)])
    scale = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.005, 0.005, 0.02, 0.05][: 9 + n_free])
    prior = np.concatenate([np.full(6, np.inf), [0.02, 0.02, 0.06], np.full(n_free, 0.05)])  # roughly known

    def whole(x: np.ndarray) -> np.ndarray:
        return np.concatenate([residual(x), 1e-3 * (x - start)[6:] / prior[6:]])

    fit = least_squares(whole, start, x_scale=scale, loss="soft_l1", f_scale=2.0, method="trf")
    x = fit.x
    rot, shift = unpack(x)
    camera = pose(rot.T, -rot.T @ shift)
    errors = np.linalg.norm(residual(x).reshape(-1, 2), axis=1)
    jac = fit.jac / scale
    return TipFit(camera, x[6:9], float(np.sqrt(np.mean(errors**2))), errors, float(np.linalg.cond(jac)), float(x[9]) if n_free else 0.0)
