"""The pen-tip sighting calibration, on simulated sightings."""

import numpy as np
import pytest

from tablet_tap.calibration.tip_pnp import solve_tip_pnp
from tablet_tap.core.kinematics import PenArm
from tablet_tap.core.kinematics import pose
from tablet_tap.core.pinhole import CameraModel
from tablet_tap.errors import CalibrationError

MODEL = CameraModel(1920, 1080, 1360.0, 1360.0, 960.0, 540.0)
TRUE_TIP = np.array([-0.002, 0.0005, 0.082])


def true_camera() -> np.ndarray:
    z = np.array([-0.2, 0.15, -1.0]) / np.linalg.norm([-0.2, 0.15, -1.0])  # looks down at the arm from 0.6 m
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    return pose(np.stack([x, np.cross(z, x), z], 1), np.array([0.28, -0.2, 0.62]))


def poses(rng: np.random.Generator, n: int = 14) -> np.ndarray:
    base = np.array([0.0, 0.5, -0.6, 0.0, 0.5, 0.0])
    return base + rng.uniform(-1, 1, (n, 6)) * [0.6, 0.35, 0.35, 0.6, 0.8, 0.8]


def sightings(joints: np.ndarray, offset: float, rng: np.random.Generator, noise: float) -> np.ndarray:
    moved = joints.copy()
    moved[:, 4] += offset
    reach = PenArm(tip_len=0.0).fk(moved)
    tips = reach[:, :3, :3] @ TRUE_TIP + reach[:, :3, 3]
    inverse = np.linalg.inv(true_camera())
    return MODEL.project(tips @ inverse[:3, :3].T + inverse[:3, 3]) + rng.normal(0, noise, (len(joints), 2))


def guess() -> np.ndarray:
    return pose(true_camera()[:3, :3], true_camera()[:3, 3] + [0.03, -0.02, 0.04])


def test_recovers_camera_and_tip():
    rng = np.random.default_rng(0)
    joints = poses(rng)
    fit = solve_tip_pnp(joints, sightings(joints, 0.0, rng, 0.5), MODEL, guess(), 0.085)
    assert np.linalg.norm(fit.camera_in_base[:3, 3] - true_camera()[:3, 3]) < 5e-3
    assert np.linalg.norm(fit.tip_in_fingertip - TRUE_TIP) < 3e-3
    assert fit.rms_px < 1.5 and fit.joint_offset_rad == 0.0


def test_free_joint_recovers_a_zero_error():
    rng = np.random.default_rng(1)
    joints = poses(rng)
    px = sightings(joints, np.radians(-2.4), rng, 0.5)
    fixed = solve_tip_pnp(joints, px, MODEL, guess(), 0.085)
    free = solve_tip_pnp(joints, px, MODEL, guess(), 0.085, free_joint=4)
    assert free.joint_offset_rad == pytest.approx(np.radians(-2.4), abs=np.radians(0.4))
    assert free.rms_px < 1.5 < fixed.rms_px


def test_too_few_views():
    with pytest.raises(CalibrationError):
        solve_tip_pnp(np.zeros((3, 6)), np.zeros((3, 2)), MODEL, guess(), 0.085)
