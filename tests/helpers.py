"""What several test files share: a synthetic screen in front of the arm, and touches of it."""

import numpy as np

from tablet_tap.core.kinematics import PenArm
from tablet_tap.core.kinematics import pose
from tablet_tap.core.screen_map import ScreenMap

ARM = PenArm()
SEED = np.array([0.0, 1.5, -1.1, 0.0, 1.0, 0.0])
PARK = np.array([0.0, 0.6, -0.5, 0.0, 0.3, 0.0])
ASPECT = 0.25 / 0.16


def true_screen(tilt_deg: float = 12.0, centre=(0.32, 0.0, 0.03), yaw_deg: float = 4.0) -> ScreenMap:
    """A 250 x 160 mm screen in front of the arm: its top edge is far from the robot and higher (stand tilt)."""
    tilt, yaw = np.radians(tilt_deg), np.radians(yaw_deg)
    right = np.array([np.sin(yaw), -np.cos(yaw), 0.0])  # the robot looks at the screen from its bottom edge
    up_slope = np.array([np.cos(yaw) * np.cos(tilt), np.sin(yaw) * np.cos(tilt), np.sin(tilt)])
    down = -up_slope
    origin = np.asarray(centre) - 0.125 * right - 0.08 * down
    return ScreenMap(origin, 0.25 * right, 0.16 * down, tool_x=-down)  # the gripper faces away from the robot


def touches(screen: ScreenMap, uv, noise_mm: float = 0.0, wobble_deg: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    rot = screen.tool_rotation()
    poses = []
    for u, v in uv:
        tilt = rng.normal(0, np.radians(wobble_deg), 3)
        skew = np.array([[0, -tilt[2], tilt[1]], [tilt[2], 0, -tilt[0]], [-tilt[1], tilt[0], 0]])
        r, _ = np.linalg.qr(rot @ (np.eye(3) + skew))
        r = r * np.sign(np.sum(r * rot, axis=0))  # keep the axes' directions
        poses.append(pose(r, screen.point(u, v) + rng.normal(0, noise_mm * 1e-3, 3)))
    return np.asarray(uv, float), np.asarray(poses)
