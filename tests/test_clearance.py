"""The park move when the arm cannot lift the pen 60 mm over the screen (a tablet lying flat on the table)."""

import numpy as np
import pytest

from tablet_tap.core.kinematics import PenArm
from tablet_tap.errors import Unreachable
from tablet_tap.robot.arm import FakeArm
from tablet_tap.robot.tablet_robot import ArmPlace
from tablet_tap.robot.tablet_robot import TabletRobot
from tablet_tap.web import sim

PEN = PenArm()
PARK = np.array([0.2, 0.9, -0.8, 0.0, 0.6, 0.0])
SEED = np.array([0.18, 1.5, -1.1, 0.0, 1.0, 0.0])


def test_clear_height_adapts_and_parks_from_a_corner():
    screen = sim.flat_screen()  # 60 mm over its centre is beyond the wrist's pitch limit with the pen upright
    with pytest.raises(Unreachable):
        PEN.solve(
            TabletRobot(FakeArm(PARK), PEN, screen, PARK, SEED, realtime=False).planner.tip_pose(0.5, 0.5, 0.06), SEED
        )
    arm = FakeArm(PARK)
    robot = TabletRobot(arm, PEN, screen, PARK, SEED, realtime=False)
    assert 0.03 <= robot.clear_m < 0.06
    robot.tap(0.95, 0.05)  # a far corner: straight up from here the clear height is out of reach
    assert robot.where is ArmPlace.HOVER
    arm.commands.clear()
    robot.to_park()
    assert robot.where is ArmPlace.PARK and np.allclose(arm.q, PARK, atol=1e-6)
    heights = np.array([screen.uv_of(tip[:3, 3]) for tip in PEN.fk(np.asarray(arm.commands))])
    over = heights[(heights[:, 0] > -0.15) & (heights[:, 0] < 1.15) & (heights[:, 1] > -0.15) & (heights[:, 1] < 1.15)]
    assert over[:, 2].min() > 0.9 * robot.limits.hover_m  # never low over the tablet on the way out


def test_no_clearance_at_all_is_reported():
    high = sim.flat_screen(centre=(0.27, 0.05, 0.035))  # hover is still within reach, 1.5 x hover above it is not...
    try:
        robot = TabletRobot(FakeArm(PARK), PEN, high, PARK, SEED, realtime=False)
    except Unreachable as exc:
        assert "够不到" in str(exc)
    else:
        assert robot.clear_m >= 0.03
