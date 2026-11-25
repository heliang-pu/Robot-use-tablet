"""Backing off only as far as the camera's view asks, and getting back from where a stop left the arm."""

import numpy as np
import pytest

from tablet_tap.core.kinematics import PenArm
from tablet_tap.core.motion import MotionLimits
from tablet_tap.core.pinhole import CameraModel
from tablet_tap.core.view_pose import clear_share
from tablet_tap.core.view_pose import gap_px
from tablet_tap.errors import PathError
from tablet_tap.errors import Stopped
from tablet_tap.robot.arm import FakeArm
from tablet_tap.robot.tablet_robot import ArmPlace
from tablet_tap.robot.tablet_robot import TabletRobot
from tablet_tap.web import sim

PEN = PenArm()
PARK = np.array([0.2, 0.9, -0.8, 0.0, 0.6, 0.0])
SEED = np.array([0.18, 1.5, -1.1, 0.0, 1.0, 0.0])
CAMERA = CameraModel(1920, 1080, 1380.0, 1380.0, 960.0, 540.0)
LIMITS = MotionLimits(joint_move_speed=0.25, max_joint_speed=0.6, travel_speed=0.05, press_speed=0.02)


def robot(arm, **kwargs):
    return TabletRobot(arm, PEN, sim.flat_screen(), PARK, SEED, LIMITS, realtime=False, **kwargs)


def looking_down(centre, height=0.6):
    pose = np.eye(4)
    pose[:3, :3] = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], float)
    pose[:3, 3] = [centre[0], centre[1], centre[2] + height]
    return pose


def test_view_share_keeps_the_screen_free_and_saves_the_way():
    screen = sim.flat_screen()
    corners = np.array([screen.point(u, v) for u, v in ((0, 0), (1, 0), (1, 1), (0, 1))])
    bot = robot(FakeArm(PARK))
    pose = looking_down(screen.point(0.5, 0.5))
    assert gap_px(PEN, bot._high_q, CAMERA, pose, corners) < 0  # over the screen the arm hides it
    turned = bot.park_q + [1.2, 0, 0, 0, 0, 0]  # folded and swung aside
    share = clear_share(PEN, turned, bot._high_q, CAMERA, pose, corners)
    assert 0.0 <= share <= 0.5
    for s in np.linspace(0, share, 5):
        if share:
            assert gap_px(PEN, turned + s * (bot._high_q - turned), CAMERA, pose, corners) > 0


def test_a_run_goes_no_farther_back_than_the_view_pose():
    full, quick = FakeArm(PARK), FakeArm(PARK)
    slow_bot, bot = robot(full), robot(quick, view_share=0.5)
    for b in (slow_bot, bot):
        for u, v in ((0.2, 0.3), (0.85, 0.7), (0.95, 0.05)):
            b.tap(u, v)
            b.to_view()
    assert slow_bot.where is ArmPlace.PARK and bot.where is ArmPlace.VIEW
    assert np.allclose(quick.q, bot.view_q, atol=1e-6)
    assert len(quick.commands) < 0.75 * len(full.commands)
    screen = bot.screen
    uvh = np.array([screen.uv_of(t[:3, 3]) for t in PEN.fk(np.asarray(quick.commands))])
    pressed = uvh[uvh[:, 2] < -0.8 * LIMITS.press_m]
    for u, v in ((0.2, 0.3), (0.85, 0.7), (0.95, 0.05)):
        assert min(screen.error_mm((u, v), tuple(p[:2])) for p in pressed) < 0.3
    low = uvh[uvh[:, 2] < 0.9 * LIMITS.hover_m]  # low only straight above the three points
    for u, v, _ in low:
        assert min(screen.error_mm((u, v), p) for p in ((0.2, 0.3), (0.85, 0.7), (0.95, 0.05))) < 0.3
    assert np.abs(np.diff(np.asarray(quick.commands), axis=0)).max() * LIMITS.hz <= LIMITS.max_joint_speed * 1.05
    bot.to_park()
    assert bot.where is ArmPlace.PARK and np.allclose(quick.q, PARK, atol=1e-6)


@pytest.mark.parametrize("after", [30, 150, 300, 420, 450, 470, 500, 560])
def test_recover_from_a_stop_anywhere(after):
    arm = FakeArm(PARK)
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > after

    bot = robot(arm, should_stop=stop)
    with pytest.raises(Stopped):
        for _ in range(10):
            bot.tap(0.8, 0.3)
            bot.to_park()
    arm.commands.clear()
    again = robot(arm)
    start = PEN.fk(arm.q)
    if again.where is ArmPlace.UNKNOWN:
        again.recover()
    else:
        again.to_park()
    assert again.where is ArmPlace.PARK and np.abs(arm.q - PARK).max() < 0.06
    if not arm.commands:
        return  # it was still at the park pose
    uvh = np.array([again.screen.uv_of(t[:3, 3]) for t in PEN.fk(np.asarray(arm.commands))])
    u0, v0, _ = again.screen.uv_of(start[:3, 3])
    low = uvh[uvh[:, 2] < 0.8 * LIMITS.hover_m]
    for u, v, _ in low:  # low only while lifting straight up
        if -0.15 < u < 1.15 and -0.15 < v < 1.15:
            assert again.screen.error_mm((u0, v0), (u, v)) < 1.0


def test_recover_refuses_far_away_places():
    arm = FakeArm(PARK + [1.5, 0, 0, 0, 0, 0])
    bot = robot(arm)
    assert bot.where is ArmPlace.UNKNOWN
    with pytest.raises(PathError, match="太远"):
        bot.recover()
    assert not arm.commands
