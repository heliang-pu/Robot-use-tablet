"""Planning and execution against a synthetic screen and a software arm: no hardware."""

from helpers import ARM
from helpers import ASPECT
from helpers import PARK
from helpers import SEED
from helpers import touches
from helpers import true_screen
import numpy as np
import pytest

from tablet_tap.core.kinematics import JOINT_LIMITS_RAD
from tablet_tap.core.kinematics import pose
from tablet_tap.core.motion import MotionLimits
from tablet_tap.core.motion import TapPlanner
from tablet_tap.core.screen_map import ScreenMap
from tablet_tap.core.screen_map import fit_screen_map
from tablet_tap.errors import CalibrationError
from tablet_tap.errors import PathError
from tablet_tap.errors import Stopped
from tablet_tap.errors import Unreachable
from tablet_tap.robot.arm import FakeArm
from tablet_tap.robot.executor import run_segments
from tablet_tap.robot.tablet_robot import ArmPlace
from tablet_tap.robot.tablet_robot import TabletRobot


def test_limits_match_the_robot_code():
    from real_robot import conversions as rc

    assert np.allclose(JOINT_LIMITS_RAD, rc.REAL_JOINT_LIMITS_RAD, atol=1e-4)


def test_fk_ik_round_trip():
    target = ARM.fk(SEED)
    q = ARM.solve(target, SEED + 0.2)
    assert np.allclose(ARM.fk(q)[:3, 3], target[:3, 3], atol=3e-4)
    assert np.all(q >= ARM.lower - 1e-9) and np.all(q <= ARM.upper + 1e-9)
    far = target.copy()
    far[0, 3] += 1.0
    with pytest.raises(Unreachable):
        ARM.solve(far, SEED)


def test_fit_recovers_the_screen():
    screen = true_screen()
    uv = [(u, v) for u in (0.1, 0.5, 0.9) for v in (0.1, 0.5, 0.9)]
    fitted = fit_screen_map(*touches(screen, uv, noise_mm=0.3, wobble_deg=3.0), aspect=ASPECT)
    assert fitted.n_points == 9 and fitted.residual_mm < 1.0
    for u, v in ((0.0, 0.0), (1.0, 1.0), (0.3, 0.8)):
        assert np.linalg.norm(fitted.point(u, v) - screen.point(u, v)) < 0.8e-3
    assert np.degrees(np.arccos(fitted.normal @ screen.normal)) < 0.6
    assert fitted.normal[2] > 0.9  # out of the screen is up
    u, v, h = fitted.uv_of(fitted.point(0.3, 0.7, 0.01))
    assert (round(u, 6), round(v, 6), round(h, 6)) == (0.3, 0.7, 0.01)
    assert abs(fitted.error_mm((0.5, 0.5), (0.5 + 0.004, 0.5)) - 1.0) < 0.05  # 0.004 of 250 mm


def test_fit_rejects_bad_touches():
    screen = true_screen()
    with pytest.raises(CalibrationError, match="至少 3"):
        fit_screen_map(*touches(screen, [(0.1, 0.1), (0.9, 0.9)]))
    with pytest.raises(CalibrationError, match="直线"):
        fit_screen_map(*touches(screen, [(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)]))
    uv, poses = touches(screen, [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)])
    with pytest.raises(CalibrationError, match="法线"):
        fit_screen_map(np.stack([1 - uv[:, 0], uv[:, 1]], axis=1), poses)  # u mirrored
    with pytest.raises(CalibrationError, match="宽高比"):
        fit_screen_map(uv, poses, aspect=1.0)


def robot(screen=None, follow=1.0, limits=None, start=PARK):
    screen = screen or true_screen()
    io = FakeArm(start, follow=follow)
    return TabletRobot(io, ARM, screen, PARK, SEED, limits, realtime=False), io, screen


def tips(io, screen):
    return np.array([screen.uv_of(t[:3, 3]) for t in ARM.fk(np.asarray(io.commands))])


def test_tap_lands_on_the_point_and_stays_clear():
    bot, io, screen = robot()
    assert bot.where is ArmPlace.PARK
    bot.tap(0.2, 0.3)
    bot.tap(0.9, 0.85)
    uvh = tips(io, screen)
    lim = bot.limits
    pressed = uvh[uvh[:, 2] < -0.8 * lim.press_m]
    assert len(pressed) > 4
    first, second = pressed[pressed[:, 0] < 0.5], pressed[pressed[:, 0] > 0.5]
    for group, (u, v) in ((first, (0.2, 0.3)), (second, (0.9, 0.85))):
        assert screen.error_mm((u, v), tuple(group[:, :2].mean(axis=0))) < 0.3
    assert uvh[:, 2].min() > -lim.press_m - 2e-4  # never deeper than the press
    # below hover height the pen only moves along the normal: within 0.3 mm of one of the two points
    low = uvh[uvh[:, 2] < 0.9 * lim.hover_m]
    for u, v, _ in low:
        assert min(screen.error_mm((u, v), (0.2, 0.3)), screen.error_mm((u, v), (0.9, 0.85))) < 0.3
    steps = np.abs(np.diff(np.asarray(io.commands), axis=0)).max()
    assert steps * lim.hz <= lim.max_joint_speed * 1.05
    # the pen stays along the screen normal while it is near the screen
    for q in np.asarray(io.commands)[uvh[:, 2] < 0.02]:
        assert np.degrees(np.arccos(np.clip(-ARM.fk(q)[:3, 2] @ screen.normal, -1, 1))) < 0.5
    assert bot.where is ArmPlace.HOVER
    bot.to_park()
    assert bot.where is ArmPlace.PARK and np.allclose(io.q, PARK, atol=1e-6)


def test_long_press_and_swipe():
    bot, io, screen = robot()
    bot.tap(0.5, 0.5, hold_s=0.8)
    held = tips(io, screen)
    assert np.sum(held[:, 2] < -0.9 * bot.limits.press_m) >= 0.8 * bot.limits.hz
    io.commands.clear()
    bot.swipe(0.5, 1.01, 0.5, 0.55, duration_s=0.3)  # the home gesture: from the bottom bezel upward
    uvh = tips(io, screen)
    on_glass = uvh[uvh[:, 2] < -0.8 * bot.limits.press_m]
    assert on_glass[0, 1] > 1.0 and on_glass[-1, 1] < 0.56 and np.abs(on_glass[:, 0] - 0.5).max() < 0.003
    assert np.all(np.diff(on_glass[:, 1]) <= 1e-4)  # one way only (IK tolerance aside)


def test_probe_stops_at_the_touch():
    bot, io, screen = robot()
    lower = ScreenMap(
        screen.origin - 0.002 * screen.normal, screen.right, screen.down, screen.tool_x
    )  # real glass 2 mm lower
    contact = bot.probe(0.4, 0.6, touched=lambda: lower.uv_of(ARM.fk(io.q)[:3, 3])[2] <= 0.0)
    assert contact is not None
    u, v, h = lower.uv_of(contact[:3, 3])
    assert abs(h) < 2e-4 and screen.error_mm((0.4, 0.6), (u, v)) < 0.3
    assert bot.where is ArmPlace.HOVER and abs(screen.uv_of(ARM.fk(bot.q)[:3, 3])[2] - bot.limits.hover_m) < 2e-4
    assert bot.probe(0.4, 0.6, touched=lambda: False) is None


def test_refuses_unsafe_paths():
    bot, io, screen = robot()
    with pytest.raises(Unreachable, match="够不到"):
        bot.tap(4.0, 0.5)
    bot, io, _ = robot(start=PARK + 0.3)
    assert bot.where is ArmPlace.UNKNOWN
    with pytest.raises(PathError, match="停靠"):
        bot.tap(0.5, 0.5)
    assert not io.commands
    # a park pose right above the glass: the joint move would pass low over the tablet
    low_park = ARM.solve(pose(screen.tool_rotation(), screen.point(0.1, 0.5, 0.004)), SEED)
    low = TabletRobot(FakeArm(low_park), ARM, screen, low_park, SEED, realtime=False)
    with pytest.raises(PathError, match="经过"):
        low.to_hover()
    # a path that does not start at the arm
    with pytest.raises(PathError, match="起点"):
        run_segments(FakeArm(PARK), TapPlanner(ARM, screen).tap(bot.centre_q, 0.5, 0.5), hz=50, realtime=False)


def test_stop_and_lagging_arm():
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 40

    screen = true_screen()
    io = FakeArm(PARK)
    bot = TabletRobot(io, ARM, screen, PARK, SEED, should_stop=stop, realtime=False)
    with pytest.raises(Stopped):
        bot.tap(0.5, 0.5)
    assert bot.where is ArmPlace.UNKNOWN and len(io.commands) == 40
    # an arm that lags behind its targets: the press waits until it has arrived before the dwell counts
    bot, io, screen = robot(follow=0.25)
    bot.tap(0.5, 0.5)
    measured_low = min(screen.uv_of(ARM.fk(io.q)[:3, 3])[2] for _ in [0])
    assert bot.where is ArmPlace.HOVER and measured_low > 0  # ended hovering
    arm = FakeArm(PARK, follow=0.0)  # an arm that does not move at all
    stuck = TabletRobot(arm, ARM, screen, PARK, SEED, MotionLimits(), realtime=False)
    with pytest.raises(PathError, match="没有到位"):
        stuck.to_hover()


def test_reach_report():
    planner = TapPlanner(ARM, true_screen())
    report = planner.reach_report(planner.hover_q(0.5, 0.5, SEED))
    assert report["n"] == 35 and report["unreachable"] == []
    far = TapPlanner(ARM, true_screen(centre=(0.46, 0.0, 0.05)))
    assert far.reach_report(SEED)["unreachable"]



def test_a_leaning_pen_keeps_the_contact_point_and_leans_toward_x(tmp_path):
    straight = true_screen()
    leaning = ScreenMap(straight.origin, straight.right, straight.down, straight.tool_x, tilt_deg=8.0)
    rot = leaning.tool_rotation()
    assert np.allclose(rot.T @ rot, np.eye(3)) and np.linalg.det(rot) > 0
    assert np.degrees(np.arccos(-rot[:, 2] @ leaning.normal)) == pytest.approx(8.0)
    assert rot[:, 2] @ leaning.tool_x > 0.1  # the tip leans toward x
    tip = TapPlanner(ARM, leaning).tip_pose(0.3, 0.7, 0.0)
    assert np.allclose(tip[:3, 3], straight.point(0.3, 0.7))
    leaning.save(tmp_path / "s.json")
    assert ScreenMap.load(tmp_path / "s.json").tilt_deg == 8.0
