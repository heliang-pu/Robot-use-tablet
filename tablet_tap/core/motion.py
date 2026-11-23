"""From a point on the screen to joint trajectories of the arm that holds the pen.

``TapPlanner`` turns tap / long press / swipe into joint paths sampled at the control rate. Paths start and end at the
hover height above the screen; lateral moves happen at hover height only, so nothing is dragged over the glass.
``joint_move`` and ``axis_move`` are the two moves that need no screen calibration (teaching).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from tablet_tap.core.kinematics import PenArm
from tablet_tap.core.kinematics import pose
from tablet_tap.core.screen_map import ScreenMap
from tablet_tap.errors import Unreachable


@dataclasses.dataclass(frozen=True)
class MotionLimits:
    hz: float = 50.0  # rate at which the joint path is sampled and sent
    hover_m: float = 0.015  # height of lateral moves above the screen
    press_m: float = 0.0025  # how far past the calibrated touch plane a press goes (the holder's flexure takes it up)
    travel_speed: float = 0.12  # m/s, average over a lateral move
    press_speed: float = 0.04  # m/s, average going down and up
    dwell_s: float = 0.08  # contact time of a tap
    max_joint_speed: float = 1.2  # rad/s: a segment is slowed down until no joint is faster
    joint_move_speed: float = 0.6  # rad/s, average of the fastest joint in a joint-space move (park <-> screen)
    min_segment_s: float = 0.12


def _min_jerk(n: int) -> np.ndarray:
    """(n,) progress 0 < s <= 1 of a minimum-jerk move sampled at n steps."""
    t = np.arange(1, n + 1) / n
    return 10 * t**3 - 15 * t**4 + 6 * t**5


@dataclasses.dataclass
class Segment:
    """A stretch of joint path. ``settle`` asks the executor to wait until the arm has arrived before going on (before
    touching and before a held contact); ``hold_s`` is how long to stay at its end."""

    name: str
    q: np.ndarray  # (N, 6)
    settle: bool = False
    hold_s: float = 0.0


def joint_move(
    name: str, q_start: np.ndarray, q_goal: np.ndarray, limits: MotionLimits | None = None, **kwargs
) -> Segment:
    """Joint-space move with a minimum-jerk profile, as slow as ``limits`` asks."""
    lim = limits or MotionLimits()
    q_start, q_goal = np.asarray(q_start, np.float64), np.asarray(q_goal, np.float64)
    span = float(np.max(np.abs(q_goal - q_start)))
    peak = 1.875  # peak / average speed of a minimum-jerk profile
    seconds = max(span / lim.joint_move_speed, span * peak / lim.max_joint_speed, lim.min_segment_s)
    n = max(int(np.ceil(seconds * lim.hz)), 2)
    return Segment(name, q_start + _min_jerk(n)[:, None] * (q_goal - q_start), **kwargs)


def axis_move(
    name: str,
    arm: PenArm,
    q_start: np.ndarray,
    distance_m: float,
    speed: float,
    limits: MotionLimits | None = None,
    **kwargs,
) -> Segment:
    """Straight move of the pen tip along the pen's own axis by ``distance_m`` (positive: out of the gripper, toward
    whatever the pen points at), keeping its orientation. Needs no screen calibration: the teach tool's guarded
    descent and its lift."""
    lim = limits or MotionLimits()
    start = arm.fk(np.asarray(q_start, np.float64))
    n = max(int(np.ceil(abs(distance_m) / speed * lim.hz)), 2)
    q, path = np.asarray(q_start, np.float64), []
    for s in np.arange(1, n + 1) / n:  # constant speed: the move may be cut short at any step
        target = start.copy()
        target[:3, 3] = start[:3, 3] + s * distance_m * start[:3, 2]
        q = arm.solve(target, q)
        path.append(q)
    path = np.asarray(path)
    fastest = float(
        np.max(np.abs(np.diff(np.concatenate([np.asarray(q_start, np.float64)[None], path]), axis=0))) * lim.hz
    )
    if fastest > lim.max_joint_speed:
        raise Unreachable(f"{name}: 关节速度 {fastest:.2f} rad/s 超过上限（靠近奇异位形），换个姿态再试。")
    return Segment(name, path, **kwargs)


class TapPlanner:
    def __init__(self, arm: PenArm, screen: ScreenMap, limits: MotionLimits | None = None) -> None:
        self.arm, self.screen, self.limits = arm, screen, limits or MotionLimits()
        self.rot = screen.tool_rotation()

    # ---- poses ---------------------------------------------------------------------------------------------------
    def tip_pose(self, u: float, v: float, height: float) -> np.ndarray:
        return pose(self.rot, self.screen.point(u, v, height))

    def hover_q(self, u: float, v: float, q_seed: np.ndarray) -> np.ndarray:
        return self.arm.solve(self.tip_pose(u, v, self.limits.hover_m), q_seed)

    # ---- segments ------------------------------------------------------------------------------------------------
    def line(
        self,
        name: str,
        start: np.ndarray,
        goal: np.ndarray,
        q_start: np.ndarray,
        speed: float,
        duration: float | None = None,
        **kwargs,
    ) -> Segment:
        """Straight pen-tip move from ``start`` to ``goal`` (4, 4, same orientation) beginning at joints ``q_start``."""
        lim = self.limits
        distance = float(np.linalg.norm(goal[:3, 3] - start[:3, 3]))
        seconds = max(duration if duration is not None else distance / speed, lim.min_segment_s)
        while True:
            n = max(int(np.ceil(seconds * lim.hz)), 2)
            q, path = np.asarray(q_start, np.float64), []
            for s in _min_jerk(n):
                target = start.copy()
                target[:3, 3] = start[:3, 3] + s * (goal[:3, 3] - start[:3, 3])
                try:
                    q = self.arm.solve(target, q)
                except Unreachable as exc:
                    u, v, h = self.screen.uv_of(target[:3, 3])
                    raise Unreachable(f"屏幕点 (u={u:.3f}, v={v:.3f}, 高 {h * 1e3:.1f} mm) 够不到：{exc}") from exc
                path.append(q)
            path = np.asarray(path)
            fastest = float(np.max(np.abs(np.diff(np.concatenate([q_start[None], path]), axis=0))) * lim.hz)
            if fastest <= lim.max_joint_speed or seconds > 20.0:
                if fastest > lim.max_joint_speed:
                    raise Unreachable(f"{name}: 关节速度降不下来（{fastest:.2f} rad/s），路径可能经过奇异位形。")
                return Segment(name, path, **kwargs)
            seconds *= min(max(fastest / lim.max_joint_speed, 1.1), 2.0)

    def joint_move(self, name: str, q_start: np.ndarray, q_goal: np.ndarray, **kwargs) -> Segment:
        """Joint-space move (minimum jerk); between the park pose and the screen."""
        return joint_move(name, q_start, q_goal, self.limits, **kwargs)

    # ---- gestures: each starts from joints ``q`` hovering anywhere over the screen and ends hovering ----------------
    def hover_pose_of(self, q: np.ndarray) -> np.ndarray:
        """The pen-tip pose of ``q``, checked to be a hover pose of this screen."""
        tip = self.arm.fk(q)
        u, v, h = self.screen.uv_of(tip[:3, 3])
        tilt = np.degrees(np.arccos(np.clip(tip[:3, 2] @ self.rot[:, 2], -1.0, 1.0)))
        if h < 0.5 * self.limits.hover_m or tilt > 3.0:
            raise ValueError(
                f"起点不是屏幕上方的悬停位姿（高 {h * 1e3:.1f} mm，笔偏离法线 {tilt:.1f}°）；先执行 to_hover。"
            )
        return tip

    def to_point(self, q: np.ndarray, u: float, v: float) -> Segment:
        """Lateral move at hover height to above (u, v)."""
        start = self.hover_pose_of(q)
        start_level = pose(self.rot, start[:3, 3])
        return self.line(
            "到位", start_level, self.tip_pose(u, v, self.limits.hover_m), q, self.limits.travel_speed, settle=True
        )

    def tap(self, q: np.ndarray, u: float, v: float, hold_s: float | None = None) -> list[Segment]:
        """Move over (u, v), press, hold (``dwell_s``, or ``hold_s`` for a long press), lift."""
        lim = self.limits
        over = self.to_point(q, u, v)
        up, down = self.tip_pose(u, v, lim.hover_m), self.tip_pose(u, v, -lim.press_m)
        press = self.line(
            "下压", up, down, over.q[-1], lim.press_speed, settle=True, hold_s=lim.dwell_s if hold_s is None else hold_s
        )
        lift = self.line("抬起", down, up, press.q[-1], lim.press_speed)
        return [over, press, lift]

    def swipe(
        self, q: np.ndarray, u1: float, v1: float, u2: float, v2: float, duration_s: float = 0.5
    ) -> list[Segment]:
        """Press at (u1, v1), slide on the glass to (u2, v2) in ``duration_s``, lift. Points may lie slightly outside
        [0, 1] (edge gestures start on the bezel)."""
        lim = self.limits
        over = self.to_point(q, u1, v1)
        up1, down1 = self.tip_pose(u1, v1, lim.hover_m), self.tip_pose(u1, v1, -lim.press_m)
        down2, up2 = self.tip_pose(u2, v2, -lim.press_m), self.tip_pose(u2, v2, lim.hover_m)
        press = self.line("下压", up1, down1, over.q[-1], lim.press_speed, settle=True, hold_s=0.05)
        slide = self.line("滑动", down1, down2, press.q[-1], lim.travel_speed, duration=duration_s)
        lift = self.line("抬起", down2, up2, slide.q[-1], lim.press_speed)
        return [over, press, slide, lift]

    def probe(self, q: np.ndarray, u: float, v: float, depth_m: float = 0.006, speed: float = 0.006) -> list[Segment]:
        """For the automatic calibration: move over (u, v), then go down slowly to ``depth_m`` below the current plane.
        The executor stops the second segment at the first touch the screen reports."""
        over = self.to_point(q, u, v)
        up, down = self.tip_pose(u, v, self.limits.hover_m), self.tip_pose(u, v, -depth_m)
        return [over, self.line("探触", up, down, over.q[-1], speed)]

    # ---- reach ---------------------------------------------------------------------------------------------------
    def reach_report(self, q_seed: np.ndarray, grid: tuple[int, int] = (7, 5), margin: float = 0.02) -> dict:
        """Which screen points the arm reaches at hover height and pressed: for a check right after calibrating.

        ``margin`` extends the grid beyond the screen edges (edge gestures start on the bezel).
        """
        us, vs = np.linspace(-margin, 1 + margin, grid[0]), np.linspace(-margin, 1 + margin, grid[1])
        unreachable, q_row = [], np.asarray(q_seed, np.float64)
        for v in vs:
            q = q_row
            for i, u in enumerate(us):
                try:
                    q_up = self.arm.solve(self.tip_pose(u, v, self.limits.hover_m), q)
                    self.arm.solve(self.tip_pose(u, v, -self.limits.press_m), q_up)
                    q = q_up
                    if i == 0:
                        q_row = q_up
                except Unreachable:
                    unreachable.append((round(float(u), 3), round(float(v), 3)))
        return {"n": int(len(us) * len(vs)), "unreachable": unreachable}
