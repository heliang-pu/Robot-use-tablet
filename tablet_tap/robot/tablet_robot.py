"""The arm with the pen over one calibrated screen: the facade the rest of the program talks to.

``TabletRobot`` knows where the arm is (``ArmPlace``: parked out of the camera's view, hovering over the screen, or
unknown), plans gestures with ``TapPlanner`` and runs them with ``run_segments``. Gestures are refused from an unknown
place; joint-space moves are checked against passing low over the tablet.
"""

from __future__ import annotations

from collections.abc import Callable
import enum

import numpy as np

from tablet_tap.core.kinematics import PenArm
from tablet_tap.core.kinematics import pose
from tablet_tap.core.motion import MotionLimits
from tablet_tap.core.motion import Segment
from tablet_tap.core.motion import TapPlanner
from tablet_tap.core.screen_map import ScreenMap
from tablet_tap.errors import PathError
from tablet_tap.errors import Unreachable
from tablet_tap.robot.arm import ArmIO
from tablet_tap.robot.executor import START_TOL_RAD
from tablet_tap.robot.executor import run_segments

# The pen passes this high over the screen centre when coming from / going to the park pose: the first height the arm
# reaches. With the pen along the screen normal the wrist's pitch limit caps how high the tip can go (about 50 mm above
# the arm's base plane with a 65 mm pen), so 60 mm is not always there.
CLEAR_HEIGHTS_M = (0.06, 0.05, 0.04, 0.03)


class ArmPlace(enum.Enum):
    """Where the arm is, as far as the gestures are concerned."""

    PARK = "park"  # at the park pose, out of the camera's view
    HOVER = "hover"  # over the screen at hover height, pen along the screen normal
    UNKNOWN = "unknown"  # anywhere else, e.g. after a stop: only an explicit ``park`` move leaves it


class TabletRobot:
    """The arm with the pen over one calibrated screen."""

    def __init__(
        self,
        arm_io: ArmIO,
        pen_arm: PenArm,
        screen: ScreenMap,
        park_q: np.ndarray,
        seed_q: np.ndarray,
        limits: MotionLimits | None = None,
        should_stop: Callable[[], bool] | None = None,
        realtime: bool = True,
        aim: Callable[[float, float], tuple[float, float]] | None = None,
    ) -> None:
        self.io, self.pen, self.screen = arm_io, pen_arm, screen
        self.aim = aim or (lambda u, v: (u, v))  # where to send the pen so that the screen is touched at (u, v)
        self.planner = TapPlanner(pen_arm, screen, limits)
        self.limits = self.planner.limits
        self.park_q = np.asarray(park_q, np.float64)
        self._should_stop, self._realtime = should_stop, realtime
        # the hover pose over the screen centre, on the IK branch of the calibration touches
        self.centre_q = self.planner.hover_q(0.5, 0.5, np.asarray(seed_q, np.float64))
        self.clear_m, self._high_q = self._clear_pose()
        self.q = arm_io.read()  # last target sent (the measured joints until something is sent)
        self.where = self._locate(self.q)

    def _clear_pose(self) -> tuple[float, np.ndarray]:
        """The height over the screen centre from which the park move starts, and the joints there."""
        for height in CLEAR_HEIGHTS_M:
            if height < 1.5 * self.limits.hover_m:
                break
            try:
                return height, self.pen.solve(self.planner.tip_pose(0.5, 0.5, height), self.centre_q)
            except Unreachable:
                continue
        raise Unreachable(
            f"屏幕中心上方 {1.5 * self.limits.hover_m * 1e3:.0f} mm 以上的位置机械臂都够不到（笔垂直屏幕时手腕到了极限）；"
            "把平板放低、放近一些，或者缩短笔尖伸出的长度。"
        )

    def _locate(self, q: np.ndarray) -> ArmPlace:
        if float(np.max(np.abs(q - self.park_q))) < START_TOL_RAD:
            return ArmPlace.PARK
        try:
            self.planner.hover_pose_of(q)
        except ValueError:
            return ArmPlace.UNKNOWN
        return ArmPlace.HOVER

    def _run(self, segments: list[Segment], **kwargs) -> None:
        try:
            self.q = run_segments(
                self.io, segments, hz=self.limits.hz, should_stop=self._should_stop, realtime=self._realtime, **kwargs
            )
        except BaseException:
            self.where = ArmPlace.UNKNOWN
            self.q = self.io.read()
            raise

    def _check_clear(self, segment: Segment) -> None:
        """A joint-space move must not bring the pen low over the tablet."""
        for tip in self.pen.fk(segment.q):
            u, v, h = self.screen.uv_of(tip[:3, 3])
            if -0.15 < u < 1.15 and -0.15 < v < 1.15 and h < self.limits.hover_m * 0.9:
                raise PathError(
                    f"“{segment.name}”会让笔尖在平板上方只有 {h * 1e3:.0f} mm 处经过（u={u:.2f}, v={v:.2f}）；"
                    "请把停靠位姿改到更高、更靠外的位置。"
                )

    def to_hover(self) -> None:
        """From the park pose to hovering over the screen centre: up and over in joint space, then straight down."""
        if self.where is ArmPlace.HOVER:
            return
        if self.where is not ArmPlace.PARK:
            raise PathError(
                "机械臂既不在停靠位姿也不在屏幕上方；请先用 park 命令（低速、确认路径无障碍）回到停靠位姿。"
            )
        over = self.planner.joint_move("去屏幕上方", self.q, self._high_q)
        self._check_clear(over)
        high, hover = (
            self.planner.tip_pose(0.5, 0.5, self.clear_m),
            self.planner.tip_pose(0.5, 0.5, self.limits.hover_m),
        )
        down = self.planner.line("降到悬停", high, hover, over.q[-1], self.limits.travel_speed, settle=True)
        self._run([over, down])
        self.where = ArmPlace.HOVER

    def to_park(self) -> None:
        """Out of the camera's view: at hover height to over the screen centre, straight up there (the one place where
        the clear height is known to be within reach), then to the park pose in joint space."""
        if self.where is ArmPlace.PARK:
            return
        if self.where is not ArmPlace.HOVER:
            raise PathError("机械臂位置未知；请先用 park 命令（低速、确认路径无障碍）回到停靠位姿。")
        centre = self.planner.to_point(self.q, 0.5, 0.5)
        low, high = self.planner.tip_pose(0.5, 0.5, self.limits.hover_m), self.planner.tip_pose(0.5, 0.5, self.clear_m)
        up = self.planner.line("升高", low, high, centre.q[-1], self.limits.travel_speed)
        away = self.planner.joint_move("回停靠位", up.q[-1], self.park_q, settle=True)
        self._check_clear(away)
        self._run([centre, up, away])
        self.where = ArmPlace.PARK

    def move_to(self, q_goal: np.ndarray, name: str = "关节移动") -> None:
        """A plain joint-space move from wherever the arm is (teaching; the operator has checked the way is free)."""
        self._run([self.planner.joint_move(name, self.io.read(), np.asarray(q_goal, np.float64), settle=True)])
        self.where = self._locate(self.q)

    def tap(self, u: float, v: float, hold_s: float | None = None) -> None:
        self.to_hover()
        self._run(self.planner.tap(self.q, *self.aim(u, v), hold_s))

    def swipe(self, u1: float, v1: float, u2: float, v2: float, duration_s: float = 0.5) -> None:
        self.to_hover()
        self._run(self.planner.swipe(self.q, *self.aim(u1, v1), *self.aim(u2, v2), duration_s))

    def probe(self, u: float, v: float, touched: Callable[[], bool], depth_m: float = 0.006) -> np.ndarray | None:
        """Go down slowly at (u, v) until ``touched()``; returns the measured pen-tip pose at that moment (None: the
        screen reported nothing down to ``depth_m`` below the current plane), then lifts back to hover."""
        self.to_hover()
        hit = {"pose": None}

        def check() -> bool:
            if touched():
                hit["pose"] = self.pen.fk(self.io.read())
                return True
            return False

        self._run(self.planner.probe(self.q, u, v, depth_m), stop_last_when=check)
        here = self.pen.fk(self.q)
        up = self.planner.tip_pose(u, v, self.limits.hover_m)
        self._run(
            [
                self.planner.line(
                    "抬起", pose(self.planner.rot, here[:3, 3]), up, self.q, self.limits.press_speed, settle=True
                )
            ]
        )
        return hit["pose"]
