"""The arm with the pen over one calibrated screen: the facade the rest of the program talks to.

``TabletRobot`` knows where the arm is (``ArmPlace``: parked out of the camera's view, hovering over the screen, or
unknown), plans gestures with ``TapPlanner`` and runs them with ``run_segments``. Gestures are refused from an unknown
place; joint-space moves are checked against passing low over the tablet.
"""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import enum

import numpy as np

from tablet_tap.core.kinematics import PenArm
from tablet_tap.core.kinematics import pose
from tablet_tap.core.motion import MotionLimits
from tablet_tap.core.motion import Segment
from tablet_tap.core.motion import TapPlanner
from tablet_tap.core.motion import joint_move
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

# recovering from where a stop left the arm
RECOVER_SPEED = 0.2  # rad/s, at most
CORRIDOR_RAD = 0.06  # this near the line between the park pose and over the screen the arm is on the way it always goes
RECOVER_ROOM_RAD = 0.4  # every joint at most this far outside what it goes through between park pose and screen
TABLE_ROOM_M = 0.03  # the pen tip stays this far above the base plane on the way back


class ArmPlace(enum.Enum):
    """Where the arm is, as far as the gestures are concerned."""

    PARK = "park"  # at the park pose, out of the camera's view
    HOVER = "hover"  # over the screen at hover height, pen along the screen normal
    VIEW = "view"  # on the way between the two, as near the screen as the camera's view of it allows
    UNKNOWN = "unknown"  # anywhere else, e.g. after a stop: only ``recover`` or an explicit ``park`` move leaves it


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
        view_share: float | None = None,
    ) -> None:
        """``view_share``: the share of the way from the park pose to over the screen at which the camera already sees
        all of the screen (``core.view_pose``); ``to_view`` then goes no farther back than that, and the moves between
        there and a point of the screen go straight, not by way of the screen's centre."""
        self.io, self.pen, self.screen = arm_io, pen_arm, screen
        self.aim = aim or (lambda u, v: (u, v))  # where to send the pen so that the screen is touched at (u, v)
        self.planner = TapPlanner(pen_arm, screen, limits)
        self.limits = self.planner.limits
        self.park_q = np.asarray(park_q, np.float64)
        self._should_stop, self._realtime = should_stop, realtime
        # the hover pose over the screen centre, on the IK branch of the calibration touches
        self.centre_q = self.planner.hover_q(0.5, 0.5, np.asarray(seed_q, np.float64))
        self.clear_m, self._high_q = self._clear_pose()
        self.view_q = self.park_q + float(view_share) * (self._high_q - self.park_q) if view_share else None
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

    def rescreen(self, screen: ScreenMap, seed_q: np.ndarray, view_share: Callable[[TabletRobot], float | None] | None = None) -> None:
        """The tablet was moved, the arm being back from it: plan on ``screen`` from now on."""
        if self.where not in (ArmPlace.PARK, ArmPlace.VIEW):
            raise PathError("机械臂不在看图位或停靠位，不能换屏幕位置。")
        self.screen = screen
        self.planner = TapPlanner(self.pen, screen, self.limits)
        self.centre_q = self.planner.hover_q(0.5, 0.5, np.asarray(seed_q, np.float64))
        self.clear_m, self._high_q = self._clear_pose()
        share = view_share(self) if view_share is not None else None
        self.view_q = self.park_q + float(share) * (self._high_q - self.park_q) if share else None

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

    def to_hover(self, u: float = 0.5, v: float = 0.5) -> None:
        """From the park pose to hovering over the screen: up and over in joint space, then straight down. Over the
        centre, or with ``view_share`` over (u, v) where the arm reaches high enough there."""
        if self.where is ArmPlace.HOVER:
            return
        if self.where is ArmPlace.UNKNOWN:
            raise PathError(
                "机械臂既不在停靠位姿也不在屏幕上方；请先用 park 命令（低速、确认路径无障碍）回到停靠位姿。"
            )
        segments = None
        if self.view_q is not None and (u, v) != (0.5, 0.5):
            try:
                segments = self._down_to(u, v, self._above(u, v))
            except (Unreachable, PathError):
                segments = None
        self._run(segments or self._down_to(0.5, 0.5, (self.clear_m, self._high_q)))
        self.where = ArmPlace.HOVER

    def _above(self, u: float, v: float) -> tuple[float, np.ndarray]:
        for height in CLEAR_HEIGHTS_M:
            if 1.5 * self.limits.hover_m <= height <= self.clear_m:
                try:
                    return height, self.pen.solve(self.planner.tip_pose(u, v, height), self.centre_q)
                except Unreachable:
                    continue
        raise Unreachable(f"(u={u:.3f}, v={v:.3f}) 上方抬不高")

    def _down_to(self, u: float, v: float, above: tuple[float, np.ndarray]) -> list[Segment]:
        height, high_q = above
        over = self.planner.joint_move("去屏幕上方", self.q, high_q)
        self._check_clear(over)
        high, hover = self.planner.tip_pose(u, v, height), self.planner.tip_pose(u, v, self.limits.hover_m)
        return [over, self.planner.line("降到悬停", high, hover, over.q[-1], self.limits.travel_speed, settle=True)]

    def _away(self, goal_q: np.ndarray, name: str) -> list[Segment]:
        """From hovering to ``goal_q``: to over the screen centre, straight up there (the one place where the clear
        height is known to be within reach), then in joint space. With ``view_share`` straight up from where the pen
        is, if that can be done."""
        if self.view_q is not None:
            try:
                start = self.planner.hover_pose_of(self.q)
                u, v, _ = self.screen.uv_of(start[:3, 3])
                height, _ = self._above(u, v)
                up = self.planner.line("升高", start, self.planner.tip_pose(u, v, height), self.q, self.limits.travel_speed)
                away = self.planner.joint_move(name, up.q[-1], goal_q, settle=True)
                self._check_clear(away)
                return [up, away]
            except (Unreachable, PathError, ValueError):
                pass
        centre = self.planner.to_point(self.q, 0.5, 0.5)
        low, high = self.planner.tip_pose(0.5, 0.5, self.limits.hover_m), self.planner.tip_pose(0.5, 0.5, self.clear_m)
        up = self.planner.line("升高", low, high, centre.q[-1], self.limits.travel_speed)
        away = self.planner.joint_move(name, up.q[-1], goal_q, settle=True)
        self._check_clear(away)
        return [centre, up, away]

    def to_park(self) -> None:
        """To the park pose, out of the camera's view."""
        if self.where is ArmPlace.PARK:
            return
        if self.where is ArmPlace.UNKNOWN:
            raise PathError("机械臂位置未知；请先用 park 命令（低速、确认路径无障碍）回到停靠位姿。")
        if self.where is ArmPlace.VIEW:  # on the line between the park pose and over the screen
            self._run([self.planner.joint_move("回停靠位", self.q, self.park_q, settle=True)])
        else:
            self._run(self._away(self.park_q, "回停靠位"))
        self.where = ArmPlace.PARK

    def to_view(self) -> None:
        """Where the camera sees all of the screen: the park pose, or with ``view_share`` a pose on the way there."""
        if self.view_q is None or self.where in (ArmPlace.PARK, ArmPlace.VIEW):
            if self.where is not ArmPlace.VIEW:
                self.to_park()
            return
        if self.where is ArmPlace.UNKNOWN:
            raise PathError("机械臂位置未知；请先用 park 命令（低速、确认路径无障碍）回到停靠位姿。")
        self._run(self._away(self.view_q, "去看图位"))
        self.where = ArmPlace.VIEW

    def recover(self) -> None:
        """From wherever a stop left the arm back to the park pose, slowly. Only from the space the arm works in: on the
        way between the park pose and the screen it just goes on; otherwise it lifts first, and the rest of the way is
        refused when it would bring the pen low over the tablet or the table."""
        if self.where is not ArmPlace.UNKNOWN:
            return
        q = self.io.read()
        ends = np.stack([self.park_q, self._high_q, self.centre_q])
        if np.any(q < ends.min(0) - RECOVER_ROOM_RAD) or np.any(q > ends.max(0) + RECOVER_ROOM_RAD):
            raise PathError("机械臂离平时走的地方太远，不自动回停靠位；请看好周围，在“点按”页点“让开”低速回去。")
        slow = dataclasses.replace(self.limits, joint_move_speed=min(self.limits.joint_move_speed, RECOVER_SPEED))

        def move(name: str, start: np.ndarray, goal: np.ndarray) -> Segment:
            segment = joint_move(name, start, goal, slow, settle=True)
            self._check_clear(segment)
            heights = self.pen.fk(segment.q)[:, 2, 3]
            if heights.min() < min(TABLE_ROOM_M, float(self.pen.fk(start)[2, 3])) - 1e-3:
                raise PathError(f"“{name}”会让笔尖离桌面太近；请看好周围，在“点按”页点“让开”。")
            return segment

        line = self._high_q - self.park_q
        share = float(np.clip((q - self.park_q) @ line / (line @ line), 0.0, 1.0))
        if float(np.max(np.abs(q - self.park_q - share * line))) < CORRIDOR_RAD:
            segments = [move("回停靠位", q, self.park_q)]
        else:
            segments, start = [], q
            tip = self.pen.fk(q)
            _, _, height = self.screen.uv_of(tip[:3, 3])
            if height < self.clear_m:  # along the screen's normal, so that nothing is dragged over the glass
                for wanted in (*(h for h in CLEAR_HEIGHTS_M if h <= self.clear_m), 1.5 * self.limits.hover_m):
                    if wanted <= height + 1e-3:
                        continue
                    above = tip.copy()
                    above[:3, 3] += (wanted - height) * self.screen.normal
                    try:
                        lift = self.planner.line("抬起", tip, above, q, self.limits.press_speed, settle=True)
                    except Unreachable:
                        continue
                    segments, start = [lift], lift.q[-1]
                    break
            try:
                segments.append(move("回停靠位", start, self.park_q))
            except PathError:
                segments += [move("去屏幕上方", start, self._high_q), move("回停靠位", self._high_q, self.park_q)]
        self._run(segments)
        self.where = ArmPlace.PARK

    def move_to(self, q_goal: np.ndarray, name: str = "关节移动") -> None:
        """A plain joint-space move from wherever the arm is (teaching; the operator has checked the way is free)."""
        self._run([self.planner.joint_move(name, self.io.read(), np.asarray(q_goal, np.float64), settle=True)])
        self.where = self._locate(self.q)

    def tap(self, u: float, v: float, hold_s: float | None = None) -> None:
        u, v = self.aim(u, v)
        self.to_hover(u, v)
        self._run(self.planner.tap(self.q, u, v, hold_s))

    def swipe(self, u1: float, v1: float, u2: float, v2: float, duration_s: float = 0.5) -> None:
        start = self.aim(u1, v1)
        self.to_hover(*start)
        self._run(self.planner.swipe(self.q, *start, *self.aim(u2, v2), duration_s))

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
