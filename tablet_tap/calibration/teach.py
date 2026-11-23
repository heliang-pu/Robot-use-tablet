"""Teaching by hand: jog the pen with the keyboard, touch the screen a few times, and fit where the screen is.

``TeachSession`` holds one session's state (the jog target, the step size, the park pose, the touches) and maps each
key to a handler; ``guarded_descent`` is the slow move along the pen that stops when the page reports a touch.
"""

from __future__ import annotations

from collections.abc import Callable
import contextlib
import dataclasses
import select
import sys
import termios
import tty

import numpy as np
from scipy.spatial.transform import Rotation

from tablet_tap.calibration.page_client import TouchPage
from tablet_tap.core.kinematics import PenArm
from tablet_tap.core.motion import MotionLimits
from tablet_tap.core.motion import axis_move
from tablet_tap.core.motion import joint_move
from tablet_tap.core.screen_map import ScreenMap
from tablet_tap.core.screen_map import fit_screen_map
from tablet_tap.errors import PathError
from tablet_tap.errors import Unreachable
from tablet_tap.robot.arm import ArmIO
from tablet_tap.robot.arm import GripsPen
from tablet_tap.robot.executor import run_segments

HELP = """
点动（基座坐标系，x 向前、y 向左、z 向上）：
  w / s  前 / 后        a / d  左 / 右        r / f  上 / 下        e / c  沿笔轴 伸出 / 收回
  i / k  俯仰 ±         j / l  横滚 ±         u / o  绕笔轴转 ±
  1 2 3 4  步长 0.2 / 1 / 5 / 20 mm（转动 0.2 / 1 / 2 / 5 度）
探触：   z  沿笔轴慢速下探，直到标定页报告触摸就停（最多 30 mm），自动记录这个触点，再抬起 15 mm
            用法：先把笔尖点动到屏幕上方 1–2 cm、大致垂直于屏幕，再按 z。比手动一点点往下点动安全
夹爪：   g  夹紧笔夹      G  张开
记录：   p  把当前位姿记为“停靠位姿”（相机看屏幕时机械臂待的地方，要高、要在相机视野外）
         平板标定页上每出现一次新的触点，会自动记录（需要至少 4 个，分布在屏幕四角附近和中间）
         x  删除最后一个触点
标记：   m  把笔尖现在的位置记为一个手动标记点（点标定板的角点时用：屏幕不会报告触摸）
         n  跳过一个标记点（够不到那个角时）      b  删除最后一个标记点
其它：   h  再看一遍帮助      q  结束：拟合屏幕平面并保存
"""
STEPS = {"1": (0.0002, 0.2), "2": (0.001, 1.0), "3": (0.005, 2.0), "4": (0.02, 5.0)}  # key -> (metres, degrees)
TRANSLATE = {"w": (1, 0, 0), "s": (-1, 0, 0), "a": (0, 1, 0), "d": (0, -1, 0), "r": (0, 0, 1), "f": (0, 0, -1)}
ROTATE = {"i": (0, 1, 0), "k": (0, -1, 0), "j": (1, 0, 0), "l": (-1, 0, 0), "u": (0, 0, 1), "o": (0, 0, -1)}
ALONG_PEN = {"e": 1.0, "c": -1.0}
MIN_TOUCHES = 4
GRIP_OPEN_M = 0.06


@dataclasses.dataclass(frozen=True)
class Touch:
    """One touch of the screen: where the page saw it and the arm's joints at that moment."""

    u: float
    v: float
    q: list[float]


@contextlib.contextmanager
def raw_keys():
    """Single key presses from the terminal without Enter; yields ``get(timeout) -> str | None``."""
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    def get(timeout: float) -> str | None:
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        return sys.stdin.read(1) if ready else None

    try:
        yield get
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def record_touches(
    page: TouchPage, seq: int, q: np.ndarray, touches: list[Touch], say: Callable[[str], None] = print
) -> int:
    """Append the page's new pointerdown events to ``touches`` with the joints ``q``; returns the new sequence number."""
    for event in page.touches(seq, "pointerdown"):
        seq = max(seq, int(event["seq"]))
        touches.append(Touch(float(event["u"]), float(event["v"]), np.asarray(q, np.float64).tolist()))
        say(f"  触点 {len(touches)}：u={event['u']:.4f} v={event['v']:.4f}")
    return seq


def guarded_descent(
    arm: ArmIO,
    pen: PenArm,
    limits: MotionLimits,
    touched: Callable[[], bool],
    should_stop: Callable[[], bool] | None,
    *,
    realtime: bool = True,
    depth_m: float = 0.03,
    speed: float = 0.005,
    lift_m: float = 0.015,
) -> np.ndarray | None:
    """Move the pen along its axis until ``touched()`` (at most ``depth_m``), then lift: by ``lift_m`` after a touch,
    back to the start after none. Returns the measured joints at the touch, or None when nothing was touched."""
    q0 = arm.read()
    down = axis_move("探触", pen, q0, depth_m, speed, limits)
    hit: dict[str, np.ndarray | None] = {"q": None}

    def check() -> bool:
        if touched():
            hit["q"] = arm.read()
            return True
        return False

    sent = run_segments(arm, [down], hz=limits.hz, should_stop=should_stop, stop_last_when=check, realtime=realtime)
    travelled = float(np.linalg.norm(pen.fk(sent)[:3, 3] - pen.fk(q0)[:3, 3]))
    back = lift_m if hit["q"] is not None else travelled
    up = axis_move("抬起", pen, sent, -back, 4 * speed, limits, settle=True)
    run_segments(arm, [up], hz=limits.hz, should_stop=should_stop, realtime=realtime)
    return hit["q"]


class TeachSession:
    """One teaching session at the keyboard.

    ``moves_arm`` False is the read-only session: the operator moves the arm by hand (drag teaching) and this only
    records the touches the page reports and the park pose.
    """

    def __init__(
        self,
        arm: ArmIO,
        pen: PenArm,
        limits: MotionLimits,
        *,
        page: TouchPage | None,
        moves_arm: bool = True,
        should_stop: Callable[[], bool] | None = None,
        realtime: bool = True,
        say: Callable[[str], None] = print,
    ) -> None:
        self.arm, self.pen, self.limits, self.page = arm, pen, limits, page
        self._moves_arm, self._should_stop, self._realtime, self._say = moves_arm, should_stop, realtime, say
        self.touches: list[Touch] = []
        self.marks: list[list[float] | None] = []  # joints at each manual mark, in order; None: that one was skipped
        self.park_q: list[float] | None = None
        self.probe_depth_m = 0.03  # how far the z key may go down
        self._step_m, self._step_deg = STEPS["3"]
        self._target = pen.fk(arm.read())  # where the jog keys move from: the last commanded pen-tip pose
        self._seq = page.last_down_seq() if page is not None else 0
        self._handlers: dict[str, Callable[[str], None]] = {"h": self._help, "p": self._set_park, "x": self._drop_touch}
        self._handlers.update({"m": self._mark, "n": self._mark, "b": self._drop_mark})
        self._handlers.update(dict.fromkeys(STEPS, self._set_step))
        if moves_arm:
            self._handlers.update({"g": self._grip, "G": self._grip, "z": self._probe})
            self._handlers.update(dict.fromkeys([*TRANSLATE, *ROTATE, *ALONG_PEN], self._jog))

    @property
    def step_m(self) -> float:
        return self._step_m

    # ---- the loop ----------------------------------------------------------------------------------------------------
    def run(self, get_key: Callable[[float], str | None]) -> None:
        """Handle keys until ``q`` or a stop; touches reported by the page are recorded all the while."""
        self._say(HELP)
        if not self._moves_arm:
            self._say("【只读示教】本程序不发指令：用拖动示教或主臂把笔带到屏幕上，触点同样会自动记录。")
        while not (self._should_stop is not None and self._should_stop()):
            key = get_key(0.05)
            self.poll()
            if key is not None and not self.press(key):
                break

    def poll(self) -> None:
        """Record the touches that arrived since the last look, with the joints the arm has now."""
        if self.page is not None:
            self._seq = record_touches(self.page, self._seq, self.arm.read(), self.touches, self._say)

    def press(self, key: str) -> bool:
        """Handle one key; False when the session is over."""
        if key == "q":
            return False
        handler = self._handlers.get(key)
        if handler is not None:
            handler(key)
        return True

    # ---- the keys ----------------------------------------------------------------------------------------------------
    def _help(self, key: str) -> None:
        self._say(HELP)

    def _set_step(self, key: str) -> None:
        self._step_m, self._step_deg = STEPS[key]
        self._say(f"  步长 {self._step_m * 1e3:g} mm / {self._step_deg:g} 度")

    def _set_park(self, key: str) -> None:
        self.park_q = self.arm.read().tolist()
        self._say(f"  停靠位姿已记录：{np.round(self.park_q, 3).tolist()}")

    def _drop_touch(self, key: str) -> None:
        if self.touches:
            self.touches.pop()
            self._say(f"  已删除，剩 {len(self.touches)} 个触点")

    def _mark(self, key: str) -> None:
        self.marks.append(self.arm.read().tolist() if key == "m" else None)
        self._say(f"  标记点 {len(self.marks)}：" + ("已记录" if key == "m" else "跳过"))

    def _drop_mark(self, key: str) -> None:
        if self.marks:
            self.marks.pop()
            self._say(f"  已删除，剩 {len(self.marks)} 个标记点")

    def _grip(self, key: str) -> None:
        if not isinstance(self.arm, GripsPen):
            return  # a software arm has no gripper
        self.arm.set_grip_width(0.0 if key == "g" else GRIP_OPEN_M)
        self.arm.command(self.arm.read())
        self._say("  夹爪：" + ("夹紧" if key == "g" else "张开"))

    def _probe(self, key: str) -> None:
        if self.page is None:
            self._say("  探触需要标定页（不要加 --no_page）")
            return
        page, seq = self.page, self._seq
        try:
            q_contact = guarded_descent(
                self.arm,
                self.pen,
                self.limits,
                lambda: page.last_down_seq() > seq,
                self._should_stop,
                realtime=self._realtime,
                depth_m=self.probe_depth_m,
            )
        except (Unreachable, PathError) as exc:
            self._say(f"  未执行：{exc}")
        else:
            if q_contact is None:
                self._say(f"  下探 {self.probe_depth_m * 1e3:.0f} mm 屏幕仍没有报告触摸：笔尖离屏幕太远，或者笔没有导电。已抬回原处。")
            else:  # with the joints of the contact, not of the lift that followed
                self._seq = record_touches(page, seq, q_contact, self.touches, self._say)
        self._target = self.pen.fk(self.arm.read())

    def _jog(self, key: str) -> None:
        goal = self._target.copy()
        if key in TRANSLATE:
            goal[:3, 3] += self._step_m * np.asarray(TRANSLATE[key], float)
        elif key in ALONG_PEN:
            goal[:3, 3] += ALONG_PEN[key] * self._step_m * self._target[:3, 2]
        else:  # about the pen tip's own axes: the tip stays where it is
            turn = Rotation.from_rotvec(np.radians(self._step_deg) * np.asarray(ROTATE[key], float)).as_matrix()
            goal[:3, :3] = self._target[:3, :3] @ turn
        q = self.arm.read()
        try:
            q_goal = self.pen.solve(goal, q)
        except Unreachable as exc:
            self._say(f"  到不了：{exc}")
            return
        try:
            run_segments(
                self.arm,
                [joint_move("点动", q, q_goal, self.limits)],
                hz=self.limits.hz,
                should_stop=self._should_stop,
                realtime=self._realtime,
            )
        except PathError as exc:
            self._say(f"  未执行：{exc}")
            self._target = self.pen.fk(self.arm.read())
            return
        self._target = goal

    # ---- the result --------------------------------------------------------------------------------------------------
    def fit(self, aspect: float | None) -> tuple[ScreenMap, list[float]] | None:
        """The screen fitted to the touches, and the joints of the touch nearest the screen centre (the IK seed).
        None with fewer than ``MIN_TOUCHES`` touches; ``CalibrationError`` when they do not fit a screen."""
        if len(self.touches) < MIN_TOUCHES:
            return None
        uv = np.array([[t.u, t.v] for t in self.touches])
        qs = np.array([t.q for t in self.touches])
        fitted = fit_screen_map(uv, self.pen.fk(qs), aspect=aspect)
        return fitted, qs[int(np.argmin(np.linalg.norm(uv - 0.5, axis=1)))].tolist()
