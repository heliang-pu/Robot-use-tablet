"""The composition root: what a front end (the command line, the browser console) works with.

``AppContext`` holds the setup and builds the arm, the robot, the camera and the calibration page from it. Nothing is
opened until it is asked for, so a front end that only looks never connects to the arm, and one that only reads never
commands it.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
import contextlib
import dataclasses
import enum
import os
import pathlib
import signal
import threading

import numpy as np

from tablet_tap.agent.robot_device import RobotTabletDevice
from tablet_tap.calibration.page_client import CalibClient
from tablet_tap.config import BENCH_TOP_CAMERA
from tablet_tap.config import Setup
from tablet_tap.config import SetupStore
from tablet_tap.core.kinematics import PenArm
from tablet_tap.core.motion import MotionLimits
from tablet_tap.errors import SetupError
from tablet_tap.robot.arm import ArmIO
from tablet_tap.robot.factory import ArmMode
from tablet_tap.robot.factory import open_arm
from tablet_tap.robot.tablet_robot import TabletRobot
from tablet_tap.vision.camera import Camera
from tablet_tap.vision.camera import open_camera
from tablet_tap.vision.rectifier import ScreenRectifier

HARDWARE_LOCK = "TABLET_TAP_NO_HARDWARE"  # set to 1: this process must not open the arm, its CAN channel or a camera


def hardware_locked() -> bool:
    return os.environ.get(HARDWARE_LOCK, "") not in ("", "0")


def require_hardware(what: str) -> None:
    """Refuse to touch the bench while the lock is set (development and tests: only --fake and files work)."""
    if hardware_locked():
        raise SetupError(f"环境变量 {HARDWARE_LOCK} 已设置，本进程不允许{what}；用 --fake 或去掉这个变量。")


class ArmUse(enum.Enum):
    """What a command does with the arm; decides how it is opened and whether the operator is asked first."""

    NONE = "none"
    READS = "reads"
    MOVES = "moves"


@dataclasses.dataclass(frozen=True)
class Options:
    """The options every sub-command shares."""

    setup_path: pathlib.Path
    view_path: pathlib.Path
    calib_url: str
    fake: bool = False  # no hardware at all
    dry_run: bool = False  # read the real joints once, move a software arm
    yes: bool = False  # do not ask before the first motion
    force: bool = False  # connect although another program drives the arms
    channel: str | None = None  # the four below are written into the setup
    camera: str | None = None
    press_mm: float | None = None
    tip_len_mm: float | None = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> Options:
        return cls(
            setup_path=pathlib.Path(args.setup),
            view_path=pathlib.Path(args.view),
            **{name: getattr(args, name) for name in ("calib_url", "fake", "dry_run", "yes", "force")},
            **{name: getattr(args, name, None) for name in ("channel", "camera", "press_mm", "tip_len_mm")},
        )


class AppContext:
    def __init__(self, options: Options, arm_use: ArmUse = ArmUse.MOVES) -> None:
        self.options = options
        self.store = SetupStore(options.setup_path)
        self.setup: Setup = self.store.load()
        self._apply_options()
        self.stop = threading.Event()  # set by Ctrl-C: the executor then stops and the arm holds
        if options.fake:
            self.arm_mode = ArmMode.FAKE
        elif options.dry_run:
            self.arm_mode = ArmMode.DRY_RUN
        else:
            self.arm_mode = ArmMode.COMMAND if arm_use is ArmUse.MOVES else ArmMode.READ_ONLY
        self._arm: ArmIO | None = None
        self._confirmed = False

    def _apply_options(self) -> None:
        options, setup = self.options, self.setup
        if options.channel:
            setup.arm.channel = options.channel
        if options.camera:
            setup.camera = options.camera
        elif not setup.camera:
            setup.camera = BENCH_TOP_CAMERA
        if options.press_mm is not None:
            setup.limits["press_m"] = options.press_mm * 1e-3
        if options.tip_len_mm is not None:
            setup.arm.tip_len = options.tip_len_mm * 1e-3

    # ---- the arm -----------------------------------------------------------------------------------------------------
    @property
    def arm(self) -> ArmIO:
        if self._arm is None:
            if self.arm_mode is not ArmMode.FAKE:
                require_hardware("连接机械臂")
            start = None if self.setup.park_q is None else np.asarray(self.setup.park_q, np.float64)
            self._arm = open_arm(
                self.setup.arm, self.arm_mode, fake_start=start, allow_other_programs=self.options.force
            )
        return self._arm

    @property
    def commands_arm(self) -> bool:
        """Whether targets from this program reach the real motors."""
        return self.arm_mode is ArmMode.COMMAND

    @property
    def realtime(self) -> bool:
        """False with the software arm of ``--fake``: paths are then run without waiting between targets."""
        return self.arm_mode is not ArmMode.FAKE

    @property
    def pen(self) -> PenArm:
        return self.setup.pen_arm()

    def confirm(self, what: str) -> None:
        """Ask once before the first motion of a command on the real arm."""
        if not self.commands_arm or self._confirmed or self.options.yes:
            return
        answer = input(f"机械臂将要运动（{what}）。确认有人在急停旁边、路径上没有障碍 [y/N] ")
        if answer.strip().lower() != "y":
            raise SystemExit("已取消。")
        self._confirmed = True

    def robot(self, limits: MotionLimits | None = None) -> TabletRobot:
        """The arm over the calibrated screen; ``limits`` replaces the setup's motion limits for this one robot."""
        setup = self.setup
        setup.require("screen_map", "park_q", "seed_q", hint="先运行 teach 完成示教标定。")
        return TabletRobot(
            self.arm,
            self.pen,
            setup.screen_map,
            np.asarray(setup.park_q),
            np.asarray(setup.seed_q),
            limits or setup.motion_limits(),
            should_stop=self.stop.is_set,
            realtime=self.realtime,
        )

    # ---- seeing and the calibration page -----------------------------------------------------------------------------
    def page(self) -> CalibClient:
        return CalibClient(self.options.calib_url)

    def camera(self) -> Camera:
        self.setup.require("camera")
        if not self.setup.camera.startswith("file:"):
            require_hardware("打开相机")
        return open_camera(self.setup.camera)

    def rectifier(self) -> ScreenRectifier:
        if not self.options.view_path.exists():
            raise SetupError(f"还没有 {self.options.view_path}；先运行 view 做相机标定。")
        return ScreenRectifier.load(self.options.view_path)

    def device(self) -> RobotTabletDevice:
        """The tablet as the GUI agent uses it: this camera, this calibration, this robot."""
        camera = self.camera()
        return RobotTabletDevice(
            self.robot(), camera, self.rectifier(), nav=self.setup.nav, keyboard=self.setup.on_screen_keyboard()
        )

    # ---- housekeeping ------------------------------------------------------------------------------------------------
    def save_setup(self) -> None:
        self.store.save(self.setup)
        print(f"已保存 {self.store.path}")

    @contextlib.contextmanager
    def interruptible(self) -> Iterator[None]:
        """Ctrl-C sets the stop flag (the executor then raises ``Stopped`` and the arm holds); a second one interrupts."""
        previous = signal.getsignal(signal.SIGINT)

        def handler(signum, frame):
            if self.stop.is_set():
                signal.signal(signal.SIGINT, previous)
                raise KeyboardInterrupt
            self.stop.set()
            print("\n收到 Ctrl-C：机械臂停在当前位置。再按一次强制退出。", flush=True)

        signal.signal(signal.SIGINT, handler)
        try:
            yield
        finally:
            signal.signal(signal.SIGINT, previous)
