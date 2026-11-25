"""The setup file (calib/setup.json): which arm and camera, the motion limits, and what the calibration steps found.

``Setup`` is the typed view of it; ``SetupStore`` reads and writes the file and keeps a dated copy of every version it
replaces. The file's keys are the dataclass fields; keys this version does not know are kept as they are.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import pathlib
import shutil
from typing import Any

import numpy as np

from tablet_tap.core.keyboard import Keyboard
from tablet_tap.core.kinematics import DEFAULT_TIP_LEN_M
from tablet_tap.core.kinematics import JOINT_LIMITS_RAD
from tablet_tap.core.kinematics import LIMIT_MARGIN_RAD
from tablet_tap.core.kinematics import PenArm
from tablet_tap.core.motion import MotionLimits
from tablet_tap.core.pinhole import CameraModel
from tablet_tap.core.screen_map import ScreenMap
from tablet_tap.errors import SetupError

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_SETUP_PATH = PROJECT_ROOT / "calib" / "setup.json"
DEFAULT_VIEW_PATH = PROJECT_ROOT / "calib" / "screen_view.json"
# The camera that looks at the tablet: the bench's top RealSense at its highest resolution (the user's choice of
# 2026-11-23). The command line uses it when neither --camera nor the setup file names another one.
BENCH_TOP_CAMERA = "realsense:262422077281:1920x1080"

MISSING_HINTS = {
    "arm.channel": "还没有指定拿笔的机械臂：加上 --channel can_l_follower（左臂）或 can_r_follower（右臂）。",
    "camera": "还没有指定相机：加上 --camera realsense:<序列号>:1920x1080（或 cv:0、file:<图片>）。",
    "park_q": "还没有停靠位姿；先在 teach 里按 p 记录。",
    "camera_pose": "还没有相机在机械臂坐标系里的位姿；先用标定板运行 board 做手眼标定。",
    "camera_model": "还没有相机内参；先用标定板运行 board 做手眼标定。",
    "screen_size_m": "还不知道屏幕发光区域的尺寸；量一下宽和高，加上 --screen_mm <宽> <高>。",
}


def runs_dir() -> pathlib.Path:
    """Where runs leave their records (screenshots, action logs, test results)."""
    return PROJECT_ROOT / "runs"


@dataclasses.dataclass
class ArmSetup:
    channel: str | None = None  # CAN channel of the arm that holds the pen
    joint_signs: list[float] = dataclasses.field(default_factory=lambda: [1.0] * 6)
    joint_offsets: list[float] = dataclasses.field(default_factory=lambda: [0.0] * 6)
    firmware: str = "auto"
    speed_percent: int = 30
    grip_force_n: float = 5.0
    tip_len: float = DEFAULT_TIP_LEN_M  # pen tip beyond the fingertip plane (m)
    # Where each arm's firmware stops its joints, by CAN channel: six (lowest, highest) in degrees of the arm's own
    # readings, as `limits` last read them. Planning and the driver's clamp keep to the row of ``channel``; an arm that
    # was never read is taken to have the URDF's limits, which are the left arm's. (An arm stops at its firmware's
    # limit: the left arm, sent to 70.9 deg on joint 5 on 2026-11-23 when that was 70, stayed at 69.98 and the move
    # never settled.)
    firmware_limits_deg: dict[str, list[list[float]]] = dataclasses.field(default_factory=dict)
    limit_margin_rad: float = LIMIT_MARGIN_RAD  # how far inside the limits to plan (the driver clamps at 0.02)

    def reading_limits_rad(self) -> np.ndarray:
        """(6, 2) the lowest and highest reading of each joint that this arm's firmware allows."""
        rows = self.firmware_limits_deg.get(self.channel or "")
        if rows is None:
            return JOINT_LIMITS_RAD.copy()
        if np.shape(rows) != (6, 2):
            raise SetupError(f"配置里 {self.channel} 的固件限位应该是 6 对（最小、最大，度）；用 limits 再读一次。")
        return np.radians(np.asarray(rows, np.float64))


@dataclasses.dataclass
class Setup:
    arm: ArmSetup = dataclasses.field(default_factory=ArmSetup)
    camera: str | None = None  # realsense:<serial>[:WxH] | cv:<index-or-url> | file:<path>
    park_q: list[float] | None = None  # joints of the pose out of the camera's view
    seed_q: list[float] | None = None  # joints of a touch near the screen centre: picks the IK branch
    screen_map: ScreenMap | None = None  # where the screen is, in the arm's base frame
    aspect: float | None = None  # screen width / height, from the calibration page
    nav: dict[str, list[float]] | None = None  # {"back": [u, v], "home": [u, v]} with three-button navigation
    keyboard: dict[str, list[float]] | None = None  # key -> (u, v) of the on-screen keyboard's anchor keys
    limits: dict[str, float] = dataclasses.field(default_factory=dict)  # overrides of MotionLimits' fields
    camera_model: CameraModel | None = None  # intrinsics of ``camera`` at the resolution it is opened with
    camera_pose: list[list[float]] | None = None  # (4, 4) the camera in the arm's base frame, from the board
    screen_size_m: list[float] | None = None  # width and height of the lit display
    records: dict[str, Any] = dataclasses.field(default_factory=dict)  # what the calibration steps measured, kept

    def motion_limits(self) -> MotionLimits:
        known = {f.name for f in dataclasses.fields(MotionLimits)}
        unknown = sorted(set(self.limits) - known)
        if unknown:
            raise SetupError(f"配置里的运动参数 {unknown} 不认识；可用的有 {sorted(known)}。")
        return dataclasses.replace(MotionLimits(), **self.limits)

    def pen_arm(self) -> PenArm:
        # The arm's limits are on its own readings (reading = sign * joint + offset): planning, which works on the
        # corrected joints, has to keep to the same readings or the arm stops short of the target.
        signs, offsets = np.array(self.arm.joint_signs)[:, None], np.array(self.arm.joint_offsets)[:, None]
        limits = np.sort((self.arm.reading_limits_rad() - offsets) * signs, axis=1)
        return PenArm(tip_len=float(self.arm.tip_len), limits=limits, margin=float(self.arm.limit_margin_rad))

    def on_screen_keyboard(self) -> Keyboard | None:
        """The calibrated on-screen keyboard."""
        if self.keyboard is None:
            return None
        return Keyboard({key: tuple(value) for key, value in self.keyboard.items()})

    def require(self, *names: str, hint: str = "") -> None:
        """Raise ``SetupError`` when one of the named entries (``park_q``, ``arm.channel``, ...) is not there yet."""
        missing = []
        for name in names:
            value: Any = self
            for part in name.split("."):
                value = getattr(value, part)
            if value is None:
                missing.append(name)
        if not missing:
            return
        if len(missing) == 1 and not hint and missing[0] in MISSING_HINTS:
            raise SetupError(MISSING_HINTS[missing[0]])
        raise SetupError(f"配置里还没有 {missing}；{hint or '先完成对应的标定步骤。'}")

    def to_json(self) -> dict[str, Any]:
        data = dict(self.records)
        data.update(
            arm=dataclasses.asdict(self.arm),
            camera=self.camera,
            park_q=self.park_q,
            seed_q=self.seed_q,
            screen_map=None if self.screen_map is None else self.screen_map.to_json(),
            aspect=self.aspect,
            nav=self.nav,
            keyboard=self.keyboard,
            limits=self.limits,
            camera_model=None if self.camera_model is None else self.camera_model.to_json(),
            camera_pose=self.camera_pose,
            screen_size_m=self.screen_size_m,
        )
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Setup:
        data = dict(data)
        arm_data = dict(data.pop("arm", None) or {})
        # files of 2026-11-23 have one number instead of the rows: the highest reading of joint 5 that `limits` wrote
        joint5_top = arm_data.pop("joint5_limit_deg", None)
        arm = ArmSetup(**{**dataclasses.asdict(ArmSetup()), **arm_data})
        if joint5_top is not None and arm.channel and arm.channel not in arm.firmware_limits_deg:
            rows = np.degrees(JOINT_LIMITS_RAD).round(1).tolist()
            rows[4][1] = float(joint5_top)
            arm.firmware_limits_deg[arm.channel] = rows
        screen_map = data.pop("screen_map", None)
        names = ("camera", "park_q", "seed_q", "aspect", "nav", "keyboard", "camera_pose", "screen_size_m")
        fields = {name: data.pop(name, None) for name in names}
        camera_model = data.pop("camera_model", None)
        return cls(
            arm=arm,
            screen_map=None if screen_map is None else ScreenMap.from_json(screen_map),
            camera_model=None if camera_model is None else CameraModel.from_json(camera_model),
            limits=dict(data.pop("limits", None) or {}),
            records=data,
            **fields,
        )


class SetupStore:
    """The setup file on disk."""

    def __init__(self, path: str | pathlib.Path = DEFAULT_SETUP_PATH) -> None:
        self.path = pathlib.Path(path)

    def load(self) -> Setup:
        if not self.path.exists():
            return Setup()
        return Setup.from_json(json.loads(self.path.read_text()))

    def save(self, setup: Setup, *, backup: bool = True) -> None:
        """Write the file; the version it replaces is kept beside it as ``<name>.<time>.bak.json``. ``backup`` False
        is for small frequent updates (one more learned tap), which would bury the copies worth keeping."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if backup and self.path.exists():
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.copy2(self.path, self.path.with_name(f"{self.path.stem}.{stamp}.bak.json"))
        self.path.write_text(json.dumps(setup.to_json(), indent=2, ensure_ascii=False))
