"""The wrist camera's own file (calib/wrist.json) and the arm that looks with it.

Kept apart from ``calib/setup.json``: the top-camera flow neither reads nor changes it. The look pose plays the part
that the park pose plays there: ``TabletRobot`` goes to it to see and comes from it to touch.
"""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import datetime
import json
import pathlib
import shutil
from typing import Any

import numpy as np

from tablet_tap.core.kinematics import PenArm
from tablet_tap.core.look import COVERED_PX
from tablet_tap.core.look import MOUNTS
from tablet_tap.core.look import Look
from tablet_tap.core.look import LookPlanner
from tablet_tap.core.look import LookRules
from tablet_tap.core.look import WristCamera
from tablet_tap.core.look import camera_in_flange
from tablet_tap.core.look import flange_pose
from tablet_tap.core.motion import MotionLimits
from tablet_tap.core.pinhole import CameraModel
from tablet_tap.core.screen_map import ScreenMap
from tablet_tap.errors import PathError
from tablet_tap.errors import SetupError
from tablet_tap.robot.arm import ArmIO
from tablet_tap.robot.tablet_robot import ArmPlace
from tablet_tap.robot.tablet_robot import TabletRobot

BENCH_WRIST_CAMERA = "realsense:262322071383:1920x1080"  # the left arm's
# the factory values of that camera at 640x480 (docs/bench_twin_params.json), at 1080p: 2.25 times, 4:3 cropped out of 16:9
NOMINAL_MODEL = CameraModel(1920, 1080, 610.064 * 2.25, 609.119 * 2.25, 321.108 * 2.25 + 240.0, 252.534 * 2.25)
MOVED_M, MOVED_DEG = 0.002, 0.4  # the tablet is not where it was when the wrist's picture was calibrated


@dataclasses.dataclass
class WristSetup:
    camera: str = BENCH_WRIST_CAMERA
    mount: str = "tap"  # which of ``look.MOUNTS`` holds the camera
    model: CameraModel | None = None  # the camera's own calibration, read from it when it was first opened
    in_flange: list[list[float]] | None = None  # (4, 4) from the calibration; None: the mount's drawing
    look_q: list[float] | None = None
    look_screen: dict[str, Any] | None = None  # the screen the look pose and the picture belong to
    records: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def calibrated(self) -> bool:
        return self.in_flange is not None

    def wrist_camera(self) -> WristCamera:
        if self.mount not in MOUNTS:
            raise SetupError(f"不认识的腕部支架 {self.mount!r}；可用的有 {sorted(MOUNTS)}。")
        drawn = camera_in_flange(*MOUNTS[self.mount])
        return WristCamera(self.model or NOMINAL_MODEL, drawn if self.in_flange is None else np.asarray(self.in_flange, np.float64))

    def planner(self, pen: PenArm, rules: LookRules | None = None, seed: int = 0) -> LookPlanner:
        camera = self.wrist_camera()
        full = (camera.model.width, camera.model.height) == (1920, 1080)
        return LookPlanner(pen, camera, rules, seed, COVERED_PX.get(self.mount) if full else None)

    def remember(self, look: Look, screen: ScreenMap) -> None:
        self.look_q, self.look_screen = look.q.tolist(), screen.to_json()

    def screen_moved(self, screen: ScreenMap) -> str | None:
        """Why the look pose no longer belongs to ``screen``; None when it does."""
        if self.look_q is None or self.look_screen is None:
            return "还没有看屏位"
        then = ScreenMap.from_json(self.look_screen)
        shift = max(float(np.linalg.norm(then.point(u, v) - screen.point(u, v))) for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)))
        turn = float(np.degrees(np.arccos(np.clip(then.normal @ screen.normal, -1, 1))))
        if shift > MOVED_M or turn > MOVED_DEG:
            return f"平板离看屏位标定时的位置差 {shift * 1e3:.1f} mm、{turn:.1f}°"
        return None

    def predicted_homography(self, pen: PenArm, q: np.ndarray, screen: ScreenMap) -> np.ndarray:
        """(3, 3) camera pixels -> (u, v), from where the arm, the camera and the screen are said to be."""
        camera = self.wrist_camera()
        to_camera = np.linalg.inv(flange_pose(pen, q) @ camera.in_flange)
        columns = np.column_stack([to_camera[:3, :3] @ screen.right, to_camera[:3, :3] @ screen.down, to_camera[:3, :3] @ screen.origin + to_camera[:3, 3]])
        return np.linalg.inv(camera.model.matrix @ columns)

    def to_json(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["model"] = None if self.model is None else self.model.to_json()
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> WristSetup:
        data = dict(data)
        model = data.pop("model", None)
        unknown = sorted(set(data) - {f.name for f in dataclasses.fields(cls)})
        if unknown:
            raise SetupError(f"腕部相机配置里有不认识的项：{unknown}")
        return cls(model=None if model is None else CameraModel.from_json(model), **data)


class WristStore:
    """The file on disk; every version it replaces is kept beside it."""

    def __init__(self, path: str | pathlib.Path) -> None:
        self.path = pathlib.Path(path)

    @property
    def view_path(self) -> pathlib.Path:
        return self.path.with_name(self.path.stem + "_view.json")

    def load(self) -> WristSetup:
        return WristSetup.from_json(json.loads(self.path.read_text())) if self.path.exists() else WristSetup()

    def save(self, setup: WristSetup) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            shutil.copy2(self.path, self.path.with_name(f"{self.path.stem}.{stamp}.bak.json"))
        self.path.write_text(json.dumps(setup.to_json(), indent=2, ensure_ascii=False))


class LookRobot(TabletRobot):
    """``TabletRobot`` whose park pose is the look pose, and that knows the way between it and the real park pose."""

    def __init__(
        self,
        arm_io: ArmIO,
        pen_arm: PenArm,
        screen: ScreenMap,
        look_q: np.ndarray,
        rest_q: np.ndarray,
        seed_q: np.ndarray,
        planner: LookPlanner,
        limits: MotionLimits | None = None,
        should_stop: Callable[[], bool] | None = None,
        realtime: bool = True,
        aim: Callable[[float, float], tuple[float, float]] | None = None,
    ) -> None:
        self.rest_q, self.look_planner = np.asarray(rest_q, np.float64), planner
        super().__init__(arm_io, pen_arm, screen, look_q, seed_q, limits, should_stop, realtime, aim)
        for name, q in (("停靠位和看屏位之间", self.rest_q), ("看屏位和屏幕上方之间", self._high_q)):
            problem = planner.path_problem(screen, q, self.park_q)
            if problem:
                raise PathError(f"{name}的路上离{problem}；换一个看屏位（重新运行 wrist view）。")

    def _at(self, q: np.ndarray) -> bool:
        return float(np.max(np.abs(self.io.read() - q))) < 0.06

    def to_look(self) -> None:
        """From the rest pose (or from over the screen) to the look pose."""
        if self.where is ArmPlace.UNKNOWN and self._at(self.rest_q):
            self.move_to(self.park_q, "去看屏位")
        self.to_park()

    def to_rest(self) -> None:
        """Back to the real park pose."""
        if self._at(self.rest_q):
            return
        self.to_park()
        self.move_to(self.rest_q, "回停靠位")

