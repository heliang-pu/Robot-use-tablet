"""The calibration console in the browser.

One page from which the operator connects the arm, jogs the pen, records touches, fits the screen, calibrates the
camera, runs the automatic probe and the tap test, and stops the arm. It lives in the same process as the tablet's
calibration page (``calibration.page_server``), so touches reach it without HTTP polling.

    bin/py -m tablet_tap.web.console [--port 8801] [--fake]
    tablet:   http://<pc>:8801/calib      (full screen)
    operator: http://<pc>:8801/console

Everything that moves the arm runs in one worker thread, one job at a time: a command sent while a job runs is refused,
never queued, so that no motion waits behind another. Stop is not a job: it sets a flag that the path executor checks
before every target, and the arm then holds where it is.
"""

from __future__ import annotations

import argparse
import collections
from collections.abc import Callable
import dataclasses
import datetime
import faulthandler
import importlib.util
import json
import logging
import pathlib
import queue
import signal
import subprocess
import threading
import time
from typing import Any

import cv2
import flask
import numpy as np

from tablet_tap import config
from tablet_tap.agent.robot_device import RobotTabletDevice
from tablet_tap.app import require_hardware
from tablet_tap.calibration import hand_eye
from tablet_tap.calibration import page_server
from tablet_tap.calibration import procedures
from tablet_tap.calibration import teach as teach_mod
from tablet_tap.calibration.live_screen import TAG_IDS
from tablet_tap.calibration.live_screen import LiveScreen
from tablet_tap.calibration.live_screen import mean_pose
from tablet_tap.calibration.tap_correction import TapCorrection
from tablet_tap.core.motion import TapPlanner
from tablet_tap.core.motion import joint_move
from tablet_tap.errors import CalibrationError
from tablet_tap.errors import SetupError
from tablet_tap.errors import Stopped
from tablet_tap.errors import TabletTapError
from tablet_tap.errors import Unreachable
from tablet_tap.robot.executor import run_segments
from tablet_tap.robot.factory import ArmMode
from tablet_tap.robot.factory import open_arm
from tablet_tap.robot.tablet_robot import ArmPlace
from tablet_tap.robot.tablet_robot import TabletRobot
from tablet_tap.vision import layout as screen_layout
from tablet_tap.vision.camera import open_camera
from tablet_tap.vision.rectifier import ScreenRectifier
from tablet_tap.vision.rectifier import detect_markers
from tablet_tap.web import sim

logger = logging.getLogger("tablet_tap.console")
STATIC_DIR = pathlib.Path(__file__).resolve().parent / "static"
PARK_SPEED = 0.25  # rad/s: the park move from an unknown pose is not collision-checked, so it is slow
JOG_KEYS = frozenset([*teach_mod.TRANSLATE, *teach_mod.ROTATE, *teach_mod.ALONG_PEN])
MOTION_KEYS = JOG_KEYS | {"g", "G", "z"}
TABLE_TAGS = cv2.aruco.DICT_APRILTAG_36h11  # the paper tags on the table: a reference for "did the camera move"
MIN_SCREEN_PX = 700
# A screen plane that is the camera's estimate (cli "place") may be off by millimetres: the probe then stays higher
# between the points and goes deeper, as cli/commands/probe.py does; nothing else may touch the screen before it.
CAMERA_SOURCE = "camera"
CAMERA_HOVER_M = 0.03
CAMERA_DEPTH_MM = 3.0  # below the estimate: a touch that is not reported presses the glass this deep, and no deeper
MIN_PLACE_MARKERS = 4
NEAR_DEPTH_MM = 2.5  # once a contact has corrected the estimate's height
BLOCKED_RAD = 0.008  # a joint this far behind its target while going down slowly: the pen stands on something
LEANS_DEG = (0.0, 3.0, 5.0, 8.0, 10.0, 15.0)  # the least that reaches the whole screen is used
FOLLOW_M, FOLLOW_DEG = 0.0015, 0.4  # the tablet has been moved when it is seen this far from where it was
STEADY_M, STEADY_DEG = 0.0008, 0.2  # two looks this close: it lies still
SCREEN_MM, TILT_DEG = (236.9, 148.1), 15.0  # an 11-inch 16:10 tablet; the pen's lean that keeps joint 5 off its limit
CONTACT_GATE_M, JOG_CONTACT_STEP_M = 0.025, 0.001
OFF_PLANE_M, OFF_PLANE_DEG = 0.005, 2.0  # the tablet lies on another plane (another stand, another slant)
PLANE_LOOKS = 5  # looks averaged into a plane
# A job that touches the screen is stopped when the tablet is seen this far from where the job has it, this many looks
# in a row: the taps would land beside their targets.
MOVED_STOP_M, MOVED_STOP_DEG, MOVED_LOOKS = 0.005, 1.5, 3
TOUCH_JOBS = frozenset(["点击位置", "一键校准", "执行指令", "点按", "滑动", "回桌面", "返回", "自动探触", "点按测试"])
STALE_M, STALE_DEG = 0.03, 4.0  # moved further than this, the learned taps no longer fit the arm's pose
GRIP_OPEN_M, GRIP_TIME_S = 0.075, 1.5
SLOW_GRID, FAST_GRID, TEST_TAPS = (3, 3), (5, 4), 12
MAX_MISSES = 2  # taps in a row that the screen does not report end a calibration
BEST_CENTRE_M = (0.245, 0.0)  # in front of the arm's base: there the pen reaches the whole screen upright
PARK_LIFTS_M = (0.06, 0.04, 0.02)
CAMERA_PERIOD_S = 0.2


class LocalPage:
    """``calibration.page_client.TouchPage`` on the page state of this process."""

    def __init__(self, state: page_server.CalibState) -> None:
        self.state = state

    def screen(self) -> dict[str, Any] | None:
        return self.state.snapshot()["screen"]

    def layout(self) -> screen_layout.ScreenLayout:
        screen = self.screen()
        if screen is None:
            raise CalibrationError("标定页还没有连上：先在平板的浏览器里打开 /calib 并进入全屏。")
        return screen_layout.default_layout(screen["aspect"])

    def set_mode(self, mode: str, target: tuple[float, float] | None = None) -> None:
        self.state.set_mode(mode, target)

    def clear(self) -> None:
        self.state.clear()

    def last_down_seq(self) -> int:
        last = self.state.snapshot()["last_down"]
        return 0 if last is None else int(last["seq"])

    def touches(self, since_seq: int = 0, kind: str | None = None) -> list[dict[str, Any]]:
        return self.state.touches_after(since_seq, kind)[0]

    def wait_down(self, after_seq: int, timeout_s: float = 5.0) -> tuple[float, float] | None:
        deadline = time.monotonic() + timeout_s
        while True:
            downs = self.touches(after_seq, "pointerdown")
            if downs:
                return float(downs[0]["u"]), float(downs[0]["v"])
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.01)


class LockedArm:
    """An arm shared by the worker (which commands it) and the status reads of the page."""

    def __init__(self, arm: Any) -> None:
        self._arm, self._lock = arm, threading.Lock()
        self.sent: np.ndarray | None = None  # the newest target

    def read(self) -> np.ndarray:
        with self._lock:
            return self._arm.read()

    def command(self, q: np.ndarray) -> None:
        with self._lock:
            self._arm.command(q)
            self.sent = np.asarray(q, np.float64).copy()

    def torque(self) -> list[float] | None:
        """The motors' torques (N·m); None for an arm that does not report them."""
        hardware = getattr(self._arm, "_arm", None)
        if hardware is None or not hasattr(hardware, "read_extras"):
            return None
        extras = hardware.read_extras()
        return np.round(extras["joint_torque"], 3).tolist() if hardware.extras_complete else None

    def lag(self) -> float:
        """How far the joints are behind the newest target (rad, the worst joint)."""
        return 0.0 if self.sent is None else float(np.max(np.abs(self.read() - self.sent)))

    def close(self) -> None:
        with self._lock:
            self._arm.close()

    def __getattr__(self, name: str) -> Any:  # set_grip_width and the like, only where the arm has them
        return getattr(self._arm, name)


def detect_table_tags(image_bgr: np.ndarray) -> list[dict[str, Any]]:
    """The paper tags on the table: [{"id", "corners" (4, 2)}], several with one id allowed."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, _ = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(TABLE_TAGS), parameters).detectMarkers(
        gray
    )
    if ids is None:
        return []
    return [{"id": int(i), "corners": np.asarray(c, np.float64).reshape(4, 2)} for c, i in zip(corners, ids.ravel())]


def tag_drift(reference: list[dict[str, Any]], current: list[dict[str, Any]]) -> dict[str, Any]:
    """How far the table tags have moved in the picture since the camera calibration: each reference tag is matched to
    the nearest tag of its id now."""
    moved, missing = [], 0
    for ref in reference:
        ref_corners = np.asarray(ref["corners"], np.float64)
        same = [np.asarray(t["corners"], np.float64) for t in current if t["id"] == ref["id"]]
        if not same:
            missing += 1
            continue
        nearest = min(same, key=lambda c: float(np.linalg.norm(c.mean(0) - ref_corners.mean(0))))
        moved.append(float(np.linalg.norm(nearest - ref_corners, axis=1).mean()))
    return {"n": len(reference), "missing": missing, "max_px": max(moved) if moved else None}


class Console:
    """The bench as the page sees it: setup, arm, teaching session, camera, and the worker that runs the jobs."""

    def __init__(
        self,
        store: config.SetupStore,
        view_path: pathlib.Path,
        calib: page_server.CalibState,
        *,
        fake: bool = False,
        realtime: bool | None = None,
        allow_other_programs: bool = False,
    ) -> None:
        self.store, self.view_path, self.calib = store, pathlib.Path(view_path), calib
        self.fake = fake
        self.realtime = (not fake) if realtime is None else realtime
        self.allow_other_programs = allow_other_programs
        self.setup = store.load()
        self.page = LocalPage(calib)
        self.stop = threading.Event()
        self.lock = threading.RLock()
        self.arm: LockedArm | None = None
        self.arm_mode: ArmMode | None = None
        self.teach: teach_mod.TeachSession | None = None
        self.camera: Any = None
        self.rectifier: ScreenRectifier | None = None
        self.tag_reference: list[dict[str, Any]] = []
        self._load_view()
        self._frame: np.ndarray | None = None
        self._frame_time = 0.0
        self._camera_error = ""
        self._seen: dict[str, Any] = {"markers": 0, "tags": 0, "drift": None}
        self._jpeg_cache: dict[str, tuple[float, bytes]] = {}
        self.busy: str | None = None
        self.progress: dict[str, Any] | None = None
        self.results: dict[str, Any] = {}
        self.taps: collections.deque[dict[str, Any]] = collections.deque(maxlen=50)  # clicks: wanted and touched
        self.live = self._load_live()
        self.correction = self._load_correction()
        self._plane_looks: list[np.ndarray] = []
        self._last_look: np.ndarray | None = None  # the tablet's pose in the previous look
        self._tablet = {"seen": 0, "state": "none"}
        self._reach = ""
        self._located = False  # the camera has seen the tablet lie still in this session
        self._off_looks = 0  # looks in a row that saw the tablet away from where a running job has it
        self.https_port = 0  # of the second, https, server (the browser's microphone needs it); 0: there is none
        self.fault = ""
        self._log: collections.deque[dict[str, Any]] = collections.deque(maxlen=400)
        self._log_seq = 0
        self._jobs: queue.Queue[tuple[str, Callable[[], None]]] = queue.Queue(maxsize=1)
        self._closing = threading.Event()
        threading.Thread(target=self._work, daemon=True, name="console-worker").start()
        threading.Thread(target=self._watch, daemon=True, name="console-watch").start()
        self.say("控制台已启动" + ("（模拟模式：软件机械臂和虚拟平板，不连任何硬件）" if fake else ""))

    # ---- log -----------------------------------------------------------------------------------------------------
    def say(self, text: str) -> None:
        text = str(text).strip("\n")
        if not text.strip() or text.lstrip().startswith("步长 "):  # the step is the page's own state
            return
        with self.lock:
            for line in text.split("\n"):
                self._log_seq += 1
                self._log.append({"i": self._log_seq, "t": time.strftime("%H:%M:%S"), "text": line.rstrip()})
        logger.info("%s", text)

    # ---- jobs ----------------------------------------------------------------------------------------------------
    def submit(self, name: str, job: Callable[[], None]) -> bool:
        """Start ``job`` in the worker; False (nothing started) while another one runs."""
        with self.lock:
            if self.busy is not None:
                return False
            self.busy = name
        self.stop.clear()
        self._jobs.put((name, job))
        return True

    def _work(self) -> None:
        while not self._closing.is_set():
            try:
                name, job = self._jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                job()
            except Stopped as exc:
                self.say(f"■ 已停止：{exc}")
            except (TabletTapError, RuntimeError, ValueError, OSError) as exc:
                self.say(f"✗ {name}：{exc}")
            except Exception as exc:  # noqa: BLE001 - the worker must survive whatever a job does
                logger.exception("job %s failed", name)
                self.say(f"✗ {name} 出错：{type(exc).__name__}: {exc}")
            finally:
                if self.teach is not None and name != "点动":
                    self._renew_teach()  # touches a procedure used up must not be taken for teaching touches
                with self.lock:
                    self.busy, self.progress = None, None

    def _watch(self) -> None:
        """Camera frames, and in the read-only session the touches made by hand."""
        last_seen = 0.0
        while not self._closing.is_set():
            time.sleep(0.05)
            camera = self.camera
            if camera is not None and time.monotonic() - self._frame_time >= CAMERA_PERIOD_S:
                try:
                    frame = camera.read()
                except Exception as exc:  # noqa: BLE001 - a camera that dropped out must not end the thread
                    self._camera_error = f"{type(exc).__name__}: {exc}"
                else:
                    self._frame, self._frame_time, self._camera_error = frame, time.monotonic(), ""
                    if time.monotonic() - last_seen > 0.4:
                        last_seen = time.monotonic()
                        self._look(frame)
            with self.lock:
                if self.busy is None and self.teach is not None and self.arm_mode is ArmMode.READ_ONLY:
                    try:
                        self.teach.poll()
                    except Exception as exc:  # noqa: BLE001
                        self.fault = f"{type(exc).__name__}: {exc}"

    def _load_live(self) -> LiveScreen | None:
        data = self.setup.records.get("live_screen")
        if data is not None and self.setup.screen_map is not None:
            return LiveScreen.from_json(data)
        return None

    def _load_correction(self) -> TapCorrection | None:
        if self.live is None:
            return None
        data = self.setup.records.get("tap_correction") or []
        return TapCorrection(self.live.size_m, [tuple(x) for x in data])

    def _aim(self, u: float, v: float) -> tuple[float, float]:
        return (u, v) if self.correction is None else self.correction.aim(u, v)

    def _forget(self, why: str, *, contacts: bool = True) -> None:
        """Drop what was learned with a pen or a tablet that is no longer as it was."""
        if self.correction is not None and self.correction.samples:
            self.correction.samples = []
            self.say(f"{why}，点按修正已清空，请重新校准")
        self.setup.records["tap_correction"] = []
        if contacts and self.live is not None and self.live.contacts:
            self.live.contacts = []

    def _apply_live(self, *, save: bool = True, lean_free: bool = False) -> None:
        """The map the gestures use, from the tablet's pose and the contacts; the picture of it as well."""
        live, setup = self.live, self.setup
        assert live is not None
        setup.screen_map, setup.screen_size_m = live.screen_map(), list(live.size_m)
        setup.records["live_screen"] = live.to_json()
        setup.records["screen_map_source"] = "touch" if live.affine else CAMERA_SOURCE
        # Upright if possible: a leaning pen slides on the glass. Changing the lean throws the contacts away, so with
        # contacts it is only reconsidered when the tablet has moved (``lean_free``), and kept if it still reaches.
        best: tuple[int, float, np.ndarray] | None = None
        judged = dataclasses.replace(self.setup.motion_limits(), hover_m=CAMERA_HOVER_M)
        current = live.tilt_deg if live.contacts else None
        for lean in [current] if current is not None and not lean_free else ([] if current is None else [current]) + list(LEANS_DEG):
            screen = dataclasses.replace(setup.screen_map, tilt_deg=lean)
            try:
                seed = hand_eye.seed_joints(setup.pen_arm(), screen, judged)
            except Unreachable:
                continue
            report = TapPlanner(setup.pen_arm(), screen, judged).reach_report(seed, (5, 4), -0.05)
            n = len(report["unreachable"])
            if best is None or n < best[0]:
                best = (n, lean, seed)
            if n == 0:
                break
        if best is None:
            self._reach = "够不到：平板往左臂底座方向挪近"
        else:
            n, lean, seed = best
            if lean != live.tilt_deg and live.contacts:
                self._forget("笔的倾角变了")  # the contacts were made with the pen leaning differently
                live.tilt_deg = lean
                return self._apply_live(save=save, lean_free=True)
            live.tilt_deg, setup.seed_q = lean, seed.tolist()
            setup.screen_map = dataclasses.replace(setup.screen_map, tilt_deg=lean)
            setup.records["live_screen"] = live.to_json()
            self._reach = "" if n == 0 else f"20 个检查点有 {n} 个够不到，平板往底座挪近一些"
            if not self._reach and lean > 5:
                self._reach = f"笔要斜 {lean:g}° 才够得到，会打滑：平板往底座挪近（中心离底座 24–28 cm 最好）"
        if setup.camera_model is not None and setup.camera_pose is not None:
            inverse = np.linalg.inv(np.asarray(setup.camera_pose))
            quad = setup.camera_model.project(live.corners() @ inverse[:3, :3].T + inverse[:3, 3])
            to_norm = cv2.getPerspectiveTransform(quad.astype(np.float32), np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]))
            model = setup.camera_model
            self.rectifier = ScreenRectifier(to_norm, live.size_m[0] / live.size_m[1], 1280,
                                             image_size=(model.width, model.height))  # fmt: skip
        if save:
            self._save()

    def _hover_limits(self) -> Any:
        """Hovering higher while the plane is only an estimate."""
        limits = self.setup.motion_limits()
        if self.from_camera:
            corrected = self.live is not None and self.live.contacts
            limits = dataclasses.replace(limits, hover_m=max(limits.hover_m, 0.02 if corrected else CAMERA_HOVER_M))
        return limits

    def _follow(self, frame: np.ndarray, found: dict[int, np.ndarray]) -> None:
        """Keep up with the tablet: adopt where the camera sees it once it lies still somewhere else."""
        setup, model = self.setup, self.setup.camera_model
        screen = self.page.screen()
        if model is None or setup.camera_pose is None or screen is None or frame.shape[1::-1] != (model.width, model.height):
            return
        if not screen["covers_display"]:
            return
        layout = self.page.layout()
        markers = {k: v for k, v in found.items() if k in layout.markers or k in TAG_IDS}
        on_page = [k for k in markers if k in layout.markers]
        self._tablet = {"seen": len(on_page), "state": "tracked", "tags": len(markers) - len(on_page)}
        cam = np.asarray(setup.camera_pose)
        if self.live is None:
            if len(on_page) < MIN_PLACE_MARKERS:
                self._tablet["state"] = "lost"
                return
            size = tuple(setup.screen_size_m or [x * 1e-3 for x in SCREEN_MM])
            first, rms, _ = hand_eye.screen_from_camera(markers, layout, size, model, cam)
            if rms > 3.0:
                return
            live = LiveScreen.from_map(dataclasses.replace(first, tilt_deg=TILT_DEG))
            seen = live.pose
        else:
            live = self.live
            if self._other_plane(live, markers, on_page, layout, model, cam):
                return
            seen = live.track(markers, layout, model, cam)
            if seen is None:
                self._tablet["state"] = "lost"
                self._last_look = None
                return
        self._watch_job(live, seen)
        previous, self._last_look = self._last_look, seen
        if previous is None:
            return
        probe = LiveScreen(seen, live.size_m)
        distance, turn = probe.moved(previous)
        if distance > STEADY_M or turn > STEADY_DEG:
            self._tablet["state"] = "moving"
            return
        distance, turn = live.moved(seen)
        with self.lock:  # never under a running job: it has planned on the map it started with
            if self.busy is None and len(on_page) >= MIN_PLACE_MARKERS and self._located and self.rectifier is not None:
                live.learn_tags(markers, self.rectifier.homography)
            if self.busy is None and (not self._located or distance > FOLLOW_M or turn > FOLLOW_DEG):
                if self._located and (distance > STALE_M or turn > STALE_DEG):
                    self._forget("平板挪动较大", contacts=False)
                if self.correction is None:
                    self.correction = TapCorrection(live.size_m)
                live.pose, self.live, setup.aspect = seen, live, layout.aspect
                self._apply_live(lean_free=True)
                self._located = True  # only now do the map and the picture belong to the tablet as it lies
                self.say(f"平板在新位置（移动 {distance * 1e3:.0f} mm、转 {turn:.1f}°），已跟上" + (f"；{self._reach}" if self._reach else ""))

    def _watch_job(self, live: LiveScreen, seen: np.ndarray) -> None:
        """Stop a job that touches the screen when the tablet is no longer where the job has it."""
        if not self._located or self.busy not in TOUCH_JOBS:
            self._off_looks = 0
            return
        distance, turn = live.moved(seen)
        self._off_looks = self._off_looks + 1 if distance > MOVED_STOP_M or turn > MOVED_STOP_DEG else 0
        if self._off_looks >= MOVED_LOOKS and not self.stop.is_set():
            self.stop.set()
            self.say(f"⚠ 平板被挪动了（{distance * 1e3:.0f} mm、{turn:.1f}°），已停下。放稳后点“让开”，再重新开始")

    def _other_plane(self, live: LiveScreen, markers, on_page, layout, model, cam) -> bool:  # noqa: ANN001
        """The tablet now lies at another height or slant: adopt it from several looks that agree. True while the
        pictures say so, when tracking on the old plane would be wrong."""
        if len(on_page) < MIN_PLACE_MARKERS:
            self._plane_looks = []
            return False
        there, rms, ratio = hand_eye.screen_from_camera(markers, layout, live.size_m, model, cam)
        seen = LiveScreen.from_map(there).pose
        height, angle = live.off_plane(seen)
        if rms > 3.0 or ratio < 2.0 or (height < OFF_PLANE_M and angle < OFF_PLANE_DEG):
            self._plane_looks = []
            return False
        if self._plane_looks:
            distance, turn = LiveScreen(seen, live.size_m).moved(self._plane_looks[-1])
            if distance > 5 * STEADY_M or turn > 5 * STEADY_DEG:
                self._plane_looks = []
        self._plane_looks.append(seen)
        self._tablet["state"] = "moving"
        with self.lock:
            if len(self._plane_looks) >= PLANE_LOOKS and self.busy is None:
                live.pose, self._plane_looks = mean_pose(self._plane_looks), []
                live.tags = {}
                self._forget("平板换了高度或倾角")
                self.setup.aspect = layout.aspect
                self._apply_live(lean_free=True)
                self._located, self._last_look = True, None
                tilt = np.degrees(np.arccos(np.clip(-live.pose[2, 2], -1, 1)))
                self.say(f"平板现在倾斜 {tilt:.1f}°，已重新识别" + (f"；{self._reach}" if self._reach else ""))
        return True

    def _look(self, frame: np.ndarray) -> None:
        tags = detect_table_tags(frame)
        found = detect_markers(frame)
        try:
            self._follow(frame, found)
        except (TabletTapError, ValueError, np.linalg.LinAlgError) as exc:
            self._tablet = {"seen": 0, "state": "lost"}
            logger.warning("following the tablet: %s", exc)
        self._seen = {
            "markers": len(found),
            "tags": len(tags),
            "drift": tag_drift(self.tag_reference, tags) if self.tag_reference else None,
        }

    # ---- setup ---------------------------------------------------------------------------------------------------
    def _save(self, *, backup: bool = True) -> None:
        self.store.save(self.setup, backup=backup)

    def _load_view(self) -> None:
        if self.view_path.exists():
            try:
                self.rectifier = ScreenRectifier.load(self.view_path)
            except (ValueError, KeyError, OSError) as exc:
                logger.warning("cannot load %s: %s", self.view_path, exc)
        tags_path = self.view_path.with_suffix(".tags.json")
        if tags_path.exists():
            self.tag_reference = json.loads(tags_path.read_text())

    def apply_settings(self, tip_len_mm: float | None = None, press_mm: float | None = None) -> None:
        if tip_len_mm is not None:
            if not 0 <= tip_len_mm <= 150:
                raise ValueError("笔尖伸出长度应在 0–150 mm 之间")
            self.setup.arm.tip_len = tip_len_mm * 1e-3
        if press_mm is not None:
            if not 0 <= press_mm <= 5:
                raise ValueError("下压深度应在 0–5 mm 之间")
            self.setup.limits["press_m"] = press_mm * 1e-3
        self._save()
        if self.teach is not None:
            self._renew_teach()
        self.say(f"参数已保存：笔尖伸出 {self.setup.arm.tip_len * 1e3:.0f} mm，下压 "
                 f"{self.setup.motion_limits().press_m * 1e3:.1f} mm")  # fmt: skip

    # ---- the arm -------------------------------------------------------------------------------------------------
    @property
    def moves(self) -> bool:
        return self.arm is not None and self.arm_mode in (ArmMode.COMMAND, ArmMode.FAKE)

    def _renew_teach(self) -> None:
        old = self.teach
        if self.arm is None:
            self.teach = None
            return
        new = teach_mod.TeachSession(
            self.arm, self.setup.pen_arm(), self.setup.motion_limits(), page=self.page,
            moves_arm=self.moves, should_stop=self.stop.is_set, realtime=self.realtime, say=self.say,
        )  # fmt: skip
        if old is not None:
            new.touches, new.park_q = old.touches, old.park_q
        self.teach = new

    def _hardware_free(self, what: str) -> None:
        """Refuse to go near the bench while another program uses it (a second opener can disturb its cameras and its
        arms), or while the hardware lock of ``tablet_tap.app`` is set."""
        require_hardware(what)
        from tablet_tap.robot.piper import other_arm_programs  # noqa: PLC0415 - looks at the process list only

        running = other_arm_programs()
        if running and not self.allow_other_programs:
            raise SetupError(
                f"不能{what}：另一个程序正在使用机械臂和相机，为了不打扰它，这里不去碰硬件。先停掉它：\n  "
                + "\n  ".join(line[:120] for line in running)
            )

    def connect(self, channel: str, mode: str) -> None:
        if not self.fake:
            self._hardware_free("连接机械臂")
        self.disconnect()
        self.fault = ""
        if self.fake:
            pen = self.setup.pen_arm()
            start = np.asarray(self.setup.park_q or (0.2, 0.9, -0.8, 0.0, 0.6, 0.0), np.float64)
            arm: Any = sim.VirtualTablet(start, pen, sim.flat_screen(), self.calib)
            arm_mode = ArmMode.FAKE
        else:
            self.setup.arm.channel = channel
            arm_mode = ArmMode.COMMAND if mode == "command" else ArmMode.READ_ONLY
            arm = open_arm(self.setup.arm, arm_mode, allow_other_programs=self.allow_other_programs)
            self._save()
        self.arm, self.arm_mode = LockedArm(arm), arm_mode
        self._renew_teach()
        names = {ArmMode.COMMAND: "可运动", ArmMode.READ_ONLY: "只读", ArmMode.FAKE: "模拟"}
        self.say(f"已连接机械臂 {channel}（{names[arm_mode]}）")

    def disconnect(self) -> None:
        if self.arm is not None:
            self.arm.close()
            self.say("已断开机械臂（手臂保持使能，停在原处）")
        self.arm, self.arm_mode, self.teach = None, None, None

    def _need_arm(self, moving: bool = True) -> LockedArm:
        if self.arm is None:
            raise SetupError("还没有连接机械臂")
        if moving and not self.moves:
            raise SetupError("现在是只读连接，不能发运动指令；要运动请断开后选“可运动”重新连接")
        return self.arm

    def press(self, key: str, step: str | None = None) -> None:
        """One key of the teaching session (calibration.teach), with the step size the page has selected."""
        self._need_arm(moving=key in MOTION_KEYS)
        assert self.teach is not None
        if key == "z":
            self._renew_teach()  # a touch made by hand since the session began must not count as this contact
            self.teach.probe_depth_m = self._descent_m()
        if step in teach_mod.STEPS:
            self.teach.press(step)
        seq, before = self.page.last_down_seq(), len(self.teach.touches)
        self.teach.press(key)
        if key in JOG_KEYS | {"z"}:
            self._note_contact(seq, self.teach.touches[before:])
        if key == "p" and self.teach.park_q is not None:
            self.setup.park_q = self.teach.park_q
            self._save()

    def _descent_m(self) -> float:
        """How far the z key may go down: to the screen as far as it is known, and the allowed press."""
        if self.arm is None or self.setup.screen_map is None or self.setup.camera_pose is None:
            return 0.03
        tip = self.setup.pen_arm().fk(self.arm.read())
        height = self.setup.screen_map.uv_of(tip[:3, 3])[2]
        if height < 0.001:
            raise SetupError("笔尖已经在屏幕面上或更低，不再下探")
        return min(height + CAMERA_DEPTH_MM * 1e-3, 0.06)

    def _note_contact(self, seq: int, probed: list[Any]) -> None:
        """A touch that came while the pen was jogged or lowered by hand corrects the map as a click does, when the
        pen really was there (a finger elsewhere on the screen must not)."""
        if self.live is None or self.arm is None:
            return
        pen = self.setup.pen_arm()
        touches = [((t.u, t.v), pen.fk(np.asarray(t.q))[:3, 3]) for t in probed]
        if not probed and self.teach is not None and self.teach.step_m <= JOG_CONTACT_STEP_M:
            downs = self.page.touches(seq, "pointerdown")
            touches = [((downs[-1]["u"], downs[-1]["v"]), pen.fk(self.arm.read())[:3, 3])] if downs else []
        for uv, tip in touches:
            gap = float(np.linalg.norm(self.setup.screen_map.point(*uv) - tip))
            if gap > CONTACT_GATE_M:
                self.say(f"屏幕报告的触点离笔尖 {gap * 1e3:.0f} mm，不像是笔点的，没有采用")
                continue
            self.live.add(uv, tip)
            self._apply_live()
            self.say(f"笔碰到屏幕 u={uv[0]:.3f} v={uv[1]:.3f}，和估计差 {gap * 1e3:.1f} mm，已修正（共 {len(self.live.contacts)} 个触点）")

    def fit(self) -> None:
        if self.teach is None:
            raise SetupError("还没有连接机械臂")
        screen = self.page.screen()
        aspect = None if screen is None else float(screen["aspect"])
        result = self.teach.fit(aspect)
        if result is None:
            raise CalibrationError(f"只有 {len(self.teach.touches)} 个触点，至少要 {teach_mod.MIN_TOUCHES} 个。")
        fitted, seed_q = result
        self.setup.screen_map, self.setup.seed_q, self.setup.aspect = fitted, seed_q, aspect
        self.setup.screen_size_m = list(fitted.size_m)
        self.setup.records["teach_touches"] = [dataclasses.asdict(t) for t in self.teach.touches]
        self.setup.records["screen_map_source"] = "touch"
        lines = procedures.fit_report(fitted, self.setup.pen_arm(), self.setup.motion_limits(), seed_q)
        self.results["fit"] = lines
        self._save()
        self.say("拟合完成，已保存。\n" + "\n".join(lines))

    @property
    def from_camera(self) -> bool:
        """The screen plane is the camera's estimate and has not been probed yet."""
        return self.setup.records.get("screen_map_source") == CAMERA_SOURCE

    def _robot(self, limits: Any = None) -> TabletRobot:
        arm = self._need_arm()
        setup = self.setup
        setup.require("screen_map", "park_q", "seed_q", hint="先记录停靠位姿、点几个触点并拟合。")
        try:
            return TabletRobot(arm, setup.pen_arm(), setup.screen_map, np.asarray(setup.park_q), np.asarray(setup.seed_q),
                               limits or setup.motion_limits(), should_stop=self.stop.is_set, realtime=self.realtime,
                               aim=self._aim)  # fmt: skip
        except Unreachable as exc:
            raise Unreachable(
                f"屏幕中心上方的悬停位姿机械臂够不到（{exc}）：把平板往拿笔的机械臂挪近、放低。"
            ) from None

    def touching_robot(self) -> TabletRobot:
        """The robot for gestures that press the screen: not on a plane that is only the camera's estimate."""
        if self.from_camera:
            raise SetupError("屏幕平面还只是相机估计的，可能差几毫米：先做一次“自动探触并重新拟合”，再点按。")
        return self._robot()

    _touching_robot = touching_robot  # the name the run page was first told

    def _reachable(
        self, robot: TabletRobot, points: list[tuple[float, float]], below_m: float
    ) -> list[tuple[float, float]]:
        """The points whose hover pose and deepest pose the arm reaches; the others are reported and left out, so
        that one corner out of reach does not end a whole run half way."""
        kept = []
        for u, v in points:
            try:
                q = robot.pen.solve(robot.planner.tip_pose(u, v, robot.limits.hover_m), robot.centre_q)
                robot.pen.solve(robot.planner.tip_pose(u, v, -below_m), q)
            except Unreachable:
                self.say(f"  跳过 u={u:.3f} v={v:.3f}：机械臂够不到")
            else:
                kept.append((u, v))
        return kept

    def park(self) -> None:
        arm = self._need_arm()
        self.setup.require("park_q")
        try:
            robot: TabletRobot | None = self._robot()
        except TabletTapError:
            robot = None  # not calibrated yet: the plain joint move below
        if robot is not None and robot.where is not ArmPlace.UNKNOWN:
            robot.to_park()
        else:
            limits = dataclasses.replace(self.setup.motion_limits(), joint_move_speed=PARK_SPEED)
            self._lift(arm, limits)
            q, goal = arm.read(), np.asarray(self.setup.park_q, np.float64)
            self.say(
                f"从当前位姿沿关节空间直线低速回停靠位姿（最大关节位移 {np.max(np.abs(goal - q)):.2f} rad，无碰撞检查）"
            )
            run_segments(arm, [joint_move("回停靠位", q, goal, limits, settle=True)], hz=limits.hz,
                         should_stop=self.stop.is_set, realtime=self.realtime)  # fmt: skip
        self.say("已到停靠位姿")

    def _lift(self, arm: LockedArm, limits: Any) -> None:
        """Straight up before a joint-space move, when the pen may be low over the tablet."""
        pen, q = self.setup.pen_arm(), arm.read()
        tip = pen.fk(q)
        if tip[2, 3] > 0.08:
            return
        for height in PARK_LIFTS_M:
            goal = tip.copy()
            goal[2, 3] += height
            try:
                above = pen.solve(goal, q)
            except Unreachable:
                continue
            run_segments(arm, [joint_move("抬高", q, above, limits, settle=True)], hz=limits.hz,
                         should_stop=self.stop.is_set, realtime=self.realtime)  # fmt: skip
            return

    def gesture(self, name: str, args: dict[str, Any]) -> None:
        if name == "hover":  # over an estimated plane: higher, so that the slow descent (Z) starts well clear of it
            robot = self._robot(self._hover_limits())
        else:
            robot = self.touching_robot()
        device = RobotTabletDevice(robot, None, None, nav=self.setup.nav)
        if name == "hover":
            robot.to_hover()
        elif name == "tap":
            u, v = float(args["u"]), float(args["v"])
            if not (0 <= u <= 1 and 0 <= v <= 1):
                raise ValueError("u、v 应在 0 到 1 之间")
            hold = args.get("hold_s")
            seq = self.page.last_down_seq()
            robot.tap(u, v, hold_s=None if hold in (None, "") else float(hold))
            touched = self.page.wait_down(seq, 0.3)
            where = (
                "屏幕没有报告触摸"
                if touched is None
                else f"屏幕报告 u={touched[0]:.3f} v={touched[1]:.3f}，偏差 {robot.screen.error_mm((u, v), touched):.2f} mm"
            )  # noqa: E501
            self.say(f"点按 u={u:.3f} v={v:.3f}：{where}")
        elif name == "swipe":
            robot.swipe(
                float(args["u"]),
                float(args["v"]),
                float(args["u2"]),
                float(args["v2"]),
                float(args.get("duration_s", 0.5)),
            )
        elif name == "home":
            device.home()
        elif name == "back":
            device.back()
        else:
            raise ValueError(f"未知动作 {name}")
        if args.get("park"):
            robot.to_park()

    def point(self, where: tuple[float, float], on_camera: bool, park: bool) -> None:
        """Touch what the operator clicked: ``where`` is a fraction of the rectified screen, or of the camera picture.
        While the plane is an estimate the pen goes down slowly until the screen reports the touch; afterwards it taps.
        What the screen reports is kept for the accuracy and corrects the map."""
        u, v = where
        if on_camera:
            if self.rectifier is None or self._frame is None:
                raise SetupError("先打开相机、让相机认出平板，才知道画面上的点是屏幕哪里。")
            height, width = self._frame.shape[:2]
            u, v = (float(x) for x in np.ravel(self.rectifier.cam_to_norm(np.array([[u * width, v * height]]))))
        if not (0 <= u <= 1 and 0 <= v <= 1):
            raise ValueError("点在平板屏幕外面，没有动")
        self.setup.require("screen_map", hint="相机还没有认出平板。")
        if self.setup.camera_pose is not None and (not self._located or self._tablet["state"] == "moving"):
            raise SetupError("相机还没认出平板现在的位置（标定页全屏、别挡住、放稳），没有动")
        if self.live is None:
            self.live = LiveScreen.from_map(self.setup.screen_map)
        slow = self.from_camera
        depth_mm = (NEAR_DEPTH_MM if self.live.contacts else CAMERA_DEPTH_MM) if slow else self.setup.motion_limits().press_m * 1e3
        robot = self._robot(self._hover_limits())
        if robot.where is ArmPlace.UNKNOWN:
            raise SetupError("手臂不在让开的位置：先点“让开”。")
        if not self._reachable(robot, [(u, v)], depth_mm * 1e-3):
            raise Unreachable("这个点够不到，没有动" + (f"（{self._reach}）" if self._reach else ""))
        seq, contact = self.page.last_down_seq(), None
        blocked, trace = False, []
        if slow:

            def stop_here() -> bool:
                nonlocal blocked
                q, lag = arm.read(), arm.lag()
                height = robot.screen.uv_of(robot.pen.fk(q)[:3, 3])[2] * 1e3
                trace.append([round(time.time(), 3), round(float(height), 2), round(lag, 4), arm.torque()])
                if self.page.last_down_seq() > seq:
                    return True
                blocked = lag > BLOCKED_RAD
                return blocked

            arm = self._need_arm()
            contact = robot.probe(u, v, stop_here, depth_m=depth_mm * 1e-3)
        else:
            robot.tap(u, v)  # which aims with the correction
        touched = self.page.wait_down(seq, 0.5)
        if touched is not None and not slow and self.correction is not None:
            self.correction.add(self._aim(u, v), touched)
            self.setup.records["tap_correction"] = [list(x) for x in self.correction.samples]
            self._save(backup=False)
        tap = {"u": u, "v": v, "tu": None, "tv": None, "error_mm": None, "slow": slow}
        if touched is None and blocked:
            self.say(f"要点 ({u:.3f}, {v:.3f})：笔顶住了但屏幕没有报告触摸，已停下抬起。检查笔的导电")
        elif touched is None:
            self.say(f"要点 ({u:.3f}, {v:.3f})：" + (f"慢速降到估计面以下 {depth_mm:g} mm 屏幕没有报告触摸，已抬起。先不要再点"
                                                     if slow else "点了，屏幕没有报告触摸（标定页没开着就属正常）"))  # fmt: skip
        else:
            error = robot.screen.error_mm((u, v), touched)
            tap.update(tu=touched[0], tv=touched[1], error_mm=error)
            self.say(f"要点 ({u:.3f}, {v:.3f})，点到 ({touched[0]:.3f}, {touched[1]:.3f})，偏 {error:.1f} mm")
        self.taps.append(tap)
        if not self.fake:
            record = {"when": datetime.datetime.now().isoformat(timespec="seconds"), **tap, "blocked": blocked,
                      "lean_deg": robot.screen.tilt_deg, "pose": self.live.pose.tolist(), "trace": trace}  # fmt: skip
            path = config.runs_dir() / "web_taps.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as out:
                out.write(json.dumps(record) + "\n")
        if park:
            robot.to_park()
        if contact is not None and touched is not None and not blocked:
            self.live.add(touched, contact[:3, 3])
            self._apply_live()

    def forget_contacts(self) -> None:
        """Start the correction over (another pen, a pen that slipped)."""
        if self.live is None:
            raise SetupError("相机还没有认出平板")
        self._forget("手动清空")
        self.live.contacts = []
        self.taps.clear()
        self._apply_live()
        self.say("修正已清空")

    def accuracy(self) -> dict[str, Any]:
        """Of the clicks so far. Every error is of a click made before its own touch corrected anything."""
        errors = [t["error_mm"] for t in self.taps if t["error_mm"] is not None]
        fast = [t["error_mm"] for t in self.taps if t["error_mm"] is not None and not t["slow"]]
        last = self.taps[-1] if self.taps else None
        return {"n": len(self.taps), "registered": len(errors), "last": last,
                "fast_n": len(fast), "fast_mean": float(np.mean(fast)) if fast else None,
                "fast_max": float(np.max(fast)) if fast else None, "contacts": 0 if self.live is None else len(self.live.contacts),
                "correction": "" if self.correction is None else self.correction.describe(),
                "affine": bool(self.live is not None and self.live.affine)}  # fmt: skip

    def grip(self, close: bool) -> None:
        """The gripper around the pen holder: squeeze, or let go."""
        arm = self._need_arm()
        if not hasattr(arm, "set_grip_width"):
            self.say("这只手臂没有夹爪")
            return
        arm.set_grip_width(0.0 if close else GRIP_OPEN_M)
        q = arm.read()
        for _ in range(int(GRIP_TIME_S * self.setup.motion_limits().hz)):  # the target goes out with the joints' own
            if self.stop.is_set():
                raise Stopped("夹爪动作已停止")
            arm.command(q)
            time.sleep(1.0 / self.setup.motion_limits().hz if self.realtime else 0.0)
        arm.read()
        width = float(getattr(arm, "gripper_width_m", float("nan"))) * 1e3
        self._forget("夹爪动过，笔的位置可能变了")
        if self.live is not None:
            self._apply_live()
        if not close:
            self.say(f"夹爪已松开（{width:.0f} mm）")
        elif np.isfinite(width) and width < 8:
            self.say(f"⚠ 夹爪合到 {width:.0f} mm，没有夹到笔夹")
        else:
            self.say(f"夹爪已夹紧，夹住 {width:.1f} mm")

    def calibrate_taps(self) -> None:
        """Everything to make taps exact, in one go: slow touches until the plane is known, a grid of taps to learn what is
        left of the error, then taps at other places to measure what remains. The measured points are not learned from."""
        if self.setup.camera_pose is not None and (not self._located or self._tablet["state"] == "moving"):
            raise SetupError("相机还没认出平板现在的位置（标定页全屏、别挡住、放稳），没有动")
        if self.live is None:
            self.setup.require("screen_map", hint="相机还没有认出平板。")
            self.live = LiveScreen.from_map(self.setup.screen_map)
        if self.correction is None:
            self.correction = TapCorrection(self.live.size_m)
        self.page.clear()
        if not self.live.affine:
            self._slow_touches()
        robot = self._robot()
        if robot.where is ArmPlace.UNKNOWN:
            raise SetupError("手臂不在让开的位置：先点“让开”。")
        press = robot.limits.press_m
        grid = self._reachable(robot, procedures.grid_points(*FAST_GRID, 0.08), press)
        learned = self._tap_all(robot, grid, "学习误差")
        for tap in learned:
            self.correction.add(tap["aimed"], (tap["tu"], tap["tv"]))
        self.setup.records["tap_correction"] = [list(x) for x in self.correction.samples]
        rng = np.random.default_rng(int(time.time()))
        fresh = [(float(u), float(v)) for u, v in rng.uniform(0.08, 0.92, (TEST_TAPS, 2))]
        tested = self._tap_all(robot, self._reachable(robot, fresh, press), "检验精度")
        robot.to_park()
        before = np.array([t["error_mm"] for t in learned])
        errors = np.array([t["error_mm"] for t in tested])
        self._save()
        summary = {"n": len(errors), "mean_mm": float(errors.mean()), "median_mm": float(np.median(errors)),
                   "p95_mm": float(np.percentile(errors, 95)), "max_mm": float(errors.max()),
                   "within_1mm": int((errors <= 1.0).sum()), "before_mean_mm": float(before.mean()),
                   "lean_deg": robot.screen.tilt_deg, "model": self.correction.describe()}  # fmt: skip
        self.results["calibration"] = summary
        path = config.runs_dir() / f"tapcal_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"summary": summary, "learned": learned, "tested": tested, "pose": self.live.pose.tolist()}, indent=1))
        self.say(f"校准完成：检验 {len(errors)} 个没学过的点，平均 {summary['mean_mm']:.2f} mm，95% {summary['p95_mm']:.2f}，"
                 f"最大 {summary['max_mm']:.2f}，≤1 mm 的 {summary['within_1mm']}/{len(errors)}（校准前平均 {summary['before_mean_mm']:.2f}）。"
                 f"{summary['model']}。记录 {path}")  # fmt: skip

    def _slow_touches(self) -> None:
        robot = self._robot(self._hover_limits())
        if robot.where is ArmPlace.UNKNOWN:
            raise SetupError("手臂不在让开的位置：先点“让开”。")
        missed = 0
        points = [(0.5, 0.5), *[p for p in procedures.grid_points(*SLOW_GRID, 0.12) if abs(p[0] - 0.5) + abs(p[1] - 0.5) > 0.1]]
        for i, (u, v) in enumerate(points, 1):
            self.progress = {"job": "慢速找屏幕", "i": i, "n": len(points)}
            depth = (NEAR_DEPTH_MM if self.live.contacts else CAMERA_DEPTH_MM) * 1e-3
            robot = self._robot(self._hover_limits())
            if not self._reachable(robot, [(u, v)], depth):
                continue
            seq, arm, blocked = self.page.last_down_seq(), self._need_arm(), []

            def stop_here(seq=seq, arm=arm, blocked=blocked) -> bool:  # noqa: ANN001
                if self.page.last_down_seq() > seq:
                    return True
                if arm.lag() > BLOCKED_RAD:
                    blocked.append(True)
                    return True
                return False

            contact = robot.probe(u, v, stop_here, depth_m=depth)
            touched = None if contact is None or blocked else self.page.wait_down(seq, 0.5)
            if touched is None:
                missed += 1
                self.say(f"  慢速 ({u:.2f}, {v:.2f})：" + ("顶住了，" if blocked else "") + "屏幕没有报告触摸")
                if missed >= MAX_MISSES:
                    robot.to_park()
                    raise CalibrationError("连续碰不到或不触发，已停止：检查笔的导电（笔杆接导线）和平板有没有动过。")
                continue
            missed = 0
            self.live.add(touched, contact[:3, 3])
            self._apply_live(save=False)
        robot.to_park()
        if not self.live.affine:
            raise CalibrationError("有效触点不够（至少 4 个，四处散开），没有继续。")

    def _tap_all(self, robot: TabletRobot, points: list[tuple[float, float]], job: str) -> list[dict[str, Any]]:
        done, missed = [], 0
        for i, (u, v) in enumerate(points, 1):
            self.progress = {"job": job, "i": i, "n": len(points)}
            aimed, seq = self._aim(u, v), self.page.last_down_seq()
            robot.tap(u, v)
            touched = self.page.wait_down(seq, 0.5)
            if touched is None:
                missed += 1
                self.say(f"  {job} ({u:.2f}, {v:.2f})：屏幕没有报告触摸")
                if missed >= MAX_MISSES:
                    robot.to_park()
                    raise CalibrationError("连续两次点了没有触发，已停止：检查笔的导电（笔杆接导线）。")
                continue
            missed = 0
            error = robot.screen.error_mm((u, v), touched)
            tap = {"u": u, "v": v, "aimed": aimed, "tu": touched[0], "tv": touched[1], "error_mm": error, "slow": False}
            done.append(tap)
            self.taps.append(tap)
        if len(done) < 4:
            raise CalibrationError("有效点按太少，没有继续。")
        return done

    def start(self, channel: str) -> None:
        """Arm and top camera in one go."""
        if self.arm is None or not self.moves:
            self.connect(channel, "command")
        if self.camera is None:
            self.open_camera(self.setup.camera or config.BENCH_TOP_CAMERA)

    # ---- procedures ----------------------------------------------------------------------------------------------
    def probe(self, nu: int, nv: int, margin: float, depth_mm: float) -> None:
        from_camera = self.from_camera
        limits = self.setup.motion_limits()
        if from_camera:
            depth_mm = CAMERA_DEPTH_MM
            limits = dataclasses.replace(limits, hover_m=max(limits.hover_m, CAMERA_HOVER_M))
            self.say(f"屏幕平面来自相机估计：悬停高度提到 {limits.hover_m * 1e3:.0f} mm，探触深度 {depth_mm:g} mm。")
        robot = self._robot(limits)
        wanted = procedures.grid_points(nu, nv, margin)
        points = self._reachable(robot, wanted, depth_mm * 1e-3)
        if len(points) < procedures.MIN_PROBE_POINTS:
            raise CalibrationError(f"{len(wanted)} 个探触点里机械臂只够得到 {len(points)} 个；把平板往机械臂挪近再试。")
        self.say(f"自动探触 {len(points)} 个点……")

        def heard(index: int, uv: tuple[float, float], point: procedures.ProbePoint | Unreachable | None) -> None:
            self.progress = {"job": "自动探触", "i": index, "n": len(points)}
            if isinstance(point, Unreachable):
                self.say(f"  {index:2d}/{len(points)}  u={uv[0]:.3f} v={uv[1]:.3f}  机械臂够不到，跳过")
            elif point is None:
                self.say(f"  {index:2d}/{len(points)}  u={uv[0]:.3f} v={uv[1]:.3f}  屏幕没有报告触摸")
            else:
                self.say(f"  {index:2d}/{len(points)}  u={uv[0]:.3f} v={uv[1]:.3f}  落点偏差 {point.error_mm:.2f} mm，"
                         f"接触面比原平面高 {point.height_mm:+.2f} mm")  # fmt: skip

        found = procedures.probe_grid(robot, self.page, points, depth_mm * 1e-3, on_point=heard)
        self.page.set_mode("markers")
        if found:
            errors = np.array([p.error_mm for p in found])
            self.say(
                f"原标定：{len(found)} / {len(points)} 个点触发，落点偏差 平均 {errors.mean():.2f} mm，最大 {errors.max():.2f} mm"
            )
        fitted = procedures.refit(found, self.setup.aspect, robot.screen.tilt_deg)
        self.setup.screen_map, self.setup.screen_size_m = fitted, list(fitted.size_m)
        self.setup.records["probe_points"] = [dataclasses.asdict(p) for p in found]
        self.setup.records["screen_map_source"] = "touch"
        if self.live is not None:
            self.live.contacts = []
            for p in found:
                self.live.add((p.u, p.v), np.array(p.pose)[:3, 3])
            self._apply_live(save=False)
        lines = procedures.fit_report(fitted, self.setup.pen_arm(), self.setup.motion_limits(), self.setup.seed_q)
        self.results["fit"] = lines
        self.results["probe"] = {"n": len(points), "registered": len(found)}
        self._save()
        self.say("已用探触结果重新拟合并保存。\n" + "\n".join(lines))

    def taptest(self, nu: int, nv: int, margin: float, repeat: int) -> None:
        robot = self.touching_robot()
        points = self._reachable(robot, procedures.grid_points(nu, nv, margin), robot.limits.press_m) * repeat
        if not points:
            raise CalibrationError("测试点机械臂一个都够不到；把平板往机械臂挪近再试。")
        self.say(f"点按精度测试 {len(points)} 次……")
        self.results["taptest"] = {"summary": None, "points": []}

        def heard(index: int, outcome: procedures.TapOutcome) -> None:
            self.progress = {"job": "点按测试", "i": index, "n": len(points)}
            self.results["taptest"]["points"].append(dataclasses.asdict(outcome))
            text = "没有触发" if outcome.error_mm is None else f"偏差 {outcome.error_mm:.2f} mm"
            self.say(f"  {index:3d}/{len(points)}  u={outcome.u:.3f} v={outcome.v:.3f}  {text}")

        outcomes = procedures.tap_test(robot, self.page, points, on_point=heard)
        summary = procedures.summarize_taps(outcomes)
        self.results["taptest"]["summary"] = summary
        out = config.runs_dir() / f"taptest_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"summary": summary, "limits": dataclasses.asdict(robot.limits),
                                   "points": [dataclasses.asdict(o) for o in outcomes]}, indent=2))  # fmt: skip
        text = f"触发 {summary['registered']} / {summary['n']} 次"
        if summary["registered"]:
            text += (f"；偏差 平均 {summary['mean_mm']:.2f}，中位 {summary['median_mm']:.2f}，95 分位 {summary['p95_mm']:.2f}，"
                     f"最大 {summary['max_mm']:.2f} mm；≤ 2 mm 的有 {summary['within_2mm']} / {summary['registered']} 次")  # fmt: skip
        self.say(text + f"\n结果已保存 {out}")

    # ---- the camera ----------------------------------------------------------------------------------------------
    def open_camera(self, spec: str) -> None:
        if not self.fake and not spec.startswith("file:"):
            self._hardware_free("打开相机")
        self.close_camera()
        self.camera = sim.SyntheticCamera(self.calib) if self.fake else open_camera(spec)
        self.setup.camera = spec
        self._save()
        self.say(f"相机已打开：{spec}")

    def close_camera(self) -> None:
        camera, self.camera = self.camera, None
        if camera is not None:
            time.sleep(2 * CAMERA_PERIOD_S)  # the watcher's read in flight
            camera.close()
            self.say("相机已释放")
        self._frame = None

    def fresh_frame(self, timeout_s: float = 3.0) -> np.ndarray:
        """A camera frame taken after this call. The watcher thread is the only reader of the camera: everything else
        (the camera calibration, the run page's screenshots) takes its frames from here."""
        if self.camera is None:
            raise SetupError("还没有打开相机")
        asked = time.monotonic()
        deadline = asked + timeout_s
        while self._frame is None or self._frame_time < asked:
            if time.monotonic() > deadline:
                raise CalibrationError(f"相机没有出图：{self._camera_error or '超时'}")
            time.sleep(0.02)
        return self._frame

    def calibrate_view(self, out_width: int) -> None:
        if self.camera is None:
            raise SetupError("还没有打开相机")
        self.page.set_mode("markers")
        self.page.clear()
        time.sleep(0.8 if self.realtime else 0.0)  # the page redraws
        frame = self.fresh_frame()
        rectifier = ScreenRectifier.from_marker_image(frame, self.page.layout(), out_width=out_width)
        self.view_path.parent.mkdir(parents=True, exist_ok=True)
        rectifier.save(self.view_path)
        cv2.imwrite(str(self.view_path.with_suffix(".rectified.png")), rectifier.rectify(frame))
        tags = [{"id": t["id"], "corners": t["corners"].tolist()} for t in detect_table_tags(frame)]
        self.view_path.with_suffix(".tags.json").write_text(json.dumps(tags))
        self.rectifier, self.tag_reference = rectifier, tags
        self._jpeg_cache.clear()
        quad = rectifier.screen_quad_cam()
        width_px = float(np.linalg.norm(quad[1] - quad[0]))
        self.results["view"] = {
            "reprojection_px": float(rectifier.reprojection_px),
            "screen_px": width_px,
            "tags": len(tags),
        }
        self.say(f"相机标定完成：重投影误差 {rectifier.reprojection_px:.2f} px，屏幕在画面里宽 {width_px:.0f} px，"
                 f"记下了桌面上 {len(tags)} 个码的位置作参照。")  # fmt: skip
        if width_px < MIN_SCREEN_PX:
            self.say(f"⚠ 屏幕在画面里不到 {MIN_SCREEN_PX} px 宽：标定没问题，但模型看小图标和文字会吃力。")

    def place(self, width_mm: float, height_mm: float, tilt_deg: float) -> None:
        """Where the screen is, from the calibration page's markers and the camera's pose (cli ``place``); the probe
        refines it."""
        setup = self.setup
        setup.require("camera_pose", "camera_model", hint="先做相机手眼标定。")
        self.page.set_mode("markers")
        self.page.clear()
        time.sleep(0.8 if self.realtime else 0.0)
        frame = self.fresh_frame()
        model = setup.camera_model
        if frame.shape[1::-1] != (model.width, model.height):
            raise CalibrationError(f"相机要用 {model.width}×{model.height}（手眼标定时的分辨率）。")
        layout = self.page.layout()
        markers = {k: v for k, v in detect_markers(frame).items() if k in layout.markers}
        if len(markers) < MIN_PLACE_MARKERS:
            raise CalibrationError(f"只认出 {len(markers)} 个标定码：把手臂移开、平板进全屏后再试。")
        screen, rms, ratio = hand_eye.screen_from_camera(
            markers, layout, (width_mm * 1e-3, height_mm * 1e-3), model, np.asarray(setup.camera_pose)
        )
        screen = dataclasses.replace(screen, tilt_deg=tilt_deg)
        limits = setup.motion_limits()
        limits = dataclasses.replace(limits, hover_m=max(limits.hover_m, CAMERA_HOVER_M))
        seed = hand_eye.seed_joints(setup.pen_arm(), screen, limits)
        contacts = [] if self.live is None else self.live.contacts
        self.live = LiveScreen.from_map(screen)
        self.live.contacts = contacts if self.live.size_m == tuple(setup.screen_size_m or ()) else []
        setup.screen_map, setup.seed_q, setup.aspect = screen, seed.tolist(), layout.aspect
        self._apply_live(save=False)
        lines = procedures.fit_report(screen, setup.pen_arm(), limits, setup.seed_q)
        self._save()
        self.say(f"相机估计屏幕位置：{len(markers)} 个码，重投影 {rms:.2f} px，倾角判别比 {ratio:.1f}，笔倾 {tilt_deg:g}°\n"
                 + "\n".join(lines) + "\n高度可能差 1 cm：先“到屏幕上方”，再按 Z 慢速下探一个点。")  # fmt: skip

    def _best_place(self) -> np.ndarray | None:
        """(4, 2) where in the camera picture the lit screen should lie for the pen to stand upright."""
        setup = self.setup
        if setup.camera_model is None or setup.camera_pose is None or self.live is None:
            return None
        corners = self.live.corners()  # as it is turned and slanted now, only moved
        corners = corners + np.array([*(np.array(BEST_CENTRE_M) - corners.mean(axis=0)[:2]), 0.0])
        inverse = np.linalg.inv(np.asarray(setup.camera_pose))
        return setup.camera_model.project(corners @ inverse[:3, :3].T + inverse[:3, 3])

    def jpeg(self, kind: str, width: int = 960) -> bytes | None:
        """The newest camera frame with what was detected drawn on it (``camera``), or the rectified screen."""
        frame, stamp = self._frame, self._frame_time
        if frame is None:
            return None
        cached = self._jpeg_cache.get(kind)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        if kind == "screen":
            if self.rectifier is None:
                return None
            image = self.rectifier.rectify(frame)
        else:
            image = frame.copy()
            thick = max(2, image.shape[1] // 600)
            for corners in detect_markers(image).values():
                cv2.polylines(image, [corners.astype(np.int32)], True, (0, 220, 0), thick)
            for tag in detect_table_tags(image):
                cv2.polylines(image, [tag["corners"].astype(np.int32)], True, (255, 160, 0), thick)
                cv2.putText(image, str(tag["id"]), tuple(tag["corners"][0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX,
                            image.shape[1] / 1100, (0, 0, 255), thick)  # fmt: skip
            if self.rectifier is not None:
                cv2.polylines(image, [self.rectifier.screen_quad_cam().astype(np.int32)], True, (255, 0, 255), thick)
            target = self._best_place() if self._reach else None
            if target is not None:
                cv2.polylines(image, [target.astype(np.int32)], True, (0, 215, 255), 2 * thick)
                cv2.putText(image, "PUT TABLET HERE", tuple((target.mean(0) - (150, 0)).astype(int)), cv2.FONT_HERSHEY_SIMPLEX,
                            image.shape[1] / 1700, (0, 215, 255), thick)  # fmt: skip
        scale = width / image.shape[1]
        if scale < 1:
            image = cv2.resize(image, (width, round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return None
        self._jpeg_cache[kind] = (stamp, data.tobytes())
        return data.tobytes()

    # ---- what the page shows -----------------------------------------------------------------------------------------
    def state(self, log_since: int = 0) -> dict[str, Any]:
        setup = self.setup
        arm: dict[str, Any] = {"connected": self.arm is not None, "mode": None if self.arm_mode is None else self.arm_mode.value,
                               "channel": setup.arm.channel, "moves": self.moves}  # fmt: skip
        place = None
        if self.arm is not None:
            try:
                q = self.arm.read()
            except Exception as exc:  # noqa: BLE001 - a fault of the arm is shown, not raised
                self.fault = f"{type(exc).__name__}: {exc}"
            else:
                tip = setup.pen_arm().fk(q)
                arm.update(q=np.round(q, 4).tolist(), tip_mm=np.round(tip[:3, 3] * 1e3, 1).tolist(),
                           axis=np.round(tip[:3, 2], 3).tolist())  # fmt: skip
                if setup.screen_map is not None:
                    u, v, h = setup.screen_map.uv_of(tip[:3, 3])
                    place = {"u": round(u, 4), "v": round(v, 4), "height_mm": round(h * 1e3, 1), "where": "unknown"}
                    if setup.park_q is not None and float(np.max(np.abs(q - np.asarray(setup.park_q)))) < 0.06:
                        place["where"] = "park"
                    else:
                        try:
                            TapPlanner(setup.pen_arm(), setup.screen_map, setup.motion_limits()).hover_pose_of(q)
                            place["where"] = "hover"
                        except ValueError:
                            pass
                elif setup.park_q is not None and float(np.max(np.abs(q - np.asarray(setup.park_q)))) < 0.06:
                    place = {"where": "park"}
        screen_map = None
        if setup.screen_map is not None:
            width, height = setup.screen_map.size_m
            screen_map = {"width_mm": round(width * 1e3, 1), "height_mm": round(height * 1e3, 1),
                          "tilt_deg": round(float(np.degrees(np.arccos(np.clip(setup.screen_map.normal[2], -1, 1)))), 1),
                          "residual_mm": float(setup.screen_map.residual_mm) if np.isfinite(setup.screen_map.residual_mm) else None,
                          "n_points": setup.screen_map.n_points,
                          "source": setup.records.get("screen_map_source")}  # fmt: skip
        teach = self.teach
        with self.lock:
            log = [line for line in self._log if line["i"] > log_since]
        return {
            "fake": self.fake,
            "run_page": importlib.util.find_spec("tablet_tap.web.run_page") is not None,
            "busy": self.busy,
            "progress": self.progress,
            "stopping": self.stop.is_set() and self.busy is not None,
            "fault": self.fault,
            "arm": arm,
            "place": place,
            "touches": [] if teach is None else [{"u": round(t.u, 4), "v": round(t.v, 4)} for t in teach.touches],
            "setup": {
                "tip_len_mm": round(setup.arm.tip_len * 1e3, 1),
                "press_mm": round(setup.motion_limits().press_m * 1e3, 2),
                "park_set": setup.park_q is not None,
                "screen_map": screen_map,
                "camera": setup.camera,
            },  # fmt: skip
            "camera": {
                "open": self.camera is not None,
                "error": self._camera_error,
                "has_frame": self._frame is not None,
                "size": None if self._frame is None else [self._frame.shape[1], self._frame.shape[0]],
                "calibrated": self.rectifier is not None,
                **self._seen,
            },  # fmt: skip
            "page": self.calib.snapshot(),
            "accuracy": self.accuracy(),
            "gripper_mm": None if self.arm is None else self._gripper_mm(),
            "https_port": self.https_port,
            "tablet": {**self._tablet, "reach": self._reach},
            "results": self.results,
            "log": log,
        }

    def _gripper_mm(self) -> float | None:
        width = getattr(self.arm, "gripper_width_m", None)
        return None if width is None or not np.isfinite(width) else round(float(width) * 1e3, 1)

    def close(self) -> None:
        self._closing.set()
        self.stop.set()
        if self.camera is not None:
            self.camera.close()


# ---- HTTP ----------------------------------------------------------------------------------------------------------
JOB_NAMES = {"connect": "连接", "disconnect": "断开", "key": "点动", "fit": "拟合", "park": "回停靠位", "hover": "到屏幕上方",
             "tap": "点按", "swipe": "滑动", "home": "回桌面", "back": "返回", "probe": "自动探触", "taptest": "点按测试",
             "camera_open": "打开相机", "camera_close": "释放相机", "view": "相机标定", "place": "相机估计屏幕位置", "point": "点击位置",
             "forget_contacts": "清空修正", "start": "连接",
             "grip": "夹爪", "calibrate_taps": "一键校准", "set": "保存参数"}  # fmt: skip


def _job(console: Console, cmd: str, args: dict[str, Any]) -> Callable[[], None]:
    """The job for a command of the page; ValueError (Chinese) when its arguments are not usable."""

    def number(name: str, default: float, low: float, high: float) -> float:
        value = float(args.get(name, default))
        if not low <= value <= high:
            raise ValueError(f"{name} 应在 {low:g} 到 {high:g} 之间")
        return value

    if cmd == "connect":
        channel, mode = str(args.get("channel", "")), str(args.get("mode", "read_only"))
        if not console.fake and channel not in ("can_l_follower", "can_r_follower"):
            raise ValueError("请选择拿笔的机械臂（左从臂或右从臂）")
        if mode not in ("command", "read_only"):
            raise ValueError("连接方式应为 command 或 read_only")
        if mode == "command" and not console.fake and not args.get("attended"):
            raise ValueError("要让机械臂运动，先勾选“有人在急停旁边”")
        return lambda: console.connect(channel or "模拟", mode)
    if cmd == "disconnect":
        return console.disconnect
    if cmd == "key":
        key, step = str(args.get("key", "")), args.get("step")
        if key not in MOTION_KEYS | {"p", "x"}:
            raise ValueError(f"不认识的按键 {key!r}")
        return lambda: console.press(key, None if step is None else str(step))
    if cmd == "fit":
        return console.fit
    if cmd == "park":
        return console.park
    if cmd in ("hover", "tap", "swipe", "home", "back"):
        return lambda: console.gesture(cmd, args)
    if cmd == "probe":
        nu, nv = int(number("nu", 5, 2, 9)), int(number("nv", 4, 2, 9))
        margin, depth = number("margin", 0.08, 0.02, 0.4), number("depth_mm", 6.0, 1.0, 20.0)
        return lambda: console.probe(nu, nv, margin, depth)
    if cmd == "taptest":
        nu, nv = int(number("nu", 5, 2, 9)), int(number("nv", 4, 2, 9))
        margin, repeat = number("margin", 0.08, 0.02, 0.4), int(number("repeat", 1, 1, 10))
        return lambda: console.taptest(nu, nv, margin, repeat)
    if cmd == "camera_open":
        spec = str(args.get("spec", "")).strip()
        if not spec and not console.fake:
            raise ValueError("请填相机，例如 realsense:262422077281:1920x1080")
        return lambda: console.open_camera(spec or "模拟相机")
    if cmd == "camera_close":
        return console.close_camera
    if cmd == "view":
        out_width = int(number("out_width", 1792, 640, 2560))
        return lambda: console.calibrate_view(out_width)
    if cmd == "point":
        x, y = number("x", -1.0, 0.0, 1.0), number("y", -1.0, 0.0, 1.0)
        on_camera, park = args.get("on") == "camera", bool(args.get("park", True))
        return lambda: console.point((x, y), on_camera, park)
    if cmd == "grip":
        return lambda: console.grip(bool(args.get("close", True)))
    if cmd == "calibrate_taps":
        return console.calibrate_taps
    if cmd == "forget_contacts":
        return console.forget_contacts
    if cmd == "start":
        channel = str(args.get("channel", "can_l_follower"))
        if not console.fake and channel not in ("can_l_follower", "can_r_follower"):
            raise ValueError("请选择拿笔的机械臂（左从臂或右从臂）")
        if not console.fake and not args.get("attended"):
            raise ValueError("先勾选“有人能断电”")
        return lambda: console.start(channel)
    if cmd == "place":
        width, height = number("width_mm", 236.9, 50, 500), number("height_mm", 148.1, 30, 400)
        tilt = number("tilt_deg", 15.0, 0, 25)
        return lambda: console.place(width, height, tilt)
    if cmd == "set":
        tip = None if args.get("tip_len_mm") in (None, "") else float(args["tip_len_mm"])
        press = None if args.get("press_mm") in (None, "") else float(args["press_mm"])
        return lambda: console.apply_settings(tip, press)
    raise ValueError(f"未知命令 {cmd!r}")


def register(app: flask.Flask, console: Console, token: str = "") -> None:
    """Add the console's routes to the calibration page's app."""

    def refused() -> Any:
        if token and flask.request.args.get("token", "") != token and flask.request.headers.get("X-Token", "") != token:
            return flask.jsonify({"error": "需要口令（网址后加 ?token=…）"}), 401
        return None

    @app.get("/console")
    def console_page() -> Any:
        return refused() or flask.send_from_directory(STATIC_DIR, "tap.html")

    @app.get("/console/full")
    def full_console_page() -> Any:
        return refused() or flask.send_from_directory(STATIC_DIR, "console.html")

    @app.get("/api/state")
    def api_state() -> Any:
        return refused() or flask.jsonify(console.state(int(flask.request.args.get("log_since", 0))))

    @app.post("/api/stop")
    def api_stop() -> Any:
        denied = refused()
        if denied is not None:
            return denied
        console.stop.set()
        console.say("■ 收到停止：机械臂保持在当前位置")
        return flask.jsonify({"ok": True})

    @app.post("/api/cmd")
    def api_cmd() -> Any:
        denied = refused()
        if denied is not None:
            return denied
        if not flask.request.is_json:  # a form on some other web page cannot send JSON without asking first
            return flask.jsonify({"error": "请求必须是 JSON"}), 415
        args = flask.request.get_json(silent=True)
        if not isinstance(args, dict) or "cmd" not in args:
            return flask.jsonify({"error": "缺少 cmd"}), 400
        cmd = str(args["cmd"])
        if cmd == "page_mode":
            try:
                console.page.set_mode(str(args.get("mode", "markers")), args.get("target"))
            except ValueError as exc:
                return flask.jsonify({"error": str(exc)}), 400
            return flask.jsonify({"ok": True})
        try:
            job = _job(console, cmd, args)
        except (ValueError, TypeError, KeyError) as exc:
            return flask.jsonify({"error": str(exc)}), 400
        if not console.submit(JOB_NAMES.get(cmd, cmd), job):
            return flask.jsonify({"error": f"正在执行“{console.busy}”，这条指令没有执行（不排队）"}), 409
        return flask.jsonify({"ok": True}), 202

    @app.get("/api/<kind>.jpg")
    def api_image(kind: str) -> Any:
        denied = refused()
        if denied is not None:
            return denied
        if kind not in ("camera", "screen"):
            return flask.jsonify({"error": "没有这张图"}), 404
        data = console.jpeg(kind, int(flask.request.args.get("w", 960)))
        if data is None:
            return flask.jsonify({"error": "还没有画面"}), 503
        return flask.Response(data, mimetype="image/jpeg")


def create_app(console: Console, token: str = "") -> flask.Flask:
    app = page_server.create_app(console.calib)
    register(app, console, token)
    if (
        importlib.util.find_spec("tablet_tap.web.run_page") is not None
    ):  # the run page (instruction -> agent), when present
        from tablet_tap.web import run_page  # noqa: PLC0415

        run_page.register(app, console, token)
    return app


def ensure_certificate(folder: pathlib.Path, hosts: list[str]) -> tuple[pathlib.Path, pathlib.Path] | None:
    """A self-signed certificate and its key for the console's https port, made once with openssl; None when that fails.
    A browser lets a page use the microphone only on https or localhost, and the operator's browser is another machine."""
    cert, key = folder / "console_cert.pem", folder / "console_key.pem"
    if cert.exists() and key.exists():
        return cert, key
    names = ["DNS:localhost", "IP:127.0.0.1", *(f"IP:{host}" for host in hosts)]
    folder.mkdir(parents=True, exist_ok=True)
    command = ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "825", "-subj", "/CN=tablet-tap console",
               "-addext", "subjectAltName=" + ",".join(names), "-keyout", str(key), "-out", str(cert)]  # fmt: skip
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("could not make the https certificate: %s", exc)
        return None
    key.chmod(0o600)
    return cert, key


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="标定控制台（网页）：平板打开 /calib，操作的人打开 /console")
    parser.add_argument("--https_port", type=int, default=0, help="再开一个 https 端口（浏览器里用麦克风说指令需要），例如 8443")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8801)
    parser.add_argument("--setup", default=str(config.DEFAULT_SETUP_PATH), help="标定和配置文件")
    parser.add_argument("--view", default=str(config.DEFAULT_VIEW_PATH), help="相机到屏幕的标定文件")
    parser.add_argument("--token", default="", help="口令；设了以后控制台的网址要带 ?token=…")
    parser.add_argument("--fake", action="store_true", help="模拟模式：软件机械臂 + 虚拟平板 + 合成相机，不连硬件")
    parser.add_argument("--force", action="store_true", help="即使 run_real 等程序在运行也允许连接机械臂（不要常用）")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("werkzeug").setLevel(logging.WARNING)  # the page polls several times a second
    from werkzeug.serving import make_server  # noqa: PLC0415

    console = Console(config.SetupStore(args.setup), pathlib.Path(args.view), page_server.CalibState(), fake=args.fake,
                      allow_other_programs=args.force)  # fmt: skip
    app = create_app(console, args.token)
    server = make_server(args.host, args.port, app, threaded=True)
    if args.https_port:
        pair = ensure_certificate(config.PROJECT_ROOT / "calib", page_server.local_addresses())
        if pair is None:
            print("没有做出 https 证书（需要 openssl）：浏览器里的麦克风用不了，其它不受影响。")
        else:
            secure = make_server(args.host, args.https_port, app, threaded=True, ssl_context=(str(pair[0]), str(pair[1])))
            threading.Thread(target=secure.serve_forever, daemon=True, name="console-https").start()
            console.https_port = args.https_port
            print(f"  要用麦克风说指令：https://<这台电脑的地址>:{args.https_port}/console（证书是自签的，浏览器里点“继续访问”）")

    def leave(signum: int, frame: Any) -> None:  # so that the log says why the console went away
        logger.warning("收到信号 %s，控制台退出", signal.Signals(signum).name)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, leave)
    faulthandler.enable()  # a crash inside a camera or CAN library leaves its stack in the log
    query = f"?token={args.token}" if args.token else ""
    print(f"标定控制台已启动（端口 {args.port}）。")
    print(f"  操作的人打开： http://<这台电脑的地址>:{args.port}/console{query}")
    print(f"  平板浏览器打开：http://<这台电脑的地址>:{args.port}/calib ，并进入全屏")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        console.close()
        logger.info("控制台已退出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
