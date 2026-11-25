"""The wrist camera in the browser console: try the plan, calibrate the camera on the flange, take the look pose's
picture, and let instructions be carried out looking through it.

The steps are those of ``cli/commands/wrist.py`` (which runs them from the command line, with the arm to itself). Here
they run in the console's one worker, on the arm the console holds, so nothing else moves the arm meanwhile. The top
camera's part is the coarse one: it says where the tablet lies, well enough to plan the looks; what the model sees and
what the look pose's picture is laid over come from the wrist camera. The wrist camera keeps its own files beside the
setup (``wrist.json``, ``wrist_view.json``).
"""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import datetime
import json
import pathlib
import threading
import time
from typing import Any

import cv2
import flask
import numpy as np
from PIL import Image

from tablet_tap import config
from tablet_tap.agent.robot_device import RobotTabletDevice
from tablet_tap.calibration import hand_eye
from tablet_tap.calibration import look_search
from tablet_tap.calibration.eye_in_hand import mount_error
from tablet_tap.calibration.eye_in_hand import solve_eye_in_hand
from tablet_tap.calibration.live_screen import LiveScreen
from tablet_tap.calibration.tap_correction import TapCorrection
from tablet_tap.calibration.wrist_procedures import collect_views
from tablet_tap.calibration.wrist_procedures import look_quality
from tablet_tap.calibration.wrist_procedures import still_picture
from tablet_tap.core.look import CALIBRATION_RULES
from tablet_tap.core.look import MOUNTS
from tablet_tap.core.look import Look
from tablet_tap.core.look import LookRules
from tablet_tap.core.look import camera_in_flange
from tablet_tap.core.look import flange_pose
from tablet_tap.core.motion import joint_move
from tablet_tap.core.screen_map import ScreenMap
from tablet_tap.errors import CalibrationError
from tablet_tap.errors import PathError
from tablet_tap.errors import SetupError
from tablet_tap.errors import Unreachable
from tablet_tap.robot.executor import START_TOL_RAD
from tablet_tap.robot.executor import run_segments
from tablet_tap.robot.tablet_robot import ArmPlace
from tablet_tap.robot.wrist import MOVED_DEG
from tablet_tap.robot.wrist import MOVED_M
from tablet_tap.robot.wrist import LookRobot
from tablet_tap.robot.wrist import WristStore
from tablet_tap.vision import layout as screen_layout
from tablet_tap.vision.camera import Camera
from tablet_tap.vision.camera import camera_model
from tablet_tap.vision.camera import open_camera
from tablet_tap.vision.rectifier import ScreenRectifier
from tablet_tap.vision.rectifier import detect_markers
from tablet_tap.vision.rectifier import draw_overlay

JOBS = {"plan": "腕部相机：试算", "calibrate": "腕部相机：手眼标定", "view": "腕部相机：看屏位", "shot": "腕部相机：拍照",
        "find": "腕部相机：找平板", "view2": "腕部相机：第二个看屏位"}  # fmt: skip
WASHED_OUT = 35  # grey levels brighter than the other look shows the same place: a reflection, not the display
# Where the tablet is asked to be put when nothing has seen it yet: straight ahead of the arm's base, far enough for
# the wrist camera to back off from it, on its stand. The first look is planned for this place, and high.
ASKED_AHEAD_M, ASKED_SLANT_DEG, FIRST_LOOK_TIP_M = 0.32, 12.0, 0.08
SCREEN_M = (0.2369, 0.1481)
MIN_MARKERS = 4
SLOW_RAD_S = 0.25  # the calibration's joint moves, as slow as the park move
MOUNT_MM, MOUNT_DEG = 15.0, 6.0  # further than this from the drawing: the wrong camera, mount or joint convention
GOOD_WIDTH_PX, GOOD_EDGE_PX = 900.0, 4.0
GLARE_SHARE = 0.005  # more of the page's dark ground than this shining: a lamp is mirrored in the glass
APART_M, APART_DEG = 0.06, 12.0  # another look pose is at least this far from those tried, in place or in direction
SLANTED_LOOK_DEG = (
    40.0  # how far off the screen's normal another look may be: a lamp's reflection goes where the camera is not
)
# A tablet pushed by less than this stays inside the look pose's picture: the picture is laid over it again from how
# far the top camera saw it go. Beyond it the look pose itself has to be found again.
FOLLOW_M, FOLLOW_DEG = 0.008, 2.0
TURNED = np.diag([-1.0, -1.0, 1.0, 1.0])  # half a turn about the optical axis: the camera the other way up in its mount
BAD_FIT_PX = 8.0  # a fit worse than this from the drawing's guess: try the camera the other way up
MIN_VIEWS, MAX_VIEWS = 10, 24


def asked_screen(size_m: tuple[float, float]) -> ScreenMap:
    """The screen where the operator is asked to put it, its lower edge toward the arm."""
    slant = np.radians(ASKED_SLANT_DEG)
    right, up = np.array([0.0, -1.0, 0.0]), np.array([np.cos(slant), 0.0, np.sin(slant)])
    centre = np.array([ASKED_AHEAD_M, 0.0, 0.5 * size_m[1] * np.sin(slant) + 0.005])
    return ScreenMap(centre - 0.5 * size_m[0] * right + 0.5 * size_m[1] * up, size_m[0] * right, -size_m[1] * up, up)


def screen_of(tablet: np.ndarray, size_m: tuple[float, float]) -> ScreenMap:
    """The screen whose frame is ``tablet`` (4, 4): x right, y down, z into the glass, origin at the top-left corner."""
    rot, corner = tablet[:3, :3], tablet[:3, 3]
    centre = corner + 0.5 * size_m[0] * rot[:, 0] + 0.5 * size_m[1] * rot[:, 1]
    outward = np.array([centre[0], centre[1], 0.0])
    tool_x = outward - (outward @ rot[:, 2]) * rot[:, 2]
    return ScreenMap(corner, size_m[0] * rot[:, 0], size_m[1] * rot[:, 1], tool_x / np.linalg.norm(tool_x))


def glare_share(rectified_bgr: np.ndarray, layout: screen_layout.ScreenLayout) -> float:
    """How much of the calibration page's dark ground is bright in the rectified picture: a lamp mirrored in the
    glass. The page is black but for its markers, so whatever shines beside them is not the display."""
    height, width = rectified_bgr.shape[:2]
    drawn = cv2.cvtColor(screen_layout.render_layout(layout, width), cv2.COLOR_BGR2GRAY)
    drawn = cv2.resize(drawn, (width, height), interpolation=cv2.INTER_AREA)
    grow = max(3, width // 60) | 1  # the markers' white border, the touch dot and a little misfit are left out
    dark = cv2.dilate((drawn > 60).astype(np.uint8), np.ones((grow, grow), np.uint8)) == 0
    bright = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2GRAY) > 200
    return float(np.count_nonzero(bright & dark)) / max(int(np.count_nonzero(dark)), 1)


def merge_looks(first_bgr: np.ndarray, second_bgr: np.ndarray, fill: np.ndarray | None = None) -> np.ndarray:
    """The first rectified picture, with what a lamp's reflection has washed out in it taken from the second. A
    reflection sits elsewhere on the glass for a camera elsewhere, and it only ever adds light: where the first look is
    much the brighter of the two, the second is the one that shows the display. ``fill`` (0 to 1, the picture's size):
    where the first look does not show the screen at all (the pen is in front of it), taken from the second as well."""
    if first_bgr.shape != second_bgr.shape:
        second_bgr = cv2.resize(second_bgr, first_bgr.shape[1::-1], interpolation=cv2.INTER_LINEAR)
    grey = [
        cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), (0, 0), 2.0).astype(np.int16)
        for x in (first_bgr, second_bgr)
    ]
    washed = ((grey[0] - grey[1]) > WASHED_OUT).astype(np.uint8)
    washed = cv2.morphologyEx(washed, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # specks are noise, not a lamp
    washed = cv2.dilate(washed, np.ones((15, 15), np.uint8))
    blend = cv2.GaussianBlur(washed.astype(np.float32), (0, 0), 4.0)
    if fill is not None:
        if fill.shape != blend.shape:
            fill = cv2.resize(fill, blend.shape[::-1], interpolation=cv2.INTER_LINEAR)
        blend = np.maximum(blend, fill)
    blend = blend[..., None]
    return np.clip(first_bgr * (1.0 - blend) + second_bgr * blend, 0, 255).astype(np.uint8)


class TwoLookDevice(RobotTabletDevice):
    """The tablet seen from two look poses: every screenshot is the first look's picture mended with the second's.
    The arm is back at the first look pose, which is its park pose here, before anything is touched."""

    def __init__(
        self, *args: Any, second_q: np.ndarray, second_rectifier: ScreenRectifier, fill: np.ndarray | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.second_q, self.second_rectifier, self.fill = np.asarray(second_q, np.float64), second_rectifier, fill

    def screenshot(self) -> Image.Image:
        first = cv2.cvtColor(np.asarray(super().screenshot()), cv2.COLOR_RGB2BGR)
        self.robot.move_to(self.second_q, "去第二个看屏位")
        time.sleep(self.view_settle_s)
        frames = [self.camera.read() for _ in range(max(self.keep_frames, 1))]
        frame = np.median(np.stack(frames), axis=0).astype(np.uint8) if len(frames) > 2 else frames[-1]
        self.robot.move_to(self.robot.park_q, "回第一个看屏位")
        merged = merge_looks(first, self.second_rectifier.rectify(frame), self.fill)
        return Image.fromarray(cv2.cvtColor(merged, cv2.COLOR_BGR2RGB))


def seen_tablet(
    markers: dict[int, np.ndarray], layout: screen_layout.ScreenLayout, size_m: tuple[float, float], model: Any
) -> tuple[np.ndarray, float, int]:
    """The tablet's frame in the camera that saw ``markers`` (and the fit's RMS in pixels, the number of markers)."""
    known = sorted(set(markers) & set(layout.markers))
    if len(known) < MIN_MARKERS:
        raise CalibrationError(f"只看到 {len(known)} 个标定码，至少要 {MIN_MARKERS} 个")
    on_glass, pixels = [], []
    for k in known:
        u0, v0, u1, v1 = layout.markers[k]
        on_glass += [(u * size_m[0], v * size_m[1], 0.0) for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))]
        pixels += np.asarray(markers[k], np.float64).reshape(4, 2).tolist()
    tablet, rms, ratio = hand_eye.planar_pose(np.array(on_glass), np.array(pixels), model)
    if ratio < 1.5:
        raise CalibrationError("标定码在画面里太正、太小，分不清平板往哪边斜")
    return tablet, rms, len(known)


class WristSession:
    """The wrist camera's steps on the console's arm."""

    def __init__(self, console: Any) -> None:
        self.console = console
        self.store = WristStore(pathlib.Path(console.store.path).with_name("wrist.json"))
        self.wrist = self.store.load()
        self.camera_factory: Callable[[str], Camera] | None = None  # the rehearsal's: instead of the real camera
        self.results: dict[str, Any] = {}
        self._top: np.ndarray | None = None  # where the last calibration has the top camera, not yet taken over
        self._pictures: dict[str, bytes] = {}
        self._lock = threading.Lock()

    # ---- what the steps need -----------------------------------------------------------------------------------------
    @property
    def use(self) -> bool:
        """Whether instructions are carried out looking through the wrist camera."""
        return bool(self.wrist.records.get("use_for_runs"))

    def set_use(self, on: bool) -> None:
        if on:
            wants = self.missing()
            if wants:
                raise SetupError("还不能用腕部相机看屏幕：" + "；".join(wants))
        self.wrist.records["use_for_runs"] = bool(on)
        self.store.save(self.wrist)
        self.console.say("指令改用腕部相机看屏幕" if on else "指令改回用顶部相机看屏幕")

    def set_mount(self, mount: str) -> None:
        if mount not in MOUNTS:
            raise ValueError(f"不认识的支架 {mount!r}；可用的有 {sorted(MOUNTS)}")
        if mount != self.wrist.mount:  # the calibration belonged to the other mount
            self.wrist.mount = mount
            self.wrist.in_flange = self.wrist.look_q = self.wrist.look_screen = None
            self.wrist.records.pop("use_for_runs", None)
            self.store.save(self.wrist)
            self.console.say(f"腕部支架改成 {mount}：手眼标定和看屏位要重做")

    def _screen(self) -> ScreenMap:
        setup = self.console.setup
        setup.require(
            "screen_map", "park_q", "seed_q", hint="先让顶部相机认出平板（“点按”页连接后，平板全屏显示标定页）。"
        )
        return setup.screen_map

    def _layout(self) -> screen_layout.ScreenLayout:
        con = self.console
        screen = con.page.screen()
        if screen is None or not screen["covers_display"]:
            raise SetupError("平板要全屏显示标定页（腕部相机靠页上的六个码）。")
        return con.page.layout()

    def _open(self) -> Camera:
        con = self.console
        if self.camera_factory is not None:
            camera = self.camera_factory(self.wrist.camera)
        elif con.fake:
            raise SetupError("模拟模式没有腕部相机；腕部流程的排练用 bin/py -m tablet_tap.web.rehearsal --wrist")
        else:
            con._hardware_free("打开腕部相机")  # noqa: SLF001 - the console's own rule for touching the bench
            camera = open_camera(self.wrist.camera)
        model = camera_model(camera)
        if model is not None:
            self.wrist.model = model
        return camera

    def _move(self, q_goal: np.ndarray, name: str) -> None:
        """A slow joint move; the legs were checked when they were planned."""
        con = self.console
        arm, limits = con._need_arm(), con.setup.motion_limits()  # noqa: SLF001
        slow = dataclasses.replace(limits, joint_move_speed=min(limits.joint_move_speed, SLOW_RAD_S))
        run_segments(arm, [joint_move(name, arm.read(), np.asarray(q_goal, np.float64), slow, settle=True)], hz=limits.hz,
                     should_stop=con.stop.is_set, realtime=con.realtime)  # fmt: skip

    def _must_be_parked(self) -> None:
        con = self.console
        gap = float(np.max(np.abs(con._need_arm().read() - np.asarray(con.setup.park_q))))  # noqa: SLF001
        if gap > START_TOL_RAD:
            raise PathError("机械臂不在停靠位姿：先点“让开”。")

    def _look_robot(self, look_q: np.ndarray | None = None) -> LookRobot:
        con, setup, screen = self.console, self.console.setup, self._screen()
        if look_q is None and self.wrist.look_q is None:
            raise SetupError("还没有看屏位：先点“去看屏位”。")
        return LookRobot(
            con._need_arm(),  # noqa: SLF001
            setup.pen_arm(),
            screen,
            np.asarray(self.wrist.look_q if look_q is None else look_q),
            np.asarray(setup.park_q),
            np.asarray(setup.seed_q),
            self.wrist.planner(setup.pen_arm()),
            setup.motion_limits(),
            should_stop=con.stop.is_set,
            realtime=con.realtime,
            aim=con._aim,  # noqa: SLF001
        )

    def _tablet_pose(self) -> list[list[float]] | None:
        """Where the top camera has the tablet now: what tells whether it was moved since the look pose's picture."""
        live = self.console.live
        return None if live is None else np.asarray(live.pose).tolist()

    @property
    def second_path(self) -> pathlib.Path:
        return self.store.path.with_name(self.store.path.stem + "_view2.json")

    @property
    def occluder_path(self) -> pathlib.Path:
        return self.store.path.with_name(self.store.path.stem + "_occluder.png")

    def _measure_occluder(self, folder: pathlib.Path, planner: Any) -> np.ndarray | None:
        """What pen, holder and fingers hide of the wrist camera's picture, from the calibration's own pictures; kept
        beside the wrist camera's file. None when the folder does not have enough of them."""
        model = planner.camera.model
        pictures = [cv2.imread(str(path)) for path in sorted(pathlib.Path(folder).glob("view_*.png"))]
        pictures = [x for x in pictures if x is not None and x.shape[1::-1] == (model.width, model.height)]
        drawn = look_search.drawn_mask(planner)
        measured = look_search.occluder_from_pictures(pictures, drawn)
        if measured is None:
            return None
        cv2.imwrite(str(self.occluder_path), measured * 255)
        self.wrist.records["occluder"] = {"mount": self.wrist.mount, "pictures": len(pictures), "share": float(measured.mean()),
                                          "drawn_share": float(drawn.mean()), "from": str(folder), "at": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
        self.store.save(self.wrist)
        self.console.say(f"按 {len(pictures)} 张标定照片量出笔和笔夹在腕部画面里挡住 {measured.mean():.1%}（图纸估的是 {drawn.mean():.1%}），以后按量出来的算。")
        return measured

    def _hide(self, planner: Any) -> np.ndarray:
        """Where the wrist camera's picture does not show the scene, for the mount as it is: measured when the
        calibration's pictures are there, else the drawing's boxes."""
        model, record, measured = planner.camera.model, self.wrist.records.get("occluder") or {}, None
        if record.get("mount") == self.wrist.mount and self.occluder_path.exists():
            measured = cv2.imread(str(self.occluder_path), cv2.IMREAD_GRAYSCALE)
            measured = None if measured is None or measured.shape != (model.height, model.width) else (measured > 0).astype(np.uint8)
        elif self.wrist.calibrated and (self.wrist.records.get("eye_in_hand") or {}).get("folder"):
            measured = self._measure_occluder(pathlib.Path(self.wrist.records["eye_in_hand"]["folder"]), planner)
        return look_search.hide_mask(planner, measured)

    def _looks(self, screen: ScreenMap) -> tuple[Look, Look | None]:
        """The look pose for the tablet where it lies: one that shows the whole screen; when the pen is in the way of
        every such pose, the best one with a second that shows what it misses."""
        setup = self.console.setup
        planner, seed, park = self.wrist.planner(setup.pen_arm()), np.asarray(setup.seed_q), np.asarray(setup.park_q)
        try:
            return planner.best(screen, seed, park), None
        except Unreachable:
            first, second = look_search.find_looks(planner, self._hide(planner), screen, seed, park)
            return first.look, None if second is None else second.look

    def _fill(self, rectifier: ScreenRectifier) -> np.ndarray | None:
        """Where the first look's rectified picture does not show the screen, when the two looks are a pair of that
        kind; None when the first look shows all of it."""
        if not (self.wrist.records.get("view2") or {}).get("mosaic"):
            return None
        return look_search.unseen_in_rectified(rectifier, self._hide(self.wrist.planner(self.console.setup.pen_arm())))

    @property
    def merge(self) -> bool:
        """Whether every screenshot is mended with a second look (against a lamp's reflection)."""
        return bool(self.wrist.records.get("merge_two_looks")) and self.second_ready()

    def second_ready(self) -> bool:
        second, first = self.wrist.records.get("view2"), self.wrist.records.get("view")
        return bool(second and first and second.get("first_at") == first.get("at") and self.second_path.exists())

    def set_merge(self, on: bool) -> None:
        if on and not self.second_ready():
            raise SetupError("还没有第二个看屏位的画面：先点“从第二个角度再拍一张”")
        if not on and self.second_ready() and self.wrist.records["view2"].get("mosaic"):
            raise SetupError("这个支架上一个角度看不全屏幕（笔挡住一块），两个角度是拼起来用的，关不了。")
        self.wrist.records["merge_two_looks"] = bool(on)
        self.store.save(self.wrist)
        self.console.say("看屏幕改成两个角度合成（去掉灯的反光，每一步多走一趟）" if on else "看屏幕改回只用一个角度")

    def _since_view(self, record: str = "view") -> tuple[np.ndarray, float, float] | None:
        """How the tablet has moved on its plane since the look pose's picture was taken, as the top camera saw it:
        ((3, 3) the move in the tablet's own frame of then, metres; how far; the turn in degrees). None when either
        placement is not known."""
        then, live = (self.wrist.records.get(record) or {}).get("tablet_pose"), self.console.live
        if then is None or live is None:
            return None
        then = np.asarray(then, np.float64)
        shift, turn = live.moved(then)
        step = np.linalg.inv(then) @ np.asarray(live.pose, np.float64)
        move = np.eye(3)
        move[:2, :2], move[:2, 2] = step[:2, :2], step[:2, 3]
        return move, shift, turn

    def moved(self) -> str | None:
        """Why the look pose's picture no longer belongs to the tablet as it lies; None when it does. The top camera's
        own placement is compared, not the map the touches have corrected: touching does not move the tablet. A small
        push is followed (``rectifier``) and is no reason."""
        if self.wrist.look_q is None or not self.store.view_path.exists():
            return "还没有看屏位的画面"
        since = self._since_view()
        if since is None:
            return self.wrist.screen_moved(self._screen())
        _, shift, turn = since
        if (
            shift > FOLLOW_M
            or turn > FOLLOW_DEG
            or abs(
                np.asarray(self.console.live.pose)[2, 3] - np.asarray(self.wrist.records["view"]["tablet_pose"])[2, 3]
            )
            > 0.004
        ):
            return f"平板在看屏位拍照之后挪动了 {shift * 1e3:.0f} mm、{turn:.1f}°，出了看屏位的画面"
        return None

    def missing(self) -> list[str]:
        """What stands between the operator and looking through the wrist camera, in words."""
        wants = []
        if not self.wrist.calibrated:
            wants.append("腕部相机还没有做手眼标定")
        moved = self.moved()
        if moved:
            wants.append(moved + "（平板显示标定页，点“去看屏位”）")
        return wants

    # ---- the steps (each runs in the console's worker) ---------------------------------------------------------------
    def _plan(self, n_views: int) -> tuple[Look, list[Look]]:
        con, setup, screen = self.console, self.console.setup, self._screen()
        pen, park = setup.pen_arm(), np.asarray(setup.park_q)
        look, _ = self._looks(screen)
        page = con.page.screen()
        layout = con.page.layout() if page is not None else screen_layout.default_layout(setup.aspect or 1.6)
        looks = self.wrist.planner(pen, CALIBRATION_RULES).calibration_views(
            screen, look, park, list(layout.markers.values()), n_views
        )
        return look, looks

    def find(self) -> None:
        """Nothing can say where the tablet is (the top camera was moved, or never calibrated): look down from over the
        place the tablet is asked to be put at, and place it from the page's markers as the wrist camera sees them, the
        camera being where the calibration, or else the mount's drawing, has it. Good to about a centimetre, which is
        what planning the calibration's looks needs. From then on the top camera has no say until its place has been
        worked out again (``calibrate`` does that)."""
        con, setup = self.console, self.console.setup
        layout = self._layout()
        setup.require("park_q", hint="先记录停靠位姿。")
        self._must_be_parked()
        size = tuple(setup.screen_size_m or SCREEN_M)
        pen, park, asked = setup.pen_arm(), np.asarray(setup.park_q), asked_screen(size)
        seed = hand_eye.seed_joints(pen, asked, dataclasses.replace(setup.motion_limits(), hover_m=0.03))
        look = self.wrist.planner(pen, dataclasses.replace(LookRules(), tip_m=FIRST_LOOK_TIP_M)).best(asked, seed, park)
        con.say(f"去平板应该在的位置上方看一眼（笔尖离那里的屏幕 {look.tip_height_m * 1e3:.0f} mm），再回停靠位。")
        camera = self._open()
        try:
            con.page.set_mode("markers")
            con.page.clear()
            self._move(look.q, "去上方看一眼")
            q, picture = still_picture(con._need_arm().read, camera, settle_s=0.6 if con.realtime else 0.0)  # noqa: SLF001
            self._move(park, "回停靠位")
        finally:
            camera.close()
        self._show("picture", picture)
        markers = {k: quad for k, quad in detect_markers(picture).items() if k in layout.markers}
        try:
            in_camera, rms, n = seen_tablet(markers, layout, size, self.wrist.wrist_camera().model)
        except CalibrationError as exc:
            raise CalibrationError(
                f"腕部相机在上方没认出平板（{exc}）：把平板放到左臂正前方、中心离底座约 {ASKED_AHEAD_M * 1e2:.0f} cm，全屏显示标定页，再试。"
            ) from None
        tablet = flange_pose(pen, q) @ self.wrist.wrist_camera().in_flange @ in_camera
        if setup.camera_pose is not None:  # the top camera's own placement would be taken up again at its next look
            setup.records["camera_pose_before_wrist"] = setup.camera_pose
            setup.camera_pose = None
        con._forget("平板的位置改由腕部相机确定")  # noqa: SLF001
        con.live = LiveScreen.from_map(screen_of(tablet, size))
        con.correction = TapCorrection(con.live.size_m)
        setup.aspect = layout.aspect
        con._apply_live(lean_free=True)  # noqa: SLF001
        con._located = True  # noqa: SLF001
        centre = setup.screen_map.point(0.5, 0.5)
        slant = float(np.degrees(np.arccos(np.clip(setup.screen_map.normal[2], -1, 1))))
        self.results["find"] = {"markers": n, "rms_px": rms, "centre_cm": (centre * 1e2).round(1).tolist(), "slant_deg": slant,
                                "at": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
        con.say(
            f"腕部相机认出平板：{n} 个码，重投影 {rms:.1f} px；屏幕中心在底座前 {centre[0] * 1e2:.1f} cm、左 {centre[1] * 1e2:.1f} cm、高 {centre[2] * 1e2:.1f} cm，"
            f"倾斜 {slant:.0f}°（按{'手眼标定' if self.wrist.calibrated else '支架图纸'}算，准到 1 cm 左右）。顶部相机暂时不参与，标定后会重新算出它的位置。"
            + (f" {con._reach}" if con._reach else "")  # noqa: SLF001
        )

    def top_camera(self, tablet: np.ndarray, size: tuple[float, float]) -> dict[str, Any]:
        """Where the top camera is, from the calibration page as it sees it now and from where the wrist camera's
        calibration has the tablet (``tablet``: its frame in the arm's base). Taken over at once when the top camera
        had no place; otherwise kept for the operator to take over."""
        con, setup = self.console, self.console.setup
        if setup.camera_model is None or con.camera is None:
            raise SetupError("顶部相机没有打开，或者没有它的内参")
        markers = {k: quad for k, quad in detect_markers(con.fresh_frame()).items() if k in con.page.layout().markers}
        in_camera, rms, n = seen_tablet(markers, con.page.layout(), size, setup.camera_model)
        pose = tablet @ np.linalg.inv(in_camera)
        result: dict[str, Any] = {"markers": n, "rms_px": rms, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        if setup.camera_pose is not None:
            old = np.asarray(setup.camera_pose, np.float64)
            result["shift_mm"], result["turn_deg"] = mount_error(pose, old)
            self._top, result["adopted"] = pose, False
            con.say(
                f"按这次标定，顶部相机的位置和现在用的差 {result['shift_mm']:.0f} mm、{result['turn_deg']:.1f}°（{n} 个码，{rms:.1f} px）。要改用新的，点“改用腕部标定算出的顶部相机位置”。"
            )
        else:
            self.adopt_top(pose, tablet, size)
            result["adopted"] = True
            con.say(
                f"顶部相机的位置已经由腕部标定算出（{n} 个码，{rms:.1f} px）：离底座前 {pose[0, 3] * 1e2:.0f} cm、左 {pose[1, 3] * 1e2:.0f} cm、高 {pose[2, 3] * 1e2:.0f} cm。它重新负责粗定位和跟踪。"
            )
        self.results["top"] = result
        return result

    def adopt_top(
        self, pose: np.ndarray | None = None, tablet: np.ndarray | None = None, size: tuple[float, float] | None = None
    ) -> None:
        """Take over the top camera's place as the last calibration has it."""
        con, setup = self.console, self.console.setup
        if pose is None:
            if self._top is None:
                raise SetupError("还没有算出顶部相机的位置：先做腕部相机的手眼标定")
            pose = self._top
        if setup.camera_pose is not None:
            setup.records["camera_pose_before_wrist"] = setup.camera_pose
        setup.camera_pose = np.asarray(pose).tolist()
        setup.records["camera_pose_source"] = f"腕部相机手眼标定 {time.strftime('%Y-%m-%d %H:%M:%S')}"
        if tablet is not None and size is not None and con.live is not None and not con.live.contacts:
            con.live.pose = np.asarray(tablet, np.float64)  # what the top camera will now see it as: nothing jumps
            con._apply_live(lean_free=True)  # noqa: SLF001
        else:
            con._save()  # noqa: SLF001
        self._top = None
        if "top" in self.results:
            self.results["top"]["adopted"] = True

    def plan(self, n_views: int = 14) -> None:
        """Only compute: the look pose and the calibration's poses. Nothing moves."""
        started = time.monotonic()
        look, looks = self._plan(n_views)
        park = np.asarray(self.console.setup.park_q)
        reach = np.degrees(max(float(np.max(np.abs(v.q - park))) for v in looks))
        self.results["plan"] = {"look": look.describe(), "look_q": np.round(look.q, 3).tolist(), "views": len(looks),
                                "lowest_tip_mm": round(min(v.tip_height_m for v in looks) * 1e3), "reach_deg": round(reach),
                                "mount": self.wrist.mount, "calibrated": self.wrist.calibrated}  # fmt: skip
        self.console.say(
            f"腕部相机试算（{time.monotonic() - started:.0f} 秒，没有动）：看屏位 {look.describe()}\n"
            f"标定位姿 {len(looks)} 个，笔尖最低离屏 {min(v.tip_height_m for v in looks) * 1e3:.0f} mm，关节离停靠位最多 {reach:.0f}°"
            + ("" if self.wrist.calibrated else "（相机位置按支架图纸算）")
        )

    def calibrate(self, n_views: int = 14) -> None:
        """Eye in hand: the tablet shows its markers, the arm looks at them from a set of poses, nothing is touched."""
        con, setup = self.console, self.console.setup
        layout, screen, park = self._layout(), self._screen(), np.asarray(setup.park_q)
        self._must_be_parked()
        _, looks = self._plan(n_views)
        con.say(
            f"将低速走 {len(looks)} 个位姿拍标定页（笔不碰屏幕，最低离屏 {min(v.tip_height_m for v in looks) * 1e3:.0f} mm），最后回停靠位。"
        )
        out = config.runs_dir() / f"wrist_calib_{datetime.datetime.now():%Y%m%d_%H%M%S}"
        out.mkdir(parents=True, exist_ok=True)
        camera = self._open()

        def keep(k: int, picture: np.ndarray) -> None:
            con.progress = {"job": "腕部手眼标定", "i": k, "n": len(looks)}
            cv2.imwrite(str(out / f"view_{k:02d}.png"), picture)
            self._show("picture", picture)

        try:
            con.page.set_mode("markers")
            con.page.clear()
            arm = con._need_arm()  # noqa: SLF001
            views = collect_views(looks, park, self._move, arm.read, camera, layout,
                                  settle_s=0.6 if con.realtime else 0.0, say=con.say, keep=keep)  # fmt: skip
        finally:
            camera.close()
        (out / "views.json").write_text(
            json.dumps(
                [{"q": v.q.tolist(), "markers": {str(k): m.tolist() for k, m in v.markers.items()}} for v in views]
            )
        )
        drawn = camera_in_flange(*MOUNTS[self.wrist.mount])
        model = self.wrist.wrist_camera().model
        size = tuple(setup.screen_size_m or screen.size_m)

        def solve(guess: np.ndarray) -> Any:  # the display's size is known: it is not fitted
            try:
                return solve_eye_in_hand(views, layout, size, model, setup.pen_arm(), guess, fit_size=False)
            except (CalibrationError, np.linalg.LinAlgError) as exc:
                return exc

        found, upright = solve(drawn), True
        if isinstance(found, Exception) or found.rms_px > BAD_FIT_PX:
            other = solve(drawn @ TURNED)
            if not isinstance(other, Exception) and (isinstance(found, Exception) or other.rms_px < 0.5 * found.rms_px):
                found, drawn, upright = other, drawn @ TURNED, False
                con.say("⚠ 相机在支架里的方向和图纸相反（绕光轴转了半圈）：按实际方向标定，后面都按它算。")
        if isinstance(found, Exception):
            raise CalibrationError(f"手眼标定没有解出来：{found}（照片和数据在 {out}）")
        (out / "result.json").write_text(json.dumps(found.to_json(), indent=1))
        off_mm, off_deg = mount_error(found.camera_in_flange, drawn)
        seen = found.screen_map()
        gaps = [
            float(np.linalg.norm(seen.point(u, v) - screen.point(u, v)) * 1e3)
            for u, v in ((0, 0), (1, 0), (1, 1), (0, 1))
        ]
        summary = {"rms_px": found.rms_px, "held_out_px": found.held_out_px, "left_out": len(found.left_out), "views": len(views),
                   "position_sigma_mm": found.position_sigma_mm, "angle_sigma_deg": found.angle_sigma_deg,
                   "off_drawing_mm": off_mm, "off_drawing_deg": off_deg, "screen_size_mm": [x * 1e3 for x in found.screen_size_m],
                   "screen_gap_mm": gaps, "folder": str(out), "at": time.strftime("%Y-%m-%d %H:%M:%S"), "upright": upright}  # fmt: skip
        self.results["calibration"] = summary
        con.say(
            f"腕部手眼标定：拟合残差 {found.rms_px:.2f} px，没参加拟合的位姿 {found.held_out_px:.2f} px，去掉 {len(found.left_out)} 个位姿；"
            f"相机位置的不确定度约 {found.position_sigma_mm:.1f} mm / {found.angle_sigma_deg:.2f}°\n"
            f"和支架图纸差 {off_mm:.1f} mm、{off_deg:.1f}°；屏幕尺寸 {found.screen_size_m[0] * 1e3:.1f} × {found.screen_size_m[1] * 1e3:.1f} mm；"
            f"它看到的屏幕和标定前估计的四角差 {' / '.join(f'{g:.1f}' for g in gaps)} mm。记录 {out}"
        )
        if off_mm > MOUNT_MM or off_deg > MOUNT_DEG:
            raise CalibrationError(
                f"标定结果离支架图纸太远（上限 {MOUNT_MM:.0f} mm、{MOUNT_DEG:.0f}°），没有保存：检查相机是不是左腕那台、支架选对没有。"
            )
        self.wrist.in_flange = found.camera_in_flange.tolist()
        self.wrist.look_q = self.wrist.look_screen = None  # the look pose was planned with the old camera
        self.wrist.records.pop("view", None)
        self.wrist.records["eye_in_hand"] = {
            k: v for k, v in found.to_json().items() if k != "camera_in_flange"
        } | summary
        self.store.save(self.wrist)
        con.say(f"已保存 {self.store.path}。接着点“去看屏位”。")
        self._measure_occluder(out, self.wrist.planner(setup.pen_arm()))
        try:  # the arm is back at the park pose and the page still shows its markers: where is the top camera?
            self.top_camera(found.screen_in_base, size)
        except (SetupError, CalibrationError, TimeoutError) as exc:
            con.say(f"顶部相机的位置这次没有核对（{exc}）")

    def _another_look(self, screen: ScreenMap, tried: list[Any], reach_from: np.ndarray | None = None) -> Look:
        """A look pose apart from those ``tried`` (camera poses) for this place of the tablet: for when a lamp is
        mirrored in the glass from where the best one looks. ``reach_from``: joints from which a straight joint move to
        it must be free."""
        setup = self.console.setup
        rules = dataclasses.replace(
            LookRules(), off_normal_deg=SLANTED_LOOK_DEG
        )  # more slanted than the best look may be
        planner, park = self.wrist.planner(setup.pen_arm(), rules), np.asarray(setup.park_q)
        found = planner._from_postures(screen, np.asarray(setup.seed_q), park)  # noqa: SLF001 - every look, not the best one
        if not found:
            raise SetupError("这个位置只找得到一个看屏位；换不了角度，把灯挡一下或者挪一下平板。")

        def score(look: Look) -> float:
            travel = float(np.degrees(np.max(np.abs(look.q - park))))
            return look.detail_px + 8 * min(look.limit_deg, 15.0) - travel - 3 * look.off_normal_deg

        def apart(a: np.ndarray, b: np.ndarray) -> bool:
            turn = np.degrees(np.arccos(np.clip(a[:3, 2] @ b[:3, 2], -1.0, 1.0)))
            return float(np.linalg.norm(a[:3, 3] - b[:3, 3])) > APART_M or turn > APART_DEG

        tried = [np.asarray(t, np.float64) for t in tried]
        fresh = [look for look in found if all(apart(look.camera_in_base, t) for t in tried)]
        if reach_from is not None:
            fresh = [look for look in fresh if planner.path_problem(screen, reach_from, look.q) is None]
        if not fresh:
            raise SetupError(f"能看全屏幕的角度都试过了（{len(tried)} 个）：把灯挡一下，或者挪一下平板再来。")

        def away(look: Look) -> float:  # the reflection moves with the camera: the furthest from those tried first
            return min((float(np.linalg.norm(look.camera_in_base[:3, 3] - t[:3, 3])) for t in tried), default=0.0)

        return max(fresh, key=lambda look: 1500.0 * away(look) + score(look))

    def view(self, out_width: int = 1792, keep_pose: bool = False, another: bool = False) -> None:
        """Go to the look pose and lay the picture over the screen with the page's markers; again after the tablet was
        moved. ``another``: from a different look pose than those tried so far."""
        con, setup = self.console, self.console.setup
        layout, screen = self._layout(), self._screen()
        if keep_pose and self.wrist.look_q is not None and not another:
            look_q = np.asarray(self.wrist.look_q)
        else:
            if another:
                look = self._another_look(screen, self.wrist.records.get("looks_tried", []))
            else:
                look, second = self._looks(screen)
                if second is not None:
                    return self._view_pair(look, second, layout, screen, int(out_width))
                self.wrist.records["looks_tried"] = []
            self.wrist.records["seconds_tried"] = []  # second looks belonged to the first look that is replaced now
            self.wrist.records["looks_tried"] = [
                *self.wrist.records.get("looks_tried", []),
                look.camera_in_base.tolist(),
            ]
            con.say(f"看屏位{'（换了一个角度）' if another else ''}：{look.describe()}")
            look_q = look.q
        if not self.wrist.calibrated:
            con.say("⚠ 腕部相机还没做手眼标定，看屏位按支架图纸算，画面可能偏几度。")
        robot = self._look_robot(look_q)
        camera = self._open()
        try:
            con.page.set_mode("markers")
            con.page.clear()
            robot.to_look()
            q, picture = still_picture(con._need_arm().read, camera, settle_s=0.6 if con.realtime else 0.0)  # noqa: SLF001
            robot.to_rest()
        finally:
            camera.close()
        path = self.store.view_path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._show("picture", picture)
        try:
            rectifier = ScreenRectifier.from_marker_image(picture, layout, out_width=int(out_width))
        except CalibrationError as exc:
            cv2.imwrite(str(path.with_suffix(".failed.jpg")), picture)
            raise CalibrationError(
                f"看屏位的画面标定失败：{exc}（画面存在 {path.with_suffix('.failed.jpg')}）。码被灯的反光盖住的话，点“换一个角度再拍”。"
            ) from None
        rectifier.save(path)
        self._show("picture", draw_overlay(picture, rectifier))
        rectified = rectifier.rectify(picture)
        self._show("screen", rectified)
        glare, n_markers = glare_share(rectified, layout), len(rectifier.marker_ids)
        self.wrist.look_q, self.wrist.look_screen = look_q.tolist(), screen.to_json()
        quality = look_quality(picture, rectifier)
        expected, worst = self.wrist.predicted_homography(setup.pen_arm(), q, screen), 0.0
        for u, v in ((0.05, 0.05), (0.95, 0.05), (0.95, 0.95), (0.05, 0.95), (0.5, 0.5)):
            x, y, w = expected @ [*rectifier.norm_to_cam(np.array([u, v])), 1.0]
            worst = max(worst, screen.error_mm((u, v), (x / w, y / w)))
        self.wrist.records["view"] = quality | {"hand_eye_gap_mm": worst, "at": time.strftime("%Y-%m-%d %H:%M:%S"), "glare": glare,
                                                "markers": n_markers, "tablet_pose": self._tablet_pose()}  # fmt: skip
        self.store.save(self.wrist)
        edge = quality["edge_px"]
        con.say(
            f"看屏位的画面已标定：屏幕在画面里 {quality['width_px']:.0f} × {quality['height_px']:.0f} px，码的重投影误差 {quality['reprojection_px']:.2f} px，"
            f"用到 {len(rectifier.marker_ids)} 个码，边缘过渡 {'量不出' if edge is None else f'{edge:.1f} px'}；"
            f"手眼标定预测的画面和实际最大差 {worst:.1f} mm（只作检查）"
        )
        if glare > GLARE_SHARE or n_markers < len(layout.markers):
            con.say(
                f"⚠ 屏幕上有反光（暗处有 {glare:.1%} 发亮，{len(layout.markers)} 个码只认出 {n_markers} 个）：灯映在玻璃上，模型会看不清那一块。"
                "点“换一个角度再拍”，或者把灯挡一下。"
            )
        if quality["width_px"] < GOOD_WIDTH_PX:
            con.say(f"⚠ 屏幕不到 {GOOD_WIDTH_PX:.0f} px 宽，比预期小。")
        if edge is not None and edge > GOOD_EDGE_PX:
            con.say(f"⚠ 边缘过渡超过 {GOOD_EDGE_PX:.0f} px：离焦或抖动。")

    def view_second(self, out_width: int = 1792) -> None:
        """A second look pose and its picture, for mending what a lamp's reflection washes out in the first: both
        pictures are taken now, laid over the screen, and what the two together still lose is reported."""
        con = self.console
        layout, screen = self._layout(), self._screen()
        first = self.rectifier()  # says so when there is no first look, or the tablet was moved since
        first_q = np.asarray(self.wrist.look_q)
        first_pose = flange_pose(self.console.setup.pen_arm(), first_q) @ self.wrist.wrist_camera().in_flange
        look = self._another_look(
            screen, [first_pose, *self.wrist.records.get("seconds_tried", [])], reach_from=first_q
        )
        self.wrist.records["seconds_tried"] = [
            *self.wrist.records.get("seconds_tried", []),
            look.camera_in_base.tolist(),
        ]
        con.say(f"第二个看屏位：{look.describe()}")
        robot = self._look_robot(first_q)
        camera = self._open()
        try:
            con.page.set_mode("markers")
            con.page.clear()
            robot.to_look()
            settle = 0.6 if con.realtime else 0.0
            _, picture1 = still_picture(con._need_arm().read, camera, settle_s=settle)  # noqa: SLF001
            robot.move_to(look.q, "去第二个看屏位")
            _, picture2 = still_picture(con._need_arm().read, camera, settle_s=settle)  # noqa: SLF001
            robot.move_to(first_q, "回第一个看屏位")
            robot.to_rest()
        finally:
            camera.close()
        try:
            second = ScreenRectifier.from_marker_image(picture2, layout, out_width=int(out_width))
        except CalibrationError as exc:
            raise CalibrationError(f"第二个看屏位的画面标定失败：{exc}。再点一次会换另一个角度。") from None
        second.save(self.second_path)
        a, b = first.rectify(picture1), second.rectify(picture2)
        merged = merge_looks(a, b)
        self._show("picture", draw_overlay(picture2, second))
        self._show("screen", merged)
        glare = {
            "first": glare_share(a, layout),
            "second": glare_share(b, layout),
            "merged": glare_share(merged, layout),
        }
        self.wrist.records["view2"] = {"look_q": look.q.tolist(), "glare": glare, "markers": len(second.marker_ids),
                                       "at": time.strftime("%Y-%m-%d %H:%M:%S"), "tablet_pose": self._tablet_pose(),
                                       "first_at": (self.wrist.records.get("view") or {}).get("at")}  # fmt: skip
        self.store.save(self.wrist)
        con.say(
            f"第二个看屏位的画面已标定（{len(second.marker_ids)} 个码）。暗处发亮：第一个角度 {glare['first']:.1%}，第二个 {glare['second']:.1%}，"
            f"两个合成后 {glare['merged']:.1%}。"
            + (
                "可以勾“两个角度合成”。"
                if glare["merged"] <= GLARE_SHARE
                else "⚠ 合成后还有反光：两处反光重叠了，再点一次换个角度，或者把灯挡一下。"
            )
        )

    def _view_pair(self, first: Look, second: Look, layout: Any, screen: ScreenMap, out_width: int) -> None:
        """The pen is in front of some of the screen from every pose: two looks, the second showing what the first
        misses. Both pictures are taken and laid over the screen now; from then on every screenshot is the two together."""
        con, setup = self.console, self.console.setup
        con.say(f"一个角度看不全屏幕（笔挡住一块）：用两个角度拼。第一个：{first.describe()}\n第二个：{second.describe()}")
        robot = self._look_robot(first.q)
        camera = self._open()
        try:
            con.page.set_mode("markers")
            con.page.clear()
            robot.to_look()
            settle = 0.6 if con.realtime else 0.0
            _, picture1 = still_picture(con._need_arm().read, camera, settle_s=settle)  # noqa: SLF001
            robot.move_to(second.q, "去第二个看屏位")
            _, picture2 = still_picture(con._need_arm().read, camera, settle_s=settle)  # noqa: SLF001
            robot.move_to(first.q, "回第一个看屏位")
            robot.to_rest()
        finally:
            camera.close()
        self._show("picture", picture1)
        try:
            rect1 = ScreenRectifier.from_marker_image(picture1, layout, out_width=out_width)
            rect2 = ScreenRectifier.from_marker_image(picture2, layout, out_width=out_width)
        except CalibrationError as exc:
            cv2.imwrite(str(self.store.view_path.with_suffix(".failed.jpg")), picture1)
            raise CalibrationError(f"看屏位的画面标定失败：{exc}（码被笔或反光盖住太多；挪一下平板再试）") from None
        self.store.view_path.parent.mkdir(parents=True, exist_ok=True)
        rect1.save(self.store.view_path)
        rect2.save(self.second_path)
        hide = self._hide(self.wrist.planner(setup.pen_arm()))
        fill, fill2 = look_search.unseen_in_rectified(rect1, hide), look_search.unseen_in_rectified(rect2, hide)
        a, b = rect1.rectify(picture1), rect2.rectify(picture2)
        merged = merge_looks(a, b, fill)
        missed = float(np.mean((fill > 0.5) & (fill2 > 0.5)))  # what neither look shows
        self._show("picture", draw_overlay(picture1, rect1))
        self._show("screen", merged)
        now, glare = time.strftime("%Y-%m-%d %H:%M:%S"), glare_share(merged, layout)
        self.wrist.look_q, self.wrist.look_screen = first.q.tolist(), screen.to_json()
        self.wrist.records["looks_tried"] = self.wrist.records["seconds_tried"] = []
        self.wrist.records["view"] = look_quality(picture1, rect1) | {
            "hand_eye_gap_mm": float("nan"), "at": now, "glare": glare, "markers": len(rect1.marker_ids),
            "tablet_pose": self._tablet_pose(), "shown": float(1.0 - np.mean(fill > 0.5)),
        }  # fmt: skip
        self.wrist.records["view2"] = {"look_q": second.q.tolist(), "glare": {"first": glare_share(a, layout), "second": glare_share(b, layout), "merged": glare},
                                       "markers": len(rect2.marker_ids), "at": now, "tablet_pose": self._tablet_pose(), "first_at": now,
                                       "mosaic": True, "missed": missed}  # fmt: skip
        self.wrist.records["merge_two_looks"] = True
        self.store.save(self.wrist)
        quality = self.wrist.records["view"]
        con.say(
            f"两个看屏位的画面已标定：第一个看到屏幕的 {quality['shown']:.0%}（屏幕在画面里约 {quality['width_px']:.0f} px 宽，{len(rect1.marker_ids)} 个码），"
            f"第二个补上其余（{len(rect2.marker_ids)} 个码）；两个都看不到的占 {missed:.1%}，拼好后暗处发亮 {glare:.1%}。以后每次看屏幕走两个位姿。"
        )
        if missed > 0.005:
            con.say("⚠ 有一小块两个角度都看不到：挪一下平板再点一次。")
        if glare > GLARE_SHARE:
            con.say("⚠ 拼好后还有反光：把灯挡一下，或者挪一下平板再点一次。")

    def rectifier(self) -> ScreenRectifier:
        """The look pose's picture laid over the screen. After a small push of the tablet it is laid over it again
        from how far the top camera saw the tablet go: the camera and the plane are where they were, so a pixel shows
        the same point of the plane, which now is another point of the screen."""
        moved = self.moved()
        if moved:
            raise SetupError(f"{moved}：平板显示标定页，重新点“去看屏位”。")
        return self._followed(ScreenRectifier.load(self.store.view_path), "view")

    def _followed(self, taken: ScreenRectifier, record: str) -> ScreenRectifier:
        since = self._since_view(record)
        if since is None or (since[1] < MOVED_M and since[2] < MOVED_DEG):
            return taken
        size = np.diag([*self.console.live.size_m, 1.0])
        follow = np.linalg.inv(size) @ np.linalg.inv(since[0]) @ size @ taken.homography
        return ScreenRectifier(
            follow, taken.aspect, taken.out_width, taken.reprojection_px, taken.image_size, taken.marker_ids
        )

    def shot(self) -> None:
        """What the model would be shown now: go to the look pose, take the picture, come back."""
        rectifier, robot = self.rectifier(), self._look_robot()
        settle = 0.5 if self.console.realtime else 0.0
        camera = self._open()
        try:
            if self.merge:  # what a run would be shown: the two looks together
                device: RobotTabletDevice = TwoLookDevice(
                    robot, camera, rectifier, view_settle_s=settle, second_q=np.asarray(self.wrist.records["view2"]["look_q"]),
                    second_rectifier=self._followed(ScreenRectifier.load(self.second_path), "view2"), fill=self._fill(rectifier),
                )  # fmt: skip
            else:
                device = RobotTabletDevice(robot, camera, rectifier, view_settle_s=settle)
            robot.to_look()
            image = device.screenshot()
            robot.to_rest()
        finally:
            camera.close()
        screen = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        self._show("screen", screen)
        if device.last_frame is not None:
            self._show("picture", device.last_frame)
        out = config.runs_dir() / "wrist_shot.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), screen)
        self.console.say(f"腕部相机拍了一张（{image.width} × {image.height}），存在 {out}")

    def run_device(self, keyboard: Any) -> tuple[RobotTabletDevice, Callable[[], None]]:
        """The tablet as the agent sees it through the wrist camera, the arm already at the look pose; and what to call
        when the run is over (back to the park pose, camera released)."""
        con = self.console
        if con.from_camera:
            raise SetupError("屏幕还没有校准（在“点按”页点“一键校准精度”）")
        rectifier, robot = self.rectifier(), self._look_robot()
        settle = 0.5 if con.realtime else 0.0
        second = None
        if self.merge:
            second_q = np.asarray(self.wrist.records["view2"]["look_q"])
            problem = robot.look_planner.path_problem(self._screen(), robot.park_q, second_q)
            if problem:
                raise SetupError(f"两个看屏位之间的路上离{problem}：重新拍第二个角度，或者取消“两个角度合成”。")
            second = self._followed(ScreenRectifier.load(self.second_path), "view2")
        camera = self._open()
        try:
            if second is None:
                device = RobotTabletDevice(
                    robot, camera, rectifier, nav=con.setup.nav, keyboard=keyboard, view_settle_s=settle
                )
            else:
                device = TwoLookDevice(robot, camera, rectifier, nav=con.setup.nav, keyboard=keyboard, view_settle_s=settle,
                                       second_q=second_q, second_rectifier=second, fill=self._fill(rectifier))  # fmt: skip
            robot.to_look()
        except BaseException:
            camera.close()
            raise

        def done() -> None:
            try:
                if robot.where is not ArmPlace.UNKNOWN and not con.stop.is_set():
                    robot.to_rest()
            finally:
                camera.close()

        return device, done

    # ---- what the page shows -----------------------------------------------------------------------------------------
    def _show(self, kind: str, image_bgr: np.ndarray, width: int = 1280) -> None:
        scale = width / image_bgr.shape[1]
        if scale < 1:
            image_bgr = cv2.resize(image_bgr, (width, round(image_bgr.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        ok, data = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            with self._lock:
                self._pictures[kind] = data.tobytes()

    def picture(self, kind: str) -> bytes | None:
        with self._lock:
            return self._pictures.get(kind)

    def state(self) -> dict[str, Any]:
        wrist, con = self.wrist, self.console
        ready = con.arm is not None and con.moves and con.setup.screen_map is not None and con.setup.park_q is not None
        hand_eye = wrist.records.get("eye_in_hand") or {}
        view = {k: v for k, v in (wrist.records.get("view") or {}).items() if k != "tablet_pose"}
        try:
            moved = self.moved() if con.setup.screen_map is not None else "顶部相机还没有认出平板"
        except SetupError as exc:
            moved = str(exc)
        return {
            "camera": wrist.camera,
            "mount": wrist.mount,
            "mounts": sorted(MOUNTS),
            "calibrated": wrist.calibrated,
            "hand_eye": {
                k: hand_eye.get(k)
                for k in (
                    "rms_px",
                    "held_out_px",
                    "off_drawing_mm",
                    "off_drawing_deg",
                    "position_sigma_mm",
                    "at",
                    "views",
                )
            }
            if hand_eye
            else None,
            "look": wrist.look_q is not None,
            "view": view or None,
            "moved": moved,
            "use": self.use,
            "merge": self.merge,
            "view2": {k: v for k, v in (wrist.records.get("view2") or {}).items() if k in ("glare", "markers", "at")}
            if self.second_ready()
            else None,
            "ready": ready,
            "busy": con.busy,
            "progress": con.progress,
            "results": self.results,
            "top": {"known": con.setup.camera_pose is not None, "pending": self._top is not None},
            "can_find": con.arm is not None and con.moves and con.setup.park_q is not None,
            "pictures": sorted(self._pictures),
            "simulated": self.camera_factory is not None,
        }


def register(app: flask.Flask, console: Any, token: str = "", session: WristSession | None = None) -> WristSession:
    """Add the wrist camera's routes to the console's app; the session is also the console's ``wrist``."""
    wrist = session or WristSession(console)
    console.wrist = wrist

    def refused() -> Any:
        given = flask.request.args.get("token", "") or flask.request.headers.get("X-Token", "")
        if token and given != token:
            return flask.jsonify({"error": "需要口令（网址后加 ?token=…）"}), 401
        return None

    @app.get("/api/wrist/state")
    def wrist_state() -> Any:
        return refused() or flask.jsonify(wrist.state())

    @app.get("/api/wrist/<kind>.jpg")
    def wrist_picture(kind: str) -> Any:
        denied = refused()
        if denied is not None:
            return denied
        data = wrist.picture(kind) if kind in ("picture", "screen") else None
        if data is None:
            return flask.jsonify({"error": "还没有这张图"}), 404
        return flask.Response(data, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/api/wrist/cmd")
    def wrist_cmd() -> Any:
        denied = refused()
        if denied is not None:
            return denied
        args = flask.request.get_json(silent=True) if flask.request.is_json else None
        if not isinstance(args, dict) or "cmd" not in args:
            return flask.jsonify({"error": "请求必须是带 cmd 的 JSON"}), 400
        cmd = str(args["cmd"])
        try:
            if cmd == "use":
                wrist.set_use(bool(args.get("on")))
                return flask.jsonify({"ok": True})
            if cmd == "mount":
                if console.busy is not None:
                    return flask.jsonify({"error": f"正在执行“{console.busy}”"}), 409
                wrist.set_mount(str(args.get("mount", "")))
                return flask.jsonify({"ok": True})
            if cmd == "merge":
                wrist.set_merge(bool(args.get("on")))
                return flask.jsonify({"ok": True})
            if cmd == "top":
                if console.busy is not None:
                    return flask.jsonify({"error": f"正在执行“{console.busy}”"}), 409
                wrist.adopt_top()
                console.say("顶部相机改用腕部标定算出的位置")
                return flask.jsonify({"ok": True})
            if cmd not in JOBS:
                raise ValueError(f"未知命令 {cmd!r}")
            n_views = int(args.get("views", 14))
            if not MIN_VIEWS <= n_views <= MAX_VIEWS:
                raise ValueError(f"标定位姿数应在 {MIN_VIEWS}–{MAX_VIEWS} 之间")
            if cmd in ("calibrate", "view", "view2", "shot", "find") and not console.fake and not args.get("attended"):
                raise ValueError("手臂会自己走位：先勾选“有人能断电”")
        except (SetupError, ValueError, TypeError) as exc:
            return flask.jsonify({"error": str(exc)}), 400
        job = {
            "plan": lambda: wrist.plan(n_views),
            "calibrate": lambda: wrist.calibrate(n_views),
            "view": lambda: wrist.view(keep_pose=bool(args.get("keep_pose")), another=bool(args.get("another"))),
            "shot": wrist.shot,
            "find": wrist.find,
            "view2": wrist.view_second,
        }[cmd]
        if not console.submit(JOBS[cmd], job):
            return flask.jsonify({"error": f"正在执行“{console.busy}”，这条指令没有执行（不排队）"}), 409
        return flask.jsonify({"ok": True}), 202

    return wrist
