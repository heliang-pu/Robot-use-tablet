"""The whole chain rehearsed without the bench: the console as the operator uses it, on a tablet, an arm and a camera
made in software to the real bench's numbers.

    bin/py -m tablet_tap.web.rehearsal                         # flat tablet; the model server of --vlm_url
    bin/py -m tablet_tap.web.rehearsal --slant 12 --llm scripted

Real here: the console's own code (following the tablet, slow touches, tap correction, the run page, the agent's prompt
and parsing), the camera's intrinsics, pose and park pose of ``calib/setup.json``, and the model and speech services
when they are up. Made up: the tablet (the calibration page, or a small Android of three screens that reacts to
touches), what the camera sees of it (projected with the real camera, the arm drawn as rods, blur and noise), and how
the arm differs from its model (joint zeros, a pen that is not on the axis, scatter from tap to tap). The console is
told a camera pose and a screen size that are a little wrong, as on the bench. Nothing here opens hardware.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import logging
import pathlib
import subprocess
import time
from typing import Any

import cv2
import numpy as np
import pinocchio as pin
import requests

from tablet_tap import config
from tablet_tap.calibration import page_server
from tablet_tap.core.kinematics import FINGERTIP_IN_FLANGE
from tablet_tap.core.kinematics import PenArm
from tablet_tap.core.kinematics import pose
from tablet_tap.core.look import HOLDER_BOX
from tablet_tap.core.look import MOUNTS
from tablet_tap.core.look import PEN_HALF_M
from tablet_tap.core.look import camera_in_flange
from tablet_tap.core.look import tip_in_flange
from tablet_tap.core.pinhole import CameraModel
from tablet_tap.core.screen_map import ScreenMap
from tablet_tap.robot.arm import FakeArm
from tablet_tap.robot.factory import ArmMode
from tablet_tap.robot.wrist import NOMINAL_MODEL
from tablet_tap.vision import layout as screen_layout
from tablet_tap.web import console as web
from tablet_tap.web import run_page
from tablet_tap.web import wrist_page

REPO = pathlib.Path(__file__).resolve().parents[2]
SCREENS_DIR = REPO / "tests" / "data" / "screens"
SPEECH_SAMPLE = REPO / "weights" / "SenseVoiceSmall" / "example" / "zh.mp3"
NEAR_PLACE_M = (0.245, 0.0)  # the tablet's centre without --dx_mm / --dy_mm: where only the top camera is used
PICTURE_PX = (640, 400)  # what the display shows, at about the size the camera sees it
NAV_BAR = 0.93  # the three buttons take the screen below this
NAV_AT = {"recents": 0.25, "home": 0.5, "back": 0.75}  # as config's nav has them
BEZEL_M = 0.009
TAG_SIDE_M, TAG_DROP_M = 0.036, 0.008  # printed tags lie on the stand, under the plane of the glass
ARM_RODS_M = (0.055, 0.05, 0.04, 0.035, 0.035, 0.035, 0.04, 0.008)  # base .. joint 6, gripper, pen: radii


def tablet_screen(
    centre: tuple[float, float, float], slant_deg: float, yaw_deg: float, size_m: tuple[float, float]
) -> ScreenMap:
    """A screen whose bottom edge is toward the robot, raised by ``slant_deg`` at the far edge and turned by ``yaw``."""
    yaw, slant = np.radians(yaw_deg), np.radians(slant_deg)
    turn = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    right, forward = turn @ [0.0, -1.0, 0.0], turn @ [1.0, 0.0, 0.0]
    up_slope = np.cos(slant) * forward + np.sin(slant) * np.array([0.0, 0.0, 1.0])
    origin = np.asarray(centre, np.float64) - 0.5 * size_m[0] * right + 0.5 * size_m[1] * up_slope
    return ScreenMap(origin, size_m[0] * right, -size_m[1] * up_slope, tool_x=up_slope)


class Tablet:
    """What the tablet shows and what a touch does: the calibration page in its browser, or three Android screens."""

    def __init__(self, calib: page_server.CalibState) -> None:
        self.calib = calib
        self.app, self.playing, self.track = "calib", False, 0
        self.events: list[dict[str, Any]] = []
        self._cache: dict[Any, np.ndarray] = {}
        self.screens: dict[str, tuple[np.ndarray, dict[str, tuple[float, float, float, float]]]] = {}
        for name in ("home", "player", "settings"):
            meta = json.loads((SCREENS_DIR / f"{name}.json").read_text())
            width, height = meta["size"]
            boxes = {key: (el["box"][0] / width, el["box"][1] / height * NAV_BAR, el["box"][2] / width, el["box"][3] / height * NAV_BAR)
                     for key, el in meta["elements"].items()}  # fmt: skip
            self.screens[name] = (cv2.imread(str(SCREENS_DIR / f"{name}.png")), boxes)
        calib.set_screen({"inner_w": 1280, "inner_h": 800, "screen_w": 1280, "screen_h": 800, "dpr": 2, "fullscreen": True,
                          "user_agent": "tablet_tap rehearsal tablet"})  # fmt: skip

    def show(self, app: str) -> None:
        self.app = app

    def box(self, name: str) -> tuple[float, float, float, float]:
        """(u0, v0, u1, v1) of something on the screen now, a button of the navigation bar included."""
        if name in NAV_AT:
            return NAV_AT[name] - 0.03, NAV_BAR + 0.01, NAV_AT[name] + 0.03, 0.995
        return self.screens[self.app][1][name]

    def picture(self, width: int = PICTURE_PX[0]) -> np.ndarray:
        """What the display shows, ``width`` pixels wide: about what the camera that asks can resolve."""
        size = (width, round(width * PICTURE_PX[1] / PICTURE_PX[0]))
        if self.app == "calib":
            snap = self.calib.snapshot()
            last, target = snap["last_down"], snap["target"]
            key: Any = (
                width,
                "calib",
                snap["mode"],
                None if target is None else tuple(target),
                None if last is None else last["seq"],
            )
            if key not in self._cache:
                self._cache = {
                    k: v for k, v in self._cache.items() if k[1:] == key[1:]
                }  # the same picture, other sizes
                self._cache[key] = screen_layout.render_layout(
                    screen_layout.default_layout(1.6), width, mode=snap["mode"],
                    target=None if target is None else tuple(target), dot=None if last is None else (last["u"], last["v"]))  # fmt: skip
            return self._cache[key]
        key = (width, self.app, self.playing, self.track)
        if key not in self._cache:
            self._cache = {k: v for k, v in self._cache.items() if k[1:] == key[1:]}
            image = self.screens[self.app][0].copy()
            if self.app == "player":
                if self.playing:  # the round button shows two bars instead of the triangle
                    cv2.circle(image, (912, 570), 46, (255, 255, 255), -1, cv2.LINE_AA)
                    for x in (896, 918):
                        cv2.rectangle(image, (x, 548), (x + 10, 592), (44, 26, 30), -1)
                if self.track:
                    cv2.putText(
                        image,
                        f"#{self.track + 1}",
                        (1040, 96),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
            height, width = image.shape[:2]
            body = cv2.resize(image, (width, round(height * NAV_BAR)), interpolation=cv2.INTER_AREA)
            image[:] = (18, 14, 16)
            image[: body.shape[0]] = body
            y, s = round(height * (NAV_BAR + 1) / 2), 13
            x = round(NAV_AT["recents"] * width)
            cv2.rectangle(image, (x - s, y - s), (x + s, y + s), (235, 235, 235), 3, cv2.LINE_AA)
            cv2.circle(image, (round(NAV_AT["home"] * width), y), s + 2, (235, 235, 235), 3, cv2.LINE_AA)
            x = round(NAV_AT["back"] * width)
            cv2.polylines(
                image,
                [np.array([[x + s, y - s - 2], [x - s, y], [x + s, y + s + 2]])],
                True,
                (235, 235, 235),
                3,
                cv2.LINE_AA,
            )
            self._cache[key] = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        return self._cache[key]

    def full_picture(self) -> np.ndarray:
        """The display at 1280 px wide (what a screenshot of the tablet itself would be)."""
        return cv2.resize(self.picture(), (1280, 800), interpolation=cv2.INTER_CUBIC)

    def touch(self, u: float, v: float) -> None:
        if self.app == "calib":
            self.calib.add_touch({"type": "pointerdown", "u": u, "v": v, "t_ms": 0, "pointer_type": "sim"})
            return
        before, hit = self.app, None
        if v >= NAV_BAR:
            hit = next((name for name, at in NAV_AT.items() if abs(u - at) < 0.04), None)
            if hit in ("home", "back"):
                self.app = "home"
        else:
            hit = next(
                (
                    name
                    for name, (u0, v0, u1, v1) in self.screens[self.app][1].items()
                    if u0 <= u <= u1 and v0 <= v <= v1
                ),
                None,
            )
            if self.app == "home" and hit in ("music", "settings"):
                self.app = {"music": "player", "settings": "settings"}[hit]
            elif self.app == "player" and hit == "play":
                self.playing = not self.playing
            elif self.app == "player" and hit in ("next", "prev"):
                self.track += 1 if hit == "next" else -1
            elif self.app == "player" and hit == "collapse":
                self.app = "home"
        self.events.append(
            {"t": time.time(), "app": before, "u": u, "v": v, "hit": hit, "then": self.app, "playing": self.playing}
        )

    def release(self, u: float, v: float) -> None:
        if self.app == "calib":
            self.calib.add_touch({"type": "pointerup", "u": u, "v": v, "t_ms": 0, "pointer_type": "sim"})


@dataclasses.dataclass
class Bench:
    """The bench as it really is. The console's setup says nearly, not exactly, the same."""

    pen: PenArm  # the arm's model
    model: CameraModel
    camera: np.ndarray  # (4, 4) where the camera really is
    screen: ScreenMap  # where the display really is, with its real size
    tablet: Tablet
    joint_zero_rad: np.ndarray  # the real joints are the readings plus these
    tool_offset_m: np.ndarray  # the real pen tip, in the model's pen-tip frame
    scatter_m: float = 0.0003  # from one tap to the next, per axis
    backlash_m: float = 0.0  # the pen stops this far short of where it was sent, along the way it came
    trigger_p: float = 1.0  # how often the screen notices the pen
    trigger_m: float = 0.0003  # a rubber tip needs a little pressure
    give_m: float = 0.004  # the holder bends this far before the glass stops the joints
    seed: int = 0
    arm: BenchArm | None = None
    wrist_in_flange: np.ndarray | None = None  # (4, 4) where the wrist camera really sits; the drawing says nearly so
    lamp: np.ndarray | None = None  # (2, 3) the ends of a tube lamp over the bench: it is mirrored in the glass

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def flange(self, q: np.ndarray) -> np.ndarray:
        """(4, 4) the flange (link 6) as it really is for readings ``q``."""
        model, data, _ = self.pen._kinematics()  # noqa: SLF001
        pin.forwardKinematics(model, data, np.asarray(q, np.float64) + self.joint_zero_rad)
        return data.oMi[model.njoints - 1].homogeneous

    def joints(self, q: np.ndarray) -> list[np.ndarray]:
        """Base, the six joints, the fingertips and the real pen tip, for readings ``q`` (base frame)."""
        model, data, tip = self.pen._kinematics()  # noqa: SLF001 - the one place that wants every link
        pin.forwardKinematics(model, data, np.asarray(q, np.float64) + self.joint_zero_rad)
        points = [np.zeros(3)] + [data.oMi[i].translation.copy() for i in range(1, model.njoints)]
        points.append((data.oMi[model.njoints - 1] * FINGERTIP_IN_FLANGE).translation.copy())
        points.append(pin.updateFramePlacement(model, data, tip).act(self.tool_offset_m))
        return points

    def tip(self, q: np.ndarray) -> np.ndarray:
        return self.joints(q)[-1]

    def move_tablet(self, along_right_m: float, along_up_m: float, yaw_deg: float = 0.0) -> None:
        """Somebody pushes the tablet (and its stand) on its plane."""
        s = self.screen
        right, down, normal = s.right / np.linalg.norm(s.right), s.down / np.linalg.norm(s.down), s.normal
        centre = s.point(0.5, 0.5) + along_right_m * right - along_up_m * down
        turn = cv2.Rodrigues(normal * np.radians(yaw_deg))[0]
        new_right, new_down = turn @ s.right, turn @ s.down
        self.screen = dataclasses.replace(s, origin=centre - 0.5 * new_right - 0.5 * new_down, right=new_right, down=new_down,
                                          tool_x=turn @ s.tool_x)  # fmt: skip

    def new_arm(self, q: np.ndarray) -> BenchArm:
        self.arm = BenchArm(q, self)
        return self.arm


class BenchArm(FakeArm):
    """The software arm with the bench's errors; its pen presses the bench's tablet, and the glass stops it."""

    def __init__(self, q: np.ndarray, bench: Bench) -> None:
        super().__init__(q)
        self.bench, self.down, self.ticks = bench, False, 0
        self.gripper_width_m = 0.075
        self._slip, self._last = np.zeros(2), None

    def set_grip_width(self, width_m: float) -> None:
        self.gripper_width_m = max(float(width_m), 0.0212)  # the holder's block is between the fingers

    def command(self, q: np.ndarray) -> None:
        bench, q = self.bench, np.asarray(q, np.float64)
        self.ticks += 1
        self.commands = self.commands[-1:]  # a rehearsal sends hundreds of thousands
        u, v, h = bench.screen.uv_of(bench.tip(q))
        over_glass = -0.04 <= u <= 1.04 and -0.06 <= v <= 1.06
        if not (over_glass and h < -bench.give_m):
            self.q = q.copy()  # else the glass holds the pen: the joints stay behind their target
        u, v, h = bench.screen.uv_of(bench.tip(self.q))
        pressed = h <= -bench.trigger_m and 0 <= u <= 1 and 0 <= v <= 1
        if pressed and not self.down:
            width, height = bench.screen.size_m
            self._slip = bench.rng.normal(0.0, bench.scatter_m, 2) / (width, height)
            if self._last is not None and bench.backlash_m:
                came = np.array([(u - self._last[0]) * width, (v - self._last[1]) * height])
                if np.linalg.norm(came) > 1e-4:
                    self._slip -= bench.backlash_m * came / np.linalg.norm(came) / (width, height)
            self._last = (u, v)
            if bench.rng.random() < bench.trigger_p:
                bench.tablet.touch(float(u + self._slip[0]), float(v + self._slip[1]))
        elif self.down and not pressed:
            bench.tablet.release(float(u + self._slip[0]), float(v + self._slip[1]))
        self.down = pressed


class BenchCamera:
    """The top camera: the real intrinsics and pose, looking at the bench's tablet, its stand's tags and the arm."""

    picture_px = PICTURE_PX[0]  # the detail of the display's picture that this camera can use
    near = False  # looks at the glass from close by, nearly square on: lamps over the bench are mirrored into it

    def __init__(
        self, bench: Bench, tags: bool = True, noise: float = 2.5, blur: float = 0.9, model: CameraModel | None = None
    ) -> None:
        self.bench, self.noise, self.blur = bench, noise, blur
        self.model = model = model or bench.model
        rng = np.random.default_rng(1)
        table = np.full((model.height, model.width, 3), (118, 124, 128), np.uint8)
        grain = cv2.GaussianBlur(rng.normal(0, 12, (model.height, model.width)).astype(np.float32), (0, 0), 6)
        self.table = np.clip(table + grain[..., None], 0, 255).astype(np.uint8)
        dictionary = cv2.aruco.getPredefinedDictionary(screen_layout.ARUCO_DICT)
        self.tags: dict[int, tuple[np.ndarray, tuple[float, float]]] = {}
        width, height = bench.screen.size_m
        places = {
            40: (-0.042, 0.4 * height),
            41: (width + 0.042, 0.4 * height),
            42: (0.22 * width, -0.04),
            43: (0.78 * width, -0.04),
        }
        for tag_id, centre in places.items() if tags else ():
            paper = np.full((140, 140), 255, np.uint8)
            paper[20:120, 20:120] = cv2.aruco.generateImageMarker(dictionary, tag_id, 100)
            self.tags[tag_id] = (cv2.cvtColor(paper, cv2.COLOR_GRAY2BGR), centre)
        self.hidden = 0.0  # the share of the display that the arm hid in the newest frame

    def pose(self) -> np.ndarray:
        """(4, 4) where the camera is now, in the arm's base frame."""
        return self.bench.camera

    def pixels(self, points: np.ndarray, pose: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(n, 2) pixels and (n,) depths of points in the base frame."""
        inverse = np.linalg.inv(self.pose() if pose is None else pose)
        in_camera = np.asarray(points, np.float64) @ inverse[:3, :3].T + inverse[:3, 3]
        return self.model.project(in_camera), in_camera[:, 2]

    def _paste(
        self, image: np.ndarray, patch: np.ndarray, corners_base: np.ndarray, pose: np.ndarray
    ) -> np.ndarray | None:
        quad, depth = self.pixels(corners_base, pose)
        if np.any(depth <= 0.05):
            return None
        lo = np.maximum(np.floor(quad.min(0)).astype(int) - 2, 0)
        hi = np.minimum(np.ceil(quad.max(0)).astype(int) + 2, (image.shape[1], image.shape[0]))
        if np.any(hi - lo < 2):
            return None
        h, w = patch.shape[:2]
        # pixel centres: the patch's outline is half a pixel outside its first and last samples
        source = np.float32([[-0.5, -0.5], [w - 0.5, -0.5], [w - 0.5, h - 0.5], [-0.5, h - 0.5]])
        warp = cv2.getPerspectiveTransform(source, (quad - lo).astype(np.float32))
        size = (int(hi[0] - lo[0]), int(hi[1] - lo[1]))
        seen = cv2.warpPerspective(patch, warp, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        alpha = (
            cv2.warpPerspective(np.full((h, w), 255, np.uint8), warp, size, flags=cv2.INTER_LINEAR)[..., None] / 255.0
        )
        region = image[lo[1] : hi[1], lo[0] : hi[0]]
        region[:] = (alpha * seen + (1 - alpha) * region).astype(np.uint8)
        return quad

    def read(self) -> np.ndarray:
        bench, image, pose = self.bench, self.table.copy(), self.pose()
        s = bench.screen
        (width, height), normal = s.size_m, s.normal
        right, down = s.right / width, s.down / height

        def at(x: float, y: float, z: float = 0.0) -> np.ndarray:
            return s.origin + x * right + y * down + z * normal

        for patch, (cx, cy) in self.tags.values():
            half = 0.5 * TAG_SIDE_M * 1.4  # the paper around the code
            self._paste(image, patch, np.array([at(cx - half, cy - half, -TAG_DROP_M), at(cx + half, cy - half, -TAG_DROP_M),
                                                at(cx + half, cy + half, -TAG_DROP_M), at(cx - half, cy + half, -TAG_DROP_M)]), pose)  # fmt: skip
        b = BEZEL_M
        self._paste(image, np.full((420, 640, 3), 24, np.uint8),
                    np.array([at(-b, -b), at(width + b, -b), at(width + b, height + b), at(-b, height + b)]), pose)  # fmt: skip
        quad = self._paste(
            image,
            self._mirrored_lamp(bench.tablet.picture(self.picture_px), pose),
            np.array([at(0, 0), at(width, 0), at(width, height), at(0, height)]),
            pose,
        )
        rods = np.zeros(image.shape[:2], np.uint8)
        if bench.arm is not None:
            self._arm(image, rods, pose)
        if quad is not None:
            lit = np.zeros(image.shape[:2], np.uint8)
            cv2.fillConvexPoly(lit, np.round(quad).astype(np.int32), 255)
            self.hidden = float(np.count_nonzero(rods & lit)) / max(np.count_nonzero(lit), 1)
        image = cv2.GaussianBlur(image, (0, 0), self.blur)
        grain = np.empty(image.shape, np.int16)
        cv2.randn(grain, 0, self.noise)
        return np.clip(image.astype(np.int16) + grain, 0, 255).astype(np.uint8)

    def _mirrored_lamp(self, picture: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """The display with the lamp's reflection on it, as this camera sees it: where the glass mirrors the tube
        depends on where the camera is."""
        lamp, s = self.bench.lamp, self.bench.screen
        if lamp is None or not self.near:  # on the bench the top camera's picture had no reflection (2026-11-23)
            return picture
        eye, normal = pose[:3, 3], s.normal
        height, width = picture.shape[:2]
        line = []
        for t in np.linspace(0.0, 1.0, 25):
            source = lamp[0] + t * (lamp[1] - lamp[0])
            mirrored = source - 2.0 * ((source - s.origin) @ normal) * normal
            along = ((s.origin - eye) @ normal) / ((mirrored - eye) @ normal)
            u, v, _ = s.uv_of(eye + along * (mirrored - eye))
            line.append((u * width, v * height))
        shine = np.zeros((height, width), np.uint8)
        cv2.polylines(
            shine,
            [np.round(line).astype(np.int32)],
            False,
            255,
            max(2, round(0.014 / s.size_m[0] * width)),
            cv2.LINE_AA,
        )
        shine = cv2.GaussianBlur(shine, (0, 0), max(1.0, 0.002 / s.size_m[0] * width))
        return np.clip(picture.astype(np.int16) + shine[..., None], 0, 255).astype(np.uint8)

    def _arm(self, image: np.ndarray, rods: np.ndarray, pose: np.ndarray) -> None:
        """The arm as the top camera sees it: a rod from joint to joint, the gripper and the pen."""
        points = np.array(self.bench.joints(self.bench.arm.q))
        pixels, depth = self.pixels(points, pose)
        for i, radius in enumerate(ARM_RODS_M):
            if min(depth[i], depth[i + 1]) <= 0.05:
                continue
            thick = max(1, round(2 * radius * self.model.fx / (0.5 * (depth[i] + depth[i + 1]))))
            a, c = (tuple(int(x) for x in np.round(pixels[j])) for j in (i, i + 1))
            colour = (150, 150, 155) if i == len(ARM_RODS_M) - 1 else (38, 38, 42)
            cv2.line(image, a, c, colour, thick, cv2.LINE_AA)
            cv2.line(rods, a, c, 255, thick)

    def close(self) -> None:
        pass


class BenchWristCamera(BenchCamera):
    """The camera on the wrist: it rides on the flange as it really is, sits not quite where the mount's drawing has
    it, is a little out of focus this near, and has the pen and its holder in the lower part of its picture."""

    picture_px = 1280
    near = True

    def __init__(self, bench: Bench, tags: bool = True) -> None:
        super().__init__(bench, tags, noise=2.5, blur=1.5, model=NOMINAL_MODEL)

    def pose(self) -> np.ndarray:
        return self.bench.flange(self.bench.arm.q) @ self.bench.wrist_in_flange

    def _arm(self, image: np.ndarray, rods: np.ndarray, pose: np.ndarray) -> None:
        bench = self.bench
        flange, tip = bench.flange(bench.arm.q), float(tip_in_flange(bench.pen)[2, 3])
        half = PEN_HALF_M
        for (low, high), colour in (
            (HOLDER_BOX, (36, 36, 40)),
            (((-half, -half, 0.16), (half, half, tip)), (150, 150, 155)),
        ):
            corners = np.array(
                [[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])]
            )
            pixels, depth = self.pixels(corners @ flange[:3, :3].T + flange[:3, 3], pose)
            if np.count_nonzero(depth > 0.02) >= 3:
                hull = cv2.convexHull(np.round(pixels[depth > 0.02]).astype(np.int32))
                cv2.fillConvexPoly(image, hull, colour, cv2.LINE_AA)
                cv2.fillConvexPoly(rods, hull, 255)


class BenchConsole(web.Console):
    """The console on the bench in software: its arm and its camera come from the ``Bench``."""

    def __init__(self, *args: Any, bench: Bench, camera: BenchCamera, **kwargs: Any) -> None:
        self.bench, self.bench_camera = bench, camera
        super().__init__(*args, fake=True, **kwargs)

    def connect(self, channel: str, mode: str) -> None:
        self.disconnect()
        self.fault = ""
        start = np.asarray(self.setup.park_q, np.float64)
        self.arm, self.arm_mode = web.LockedArm(self.bench.new_arm(start)), ArmMode.FAKE
        self._renew_teach()
        self.say(f"已连接机械臂 {channel}（排练：软件机械臂，带零位误差和重复性噪声）")

    def open_camera(self, spec: str) -> None:
        self.close_camera()
        self.camera = self.bench_camera
        self.say("相机已打开：排练相机（真实内参和位姿，画面是合成的）")


class ScriptedLlm:
    """Stands in for the model: it knows the tablet's state, finds what to press in the picture the camera gave, and
    answers in GUI-Owl's format. Perception is perfect; everything after it is the real chain."""

    def __init__(self, console: BenchConsole, eye: Any) -> None:
        self.console, self.eye, self.steps = console, eye, 0

    def _where(self, name: str) -> list[float]:
        """The middle of ``name`` as the rectified picture shows it, in the model's 0-1000 coordinates."""
        bench = self.console.bench
        camera, pose, rectifier = self.eye()
        u0, v0, u1, v1 = bench.tablet.box(name)
        pixel, _ = camera.pixels(bench.screen.point(0.5 * (u0 + u1), 0.5 * (v0 + v1))[None], pose)
        u, v = np.ravel(rectifier.cam_to_norm(pixel))
        return [round(1000 * float(u), 1), round(1000 * float(v), 1)]

    def predict_mm(self, messages: list[dict]) -> tuple[str, Any, Any]:
        text = json.dumps(messages, ensure_ascii=False)
        tablet = self.console.bench.tablet
        self.steps += 1
        done = {"action": "terminate", "status": "success"}
        if "桌面" in text.split("附加", 1)[0] and "回到桌面" in text:
            action, why = (
                (done, "已经在桌面")
                if tablet.app == "home"
                else ({"action": "system_button", "button": "Home"}, "按主页键")
            )
        elif "设置" in text and "打开设置" in text:
            action, why = (
                (done, "设置已打开")
                if tablet.app == "settings"
                else ({"action": "click", "coordinate": self._where("settings")}, "点设置图标")
            )
        elif "暂停" in text:
            action, why = (
                ({"action": "click", "coordinate": self._where("play")}, "点暂停")
                if tablet.playing
                else (done, "已暂停")
            )
        elif tablet.app == "home":
            action, why = {"action": "click", "coordinate": self._where("music")}, "点音乐图标"
        elif tablet.app == "player" and not tablet.playing:
            action, why = {"action": "click", "coordinate": self._where("play")}, "点播放按钮"
        else:
            action, why = done, "音乐在播放"
        answer = (
            f"Action: {why}\n<tool_call>\n" + json.dumps({"name": "mobile_use", "arguments": action}) + "\n</tool_call>"
        )
        return answer, None, None


# ---- the rehearsal ------------------------------------------------------------------------------------------------
EXPECTED = {  # instruction -> what the tablet must show afterwards
    "打开音乐，播放音乐": lambda t: t.app == "player" and t.playing,
    "暂停播放": lambda t: t.app == "player" and not t.playing,
    "回到桌面": lambda t: t.app == "home",
    "打开设置": lambda t: t.app == "settings",
}


class Rehearsal:
    def __init__(self, out: pathlib.Path, args: argparse.Namespace) -> None:
        self.out, self.args = out, args
        self.rows: list[tuple[str, str, str]] = []  # step, verdict, what was measured
        self.problems: list[str] = []
        self.data: dict[str, Any] = {"args": vars(args)}
        out.mkdir(parents=True, exist_ok=True)
        config.PROJECT_ROOT = out  # runs/ and calib/ of this rehearsal; the bench's own files are only read
        self.log_file = logging.FileHandler(out / "console.log", mode="w")
        self.log_file.setFormatter(logging.Formatter("%(asctime)s %(name)s: %(message)s", "%H:%M:%S"))
        logging.getLogger("tablet_tap").addHandler(self.log_file)
        logging.getLogger("tablet_tap").setLevel(logging.INFO)
        real = config.SetupStore(args.setup).load()
        if real.camera_model is None or real.camera_pose is None or real.park_q is None:
            raise SystemExit(f"{args.setup} 里没有相机标定或停靠位姿，排练需要它们。")
        setup = real
        setup.screen_map, setup.seed_q, setup.keyboard = None, None, None
        for name in ("live_screen", "tap_correction", "screen_map_source"):
            setup.records.pop(name, None)
        if args.park:
            setup.park_q = [float(x) for x in args.park.split(",")]
        if args.park_j1 is not None:
            setup.park_q = [args.park_j1, *setup.park_q[1:]]
        store = config.SetupStore(out / "calib" / "setup.json")
        store.save(setup, backup=False)
        believed = np.asarray(setup.camera_pose, np.float64)
        if args.top_moved:  # the camera was taken off and put back: where the setup has it is centimetres off
            args.wrist, args.camera_error_deg, args.camera_error_mm = True, 5.0, 70.0
        wrong = np.eye(4)  # the camera is not quite where the hand-eye calibration put it
        wrong[:3, :3] = cv2.Rodrigues(
            np.radians(args.camera_error_deg) * np.array([0.6, 0.7, 0.4]) / np.linalg.norm([0.6, 0.7, 0.4])
        )[0]
        wrong[:3, 3] = np.array([0.5, -0.4, 0.75]) * args.camera_error_mm * 1e-3
        size = tuple(x * (1 + args.size_error) for x in (setup.screen_size_m or [0.2369, 0.1481]))
        centre = (NEAR_PLACE_M[0] + args.dx_mm * 1e-3, NEAR_PLACE_M[1] + args.dy_mm * 1e-3,
                  args.height_mm * 1e-3 + 0.5 * size[1] * np.sin(np.radians(args.slant)))  # fmt: skip
        mount_off = np.eye(4)  # the printed mount and the camera in it are not quite the drawing
        mount_off[:3, :3] = cv2.Rodrigues(
            np.radians(1.2 * args.mount_error) * np.array([0.5, 0.7, 0.5]) / np.linalg.norm([0.5, 0.7, 0.5])
        )[0]
        mount_off[:3, 3] = np.array([0.002, -0.0015, 0.003]) * args.mount_error
        calib = page_server.CalibState()
        self.tablet = Tablet(calib)
        self.bench = Bench(
            pen=setup.pen_arm(), model=setup.camera_model, camera=believed @ wrong,
            screen=tablet_screen(centre, args.slant, args.yaw, size), tablet=self.tablet,
            joint_zero_rad=np.radians([0.3, -0.3, 0.4, 0.4, -0.5, 0.0]) * args.arm_error,
            tool_offset_m=np.array([0.0015, -0.001, 0.001]) * args.arm_error,
            scatter_m=args.scatter_mm * 1e-3, backlash_m=args.backlash_mm * 1e-3, trigger_p=args.trigger_p, seed=args.seed,
            wrist_in_flange=camera_in_flange(*MOUNTS[args.mount]) @ (np.diag([-1.0, -1.0, 1.0, 1.0]) if args.mount_turned else np.eye(4)) @ mount_off,
        )  # fmt: skip
        if (
            args.lamp
        ):  # a tube over the tablet, where the glass mirrors it into a camera looking down from the arm's side
            s = self.bench.screen
            toward = 0.99 * s.normal - 0.14 * s.down / np.linalg.norm(s.down)
            middle = s.point(0.5, 0.5) + 1.4 * toward
            self.bench.lamp = np.array([middle + [0.0, -0.6, 0.0], middle + [0.0, 0.6, 0.0]])
        self.camera = BenchCamera(self.bench, tags=not args.no_tags)
        self.con = BenchConsole(
            store,
            out / "calib" / "screen_view.json",
            calib,
            bench=self.bench,
            camera=self.camera,
            realtime=args.realtime,
        )
        app = page_server.create_app(calib)
        web.register(app, self.con, "")
        options = run_page.RunOptions(vlm_url=args.vlm_url)
        factory = (lambda: ScriptedLlm(self.con, self.eye)) if args.llm == "scripted" else None
        self.session = run_page.register(app, self.con, "", run_page.RunSession(self.con, options, factory))
        self.wrist_camera = BenchWristCamera(self.bench, tags=not args.no_tags)
        self.wrist = wrist_page.register(app, self.con, "")
        self.wrist.camera_factory = lambda spec: self.wrist_camera
        self.wrist.wrist.mount = args.mount
        if args.wrist_calibrated:  # as after another program's calibration: the camera known to a few millimetres
            known = pose(cv2.Rodrigues(np.radians([0.3, -0.2, 0.25]))[0], np.array([0.0015, -0.001, 0.002]))
            self.wrist.wrist.in_flange = (self.bench.wrist_in_flange @ known).tolist()
            args.wrist = True
        self.app, self.client = app, app.test_client()
        self.hz = setup.motion_limits().hz

    # ---- helpers -----------------------------------------------------------------------------------------------------
    def note(self, step: str, ok: bool | None, detail: str) -> None:
        verdict = {True: "通过", False: "没通过", None: "—"}[ok]
        self.rows.append((step, verdict, detail))
        print(f"[{verdict}] {step}：{detail}", flush=True)
        if ok is False:
            self.problems.append(f"{step}：{detail}")

    def state(self) -> dict[str, Any]:
        return self.client.get("/api/state").get_json()

    def idle(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while self.con.busy is not None:
            if time.monotonic() > deadline:
                raise TimeoutError(f"“{self.con.busy}”{timeout_s:g} 秒还没完")
            time.sleep(0.02)

    def cmd(self, timeout_s: float = 120.0, wait: bool = True, **body: Any) -> list[str]:
        """Send a command as the page does; the log lines it produced."""
        since, ticks = self.con._log_seq, 0 if self.bench.arm is None else self.bench.arm.ticks  # noqa: SLF001
        answer = self.client.post("/api/cmd", json=body)
        if answer.status_code != 202:
            raise RuntimeError(f"{body['cmd']} 被拒绝（{answer.status_code}）：{answer.get_json()}")
        if wait:
            self.idle(timeout_s)
        self.arm_seconds = 0.0 if self.bench.arm is None else (self.bench.arm.ticks - ticks) / self.hz
        return [line["text"] for line in self.state()["log"] if line["i"] > since]

    def wait_for(self, what: Any, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if what():
                return True
            time.sleep(0.1)
        return False

    def picture(self, kind: str, name: str) -> None:
        time.sleep(0.5)  # a frame of the scene as it is now
        answer = self.client.get(f"/api/{kind}.jpg?w=1280")
        if answer.status_code == 200:
            (self.out / name).write_bytes(answer.data)

    def map_error(self) -> dict[str, float]:
        """How far the console's screen is from the real one: at the centre and the corners, on the glass and off it."""
        believed, real = self.con.setup.screen_map, self.bench.screen
        on_glass, off_glass = [], []
        for u, v in ((0.5, 0.5), (0.05, 0.05), (0.95, 0.05), (0.95, 0.95), (0.05, 0.95)):
            du, dv, h = real.uv_of(believed.point(u, v))
            on_glass.append(np.linalg.norm((du - u) * real.right + (dv - v) * real.down) * 1e3)
            off_glass.append(h * 1e3)
        angle = np.degrees(np.arccos(np.clip(believed.normal @ real.normal, -1, 1)))
        return {"centre_mm": float(on_glass[0]), "worst_mm": float(max(on_glass)), "height_mm": float(off_glass[0]),
                "height_worst_mm": float(max(off_glass, key=abs)), "angle_deg": float(angle)}  # fmt: skip

    def eye(self) -> tuple[BenchCamera, np.ndarray, Any]:
        """The camera instructions are carried out with, where it really is when it looks, and its rectifier."""
        if self.wrist.use:
            look_q = np.asarray(self.wrist.wrist.look_q)
            return self.wrist_camera, self.bench.flange(look_q) @ self.bench.wrist_in_flange, self.wrist.rectifier()
        return self.camera, self.bench.camera, self.con.rectifier

    def picture_error_mm(self, top: bool = False) -> float:
        """How far the rectified picture puts things from where they are on the glass (the worst of a grid); of the
        camera the instructions use, or of the top camera."""
        real, worst = self.bench.screen, 0.0
        camera, pose, rectifier = (self.camera, self.bench.camera, self.con.rectifier) if top else self.eye()
        for u in np.linspace(0.05, 0.95, 5):
            for v in np.linspace(0.05, 0.95, 4):
                pixel, _ = camera.pixels(real.point(u, v)[None], pose)
                shown = np.ravel(rectifier.cam_to_norm(pixel))
                worst = max(worst, real.error_mm((u, v), (float(shown[0]), float(shown[1]))))
        return worst

    def true_uv(self, u: float, v: float) -> tuple[float, float]:
        """The point of the glass that the rectified picture shows at (u, v)."""
        camera, cam, rectifier = self.eye()
        quad = rectifier.screen_quad_cam().astype(np.float32)
        to_camera = cv2.getPerspectiveTransform(np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]), quad)
        x, y, w = to_camera @ [u, v, 1.0]
        model = camera.model
        ray = cam[:3, :3] @ [(x / w - model.cx) / model.fx, (y / w - model.cy) / model.fy, 1.0]
        s = self.bench.screen
        along = ((s.origin - cam[:3, 3]) @ s.normal) / (ray @ s.normal)
        tu, tv, _ = s.uv_of(cam[:3, 3] + along * ray)
        return tu, tv

    # ---- the steps ---------------------------------------------------------------------------------------------------
    def pages(self) -> None:
        codes = {
            path: self.client.get(path).status_code
            for path in ("/console", "/console/full", "/calib", "/run", "/api/state")
        }
        self.note(
            "页面能打开", all(code == 200 for code in codes.values()), "、".join(f"{p} {c}" for p, c in codes.items())
        )

    def connect(self) -> None:
        self.cmd(cmd="start", channel="can_l_follower", attended=True)
        s = self.state()
        self.note(
            "连接机械臂、打开相机",
            s["arm"]["connected"] and s["camera"]["open"],
            f"机械臂 {s['arm']['mode']}，相机 {'开' if s['camera']['open'] else '没开'}",
        )
        self.cmd(cmd="grip", close=False)
        lines = self.cmd(cmd="grip", close=True)
        width = self.state()["gripper_mm"]
        self.note(
            "夹爪开关：松开再夹紧",
            width is not None and 15 < width < 30,
            f"夹住 {width} mm；{lines[-1] if lines else ''}",
        )

    def locate(self) -> bool:
        seen = self.wait_for(lambda: self.con._located and self.state()["tablet"]["state"] == "tracked", 20)  # noqa: SLF001
        s = self.state()
        self.picture("camera", "01_相机画面_标定页.jpg")
        self.picture("screen", "02_拉正后的屏幕_标定页.jpg")
        if not seen:
            self.note(
                "相机认出平板",
                False,
                f"20 秒内没有认出：{s['tablet']}，看到标定码 {s['camera']['markers']} 个，手臂挡住屏幕 {self.camera.hidden:.0%}",
            )
            return False
        error = self.map_error()
        self.data["located"] = error | {
            "hidden": self.camera.hidden,
            "lean_deg": self.con.live.tilt_deg,
            "reach": s["tablet"]["reach"],
        }
        self.note("相机认出平板（只靠相机）", error["worst_mm"] < 8 and abs(error["height_worst_mm"]) < 8,
                  f"屏幕上偏 {error['centre_mm']:.1f} mm（角上最大 {error['worst_mm']:.1f}），高度差 {error['height_mm']:+.1f} mm"
                  f"（最大 {error['height_worst_mm']:+.1f}），倾角差 {error['angle_deg']:.2f}°；看到 {s['tablet']['seen']} 个标定码；"
                  f"笔的倾角 {self.con.live.tilt_deg:g}°" + (f"；提示：{s['tablet']['reach']}" if s["tablet"]["reach"] else ""))  # fmt: skip
        self.note("停靠位挡不挡屏幕", self.camera.hidden < 0.02, f"手臂停在停靠位时挡住屏幕的 {self.camera.hidden:.0%}")
        if self.camera.tags:
            learned = self.wait_for(
                lambda: len(self.con.live.tags) >= 2 and min(n for _, n in self.con.live.tags.values()) >= 8, 12
            )
            self.note(
                "学到支架上的贴码",
                learned,
                f"学到 {sorted(self.con.live.tags)}（画面里贴了 {sorted(self.camera.tags)}）",
            )
        return True

    def calibrate(self) -> bool:
        lines = self.cmd(cmd="calibrate_taps", timeout_s=600)
        result = self.con.results.get("calibration")
        if result is None:
            self.note("一键校准精度", False, "没有做完：" + " / ".join(lines[-3:]))
            return False
        error = self.map_error()
        self.data["calibration"] = result | {"arm_seconds": self.arm_seconds, "map": error}
        self.note("一键校准精度（检验点没参加学习）", result["mean_mm"] <= 1.0 and result["max_mm"] <= 2.0,
                  f"检验 {result['n']} 点：平均 {result['mean_mm']:.2f} mm，95% {result['p95_mm']:.2f}，最大 {result['max_mm']:.2f}，"
                  f"≤1 mm {result['within_1mm']}/{result['n']}；校准前平均 {result['before_mean_mm']:.2f} mm；{result['model']}；"
                  f"机械臂运动约 {self.arm_seconds / 60:.1f} 分钟")  # fmt: skip
        self.note(
            "慢速触点之后的屏幕平面",
            None,
            f"高度差 {error['height_mm']:+.2f} mm（最大 {error['height_worst_mm']:+.2f}），倾角差 {error['angle_deg']:.2f}°",
        )
        return True

    def rescue(self) -> bool:
        """What the console tells the operator to do when the slow touches meet nothing: leave the pen over the middle of
        the screen and lower it by hand, in steps, until the screen reports it."""
        self.cmd(cmd="point", on="screen", x=0.5, y=0.5, park=False)
        seq, steps = self.con.page.last_down_seq(), []
        for _ in range(40):
            height = self.state()["place"]["height_mm"]
            step = "3" if height > 9 else "2"  # 5 mm while well above where the camera has the screen, then 1 mm
            lines = self.cmd(cmd="key", key="f", step=step)
            steps.append(5 if step == "3" else 1)
            if self.con.page.last_down_seq() > seq or self.state()["place"]["height_mm"] < -3:
                break  # touched, or as deep as a person watching would go
        touched = self.con.page.last_down_seq() > seq and bool(self.con.live.contacts)
        below = -self.state()["place"]["height_mm"] if not touched else None
        self.cmd(cmd="park")
        self.note("没碰到时的补救：手动一步步下探", touched,
                  f"按“下” {len(steps)} 次（共 {sum(steps)} mm）" + ("碰到屏幕，" + next((x for x in reversed(lines) if "笔碰到屏幕" in x), "控制台记下了触点")
                                                                 if touched else f"仍没有触摸（已到估计面以下 {below:.0f} mm）"))  # fmt: skip
        return touched

    def wrist_cmd(self, timeout_s: float = 300.0, **body: Any) -> list[str]:
        since = self.con._log_seq  # noqa: SLF001
        answer = self.client.post("/api/wrist/cmd", json=body | {"attended": True})
        if answer.status_code not in (200, 202):
            raise RuntimeError(f"腕部相机 {body['cmd']} 被拒绝（{answer.status_code}）：{answer.get_json()}")
        if answer.status_code == 202:
            self.idle(timeout_s)
        return [line["text"] for line in self.state()["log"] if line["i"] > since]

    def wrist_find(self) -> bool:
        """The top camera was moved: what it says of the tablet is far off. The wrist camera looks down from over the
        place the tablet was asked to be put at, and places it."""
        self.wait_for(lambda: self.con._located, 15)  # noqa: SLF001
        time.sleep(1.0)
        before = self.map_error() if self.con.setup.screen_map is not None else None
        self.note("顶部相机挪过以后它自己估的平板位置", None,
                  "没有认出平板" if before is None else f"屏幕上偏 {before['centre_mm']:.0f} mm，高度差 {before['height_mm']:+.0f} mm（旧标定已经不对）")  # fmt: skip
        lines = self.wrist_cmd(cmd="find")
        if "find" not in self.wrist.results:
            self.note("腕部相机：自己找平板", False, " / ".join(lines[-2:]))
            return False
        error = self.map_error()
        self.data["wrist_find"] = self.wrist.results["find"] | error
        self.note("腕部相机：自己找平板（相机位置按图纸）", error["worst_mm"] < 15 and abs(error["height_worst_mm"]) < 15 and self.con.setup.camera_pose is None,
                  f"看到 {self.wrist.results['find']['markers']} 个码；屏幕上偏 {error['centre_mm']:.1f} mm（角上最大 {error['worst_mm']:.1f}），"
                  f"高度差 {error['height_mm']:+.1f} mm，倾角差 {error['angle_deg']:.1f}°；顶部相机暂时不参与")  # fmt: skip
        self.picture_of_wrist("07a_腕部相机_找平板.jpg", "picture")
        return True

    def top_recovered(self) -> bool:
        """After the wrist camera's calibration the top camera's place is worked out from it, and it follows the tablet
        again."""
        pose = self.con.setup.camera_pose
        if pose is None:
            self.note("顶部相机的位置由腕部标定算出", False, "没有算出来：" + str(self.wrist.results.get("top")))
            return False
        off_mm, off_deg = wrist_page.mount_error(np.asarray(pose), self.bench.camera)
        seen = self.wait_for(lambda: self.con._located and self.state()["tablet"]["state"] == "tracked", 15)  # noqa: SLF001
        time.sleep(1.5)
        error = self.map_error()
        self.data["top_recovered"] = {"off_true_mm": off_mm, "off_true_deg": off_deg} | error
        self.note("顶部相机的位置由腕部标定算出，重新跟踪平板", seen and error["worst_mm"] < 10 and abs(error["height_worst_mm"]) < 6,
                  f"算出的位置离真实位置 {off_mm:.0f} mm、{off_deg:.1f}°（标定前差 70 mm、5°）；现在估的平板：屏幕上偏 {error['centre_mm']:.1f} mm，"
                  f"高度差 {error['height_mm']:+.1f} mm（最大 {error['height_worst_mm']:+.1f}）")  # fmt: skip
        if self.camera.tags:
            learned = self.wait_for(
                lambda: len(self.con.live.tags) >= 2 and min(n for _, n in self.con.live.tags.values()) >= 8, 12
            )
            self.note("学到支架上的贴码", learned, f"学到 {sorted(self.con.live.tags)}")
        return seen

    def wrist_calibrate(self) -> bool:
        """The top camera has placed the tablet roughly; the wrist camera is calibrated on the flange from there."""
        codes = {path: self.client.get(path).status_code for path in ("/api/wrist/state",)}
        lines = self.wrist_cmd(cmd="plan")
        plan = self.wrist.results.get("plan")
        self.note("腕部相机：试算看屏位和标定位姿（不动）", plan is not None and all(c == 200 for c in codes.values()),
                  next((x for x in lines if "看屏位" in x), lines[-1] if lines else "没有结果"))  # fmt: skip
        if plan is None:
            return False
        ticks = self.bench.arm.ticks
        lines = self.wrist_cmd(cmd="calibrate", timeout_s=900)
        seconds = (self.bench.arm.ticks - ticks) / self.hz
        found = self.wrist.wrist.in_flange
        if found is None:
            self.note("腕部相机：手眼标定", False, " / ".join(lines[-2:]))
            return False
        real, result = self.bench.wrist_in_flange, self.wrist.results["calibration"]
        off_mm, off_deg = wrist_page.mount_error(np.asarray(found), real)
        self.data["wrist_calibration"] = result | {
            "off_true_mm": off_mm,
            "off_true_deg": off_deg,
            "arm_seconds": seconds,
        }
        self.note("腕部相机：手眼标定", result["held_out_px"] < 8.0 and off_mm < 6.0 and off_deg < 1.0,
                  f"{result['views']} 个位姿，拟合残差 {result['rms_px']:.2f} px，没参加拟合的 {result['held_out_px']:.2f} px；"
                  f"和真实安装位置差 {off_mm:.1f} mm、{off_deg:.2f}°（真实位置离图纸 {wrist_page.mount_error(real, camera_in_flange(*MOUNTS[self.args.mount]))[0]:.1f} mm）；"
                  f"它看到的屏幕和顶部相机估的四角差 {max(result['screen_gap_mm']):.1f} mm；机械臂运动约 {seconds / 60:.1f} 分钟")  # fmt: skip
        self.picture_of_wrist("07_腕部相机_标定时最后一张.jpg", "picture")
        return True

    def picture_of_wrist(self, name: str, kind: str) -> None:
        answer = self.client.get(f"/api/wrist/{kind}.jpg")
        if answer.status_code == 200:
            (self.out / name).write_bytes(answer.data)

    def wrist_view(self, name: str) -> bool:
        """The look pose's picture, laid over the screen; from then on instructions look through the wrist camera."""
        lines = self.wrist_cmd(cmd="view")
        view = self.wrist.wrist.records.get("view")
        if view is None or self.wrist.moved():
            self.note(name, False, " / ".join(lines[-2:]))
            return False
        self.wrist_cmd(cmd="use", on=True)
        error = self.picture_error_mm()
        self.data.setdefault("wrist_views", []).append(
            {k: v for k, v in view.items() if k != "tablet_pose"} | {"picture_error_mm": error}
        )
        edge = view["edge_px"]
        self.note(name, error < 1.0 and view["width_px"] > 800,
                  ("" if "shown" not in view else f"两个角度拼（第一个看到屏幕的 {view['shown']:.0%}）；")
                  + f"屏幕在画面里 {view['width_px']:.0f} × {view['height_px']:.0f} px（顶部相机约 455 px 宽），"
                  f"边缘过渡 {'量不出' if edge is None else f'{edge:.1f} px'}；拉正后的画面和真实屏幕最大差 {error:.2f} mm；"
                  f"手眼标定预测的画面和实际差 {view['hand_eye_gap_mm']:.1f} mm")  # fmt: skip
        self.picture_of_wrist("08_腕部相机_看屏位画面.jpg", "picture")
        self.picture_of_wrist("09_腕部相机_拉正后的屏幕.jpg", "screen")
        return True

    def wrist_glare(self) -> None:
        """A lamp is mirrored in the glass from where the look pose looks: the console says so, and the picture is taken
        again from other angles until the screen is clear."""
        first = self.wrist.wrist.records["view"]
        said = any("反光" in line["text"] for line in self.state()["log"])
        self.picture_of_wrist("09a_腕部相机_有反光.jpg", "screen")
        self.note("腕部相机：认出屏幕上的灯管反光", first["glare"] > wrist_page.GLARE_SHARE and said,
                  f"暗处有 {first['glare']:.1%} 发亮，认出 {first['markers']} 个码")  # fmt: skip
        tries, view = 0, first
        while tries < 6 and (view["glare"] > wrist_page.GLARE_SHARE or view["markers"] < 6):
            tries += 1
            lines = self.wrist_cmd(cmd="view", another=True)
            view = self.wrist.wrist.records.get("view") or view
            if any(line.startswith("✗") for line in lines):
                break
        clear = view["glare"] <= wrist_page.GLARE_SHARE and view["markers"] == 6
        told = any("都试过了" in line["text"] for line in self.state()["log"])
        error = self.picture_error_mm()
        self.data["wrist_glare"] = {"first": first["glare"], "tries": tries, "last": view["glare"], "width_px": view["width_px"],
                                    "cleared": clear, "told": told, "picture_error_mm": error}  # fmt: skip
        # a tube right where the glass mirrors it cannot be dodged from the few poses the arm has: saying so is the answer
        self.note("腕部相机：换角度躲反光（躲不开就明说）", (clear or told) and error < 2.0,
                  f"换了 {tries} 次：" + ("躲开了" if clear else "躲不开，控制台提示把灯挡一下或挪平板") +
                  f"；最后暗处发亮 {view['glare']:.1%}，认出 {view['markers']} 个码，屏幕 {view['width_px']:.0f} px 宽，"
                  f"画面和真实屏幕最大差 {error:.2f} mm")  # fmt: skip
        self.picture_of_wrist("09b_腕部相机_换角度后.jpg", "screen")
        if clear:
            return
        second = None
        for _ in range(3):  # a second look whose reflection lies elsewhere mends the first
            self.wrist_cmd(cmd="view2")
            second = self.wrist.wrist.records.get("view2")
            if second and second["glare"]["merged"] <= wrist_page.GLARE_SHARE:
                break
        mended = bool(second) and second["glare"]["merged"] <= wrist_page.GLARE_SHARE
        if mended:
            self.wrist_cmd(cmd="merge", on=True)
        self.data["wrist_glare"]["two_looks"] = None if not second else second["glare"]
        self.note("腕部相机：两个角度合成去掉反光", mended and self.wrist.merge,
                  "没有拍成第二个角度" if not second else
                  f"暗处发亮：第一个角度 {second['glare']['first']:.1%}，第二个 {second['glare']['second']:.1%}，合成后 {second['glare']['merged']:.2%}")  # fmt: skip
        self.picture_of_wrist("09c_腕部相机_两个角度合成.jpg", "screen")

    def wrist_after_nudge(self) -> None:
        """The tablet was pushed by 5 mm while it showed its home screen: the wrist's picture follows by what the top
        camera saw. Pushed out of the look pose's picture, it says so and refuses; with the calibration page back on,
        taking the picture again puts things right."""
        error = self.picture_error_mm() if self.wrist.moved() is None else float("nan")
        self.wrist_cmd(cmd="shot")
        self.picture_of_wrist("10_腕部相机_模型看到的屏幕.jpg", "screen")
        shot = self.client.get("/api/wrist/screen.jpg")
        if shot.status_code == 200:  # is it the display, all of it: no pen, no holder, no reflection left in it
            seen = cv2.imdecode(np.frombuffer(shot.data, np.uint8), cv2.IMREAD_COLOR)
            real = cv2.resize(self.tablet.full_picture(), seen.shape[1::-1], interpolation=cv2.INTER_AREA)
            gap = cv2.absdiff(cv2.GaussianBlur(seen, (0, 0), 3), cv2.GaussianBlur(real, (0, 0), 3)).mean(axis=2)
            worst = float(cv2.resize(gap, (16, 10), interpolation=cv2.INTER_AREA).max())
            self.data["wrist_shot_worst_block"] = worst
            self.note("腕部相机：模型看到的就是屏幕（没有笔、没有反光）", worst < 45,
                      f"和平板真实画面比，最差的一小块平均差 {worst:.0f} 个灰度（笔或反光留在画面里会到 60 以上）"
                      + ("；两个角度拼的" if self.wrist.merge else ""))  # fmt: skip
        self.data["wrist_after_nudge_mm"] = error
        self.note(
            "腕部相机：平板被碰动 5 mm 后画面跟着走（不重拍）",
            error < 1.0,
            f"拉正后的画面和真实屏幕最大差 {error:.2f} mm",
        )
        if self.args.quick:
            return
        self.bench.move_tablet(0.02, 0.0)
        self.wait_for(lambda: self.wrist.moved() is not None, 10)
        stale = self.wrist.moved()
        refused = self.client.post("/api/run/start", json={"instruction": "回到桌面", "max_steps": 2})
        self.note("腕部相机：平板被推开 20 mm 后不再用旧画面", stale is not None and refused.status_code == 400,
                  f"{stale or '没有发现挪动'}；这时执行指令：{(refused.get_json() or {}).get('error', refused.status_code)}")  # fmt: skip
        shown = self.tablet.app
        self.tablet.show("calib")
        self.wait_for(lambda: self.state()["tablet"]["seen"] >= 4 and self.state()["tablet"]["state"] == "tracked", 8)
        time.sleep(1.5)
        self.wrist_view("腕部相机：重新去看屏位拍画面")
        self.tablet.show(shown)
        time.sleep(1.0)

    def clicks(self, name: str, n: int = 8) -> None:
        rng, errors, missed, refused = np.random.default_rng(self.args.seed + len(self.rows)), [], 0, []
        for u, v in rng.uniform(0.06, 0.94, (n, 2)):
            before = self.state()["accuracy"]["n"]
            lines = self.cmd(cmd="point", on="screen", x=float(u), y=float(v), park=False)
            accuracy = self.state()["accuracy"]
            if accuracy["n"] == before:  # the console did not move: what it said instead
                refused.append(next((x for x in lines if x.startswith("✗")), "没有执行"))
            elif accuracy["last"]["error_mm"] is None:
                missed += 1
            else:
                errors.append(accuracy["last"]["error_mm"])
        self.cmd(cmd="park")
        self.data.setdefault("clicks", {})[name] = {"errors": errors, "missed": missed, "refused": refused}
        said = (f"；{missed} 次没触发" if missed else "") + (
            f"；{len(refused)} 次被拒绝（{refused[0]}）" if refused else ""
        )
        if not errors:
            self.note(name, False, f"{n} 次里没有一次点到" + said)
            return
        self.note(name, float(np.mean(errors)) <= 1.0 and missed == 0 and not refused,
                  f"{len(errors)} 次：平均 {np.mean(errors):.2f} mm，最大 {np.max(errors):.2f}，≤1 mm {int(np.sum(np.array(errors) <= 1.0))}/{len(errors)}" + said)  # fmt: skip

    def swipe(self) -> None:
        since = self.con.page.last_down_seq()
        wanted = ((0.3, 0.62), (0.72, 0.62))
        self.cmd(
            cmd="swipe", u=wanted[0][0], v=wanted[0][1], u2=wanted[1][0], v2=wanted[1][1], duration_s=0.6, park=True
        )
        events = self.con.page.touches(since)
        down = next((e for e in events if e["type"] == "pointerdown"), None)
        up = next((e for e in reversed(events) if e["type"] == "pointerup"), None)
        if down is None or up is None:
            self.note("滑动", False, f"屏幕没有收到完整的一划（事件 {[e['type'] for e in events]}）")
            return
        real = self.bench.screen
        start, end = real.error_mm(wanted[0], (down["u"], down["v"])), real.error_mm(wanted[1], (up["u"], up["v"]))
        length = real.error_mm((down["u"], down["v"]), (up["u"], up["v"]))
        self.note(
            "滑动（从左往右 100 mm）",
            start < 1.5 and end < 1.5,
            f"起点偏 {start:.2f} mm，终点偏 {end:.2f} mm，划过 {length:.0f} mm",
        )

    def moved_while_busy(self) -> None:
        self.con.realtime = True  # the arm takes its real time, so that the camera can see the tablet go
        since, before = self.con._log_seq, self.con.setup.screen_map.point(0.5, 0.5)  # noqa: SLF001
        self.client.post("/api/cmd", json={"cmd": "taptest", "nu": 5, "nv": 4, "margin": 0.1, "repeat": 1})
        time.sleep(6.0)
        started = time.monotonic()
        self.bench.move_tablet(0.012, 0.0)
        try:
            self.idle(30)
        except TimeoutError:
            self.client.post("/api/stop")
            self.idle(30)
        took = time.monotonic() - started
        self.con.realtime = self.args.realtime
        lines = [line["text"] for line in self.state()["log"] if line["i"] > since]
        stopped = any("平板被挪动了" in line for line in lines)
        self.note(
            "执行中平板被推开 12 mm",
            stopped,
            (f"{took:.1f} 秒后停下" if stopped else "没有自己停下")
            + "；"
            + next((x for x in lines if "挪动" in x), lines[-1] if lines else ""),
        )
        self.cmd(cmd="park")

        def shift_mm() -> float:
            return float(np.linalg.norm(self.con.setup.screen_map.point(0.5, 0.5) - before) * 1e3)

        followed = self.wait_for(lambda: abs(shift_mm() - 12.0) < 1.5, 15)
        self.note("停下后跟上新位置", followed, f"控制台把屏幕挪了 {shift_mm():.1f} mm（实际推了 12 mm），触点保留 {len(self.con.live.contacts)} 个，"
                  f"修正样本 {len(self.con.correction.samples)} 个")  # fmt: skip

    def android(self) -> None:
        self.tablet.show("home")
        time.sleep(2.5)
        s = self.state()
        self.picture("camera", "03_相机画面_桌面.jpg")
        self.picture("screen", "04_模型看到的屏幕_桌面.jpg")
        cv2.imwrite(str(self.out / "04b_平板真实画面_桌面.jpg"), self.tablet.full_picture())
        error = self.picture_error_mm(top=True)
        self.data["android"] = {"picture_error_mm": error, "tablet": s["tablet"], "hidden": self.camera.hidden}
        self.note("离开标定页后（桌面）画面对得准不准", error < 1.5, f"拉正后的画面和真实屏幕最大差 {error:.2f} mm；平板状态 {s['tablet']['state']}，"
                  f"标定码 {s['tablet']['seen']} 个，贴码 {s['tablet'].get('tags', 0)} 个；手臂挡住屏幕 {self.camera.hidden:.0%}")  # fmt: skip
        self.bench.move_tablet(0.004, -0.003, 0.5)
        followed = self.wait_for(lambda: self.picture_error_mm(top=True) < 1.5, 10)
        error = self.picture_error_mm(top=True)
        self.picture("screen", "05_模型看到的屏幕_挪动后.jpg")
        self.data["android"]["after_nudge_mm"] = error
        self.note(
            "桌面状态下平板被碰动 5 mm、0.5°（" + ("只剩贴码可跟" if self.camera.tags else "没有码，靠机身边缘跟") + "）",
            followed,
            f"跟上后画面最大差 {error:.2f} mm；修正样本还有 {len(self.con.correction.samples)} 个",
        )

    def speech(self) -> str | None:
        try:
            requests.get("http://127.0.0.1:8803/health", timeout=2).raise_for_status()
        except requests.RequestException:
            self.note("语音识别", None, "语音服务没有启动，跳过")
            return None
        wav = self.out / "speech.wav"
        made = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(SPEECH_SAMPLE), "-ar", "16000", "-ac", "1", str(wav)],
                              capture_output=True, check=False)  # fmt: skip
        if made.returncode != 0:
            self.note("语音识别", None, "没有样例录音（或没有 ffmpeg），跳过")
            return None
        started = time.monotonic()
        answer = self.client.post("/api/run/transcribe", data=wav.read_bytes(), content_type="audio/wav")
        text = (answer.get_json() or {}).get("text", "")
        self.note(
            "语音识别（SenseVoice 的样例录音走网页接口）",
            answer.status_code == 200 and bool(text),
            f"{time.monotonic() - started:.1f} 秒，识别为“{text}”",
        )
        return text

    def instruct(self, instruction: str) -> None:
        tablet, first = self.tablet, len(self.tablet.events)
        started = time.monotonic()
        answer = self.client.post("/api/run/start", json={"instruction": instruction, "max_steps": self.args.max_steps})
        if answer.status_code != 202:
            self.note(f"指令“{instruction}”", False, f"没有开始：{answer.get_json()}")
            return
        run_id = answer.get_json()["id"]
        self.idle(self.args.max_steps * 240)
        run = self.client.get(f"/api/run/state?id={run_id}").get_json()["run"]
        took, steps, events = time.monotonic() - started, run["steps"], tablet.events[first:]
        lands, taps = [], iter(events)
        for step in steps:
            line = f"第 {step['step']} 步（模型 {step['llm_s']:.0f} 秒）：{step['did']}"
            point = (
                next((p for p in step["coords"].values() if isinstance(p, (list, tuple)) and len(p) == 2), None)
                if step["action"] in ("click", "long_press")
                else None
            )
            if step["action"] in ("click", "long_press", "system_button"):
                event = next(taps, None)
                if event is None:
                    line += " → 屏幕没有反应"
                else:
                    line += f" → 碰到“{event['hit'] or '空白处'}”"
                    if point is not None:
                        miss = self.bench.screen.error_mm(
                            self.true_uv(float(point[0]), float(point[1])), (event["u"], event["v"])
                        )
                        lands.append(miss)
                        line += f"，离模型指的点 {miss:.2f} mm"
            print("    " + line, flush=True)
            step["line"] = line
        check = EXPECTED.get(instruction)
        reached = None if check is None else bool(check(tablet))
        self.data.setdefault("runs", []).append({"instruction": instruction, "id": run_id, "status": run["status"], "answer": run["answer"],
                                                 "seconds": took, "reached": reached, "land_mm": lands, "steps": [s["line"] for s in steps],
                                                 "events": events})  # fmt: skip
        ok = run["status"] == "success" and reached is not False
        self.note(f"指令“{instruction}”", ok,
                  f"{run_page.STATUS_TEXT.get(run['status'], run['status'])}，{len(steps)} 步，{took:.0f} 秒；平板最后是 {tablet.app}"
                  + ("（在播放）" if tablet.playing else "") + (f"；落点离模型指的点 平均 {np.mean(lands):.2f} mm、最大 {np.max(lands):.2f} mm" if lands else "")
                  + (f"；{run['answer']}" if run["answer"] and run["status"] != "success" else ""))  # fmt: skip

    def run(self) -> int:
        try:
            self.pages()
            self.connect()
            if self.args.turn_park:  # the way out when only the top camera is used: the park pose swung aside
                lines = self.cmd(cmd="park_turn", j1_deg=40, attended=True)
                turned = abs(self.con.setup.park_q[0] - np.radians(40)) < 1e-6
                self.note("停靠位左转到 40°（网页按钮）", turned, lines[-1] if lines else "没有执行")
            # the rescue is for slow touches that met nothing at all, not for a pen the screen misses now and then
            located = self.wrist_find() if self.args.top_moved else self.locate()
            wrist = located and self.args.wrist and (self.args.wrist_calibrated or self.wrist_calibrate())
            if self.args.top_moved:
                located = wrist and self.top_recovered()
            if located and (self.calibrate() or (not self.con.live.contacts and self.rescue() and self.calibrate())):
                self.clicks("网页上点 8 个随机位置")
                self.swipe()
                if not self.args.quick:
                    self.moved_while_busy()
                    self.clicks("挪动之后不重新校准，再点 8 个位置")
                wrist = wrist and self.wrist_view("腕部相机：去看屏位标定画面")
                if wrist and self.args.lamp:
                    self.wrist_glare()
                self.android()
                if wrist:
                    self.wrist_after_nudge()
                heard = self.speech()
                if heard is not None:
                    self.data["heard"] = heard
                missing = self.session.missing(self.session.probe.check())
                if missing and self.args.llm != "scripted":
                    self.note("执行指令", False, "还不能执行：" + "；".join(missing))
                else:
                    for instruction in self.args.instructions:
                        self.instruct(instruction)
                        self.picture("screen", f"06_指令后_{instruction}.jpg")
        finally:
            self.con.close()
            self.report()
            logging.getLogger("tablet_tap").removeHandler(self.log_file)
            self.log_file.close()
        return 1 if self.problems else 0

    def serve(self, port: int) -> int:
        """The bench in software behind the real pages, for a person (or a browser test) to operate; /sim switches what
        the tablet shows and pushes it."""
        import flask  # noqa: PLC0415
        from werkzeug.serving import make_server  # noqa: PLC0415

        self.con.realtime = not self.args.fast
        app, bench, tablet = self.app, self.bench, self.tablet

        @app.get("/sim")
        def sim_page() -> Any:
            return SIM_PAGE

        @app.post("/sim/do")
        def sim_do() -> Any:
            data = flask.request.get_json(silent=True) or {}
            if data.get("show") in ("calib", "home", "player", "settings"):
                tablet.show(data["show"])
            if "push" in data:
                right, up, yaw = (float(x) for x in data["push"])
                bench.move_tablet(right * 1e-3, up * 1e-3, yaw)
            return flask.jsonify({"app": tablet.app, "playing": tablet.playing, "events": tablet.events[-5:]})

        print(
            f"模拟台架已启动：操作页 http://<这台电脑的地址>:{port}/console ，平板和推动的开关 http://<这台电脑的地址>:{port}/sim"
        )
        server = make_server("0.0.0.0", port, app, threaded=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.con.close()
        return 0

    def report(self) -> None:
        a = self.args
        lines = [f"# 全流程排练（不连真机） {datetime.datetime.now():%Y-%m-%d %H:%M}", "",
                 f"平板：倾斜 {a.slant:g}°、转 {a.yaw:g}°；相机标定误差 {a.camera_error_deg:g}° / {a.camera_error_mm:g} mm；屏幕尺寸误差 {a.size_error:.1%}；"
                 f"机械臂误差 ×{a.arm_error:g}；每次点按的随机偏差 σ={a.scatter_mm:g} mm，回差 {a.backlash_mm:g} mm；停靠位 {np.round(self.con.setup.park_q, 2).tolist()}；触发概率 {a.trigger_p:g}；模型：{a.llm}（{a.vlm_url if a.llm != 'scripted' else '替身'}）", "",
                 "## 没通过的" if self.problems else "## 全部通过", *[f"- {p}" for p in self.problems], "",
                 "## 每一步", "", "| 步骤 | 结果 | 数据 |", "|---|---|---|",
                 *[f"| {step} | {verdict} | {detail.replace('|', '/')} |" for step, verdict, detail in self.rows], ""]  # fmt: skip
        for run in self.data.get("runs", []):
            lines += [f"### 指令“{run['instruction']}”（{run['id']}）", "", *[f"- {s}" for s in run["steps"]], ""]
        (self.out / "report.md").write_text("\n".join(lines))
        (self.out / "result.json").write_text(json.dumps(self.data, indent=1, ensure_ascii=False, default=str))
        print(f"\n报告：{self.out / 'report.md'}")


SIM_PAGE = """<!doctype html><meta charset="utf-8"><title>模拟台架</title>
<body style="font:16px sans-serif;background:#0f1218;color:#d7dde8;padding:24px">
<h3>模拟台架（没有连真机）</h3>
<p>平板显示：<button onclick="go({show:'calib'})">标定页</button> <button onclick="go({show:'home'})">安卓桌面</button></p>
<p>推一下平板：<button onclick="go({push:[5,0,0]})">右 5 mm</button> <button onclick="go({push:[0,5,0]})">前 5 mm</button>
<button onclick="go({push:[12,0,0]})">右 12 mm</button> <button onclick="go({push:[0,0,1]})">转 1°</button></p>
<pre id="out"></pre><p><a style="color:#7fb4ff" href="/console">回到操作页</a></p>
<script>
async function go(what){const r=await fetch('/sim/do',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(what)});
document.getElementById('out').textContent=JSON.stringify(await r.json(),null,1)}
go({});setInterval(()=>go({}),1500)
</script>"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="全流程排练：不连机械臂、相机和平板，把从连接到执行指令走一遍")
    parser.add_argument(
        "--setup", default=str(config.DEFAULT_SETUP_PATH), help="真台架的配置（只读）：相机内参和位姿、停靠位姿、速度"
    )
    parser.add_argument("--out", default="", help="结果目录，默认 runs/rehearsal_<时间>")
    parser.add_argument("--slant", type=float, default=0.0, help="平板支架的倾角（度）")
    parser.add_argument("--yaw", type=float, default=2.0, help="平板摆歪的角度（度）")
    parser.add_argument("--dx_mm", type=float, default=0.0, help="平板中心离推荐位置：往前（毫米）")
    parser.add_argument("--dy_mm", type=float, default=0.0, help="平板中心离推荐位置：往左（毫米）")
    parser.add_argument("--height_mm", type=float, default=0.0, help="屏幕近端的高度（毫米，机械臂底座坐标）")
    parser.add_argument("--camera_error_deg", type=float, default=0.25)
    parser.add_argument("--camera_error_mm", type=float, default=2.5)
    parser.add_argument("--size_error", type=float, default=0.002, help="屏幕真实尺寸比配置里的大多少")
    parser.add_argument("--arm_error", type=float, default=1.0, help="机械臂零位和笔尖偏心的误差倍数")
    parser.add_argument("--scatter_mm", type=float, default=0.3, help="同一个点每次点按的随机偏差（每个方向的标准差）")
    parser.add_argument("--trigger_p", type=float, default=1.0, help="笔碰到屏幕时被识别的概率")
    parser.add_argument("--park", default="", help="换一个停靠位姿：六个关节角（弧度），逗号分开")
    parser.add_argument(
        "--park_j1", type=float, default=None, help="停靠位姿只改 1 号关节（弧度，正值往左转），别的不变"
    )
    parser.add_argument("--backlash_mm", type=float, default=0.0, help="回差：笔比目标少走这么多，方向随来路而变")
    parser.add_argument("--no_tags", action="store_true", help="支架上不贴码")
    parser.add_argument(
        "--llm", choices=("server", "scripted"), default="server", help="server：--vlm_url 的模型；scripted：替身"
    )
    parser.add_argument(
        "--vlm_url", default="http://127.0.0.1:8822/v1", help="模型服务（默认是 CPU 上的那个，不是 8802）"
    )
    parser.add_argument("--max_steps", type=int, default=6)
    parser.add_argument("--quick", action="store_true", help="跳过“执行中被推开”那一步（它按真实速度走）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--serve",
        type=int,
        default=0,
        metavar="PORT",
        help="不自动走流程，而是把模拟台架挂在这个端口上，让人在浏览器里照常操作（/console，/sim 切换平板画面）",
    )
    parser.add_argument(
        "--wrist", action="store_true", help="腕部相机也走一遍：手眼标定、看屏位，指令改用腕部相机看屏幕"
    )
    parser.add_argument(
        "--top_moved",
        action="store_true",
        help="顶部相机被挪过（旧标定差 7 cm、5°）：由腕部相机找平板、标定，再反算顶部相机的位置",
    )
    parser.add_argument(
        "--mount_error", type=float, default=1.0, help="腕部相机实际安装位置离图纸的误差倍数（1 = 约 4 mm、1.2°）"
    )
    parser.add_argument("--mount_turned", action="store_true", help="腕部相机在支架里倒着装（相对图纸绕光轴转半圈）")
    parser.add_argument("--mount", choices=sorted(MOUNTS), default="tap", help="腕部相机装在哪种支架上（kit：原装，笔在画面中间）")
    parser.add_argument("--wrist_calibrated", action="store_true", help="腕部相机已经标定过（跳过手眼标定这一步）")
    parser.add_argument("--lamp", action="store_true", help="平板上方有一根灯管，会映在屏幕上（腕部相机从近处看时）")
    parser.add_argument("--fast", action="store_true", help="配合 --serve：机械臂瞬间到位，不按真实速度走")
    parser.add_argument("--turn_park", action="store_true", help="连接后先按网页上的按钮把停靠位左转到 40°")
    parser.add_argument("--realtime", action="store_true", help="机械臂按真实速度走（慢，但等待和超时都和真机一样）")
    parser.add_argument("instructions", nargs="*", default=["打开音乐，播放音乐", "暂停播放", "回到桌面", "打开设置"])
    args = parser.parse_args(argv)
    root = config.PROJECT_ROOT
    out = pathlib.Path(args.out) if args.out else root / "runs" / f"rehearsal_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    rehearsal = Rehearsal(out, args)
    return rehearsal.serve(args.serve) if args.serve else rehearsal.run()


if __name__ == "__main__":
    raise SystemExit(main())
