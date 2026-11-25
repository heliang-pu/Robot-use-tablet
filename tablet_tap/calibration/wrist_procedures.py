"""The steps of the wrist camera's calibration, for any front end: they move the arm through ``move`` and look through
``camera``; what is printed and confirmed is up to the caller."""

from __future__ import annotations

from collections.abc import Callable
import time

import numpy as np

from tablet_tap.calibration.eye_in_hand import MIN_MARKERS
from tablet_tap.calibration.eye_in_hand import View
from tablet_tap.calibration.eye_in_hand import edge_width_px
from tablet_tap.core.look import Look
from tablet_tap.errors import CalibrationError
from tablet_tap.vision.camera import Camera
from tablet_tap.vision.layout import ScreenLayout
from tablet_tap.vision.rectifier import ScreenRectifier
from tablet_tap.vision.rectifier import detect_markers

STILL_RAD = 0.0015  # the arm moves less than this between two readings around the picture


def still_picture(read_joints: Callable[[], np.ndarray], camera: Camera, *, settle_s: float, frames: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """(joints, median picture) with the arm at rest: the joints are read before and after and must agree."""
    time.sleep(settle_s)
    for _ in range(5):
        before = read_joints()
        stack = [camera.read() for _ in range(frames)]
        after = read_joints()
        if float(np.max(np.abs(after - before))) < STILL_RAD:
            return 0.5 * (before + after), np.median(np.stack(stack), axis=0).astype(np.uint8)
    raise CalibrationError("机械臂没有停稳，拍不了标定照片。")


def collect_views(
    looks: list[Look],
    park_q: np.ndarray,
    move: Callable[[np.ndarray, str], None],
    read_joints: Callable[[], np.ndarray],
    camera: Camera,
    layout: ScreenLayout,
    *,
    settle_s: float = 0.6,
    say: Callable[[str], None] = print,
    keep: Callable[[int, np.ndarray], None] | None = None,
) -> list[View]:
    """Go through the calibration looks and back to the park pose; what each of them saw."""
    views = []
    for k, look in enumerate(looks, 1):
        if look.clearance.get("via_park"):
            move(np.asarray(park_q, np.float64), "经停靠位")
        move(look.q, f"标定位姿 {k}/{len(looks)}")
        q, picture = still_picture(read_joints, camera, settle_s=settle_s)
        markers = {i: quad for i, quad in detect_markers(picture).items() if i in layout.markers}
        if keep is not None:
            keep(k, picture)
        say(f"  {k}/{len(looks)}：看到 {len(markers)} 个标定码" + ("" if len(markers) >= MIN_MARKERS else "（太少，这一个不用）"))
        views.append(View(q, markers))
    move(np.asarray(park_q, np.float64), "回停靠位")
    return views


def look_quality(picture: np.ndarray, rectifier: ScreenRectifier) -> dict[str, float | None]:
    """How large and how sharp the screen is in the picture."""
    quad = rectifier.screen_quad_cam()
    edge = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    sharp = edge_width_px(picture, rectifier.detected)
    return {
        "width_px": float(min(edge[0], edge[2])),
        "height_px": float(min(edge[1], edge[3])),
        "edge_px": sharp,
        "reprojection_px": rectifier.reprojection_px,
    }
