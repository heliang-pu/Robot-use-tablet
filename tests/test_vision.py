"""Tests of the vision package and the calibration page (server and client): synthetic camera frames only, no hardware, no GPU.

    cd /home/phl/workspace/tablet_tap && bin/py -m pytest -q tests/test_vision.py        (-s prints the errors)

The browser test needs a Python with Playwright and its Chromium somewhere on the machine (the `playwright` command on
PATH points at it); it is skipped otherwise.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import threading
import types

import cv2
import numpy as np
import pytest
from werkzeug.serving import make_server

from tablet_tap.calibration import page_server
from tablet_tap.calibration.page_client import CalibClient
from tablet_tap.errors import CalibrationError
from tablet_tap.vision import camera as sv_camera
from tablet_tap.vision import layout as sv_layout
from tablet_tap.vision import rectifier as sv_rectifier
from tablet_tap.vision import tools as sv_tools

SCREEN_W, SCREEN_H = 2560, 1600
ASPECT = SCREEN_W / SCREEN_H
LAYOUT = sv_layout.default_layout(ASPECT)


# --------------------------------------------------------------------------------------------------------------------
# A synthetic camera
# --------------------------------------------------------------------------------------------------------------------


def _rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = (
        math.cos(roll),
        math.sin(roll),
        math.cos(pitch),
        math.sin(pitch),
        math.cos(yaw),
        math.sin(yaw),
    )
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return rz @ rx @ ry


def random_view(seed: int, cam_w: int, cam_h: int, aspect: float = ASPECT, upside_down: bool = False) -> np.ndarray:
    """Ground truth: the homography (u, v) -> camera pixel centres of a pinhole camera (69 deg wide, like a D435's
    colour camera) that sees the tablet obliquely, the screen 40-60 % of the frame wide and whole inside the frame."""
    rng = np.random.default_rng(seed)
    focal = 0.5 * cam_w / math.tan(math.radians(69.0 / 2))
    intrinsics = np.array([[focal, 0, cam_w / 2 - 0.5], [0, focal, cam_h / 2 - 0.5], [0, 0, 1]])
    # The glass in units of its height, centred: X = (u - 0.5) * aspect, Y = v - 0.5.
    to_plane = np.array([[aspect, 0, -aspect / 2], [0, 1, -0.5], [0, 0, 1]])
    corners = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float64)
    for _ in range(1000):
        roll = math.radians(rng.uniform(-12, 12)) + (math.pi if upside_down else 0.0)
        pitch = math.radians(rng.uniform(20, 40)) * rng.choice([-1, 1])
        yaw = math.radians(rng.uniform(-20, 20))
        rotation = _rotation(roll, pitch, yaw)
        fraction = rng.uniform(0.40, 0.60)
        centre = np.array([rng.uniform(-0.12, 0.12) * cam_w, rng.uniform(-0.08, 0.08) * cam_h])
        distance = 3.0
        for _ in range(30):  # the distance at which the screen is `fraction` of the frame wide
            translation = np.array([centre[0] * distance / focal, centre[1] * distance / focal, distance])
            homography = intrinsics @ np.column_stack([rotation[:, 0], rotation[:, 1], translation]) @ to_plane
            quad = sv_rectifier._apply_homography(homography, corners)
            distance *= np.ptp(quad[:, 0]) / (fraction * cam_w)
        margin = 0.03 * cam_h
        if quad[:, 0].min() > margin and quad[:, 1].min() > margin:
            if quad[:, 0].max() < cam_w - 1 - margin and quad[:, 1].max() < cam_h - 1 - margin:
                return homography
    raise AssertionError("no view found")


def camera_frame(
    screen: np.ndarray,
    norm_to_cam: np.ndarray,
    cam_w: int,
    cam_h: int,
    seed: int = 0,
    blur: float = 0.7,
    noise: float = 2.0,
) -> np.ndarray:
    """What the camera sees: the screen image (inside a black bezel) on a mid-grey table, then the lens and the
    sensor (area sampling to the camera's resolution, a mild Gaussian blur, noise)."""
    rng = np.random.default_rng(seed + 1000)
    screen_h, screen_w = screen.shape[:2]
    bezel = round(0.03 * screen_h)
    tablet = cv2.copyMakeBorder(screen, bezel, bezel, bezel, bezel, cv2.BORDER_CONSTANT, value=(12, 12, 12))
    oversample = max(2, math.ceil(5000 / cam_w))
    # tablet pixel centres -> (u, v) -> camera pixel centres -> pixel centres of the oversampled frame
    to_norm = np.array(
        [[1 / screen_w, 0, (0.5 - bezel) / screen_w], [0, 1 / screen_h, (0.5 - bezel) / screen_h], [0, 0, 1]]
    )
    to_fine = np.array([[oversample, 0, (oversample - 1) / 2], [0, oversample, (oversample - 1) / 2], [0, 0, 1]])
    warp = to_fine @ norm_to_cam @ to_norm
    size = (cam_w * oversample, cam_h * oversample)
    fine = cv2.warpPerspective(tablet, warp, size, flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full(tablet.shape[:2], 255, np.uint8), warp, size, flags=cv2.INTER_LINEAR)
    table = np.empty((size[1], size[0], 3), np.uint8)
    table[:] = (122, 126, 131)
    table = (table.astype(np.float32) * np.linspace(0.85, 1.1, size[0], dtype=np.float32)[None, :, None]).astype(
        np.uint8
    )
    alpha = mask.astype(np.float32)[..., None] / 255
    fine = (fine.astype(np.float32) * alpha + table.astype(np.float32) * (1 - alpha)).astype(np.uint8)
    frame = cv2.resize(fine, (cam_w, cam_h), interpolation=cv2.INTER_AREA).astype(np.float32)
    frame = cv2.GaussianBlur(frame, (0, 0), blur)
    frame += rng.normal(0.0, noise, frame.shape)
    return np.clip(np.rint(frame), 0, 255).astype(np.uint8)


def grid_error(rectifier: sv_rectifier.ScreenRectifier, norm_to_cam: np.ndarray) -> tuple[float, float]:
    """Max |u error| and |v error| of cam_to_norm over a 9x9 grid of the whole screen, corners included."""
    grid = np.stack(np.meshgrid(np.linspace(0, 1, 9), np.linspace(0, 1, 9)), axis=-1)
    error = np.abs(rectifier.cam_to_norm(sv_rectifier._apply_homography(norm_to_cam, grid)) - grid)
    return float(error[..., 0].max()), float(error[..., 1].max())


def occlude(frame: np.ndarray, norm_to_cam: np.ndarray, marker_ids: list[int]) -> np.ndarray:
    """Grey blobs (a robot arm in the way) over the given markers."""
    out = frame.copy()
    for marker_id in marker_ids:
        u0, v0, u1, v1 = LAYOUT.markers[marker_id]
        quad = sv_rectifier._apply_homography(norm_to_cam, np.array([(u0, v0), (u1, v0), (u1, v1), (u0, v1)]))
        centre = quad.mean(axis=0)
        radius = 0.75 * float(np.linalg.norm(quad - centre, axis=1).max())
        cv2.ellipse(
            out,
            (round(centre[0]), round(centre[1])),
            (round(radius), round(0.8 * radius)),
            20,
            0,
            360,
            (88, 90, 96),
            -1,
        )
    return out


@pytest.fixture(scope="module")
def screen() -> np.ndarray:
    return sv_layout.render_layout(LAYOUT, SCREEN_W)


def marker_corners_px(layout: sv_layout.ScreenLayout, marker_id: int, width: int, height: int) -> np.ndarray:
    u0, v0, u1, v1 = layout.markers[marker_id]
    return np.array([(u0, v0), (u1, v0), (u1, v1), (u0, v1)]) * (width, height) - 0.5


# --------------------------------------------------------------------------------------------------------------------
# Layout and rendering
# --------------------------------------------------------------------------------------------------------------------


def test_default_layout() -> None:
    for aspect in (4 / 3, 1.5, 1.6, 16 / 9, 2.0):
        layout = sv_layout.default_layout(aspect)
        assert sorted(layout.markers) == [0, 1, 2, 3, 4, 5]
        boxes = []
        for u0, v0, u1, v1 in layout.markers.values():
            assert v1 - v0 == pytest.approx(0.22) and (u1 - u0) * aspect == pytest.approx(0.22)  # square on the glass
            quiet_u, quiet_v = (u1 - u0) / sv_layout.MARKER_CELLS, (v1 - v0) / sv_layout.MARKER_CELLS
            box = (u0 - quiet_u, v0 - quiet_v, u1 + quiet_u, v1 + quiet_v)
            assert box[0] > 0 and box[1] > 0 and box[2] < 1 and box[3] < 1  # the quiet zone is on the screen
            boxes.append(box)
        for i, a in enumerate(boxes):
            for b in boxes[i + 1 :]:
                assert a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]  # quiet zones do not overlap
        # top row left -> right is 0, 1, 2; bottom row 3, 4, 5; inset 0.04 of the height
        assert layout.markers[0][:2] == pytest.approx((0.04 / aspect, 0.04))
        assert layout.markers[5][2:] == pytest.approx((1 - 0.04 / aspect, 0.96))
        assert (layout.markers[1][0] + layout.markers[1][2]) / 2 == pytest.approx(0.5)
        assert (layout.markers[4][0] + layout.markers[4][2]) / 2 == pytest.approx(0.5)
        assert layout.markers[4][1] > 0.5 > layout.markers[1][3]
        again = sv_layout.ScreenLayout.from_json(json.loads(json.dumps(layout.to_json())))
        assert again == layout
    with pytest.raises(ValueError, match="横屏"):
        sv_layout.default_layout(0.625)


def test_render_layout_is_what_the_layout_says(screen: np.ndarray) -> None:
    assert screen.shape == (SCREEN_H, SCREEN_W, 3) and screen.dtype == np.uint8
    assert tuple(screen[SCREEN_H // 2, SCREEN_W // 2]) == (0x20, 0x20, 0x20)
    found = sv_rectifier.detect_markers(screen)
    assert sorted(found) == sorted(LAYOUT.markers)
    for marker_id, quad in found.items():
        assert np.abs(quad - marker_corners_px(LAYOUT, marker_id, SCREEN_W, SCREEN_H)).max() < 0.3
    # the white quiet zone is one cell wide all round
    u0, v0, u1, v1 = LAYOUT.markers[0]
    cell = (v1 - v0) * SCREEN_H / sv_layout.MARKER_CELLS
    x, y = round(u0 * SCREEN_W), round(v0 * SCREEN_H)
    assert tuple(screen[y + 5, x - round(cell / 2)]) == (255, 255, 255)
    assert tuple(screen[y + 5, x + 5]) == (0, 0, 0)
    assert tuple(screen[y + 5, x - round(cell) - 3]) == (0x20, 0x20, 0x20)
    # other modes have no markers
    assert sv_rectifier.detect_markers(sv_layout.render_layout(LAYOUT, 1280, mode="blank")) == {}
    target = sv_layout.render_layout(LAYOUT, 1280, mode="target", target=(0.3, 0.6))
    assert sv_rectifier.detect_markers(target) == {}
    assert tuple(target[round(0.6 * 800), round(0.3 * 1280)]) == (255, 255, 255)


# --------------------------------------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("seed", "upside_down"), [(0, False), (1, False), (2, False), (3, True)])
def test_calibrate_1280x720(screen: np.ndarray, seed: int, upside_down: bool) -> None:
    truth = random_view(seed, 1280, 720, upside_down=upside_down)
    quad = sv_rectifier._apply_homography(truth, np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float64))
    assert 0.39 <= np.ptp(quad[:, 0]) / 1280 <= 0.61
    frame = camera_frame(screen, truth, 1280, 720, seed)
    rectifier = sv_rectifier.ScreenRectifier.from_marker_image(frame, LAYOUT)
    assert rectifier.marker_ids == (0, 1, 2, 3, 4, 5)
    assert rectifier.image_size == (1280, 720)
    error_u, error_v = grid_error(rectifier, truth)
    print(
        f"\n1280x720 seed {seed}{' upside down' if upside_down else ''}: screen {np.ptp(quad[:, 0]) / 1280:.0%} of the frame "
        f"wide, reprojection RMS {rectifier.reprojection_px:.3f} px, max |du| {error_u:.5f}, max |dv| {error_v:.5f}"
    )
    assert rectifier.reprojection_px < 0.5
    assert error_u < 0.004 and error_v < 0.004
    assert np.abs(rectifier.screen_quad_cam() - quad).max() < 2.0
    grid = np.random.default_rng(seed).uniform(0, 1, (5, 7, 2))
    assert np.abs(rectifier.cam_to_norm(rectifier.norm_to_cam(grid)) - grid).max() < 1e-9
    assert rectifier.cam_to_norm(rectifier.norm_to_cam(grid)).shape == grid.shape

    shot = rectifier.rectify(frame)
    assert shot.shape == (800, 1280, 3) and shot.dtype == np.uint8
    found = sv_rectifier.detect_markers(shot)  # upright: every marker is where the layout puts it, the right way up
    assert sorted(found) == sorted(LAYOUT.markers)
    worst = max(np.abs(found[i] - marker_corners_px(LAYOUT, i, 1280, 800)).max() for i in found)
    print(f"  rectified 1280x800: marker corners within {worst:.2f} px of the layout")
    assert worst < 2.0
    assert sv_rectifier.ScreenRectifier.from_marker_image(frame, LAYOUT, out_width=640).rectify(frame).shape == (
        400,
        640,
        3,
    )
    with pytest.raises(ValueError, match="重新标定"):
        rectifier.rectify(frame[:480, :640])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_calibrate_640x480(screen: np.ndarray, seed: int) -> None:
    truth = random_view(seed, 640, 480)
    frame = camera_frame(screen, truth, 640, 480, seed)
    rectifier = sv_rectifier.ScreenRectifier.from_marker_image(frame, LAYOUT)
    error_u, error_v = grid_error(rectifier, truth)
    print(
        f"\n640x480 seed {seed}: {len(rectifier.marker_ids)} markers, reprojection RMS {rectifier.reprojection_px:.3f} px, "
        f"max |du| {error_u:.5f}, max |dv| {error_v:.5f}"
    )
    assert len(rectifier.marker_ids) >= 4
    assert error_u < 0.008 and error_v < 0.008
    assert rectifier.rectify(frame).shape == (800, 1280, 3)


def test_partial_occlusion(screen: np.ndarray) -> None:
    truth = random_view(0, 1280, 720)
    frame = camera_frame(screen, truth, 1280, 720, 0)
    for hidden in ([0, 1, 3], [2, 4, 5], [0, 1, 2]):  # three of the six are behind the arm
        rectifier = sv_rectifier.ScreenRectifier.from_marker_image(occlude(frame, truth, hidden), LAYOUT)
        assert rectifier.marker_ids == tuple(i for i in range(6) if i not in hidden)
        error_u, error_v = grid_error(rectifier, truth)
        print(
            f"\nmarkers {hidden} hidden: RMS {rectifier.reprojection_px:.3f} px, max |du| {error_u:.5f}, max |dv| {error_v:.5f}"
        )
        assert error_u < 0.01 and error_v < 0.01
    for hidden in ([1, 2, 3, 4], [0, 1, 2, 3], [0, 1, 3, 4]):  # two markers left, the least that calibrates
        rectifier = sv_rectifier.ScreenRectifier.from_marker_image(occlude(frame, truth, hidden), LAYOUT)
        assert len(rectifier.marker_ids) == 2
        error_u, error_v = grid_error(rectifier, truth)
        print(f"markers {hidden} hidden (2 left): max |du| {error_u:.5f}, max |dv| {error_v:.5f}")
        assert error_u < 0.03 and error_v < 0.03
    with pytest.raises(CalibrationError, match="只看到 1 个"):
        sv_rectifier.ScreenRectifier.from_marker_image(occlude(frame, truth, [0, 1, 2, 3, 5]), LAYOUT)
    with pytest.raises(CalibrationError, match="只看到 0 个"):
        sv_rectifier.ScreenRectifier.from_marker_image(occlude(frame, truth, [0, 1, 2, 3, 4, 5]), LAYOUT)


def test_marker_of_another_layout_is_ignored(screen: np.ndarray) -> None:
    """A marker that is not where the layout says (another id, or a wrong position) does not bend the fit."""
    truth = random_view(1, 1280, 720)
    moved = dict(LAYOUT.markers)
    u0, v0, u1, v1 = moved[4]
    moved[4] = (u0 - 0.2, v0 - 0.3, u1 - 0.2, v1 - 0.3)  # drawn in the middle of the screen instead
    moved[17] = (0.62, 0.40, 0.62 + (u1 - u0), 0.40 + (v1 - v0))
    frame = camera_frame(sv_layout.render_layout(sv_layout.ScreenLayout(ASPECT, moved), SCREEN_W), truth, 1280, 720, 1)
    rectifier = sv_rectifier.ScreenRectifier.from_marker_image(frame, LAYOUT)
    assert rectifier.marker_ids == (0, 1, 2, 3, 5)
    assert sorted(rectifier.detected) == [0, 1, 2, 3, 4, 5]
    assert max(grid_error(rectifier, truth)) < 0.004
    overlay = sv_rectifier.draw_overlay(frame, rectifier)
    assert overlay.shape == frame.shape and np.any(overlay != frame)


def test_save_load(screen: np.ndarray, tmp_path: pathlib.Path) -> None:
    truth = random_view(2, 1280, 720)
    frame = camera_frame(screen, truth, 1280, 720, 2)
    rectifier = sv_rectifier.ScreenRectifier.from_marker_image(frame, LAYOUT, out_width=1024)
    path = tmp_path / "sub" / "screen_view.json"
    rectifier.save(path)
    loaded = sv_rectifier.ScreenRectifier.load(path)
    assert json.loads(path.read_text())["version"] == 1
    assert np.allclose(loaded.homography, rectifier.homography, rtol=0, atol=1e-15)
    assert (loaded.aspect, loaded.out_width, loaded.out_height) == (ASPECT, 1024, 640)
    assert loaded.reprojection_px == rectifier.reprojection_px
    assert loaded.image_size == (1280, 720) and loaded.marker_ids == rectifier.marker_ids
    assert np.array_equal(loaded.rectify(frame), rectifier.rectify(frame))
    assert np.allclose(loaded.screen_quad_cam(), rectifier.screen_quad_cam())
    # a hand-made rectifier: the reprojection error is unknown (nan) and survives the file as such
    plain = sv_rectifier.ScreenRectifier(np.diag([1 / 640, 1 / 400, 1.0]), 1.6, out_width=320)
    assert plain.cam_to_norm([320, 200]) == pytest.approx([0.5, 0.5])
    plain.save(tmp_path / "plain.json")
    assert math.isnan(sv_rectifier.ScreenRectifier.load(tmp_path / "plain.json").reprojection_px)
    assert plain.rectify(np.zeros((400, 640, 3), np.uint8)).shape == (200, 320, 3)


# --------------------------------------------------------------------------------------------------------------------
# The touch dot
# --------------------------------------------------------------------------------------------------------------------

DOTS = [(0.5, 0.5), (0.31, 0.62), (0.777, 0.333), (0.2, 0.5), (0.05, 0.5), (0.93, 0.48), (0.3, 0.9), (0.68, 0.07)]


@pytest.mark.parametrize(("cam_w", "cam_h", "limit"), [(1280, 720, 0.003), (640, 480, 0.006)])
def test_find_touch_dot(cam_w: int, cam_h: int, limit: float) -> None:
    truth = random_view(4, cam_w, cam_h)
    worst = 0.0
    for k, dot in enumerate(DOTS):
        frame = camera_frame(sv_layout.render_layout(LAYOUT, SCREEN_W, dot=dot), truth, cam_w, cam_h, k)
        if k == 0:
            rectifier = sv_rectifier.ScreenRectifier.from_marker_image(frame, LAYOUT)
        found = sv_rectifier.find_touch_dot(rectifier.rectify(frame))
        assert found is not None, dot
        worst = max(worst, abs(found[0] - dot[0]), abs(found[1] - dot[1]))
    print(f"\ntouch dot through a {cam_w}x{cam_h} camera: max |(u, v) error| {worst:.5f} over {len(DOTS)} positions")
    assert worst < limit
    # in the other page modes too (the calibration is reused, no markers on the screen)
    for mode, target in (("blank", None), ("target", (0.4, 0.4))):
        frame = camera_frame(
            sv_layout.render_layout(LAYOUT, SCREEN_W, mode=mode, target=target, dot=(0.42, 0.43)),
            truth,
            cam_w,
            cam_h,
            9,
        )
        found = sv_rectifier.find_touch_dot(rectifier.rectify(frame))
        assert found is not None and abs(found[0] - 0.42) < limit and abs(found[1] - 0.43) < limit


def test_find_touch_dot_none_without_a_dot(screen: np.ndarray) -> None:
    assert sv_rectifier.find_touch_dot(screen) is None
    assert sv_rectifier.find_touch_dot(sv_layout.render_layout(LAYOUT, 1280, mode="blank")) is None
    assert sv_rectifier.find_touch_dot(sv_layout.render_layout(LAYOUT, 1280, mode="target", target=(0.5, 0.5))) is None
    assert sv_rectifier.find_touch_dot(sv_layout.render_layout(LAYOUT, 1280, dot=(0.25, 0.75))) == pytest.approx(
        (0.25, 0.75), abs=2e-4
    )
    for cam_w, cam_h in ((1280, 720), (640, 480)):
        truth = random_view(5, cam_w, cam_h)
        frame = camera_frame(screen, truth, cam_w, cam_h, 5, noise=4.0)
        rectifier = sv_rectifier.ScreenRectifier.from_marker_image(frame, LAYOUT)
        assert sv_rectifier.find_touch_dot(rectifier.rectify(frame)) is None
        # A lens with colour fringes: green shifted against red and blue makes every black/white edge of the markers
        # magenta on one side. Thin lines, not a disc: still no dot.
        fringed = frame.copy()
        fringed[:, :, 1] = np.roll(frame[:, :, 1], (1, 1), axis=(0, 1))
        assert sv_rectifier.find_touch_dot(rectifier.rectify(fringed)) is None
        # ... and a real dot is still found next to those fringes
        with_dot = camera_frame(sv_layout.render_layout(LAYOUT, SCREEN_W, dot=(0.6, 0.55)), truth, cam_w, cam_h, 5)
        with_dot[:, :, 1] = np.roll(with_dot[:, :, 1], (1, 1), axis=(0, 1))
        found = sv_rectifier.find_touch_dot(rectifier.rectify(with_dot))
        assert found is not None and abs(found[0] - 0.6) < 0.006 and abs(found[1] - 0.55) < 0.006


# --------------------------------------------------------------------------------------------------------------------
# Cameras and the CLI
# --------------------------------------------------------------------------------------------------------------------


def test_file_camera_and_open_camera(tmp_path: pathlib.Path) -> None:
    for k in range(3):
        cv2.imwrite(str(tmp_path / f"frame_{k:02d}.png"), np.full((48, 64, 3), 10 * (k + 1), np.uint8))
    (tmp_path / "notes.txt").write_text("not an image")
    camera = sv_camera.open_camera(f"file:{tmp_path}")
    assert isinstance(camera, sv_camera.FileCamera)
    assert [int(camera.read()[0, 0, 0]) for _ in range(5)] == [10, 20, 30, 30, 30]  # the last one repeats
    camera.close()
    single = sv_camera.open_camera(f"file:{tmp_path / 'frame_01.png'}")
    assert [int(single.read()[0, 0, 0]) for _ in range(2)] == [20, 20]
    assert single.read().shape == (48, 64, 3)
    with pytest.raises(RuntimeError):
        sv_camera.open_camera(f"file:{tmp_path / 'missing.png'}")
    for bad in ("usb:0", "realsense:", "file:", "nothing"):
        with pytest.raises(ValueError, match="不认识的相机"):
            sv_camera.open_camera(bad)
    with pytest.raises(ValueError, match="宽x高"):
        sv_camera.open_camera("realsense:123:720p")


class _FakeRealSense(types.ModuleType):
    """The few pyrealsense2 calls RealSenseColor makes, for one camera '111' with `queued` frames already waiting.
    It checks the call order only: the real SDK is never touched in these tests."""

    def __init__(self) -> None:
        super().__init__("pyrealsense2")
        fake = self
        self.calls: list[tuple] = []
        self.queued = 0
        self.number = 0
        self.fail = False
        self.camera_info = types.SimpleNamespace(serial_number="serial_number")
        self.stream = types.SimpleNamespace(color="color")
        self.format = types.SimpleNamespace(bgr8="bgr8")

        class Frame:
            def __init__(self, number: int, shape: tuple[int, int]) -> None:
                self._image = np.full((*shape, 3), number % 256, np.uint8)

            def get_color_frame(self) -> Frame:
                return self

            def get_data(self) -> np.ndarray:
                return self._image

        class Device:
            def get_info(self, what: str) -> str:
                assert what == "serial_number"
                return "111"

        class Context:
            def query_devices(self) -> list[Device]:
                return [Device()]

        class Config:
            def enable_device(self, serial: str) -> None:
                fake.calls.append(("enable_device", serial))

            def enable_stream(self, *args: object) -> None:
                fake.calls.append(("enable_stream", *args))
                fake.shape = (args[2], args[1])

        class Pipeline:
            def __init__(self, context: Context) -> None:
                assert isinstance(context, Context)

            def start(self, config: Config) -> None:
                fake.calls.append(("start",))

            def stop(self) -> None:
                fake.calls.append(("stop",))

            def poll_for_frames(self) -> Frame | None:
                if fake.queued <= 0:
                    return None
                fake.queued -= 1
                fake.number += 1
                return Frame(fake.number, fake.shape)

            def wait_for_frames(self, timeout_ms: int = 5000) -> Frame:
                if fake.fail:
                    raise RuntimeError("Frame didn't arrive within 5000")
                fake.number += 1
                return Frame(fake.number, fake.shape)

        self.context, self.config, self.pipeline = Context, Config, Pipeline


def test_realsense_color_with_a_fake_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeRealSense()
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake)
    camera = sv_camera.open_camera("realsense:111:640x480")
    assert isinstance(camera, sv_camera.RealSenseColor)
    assert fake.calls == [("enable_device", "111"), ("enable_stream", "color", 640, 480, "bgr8", 30), ("start",)]
    assert fake.number == 30  # the warm-up frames
    fake.queued = 3  # three old frames wait in the queue: read() drops them and takes the next one
    image = camera.read()
    assert image.shape == (480, 640, 3) and int(image[0, 0, 0]) == 34
    fake.fail = True
    with pytest.raises(RuntimeError, match="没有出图"):
        camera.read()
    camera.close()
    camera.close()
    assert fake.calls.count(("stop",)) == 1
    with pytest.raises(RuntimeError, match="没有序列号为 222"):
        sv_camera.RealSenseColor("222")
    fake.calls.clear()
    with pytest.raises(RuntimeError, match="没有出图"):  # no frame during the warm-up: the stream is stopped again
        sv_camera.RealSenseColor("111")
    assert fake.calls[-2:] == [("start",), ("stop",)]
    fake.fail = False
    assert sv_camera.open_camera("realsense:111")._shape == (720, 1280, 3)
    assert ("enable_stream", "color", 1280, 720, "bgr8", 30) in fake.calls


# --------------------------------------------------------------------------------------------------------------------
# The calibration server
# --------------------------------------------------------------------------------------------------------------------

SCREEN_INFO = {"inner_w": 1280, "inner_h": 800, "screen_w": 1280, "screen_h": 800, "dpr": 2, "fullscreen": True}


def test_flask_app() -> None:
    app = page_server.create_app()
    client = app.test_client()

    assert client.get("/").status_code == 302
    page = client.get("/calib")
    assert page.status_code == 200 and "点这里进入全屏" in page.get_data(as_text=True)
    assert page.headers["Cache-Control"] == "no-store"

    # before the page has reported its screen
    state = client.get("/state").get_json()
    assert state["screen"] is None and state["mode"] == "markers" and state["target"] is None
    assert state["last_down"] is None and state["n_touches"] == 0 and state["clear_seq"] == 0
    response = client.get("/layout")
    assert response.status_code == 409 and "屏幕" in response.get_json()["error"]

    # /screen -> /layout
    response = client.post("/screen", json=SCREEN_INFO)
    assert response.status_code == 200
    answer = response.get_json()
    assert answer["screen"]["covers_display"] is True and answer["screen"]["aspect"] == pytest.approx(1.6)
    assert answer["layout"]["style"]["dot_diameter"] == sv_layout.DOT_DIAMETER
    response = client.get("/layout")
    assert response.status_code == 200
    assert sv_layout.ScreenLayout.from_json(response.get_json()) == sv_layout.default_layout(1.6)
    assert response.get_json()["covers_display"] is True
    assert client.get("/state").get_json()["screen"]["dpr"] == 2
    assert client.post("/screen", json={"inner_w": 1280}).status_code == 400
    assert client.post("/screen", data="not json", content_type="text/plain").status_code == 400
    # the page does not cover the display (the browser's bars are showing): the layout follows the page, flagged
    client.post("/screen", json=SCREEN_INFO | {"inner_h": 720, "fullscreen": False})
    data = client.get("/layout").get_json()
    assert data["covers_display"] is False and data["aspect"] == pytest.approx(1280 / 720)
    # portrait: no layout, and a message that says why
    answer = client.post(
        "/screen", json=SCREEN_INFO | {"inner_w": 800, "inner_h": 1280, "screen_w": 800, "screen_h": 1280}
    ).get_json()
    assert answer["layout"] is None and "横屏" in answer["error"]
    response = client.get("/layout")
    assert response.status_code == 409 and "横屏" in response.get_json()["error"]
    client.post("/screen", json=SCREEN_INFO)

    # markers
    response = client.get("/marker/3.png")
    assert response.status_code == 200 and response.mimetype == "image/png"
    image = cv2.imdecode(np.frombuffer(response.data, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert image.shape == (240, 240) and set(np.unique(image)) == {0, 255}
    padded = cv2.copyMakeBorder(image, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    assert list(sv_rectifier.detect_markers(padded)) == [3]
    assert client.get("/marker/50.png").status_code == 404

    # /touch -> /state -> /touches
    assert client.post(
        "/touch", json={"type": "pointerdown", "u": 0.25, "v": 0.75, "t_ms": 1000, "pointer_type": "touch"}
    ).get_json() == {
        "ok": True,
        "seq": 1,
    }
    client.post("/touch", json={"type": "pointermove", "u": 0.26, "v": 0.75, "t_ms": 1030, "pointer_type": "touch"})
    # sendBeacon posts JSON as text/plain
    response = client.post(
        "/touch", data=json.dumps({"type": "pointerup", "u": 0.27, "v": 0.75, "t_ms": 1060}), content_type="text/plain"
    )
    assert response.get_json()["seq"] == 3
    assert client.post("/touch", json={"type": "click", "u": 0.1, "v": 0.1}).status_code == 400
    assert client.post("/touch", json={"type": "pointerdown", "u": "x", "v": 0.1}).status_code == 400
    state = client.get("/state").get_json()
    assert state["n_touches"] == 3
    assert state["last_down"]["u"] == 0.25 and state["last_down"]["v"] == 0.75 and state["last_down"]["seq"] == 1
    assert state["last_down"]["t_server"] > 1e9
    touches = client.get("/touches").get_json()["touches"]
    assert [t["type"] for t in touches] == ["pointerdown", "pointermove", "pointerup"]
    assert (
        [t["seq"] for t in touches] == [1, 2, 3]
        and touches[0]["pointer_type"] == "touch"
        and touches[0]["t_ms"] == 1000
    )
    assert [t["seq"] for t in client.get("/touches?since_seq=1").get_json()["touches"]] == [2, 3]
    assert client.get("/touches?since_seq=3").get_json() == {"touches": [], "last_seq": 3}
    assert [t["seq"] for t in client.get("/touches?type=pointerdown").get_json()["touches"]] == [1]
    client.post("/touch", json={"type": "pointerdown", "u": 0.5, "v": 0.5, "t_ms": 2000, "pointer_type": "touch"})
    assert client.get("/state").get_json()["last_down"]["seq"] == 4

    # mode / target / clear
    assert client.post("/mode", json={"mode": "target", "target": [0.3, 0.6]}).status_code == 200
    state = client.get("/state").get_json()
    assert state["mode"] == "target" and state["target"] == [0.3, 0.6]
    assert client.post("/mode", json={"mode": "target"}).status_code == 400  # a target is needed
    assert client.post("/mode", json={"mode": "target", "target": [1.3, 0.6]}).status_code == 400
    assert client.post("/mode", json={"mode": "fancy"}).status_code == 400
    assert "mode" in client.post("/mode", json={"mode": "fancy"}).get_json()["error"]
    assert client.get("/state").get_json()["mode"] == "target"  # refused requests change nothing
    client.post("/mode", json={"mode": "blank"})
    state = client.get("/state").get_json()
    assert state["mode"] == "blank" and state["target"] is None
    client.post("/mode", json={"mode": "markers"})
    assert client.post("/clear").get_json()["clear_seq"] == 1
    state = client.get("/state").get_json()
    assert state["clear_seq"] == 1 and state["last_down"]["seq"] == 4 and state["page_age_s"] is None
    assert client.get("/state?page=1").get_json()["page_age_s"] == 0


def test_touches_are_bounded() -> None:
    state = page_server.CalibState(max_touches=10)
    for k in range(25):
        state.add_touch({"type": "pointermove", "u": k / 100, "v": 0.5})
    events, last = state.touches_after(0)
    assert last == 25 and [e["seq"] for e in events] == list(range(16, 26))
    assert state.snapshot()["n_touches"] == 25


@pytest.fixture
def live_server():
    """The app served by werkzeug in a thread on a free port: (base URL, state)."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    app = page_server.create_app()
    server = make_server("127.0.0.1", port, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}", app.config["CALIB_STATE"]
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_calib_client(live_server, monkeypatch: pytest.MonkeyPatch) -> None:
    base_url, state = live_server
    for name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")  # a dead proxy: the client must not use it
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    client = CalibClient(base_url, timeout_s=3.0)

    assert client.screen() is None
    with pytest.raises(RuntimeError, match="409"):
        client.layout()
    state.set_screen(SCREEN_INFO)
    assert client.screen()["inner_w"] == 1280 and client.screen()["covers_display"] is True
    assert client.layout() == sv_layout.default_layout(1.6)

    client.set_mode("target", (0.3, 0.6))
    assert (state.mode, state.target) == ("target", (0.3, 0.6))
    client.set_mode("markers")
    assert (state.mode, state.target) == ("markers", None)
    with pytest.raises(RuntimeError, match="400"):
        client.set_mode("target")
    client.clear()
    assert state.clear_seq == 1

    assert client.last_down_seq() == 0
    assert client.wait_down(0, timeout_s=0.2) is None
    state.add_touch({"type": "pointerdown", "u": 0.1, "v": 0.2})
    state.add_touch({"type": "pointerup", "u": 0.1, "v": 0.2})
    assert client.last_down_seq() == 1
    assert client.wait_down(0, timeout_s=0.2) == (0.1, 0.2)
    assert client.wait_down(1, timeout_s=0.2) is None

    # the stylus lands while the caller waits; the first pointerdown after `seq` is the one returned
    seq = client.last_down_seq()

    def tap() -> None:
        state.add_touch({"type": "pointermove", "u": 0.9, "v": 0.9})
        state.add_touch({"type": "pointerdown", "u": 0.61, "v": 0.37})
        state.add_touch({"type": "pointerup", "u": 0.61, "v": 0.37})
        state.add_touch({"type": "pointerdown", "u": 0.7, "v": 0.7})

    timer = threading.Timer(0.3, tap)
    timer.start()
    assert client.wait_down(seq, timeout_s=5.0) == (0.61, 0.37)
    timer.join()
    assert client.last_down_seq() == 6
    assert [t["type"] for t in client.touches(seq)] == [
        "pointerup",
        "pointermove",
        "pointerdown",
        "pointerup",
        "pointerdown",
    ]
    client.clear()  # clearing the dot does not rewind the sequence
    assert client.last_down_seq() == 6

    with pytest.raises(RuntimeError, match="连不上标定服务"):
        CalibClient("http://127.0.0.1:9", timeout_s=0.5).screen()


def test_cli_calibrate_and_rectify(
    live_server, screen: np.ndarray, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture
) -> None:
    base_url, state = live_server
    truth = random_view(0, 1280, 720)
    frame_path = tmp_path / "frame.png"
    cv2.imwrite(str(frame_path), camera_frame(screen, truth, 1280, 720, 0))
    out, shot = tmp_path / "calib" / "screen_view.json", tmp_path / "shot.png"
    arguments = ["calibrate", "--camera", f"file:{frame_path}", "--layout-url", f"{base_url}/layout", "--out", str(out)]

    assert sv_tools.main(arguments) == 1  # the page has not reported its screen yet
    assert "409" in capsys.readouterr().out and not out.exists()

    state.set_screen(SCREEN_INFO | {"inner_h": 790, "fullscreen": False})
    assert sv_tools.main(arguments) == 0
    assert "没有铺满" in capsys.readouterr().out  # calibrated, with the warning

    state.set_screen(SCREEN_INFO)
    assert sv_tools.main(arguments) == 0
    printed = capsys.readouterr().out
    assert "6/6" in printed and "重投影误差" in printed and "没有铺满" not in printed
    debug = out.with_name("screen_view_debug.jpg")
    assert out.exists() and debug.exists() and cv2.imread(str(debug)).shape == (720, 1280, 3)
    assert max(grid_error(sv_rectifier.ScreenRectifier.load(out), truth)) < 0.004

    assert sv_tools.main(["rectify", "--camera", f"file:{frame_path}", "--calib", str(out), "--out", str(shot)]) == 0
    assert cv2.imread(str(shot)).shape == (800, 1280, 3)

    # a frame without the tablet: the failure says how many markers were seen and keeps the frame
    grey = tmp_path / "grey.png"
    cv2.imwrite(str(grey), np.full((720, 1280, 3), 128, np.uint8))
    other = tmp_path / "other.json"
    code = sv_tools.main(
        [
            "calibrate",
            "--camera",
            f"file:{grey}",
            "--layout-url",
            f"{base_url}/layout",
            "--out",
            str(other),
            "--debug-image",
            str(tmp_path / "why.jpg"),
        ]
    )
    assert code == 1 and "只看到 0 个" in capsys.readouterr().out
    assert not other.exists() and (tmp_path / "why.jpg").exists()


# --------------------------------------------------------------------------------------------------------------------
# The page in a real (headless) browser
# --------------------------------------------------------------------------------------------------------------------

_BROWSER_SCRIPT = r"""
import json, pathlib, sys, urllib.request
from playwright.sync_api import sync_playwright

base, out = sys.argv[1], pathlib.Path(sys.argv[2])
result = {}

def post(path, payload):
    request = urllib.request.Request(base + path, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=5).read()

def markers_ready(page):
    page.wait_for_function(
        "() => { const m = document.querySelectorAll('img.marker');"
        " return m.length === 6 && [...m].every((i) => i.complete && i.naturalWidth > 0); }", timeout=10000)

with sync_playwright() as p:
    browser = p.chromium.launch()
    # the tablet: 2560x1600 physical pixels, a touch screen
    context = browser.new_context(viewport={"width": 2560, "height": 1600}, device_scale_factor=1, has_touch=True)
    page = context.new_page()
    page.add_init_script("window.shown = (id) => getComputedStyle(document.getElementById(id)).display !== 'none';")
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.goto(base + "/calib")
    markers_ready(page)
    page.wait_for_timeout(600)
    result["first"] = page.evaluate(
        "() => ({w: innerWidth, h: innerHeight, sw: screen.width, sh: screen.height,"
        " button: shown('fs'), warn: document.getElementById('warn').textContent})")
    page.screenshot(path=str(out / "markers.png"))

    page.touchscreen.tap(700, 900)
    page.wait_for_timeout(500)
    result["info"] = page.evaluate("() => document.getElementById('info').textContent")
    page.screenshot(path=str(out / "dot.png"))

    post("/mode", {"mode": "target", "target": [0.3, 0.6]})
    page.wait_for_timeout(800)
    page.screenshot(path=str(out / "target.png"))

    page.touchscreen.tap(2000.5, 300.25)
    page.wait_for_timeout(300)
    post("/mode", {"mode": "blank"})
    page.wait_for_timeout(800)
    page.screenshot(path=str(out / "blank_dot.png"))

    post("/clear", {})
    page.wait_for_timeout(800)
    result["dot_hidden_after_clear"] = page.evaluate("() => !shown('dot')")
    page.screenshot(path=str(out / "blank.png"))

    # a drag with the touch screen: the page gets the moves (the browser does not take the gesture for scrolling)
    post("/mode", {"mode": "markers"})
    session = context.new_cdp_session(page)
    point = lambda x, y: [{"x": x, "y": y, "id": 1}]
    session.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": point(1000, 800)})
    for k in range(1, 11):
        session.send("Input.dispatchTouchEvent", {"type": "touchMove", "touchPoints": point(1000 + 20 * k, 800)})
    session.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
    page.wait_for_timeout(500)

    # pointer events faster than any screen sends them (one every 4 ms for 0.4 s): reported at about 30 Hz
    page.evaluate(
        "async () => { const fire = (type, x) => window.dispatchEvent(new PointerEvent(type, {pointerId: 77,"
        " pointerType: 'touch', clientX: x, clientY: 400, bubbles: true, cancelable: true}));"
        " fire('pointerdown', 500); const start = performance.now(); let k = 0;"
        " while (performance.now() - start < 400) { fire('pointermove', 500 + (k += 1));"
        " await new Promise((r) => setTimeout(r, 4)); }"
        " fire('pointerup', 500 + k); window.__fired = k; }")
    result["fired"] = page.evaluate("() => window.__fired")
    page.wait_for_timeout(500)
    context.close()
    result["seq_main"] = json.load(urllib.request.build_opener(urllib.request.ProxyHandler({})).open(base + "/touches?since_seq=100000"))["last_seq"]

    # the same display as a real tablet reports it: 1280x800 CSS pixels at devicePixelRatio 2
    context = browser.new_context(viewport={"width": 1280, "height": 800}, device_scale_factor=2, has_touch=True)
    page = context.new_page()
    page.add_init_script("window.shown = (id) => getComputedStyle(document.getElementById(id)).display !== 'none';")
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(base + "/calib")
    markers_ready(page)
    page.wait_for_timeout(600)
    page.touchscreen.tap(333.5, 444.5)
    page.wait_for_timeout(500)
    result["dpr2"] = page.evaluate("() => ({w: innerWidth, h: innerHeight, dpr: devicePixelRatio,"
                                   " button: shown('fs'), warn: document.getElementById('warn').textContent})")
    page.screenshot(path=str(out / "dpr2.png"))
    context.close()

    # a browser window that does not cover the display (bars showing): the warning and the button
    context = browser.new_context(viewport={"width": 1280, "height": 720}, screen={"width": 1280, "height": 800}, has_touch=True)
    page = context.new_page()
    page.add_init_script("window.shown = (id) => getComputedStyle(document.getElementById(id)).display !== 'none';")
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(base + "/calib")
    markers_ready(page)
    page.wait_for_timeout(600)
    result["windowed"] = page.evaluate(
        "() => ({button: shown('fs'), warn: document.getElementById('warn').textContent,"
        " warn_hidden: !shown('warn')})")
    page.screenshot(path=str(out / "windowed.png"))
    page.tap("#fs")  # must not count as a touch on the screen; fullscreen itself may not exist in a headless browser
    page.wait_for_timeout(1200)
    result["after_button"] = page.evaluate(
        "() => ({fullscreen: !!document.fullscreenElement, w: innerWidth, h: innerHeight,"
        " button: shown('fs'), warn: document.getElementById('warn').textContent})")
    page.screenshot(path=str(out / "after_button.png"))
    browser.close()

result["errors"] = errors
(out / "result.json").write_text(json.dumps(result))
"""


def _playwright_python() -> str | None:
    """A Python interpreter that has Playwright: the one behind the `playwright` command on PATH."""
    command = shutil.which("playwright")
    candidates = [sys.executable]
    if command:
        try:
            first = pathlib.Path(command).read_text(errors="replace").splitlines()[0]
        except (OSError, IndexError):
            first = ""
        if first.startswith("#!"):
            candidates.insert(0, first[2:].strip().split()[0])
    for python in candidates:
        try:
            if (
                subprocess.run(
                    [python, "-c", "import playwright.sync_api"], capture_output=True, timeout=60, env=_clean_env()
                ).returncode
                == 0
            ):
                return python
        except (OSError, subprocess.SubprocessError):
            continue
    return None


def _clean_env() -> dict[str, str]:
    """The environment for the other interpreter: without this venv's PYTHONPATH."""
    return {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}


def _differing(shot: np.ndarray, reference: np.ndarray, levels: int = 64) -> int:
    """How many pixels of `shot` are more than `levels` away from every pixel of `reference` within one pixel of
    them. The browser puts an edge that falls inside a pixel on the pixel grid, the rendering shades that pixel:
    a difference of one pixel along an edge is not a difference."""
    smallest = np.full(shot.shape[:2], 255, np.int16)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            moved = np.roll(reference, (dy, dx), axis=(0, 1)).astype(np.int16)
            smallest = np.minimum(smallest, np.abs(shot.astype(np.int16) - moved).max(axis=2))
    return int((smallest[1:-1, 1:-1] > levels).sum())


def test_page_in_headless_browser(live_server, tmp_path: pathlib.Path) -> None:
    base_url, state = live_server
    python = _playwright_python()
    if python is None:
        pytest.skip("no Python with Playwright on this machine")
    run = subprocess.run(
        [python, "-c", _BROWSER_SCRIPT, base_url, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=180,
        env=_clean_env(),
    )
    if run.returncode != 0 and ("Executable doesn't exist" in run.stderr or "playwright install" in run.stderr):
        pytest.skip("Playwright has no browser installed")
    assert run.returncode == 0, run.stderr[-3000:]
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["errors"] == []
    keep = os.environ.get("TABLET_TAP_KEEP_SHOTS")
    if keep:
        shutil.copytree(tmp_path, keep, dirs_exist_ok=True)

    # 1. the page covers the 2560x1600 display: markers exactly where render_layout() puts them, no button, no warning
    assert result["first"] == {"w": 2560, "h": 1600, "sw": 2560, "sh": 1600, "button": False, "warn": ""}
    shot = cv2.imread(str(tmp_path / "markers.png"))
    assert shot.shape == (1600, 2560, 3)
    reference = sv_layout.render_layout(LAYOUT, 2560)
    found, expected = sv_rectifier.detect_markers(shot), sv_rectifier.detect_markers(reference)
    assert sorted(found) == sorted(expected) == [0, 1, 2, 3, 4, 5]
    worst = max(np.abs(found[i] - expected[i]).max() for i in found)
    print(
        f"\nbrowser 2560x1600: marker corners within {worst:.2f} px of render_layout; "
        f"{_differing(shot, reference)} pixels unlike the rendering"
    )
    assert worst < 1.0
    assert _differing(shot, reference) == 0
    assert sv_rectifier.ScreenRectifier.from_marker_image(shot, LAYOUT).reprojection_px < 0.5

    # 2. a tap: the server gets it in (u, v), the page draws the dot there, find_touch_dot() finds it
    events, _ = state.touches_after(0)
    downs = [e for e in events if e["type"] == "pointerdown"]
    assert (downs[0]["u"], downs[0]["v"]) == pytest.approx((700 / 2560, 900 / 1600), abs=1e-6)
    assert downs[0]["pointer_type"] == "touch" and abs(downs[0]["t_ms"] / 1000 - downs[0]["t_server"]) < 5
    assert [e["type"] for e in events[:2]] == ["pointerdown", "pointerup"]
    assert result["info"] == "u=0.2734  v=0.5625"
    dot_shot = cv2.imread(str(tmp_path / "dot.png"))
    assert sv_rectifier.find_touch_dot(dot_shot) == pytest.approx((700 / 2560, 900 / 1600), abs=1e-4)
    reference = sv_layout.render_layout(LAYOUT, 2560, dot=(700 / 2560, 900 / 1600))
    patch = (slice(900 - 60, 900 + 60), slice(700 - 60, 700 + 60))
    assert np.abs(dot_shot[patch].astype(np.int16) - reference[patch]).mean() < 2.0  # the same disc and ring
    assert sorted(sv_rectifier.detect_markers(dot_shot)) == [0, 1, 2, 3, 4, 5]

    # 3. mode "target": no markers, the crosshair where render_layout() draws it, the dot still there
    target_shot = cv2.imread(str(tmp_path / "target.png"))
    assert sv_rectifier.detect_markers(target_shot) == {}
    reference = sv_layout.render_layout(LAYOUT, 2560, mode="target", target=(0.3, 0.6), dot=(700 / 2560, 900 / 1600))
    reference[1500:, 400:1100] = target_shot[1500:, 400:1100]  # the small (u, v) text is not rendered
    assert _differing(target_shot, reference) < 40  # the ring's anti-aliasing is not the browser's to the pixel
    assert tuple(target_shot[960, 768]) == (255, 255, 255) and tuple(target_shot[960 + 40, 768 + 40]) == (
        0x20,
        0x20,
        0x20,
    )

    # 4. mode "blank" with the second tap's dot (sub-pixel position), then cleared
    assert (downs[1]["u"], downs[1]["v"]) == pytest.approx((2000.5 / 2560, 300.25 / 1600), abs=1e-6)
    blank_dot = cv2.imread(str(tmp_path / "blank_dot.png"))
    assert sv_rectifier.detect_markers(blank_dot) == {}
    assert sv_rectifier.find_touch_dot(blank_dot) == pytest.approx((2000.5 / 2560, 300.25 / 1600), abs=1e-4)
    assert result["dot_hidden_after_clear"] is True
    blank = cv2.imread(str(tmp_path / "blank.png"))
    assert sv_rectifier.find_touch_dot(blank) is None and int(blank.max()) == 0x20 and int(blank.min()) == 0x20

    # 5. a drag on the touch screen: down, moves, up, in that order, nothing cancelled
    rest = [e for e in events if downs[1]["seq"] + 1 < e["seq"] <= result["seq_main"]]
    drag = [e for e in rest if e["pointer_id"] != 77]
    kinds = [e["type"] for e in drag]
    assert kinds[0] == "pointerdown" and kinds[-1] == "pointerup" and set(kinds[1:-1]) == {"pointermove"}
    assert 1 <= kinds.count("pointermove") <= 10
    assert drag[0]["u"] == pytest.approx(1000 / 2560) and drag[-1]["u"] == pytest.approx(1200 / 2560)
    assert all(e["v"] == pytest.approx(0.5) and e["pointer_type"] == "touch" for e in drag)
    # ... and events every 4 ms are reported at about 30 Hz: the down, the up, and a move every 33 ms or more
    fast = [e for e in rest if e["pointer_id"] == 77]
    moves = [e for e in fast if e["type"] == "pointermove"]
    print(f"{result['fired']} pointermove events in 0.4 s -> {len(moves)} reported")
    assert result["fired"] >= 40
    assert [e["type"] for e in fast] == ["pointerdown"] + ["pointermove"] * len(moves) + ["pointerup"]
    assert 6 <= len(moves) <= 13
    assert min(np.diff([e["t_ms"] for e in moves])) >= 32
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))

    # 6. the tablet's own numbers: 1280x800 CSS pixels at devicePixelRatio 2 is the same picture on 2560x1600 pixels
    assert result["dpr2"] == {"w": 1280, "h": 800, "dpr": 2, "button": False, "warn": ""}
    tap = [e for e in events if e["seq"] > result["seq_main"]]
    assert [e["type"] for e in tap] == ["pointerdown", "pointerup"]
    assert (tap[0]["u"], tap[0]["v"]) == pytest.approx((333.5 / 1280, 444.5 / 800), abs=1e-6)
    dpr2 = cv2.imread(str(tmp_path / "dpr2.png"))
    assert dpr2.shape == (1600, 2560, 3)
    reference = sv_layout.render_layout(LAYOUT, 2560, dot=(333.5 / 1280, 444.5 / 800))
    reference[1500:, 400:1100] = dpr2[1500:, 400:1100]  # the small (u, v) text
    assert _differing(dpr2, reference) < 40
    assert sv_rectifier.find_touch_dot(dpr2) == pytest.approx((333.5 / 1280, 444.5 / 800), abs=1e-4)
    found = sv_rectifier.detect_markers(dpr2)
    assert max(np.abs(found[i] - expected[i]).max() for i in range(6)) < 1.0

    # 7. a window smaller than the display: the red warning and the button; the button is not a touch
    assert result["windowed"]["button"] is True and result["windowed"]["warn_hidden"] is False
    assert (
        "没有铺满" in result["windowed"]["warn"]
        and "1280×720" in result["windowed"]["warn"]
        and "1280×800" in result["windowed"]["warn"]
    )
    assert state.touches_after(0)[1] == events[-1]["seq"]  # tapping the button added no touch event
    screen_now = state.snapshot()["screen"]
    print(f"after the fullscreen button in the headless browser: {result['after_button']}")
    if result["after_button"]["fullscreen"]:
        assert result["after_button"]["button"] is False and screen_now["fullscreen"] is True
    else:
        assert screen_now["covers_display"] is False and (screen_now["inner_w"], screen_now["inner_h"]) == (1280, 720)
