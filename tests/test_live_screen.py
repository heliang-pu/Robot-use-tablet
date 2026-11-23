"""Following the tablet with the camera, and correcting the camera with pen contacts."""

from helpers import ASPECT
from helpers import true_screen
import numpy as np
import pytest

from tablet_tap.calibration.live_screen import LiveScreen
from tablet_tap.core.kinematics import pose
from tablet_tap.core.pinhole import CameraModel
from tablet_tap.core.screen_map import ScreenMap
from tablet_tap.vision.layout import default_layout

MODEL = CameraModel(1920, 1080, 1360.0, 1360.0, 960.0, 540.0)
LAYOUT = default_layout(ASPECT)
CAMERA = pose(np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]), np.array([0.3, -0.1, 0.7]))


def markers(screen, ids=None, noise=0.0, rng=None):
    inverse, out = np.linalg.inv(CAMERA), {}
    for marker_id, (u0, v0, u1, v1) in LAYOUT.markers.items():
        if ids is None or marker_id in ids:
            points = np.array([screen.point(u, v) for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))])
            out[marker_id] = MODEL.project(points @ inverse[:3, :3].T + inverse[:3, 3])
            if noise:
                out[marker_id] += rng.normal(0, noise, (4, 2))
    return out


def slid(screen, dx, dy, turn_deg):
    """The screen moved on its own plane."""
    live = LiveScreen.from_map(screen)
    move, a = np.eye(4), np.radians(turn_deg)
    move[:2, :2], move[:2, 3] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], [dx, dy]
    live.pose = live.pose @ move
    return live.screen_map(), live.pose


@pytest.mark.parametrize("ids", [None, (1, 4), (2, 5)])
def test_follows_a_moved_tablet_even_half_covered(ids):
    start = true_screen(tilt_deg=2.0)
    live = LiveScreen.from_map(start)
    there, want = slid(start, 0.08, -0.05, 12.0)
    got = live.track(markers(there, ids, 0.3, np.random.default_rng(0)), LAYOUT, MODEL, CAMERA)
    live.pose = want
    distance, turn = live.moved(got)
    assert distance < 1.5e-3 and turn < 0.8


def test_one_marker_is_not_enough_and_nonsense_is_refused():
    start = true_screen(tilt_deg=2.0)
    live = LiveScreen.from_map(start)
    assert live.track(markers(start, (1,)), LAYOUT, MODEL, CAMERA) is None
    seen = markers(start)
    seen[1] = seen[1] + 80.0
    assert live.track(seen, LAYOUT, MODEL, CAMERA) is None


def test_contacts_correct_the_camera_and_survive_a_move():
    real = true_screen(tilt_deg=2.0)
    off = np.array([0.004, -0.003, 0.006])
    live = LiveScreen.from_map(ScreenMap(real.origin + off, real.right, real.down, real.tool_x))
    live.add((0.5, 0.5), real.point(0.5, 0.5))
    assert not live.affine and np.linalg.norm(live.screen_map().point(0.2, 0.8) - real.point(0.2, 0.8)) < 1e-6
    for uv in ((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)):
        live.add(uv, real.point(*uv))
    assert live.affine and live.screen_map().residual_mm < 1e-3
    there, want = slid(real, 0.05, 0.02, 8.0)
    move = want @ np.linalg.inv(LiveScreen.from_map(real).pose)
    live.pose = move @ live.pose  # what the camera, off by the same amount, reports after the move
    again = LiveScreen.from_json(live.to_json()).screen_map()
    assert np.linalg.norm(again.point(0.3, 0.7) - there.point(0.3, 0.7)) < 1e-6


def test_one_late_touch_does_not_tilt_the_plane():
    """The bench's contacts of 2026-11-23: the first was pressed 13 mm deep and slid before the screen reported it."""
    real = true_screen(tilt_deg=2.0)
    live = LiveScreen.from_map(real)
    w, h = live.size_m
    for u, v, off in ((0.498, 0.073, (6.1, 9.7, 10.9)), (0.499, 0.895, (4.6, 1.0, -2.5)),
                      (0.888, 0.887, (5.7, 2.5, -4.1)), (0.106, 0.927, (5.7, 1.1, -1.2))):  # fmt: skip
        local = np.array([u * w, v * h, 0.0]) + np.array(off) * 1e-3
        live.add((u, v), live.pose[:3, :3] @ local + live.pose[:3, 3])
    fitted = live.screen_map()
    assert len(live.trusted()) == 3 and not live.affine
    assert np.degrees(np.arccos(np.clip(fitted.normal @ real.normal, -1, 1))) < 0.1


def test_a_small_real_tilt_is_learned_but_only_slowly():
    real, live = true_screen(tilt_deg=2.5), LiveScreen.from_map(true_screen(tilt_deg=2.0))
    for u in (0.1, 0.5, 0.9):
        for v in (0.1, 0.5, 0.9):
            live.add((u, v), real.point(u, v))
    off = [np.linalg.norm(live.screen_map().point(u, v) - real.point(u, v)) for u in (0, 1) for v in (0, 1)]
    assert live.affine and max(off) < 0.5e-3


def test_printed_tags_stand_in_for_the_page():
    start = true_screen(tilt_deg=25.0)  # a slanted stand
    live = LiveScreen.from_map(start)
    quads = {40: ((-0.2, 0.3), (-0.08, 0.3), (-0.08, 0.5), (-0.2, 0.5)), 41: ((1.08, 0.3), (1.2, 0.3), (1.2, 0.5), (1.08, 0.5))}

    def tags(screen):
        inverse = np.linalg.inv(CAMERA)
        return {k: MODEL.project(np.array([screen.point(u, v) for u, v in q]) @ inverse[:3, :3].T + inverse[:3, 3]) for k, q in quads.items()}  # fmt: skip

    inverse = np.linalg.inv(CAMERA)
    quad = MODEL.project(live.corners() @ inverse[:3, :3].T + inverse[:3, 3])
    import cv2

    to_norm = cv2.getPerspectiveTransform(quad.astype(np.float32), np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]))
    assert live.track(tags(start), LAYOUT, MODEL, CAMERA) is None  # not learned yet
    live.learn_tags({**markers(start), **tags(start)}, to_norm)
    there, want = slid(start, 0.03, 0.02, 4.0)
    got = LiveScreen.from_json(live.to_json()).track(tags(there), LAYOUT, MODEL, CAMERA)  # no page markers at all
    live.pose = want
    distance, turn = live.moved(got)
    assert distance < 1e-3 and turn < 0.3


def test_another_slant_is_noticed():
    live, slanted = LiveScreen.from_map(true_screen(tilt_deg=2.0)), LiveScreen.from_map(true_screen(tilt_deg=20.0))
    assert live.off_plane(slanted.pose)[1] == pytest.approx(18.0, abs=0.1)
    assert live.off_plane(live.pose) == pytest.approx((0.0, 0.0), abs=1e-4)
