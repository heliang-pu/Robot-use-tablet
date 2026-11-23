"""Hand-eye calibration with a chessboard, in a synthetic scene rendered through a pinhole camera: no hardware.

The scene: the bench's top camera (its real intrinsics at 1920 x 1080) above and behind the arm, a 12 x 9 chessboard of
15 mm squares lying on the table, and a tablet on a 12 degree stand showing the calibration page.
"""

import contextlib
import json
import threading

import cv2
from helpers import ARM
from helpers import PARK
from helpers import SEED
from helpers import true_screen
import numpy as np
import pytest
from werkzeug.serving import make_server

from tablet_tap import app
from tablet_tap import config
from tablet_tap.calibration import hand_eye
from tablet_tap.calibration import page_server
from tablet_tap.cli import main as cli
from tablet_tap.cli.commands import board as board_command
from tablet_tap.core.kinematics import pose
from tablet_tap.core.pinhole import CameraModel
from tablet_tap.core.screen_map import ScreenMap
from tablet_tap.errors import CalibrationError
from tablet_tap.robot.arm import FakeArm
from tablet_tap.vision import layout as screen_layout
from tablet_tap.vision.board import BoardSpec
from tablet_tap.vision.board import find_chessboard
from tablet_tap.vision.board import numbered_extremes
from tablet_tap.vision.board import numbered_points
from tablet_tap.vision.rectifier import detect_markers

MODEL = CameraModel(1920, 1080, 1361.72, 1358.84, 975.76, 569.26)
SPEC = BoardSpec(cols=11, rows=8, pitch_m=0.015)
MARGIN_M = 0.010  # white border of the printed board around its 12 x 9 squares
ASPECT = 0.25 / 0.16


def look_at(eye, target) -> np.ndarray:
    """camera_in_base of a camera at ``eye`` looking at ``target`` with the picture's top toward +x of the base."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = (target - eye) / np.linalg.norm(target - eye)
    up = np.array([1.0, 0.0, 0.0])
    y = -(up - (up @ z) * z)
    y /= np.linalg.norm(y)
    return pose(np.stack([np.cross(y, z), y, z], axis=1), eye)


CAMERA = look_at([0.02, -0.02, 0.66], [0.31, -0.02, 0.0])


def board_pose(centre=(0.27, -0.06), yaw_deg=8.0, top_z=0.0) -> np.ndarray:
    """board_in_base of a board lying on the table: z into the board (down), its inner-corner grid centred there."""
    yaw = np.radians(yaw_deg)
    x, y = np.array([np.cos(yaw), np.sin(yaw), 0.0]), np.array([np.sin(yaw), -np.cos(yaw), 0.0])
    rot = np.stack([x, y, np.cross(x, y)], axis=1)
    middle = np.array([(SPEC.cols - 1) / 2, (SPEC.rows - 1) / 2, 0.0]) * SPEC.pitch_m
    return pose(rot, np.array([*centre, top_z]) - rot @ middle)


def render_plane(texture, texture_size_m, plane_in_camera, frame) -> np.ndarray:
    """Draw a flat texture (its top-left at the plane frame's origin, x right, y down, ``texture_size_m`` wide and
    high) into ``frame`` as the camera sees it."""
    h, w = texture.shape[:2]
    to_metres = np.diag([texture_size_m[0] / w, texture_size_m[1] / h, 1.0]) @ np.array(
        [[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1.0]]
    )
    plane_to_pixels = MODEL.matrix @ plane_in_camera[:3, [0, 1, 3]]
    hom = np.array([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1.0]]) @ plane_to_pixels @ to_metres
    size = (frame.shape[1], frame.shape[0])
    warped = cv2.warpPerspective(texture, hom, size, flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), hom, size, flags=cv2.INTER_AREA)
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    return (frame * (1 - alpha) + warped * alpha).astype(np.uint8)


def table_frame() -> np.ndarray:
    frame = np.full((MODEL.height, MODEL.width, 3), (150, 140, 110), np.uint8)  # a blue-grey cloth
    return frame


def chessboard_frame(board_in_base, noise=1.5, seed=0) -> np.ndarray:
    px_per_m = 12000.0
    squares = (SPEC.cols + 1, SPEC.rows + 1)
    size_m = (squares[0] * SPEC.pitch_m + 2 * MARGIN_M, squares[1] * SPEC.pitch_m + 2 * MARGIN_M)
    texture = np.full((round(size_m[1] * px_per_m), round(size_m[0] * px_per_m), 3), 225, np.uint8)
    for i in range(squares[0]):
        for j in range(squares[1]):
            if (i + j) % 2 == 0:
                x0, y0 = (MARGIN_M + i * SPEC.pitch_m) * px_per_m, (MARGIN_M + j * SPEC.pitch_m) * px_per_m
                x1, y1 = x0 + SPEC.pitch_m * px_per_m, y0 + SPEC.pitch_m * px_per_m
                texture[round(y0) : round(y1), round(x0) : round(x1)] = 25
    # the texture's top-left corner is one square and the margin before the first inner corner
    shift = pose(np.eye(3), [-(MARGIN_M + SPEC.pitch_m), -(MARGIN_M + SPEC.pitch_m), 0.0])
    plane_in_camera = np.linalg.inv(CAMERA) @ board_in_base @ shift
    frame = render_plane(texture, size_m, plane_in_camera, table_frame())
    frame = cv2.GaussianBlur(frame, (0, 0), 0.8)
    rng = np.random.default_rng(seed)
    return np.clip(frame + rng.normal(0, noise, frame.shape), 0, 255).astype(np.uint8)


def page_frame(screen: ScreenMap, frame=None) -> np.ndarray:
    """The tablet's calibration page on the screen ``screen`` (in the base frame), as the camera sees it."""
    texture = screen_layout.render_layout(screen_layout.default_layout(ASPECT), 2000)
    w, h = screen.size_m
    rot = np.stack([screen.right / w, screen.down / h, -screen.normal], axis=1)
    plane_in_camera = np.linalg.inv(CAMERA) @ pose(rot, screen.origin)
    out = render_plane(texture, (w, h), plane_in_camera, table_frame() if frame is None else frame)
    return cv2.GaussianBlur(out, (0, 0), 0.7)


def grid_in_base(grid, board_in_base) -> dict[tuple[int, int], np.ndarray]:
    """Which physical corner each detected grid corner is: its true position in the base frame."""
    truth = SPEC.corners() @ board_in_base[:3, :3].T + board_in_base[:3, 3]
    seen = MODEL.project(truth @ np.linalg.inv(CAMERA)[:3, :3].T + np.linalg.inv(CAMERA)[:3, 3])
    return {
        (r, c): truth[int(np.argmin(np.linalg.norm(seen - grid[r, c], axis=1)))]
        for r in range(grid.shape[0])
        for c in range(grid.shape[1])
    }


def touch_pose(point, lean_deg=0.0, about=(0.0, 1.0, 0.0)) -> np.ndarray:
    """A pen-tip pose on ``point`` with the pen pointing down, optionally leaning."""
    down = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
    lean, _ = cv2.Rodrigues(np.radians(lean_deg) * np.asarray(about, float))
    return pose(lean @ down, point)


def touches_of(grid, board_in_base, order=None, noise_mm=0.0, seed=0):
    rng = np.random.default_rng(seed)
    where = grid_in_base(grid, board_in_base)
    numbered = numbered_extremes(grid)
    return [
        hand_eye.BoardTouch(r, c, touch_pose(where[numbered[i]] + rng.normal(0, noise_mm * 1e-3, 3)))
        for i, (r, c) in zip(order or range(4), numbered, strict=False)
    ]


def touches_at(grid, board_in_base, numbered, noise_mm=0.0, seed=0):
    rng = np.random.default_rng(seed)
    where = grid_in_base(grid, board_in_base)
    return [
        hand_eye.BoardTouch(r, c, touch_pose(where[(r, c)] + rng.normal(0, noise_mm * 1e-3, 3))) for r, c in numbered
    ]


def pose_error(found, truth) -> tuple[float, float]:
    """(mm, degrees) between two poses."""
    turn = np.degrees(np.linalg.norm(cv2.Rodrigues(found[:3, :3].T @ truth[:3, :3])[0]))
    return float(np.linalg.norm(found[:3, 3] - truth[:3, 3]) * 1e3), float(turn)


@pytest.fixture(scope="module")
def scene():
    board_in_base = board_pose()
    frame = chessboard_frame(board_in_base)
    return {"board": board_in_base, "frame": frame, "grid": find_chessboard(frame, SPEC.cols, SPEC.rows)}


def test_the_board_is_found_and_numbered_as_a_person_reads_the_picture(scene):
    grid = scene["grid"]
    assert grid.shape == (SPEC.rows, SPEC.cols, 2)
    assert find_chessboard(scene["frame"]).shape[:2] in ((8, 11), (11, 8))  # the size is found without being told
    with pytest.raises(CalibrationError, match="能认出的是"):
        find_chessboard(scene["frame"], cols=13, rows=9)
    with pytest.raises(CalibrationError, match="没有找到"):
        find_chessboard(table_frame())
    numbered = numbered_extremes(grid)
    points = np.array([grid[r, c] for r, c in numbered])
    assert int(np.argmin(points.sum(axis=1))) == 0  # number 1 is the corner nearest the picture's top-left
    edges = np.roll(points, -1, axis=0) - points
    turns = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    assert np.all(turns > 0)  # 1 -> 2 -> 3 -> 4 goes clockwise in the picture
    in_camera, rms, ratio = hand_eye.board_in_camera(SPEC, grid, MODEL)
    assert rms < 0.3 and ratio > 3
    # the detector may start counting at either end of the board, so compare the grid's centre, not its origin
    middle = np.array([(SPEC.cols - 1) / 2 * SPEC.pitch_m, (SPEC.rows - 1) / 2 * SPEC.pitch_m, 0.0, 1.0])
    assert np.linalg.norm((in_camera @ middle - np.linalg.inv(CAMERA) @ scene["board"] @ middle)[:3]) < 0.004


def test_eight_points_go_round_the_board_and_give_a_steadier_result(scene):
    grid = scene["grid"]
    eight, four = numbered_points(grid, 8), numbered_points(grid, 4)
    assert eight[::2] == four == numbered_extremes(grid) and len(set(eight)) == 8
    assert all(r in (0, 3, 7) and c in (0, 5, 10) for r, c in eight)  # corners and the middles of the sides
    points = np.array([grid[r, c] for r, c in eight])
    centre = points.mean(axis=0)
    angles = np.unwrap(np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0]))
    assert np.all(np.diff(angles) > 0)  # numbered clockwise in the picture (y points down)
    with pytest.raises(ValueError):
        numbered_points(grid, 6)

    def camera_scatter(numbered) -> float:
        found = [
            hand_eye.solve_hand_eye(SPEC, grid, MODEL, touches_at(grid, scene["board"], numbered, 1.0, seed))
            for seed in range(40)
        ]
        return float(np.mean([pose_error(r.camera_in_base, CAMERA)[0] for r in found]))

    assert camera_scatter(eight) < 0.85 * camera_scatter(four)


def test_camera_pose_from_exact_touches(scene):
    result = hand_eye.solve_hand_eye(SPEC, scene["grid"], MODEL, touches_of(scene["grid"], scene["board"]))
    miss_mm, miss_deg = pose_error(result.camera_in_base, CAMERA)
    assert miss_mm < 4.0 and miss_deg < 0.4, (miss_mm, miss_deg)
    assert max(result.touch_residual_mm) < 0.05 and result.span_error_mm < 0.05
    # what matters: a point the camera sees 15 cm beside the board lands within 1.5 mm in the arm's frame
    beside = scene["board"] @ np.array([-0.15, 0.05, 0.0, 1.0])
    seen = np.linalg.inv(CAMERA) @ beside
    assert np.linalg.norm((result.camera_in_base @ seen - beside)[:3]) < 1.5e-3
    assert "上方" in hand_eye.describe_camera(result.camera_in_base)
    assert json.dumps(result.to_json())  # plain numbers and lists


def test_three_touches_are_enough_and_the_uncertainty_is_reported(scene):
    touches = touches_of(scene["grid"], scene["board"], noise_mm=0.7, seed=3)
    result = hand_eye.solve_hand_eye(SPEC, scene["grid"], MODEL, touches[:3])
    assert pose_error(result.camera_in_base, CAMERA)[0] < 15.0
    spread = hand_eye.uncertainty(SPEC, scene["grid"], MODEL, touches, samples=80)
    assert 0.2 < spread["centre_mm"] < 2.0 and spread["outside_mm"] > spread["centre_mm"]  # worse away from the board
    assert 0.05 < spread["normal_deg"] < 2.0


def test_touches_that_do_not_belong_to_the_board_are_refused(scene):
    grid, board = scene["grid"], scene["board"]
    with pytest.raises(CalibrationError, match="底面朝上"):  # going round the board the other way
        hand_eye.solve_hand_eye(SPEC, grid, MODEL, touches_of(grid, board, order=[0, 3, 2, 1]))
    wrong_pitch = BoardSpec(cols=SPEC.cols, rows=SPEC.rows, pitch_m=0.020)
    with pytest.raises(CalibrationError, match="对不上"):
        hand_eye.solve_hand_eye(wrong_pitch, grid, MODEL, touches_of(grid, board))
    with pytest.raises(CalibrationError, match="至少 3"):
        hand_eye.solve_hand_eye(SPEC, grid, MODEL, touches_of(grid, board)[:2])
    row = [hand_eye.BoardTouch(0, c, touch_pose(grid_in_base(grid, board)[(0, c)])) for c in (0, 5, 10)]
    with pytest.raises(CalibrationError, match="直线"):
        hand_eye.solve_hand_eye(SPEC, grid, MODEL, row)
    # the one mistake geometry cannot see: every number taken for the corner opposite. The fit is perfect and the
    # camera ends up on the other side of the board, which is why the command asks the operator to confirm.
    turned = hand_eye.solve_hand_eye(SPEC, grid, MODEL, touches_of(grid, board, order=[2, 3, 0, 1]))
    assert max(turned.touch_residual_mm) < 0.05 and pose_error(turned.camera_in_base, CAMERA)[0] > 300


def test_glass_over_the_pattern(scene):
    """A pattern on the back of 3 mm glass: the pen touches 3 mm above what the camera sees."""
    glass = BoardSpec(cols=SPEC.cols, rows=SPEC.rows, pitch_m=SPEC.pitch_m, glass_m=0.003)
    where, numbered = grid_in_base(scene["grid"], scene["board"]), numbered_extremes(scene["grid"])
    touches = [hand_eye.BoardTouch(r, c, touch_pose(where[(r, c)] + [0, 0, 0.003])) for r, c in numbered]
    with_glass = hand_eye.solve_hand_eye(glass, scene["grid"], MODEL, touches)
    without = hand_eye.solve_hand_eye(SPEC, scene["grid"], MODEL, touches)
    assert pose_error(with_glass.camera_in_base, CAMERA)[0] < 4.0
    assert abs((without.camera_in_base[2, 3] - with_glass.camera_in_base[2, 3]) * 1e3 - 3.0) < 0.2  # 3 mm too high


def test_pen_length_shows_when_the_pen_leans_differently(scene):
    where, numbered = grid_in_base(scene["grid"], scene["board"]), numbered_extremes(scene["grid"])
    true_arm = ARM.__class__(tip_len=0.055)  # the pen really sticks out 55 mm; the setup says 65
    leans = [(0.0, (0, 1, 0)), (-25.0, (0, 1, 0)), (-12.0, (0, 1, 0)), (-18.0, (0, 1, 0))]  # leaning away from the arm
    joints = []
    for (row, col), (lean, about) in zip(numbered, leans, strict=True):
        joints.append((row, col, true_arm.solve(touch_pose(where[(row, col)], lean, about), SEED)))
    check = hand_eye.tip_length_check(ARM, SPEC, joints)
    assert abs(check["tip_len_mm"] - 55.0) < 1.0 and check["rms_mm"] < 0.3 < check["rms_now_mm"]
    straight = [(r, c, true_arm.solve(touch_pose(where[(r, c)]), SEED)) for r, c in numbered]
    assert hand_eye.tip_length_check(ARM, SPEC, straight) is None  # the same direction every time hides the length


def test_the_screen_from_the_camera(scene):
    screen = true_screen(centre=(0.30, 0.02, 0.03))
    frame = page_frame(screen)
    markers = detect_markers(frame)
    assert len(markers) == 6
    layout = screen_layout.default_layout(ASPECT)
    found, rms, ratio = hand_eye.screen_from_camera(markers, layout, screen.size_m, MODEL, CAMERA)
    assert rms < 0.5 and ratio > 2
    for u, v in ((0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)):
        assert np.linalg.norm(found.point(u, v) - screen.point(u, v)) < 2.5e-3, (u, v)
    assert np.degrees(np.arccos(np.clip(found.normal @ screen.normal, -1, 1))) < 0.6
    seed = hand_eye.seed_joints(ARM, found)
    assert abs(found.uv_of(ARM.fk(seed)[:3, 3])[2] - 0.015) < 2e-4  # hovering over the centre
    with pytest.raises(CalibrationError, match="至少要 2 个"):
        hand_eye.screen_from_camera({0: markers[0]}, layout, screen.size_m, MODEL, CAMERA)


# ---- the commands, end to end ---------------------------------------------------------------------------------------
class PressingArm(FakeArm):
    """A software arm whose pen presses a hidden screen; touches go to the calibration page's state (as test_cli)."""

    def __init__(self, q, screen, state) -> None:
        super().__init__(q)
        self.screen, self.state, self.down = screen, state, False

    def command(self, q) -> None:
        super().command(q)
        u, v, h = self.screen.uv_of(ARM.fk(self.q)[:3, 3])
        pressed = h <= -0.0003 and 0 <= u <= 1 and 0 <= v <= 1
        if pressed != self.down:
            self.state.add_touch({"type": "pointerdown" if pressed else "pointerup", "u": u, "v": v, "t_ms": 0})
        self.down = pressed


def test_board_then_place_then_probe(scene, tmp_path, monkeypatch, capsys):
    """The whole flow on files: the board gives the camera's pose, the camera gives the screen, the probe refines it."""
    state = page_server.CalibState()
    server = make_server("127.0.0.1", 0, page_server.create_app(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state.set_screen(
        {"inner_w": 2500, "inner_h": 1600, "screen_w": 2500, "screen_h": 1600, "dpr": 1, "fullscreen": True}
    )
    screen = true_screen(centre=(0.30, 0.02, 0.03))
    arm = PressingArm(PARK, screen, state)
    monkeypatch.setattr(app, "open_arm", lambda *args, **kwargs: arm)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    cv2.imwrite(str(tmp_path / "board.png"), scene["frame"])
    cv2.imwrite(str(tmp_path / "tablet.png"), page_frame(screen))
    (tmp_path / "k.json").write_text(json.dumps(MODEL.to_json()))
    setup = tmp_path / "calib" / "setup.json"
    setup.parent.mkdir()
    setup.write_text(json.dumps({"park_q": PARK.tolist()}))
    base = ["--setup", str(setup), "--view", str(tmp_path / "calib" / "view.json"), "--fake", "--yes",
            "--calib_url", f"http://127.0.0.1:{server.server_port}"]  # fmt: skip

    # the operator: put the pen on corners 1, 2 and 4 as numbered in the picture (3 is "out of reach")
    where, numbered = grid_in_base(scene["grid"], scene["board"]), numbered_extremes(scene["grid"])
    script = []
    for number, (row, col) in enumerate(numbered, 1):
        if number == 3:
            script.append(lambda: "n")
            continue
        q = ARM.solve(touch_pose(where[(row, col)]), SEED)
        script.append(lambda q=q: (setattr(arm, "q", q.copy()), "m")[1])
    script.append(lambda: "q")
    keys = iter(script)

    @contextlib.contextmanager
    def scripted_keys():
        yield lambda timeout: next(keys)()

    monkeypatch.setattr(board_command, "raw_keys", scripted_keys)
    code = cli.main([*base, "--camera", f"file:{tmp_path / 'board.png'}", "board",  # 15 mm squares: the default
                     "--intrinsics", str(tmp_path / "k.json"), "--frames", "1"])  # fmt: skip
    out = capsys.readouterr().out
    assert code == 0, out
    assert "12 × 9 格" in out and "上方" in out and (tmp_path / "calib" / "board_corners.jpg").is_file()
    saved = json.loads(setup.read_text())
    assert pose_error(np.array(saved["camera_pose"]), CAMERA)[0] < 6.0
    assert saved["camera_model"]["width"] == 1920 and len(saved["hand_eye"]["touched"]) == 3

    # the tablet is put down somewhere: the camera says where its screen is
    arm.q = PARK.copy()
    code = cli.main([*base, "--camera", f"file:{tmp_path / 'tablet.png'}", "place", "--screen_mm", "250", "160"])
    out = capsys.readouterr().out
    assert code == 0, out
    saved = json.loads(setup.read_text())
    placed = ScreenMap.from_json(saved["screen_map"])
    worst = max(np.linalg.norm(placed.point(u, v) - screen.point(u, v)) for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)))
    assert worst < 6e-3 and saved["screen_map_source"] == "camera" and "可达性" in out
    assert (tmp_path / "calib" / "view.json").is_file()

    # and the probe makes it exact, from higher up and deeper than usual
    assert cli.main([*base, "probe", "--nu", "4", "--nv", "3"]) == 0
    out = capsys.readouterr().out
    saved = json.loads(setup.read_text())
    refined = ScreenMap.from_json(saved["screen_map"])
    worst = max(np.linalg.norm(refined.point(u, v) - screen.point(u, v)) for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)))
    assert "悬停高度提到 20 mm" in out and "12 / 12 个点触发" in out
    assert worst < 0.7e-3 and saved["screen_map_source"] == "touch" and "hover_m" not in saved["limits"]
    assert np.allclose(saved["screen_size_m"], [0.25, 0.16], atol=1e-3)

    # a check against the probed plane tells how good the camera's estimate was
    assert cli.main([*base, "--camera", f"file:{tmp_path / 'tablet.png'}", "place", "--check"]) == 0
    assert "这就是手眼标定的误差" in capsys.readouterr().out
    server.shutdown()


def test_place_says_what_is_missing(tmp_path, capsys):
    base = ["--setup", str(tmp_path / "s.json"), "--fake"]
    assert cli.main([*base, "place"]) == 1
    assert "board" in capsys.readouterr().out
