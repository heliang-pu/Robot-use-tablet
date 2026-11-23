"""Where the camera is, in the arm's base frame, from a calibration board that both of them can reach.

The camera sees the board's corners (``board_in_camera``); the pen tip is put on a few of them, which places the board
in the arm's base frame (``rigid_fit``); the two together give the camera's pose (``solve_hand_eye``). With that pose,
a camera frame of the tablet's calibration page is enough to say where the screen is (``screen_from_camera``): a
starting point for the automatic probe, which no longer needs teaching by hand after the tablet has been moved.

Frames: 4 x 4 matrices named ``a_in_b`` take coordinates in a to coordinates in b.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import cv2
import numpy as np

from tablet_tap.core.kinematics import PenArm
from tablet_tap.core.kinematics import pose
from tablet_tap.core.motion import MotionLimits
from tablet_tap.core.motion import TapPlanner
from tablet_tap.core.pinhole import CameraModel
from tablet_tap.core.screen_map import ScreenMap
from tablet_tap.errors import CalibrationError
from tablet_tap.errors import Unreachable
from tablet_tap.vision.board import BoardSpec
from tablet_tap.vision.layout import ScreenLayout

MIN_TOUCHES = 3
MAX_TOUCH_RESIDUAL_M = 0.004  # a touched corner further than this from the fitted board: the touches are mixed up
SEED_SHAPE = (1.5, -1.1, 0.0, 1.0, 0.0)  # joints 2..6 of a pose with the gripper pointing down in front of the arm


def planar_pose(
    object_points: np.ndarray, image_points: np.ndarray, model: CameraModel
) -> tuple[np.ndarray, float, float]:
    """Pose of a flat target from its points in the picture: (target_in_camera, RMS reprojection error in pixels, how
    many times worse the second planar solution fits). A flat target always has two candidate poses, tilted opposite
    ways; when the second fits nearly as well (ratio near 1) the tilt cannot be trusted."""
    obj = np.asarray(object_points, np.float64).reshape(-1, 3)
    img = np.asarray(image_points, np.float64).reshape(-1, 2)
    count, rvecs, tvecs, errors = cv2.solvePnPGeneric(obj, img, model.matrix, model.distortion, flags=cv2.SOLVEPNP_IPPE)
    if count < 1:
        raise CalibrationError("由画面里的点算不出平面目标的位姿。")
    errors = np.asarray(errors, np.float64).ravel()
    best = int(np.argmin(errors))
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, model.matrix, model.distortion, rvecs[best], tvecs[best])
    projected, _ = cv2.projectPoints(obj, rvec, tvec, model.matrix, model.distortion)
    rms = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - img) ** 2, axis=1))))
    ratio = float(errors[1 - best] / max(errors[best], 1e-9)) if count > 1 else float("inf")
    return pose(cv2.Rodrigues(rvec)[0], np.asarray(tvec, np.float64).ravel()), rms, ratio


def board_in_camera(spec: BoardSpec, grid: np.ndarray, model: CameraModel) -> tuple[np.ndarray, float, float]:
    """``planar_pose`` of a board whose inner corners are ``grid`` (rows, cols, 2) in the picture."""
    if grid.shape[:2] != (spec.rows, spec.cols):
        raise CalibrationError(
            f"角点阵列是 {grid.shape[1]} × {grid.shape[0]}，和标定板规格 {spec.cols} × {spec.rows} 不一致。"
        )
    return planar_pose(spec.corners(), grid.reshape(-1, 2), model)


def rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The rotation and translation (no scale) that best take ``source`` (K, 3) onto ``target`` (K, 3), and each
    point's remaining distance (m). Needs three points that are not on one line."""
    src, dst = np.asarray(source, np.float64).reshape(-1, 3), np.asarray(target, np.float64).reshape(-1, 3)
    if len(src) != len(dst) or len(src) < MIN_TOUCHES:
        raise CalibrationError(f"需要至少 {MIN_TOUCHES} 个点，现在有 {len(src)} 个。")
    src_c, dst_c = src - src.mean(axis=0), dst - dst.mean(axis=0)
    if np.linalg.matrix_rank(src_c, tol=1e-4) < 2:
        raise CalibrationError("这些点几乎在一条直线上，定不出位姿；请点分散在板四周的角点。")
    u, _, vt = np.linalg.svd(src_c.T @ dst_c)
    flip = np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))])
    rot = vt.T @ flip @ u.T
    fitted = pose(rot, dst.mean(axis=0) - rot @ src.mean(axis=0))
    return fitted, np.linalg.norm(src @ rot.T + fitted[:3, 3] - dst, axis=1)


@dataclasses.dataclass(frozen=True)
class BoardTouch:
    """The pen tip on the board, right above the inner corner (row, col)."""

    row: int
    col: int
    tip: np.ndarray  # (4, 4) pen-tip pose in the arm's base frame at that moment


@dataclasses.dataclass(frozen=True)
class HandEye:
    camera_in_base: np.ndarray  # (4, 4)
    board_in_base: np.ndarray
    board_in_camera: np.ndarray
    reprojection_px: float  # how well the board's pose explains its corners in the picture
    tilt_ratio: float  # see planar_pose: below about 2 the board's tilt in the camera frame is doubtful
    touch_residual_mm: list[float]  # each touched corner's distance from where the fitted board puts it
    span_error_mm: float  # worst difference between a distance the arm measured between two touches and the board's

    def to_json(self) -> dict[str, Any]:
        return {name: (value.tolist() if isinstance(value, np.ndarray) else value)
                for name, value in dataclasses.asdict(self).items()}  # fmt: skip


def solve_hand_eye(spec: BoardSpec, grid: np.ndarray, model: CameraModel, touches: list[BoardTouch]) -> HandEye:
    """The camera's pose in the arm's base frame. Raises ``CalibrationError`` when the touches do not fit the board or
    were made going round it the wrong way."""
    in_camera, rms, ratio = board_in_camera(spec, grid, model)
    on_board = np.array([spec.corner(t.row, t.col, on_surface=True) for t in touches])
    tips = np.array([t.tip[:3, 3] for t in touches])
    in_base, residual = rigid_fit(on_board, tips)
    # The board's z axis points into the board (away from the camera), so the pen, which points at the board, must
    # point along it. Touches made in mirrored order fit just as well with the board turned upside down.
    pen_axis = np.mean([t.tip[:3, 2] for t in touches], axis=0)
    if in_base[:3, 2] @ pen_axis <= 0:
        raise CalibrationError(
            "按这些触点拟合出来的标定板是底面朝上的：点的顺序和图上的编号方向相反（或者编号对错了角）。请按图上 1→2→3→4 的顺序重新点。"
        )
    if residual.max() > MAX_TOUCH_RESIDUAL_M:
        raise CalibrationError(
            f"触点和标定板对不上：最大残差 {residual.max() * 1e3:.1f} mm（上限 {MAX_TOUCH_RESIDUAL_M * 1e3:.0f} mm）。"
            "检查格距（--square_mm）是否填对、每个点是否点在了对应编号的角上。"
        )
    pairs = [(i, j) for i in range(len(touches)) for j in range(i + 1, len(touches))]
    span = max(abs(np.linalg.norm(tips[i] - tips[j]) - np.linalg.norm(on_board[i] - on_board[j])) for i, j in pairs)
    return HandEye(
        camera_in_base=in_base @ np.linalg.inv(in_camera),
        board_in_base=in_base,
        board_in_camera=in_camera,
        reprojection_px=rms,
        tilt_ratio=ratio,
        touch_residual_mm=(residual * 1e3).round(3).tolist(),
        span_error_mm=float(span * 1e3),
    )


def uncertainty(
    spec: BoardSpec,
    grid: np.ndarray,
    model: CameraModel,
    touches: list[BoardTouch],
    *,
    touch_sigma_m: float = 0.001,
    corner_sigma_px: float = 0.1,
    outside_m: float = 0.15,
    samples: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """How far the result moves when the inputs are jiggled by what a person and a camera can resolve: touches by
    ``touch_sigma_m`` (a 6 mm rubber tip set on a corner by eye), corners in the picture by ``corner_sigma_px``.

    Reported for points the camera would see: the board's centre, and points ``outside_m`` beyond the board's edges in
    its plane (where a tablet would lie). Values are standard deviations: of the position in the arm's base frame
    (mm, the worst direction) and of the board plane's normal (degrees).
    """
    rng = np.random.default_rng(seed)
    nominal = solve_hand_eye(spec, grid, model, touches)
    width, height = (spec.cols - 1) * spec.pitch_m, (spec.rows - 1) * spec.pitch_m
    on_board = np.array(
        [[width / 2, height / 2, 0], [-outside_m, height / 2, 0], [width + outside_m, height / 2, 0],
         [width / 2, -outside_m, 0], [width / 2, height + outside_m, 0]]
    )  # fmt: skip
    seen = on_board @ nominal.board_in_camera[:3, :3].T + nominal.board_in_camera[:3, 3]  # fixed points, camera frame
    positions, normals = [], []
    for _ in range(samples):
        jiggled = [dataclasses.replace(t, tip=pose(t.tip[:3, :3], t.tip[:3, 3] + rng.normal(0, touch_sigma_m, 3)))
                   for t in touches]  # fmt: skip
        try:
            result = solve_hand_eye(spec, grid + rng.normal(0, corner_sigma_px, grid.shape), model, jiggled)
        except CalibrationError:
            continue
        positions.append(seen @ result.camera_in_base[:3, :3].T + result.camera_in_base[:3, 3])
        normals.append(result.board_in_base[:3, 2])
    if len(positions) < samples // 2:
        raise CalibrationError("加上人眼和相机的分辨误差后，这组触点有一半以上的情况拟合不出来：点得太少或太集中。")
    spread = np.std(np.asarray(positions), axis=0)  # (5 points, 3 axes)
    mean_normal = np.mean(normals, axis=0)
    angles = np.degrees(np.arccos(np.clip(np.asarray(normals) @ (mean_normal / np.linalg.norm(mean_normal)), -1, 1)))
    return {
        "centre_mm": float(np.linalg.norm(spread[0]) * 1e3),
        "outside_mm": float(np.linalg.norm(spread[1:], axis=1).max() * 1e3),
        "outside_m": outside_m,
        "normal_deg": float(np.sqrt(np.mean(angles**2))),
        "touch_sigma_mm": touch_sigma_m * 1e3,
        "corner_sigma_px": corner_sigma_px,
    }


def tip_length_check(
    pen: PenArm, spec: BoardSpec, touches: list[tuple[int, int, np.ndarray]]
) -> dict[str, float] | None:
    """The pen-tip length that makes the touches fit the board best, when the pen's direction varied enough between
    them to tell (a wrong length then shows as a residual). ``touches``: (row, col, joints). None: all touches were
    made with nearly the same pen direction, which hides the length."""
    joints = np.array([q for _, _, q in touches])
    axes = pen.fk(joints)[:, :3, 2]
    spread = np.degrees(np.arccos(np.clip(axes @ axes.mean(axis=0) / np.linalg.norm(axes.mean(axis=0)), -1, 1))).max()
    if spread < 8.0 or len(touches) < 4:
        return None
    on_board = np.array([spec.corner(r, c, on_surface=True) for r, c, _ in touches])

    def rms(length: float) -> float:
        tips = dataclasses.replace(pen, tip_len=length).fk(joints)[:, :3, 3]
        return float(np.sqrt(np.mean(rigid_fit(on_board, tips)[1] ** 2)))

    lengths = np.arange(pen.tip_len - 0.03, pen.tip_len + 0.0301, 0.0005)
    errors = [rms(float(length)) for length in lengths]
    best = int(np.argmin(errors))
    return {"tip_len_mm": float(lengths[best] * 1e3), "rms_mm": errors[best] * 1e3, "rms_now_mm": rms(pen.tip_len) * 1e3,
            "direction_spread_deg": float(spread)}  # fmt: skip


def describe_camera(camera_in_base: np.ndarray) -> str:
    """Where the camera is, in words an operator can check against the bench."""
    x, y, z = camera_in_base[:3, 3]
    looking = camera_in_base[:3, 2]  # the optical axis
    down = np.degrees(np.arccos(np.clip(-looking[2], -1, 1)))
    side = "左" if y >= 0 else "右"
    ahead = "前方" if x >= 0 else "后方"
    return (f"相机在拿笔手臂基座{ahead} {abs(x) * 100:.0f} cm、{side}侧 {abs(y) * 100:.0f} cm、上方 {z * 100:.0f} cm，"
            f"光轴偏离竖直向下 {down:.0f}°")  # fmt: skip


# ---- the screen from the camera -----------------------------------------------------------------------------------
def screen_from_camera(
    markers: dict[int, np.ndarray],
    layout: ScreenLayout,
    screen_size_m: tuple[float, float],
    model: CameraModel,
    camera_in_base: np.ndarray,
) -> tuple[ScreenMap, float, float]:
    """Where the screen is, from the calibration page's markers in one camera frame: (the screen, RMS reprojection
    error in pixels, tilt ratio as in ``planar_pose``). ``markers``: id -> (4, 2) corners in the picture, as
    ``vision.rectifier.detect_markers`` gives them; ``screen_size_m``: width and height of the lit display."""
    width, height = screen_size_m
    known = sorted(set(markers) & set(layout.markers))
    if len(known) < 2:
        raise CalibrationError(f"只看到 {len(known)} 个标定码，至少要 2 个才能估计屏幕的位姿。")
    on_screen, in_picture = [], []
    for marker_id in known:
        u0, v0, u1, v1 = layout.markers[marker_id]
        on_screen += [(u0 * width, v0 * height, 0.0), (u1 * width, v0 * height, 0.0),
                      (u1 * width, v1 * height, 0.0), (u0 * width, v1 * height, 0.0)]  # fmt: skip
        in_picture.append(np.asarray(markers[marker_id], np.float64).reshape(4, 2))
    screen_in_camera, rms, ratio = planar_pose(np.array(on_screen), np.concatenate(in_picture), model)
    screen_in_base = camera_in_base @ screen_in_camera
    origin, rot = screen_in_base[:3, 3], screen_in_base[:3, :3]
    right, down = rot[:, 0] * width, rot[:, 1] * height
    # Hold the pen so that its x axis points away from the arm along the screen (the wrist then stays mid-range).
    centre = origin + 0.5 * right + 0.5 * down
    normal = np.cross(down, right) / np.linalg.norm(np.cross(down, right))
    outward = np.array([centre[0], centre[1], 0.0])
    tool_x = outward - (outward @ normal) * normal
    if np.linalg.norm(tool_x) < 1e-6:
        raise CalibrationError("屏幕中心几乎在机械臂基座正上方，定不出持笔的朝向。")
    return ScreenMap(origin, right, down, tool_x / np.linalg.norm(tool_x)), rms, ratio


def seed_joints(pen: PenArm, screen: ScreenMap, limits: MotionLimits | None = None) -> np.ndarray:
    """Joints that hover the pen over the screen's centre, found from a pose with the gripper pointing down in the
    centre's direction. ``Unreachable`` when the arm cannot hover there."""
    planner = TapPlanner(pen, screen, limits)
    centre = screen.point(0.5, 0.5)
    seed = np.array([np.arctan2(centre[1], centre[0]), *SEED_SHAPE])
    try:
        return planner.hover_q(0.5, 0.5, seed)
    except Unreachable as exc:
        raise Unreachable(f"屏幕中心上方的悬停位姿够不到（{exc}）；把平板挪到拿笔手臂正前方 24–36 cm 处。") from exc
