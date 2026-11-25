"""Calibrate the wrist camera as it is mounted now (the kit's 20-degree mount) on the chessboard lying on the table.

    bin/py scripts/calibrate_wrist_now.py plan  [--top <picture.png>]   # no arm: where is the tablet, which poses
    bin/py scripts/calibrate_wrist_now.py sim   [--top <picture.png>]   # the whole run on a software arm, made-up errors
    bin/py scripts/calibrate_wrist_now.py run --attended                # moves the LEFT arm slowly; someone at the e-stop

Where board and tablet lie comes from the top camera; every view shows the whole board past pen and holder, and no leg
comes near the board, the tablet, the table, the post or the other arm's side.
The result goes to calib/wrist.json (the previous one is kept beside it); nothing else is changed.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import os
import pathlib
import signal
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tablet_tap import config  # noqa: E402
from tablet_tap.calibration import hand_eye  # noqa: E402
from tablet_tap.calibration.eye_in_hand import View  # noqa: E402
from tablet_tap.calibration.eye_in_hand import mount_error  # noqa: E402
from tablet_tap.calibration.eye_in_hand import solve_eye_in_hand  # noqa: E402
from tablet_tap.calibration.wrist_procedures import still_picture  # noqa: E402
from tablet_tap.core.kinematics import pose  # noqa: E402
from tablet_tap.core.look import CALIBRATION_RULES  # noqa: E402
from tablet_tap.core.look import MOUNTS  # noqa: E402
from tablet_tap.core.look import LookPlanner  # noqa: E402
from tablet_tap.core.look import WristCamera  # noqa: E402
from tablet_tap.core.look import camera_in_flange  # noqa: E402
from tablet_tap.core.look import flange_pose  # noqa: E402
from tablet_tap.core.motion import joint_move  # noqa: E402
from tablet_tap.core.pinhole import CameraModel  # noqa: E402
from tablet_tap.core.screen_map import ScreenMap  # noqa: E402
from tablet_tap.errors import PathError  # noqa: E402
from tablet_tap.robot.arm import FakeArm  # noqa: E402
from tablet_tap.robot.executor import run_segments  # noqa: E402
from tablet_tap.robot.wrist import NOMINAL_MODEL  # noqa: E402
from tablet_tap.robot.wrist import WristStore  # noqa: E402
from tablet_tap.vision import layout as screen_layout  # noqa: E402
from tablet_tap.vision.camera import camera_model  # noqa: E402
from tablet_tap.vision.camera import open_camera  # noqa: E402
from tablet_tap.vision.rectifier import detect_markers  # noqa: E402

SPEED = 0.20  # rad/s, average of the fastest joint
TIP_M = 0.05  # pen and gripper stay this far above the glass
LAYOUT = screen_layout.default_layout(1.6)
STOP = {"now": False}


def screen_from_top(setup, picture):
    found = {k: v for k, v in detect_markers(picture).items() if k in LAYOUT.markers}
    if len(found) < 4:
        raise SystemExit(f"顶部相机只看到 {len(found)} 个标定码；平板要显示标定页、别被挡住。")
    screen, rms, ratio = hand_eye.screen_from_camera(found, LAYOUT, tuple(setup.screen_size_m), setup.camera_model, np.asarray(setup.camera_pose))
    lean = np.degrees(np.arccos(screen.normal[2]))
    print(f"顶部相机看到 {len(found)} 个码（误差 {rms:.1f} px，倾角判别比 {ratio:.1f}）：平板中心 {np.round(screen.point(0.5, 0.5) * 100, 1).tolist()} cm，倾斜 {lean:.1f}°")
    return screen


BOARD_CELLS = (12, 9)  # squares of the bench's glass board
PITCH_M = 0.015
COLS, ROWS = BOARD_CELLS[0] - 1, BOARD_CELLS[1] - 1
BOARD_M = ((COLS - 1) * PITCH_M, (ROWS - 1) * PITCH_M)
CELLS = {r * (COLS - 1) + c: (r, c) for r in range(ROWS - 1) for c in range(COLS - 1) if (r + c) % 2 == 0}
BOARD_LAYOUT = screen_layout.ScreenLayout(
    aspect=BOARD_M[0] / BOARD_M[1],
    markers={k: (c / (COLS - 1), r / (ROWS - 1), (c + 1) / (COLS - 1), (r + 1) / (ROWS - 1)) for k, (r, c) in CELLS.items()},
)


def find_board(picture, near=None):
    """(ROWS, COLS, 2) inner corners, or None. ``near``: where they should be about; it fixes which end is which."""
    gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
    x0, y0, x1, y1 = 0, 0, gray.shape[1], gray.shape[0]
    if near is not None:
        lo, hi = near.reshape(-1, 2).min(0) - 160, near.reshape(-1, 2).max(0) + 160
        x0, y0 = max(int(lo[0]), 0), max(int(lo[1]), 0)
        x1, y1 = min(int(hi[0]), gray.shape[1]), min(int(hi[1]), gray.shape[0])
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    grid = None
    for scale in (1.0, 2.0, 1.5, 0.6):
        patch = cv2.resize(gray[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
        ok, corners = cv2.findChessboardCornersSB(patch, (COLS, ROWS), flags)
        if not ok:
            ok, corners = cv2.findChessboardCorners(patch, (COLS, ROWS), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            points = (corners.reshape(-1, 2) + 0.5) / scale - 0.5 + (x0, y0)
            points = cv2.cornerSubPix(gray, points.astype(np.float32).reshape(-1, 1, 2), (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 0.005))
            grid = points.reshape(ROWS, COLS, 2).astype(np.float64)
            break
    if grid is None:
        return None
    ex, ey = grid[0, 1] - grid[0, 0], grid[1, 0] - grid[0, 0]
    if ex[0] * ey[1] - ex[1] * ey[0] < 0:
        grid = grid[::-1]
    if near is not None:
        flipped = grid[::-1, ::-1]
        if np.linalg.norm(flipped - near, axis=2).mean() < np.linalg.norm(grid - near, axis=2).mean():
            grid = flipped
    return grid


def board_from_top(setup, picture):
    grid = find_board(picture)
    if grid is None:  # large and oblique: look at each half of the picture enlarged
        h, w = picture.shape[:2]
        for x in (0, w // 4, w // 2):
            part = picture[: int(0.6 * h), x : x + w // 2]
            found = find_board(cv2.resize(part, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
            if found is not None:
                grid = (found + 0.5) / 2 - 0.5 + (x, 0)
                break
    if grid is None:
        raise SystemExit("顶部相机没有认出标定板（12 × 9 格）：看看有没有被手臂挡住或反光。")
    points = np.array([(c * PITCH_M, r * PITCH_M, 0.0) for r in range(ROWS) for c in range(COLS)])
    in_camera, rms, ratio = hand_eye.planar_pose(points, grid.reshape(-1, 2), setup.camera_model)
    at = np.asarray(setup.camera_pose) @ in_camera
    screen = ScreenMap(at[:3, 3], at[:3, 0] * BOARD_M[0], at[:3, 1] * BOARD_M[1], np.array([1.0, 0, 0]))
    centre = screen.point(0.5, 0.5)
    print(f"标定板：误差 {rms:.2f} px，倾角判别比 {ratio:.1f}，中心 {np.round(centre * 100, 1).tolist()} cm，和水平差 {np.degrees(np.arccos(abs(screen.normal[2]))):.1f}°")
    return screen


class Careful(LookPlanner):
    """A planner whose paths also keep off the tablet and its stand."""

    others: list = []

    def path_problem(self, screen, q_start, q_goal, *, tip_m=0.03):
        for other in [screen, *self.others]:
            problem = super().path_problem(other, q_start, q_goal, tip_m=tip_m)
            if problem:
                return problem
        return None


# what pen, holder and fingers cover in each mount's picture, from pictures of 2026-11-23 (the same at every pose)
COVERED = {
    "kit": [np.float32([[925, 575], [1020, 575], [1235, 725], [1440, 1080], [620, 1080], [770, 860], [805, 725]])],
    "sim": [np.float32([[990, 555], [1070, 555], [1080, 700], [990, 700]]), np.float32([[930, 665], [1090, 665], [1250, 1080], [920, 1080]]),
            np.float32([[890, 850], [1240, 850], [1390, 1080], [815, 1080]])],
}  # fmt: skip
HOLDER_AT = {"kit": (780, 1260, 660, 1060), "sim": (860, 1300, 680, 1060)}  # x0, x1, y0, y1 of the pen holder in the picture
START = {"sim": (0.129, 0.007, 0.043, 34.5)}  # where a single picture put it, for planning (the gate is the drawing)


def pool_of(planner, screen, park, want=400):
    """Safe poses that see the whole board, from all over the arm's reach."""
    rng, rules = np.random.default_rng(7), planner.rules
    quad = np.array([screen.point(u, v) for u, v in OUTLINE])
    size = np.array([planner.camera.model.width, planner.camera.model.height])
    centre = screen.point(0.5, 0.5)
    turn = np.arctan2(centre[1], centre[0])
    pool = []
    for _ in range(400000):
        if len(pool) >= want:
            break
        q = np.array([turn + rng.uniform(-0.7, 0.7), rng.uniform(0.2, 2.2), rng.uniform(-2.0, -0.2), rng.uniform(-1.2, 1.2), rng.uniform(-0.3, 1.2), rng.uniform(-1.5, 1.5)])
        limit = float(np.degrees(np.min(np.minimum(q - planner.pen.lower, planner.pen.upper - q))))
        if limit < rules.limit_deg:
            continue
        cam = flange_pose(planner.pen, q) @ planner.camera.in_flange
        px, depth = planner.camera.pixels(quad, cam)
        if depth.min() < 0.15 or not np.all((px > 40) & (px < size - 41)) or planner.hides(px, 25):
            continue
        height = float((planner.pen.fk(q)[:3, 3] - screen.origin) @ screen.normal)
        gaps = planner.shape.clearance(q)
        if height < TIP_M + 0.03 or gaps["table"] < 0.06 or gaps["post"] < 0.06 or gaps["camera"] < 0.05 or gaps["other_arm"] < 0.02:
            continue
        look = planner._look(screen, q, gaps, height, limit)  # noqa: SLF001
        if look.off_normal_deg > 45 or look.distance_m > 0.55 or planner.path_problem(screen, park, q) is not None:
            continue
        pool.append(look)
    return pool


def choose(planner, screen, park, pool, n):
    """``n`` poses as different as the pool has them, in an order the arm can go through."""
    feature = []
    for look in pool:
        away = look.camera_in_base[:3, 3] - screen.point(0.5, 0.5)
        away = away / np.linalg.norm(away)
        down = look.camera_in_base[:3, 1]
        roll = np.arctan2(down[1], down[0])
        feature.append([1.5 * away[0], 1.5 * away[1], 0.6 * np.cos(roll), 0.6 * np.sin(roll), 2.5 * look.distance_m])
    feature = np.array(feature)
    chosen = [int(np.argmax(np.linalg.norm(feature - feature.mean(0), axis=1)))]
    while len(chosen) < n:
        gap = np.min(np.linalg.norm(feature[:, None] - feature[chosen], axis=2), axis=1)
        chosen.append(int(np.argmax(gap)))
    rest, route, here = [pool[k] for k in chosen], [], park
    while rest:
        rest.sort(key=lambda look: float(np.max(np.abs(look.q - here))))  # noqa: B023
        step = next((look for look in rest if planner.path_problem(screen, here, look.q) is None), None)
        if step is None:
            step = rest[0]
            route.append(dataclasses.replace(step, clearance=step.clearance | {"via_park": 1.0}))
        else:
            route.append(step)
        rest.remove(step)
        here = step.q
    return route


OUTLINE = [(-0.12, -0.16), (1.12, -0.16), (1.12, 1.16), (-0.12, 1.16)]  # the board's squares reach one pitch past the inner corners


def plan(setup, screen, others, mount, n, seed=0):
    pen, park = setup.pen_arm(), np.asarray(setup.park_q)
    camera = WristCamera(NOMINAL_MODEL, camera_in_flange(*START.get(mount, MOUNTS[mount])))
    rules = dataclasses.replace(CALIBRATION_RULES, tip_m=TIP_M, table_m=0.04, post_m=0.05, camera_m=0.04, limit_deg=4.0, margin_px=30.0, pen_px=20.0)
    planner = Careful(pen, camera, rules, seed)
    planner.others = others
    if mount in COVERED:
        planner._hidden = [cv2.convexHull(hull) for hull in COVERED[mount]]  # noqa: SLF001
    pool = pool_of(planner, screen, park)
    print(f"可用位姿 {len(pool)} 个")
    if len(pool) < n:
        raise SystemExit("能看到整块标定板的安全位姿太少；把板挪到左臂正前方 30–35 cm 再试。")
    looks = choose(planner, screen, park, pool, n)
    here = park
    for k, look in enumerate(looks, 1):
        start = park if look.clearance.get("via_park") else here
        problem = planner.path_problem(screen, start, look.q) or planner.path_problem(screen, park, look.q)
        if problem:
            raise SystemExit(f"第 {k} 段路离{problem}，没有采用。")
        here = look.q
        print(f"  {k:2d}: 距板 {look.distance_m * 100:.0f} cm，偏法线 {look.off_normal_deg:.0f}°，笔尖高 {look.tip_height_m * 1e3:.0f} mm，离限位 {look.limit_deg:.0f}°"
              + ("，经停靠位" if look.clearance.get("via_park") else "") + f"，关节 {np.degrees(look.q).round().astype(int).tolist()}")  # fmt: skip
    for look in looks[:REPEATS]:  # the first poses once more, by the park pose, from which every pose was checked
        looks.append(dataclasses.replace(look, clearance=look.clearance | {"via_park": 1.0}))
        here = look.q
    if planner.path_problem(screen, here, park):
        raise SystemExit("最后回停靠位的路不通。")
    return planner, looks


def board_view(planner, screen, q, picture):
    """The board's cells in one picture, numbered by where the board should be in it."""
    cam = flange_pose(planner.pen, q) @ planner.camera.in_flange
    near, _ = planner.camera.pixels(np.array([[screen.point(c / (COLS - 1), r / (ROWS - 1)) for c in range(COLS)] for r in range(ROWS)]), cam)
    grid = find_board(picture, near)
    if grid is None:
        return {}, None
    return {k: np.array([grid[r, c], grid[r, c + 1], grid[r + 1, c + 1], grid[r + 1, c]]) for k, (r, c) in CELLS.items()}, float(np.linalg.norm(grid - near, axis=2).mean())


def own_intrinsics(views, model):
    """The camera's intrinsics with lens distortion from the board pictures themselves (the factory says there is none),
    kept only when they describe the pictures clearly better: (model, RMS px before, after)."""
    flat = np.array([c for k in sorted(CELLS) for u0, v0, u1, v1 in (BOARD_LAYOUT.markers[k],) for c in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))])
    flat = np.column_stack([flat * BOARD_M, np.zeros(len(flat))]).astype(np.float32)
    seen = [np.concatenate([v.markers[k] for k in sorted(CELLS)]).astype(np.float32).reshape(-1, 1, 2) for v in views if len(v.markers) >= 4]
    before = float(np.sqrt(np.mean([hand_eye.planar_pose(flat, px, model)[1] ** 2 for px in seen])))
    if len(seen) < 10:
        return model, before, before
    after, k, dist, _, _ = cv2.calibrateCamera([flat] * len(seen), seen, (model.width, model.height), model.matrix.copy(), np.zeros(5), flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    own = CameraModel(model.width, model.height, float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]), tuple(float(x) for x in dist.ravel()[:5]))
    near = abs(own.fx / model.fx - 1) < 0.03 and abs(own.fy / model.fy - 1) < 0.03 and np.hypot(own.cx - model.cx, own.cy - model.cy) < 30
    return (own if near and after < 0.6 * before else model), before, float(after)


LOOSE_DEG = 0.6
REPEATS = 3
FINE_MM, USABLE_MM = 2.0, 6.0  # held-out error of the board: below the first the result is good, above the second useless


def holder_turns(pictures, at):
    """How far the pen holder turns in the picture from pose to pose (degrees, against the median). The holder is clamped
    in the gripper: when it turns in the picture, the camera or its mount is loose."""
    x0, x1, y0, y1 = at

    def edges(picture):
        g = cv2.GaussianBlur(cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY), (0, 0), 2.0).astype(np.float32)
        m = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
        return (m / (m + 20))[y0:y1, x0:x1].astype(np.float32)

    first, turns = edges(pictures[0]), []
    for picture in pictures:
        try:
            score, warp = cv2.findTransformECC(first, edges(picture), np.float32([[1, 0, 0], [0, 1, 0]]), cv2.MOTION_EUCLIDEAN,
                                               (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-6), None, 5)  # fmt: skip
            turns.append(float(np.degrees(np.arctan2(warp[1, 0], warp[0, 0]))) if score > 0.5 else float("nan"))
        except cv2.error:
            turns.append(float("nan"))
    return np.array(turns) - np.nanmedian(turns)


def with_zeros(setup, views, model, start, prior_deg=0.7):
    """The same fit with the zeros of joints 2-5 free (joint 1 is the board's place, joint 6 the camera's turn):
    (camera on the flange, zeros in degrees, RMS px, RMS px of views predicted by fits that did not use them)."""
    pen = setup.pen_arm()
    usable = [v for v in views if len(v.markers) >= 4]
    corners = np.array([[BOARD_LAYOUT.markers[k][a] for a in (0, 1, 2, 1, 2, 3, 0, 3)] for k in sorted(CELLS)]).reshape(-1, 4, 2).reshape(-1, 2)
    flat = np.column_stack([corners * BOARD_M, np.zeros(len(corners))])
    seen = [np.concatenate([v.markers[k] for k in sorted(CELLS)]) for v in usable]

    def unpack(x):
        turn = lambda r, t: pose(cv2.Rodrigues(np.asarray(r))[0], np.asarray(t))  # noqa: E731
        dq = np.zeros(6)
        dq[1:5] = x[12:16]
        return start.camera_in_flange @ turn(x[:3], x[3:6]), start.screen_in_base @ turn(x[6:9], x[9:12]), dq

    def errors(x, rows):
        cam, board, dq = unpack(x)
        out = []
        for k in rows:
            at = np.linalg.inv(flange_pose(pen, usable[k].q + dq) @ cam) @ board
            px, _ = cv2.projectPoints(flat, cv2.Rodrigues(at[:3, :3])[0], at[:3, 3], model.matrix, model.distortion)
            out.append(px.reshape(-1, 2) - seen[k])
        return out

    def fit(rows):
        def residual(x):
            e = np.concatenate([d.ravel() for d in errors(x, rows)]) / np.sqrt(len(flat))
            return np.concatenate([e, x[12:16] / np.radians(prior_deg)])

        return least_squares(residual, np.zeros(16), x_scale=np.array([0.02] * 3 + [0.01] * 3 + [0.02] * 3 + [0.01] * 3 + [0.01] * 4), loss="soft_l1", f_scale=3.0).x

    every = list(range(len(usable)))
    x = fit(every)
    rms = lambda es: float(np.sqrt(np.mean([np.mean(np.sum(d**2, 1)) for d in es])))  # noqa: E731
    held = []
    for part in range(4):
        held += errors(fit([k for k in every if k % 4 != part]), [k for k in every if k % 4 == part])
    cam, board, dq = unpack(x)
    return cam, np.degrees(dq), rms(errors(x, every)), rms(held), board


def board_pose(view, model):
    flat = np.array([c for k in sorted(CELLS) for u0, v0, u1, v1 in (BOARD_LAYOUT.markers[k],) for c in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))])
    return hand_eye.planar_pose(np.column_stack([flat * BOARD_M, np.zeros(len(flat))]), np.concatenate([view.markers[k] for k in sorted(CELLS)]), model)[0]


def held_out(setup, views, model, drawn):
    """How far views place the board from where fits without them place it: (mm, degrees), one per view."""
    pen, mm, deg = setup.pen_arm(), [], []
    centre = np.array([BOARD_M[0] / 2, BOARD_M[1] / 2, 0, 1])
    for part in range(4):
        f = solve_eye_in_hand([v for i, v in enumerate(views) if i % 4 != part], BOARD_LAYOUT, BOARD_M, model, pen, drawn, fit_size=False)
        for v in views[part::4]:
            at = flange_pose(pen, v.q) @ f.camera_in_flange @ board_pose(v, model)
            mm.append(float(np.linalg.norm((at @ centre - f.screen_in_base @ centre)[:3]) * 1e3))
            deg.append(float(np.degrees(np.linalg.norm(cv2.Rodrigues(at[:3, :3] @ f.screen_in_base[:3, :3].T)[0]))))
    return np.array(mm), np.array(deg)


def solve(setup, views, screen, model, mount, out):
    drawn = camera_in_flange(*MOUNTS[mount])
    (out / "views.json").write_text(json.dumps([{"q": v.q.tolist(), "markers": {str(k): m.tolist() for k, m in v.markers.items()}} for v in views]))
    if any(v.markers for v in views):
        model, before, after = own_intrinsics(views, model)
        print(f"内参：出厂值的单张误差 {before:.2f} px，自己拟合后 {after:.2f} px，畸变 {np.round(model.dist, 3).tolist()}")
    found = solve_eye_in_hand(views, BOARD_LAYOUT, BOARD_M, model, setup.pen_arm(), drawn, fit_size=False)
    free = solve_eye_in_hand(views, BOARD_LAYOUT, BOARD_M, model, setup.pen_arm(), drawn, fit_size=True)
    off = mount_error(found.camera_in_flange, drawn)
    seen = found.screen_map()
    gaps = [float(np.linalg.norm(seen.point(u, v) - screen.point(u, v)) * 1e3) for u, v in ((0, 0), (1, 0), (1, 1), (0, 1))]
    record = {"off_drawing": off, "top_camera_gap_mm": gaps, "free_size_scale": free.screen_size_m[0] / BOARD_M[0], "free_size_rms_px": free.rms_px,
              "free_size_shift": mount_error(free.camera_in_flange, found.camera_in_flange)}  # fmt: skip
    (out / "result.json").write_text(json.dumps(found.to_json() | record, indent=1))
    print(f"拟合残差 {found.rms_px:.2f} px，没参加拟合的位姿 {found.held_out_px:.2f} px，去掉 {len(found.left_out)} 个；"
          f"不确定度 {found.position_sigma_mm:.1f} mm / {found.angle_sigma_deg:.2f}°；和图纸差 {off[0]:.1f} mm、{off[1]:.1f}°；"
          f"各位姿残差 {found.view_px}\n"
          f"和顶部相机看到的板四角差 {' / '.join(f'{g:.1f}' for g in gaps)} mm；放开板的尺寸：比例 {record['free_size_scale']:.4f}，残差 {free.rms_px:.2f} px，"
          f"相机只变 {record['free_size_shift'][0]:.1f} mm / {record['free_size_shift'][1]:.2f}°")  # fmt: skip
    cam, dq, rms, held, board = with_zeros(setup, views, model, found)
    np.savez(out / "with_zeros.npz", camera=cam, zeros_deg=dq, board=board)
    print(f"同时放开 2–5 号关节零位：残差 {rms:.2f} px，没参加拟合的 {held:.2f} px，零位（度）{dq[1:5].round(2).tolist()}，"
          f"相机变 {mount_error(cam, found.camera_in_flange)[0]:.1f} mm / {mount_error(cam, found.camera_in_flange)[1]:.2f}°")  # fmt: skip
    return found, off, model


class SimCamera:
    """The board drawn where the made-up truth puts it."""

    def __init__(self, truth, screen, arm, pen, zero, mount):
        self.truth, self.screen, self.arm, self.pen, self.zero = truth, screen, arm, pen, zero
        cell = 60
        page = np.full(((BOARD_CELLS[1] + 2) * cell, (BOARD_CELLS[0] + 2) * cell, 3), 215, np.uint8)
        for r in range(BOARD_CELLS[1]):
            for c in range(BOARD_CELLS[0]):
                if (r + c) % 2 == 0:
                    page[(r + 1) * cell : (r + 2) * cell, (c + 1) * cell : (c + 2) * cell] = 25
        self.page, self.cell = page, cell
        self.hulls = [cv2.convexHull(hull) for hull in next((v for k, v in COVERED.items() if k == mount), [])]
        self.rng = np.random.default_rng(1)

    def read(self):
        cam = flange_pose(self.pen, self.arm.read() + self.zero) @ self.truth
        uv = [(-2 / (COLS - 1), -2 / (ROWS - 1)), (1 + 2 / (COLS - 1), -2 / (ROWS - 1)), (1 + 2 / (COLS - 1), 1 + 2 / (ROWS - 1)), (-2 / (COLS - 1), 1 + 2 / (ROWS - 1))]
        corners = np.array([self.screen.point(u, v) for u, v in uv])
        px = NOMINAL_MODEL.project((corners - cam[:3, 3]) @ cam[:3, :3]).astype(np.float32)
        h, w = self.page.shape[:2]
        warp = cv2.getPerspectiveTransform(np.float32([[-0.5, -0.5], [w - 0.5, -0.5], [w - 0.5, h - 0.5], [-0.5, h - 0.5]]), px)
        image = cv2.warpPerspective(self.page, warp, (1920, 1080), flags=cv2.INTER_AREA, borderValue=(170, 160, 120))
        for hull in self.hulls:
            cv2.fillConvexPoly(image, hull.astype(np.int32), (30, 30, 30))
        image = cv2.GaussianBlur(image, (0, 0), 1.2)
        return np.clip(image + self.rng.normal(0, 3, image.shape), 0, 255).astype(np.uint8)

    def close(self):
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("plan", "sim", "run"))
    ap.add_argument("--top", help="顶部相机的照片；不给就现拍")
    ap.add_argument("--mount", default="sim", choices=sorted(MOUNTS))
    ap.add_argument("--views", type=int, default=14)
    ap.add_argument("--attended", action="store_true")
    args = ap.parse_args()
    if args.what == "run" and not args.attended:
        raise SystemExit("要动机械臂：确认有人在急停旁边、手和笔夹都离开手臂之后，加 --attended。")
    setup = config.SetupStore().load()
    if args.top:
        picture = cv2.imread(args.top)
    else:
        camera = open_camera(setup.camera)
        picture = camera.read()
        camera.close()
    screen, tablet = board_from_top(setup, picture), screen_from_top(setup, picture)
    planner, looks = plan(setup, screen, [tablet], args.mount, args.views)
    park = np.asarray(setup.park_q)
    print(f"{len(looks)} 个位姿，笔尖最低离屏 {min(v.tip_height_m for v in looks) * 1e3:.0f} mm，离停靠位最大 {np.degrees(max(np.max(np.abs(v.q - park)) for v in looks)):.0f}°")
    if args.what == "plan":
        return 0
    out = REPO / "runs" / f"wrist_calib_{args.what}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True)
    cv2.imwrite(str(out / "top.png"), picture)
    limits = dataclasses.replace(setup.motion_limits(), joint_move_speed=SPEED)
    if args.what == "sim":
        pen = setup.pen_arm()
        rng = np.random.default_rng(2)
        wrong = pose(cv2.Rodrigues(np.radians([1.2, -0.8, 1.0]))[0], np.array([0.003, -0.002, 0.004]))
        zero = np.radians([0.25, -0.25, 0.3, 0.3, -0.4, 0.2])
        truth = planner.camera.in_flange @ wrong
        shifted = dataclasses.replace(screen, origin=screen.origin + rng.normal(0, 0.004, 3))
        arm, realtime = FakeArm(park), False
        camera, model = SimCamera(truth, shifted, arm, pen, zero, args.mount), NOMINAL_MODEL
    else:
        from tablet_tap.robot.factory import ArmMode  # noqa: PLC0415
        from tablet_tap.robot.factory import open_arm  # noqa: PLC0415

        arm, realtime = open_arm(setup.arm, ArmMode.COMMAND), True
        camera = open_camera(WristStore(REPO / "calib" / "wrist.json").load().camera)
        model = camera_model(camera)
        signal.signal(signal.SIGINT, lambda *_: STOP.update(now=True))

    def move(goal, name):
        print(f"{name} ……", flush=True)
        run_segments(arm, [joint_move(name, arm.read(), np.asarray(goal), limits, settle=True)], hz=limits.hz, should_stop=lambda: STOP["now"], realtime=realtime)

    try:
        here = arm.read()
        problem = planner.path_problem(screen, here, park)
        if problem:
            raise PathError(f"从现在的位置到停靠位的路离{problem}；先用手把手臂移开。")
        move(park, "到停靠位")
        views = []
        for k, look in enumerate(looks, 1):
            if look.clearance.get("via_park"):
                move(park, "经停靠位")
            move(look.q, f"标定位姿 {k}/{len(looks)}")
            q, shot = still_picture(arm.read, camera, settle_s=0.8 if realtime else 0.0)
            cv2.imwrite(str(out / f"view_{k:02d}.png"), shot)
            markers, gap = board_view(planner, screen, q, shot)
            print(f"  {'没认出标定板' if not markers else f'认出标定板，离预想位置 {gap:.0f} px'}", flush=True)
            views.append(View(q, markers))
        move(park, "回停靠位")
    finally:
        camera.close()
    used, turns = list(views), np.zeros(len(views))
    if args.what == "run" and args.mount in HOLDER_AT:
        turns = holder_turns([cv2.imread(str(out / f"view_{k:02d}.png")) for k in range(1, len(views) + 1)], HOLDER_AT[args.mount])
        print(f"笔夹在画面里的转动（度）：{np.round(turns, 1).tolist()}")
        used = [v for v, t in zip(views, turns) if not abs(t) > LOOSE_DEG]
        if len(used) < len(views):
            print(f"⚠ {len(views) - len(used)} 个位姿里相机转了 {LOOSE_DEG}° 以上，没有用")
    used = [v for v in used if v.markers]
    found, off, model = solve(setup, used, screen, model, args.mount, out)
    mm, deg = held_out(setup, used, model, camera_in_flange(*MOUNTS[args.mount]))
    print(f"没参加拟合的位姿：板中心平均差 {mm.mean():.1f} mm，最大 {mm.max():.1f} mm；转角平均 {deg.mean():.2f}°，最大 {deg.max():.2f}°")
    pen, centre = setup.pen_arm(), np.array([BOARD_M[0] / 2, BOARD_M[1] / 2, 0, 1])
    for k in range(REPEATS):  # the same pose at the beginning and at the end
        first, again = views[k], views[len(looks) - REPEATS + k]
        if first.markers and again.markers:
            a, b = (flange_pose(pen, v.q) @ found.camera_in_flange @ board_pose(v, model) for v in (first, again))
            print(f"位姿 {k + 1} 重复：板差 {np.linalg.norm((a @ centre - b @ centre)[:3]) * 1e3:.1f} mm、"
                  f"{np.degrees(np.linalg.norm(cv2.Rodrigues(a[:3, :3] @ b[:3, :3].T)[0])):.2f}°，关节读数差 {np.degrees(np.abs(first.q - again.q)).max():.2f}°")  # fmt: skip
    if args.what == "sim":
        print("离真值", "%.2f mm %.2f°" % mount_error(found.camera_in_flange, truth),
              "；放开零位后", "%.2f mm %.2f°" % mount_error(np.load(out / "with_zeros.npz")["camera"], truth))  # fmt: skip
        return 0
    if off[0] > 30 or off[1] > 8 or mm.mean() > USABLE_MM or len(used) < 12:
        raise SystemExit(f"结果不可用，没有保存（记录在 {out}）。")
    store = WristStore(REPO / "calib" / "wrist.json")
    wrist = store.load()
    wrist.mount, wrist.model, wrist.in_flange = args.mount, model, found.camera_in_flange.tolist()
    wrist.look_q = wrist.look_screen = None
    wrist.records["eye_in_hand"] = {k: v for k, v in found.to_json().items() if k != "camera_in_flange"} | {
        "folder": str(out), "coarse": bool(mm.mean() > FINE_MM), "held_out_mm_mean": float(mm.mean()), "held_out_mm_max": float(mm.max()),
        "held_out_deg_mean": float(deg.mean()), "views_left_out": len(views) - len(used)}  # fmt: skip
    store.save(wrist)
    print(f"已保存 {store.path}，记录在 {out}")
    return 0


if __name__ == "__main__":
    os.environ.pop("DISPLAY", None)
    raise SystemExit(main())
