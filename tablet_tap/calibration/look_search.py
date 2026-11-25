"""Look poses judged by what the wrist camera's picture really shows of the screen.

``core.look`` takes what pen and holder hide from the drawing's boxes. On the bench those boxes cover four times what
the real parts do (2026-11-23, stock mount: 24 % of the picture against 6.5 %), and with them no pose shows the whole
screen. Here what is hidden is a mask of the picture: measured from pictures taken at different poses when there are
such (whatever rides with the camera looks the same in all of them), else the drawing's. And a look need not show all
of the screen: two looks that between them show all of it are as good, the second filling what the first one misses.
"""

from __future__ import annotations

import dataclasses
import itertools

import cv2
import numpy as np
from scipy.spatial import cKDTree

from tablet_tap.core.kinematics import pose
from tablet_tap.core.look import Look
from tablet_tap.core.look import LookPlanner
from tablet_tap.core.look import flange_pose
from tablet_tap.core.screen_map import ScreenMap
from tablet_tap.errors import Unreachable
from tablet_tap.vision.rectifier import ScreenRectifier

GRID = (24, 15)  # the screen as cells: which of them a look shows
STILL = 22  # grey levels a pixel varies by, at most, when it shows something that rides with the camera
MIN_PICTURES = 10
GOOD_SINGLE_PX = 700.0  # one look that shows the screen at least this wide is taken before two looks are
PAIR_POOL = 60


def drawn_mask(planner: LookPlanner) -> np.ndarray:
    """(height, width) 1 where the drawing's boxes of pen and holder are in the picture."""
    model = planner.camera.model
    mask = np.zeros((model.height, model.width), np.uint8)
    for hull in planner._hidden:  # noqa: SLF001 - the planner's own outlines
        cv2.fillConvexPoly(mask, np.round(hull).astype(np.int32), 1)
    return mask


def pen_mask(planner: LookPlanner) -> np.ndarray:
    """The pen alone, from the drawing: a polished rod mirrors what is around it, so pictures do not show it as still."""
    model = planner.camera.model
    mask = np.zeros((model.height, model.width), np.uint8)
    hulls = planner._hidden  # noqa: SLF001
    if hulls:
        cv2.fillConvexPoly(mask, np.round(min(hulls, key=cv2.contourArea)).astype(np.int32), 1)
    return mask


def occluder_from_pictures(pictures: list[np.ndarray], within: np.ndarray) -> np.ndarray | None:
    """What rides with the camera (holder, fingers, mount), from pictures taken at different poses: the part of the
    picture that looks the same in all of them, and only inside ``within`` (where the drawing can have such a part at
    all: a tablecloth of one colour is as still). None with too few pictures."""
    if len(pictures) < MIN_PICTURES:
        return None
    height, width = pictures[0].shape[:2]
    small = (width // 4, height // 4)
    stack = np.stack([cv2.resize(p, small, interpolation=cv2.INTER_AREA).astype(np.float32) for p in pictures])
    spread = (np.percentile(stack, 90, axis=0) - np.percentile(stack, 10, axis=0)).max(axis=2)
    still = (spread < STILL).astype(np.uint8)
    still = cv2.morphologyEx(still, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    still = cv2.morphologyEx(still, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    still &= cv2.resize(cv2.dilate(within, np.ones((41, 41), np.uint8)), small, interpolation=cv2.INTER_NEAREST)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(still)
    kept = np.zeros_like(still)
    for k in range(1, count):
        if stats[k, cv2.CC_STAT_AREA] > 0.003 * still.size:
            kept[labels == k] = 1
    return cv2.resize(kept, (width, height), interpolation=cv2.INTER_NEAREST)


def hide_mask(planner: LookPlanner, measured: np.ndarray | None, grow_px: int | None = None) -> np.ndarray:
    """(height, width) 1 where the picture does not show the scene: the measured part and the pen, or the drawing's
    boxes when nothing was measured; grown by the planner's distance from the pen."""
    mask = drawn_mask(planner) if measured is None else (measured.astype(np.uint8) | pen_mask(planner))
    grow = int(planner.rules.pen_px if grow_px is None else grow_px)
    return cv2.dilate(mask, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)) if grow > 0 else mask


@dataclasses.dataclass(frozen=True)
class Seen:
    look: Look
    cells: np.ndarray  # (GRID[0] * GRID[1],) which cells of the screen the look shows

    @property
    def share(self) -> float:
        return float(self.cells.mean())


def _cells(screen: ScreenMap) -> np.ndarray:
    u, v = np.meshgrid((np.arange(GRID[0]) + 0.5) / GRID[0], (np.arange(GRID[1]) + 0.5) / GRID[1])
    return np.array([screen.point(a, b) for a, b in zip(u.ravel(), v.ravel())])


def seen_cells(planner: LookPlanner, hide: np.ndarray, cells: np.ndarray, camera_pose: np.ndarray) -> np.ndarray:
    """Which of the screen's cells are in the picture and not behind what ``hide`` marks."""
    model, margin = planner.camera.model, planner.rules.margin_px
    seen = (cells - camera_pose[:3, 3]) @ camera_pose[:3, :3]
    ahead = seen[:, 2] > 0.05
    pixel = model.project(np.where(ahead[:, None], seen, 1.0))
    inside = ahead & np.all((pixel > margin) & (pixel < np.array([model.width, model.height]) - 1 - margin), axis=1)
    shown = inside.copy()
    at = np.round(pixel[inside]).astype(int)
    shown[inside] = hide[at[:, 1], at[:, 0]] == 0
    return shown


def find_looks(
    planner: LookPlanner,
    hide: np.ndarray,
    screen: ScreenMap,
    seed_q: np.ndarray,
    park_q: np.ndarray,
    *,
    postures: int = 20000,
    tries: int = 500,
    min_share: float = 0.55,
) -> tuple[Seen, Seen | None]:
    """One look that shows the whole screen, or two that do between them (the first is the one the arm rests at
    between touches). ``Unreachable`` when neither exists."""
    rules, pen, size = planner.rules, planner.pen, screen.size_m
    park, cells = np.asarray(park_q, np.float64), _cells(screen)
    margin = np.radians(rules.limit_deg)
    spread = np.array([0.6, 0.9, 0.9, 1.2, 0.8, 1.8])
    cloud = np.clip(np.asarray(seed_q, np.float64) + planner.rng.uniform(-1, 1, (postures, 6)) * spread, pen.lower + margin, pen.upper - margin)
    held = flange_pose(pen, cloud) @ planner.camera.in_flange
    nearest = cKDTree(np.column_stack([held[:, :3, 3], 0.15 * held[:, :3, 2], 0.05 * held[:, :3, 1]]))
    centre, normal = screen.point(0.5, 0.5), screen.normal
    ex, ey = screen.right / size[0], screen.down / size[1]
    framings = []
    for distance in np.linspace(rules.distance_m[0] + 0.04, min(rules.distance_m[1], 0.42), 7):
        for lean in np.radians(np.arange(0.0, rules.off_normal_deg + 0.1, 8.0)):
            for azimuth in [0.0] if lean == 0 else np.linspace(0, 2 * np.pi, 12, endpoint=False):
                eye = centre + distance * (np.sin(lean) * (np.cos(azimuth) * ex + np.sin(azimuth) * ey) + np.cos(lean) * normal)
                for hint in (ey, -ey):
                    for up, aside in itertools.product((0.0, 0.25, 0.5, -0.25), (0.0, 0.3, -0.3, 0.55, -0.55)):
                        forward = centre + up * size[1] * hint + aside * size[0] * ex - eye
                        forward /= np.linalg.norm(forward)
                        down = hint - (hint @ forward) * forward
                        down /= np.linalg.norm(down)
                        camera_pose = pose(np.column_stack([np.cross(down, forward), down, forward]), eye)
                        shown = seen_cells(planner, hide, cells, camera_pose)
                        if shown.mean() >= min_share:
                            framings.append((shown.mean() / distance, camera_pose, shown))
    framings.sort(key=lambda item: -item[0])
    looks: list[Seen] = []
    for _, camera_pose, shown in framings[:tries]:
        tip = camera_pose @ planner._to_tip  # noqa: SLF001
        start = None
        for k in nearest.query(np.concatenate([camera_pose[:3, 3], 0.15 * camera_pose[:3, 2], 0.05 * camera_pose[:3, 1]]), k=3)[1]:
            q, miss, turn = pen.ik(tip, cloud[k])
            if miss < 1e-3 and turn < 5e-3:
                start = q
                break
        look = planner.feasible(screen, camera_pose, [start]) if start is not None else None
        if look is None or np.max(np.abs(look.q - park)) > np.radians(rules.travel_deg):
            continue
        if planner.path_problem(screen, park, look.q) is None:
            looks.append(Seen(look, shown))

    def score(seen: Seen) -> float:
        travel = float(np.degrees(np.max(np.abs(seen.look.q - park))))
        return seen.look.detail_px + 8 * min(seen.look.limit_deg, 15.0) - travel - 3 * seen.look.off_normal_deg

    whole = sorted((s for s in looks if s.cells.all()), key=score, reverse=True)
    if whole and whole[0].look.detail_px >= GOOD_SINGLE_PX:
        return whole[0], None
    best: tuple[float, Seen, Seen] | None = None
    firsts = sorted(looks, key=lambda s: -(s.share * 400 + score(s)))[:PAIR_POOL]  # the arm rests at the first look
    for first in firsts:
        for second in looks:  # any look will do as the second, if it shows what the first one misses
            if second is first or not np.all(first.cells | second.cells):
                continue
            value = min(score(first), score(second)) + 200 * first.share
            if (best is None or value > best[0]) and planner.path_problem(screen, first.look.q, second.look.q) is None:
                best = (value, first, second)
    if best is not None:
        return best[1], best[2]
    if whole:
        return whole[0], None
    ahead = float(np.hypot(*centre[:2])) * 1e2
    most = max((s.share for s in looks), default=0.0)
    raise Unreachable(
        f"找不到能看全屏幕的看屏位，两个角度拼也拼不全（单个角度最多看到 {most:.0%}）。平板中心现在离底座 {ahead:.0f} cm："
        "放在 12°–20° 的斜支架上、中心离底座 31–36 cm 再试。"
    )


def unseen_in_rectified(rectifier: ScreenRectifier, hide: np.ndarray, step: int = 8) -> np.ndarray:
    """(out_height, out_width) between 0 and 1: where the rectified picture of a look shows something else than the
    screen (pen, holder, or nothing because that part is outside the camera's picture); soft at its edge."""
    width, height = rectifier.out_width, rectifier.out_height
    us, vs = (np.arange(0, width, step) + 0.5 * step) / width, (np.arange(0, height, step) + 0.5 * step) / height
    u, v = np.meshgrid(us, vs)
    pixel = np.asarray(rectifier.norm_to_cam(np.column_stack([u.ravel(), v.ravel()])), np.float64).reshape(-1, 2)
    inside = np.all((pixel > 4) & (pixel < np.array([hide.shape[1], hide.shape[0]]) - 5), axis=1)
    unseen = ~inside
    at = np.round(pixel[inside]).astype(int)
    unseen[inside] = hide[at[:, 1], at[:, 0]] > 0
    coarse = unseen.reshape(len(vs), len(us)).astype(np.float32)
    full = (cv2.resize(coarse, (width, height), interpolation=cv2.INTER_LINEAR) > 0.01).astype(np.float32)
    full = cv2.dilate(full, np.ones((2 * step + 1, 2 * step + 1), np.uint8))
    return cv2.GaussianBlur(full, (0, 0), step / 2)
