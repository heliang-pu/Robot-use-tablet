"""The tablet's own outline against the table, as a camera above sees it: what can be followed when the screen shows no
markers (an app is in front) and no printed tags are stuck beside it.

The picture around where the tablet is believed to lie is laid flat (the tablet's plane, a fixed scale), so that the
body's edges run along the rows and columns. From outside inward, each column is searched for where the table ends and
the body begins (its upper edge), each row for the left and the right edge: what the screen shows, and a lamp mirrored
in it, do not matter, because the search stops at the body's rim. Three lines, or the upper one and one side, give where
the body lies on its plane. The edges' places are first learned while the calibration page's markers say where the
screen is; afterwards a measurement says how far the body has moved from there.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import cv2
import numpy as np

from tablet_tap.core.pinhole import CameraModel

MARGIN_M = 0.045  # how much of the table around the screen is laid flat with it: the body may have moved this far
M_PER_PX = 0.0004  # of the flat picture; the top camera resolves about 0.5 mm
MIN_CONTRAST = 14.0  # colour distance (Lab) by which the body has to differ from the table
RUN_PX = 6  # this many pixels in a row past the rim have to be body
TOP_SPAN = (0.10, 0.90)  # the part of the screen's width over which the upper edge is searched
SIDE_SPAN = (0.04, 0.28)  # of its height, for the side edges: from 0.3 down the bench's stand is wider than the body
INLIER_M = 0.0006
MIN_INLIERS = 0.6
MAX_SKEW_DEG = 3.0  # a side that is not square to the upper edge is something else's edge
WIDTH_TOL_M = 0.0035


@dataclasses.dataclass(frozen=True)
class Edges:
    """The body's edges in the frame of the pose they were measured with (metres; x right, y down, origin at the
    screen's top-left corner): the upper edge as y = top + slope * (x - mid_x), the sides as x at ``side_y``."""

    top: float
    slope: float
    left: float | None
    right: float | None
    mid_x: float
    side_y: float
    support: float  # the share of the searched columns and rows that agree with the lines

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Edges:
        return cls(**{f.name: data[f.name] for f in dataclasses.fields(cls)})


def flat_picture(image_bgr: np.ndarray, model: CameraModel, camera_in_base: np.ndarray, pose: np.ndarray, size_m: tuple[float, float]) -> np.ndarray:
    """The picture of the tablet's plane around the screen: pixel (column j, row i) shows the plane point
    (j * M_PER_PX - MARGIN_M, i * M_PER_PX - MARGIN_M) of the tablet's frame."""
    width, height = (round((s + 2 * MARGIN_M) / M_PER_PX) for s in size_m)
    corners = np.array([[-MARGIN_M, -MARGIN_M, 0, 1], [size_m[0] + MARGIN_M, -MARGIN_M, 0, 1],
                        [size_m[0] + MARGIN_M, size_m[1] + MARGIN_M, 0, 1], [-MARGIN_M, size_m[1] + MARGIN_M, 0, 1]])  # fmt: skip
    in_camera = corners @ (np.linalg.inv(np.asarray(camera_in_base, np.float64)) @ np.asarray(pose, np.float64)).T
    pixels = model.project(in_camera[:, :3]).astype(np.float32)
    flat = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    return cv2.warpPerspective(image_bgr, cv2.getPerspectiveTransform(pixels, flat), (width, height), flags=cv2.INTER_LINEAR)


def _not_table(flat_bgr: np.ndarray) -> np.ndarray | None:
    """How far each pixel's colour is from the table's, in units of the contrast needed; None when the picture's rim,
    where the table should be, is not of one colour (something lies against the tablet all around)."""
    lab = cv2.cvtColor(cv2.GaussianBlur(flat_bgr, (0, 0), 1.2), cv2.COLOR_BGR2LAB).astype(np.float32)
    rim = max(8, round(0.008 / M_PER_PX))
    ring = np.concatenate([lab[:rim].reshape(-1, 3), lab[:, :rim].reshape(-1, 3), lab[:, -rim:].reshape(-1, 3)])
    table = np.median(ring, axis=0)
    spread = float(np.median(np.linalg.norm(ring - table, axis=1)))
    if spread > MIN_CONTRAST:
        return None
    return np.linalg.norm(lab - table, axis=2) / max(MIN_CONTRAST, 4.0 * spread)


def _first_crossing(profile: np.ndarray) -> float | None:
    """Where, coming from the table, a profile of ``_not_table`` values enters the body: the index (with a fraction)
    of the first crossing of 1 that is followed by a run of body."""
    above = profile > 1.0
    run = np.convolve(above.astype(np.int32), np.ones(RUN_PX, np.int32), mode="valid") == RUN_PX
    starts = np.flatnonzero(run)
    if len(starts) == 0 or starts[0] == 0:
        return None
    k = int(starts[0])
    low, high = float(profile[k - 1]), float(profile[k])
    return k - 1 + (1.0 - low) / max(high - low, 1e-6)


def _line(along: np.ndarray, across: np.ndarray) -> tuple[float, float, float] | None:
    """across = offset + slope * along through most of the points (the others are a clip, a foot, a cable): (offset
    at along = 0, slope, the share of points within INLIER_M); None when no line holds most of them."""
    if len(along) < 12:
        return None
    rng, best = np.random.default_rng(0), None
    for _ in range(60):
        i, j = rng.choice(len(along), 2, replace=False)
        if abs(along[i] - along[j]) < min(0.02, 0.4 * np.ptp(along)):
            continue
        slope = (across[i] - across[j]) / (along[i] - along[j])
        near = np.abs(across - (across[i] + slope * (along - along[i]))) < INLIER_M
        if best is None or near.sum() > best.sum():
            best = near
    if best is None or best.mean() < MIN_INLIERS:
        return None
    slope, offset = np.polyfit(along[best], across[best], 1)
    near = np.abs(across - (offset + slope * along)) < INLIER_M
    slope, offset = np.polyfit(along[near], across[near], 1)
    return float(offset), float(slope), float(near.mean())


def measure(image_bgr: np.ndarray, model: CameraModel, camera_in_base: np.ndarray, pose: np.ndarray, size_m: tuple[float, float]) -> Edges | None:
    """The body's edges as the picture shows them, in the frame of ``pose`` (where the tablet is believed to lie);
    None when the upper edge is not found, or neither side."""
    flat = flat_picture(image_bgr, model, camera_in_base, pose, size_m)
    far = _not_table(flat)
    if far is None:
        return None
    width, height = size_m

    def at(index: float) -> float:
        return index * M_PER_PX - MARGIN_M

    columns = np.arange(round((TOP_SPAN[0] * width + MARGIN_M) / M_PER_PX), round((TOP_SPAN[1] * width + MARGIN_M) / M_PER_PX), 2)
    found = [(at(j), _first_crossing(far[:, j])) for j in columns]
    xs, ys = np.array([x for x, k in found if k is not None]), np.array([at(k) for _, k in found if k is not None])
    top = _line(xs, ys)
    if top is None:
        return None
    mid_x, side_y = 0.5 * width, 0.5 * (SIDE_SPAN[0] + SIDE_SPAN[1]) * height
    top_at_mid, slope, support = top[0] + top[1] * mid_x, top[1], top[2] * len(xs) / len(columns)
    rows = np.arange(round((SIDE_SPAN[0] * height + MARGIN_M) / M_PER_PX), round((SIDE_SPAN[1] * height + MARGIN_M) / M_PER_PX))
    sides: list[float | None] = []
    for mirrored in (False, True):
        crossings = [(at(i), _first_crossing(far[i, ::-1] if mirrored else far[i])) for i in rows]
        along = np.array([y for y, k in crossings if k is not None])
        across = np.array([at(far.shape[1] - 1 - k) if mirrored else at(k) for _, k in crossings if k is not None])
        fit = _line(along, across)
        # a side of the body is square to its upper edge: x grows by -slope for each metre down
        if fit is None or abs(np.degrees(np.arctan(fit[1]) + np.arctan(slope))) > MAX_SKEW_DEG or fit[2] * len(along) / len(rows) < MIN_INLIERS:
            sides.append(None)
        else:
            sides.append(fit[0] + fit[1] * side_y)
    if sides[0] is None and sides[1] is None:
        return None
    return Edges(float(top_at_mid), float(slope), sides[0], sides[1], float(mid_x), float(side_y), float(support))


def moved(learned: Edges, now: Edges) -> np.ndarray | None:
    """(4, 4) how the body has moved on its plane between ``learned`` and ``now``, both measured in the frame of the
    same pose: the pose of now is that pose times this. None when the two do not show the same body (the sides are
    not as far apart as they were: one of them is another thing's edge)."""
    if learned.left is not None and learned.right is not None and now.left is not None and now.right is not None:
        if abs((now.right - now.left) - (learned.right - learned.left)) > WIDTH_TOL_M:
            return None
    turn = np.arctan(now.slope) - np.arctan(learned.slope)
    cos, sin = np.cos(turn), np.sin(turn)
    rot = np.array([[cos, -sin], [sin, cos]])
    shifts = []
    for a, b in ((learned.left, now.left), (learned.right, now.right)):
        if a is not None and b is not None:
            shifts.append(b - (rot @ [a, learned.side_y])[0])
    if not shifts:
        return None
    step = np.eye(4)
    step[:2, :2] = rot
    step[0, 3] = float(np.mean(shifts))
    # the upper edge at the middle of the screen, after the turn and the shift aside, has to lie on the line of now
    x_mid = (rot @ [learned.mid_x, learned.top])[0] + step[0, 3]
    step[1, 3] = now.top + now.slope * (x_mid - now.mid_x) - (rot @ [learned.mid_x, learned.top])[1]
    return step


BAND_M = 0.010  # the outermost strip of the flat picture: past the body, its stand's feet and their shadows


def crossed(image_bgr: np.ndarray, model: CameraModel, camera_in_base: np.ndarray, pose: np.ndarray, size_m: tuple[float, float]) -> tuple[np.ndarray, int]:
    """Which pixels of the strip of table around the tablet (of the flat picture) are not table, and how many the strip
    has: a hand or an arm reaching in has to cross it. The part of the strip outside the picture is left out."""
    flat = flat_picture(image_bgr, model, camera_in_base, pose, size_m)
    seen = flat_picture(np.full_like(image_bgr, 255), model, camera_in_base, pose, size_m)[..., 0] > 250
    band = np.zeros(flat.shape[:2], bool)
    width = round(BAND_M / M_PER_PX)
    band[:width] = band[-width:] = band[:, :width] = band[:, -width:] = True
    band &= seen
    if not band.any():
        return band, 0
    lab = cv2.cvtColor(cv2.GaussianBlur(flat, (0, 0), 1.2), cv2.COLOR_BGR2LAB).astype(np.float32)
    table = np.median(lab[band], axis=0)
    # shades of the cloth differ in brightness; a hand, a sleeve and a robot differ in tint too or by far more
    far = np.hypot(0.5 * (lab[..., 0] - table[0]), np.linalg.norm(lab[..., 1:] - table[1:], axis=2))
    return (far > MIN_CONTRAST) & band, int(band.sum())


def besides(now: np.ndarray, known: np.ndarray | None) -> np.ndarray:
    """``now`` without what is ``known`` to lie there (give or take a few millimetres)."""
    if known is None or not known.any():
        return now
    return now & (cv2.dilate(known.astype(np.uint8), np.ones((15, 15), np.uint8)) == 0)
