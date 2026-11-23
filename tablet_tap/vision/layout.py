"""What the calibration page draws: ArUco markers at known screen positions, the touch dot, the target crosshair.

``ScreenLayout`` holds the markers' positions in normalized screen coordinates ``(u, v)`` (u in [0, 1] left to right,
v in [0, 1] top to bottom, over the full physical display). ``render_layout`` draws the same picture as the page
(calibration/static/calib.html) for tests and offline checks.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Sequence

import cv2
import numpy as np

ARUCO_DICT = cv2.aruco.DICT_4X4_50
ARUCO_DICT_NAME = "DICT_4X4_50"
# A DICT_4X4 marker is 6 cells across: 4 data cells and a black border of 1 cell on each side.
MARKER_CELLS = 6
N_MARKER_IDS = 50

# What the calibration page draws; the page gets these numbers from the server (calibration.page_server.page_style), and
# render_layout() below draws the same picture. Lengths are fractions of the screen HEIGHT.
MARKER_SIDE = 0.22
MARKER_INSET = 0.04
BACKGROUND_HEX = "#202020"
DOT_HEX = "#ff00ff"
DOT_DIAMETER = 0.035  # the filled magenta disc at the last pointerdown
DOT_RING = 0.004  # width of the white ring around it
TARGET_ARM = 0.04  # half length of the crosshair's lines
TARGET_RING = 0.03  # outer diameter of the crosshair's ring
TARGET_LINE = 0.003  # width of the crosshair's lines and of its ring

_BACKGROUND_BGR = (0x20, 0x20, 0x20)
_DOT_BGR = (255, 0, 255)
_WHITE_BGR = (255, 255, 255)


@dataclasses.dataclass(frozen=True)
class ScreenLayout:
    """What the calibration page draws, in normalized screen coordinates."""

    aspect: float  # screen width / height
    markers: dict[int, tuple[float, float, float, float]]  # marker id -> (u0, v0, u1, v1) of its outer black square

    def to_json(self) -> dict[str, Any]:
        return {
            "aspect": self.aspect,
            "markers": {str(marker_id): list(box) for marker_id, box in sorted(self.markers.items())},
            "dictionary": ARUCO_DICT_NAME,
            "marker_cells": MARKER_CELLS,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> ScreenLayout:
        dictionary = d.get("dictionary", ARUCO_DICT_NAME)
        if dictionary != ARUCO_DICT_NAME:
            raise ValueError(f"标定码字典是 {dictionary}，这份代码只认 {ARUCO_DICT_NAME}")
        markers = {int(marker_id): tuple(float(x) for x in box) for marker_id, box in d["markers"].items()}
        if any(len(box) != 4 for box in markers.values()):
            raise ValueError("标定码的位置应是 (u0, v0, u1, v1)")
        return cls(aspect=float(d["aspect"]), markers=markers)  # type: ignore[arg-type]


def default_layout(aspect: float) -> ScreenLayout:
    """Six markers: ids 0, 1, 2 along the top (left, centre, right) and 3, 4, 5 along the bottom.

    A marker's side is MARKER_SIDE of the screen height (so it is square on the glass) and it sits MARKER_INSET of
    the height from the edges. Landscape screens only: in portrait three markers do not fit side by side.
    """
    if not (math.isfinite(aspect) and aspect >= 1.0):
        raise ValueError(f"屏幕宽高比是 {aspect:.3f}：平板要横屏放置（宽 > 高）")
    side_u, inset_u = MARKER_SIDE / aspect, MARKER_INSET / aspect
    columns = (inset_u, 0.5 - side_u / 2, 1.0 - inset_u - side_u)
    rows = (MARKER_INSET, 1.0 - MARKER_INSET - MARKER_SIDE)
    markers = {}
    for r, v0 in enumerate(rows):
        for c, u0 in enumerate(columns):
            markers[r * len(columns) + c] = (u0, v0, u0 + side_u, v0 + MARKER_SIDE)
    return ScreenLayout(aspect=float(aspect), markers=markers)


# --------------------------------------------------------------------------------------------------------------------
# The calibration screen as an image (what calib.html draws)
# --------------------------------------------------------------------------------------------------------------------


def marker_bits(marker_id: int) -> np.ndarray:
    """The marker as MARKER_CELLS x MARKER_CELLS cells, 0 (black) or 255 (white), border included."""
    if not 0 <= marker_id < N_MARKER_IDS:
        raise ValueError(f"标定码编号 {marker_id} 不在 0..{N_MARKER_IDS - 1} 内")
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    return cv2.aruco.generateImageMarker(dictionary, marker_id, MARKER_CELLS)


def marker_png(marker_id: int, cell_px: int = 40) -> bytes:
    """PNG of one marker (no quiet zone), each cell `cell_px` pixels; the page scales it without smoothing."""
    cell_px = int(np.clip(cell_px, 1, 200))
    image = np.kron(marker_bits(marker_id), np.ones((cell_px, cell_px), np.uint8))
    ok, data = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("PNG 编码失败")
    return data.tobytes()


def _coverage(lo: float, hi: float, start: int, stop: int) -> np.ndarray:
    """Share of each unit pixel [i, i + 1), i in start..stop-1, that the interval [lo, hi) covers."""
    i = np.arange(start, stop, dtype=np.float64)
    return np.clip(np.minimum(i + 1.0, hi) - np.maximum(i, lo), 0.0, 1.0)


def _window(lo: float, hi: float, n: int) -> tuple[int, int]:
    """The pixels of 0..n-1 that the interval [lo, hi) touches, as a (start, stop) range."""
    return int(np.clip(math.floor(lo), 0, n)), int(np.clip(math.ceil(hi), 0, n))


def _blend(patch: np.ndarray, alpha: np.ndarray, bgr: Sequence[float]) -> None:
    patch *= 1.0 - alpha[..., None]
    patch += alpha[..., None] * np.asarray(bgr, dtype=patch.dtype)


def _draw_rect(image: np.ndarray, x0: float, y0: float, x1: float, y1: float, bgr: Sequence[float]) -> None:
    """Axis-aligned rectangle with exact area coverage (edge coordinates: pixel i spans [i, i + 1))."""
    (xa, xb), (ya, yb) = _window(x0, x1, image.shape[1]), _window(y0, y1, image.shape[0])
    _blend(image[ya:yb, xa:xb], np.outer(_coverage(y0, y1, ya, yb), _coverage(x0, x1, xa, xb)), bgr)


def _draw_disc(image: np.ndarray, cx: float, cy: float, radius: float, bgr: Sequence[float], hole: float = 0.0) -> None:
    """Anti-aliased disc (edge coordinates), or a ring when `hole` (its inner radius) is given."""
    reach = radius + 1.0
    (xa, xb), (ya, yb) = (
        _window(cx - reach, cx + reach, image.shape[1]),
        _window(cy - reach, cy + reach, image.shape[0]),
    )
    yy, xx = np.mgrid[ya:yb, xa:xb].astype(np.float64)
    distance = np.hypot(xx + 0.5 - cx, yy + 0.5 - cy)
    alpha = np.clip(radius - distance + 0.5, 0.0, 1.0)
    if hole > 0:
        alpha -= np.clip(hole - distance + 0.5, 0.0, 1.0)
    _blend(image[ya:yb, xa:xb], alpha, bgr)


def _draw_marker(image: np.ndarray, marker_id: int, x0: float, y0: float, x1: float, y1: float) -> None:
    """The marker in [x0, x1) x [y0, y1) plus a white quiet zone of one cell around it, with exact area coverage."""
    cells = np.full((MARKER_CELLS + 2, MARKER_CELLS + 2), 255.0)
    cells[1:-1, 1:-1] = marker_bits(marker_id)
    cell_w, cell_h = (x1 - x0) / MARKER_CELLS, (y1 - y0) / MARKER_CELLS
    n = MARKER_CELLS + 2
    (xa, xb), (ya, yb) = (
        _window(x0 - cell_w, x1 + cell_w, image.shape[1]),
        _window(y0 - cell_h, y1 + cell_h, image.shape[0]),
    )
    wx = np.stack([_coverage(x0 + (k - 1) * cell_w, x0 + k * cell_w, xa, xb) for k in range(n)], axis=1)
    wy = np.stack([_coverage(y0 + (k - 1) * cell_h, y0 + k * cell_h, ya, yb) for k in range(n)], axis=1)
    patch = image[ya:yb, xa:xb]
    patch *= 1.0 - np.outer(wy.sum(axis=1), wx.sum(axis=1))[..., None]
    patch += (wy @ cells @ wx.T)[..., None]


def render_layout(
    layout: ScreenLayout,
    width: int,
    mode: str = "markers",
    target: tuple[float, float] | None = None,
    dot: tuple[float, float] | None = None,
) -> np.ndarray:
    """The calibration page as a BGR image `width` pixels wide: what calib.html draws on a screen of that size.

    `mode` is the page's mode ("markers", "target" with `target` = (u, v) of the crosshair, or "blank"); `dot` is the
    (u, v) of the magenta touch marker, if any. The button, the warning banner and the small text are not drawn.
    """
    if mode not in ("markers", "target", "blank"):
        raise ValueError(f"未知的页面模式 {mode!r}")
    height = round(width / layout.aspect)
    image = np.empty((height, width, 3), dtype=np.float64)
    image[:] = _BACKGROUND_BGR
    if mode == "markers":
        for marker_id, (u0, v0, u1, v1) in layout.markers.items():
            _draw_marker(image, marker_id, u0 * width, v0 * height, u1 * width, v1 * height)
    if mode == "target" and target is not None:
        cx, cy = target[0] * width, target[1] * height
        arm, line, ring = TARGET_ARM * height, max(2.0, TARGET_LINE * height), TARGET_RING * height / 2
        _draw_rect(image, cx - arm, cy - line / 2, cx + arm, cy + line / 2, _WHITE_BGR)
        _draw_rect(image, cx - line / 2, cy - arm, cx + line / 2, cy + arm, _WHITE_BGR)
        _draw_disc(image, cx, cy, ring, _WHITE_BGR, hole=ring - line)
    if dot is not None:
        cx, cy = dot[0] * width, dot[1] * height
        radius = DOT_DIAMETER * height / 2
        _draw_disc(image, cx, cy, radius + max(1.0, DOT_RING * height), _WHITE_BGR)
        _draw_disc(image, cx, cy, radius, _DOT_BGR)
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------------------------------------------------
# Calibration and rectification
# --------------------------------------------------------------------------------------------------------------------
