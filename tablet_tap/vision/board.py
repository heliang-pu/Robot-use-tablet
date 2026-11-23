"""A printed calibration board (a chessboard) in a camera frame.

The board's own frame: the origin is the inner corner in row 0, column 0, x runs along the columns, y along the rows,
z points into the board (the pattern is seen from -z), units are metres. "Inner corners" are the points where four
squares meet: a board of 12 x 9 squares has 11 x 8 of them.
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np

from tablet_tap.errors import CalibrationError

_FLAGS = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY


@dataclasses.dataclass(frozen=True)
class BoardSpec:
    cols: int  # inner corners along x
    rows: int  # inner corners along y
    pitch_m: float  # side of one square
    glass_m: float = 0.0  # how far the pattern lies under the surface a pen touches (a pattern on the back of glass)

    def __post_init__(self) -> None:
        if self.cols < 3 or self.rows < 3 or self.pitch_m <= 0 or self.glass_m < 0:
            raise ValueError(f"标定板的规格不合理：{self}")

    def corner(self, row: int, col: int, on_surface: bool = False) -> np.ndarray:
        """(3,) board-frame position of an inner corner; ``on_surface``: of the point a pen touches right above it."""
        return np.array([col * self.pitch_m, row * self.pitch_m, -self.glass_m if on_surface else 0.0])

    def corners(self) -> np.ndarray:
        """(rows * cols, 3) all inner corners, row by row."""
        grid = np.zeros((self.rows, self.cols, 3))
        grid[..., 0] = np.arange(self.cols)[None, :] * self.pitch_m
        grid[..., 1] = np.arange(self.rows)[:, None] * self.pitch_m
        return grid.reshape(-1, 3)

    def extremes(self) -> list[tuple[int, int]]:
        """(row, col) of the four outermost inner corners, going round the board."""
        return [(0, 0), (0, self.cols - 1), (self.rows - 1, self.cols - 1), (self.rows - 1, 0)]


def _grow(gray: np.ndarray) -> tuple[int, int] | None:
    """(cols, rows) of the largest chessboard the detector can grow from a few small starting patterns."""
    best = None
    for start in ((3, 3), (4, 4), (6, 4), (7, 7)):
        found, _, meta = cv2.findChessboardCornersSBWithMeta(gray, start, _FLAGS | cv2.CALIB_CB_LARGER)
        if found and (best is None or meta.shape[0] * meta.shape[1] > best[0] * best[1]):
            best = (meta.shape[1], meta.shape[0])
    return best


def find_chessboard(image_bgr: np.ndarray, cols: int | None = None, rows: int | None = None) -> np.ndarray:
    """The inner corners of the one chessboard in the picture: (rows, cols, 2) pixels, sub-pixel.

    With ``cols`` and ``rows`` None the board's size is found too (the largest grid the detector can grow; give the
    size when part of the board might be mistaken for the whole). Seen in the picture, the rows follow the columns
    clockwise, so the board's z axis points away from the camera.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    wanted = (cols, rows) if cols is not None and rows is not None else _grow(gray)
    if wanted is None:
        raise CalibrationError("画面里没有找到棋盘格标定板：检查它是否完整在画面里、有没有反光或被挡住。")
    found, corners = cv2.findChessboardCornersSB(gray, wanted, _FLAGS)
    if not found:
        seen = _grow(gray)
        if seen is None:
            raise CalibrationError("画面里没有找到棋盘格标定板：检查它是否完整在画面里、有没有反光或被挡住。")
        raise CalibrationError(
            f"没有找到 {wanted[0]} × {wanted[1]} 个内角点的棋盘格；画面里能认出的是 {seen[0]} × {seen[1]} 个内角点"
            "（内角点数 = 每边的格数 − 1）。格数填错了，或者板的一部分被挡住、反光。"
        )
    grid = np.asarray(corners, np.float64).reshape(wanted[1], wanted[0], 2)
    edge_x, edge_y = grid[0, 1] - grid[0, 0], grid[1, 0] - grid[0, 0]
    if edge_x[0] * edge_y[1] - edge_x[1] * edge_y[0] < 0:
        grid = grid[
            ::-1
        ]  # seen in the picture, y must follow x clockwise: the board's z then points away from the camera
    return grid


def numbered_points(grid: np.ndarray, count: int = 4) -> list[tuple[int, int]]:
    """(row, col) of the inner corners a person is asked to touch, in the order they are numbered on the picture:
    the four outermost corners (``count`` 4), or those and the middle of each side (``count`` 8). Number 1 is the
    outermost corner nearest the picture's top-left; the others follow clockwise round the board."""
    if count not in (4, 8):
        raise ValueError("count must be 4 or 8")
    rows, cols = grid.shape[:2]
    last_r, last_c, mid_r, mid_c = rows - 1, cols - 1, (rows - 1) // 2, (cols - 1) // 2
    # clockwise in the picture (see find_chessboard), corners at the even places
    ring = [
        (0, 0),
        (0, mid_c),
        (0, last_c),
        (mid_r, last_c),
        (last_r, last_c),
        (last_r, mid_c),
        (last_r, 0),
        (mid_r, 0),
    ]
    first = 2 * int(np.argmin([grid[r, c].sum() for r, c in ring[::2]]))
    ring = ring[first:] + ring[:first]
    return ring if count == 8 else ring[::2]


def numbered_extremes(grid: np.ndarray) -> list[tuple[int, int]]:
    """``numbered_points`` for the four outermost corners."""
    return numbered_points(grid, 4)


def draw_board(image_bgr: np.ndarray, grid: np.ndarray, numbered: list[tuple[int, int]] | None = None) -> np.ndarray:
    """A copy of the picture with the detected corners marked, and the corners in ``numbered`` labelled 1, 2, ..."""
    out = image_bgr.copy()
    scale = max(out.shape[1] / 1280.0, 0.5)
    for point in grid.reshape(-1, 2):
        cv2.circle(out, tuple(np.round(point).astype(int)), max(int(2 * scale), 1), (0, 200, 255), -1, cv2.LINE_AA)
    centre = grid.reshape(-1, 2).mean(axis=0)
    for number, (row, col) in enumerate(numbered or [], 1):
        point = grid[row, col]
        away = (point - centre) / max(np.linalg.norm(point - centre), 1e-9)
        text_at = point + away * 45 * scale + np.array([-12.0, 12.0]) * scale
        cv2.circle(
            out, tuple(np.round(point).astype(int)), int(9 * scale), (0, 0, 255), max(int(2 * scale), 1), cv2.LINE_AA
        )
        for colour, width in (((255, 255, 255), 7), ((0, 0, 255), 3)):
            cv2.putText(out, str(number), tuple(np.round(text_at).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1.3 * scale,
                        colour, max(int(width * scale), 1), cv2.LINE_AA)  # fmt: skip
    return out
