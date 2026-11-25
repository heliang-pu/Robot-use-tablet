"""The tablet anywhere in a fixed camera's picture, without markers: the one thing on the table's cloth that has the
body's size. ``find`` gives a first guess, good to a centimetre or two, from which ``vision.outline`` measures the rim.

The tablet is on a stand that is pushed about on the table, so that it turns about the vertical and shifts along the
table; its height and its slant stay. Poses are moved only that way (``on_table``).
"""

from __future__ import annotations

import cv2
import numpy as np

from tablet_tap.core.pinhole import CameraModel

M_PER_PX = 0.001  # of the view from above
REACH_X, REACH_Y = (0.10, 0.60), (-0.60, 0.40)  # the part of the table searched, in the arm's base frame
MIN_CONTRAST = 16.0
FEET_M = 0.030  # what sticks out of the body less far than this (the stand's feet, a cable) is cut off
SIZE_TOL = 0.16  # the stand shows below the body, and the body is seen at a slant
MAX_TURN_DEG = 75.0  # the tablet is not upside down


def slide(pose: np.ndarray, size_m: tuple[float, float], dx: float, dy: float, turn: float) -> np.ndarray:
    """``pose`` with the tablet shifted along the table and turned about the vertical through the screen's centre."""
    centre = (pose @ [0.5 * size_m[0], 0.5 * size_m[1], 0.0, 1.0])[:3]
    cos, sin = np.cos(turn), np.sin(turn)
    move = np.eye(4)
    move[:2, :2] = [[cos, -sin], [sin, cos]]
    move[:3, 3] = centre + [dx, dy, 0.0] - move[:3, :3] @ centre
    return move @ pose


def _fit(pose: np.ndarray, size_m: tuple[float, float], residual, start=(0.0, 0.0, 0.0)) -> tuple[np.ndarray, float]:  # noqa: ANN001
    """The slide of ``pose`` whose ``residual`` is least (Gauss-Newton), and the RMS of what is left."""
    x = np.array(start, np.float64)
    for _ in range(15):
        r = residual(slide(pose, size_m, *x))
        jac = np.stack([(residual(slide(pose, size_m, *(x + h))) - r) / 1e-5 for h in 1e-5 * np.eye(3)], axis=1)
        step = np.linalg.lstsq(jac, -r, rcond=None)[0]
        x += step
        if np.abs(step).max() < 1e-7:
            break
    moved = slide(pose, size_m, *x)
    return moved, float(np.sqrt(np.mean(residual(moved) ** 2)))


def on_table(pose: np.ndarray, seen: np.ndarray, size_m: tuple[float, float], model: CameraModel, camera_in_base: np.ndarray) -> tuple[np.ndarray, float]:
    """``pose`` slid on the table so that it looks to the camera as ``seen`` does, and the RMS left (pixels). ``seen``
    moved within the tablet's own plane: the right picture, at the wrong height."""
    w, h = size_m
    corners = np.array([[0, 0, 0, 1], [w, 0, 0, 1], [w, h, 0, 1], [0, h, 0, 1]], np.float64)
    to_camera = np.linalg.inv(np.asarray(camera_in_base, np.float64))

    def pixels(p: np.ndarray) -> np.ndarray:
        return model.project((corners @ (to_camera @ p).T)[:, :3])

    wanted = pixels(seen)
    return _fit(pose, size_m, lambda p: (pixels(p) - wanted).ravel())


def _from_above(model: CameraModel, camera_in_base: np.ndarray, height: float) -> np.ndarray:
    """(3, 3) from the plane z = ``height`` of the base frame (x, y, 1) to the camera's pixels."""
    to_camera = np.linalg.inv(np.asarray(camera_in_base, np.float64))
    return model.matrix @ np.column_stack([to_camera[:3, 0], to_camera[:3, 1], to_camera[:3, 2] * height + to_camera[:3, 3]])


def find(
    image_bgr: np.ndarray, model: CameraModel, camera_in_base: np.ndarray, pose: np.ndarray, size_m: tuple[float, float], body: tuple[float, float, float, float]
) -> np.ndarray | None:
    """Where the tablet stands now; ``pose`` is where it stood. ``body``: left, top, right, bottom of the body in the
    tablet's frame. None when nothing on the table has its size."""
    height = float((pose @ [0.5 * size_m[0], 0.5 * size_m[1], 0.0, 1.0])[2])
    plane = _from_above(model, camera_in_base, height)
    rows, cols = round((REACH_X[1] - REACH_X[0]) / M_PER_PX), round((REACH_Y[1] - REACH_Y[0]) / M_PER_PX)
    # picture from above: row i is x = REACH_X[1] - i, column j is y = REACH_Y[1] - j (away from the arm is up)
    flat = np.array([[0, -M_PER_PX, REACH_X[1]], [-M_PER_PX, 0, REACH_Y[1]], [0, 0, 1.0]])
    to_pixels = plane @ flat
    above = cv2.warpPerspective(image_bgr, to_pixels, (cols, rows), flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)
    inside = cv2.warpPerspective(np.full(image_bgr.shape[:2], 255, np.uint8), to_pixels, (cols, rows), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    jj, ii = np.meshgrid(np.arange(cols), np.arange(rows))
    ahead = (to_pixels[2, 0] * jj + to_pixels[2, 1] * ii + to_pixels[2, 2]) > 0.05  # what is behind the camera is no picture
    valid = (inside > 0) & ahead
    if valid.mean() < 0.1:
        return None
    lab = cv2.cvtColor(cv2.GaussianBlur(above, (0, 0), 1.5), cv2.COLOR_BGR2LAB).astype(np.float32)
    # the cloth: the commonest colour of the picture
    seen = lab[valid]
    bins = np.floor(seen / 8.0).astype(np.int64)
    keys = bins[:, 0] * 4096 + bins[:, 1] * 64 + bins[:, 2]
    values, counts = np.unique(keys, return_counts=True)
    cloth = np.median(seen[keys == values[np.argmax(counts)]], axis=0)
    # its brightness changes over the table far more than its tint does
    far = np.hypot(0.6 * (lab[..., 0] - cloth[0]), np.linalg.norm(lab[..., 1:] - cloth[1:], axis=2))
    mask = ((far > MIN_CONTRAST) & valid).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    solid = np.zeros_like(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # a bright screen inside its rim is filled
    cv2.drawContours(solid, contours, -1, 1, cv2.FILLED)
    feet = round(FEET_M / M_PER_PX) * 2 + 1
    solid = cv2.morphologyEx(solid, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (feet, feet)))
    contours, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    left, top, right, bottom = body
    outline = np.array([[left, top, 0, 1], [right, top, 0, 1], [right, bottom, 0, 1], [left, bottom, 0, 1]], np.float64)
    to_flat = np.linalg.inv(to_pixels)
    to_camera = np.linalg.inv(np.asarray(camera_in_base, np.float64))

    def drawn(p: np.ndarray) -> np.ndarray:
        """The body's corners at pose ``p`` in the picture from above."""
        pixel = model.project((outline @ (to_camera @ p).T)[:, :3])
        there = np.column_stack([pixel, np.ones(4)]) @ to_flat.T
        return there[:, :2] / there[:, 2:]

    now = drawn(pose)
    sides = np.array([np.linalg.norm(now[1] - now[0]), np.linalg.norm(now[3] - now[0])])
    best: tuple[float, np.ndarray] | None = None
    for contour in contours:
        (cx, cy), box, _ = cv2.minAreaRect(contour)
        long, short = max(box), min(box)
        if abs(long / sides.max() - 1) > SIZE_TOL or abs(short / sides.min() - 1) > SIZE_TOL + 0.06:
            continue
        if cv2.contourArea(contour) < 0.85 * long * short:
            continue
        corners = cv2.boxPoints(((cx, cy), box, _)).astype(np.float64)
        for k in range(4):  # which corner of the box is the screen's top left: the turn nearest to what it was
            target = np.roll(corners, k, axis=0)
            if np.cross(target[1] - target[0], target[2] - target[0]) * np.cross(now[1] - now[0], now[2] - now[0]) < 0:
                target = target[::-1]
            if (np.linalg.norm(target[1] - target[0]) > np.linalg.norm(target[3] - target[0])) != (sides[0] > sides[1]):
                continue
            a, b = now[1] - now[0], target[1] - target[0]
            turn = np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b)  # rows run against x and columns against y: same sense
            if abs(np.degrees(turn)) > MAX_TURN_DEG:
                continue
            shift = (target.mean(0) - now.mean(0)) * M_PER_PX
            # the stand's feet pull the box down: its upper edge counts, which is the body's own
            weight = np.sqrt(np.array([2.0, 2.0, 0.5, 0.5]))[:, None]
            there, left_over = _fit(pose, size_m, lambda p: (weight * (drawn(p) - target)).ravel() * M_PER_PX,  # noqa: B023
                                    (-shift[1], -shift[0], turn))  # fmt: skip
            if left_over < 0.02 and (best is None or left_over < best[0]):
                best = (left_over, there)
    return None if best is None else best[1]
