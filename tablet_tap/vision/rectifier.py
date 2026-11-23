"""Camera frame -> upright, full-screen "screenshot" of the tablet.

The tablet shows the calibration page with ArUco markers at known screen positions (vision/layout.py). One camera
frame of it gives the homography camera pixels -> normalized screen coordinates; camera and tablet stand are fixed, so
that homography is then reused for any app content (``rectify`` never looks for markers).

Coordinates:

- normalized screen `(u, v)`: u in [0, 1] left -> right, v in [0, 1] top -> bottom, over the full physical display;
- pixels `(x, y)` of a camera frame or of a rectified image: OpenCV's convention, integer values are pixel centres. The
  left edge of the rectified image (u = 0) is therefore at x = -0.5, and u = (x + 0.5) / width.
"""

from __future__ import annotations

import json
import logging
import math
import pathlib
import time
from typing import Any, Sequence

import cv2
import numpy as np

from tablet_tap.errors import CalibrationError
from tablet_tap.vision.layout import ARUCO_DICT
from tablet_tap.vision.layout import DOT_DIAMETER
from tablet_tap.vision.layout import ScreenLayout

logger = logging.getLogger(__name__)

# from_marker_image: a marker corner further than this from the fitted homography is an outlier (camera px).
RANSAC_PX = 4.0
MIN_MARKERS = 2

# find_touch_dot: "magenta-ness" of a pixel is min(R, B) - G, 255 for the dot and 0 for any grey.
DOT_MIN_MAGENTA = 40.0


def detect_markers(image_bgr: np.ndarray) -> dict[int, np.ndarray]:
    """ArUco markers of ARUCO_DICT in an image: id -> (4, 2) float corners in pixels.

    The corners come in the marker's own order (its top-left, top-right, bottom-right, bottom-left), whatever way up
    the marker is in the picture, refined to sub-pixel. Of two detections with one id the larger is kept.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), parameters)
    corners, ids, _rejected = detector.detectMarkers(gray)
    found: dict[int, np.ndarray] = {}
    if ids is None:
        return found
    for quad, marker_id in zip(corners, ids.ravel().tolist()):
        quad = np.asarray(quad, dtype=np.float64).reshape(4, 2)
        if marker_id not in found or cv2.contourArea(quad.astype(np.float32)) > cv2.contourArea(
            found[marker_id].astype(np.float32)
        ):
            found[marker_id] = quad
    return found


def _apply_homography(h: np.ndarray, points: Any) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.shape[-1] != 2:
        raise ValueError(f"需要 (..., 2) 的坐标，得到 {points.shape}")
    x, y = points[..., 0], points[..., 1]
    w = h[2, 0] * x + h[2, 1] * y + h[2, 2]
    return np.stack(
        [(h[0, 0] * x + h[0, 1] * y + h[0, 2]) / w, (h[1, 0] * x + h[1, 1] * y + h[1, 2]) / w],
        axis=-1,
    )


class ScreenRectifier:
    """The fixed homography between camera pixels and normalized screen coordinates, and the warp that uses it."""

    def __init__(
        self,
        cam_to_norm: np.ndarray,
        aspect: float,
        out_width: int = 1280,
        reprojection_px: float = float("nan"),
        image_size: tuple[int, int] | None = None,
        marker_ids: Sequence[int] = (),
    ) -> None:
        """`cam_to_norm` is the 3x3 homography camera px -> (u, v). `image_size` = (width, height) of the camera
        frames it was calibrated on (rectify() then refuses frames of another size); `marker_ids` = the markers the
        calibration used. Both are optional records."""
        homography = np.asarray(cam_to_norm, dtype=np.float64)
        if homography.shape != (3, 3) or not np.all(np.isfinite(homography)):
            raise ValueError("cam_to_norm 应是有限的 3x3 矩阵")
        if abs(np.linalg.det(homography)) < 1e-18:
            raise ValueError("cam_to_norm 不可逆")
        if not (aspect > 0 and out_width > 0):
            raise ValueError("aspect 和 out_width 应为正数")
        self.homography = homography / np.linalg.norm(homography)
        self.homography_inv = np.linalg.inv(self.homography)
        self.aspect = float(aspect)
        self.out_width = int(out_width)
        self.out_height = round(self.out_width / self.aspect)
        self.reprojection_px = float(reprojection_px)
        self.image_size = None if image_size is None else (int(image_size[0]), int(image_size[1]))
        self.marker_ids = tuple(int(i) for i in marker_ids)
        # Markers seen by from_marker_image (id -> (4, 2) camera px), for the overlay; not saved.
        self.detected: dict[int, np.ndarray] = {}
        # (u, v) -> pixel centres of the rectified image: x = u * W - 0.5.
        to_out = np.array([[self.out_width, 0, -0.5], [0, self.out_height, -0.5], [0, 0, 1]], dtype=np.float64)
        self._cam_to_out = to_out @ self.homography

    @classmethod
    def from_marker_image(cls, image_bgr: np.ndarray, layout: ScreenLayout, out_width: int = 1280) -> ScreenRectifier:
        """Calibrate from one camera frame of the calibration page.

        Uses the four corners of every marker of `layout` that is detected (a robust fit: a mis-detected marker is
        left out). CalibrationError when fewer than MIN_MARKERS markers support the fit.
        """
        height, width = image_bgr.shape[:2]
        detected = detect_markers(image_bgr)
        known = {marker_id: quad for marker_id, quad in detected.items() if marker_id in layout.markers}
        if len(known) < MIN_MARKERS:
            raise CalibrationError(
                f"只看到 {len(known)} 个标定码（编号 {sorted(known)}），至少要 {MIN_MARKERS} 个。"
                "检查：平板是否显示标定页、有没有被挡住、反光或过曝、相机是否对着平板"
            )
        ids = sorted(known)
        camera = np.concatenate([known[marker_id] for marker_id in ids])
        norm = np.concatenate(
            [np.array([(u0, v0), (u1, v0), (u1, v1), (u0, v1)]) for u0, v0, u1, v1 in (layout.markers[i] for i in ids)]
        )
        # Fitted on the glass in units of the screen height (isotropic), so that the threshold is in camera pixels.
        plane = norm * (layout.aspect, 1.0)
        plane_to_cam, mask = cv2.findHomography(plane, camera, cv2.RANSAC, RANSAC_PX)
        if plane_to_cam is None:
            raise CalibrationError(f"看到 {len(known)} 个标定码（编号 {ids}），但拟合不出屏幕平面")
        inliers = mask.ravel().astype(bool)
        used = [marker_id for k, marker_id in enumerate(ids) if inliers[4 * k : 4 * k + 4].sum() >= 3]
        if len(used) < MIN_MARKERS:
            raise CalibrationError(
                f"看到 {len(known)} 个标定码（编号 {ids}），但位置相互吻合的只有 {len(used)} 个，至少要 {MIN_MARKERS} 个"
            )
        norm_to_cam = plane_to_cam @ np.diag([layout.aspect, 1.0, 1.0])
        error = _apply_homography(norm_to_cam, norm) - camera
        rms = float(np.sqrt((error[inliers] ** 2).sum(axis=1).mean()))
        rectifier = cls(
            np.linalg.inv(norm_to_cam),
            layout.aspect,
            out_width=out_width,
            reprojection_px=rms,
            image_size=(width, height),
            marker_ids=used,
        )
        rectifier.detected = known
        quad = rectifier.screen_quad_cam()
        edges = np.roll(quad, -1, axis=0) - quad
        turns = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
        if not (np.all(np.isfinite(quad)) and np.all(turns > 0)):
            # TL, TR, BR, BL must go clockwise in the picture: a camera looking at the glass cannot see it mirrored.
            raise CalibrationError(f"标定码（编号 {ids}）拟合出的屏幕是翻转或扭曲的，这一帧不可用")
        if len(used) == MIN_MARKERS:
            logger.warning("只用了 %d 个标定码（编号 %s），离它们远的地方误差会偏大。", len(used), used)
        return rectifier

    def rectify(self, image_bgr: np.ndarray) -> np.ndarray:
        """The full screen, upright: (round(out_width / aspect), out_width, 3). Parts outside the frame are black."""
        height, width = image_bgr.shape[:2]
        if self.image_size is not None and (width, height) != self.image_size:
            raise ValueError(
                f"这一帧是 {width}x{height}，标定用的是 {self.image_size[0]}x{self.image_size[1]}：换分辨率后要重新标定"
            )
        return cv2.warpPerspective(
            image_bgr, self._cam_to_out, (self.out_width, self.out_height), flags=cv2.INTER_CUBIC
        )

    def cam_to_norm(self, xy: Any) -> np.ndarray:
        """(..., 2) camera pixels -> (u, v)."""
        return _apply_homography(self.homography, xy)

    def norm_to_cam(self, uv: Any) -> np.ndarray:
        """(..., 2) (u, v) -> camera pixels."""
        return _apply_homography(self.homography_inv, uv)

    def screen_quad_cam(self) -> np.ndarray:
        """(4, 2) camera pixels of the screen's corners: top-left, top-right, bottom-right, bottom-left."""
        return self.norm_to_cam(np.array([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]))

    def to_json(self) -> dict[str, Any]:
        return {
            "version": 1,
            "cam_to_norm": self.homography.tolist(),
            "aspect": self.aspect,
            "out_width": self.out_width,
            "reprojection_px": self.reprojection_px if math.isfinite(self.reprojection_px) else None,
            "image_size": None if self.image_size is None else list(self.image_size),
            "marker_ids": list(self.marker_ids),
        }

    def save(self, path: str | pathlib.Path) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_json() | {"saved_at": time.strftime("%Y-%m-%d %H:%M:%S %z")}
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | pathlib.Path) -> ScreenRectifier:
        d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        reprojection = d.get("reprojection_px")
        return cls(
            np.asarray(d["cam_to_norm"], dtype=np.float64),
            d["aspect"],
            out_width=d.get("out_width", 1280),
            reprojection_px=float("nan") if reprojection is None else reprojection,
            image_size=d.get("image_size"),
            marker_ids=d.get("marker_ids", ()),
        )


def draw_overlay(image_bgr: np.ndarray, rectifier: ScreenRectifier) -> np.ndarray:
    """A copy of the camera frame with the detected markers (green, id at the marker's top-left corner) and the
    fitted screen outline (yellow, its top-left corner marked) drawn on it."""
    out = image_bgr.copy()
    scale = max(1.0, out.shape[1] / 1280)
    thickness = max(1, round(2 * scale))
    for marker_id, quad in rectifier.detected.items():
        used = marker_id in rectifier.marker_ids
        color = (0, 255, 0) if used else (0, 0, 255)
        cv2.polylines(out, [np.rint(quad).astype(np.int32)], True, color, thickness, cv2.LINE_AA)
        corner = (int(quad[0, 0]), int(quad[0, 1]))
        cv2.putText(out, str(marker_id), corner, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, color, thickness, cv2.LINE_AA)
    quad = rectifier.screen_quad_cam()
    if np.all(np.isfinite(quad)) and np.abs(quad).max() < 1e5:
        points = np.rint(quad).astype(np.int32)
        cv2.polylines(out, [points], True, (0, 255, 255), thickness, cv2.LINE_AA)
        cv2.circle(out, (int(points[0, 0]), int(points[0, 1])), round(6 * scale), (0, 255, 255), -1, cv2.LINE_AA)
    return out


def find_touch_dot(screen_bgr: np.ndarray, dot_diameter: float = DOT_DIAMETER) -> tuple[float, float] | None:
    """(u, v) of the magenta touch marker of the calibration page in a RECTIFIED screen image; None if there is none.

    A pixel's magenta-ness is min(R, B) - G: zero for black, white and grey (the markers, the background), large for
    the dot. The dot is the blob of that measure that is round and about `dot_diameter` of the image height across
    (so colour fringes along the markers' edges are not taken for it); its centre is the weighted centroid.
    """
    height, width = screen_bgr.shape[:2]
    image = screen_bgr.astype(np.float32)
    magenta = np.clip(np.minimum(image[..., 2], image[..., 0]) - image[..., 1], 0.0, None)
    magenta = cv2.GaussianBlur(magenta, (0, 0), max(0.8, 0.0012 * height))
    peak = float(magenta.max())
    if peak < DOT_MIN_MAGENTA:
        return None
    expected = dot_diameter * height
    best: tuple[float, float, float] | None = None  # (weight, x, y)
    # Blobs are taken at half the height of the strongest one still unexplained, so a dim dot beside a brighter
    # fringe is still found; a few rounds are enough.
    remaining = magenta.copy()
    for _ in range(4):
        peak = float(remaining.max())
        if peak < DOT_MIN_MAGENTA:
            break
        mask = (remaining >= 0.5 * peak).astype(np.uint8)
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        remaining[mask > 0] = 0.0
        for label in range(1, count):
            x, y, w, h, area = (int(value) for value in stats[label])
            diameter = 2.0 * math.sqrt(area / math.pi)
            if not 0.4 * expected <= diameter <= 2.0 * expected:
                continue
            if not 0.6 <= w / h <= 1 / 0.6:
                continue
            if not 0.7 <= area / (math.pi / 4 * w * h) <= 1.2:
                continue
            pad = max(2, round(0.25 * expected))
            x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(width, x + w + pad), min(height, y + h + pad)
            weights = np.clip(magenta[y0:y1, x0:x1] - 0.2 * peak, 0.0, None)
            total = float(weights.sum())
            if total <= 0:
                continue
            ys, xs = np.indices(weights.shape, dtype=np.float64)
            cx, cy = x0 + float((weights * xs).sum()) / total, y0 + float((weights * ys).sum()) / total
            if best is None or total > best[0]:
                best = (total, cx, cy)
        if best is not None:
            break
    if best is None:
        return None
    return (best[1] + 0.5) / width, (best[2] + 0.5) / height


# --------------------------------------------------------------------------------------------------------------------
# Cameras
# --------------------------------------------------------------------------------------------------------------------
