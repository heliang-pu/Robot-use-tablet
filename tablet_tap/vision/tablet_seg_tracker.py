"""无 Tag 码任意姿态平板分割与亚像素 4 角点追踪器（基于 YOLO11-seg + 四边直线交点 + 透视黑边内缩）。

核心设计：
  1. `YOLO11-seg` 分割整块平板机身外轮廓（深色/浅色/熄屏下边界均保持锐利）；
  2. 对掩码上下左右四条直边分别做抗离群点直线拟合（Huber fitLine）并两两求交点，
     消除圆角与局部遮挡抖动，获得亚像素级机身外四角 `outer_quad`（TL, TR, BR, BL）；
  3. 在平板自身透视单应坐标系内，按固定物理黑边比例 `(bx_l, bx_r, by_t, by_b)` 精确内缩，
     得到内圈发光显示区四角 `screen_quad`（即使 60°~70° 大倾角仍严格满足透视不变性）；
  4. 提供 `calibrate_bezel_from_calib_corners`（结合 `/calib` 网页一次性白嫖精确黑边比例）
     与 `fit_screen_plane_rgbd`（结合 RealSense 深度图做 RANSAC 3D 平面拟合，输出精确法向量与 3D 角点）。
"""
from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import cv2
import numpy as np


@dataclasses.dataclass
class TabletDetection:
    """单帧无 Tag 平板检测结果。"""

    conf: float
    mask: np.ndarray  # (H, W) uint8, 0 或 255
    outer_quad: np.ndarray  # (4, 2) float64, 机身外框四角 [TL, TR, BR, BL]
    screen_quad: np.ndarray  # (4, 2) float64, 内圈发光显示区四角 [TL, TR, BR, BL]


def order_quad(pts: np.ndarray) -> np.ndarray:
    """将 4 个角点按严格无自交顺时针顺序排列为 [TL, TR, BR, BL]（兼容任意 0°~360° 旋转）。"""
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    # 图像坐标系 y 向下：atan2 从 -pi 到 +pi 递增即顺时针顺序
    cw = pts[np.argsort(angles)]
    # 旋转起始索引使最靠近左上角（x+y 最小）的顶点位于 index 0
    tl_idx = int(np.argmin(cw[:, 0] + cw[:, 1]))
    return np.roll(cw, -tl_idx, axis=0)


def mask_to_quad_corners(mask_u8: np.ndarray) -> np.ndarray | None:
    """从二值分割掩码拟合四条直边并两两求交，输出亚像素级 4 角点 [TL, TR, BR, BL]。"""
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 400:
        return None

    # 先用凸包抑制机械臂/灵巧手局部遮挡造成的内凹，找到粗略 4 顶点
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    quad: np.ndarray | None = None
    for eps_f in np.linspace(0.01, 0.08, 25):
        ap = cv2.approxPolyDP(hull, eps_f * peri, True)
        if len(ap) == 4:
            quad = ap.reshape(4, 2).astype(np.float64)
            break
    if quad is None:
        quad = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float64)
    ordered = order_quad(quad)

    # 对四条直边中段（避开圆角与遮挡段）做 Huber 直线拟合，再两两求精确交点
    pts = cnt.reshape(-1, 2).astype(np.float64)
    lines: list[tuple[float, float, float]] = []
    for i in range(4):
        p1, p2 = ordered[i], ordered[(i + 1) % 4]
        v = p2 - p1
        elen = float(np.linalg.norm(v))
        if elen < 12.0:
            return ordered
        u = v / elen
        rel = pts - p1
        proj = rel @ u
        perp = np.abs(rel[:, 0] * (-u[1]) + rel[:, 1] * u[0])
        inliers = pts[(proj > 0.12 * elen) & (proj < 0.88 * elen) & (perp < max(6.0, 0.028 * elen))]
        if len(inliers) >= 12:
            vx, vy, x0, y0 = cv2.fitLine(inliers.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            a, b = float(-vy), float(vx)
            c = -(a * float(x0) + b * float(y0))
        else:
            a, b = float(-u[1]), float(u[0])
            c = -(a * p1[0] + b * p1[1])
        lines.append((a, b, c))

    refined: list[np.ndarray] = []
    for i in range(4):
        l_prev = lines[(i - 1) % 4]
        l_curr = lines[i]
        det = l_prev[0] * l_curr[1] - l_curr[0] * l_prev[1]
        if abs(det) < 1e-6:
            refined.append(ordered[i])
        else:
            x = (l_prev[1] * l_curr[2] - l_curr[1] * l_prev[2]) / det
            y = (l_curr[0] * l_prev[2] - l_prev[0] * l_curr[2]) / det
            pt = np.array([x, y], dtype=np.float64)
            refined.append(ordered[i] if np.linalg.norm(pt - ordered[i]) > 60.0 else pt)
    return order_quad(np.array(refined, dtype=np.float64))


def inset_quad_projective(
    outer_quad: np.ndarray,
    bezel_inset: tuple[float, float, float, float] = (0.040, 0.040, 0.056, 0.056),
) -> np.ndarray:
    """在透视单应空间内按固定物理黑边比例 `(左, 右, 上, 下)` 内缩，求发光显示区四角。"""
    bx_l, bx_r, by_t, by_b = bezel_inset
    unit = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(unit, np.asarray(outer_quad, dtype=np.float32))
    inner_unit = np.array(
        [[bx_l, by_t], [1.0 - bx_r, by_t], [1.0 - bx_r, 1.0 - by_b], [bx_l, 1.0 - by_b]],
        dtype=np.float32,
    ).reshape(1, 4, 2)
    inner_px = cv2.perspectiveTransform(inner_unit, H).reshape(4, 2).astype(np.float64)
    return inner_px


def calibrate_bezel_from_calib_corners(
    outer_quad: np.ndarray,
    calib_screen_quad: np.ndarray,
) -> tuple[float, float, float, float]:
    """用 `/calib` 标定页识别出的发光屏四角与机身外四角，一次性自动标定精确黑边比例。"""
    unit = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    H_inv = cv2.getPerspectiveTransform(np.asarray(outer_quad, dtype=np.float32), unit)
    uv = cv2.perspectiveTransform(np.asarray(calib_screen_quad, dtype=np.float32).reshape(1, 4, 2), H_inv).reshape(4, 2)
    bx_l = float(np.clip(0.5 * (uv[0, 0] + uv[3, 0]), 0.0, 0.25))
    bx_r = float(np.clip(1.0 - 0.5 * (uv[1, 0] + uv[2, 0]), 0.0, 0.25))
    by_t = float(np.clip(0.5 * (uv[0, 1] + uv[1, 1]), 0.0, 0.25))
    by_b = float(np.clip(1.0 - 0.5 * (uv[2, 1] + uv[3, 1]), 0.0, 0.25))
    return (round(bx_l, 5), round(bx_r, 5), round(by_t, 5), round(by_b, 5))


def fit_screen_plane_rgbd(
    det: TabletDetection,
    depth_m: np.ndarray,
    K: np.ndarray,
    *,
    ransac_iters: int = 200,
    inlier_tol_m: float = 0.004,
) -> dict[str, Any] | None:
    """基于 RealSense 对齐深度图 + 分割掩码做 RANSAC 3D 平面拟合，消除单目大倾角二义性。

    Returns:
        包含 `normal`（指向相机的单位法向量）、`d`（平面方程 n·X + d = 0）、
        `screen_corners_3d`（4×3 相机系下发光屏 3D 角点，单位米）、`rms_mm` 的字典。
    """
    h, w = depth_m.shape[:2]
    mask = det.mask
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    # 轻微内腐蚀避开边缘深度跳变飞点
    eroded = cv2.erode(mask, np.ones((15, 15), np.uint8), iterations=1)
    ys, xs = np.where((eroded > 127) & (depth_m > 0.12) & (depth_m < 2.2))
    if len(xs) < 60:
        return None
    if len(xs) > 4000:
        idx = np.linspace(0, len(xs) - 1, 4000, dtype=int)
        xs, ys = xs[idx], ys[idx]
    zs = depth_m[ys, xs].astype(np.float64)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    pts3d = np.stack([(xs - cx) * zs / fx, (ys - cy) * zs / fy, zs], axis=1)

    rng = np.random.default_rng(42)
    best_inliers = np.zeros(len(pts3d), dtype=bool)
    for _ in range(ransac_iters):
        tri = pts3d[rng.choice(len(pts3d), 3, replace=False)]
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        nn = float(np.linalg.norm(n))
        if nn < 1e-6:
            continue
        n /= nn
        d = -float(n @ tri[0])
        dist = np.abs(pts3d @ n + d)
        inl = dist < inlier_tol_m
        if int(inl.sum()) > int(best_inliers.sum()):
            best_inliers = inl

    if int(best_inliers.sum()) < 30:
        return None
    inlier_pts = pts3d[best_inliers]
    centroid = inlier_pts.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
    normal = vh[-1]
    if normal[2] > 0:  # 统一朝向相机（-Z 方向）
        normal = -normal
    d = -float(normal @ centroid)
    rms_mm = float(np.sqrt(np.mean((inlier_pts @ normal + d) ** 2)) * 1000.0)

    # 将 4 个 2D 发光屏角点射线投射到拟合出的 3D 平面上
    corners_3d = []
    for u, v in det.screen_quad:
        ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
        denom = float(normal @ ray)
        if abs(denom) < 1e-6:
            return None
        t = -d / denom
        corners_3d.append(t * ray)
    return {
        "normal": normal,
        "d": d,
        "screen_corners_3d": np.array(corners_3d, dtype=np.float64),
        "rms_mm": round(rms_mm, 3),
        "inliers": int(best_inliers.sum()),
    }


class TabletSegTracker:
    """无 Tag 码实时平板分割 + 4 角点追踪 + 透视拉正封装。"""

    def __init__(
        self,
        weights: str | pathlib.Path,
        *,
        bezel_inset: tuple[float, float, float, float] = (0.040, 0.040, 0.056, 0.056),
        conf: float = 0.35,
        imgsz: int = 640,
        device: str | int = 0,
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.bezel_inset = bezel_inset
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

    def detect(self, frame_bgr: np.ndarray) -> TabletDetection | None:
        h, w = frame_bgr.shape[:2]
        res = self.model.predict(frame_bgr, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False, retina_masks=True)
        if not res or res[0].masks is None or len(res[0].masks.xy) == 0:
            return None
        confs = res[0].boxes.conf.cpu().numpy()
        best_i = int(np.argmax(confs))
        # 使用已严格逆 letterbox 还原到原图 (w, h) 的高精度掩码 / 多边形（避免 letterbox 纵向偏移！）
        poly_xy = res[0].masks.xy[best_i]
        if poly_xy is not None and len(poly_xy) >= 4:
            m = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(m, [np.round(poly_xy).astype(np.int32)], 255)
        else:
            m = (res[0].masks.data[best_i].cpu().numpy() > 0.5).astype(np.uint8) * 255
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        outer_quad = mask_to_quad_corners(m)
        if outer_quad is None:
            return None
        screen_quad = inset_quad_projective(outer_quad, self.bezel_inset)
        return TabletDetection(conf=float(confs[best_i]), mask=m, outer_quad=outer_quad, screen_quad=screen_quad)

    def rectify_screen(
        self,
        frame_bgr: np.ndarray,
        out_size: tuple[int, int] = (1280, 800),
    ) -> tuple[np.ndarray, TabletDetection] | None:
        """一步完成无 Tag 平板检测 + 内圈发光屏拉正，直接输出正视 RGB 图供 9B Agent 使用。"""
        det = self.detect(frame_bgr)
        if det is None:
            return None
        ow, oh = out_size
        dst = np.array([[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(det.screen_quad.astype(np.float32), dst)
        rect = cv2.warpPerspective(frame_bgr, H, (ow, oh), flags=cv2.INTER_LINEAR)
        return rect, det
