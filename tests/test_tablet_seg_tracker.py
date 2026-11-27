"""无 Tag 平板分割与 4 角点 / 3D 平面拟合单元测试（不连硬件、不需 GPU）。"""
from __future__ import annotations

import math

import cv2
import numpy as np

from tablet_tap.vision.tablet_seg_tracker import calibrate_bezel_from_calib_corners
from tablet_tap.vision.tablet_seg_tracker import fit_screen_plane_rgbd
from tablet_tap.vision.tablet_seg_tracker import inset_quad_projective
from tablet_tap.vision.tablet_seg_tracker import mask_to_quad_corners
from tablet_tap.vision.tablet_seg_tracker import order_quad
from tablet_tap.vision.tablet_seg_tracker import TabletDetection


def test_mask_to_quad_corners_steep_tilt_and_occlusion() -> None:
    """模拟 60° 陡倾角平板掩码 + 机械臂局部遮挡，四边直线拟合仍应亚像素还原 4 角点。"""
    h, w = 720, 1280
    gt = np.array([[420.0, 220.0], [910.0, 260.0], [990.0, 465.0], [340.0, 415.0]], dtype=np.float64)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [gt.astype(np.int32)], 255)
    # 在顶边制造一个局部机械臂遮挡缺口，检验抗遮挡直线拟合
    cv2.rectangle(mask, (610, 200), (690, 310), 0, -1)

    pred = mask_to_quad_corners(mask)
    assert pred is not None
    err = np.linalg.norm(pred - order_quad(gt), axis=1)
    assert float(err.max()) < 2.0, f"角点误差过大: {err}"


def test_bezel_calibration_roundtrip_projective() -> None:
    """在强透视梯形下，内缩与反标定应严格可逆（与倾角无关）。"""
    outer = np.array([[360.0, 210.0], [900.0, 235.0], [1020.0, 560.0], [260.0, 520.0]], dtype=np.float64)
    bezel = (0.042, 0.038, 0.055, 0.061)
    inner = inset_quad_projective(outer, bezel)
    est = calibrate_bezel_from_calib_corners(outer, inner)
    for a, b in zip(bezel, est):
        assert abs(a - b) < 1e-3, f"黑边反标定误差过大: {bezel} vs {est}"


def test_fit_screen_plane_rgbd_60deg_tilt() -> None:
    """合成 60° 倾斜平板的真实深度图 + 离群噪点，RANSAC 平面拟合应精准恢复 60° 法向量与 3D 角点。"""
    h, w = 480, 640
    K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    # 构造法向量与相机光轴夹角 60° 的平面：n = (0, sin60, -cos60), 中心点在 (0, 0, 0.70m)
    ang = math.radians(60.0)
    n_gt = np.array([0.0, math.sin(ang), -math.cos(ang)], dtype=np.float64)
    p0 = np.array([0.0, 0.0, 0.70], dtype=np.float64)
    d_gt = -float(n_gt @ p0)

    mask = np.zeros((h, w), dtype=np.uint8)
    quad = np.array([[200.0, 160.0], [440.0, 160.0], [460.0, 310.0], [180.0, 310.0]], dtype=np.float64)
    cv2.fillPoly(mask, [quad.astype(np.int32)], 255)

    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    rx = (xs - K[0, 2]) / K[0, 0]
    ry = (ys - K[1, 2]) / K[1, 1]
    denom = n_gt[0] * rx + n_gt[1] * ry + n_gt[2]
    depth = (-d_gt / denom).astype(np.float32)
    # 注入 0.5mm 高斯噪声与 5% 飞点离群值
    rng = np.random.default_rng(7)
    depth += rng.normal(0.0, 0.0005, depth.shape).astype(np.float32)
    outliers = rng.random(depth.shape) < 0.05
    depth[outliers] += 0.15

    det = TabletDetection(conf=0.98, mask=mask, outer_quad=quad, screen_quad=quad)
    res = fit_screen_plane_rgbd(det, depth, K)
    assert res is not None
    cos_sim = abs(float(res["normal"] @ n_gt))
    angle_err_deg = math.degrees(math.acos(min(1.0, cos_sim)))
    assert angle_err_deg < 0.5, f"60° 平面法向量误差过大: {angle_err_deg:.3f}°"
    assert res["rms_mm"] < 1.0
    assert res["screen_corners_3d"].shape == (4, 3)
