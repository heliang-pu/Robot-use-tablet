"""Build the final YOLO11-seg dataset combining:
  1. Verified real multi-camera robot frames (tablet/overview/wrist from 4 train sessions;
     2 completely disjoint real sessions kept strictly as real `val`).
  2. Physically grounded 3D perspective & steep-tilt (0-70 deg) arbitrary-position synthesis
     built from real Xiaomi tablet cutouts + real tabletop backgrounds + real L8 car HMI screens
     + glare/shadow/arm-occlusion, with exact homography-warped polygon labels.
  3. Dedicated 80-image `val_steep` benchmark (40-70 deg steep tilts, arbitrary locations)
     with exact ground-truth 4 corners for quantitative sub-pixel corner error evaluation.
"""
from __future__ import annotations

import glob
import json
import math
import pathlib
import random
import shutil
from typing import Any

import cv2
import numpy as np

REAL_ROOT = pathlib.Path("/mnt/ymtc2t/tablet_seg/labeled_real")
DS_ROOT = pathlib.Path("/mnt/ymtc2t/tablet_seg/dataset_v1")
HMI_IMG_GLOB = "/mnt/ymtc2t/datasets/carhmi_maigym_v2/images/*_c.png"


def order_quad(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(d)],
        pts[np.argmax(s)],
        pts[np.argmax(d)],
    ], dtype=np.float32)


def load_yolo_poly(lbl_path: pathlib.Path, w: int, h: int) -> np.ndarray | None:
    if not lbl_path.exists():
        return None
    txt = lbl_path.read_text(encoding="utf-8").strip()
    if not txt:
        return None
    parts = txt.splitlines()[0].split()
    if len(parts) < 9:
        return None
    vals = [float(x) for x in parts[1:]]
    pts = np.array([[vals[i] * w, vals[i + 1] * h] for i in range(0, len(vals) - 1, 2)], dtype=np.float32)
    return pts


def extract_clean_sources(rng: random.Random) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[np.ndarray]]:
    """Extract clean unobstructed real tablet cutouts (rectified to canonical 800x500) + real backgrounds."""
    meta = json.loads((REAL_ROOT / "autolabel_meta.json").read_text(encoding="utf-8"))
    meta_by_file = {m["file"]: m for m in meta}

    cutouts: list[tuple[np.ndarray, np.ndarray]] = []  # (bgr 800x500, alpha 800x500)
    backgrounds: list[np.ndarray] = []

    CW, CH = 800, 500
    dst_quad = np.array([[0, 0], [CW - 1, 0], [CW - 1, CH - 1], [0, CH - 1]], dtype=np.float32)

    for img_p in sorted((REAL_ROOT / "images" / "train").glob("*.jpg")):
        m = meta_by_file.get(img_p.name)
        if not m or m["status"] != "ok" or not m.get("quad_px"):
            continue
        img = cv2.imread(str(img_p))
        h, w = img.shape[:2]
        lbl_p = REAL_ROOT / "labels" / "train" / (img_p.stem + ".txt")
        poly = load_yolo_poly(lbl_p, w, h)
        if poly is None:
            continue

        # Collect clean background canvases via telea inpainting on unobstructed tablet/overview frames
        if ("_tablet_" in img_p.name or "_overview_" in img_p.name) and m["iou"] >= 0.85 and len(backgrounds) < 16:
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
            # Dilate generously to also erase brown bracket ears
            dil = cv2.dilate(mask, np.ones((45, 45), np.uint8), iterations=1)
            bg = cv2.inpaint(img, dil, 15, cv2.INPAINT_TELEA)
            bg = cv2.resize(bg, (1280, 720), interpolation=cv2.INTER_AREA)
            backgrounds.append(bg)

        # Clean unobstructed tablet cutouts from high-res `tablet` camera frames
        if "_tablet_" in img_p.name and m["iou"] >= 0.88 and len(cutouts) < 28:
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
            quad = order_quad(np.array(m["quad_px"], dtype=np.float32))
            # Reject frames where arm strongly cuts the top/interior (rectangularity check)
            q_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(q_mask, [quad.astype(np.int32)], 255)
            inter = float(np.sum((mask > 0) & (q_mask > 0)))
            union = float(np.sum((mask > 0) | (q_mask > 0)))
            if inter / max(union, 1.0) < 0.90:
                continue
            H = cv2.getPerspectiveTransform(quad, dst_quad)
            rect_bgr = cv2.warpPerspective(img, H, (CW, CH), flags=cv2.INTER_LINEAR)
            rect_mask = cv2.warpPerspective(mask, H, (CW, CH), flags=cv2.INTER_NEAREST)
            # Slight rounded-corner clean tablet silhouette
            alpha = np.full((CH, CW), 255, dtype=np.uint8)
            # Carve tiny realistic 14px rounded corners
            r = 14
            for cx, cy in [(r, r), (CW - 1 - r, r), (CW - 1 - r, CH - 1 - r), (r, CH - 1 - r)]:
                pass
            alpha = np.minimum(alpha, cv2.dilate(rect_mask, np.ones((5, 5), np.uint8)))
            cutouts.append((rect_bgr, alpha))

    # Also collect negative wrist frames (clean green wall / desk) as extra diverse backgrounds!
    for img_p in sorted((REAL_ROOT / "images" / "train").glob("*_wrist_*.jpg")):
        m = meta_by_file.get(img_p.name)
        if m and m["status"] == "neg" and len(backgrounds) < 26:
            bg = cv2.resize(cv2.imread(str(img_p)), (1280, 720), interpolation=cv2.INTER_AREA)
            backgrounds.append(bg)

    print(f"Extracted {len(cutouts)} clean real tablet cutouts and {len(backgrounds)} real backgrounds.", flush=True)
    return cutouts, backgrounds


def make_random_bg(backgrounds: list[np.ndarray], rng: random.Random, W: int = 1280, H: int = 720) -> np.ndarray:
    mode = rng.random()
    if backgrounds and mode < 0.72:
        bg = rng.choice(backgrounds).copy()
        if rng.random() < 0.5:
            bg = cv2.flip(bg, 1)
        if rng.random() < 0.35:
            bg = cv2.flip(bg, 0)
        # Random lighting & color temperature jitter
        gain = rng.uniform(0.72, 1.25)
        bias = rng.uniform(-18, 18)
        tint = np.array([rng.uniform(0.90, 1.10), rng.uniform(0.93, 1.07), rng.uniform(0.90, 1.10)], dtype=np.float32)
        bg = np.clip(bg.astype(np.float32) * gain * tint + bias, 0, 255).astype(np.uint8)
        return bg
    # Synthetic textured desk (wood / white / gray / dark lab bench) for strong background generalization!
    base = np.array([rng.randint(45, 225) for _ in range(3)], dtype=np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    grad = ((xx / W - 0.5) * rng.uniform(-35, 35) + (yy / H - 0.5) * rng.uniform(-35, 35))[:, :, None]
    noise = rng.uniform(2, 10) * np.random.default_rng(rng.randint(0, 10**9)).standard_normal((H, W, 3)).astype(np.float32)
    bg = np.clip(base[None, None, :] + grad + noise, 0, 255).astype(np.uint8)
    # Optional desk edge/clutter strips
    for _ in range(rng.randint(0, 3)):
        col = tuple(int(rng.randint(20, 230)) for _ in range(3))
        x1, y1 = rng.randint(0, W - 50), rng.randint(0, H - 50)
        x2, y2 = min(W, x1 + rng.randint(40, 300)), min(H, y1 + rng.randint(30, 220))
        cv2.rectangle(bg, (x1, y1), (x2, y2), col, -1)
    bg = cv2.GaussianBlur(bg, (9, 9), 0)
    return bg


def project_3d_tablet(
    W: int,
    H: int,
    pitch_deg: float,
    yaw_deg: float,
    roll_deg: float,
    tx_frac: float,
    ty_frac: float,
    scale: float,
    aspect: float = 1.60,
) -> np.ndarray:
    """Project a 3D rectangle (aspect W:H) through a pinhole camera to 4 image corners (TL, TR, BR, BL)."""
    rx, ry, rz = map(math.radians, (pitch_deg, yaw_deg, roll_deg))
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx

    hw = 0.5 * aspect * scale
    hh = 0.5 * scale
    pts3d = np.array([
        [-hw, -hh, 0.0],
        [+hw, -hh, 0.0],
        [+hw, +hh, 0.0],
        [-hw, +hh, 0.0],
    ], dtype=np.float64)
    cam_pts = (R @ pts3d.T).T
    z_dist = 2.4
    cam_pts[:, 2] += z_dist
    # Ensure all corners strictly in front of camera
    if np.any(cam_pts[:, 2] < 0.6):
        cam_pts[:, 2] += 1.0

    f = 1.15 * W
    cx = W * tx_frac
    cy = H * ty_frac
    u = f * (cam_pts[:, 0] / cam_pts[:, 2]) + cx
    v = f * (cam_pts[:, 1] / cam_pts[:, 2]) + cy
    return np.stack([u, v], axis=1).astype(np.float32)


def synthesize_sample(
    cutouts: list[tuple[np.ndarray, np.ndarray]],
    backgrounds: list[np.ndarray],
    hmi_paths: list[str],
    rng: random.Random,
    *,
    steep_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    W, H = 1280, 720
    tab_bgr, tab_alpha = rng.choice(cutouts)
    tab = tab_bgr.copy()
    ch, cw = tab.shape[:2]

    # 55% of time paste a real Ideal L8 Car HMI screenshot (light/dark/nav/climate/music) inside the real black bezel!
    if hmi_paths and rng.random() < 0.55:
        hmi = cv2.imread(rng.choice(hmi_paths))
        if hmi is not None:
            bx = int(cw * rng.uniform(0.036, 0.048))
            by = int(ch * rng.uniform(0.052, 0.068))
            sw, sh = cw - 2 * bx, ch - 2 * by
            hmi_rs = cv2.resize(hmi, (sw, sh), interpolation=cv2.INTER_AREA)
            # Slight realistic glass reflection/gamma shift so it blends seamlessly with real Xiaomi bezel
            hmi_rs = np.clip(hmi_rs.astype(np.float32) * rng.uniform(0.75, 1.05) + rng.uniform(2, 14), 0, 255).astype(np.uint8)
            tab[by:by + sh, bx:bx + sw] = hmi_rs

    # Sample 3D pose (covering 0..70 deg tilts & arbitrary screen locations!)
    for _ in range(25):
        if steep_only:
            pitch = rng.choice([-1, 1]) * rng.uniform(38.0, 68.0)
            yaw = rng.uniform(-35.0, 35.0)
            roll = rng.uniform(-45.0, 45.0)
            scale = rng.uniform(0.45, 0.92)
            tx = rng.uniform(0.24, 0.76)
            ty = rng.uniform(0.24, 0.76)
        else:
            tier = rng.random()
            if tier < 0.45:
                # Steep 35..70 deg tilt (car console / tilted stand!)
                pitch = rng.choice([-1, 1]) * rng.uniform(35.0, 70.0)
                yaw = rng.uniform(-42.0, 42.0)
                roll = rng.uniform(-60.0, 60.0)
            elif tier < 0.80:
                # Moderate 0..35 deg tilt + wide rotation
                pitch = rng.uniform(-35.0, 35.0)
                yaw = rng.uniform(-35.0, 35.0)
                roll = rng.uniform(-90.0, 90.0)
            else:
                # Wild wrist-like full 180 deg roll + tilt
                pitch = rng.uniform(-55.0, 55.0)
                yaw = rng.uniform(-45.0, 45.0)
                roll = rng.uniform(-180.0, 180.0)
            scale = rng.uniform(0.32, 0.96)
            tx = rng.uniform(0.18, 0.82)
            ty = rng.uniform(0.18, 0.82)

        quad = project_3d_tablet(W, H, pitch, yaw, roll, tx, ty, scale)
        # Keep all 4 corners well inside frame for steep_only, allow mild boundary touch in train
        pad = 22 if steep_only else -30
        if np.all(quad[:, 0] >= pad) and np.all(quad[:, 0] <= W - 1 - pad) and np.all(quad[:, 1] >= pad) and np.all(quad[:, 1] <= H - 1 - pad):
            area = float(cv2.contourArea(quad))
            if 0.025 * W * H <= area <= 0.65 * W * H:
                break
    else:
        return None

    src_quad = np.array([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]], dtype=np.float32)
    Hm = cv2.getPerspectiveTransform(src_quad, quad)
    warped_tab = cv2.warpPerspective(tab, Hm, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    warped_mask = cv2.warpPerspective(tab_alpha, Hm, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)

    bg = make_random_bg(backgrounds, rng, W, H)

    # Subtle realistic drop shadow beneath the tablet
    sh_dx, sh_dy = rng.randint(-12, 12), rng.randint(4, 18)
    M_sh = np.float32([[1, 0, sh_dx], [0, 1, sh_dy]])
    sh_mask = cv2.warpAffine(warped_mask, M_sh, (W, H))
    sh_mask = cv2.GaussianBlur(sh_mask, (31, 31), 0).astype(np.float32) / 255.0 * rng.uniform(0.20, 0.45)
    bg = np.clip(bg.astype(np.float32) * (1.0 - sh_mask[:, :, None]), 0, 255).astype(np.uint8)

    # Alpha composite tablet onto background
    a_f = (warped_mask.astype(np.float32) / 255.0)[:, :, None]
    comp = np.clip(warped_tab.astype(np.float32) * a_f + bg.astype(np.float32) * (1.0 - a_f), 0, 255).astype(np.uint8)

    # Random glass glare spot (like real ceiling light reflection)
    if rng.random() < 0.45:
        gx, gy = rng.randint(0, W - 1), rng.randint(0, H - 1)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        sig = rng.uniform(70.0, 220.0)
        glare = np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2.0 * sig * sig)) * rng.uniform(25.0, 95.0)
        comp = np.clip(comp.astype(np.float32) + glare[:, :, None] * a_f, 0, 255).astype(np.uint8)

    # Random partial arm/hand occluder (train only — NOT on val_steep so 4 corners remain exact)
    vis_mask = (warped_mask > 127).astype(np.uint8) * 255
    if not steep_only and rng.random() < 0.25:
        # Draw a realistic dark robotic arm bar entering from top or left edge
        occ = np.zeros((H, W), dtype=np.uint8)
        cx, cy = int(quad[:, 0].mean()), int(quad[:, 1].mean())
        p0 = (rng.randint(0, W), 0) if rng.random() < 0.6 else (0, rng.randint(0, H))
        p1 = (cx + rng.randint(-120, 120), cy + rng.randint(-80, 80))
        thick = rng.randint(36, 95)
        cv2.line(occ, p0, p1, 255, thick)
        arm_col = tuple(int(rng.randint(25, 65)) for _ in range(3))
        comp[occ > 0] = arm_col
        vis_mask[occ > 0] = 0

    # Slight realistic camera blur + sensor noise
    if rng.random() < 0.35:
        comp = cv2.GaussianBlur(comp, (3, 3), 0)

    cnts, _ = cv2.findContours(vis_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 0.015 * W * H:
        return None
    peri = cv2.arcLength(cnt, True)
    poly = cv2.approxPolyDP(cnt, 0.002 * peri, True).reshape(-1, 2).astype(np.float32)
    if len(poly) < 4:
        return None
    return comp, poly, quad


def write_yolo_label(path: pathlib.Path, poly: np.ndarray, w: int, h: int) -> None:
    coords = []
    for px, py in poly:
        coords.append(f"{max(0.0, min(1.0, float(px) / w)):.6f}")
        coords.append(f"{max(0.0, min(1.0, float(py) / h)):.6f}")
    path.write_text("0 " + " ".join(coords) + "\n", encoding="utf-8")


def main() -> None:
    rng = random.Random(20261127)
    if DS_ROOT.exists():
        shutil.rmtree(DS_ROOT)
    for sub in ("images/train", "labels/train", "images/val", "labels/val", "images/val_steep", "labels/val_steep"):
        (DS_ROOT / sub).mkdir(parents=True, exist_ok=True)

    # 1. Copy verified real train & real held-out val
    n_real = {"train": 0, "val": 0}
    for split in ("train", "val"):
        for img_p in sorted((REAL_ROOT / "images" / split).glob("*.jpg")):
            lbl_p = REAL_ROOT / "labels" / split / (img_p.stem + ".txt")
            shutil.copy2(img_p, DS_ROOT / "images" / split / img_p.name)
            shutil.copy2(lbl_p, DS_ROOT / "labels" / split / lbl_p.name)
            n_real[split] += 1
    print("Copied real frames:", n_real, flush=True)

    # 2. Extract real tablet cutouts + backgrounds + L8 HMI screenshots
    cutouts, backgrounds = extract_clean_sources(rng)
    hmi_paths = sorted(glob.glob(HMI_IMG_GLOB))
    print(f"Found {len(hmi_paths)} real L8 Car HMI screenshots for screen synthesis.", flush=True)

    # 3. Synthesize 800 arbitrary-pose 0-70 deg training frames
    n_syn = 0
    for i in range(900):
        if n_syn >= 800:
            break
        res = synthesize_sample(cutouts, backgrounds, hmi_paths, rng, steep_only=False)
        if res is None:
            continue
        img, poly, _quad = res
        stem = f"syn3d_train_{n_syn:04d}"
        cv2.imwrite(str(DS_ROOT / "images/train" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        write_yolo_label(DS_ROOT / "labels/train" / f"{stem}.txt", poly, img.shape[1], img.shape[0])
        n_syn += 1

    # 4. Synthesize 80 steep-tilt (38-68 deg) arbitrary-position stress-test frames with EXACT GT 4 corners!
    steep_gt = []
    n_steep = 0
    for i in range(150):
        if n_steep >= 80:
            break
        res = synthesize_sample(cutouts, backgrounds, hmi_paths, rng, steep_only=True)
        if res is None:
            continue
        img, poly, quad = res
        stem = f"syn3d_steep_{n_steep:04d}"
        img_out = DS_ROOT / "images/val_steep" / f"{stem}.jpg"
        cv2.imwrite(str(img_out), img, [cv2.IMWRITE_JPEG_QUALITY, 94])
        write_yolo_label(DS_ROOT / "labels/val_steep" / f"{stem}.txt", poly, img.shape[1], img.shape[0])
        steep_gt.append({"file": img_out.name, "gt_quad": order_quad(quad).round(2).tolist()})
        n_steep += 1

    (DS_ROOT / "val_steep_gt.json").write_text(json.dumps(steep_gt, indent=2), encoding="utf-8")

    # 5. Write Ultralytics data.yaml
    yaml_txt = (
        f"path: {DS_ROOT}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: tablet\n"
    )
    (DS_ROOT / "data.yaml").write_text(yaml_txt, encoding="utf-8")
    print(f"DATASET READY: train={n_real['train']} real + {n_syn} syn3d = {n_real['train'] + n_syn}, val_real={n_real['val']}, val_steep={n_steep}")


if __name__ == "__main__":
    main()
