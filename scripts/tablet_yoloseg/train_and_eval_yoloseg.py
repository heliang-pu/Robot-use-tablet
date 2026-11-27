import json, pathlib, shutil, time
import cv2
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
import tablet_seg_tracker as tst

t0 = time.monotonic()
ds = pathlib.Path("/mnt/ymtc2t/tablet_seg/dataset_v1")
model = YOLO("/mnt/ymtc2t/tablet_seg/weights/yolo11s-seg.pt")

print("=== [1/3] Training YOLO11s-seg on 1,091 real+3D-pose images (100 epochs) ===", flush=True)
res = model.train(
    data=str(ds / "data.yaml"),
    epochs=80,
    imgsz=640,
    batch=16,
    device=0,
    workers=4,
    patience=20,
    cos_lr=True,
    degrees=25.0,
    translate=0.15,
    scale=0.45,
    shear=8.0,
    perspective=0.0006,
    fliplr=0.5,
    hsv_h=0.02,
    hsv_s=0.45,
    hsv_v=0.45,
    mosaic=0.8,
    close_mosaic=15,
    project="/mnt/ymtc2t/tablet_seg/runs",
    name="yolo11s_tablet_v1",
    exist_ok=True,
    verbose=True,
)
best_pt = pathlib.Path("/mnt/ymtc2t/tablet_seg/runs/yolo11s_tablet_v1/weights/best.pt")
dst_pt = pathlib.Path("/mnt/ymtc2t/models/tablet_yoloseg/yolo11s_tablet_v1.pt")
shutil.copy2(best_pt, dst_pt)
print(f"Saved best weights -> {dst_pt} (train elapsed {time.monotonic()-t0:.1f}s)", flush=True)

print("=== [2/3] Quantitative Eval on Held-Out Real Sessions (val) & 38-68 deg Steep Tilt (val_steep) ===", flush=True)
tracker = tst.TabletSegTracker(dst_pt, conf=0.35, imgsz=640, device=0)

# Speed benchmark
sample_img = cv2.imread(str(sorted((ds / "images/val").glob("*.jpg"))[0]))
for _ in range(5):
    tracker.detect(sample_img)
ts = []
for _ in range(30):
    t1 = time.monotonic()
    tracker.detect(sample_img)
    ts.append((time.monotonic() - t1) * 1000.0)
print(f"Inference speed (YOLO11s-seg + 4-edge Huber fit): mean={np.mean(ts):.2f} ms, p95={np.percentile(ts, 95):.2f} ms", flush=True)

# 1. Real held-out val corner error & mask IoU
meta = json.loads(pathlib.Path("/mnt/ymtc2t/tablet_seg/labeled_real/autolabel_meta.json").read_text(encoding="utf-8"))
meta_by_file = {m["file"]: m for m in meta}

real_ious = []
real_corner_errs_unocc = []
real_tp = real_fp = real_fn = real_tn = 0
vis_real = []

for ip in sorted((ds / "images/val").glob("*.jpg")):
    im = cv2.imread(str(ip))
    h, w = im.shape[:2]
    lp = ds / "labels/val" / (ip.stem + ".txt")
    txt = lp.read_text().strip()
    has_gt = bool(txt)
    det = tracker.detect(im)
    if has_gt and det is not None:
        real_tp += 1
        vals = [float(x) for x in txt.split()[1:]]
        gt_poly = np.array([[int(vals[i]*w), int(vals[i+1]*h)] for i in range(0, len(vals)-1, 2)], dtype=np.int32)
        gt_m = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(gt_m, [gt_poly], 255)
        inter = float(np.sum((gt_m > 0) & (det.mask > 0)))
        union = float(np.sum((gt_m > 0) | (det.mask > 0)))
        iou = inter / max(union, 1.0)
        real_ious.append(iou)
        # Corner error on clean unobstructed tablet/overview frames
        m_info = meta_by_file.get(ip.name)
        if m_info and m_info.get("quad_px") and ("_tablet_" in ip.name or "_overview_" in ip.name) and m_info["iou"] >= 0.90:
            gt_q = tst.order_quad(np.array(m_info["quad_px"], dtype=np.float64))
            err = np.linalg.norm(det.outer_quad - gt_q, axis=1)
            real_corner_errs_unocc.extend(err.tolist())
        vis_real.append((ip, det, f"IoU={iou:.3f}"))
    elif has_gt and det is None:
        real_fn += 1
        vis_real.append((ip, None, "FN"))
    elif not has_gt and det is not None:
        real_fp += 1
        vis_real.append((ip, det, "FP"))
    else:
        real_tn += 1
        vis_real.append((ip, None, "TN"))

print(f"REAL HELD-OUT VAL (73 frames): TP={real_tp} FN={real_fn} TN={real_tn} FP={real_fp} | mean_mask_IoU={np.mean(real_ious):.4f} (p10={np.percentile(real_ious,10):.4f})", flush=True)
if real_corner_errs_unocc:
    print(f"REAL UNOCCLUDED 4-CORNER ERR (px): median={np.median(real_corner_errs_unocc):.2f}px, mean={np.mean(real_corner_errs_unocc):.2f}px, p95={np.percentile(real_corner_errs_unocc,95):.2f}px", flush=True)

# 2. Steep 38-68 deg arbitrary-position exact-GT 4-corner benchmark!
steep_gt = json.loads((ds / "val_steep_gt.json").read_text(encoding="utf-8"))
steep_errs = []
steep_ious = []
steep_hit = 0
vis_steep = []
for row in steep_gt:
    ip = ds / "images/val_steep" / row["file"]
    im = cv2.imread(str(ip))
    h, w = im.shape[:2]
    gt_q = tst.order_quad(np.array(row["gt_quad"], dtype=np.float64))
    det = tracker.detect(im)
    if det is not None:
        steep_hit += 1
        # Match corners allowing 4 cyclic rotations (since arbitrary roll up to 45 deg)
        best_err = 1e9
        for shift in range(4):
            cand = np.roll(det.outer_quad, shift, axis=0)
            e = float(np.mean(np.linalg.norm(cand - gt_q, axis=1)))
            if e < best_err:
                best_err = e
                best_e_arr = np.linalg.norm(cand - gt_q, axis=1)
        steep_errs.extend(best_e_arr.tolist())
        gt_m = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(gt_m, [gt_q.astype(np.int32)], 255)
        inter = float(np.sum((gt_m > 0) & (det.mask > 0)))
        union = float(np.sum((gt_m > 0) | (det.mask > 0)))
        iou = inter / max(union, 1.0)
        steep_ious.append(iou)
        vis_steep.append((ip, det, f"err={best_err:.1f}px IoU={iou:.3f}"))
    else:
        vis_steep.append((ip, None, "MISS"))

print(f"STEEP 38-68 DEG ARBITRARY-POSE VAL ({len(steep_gt)} frames): detect={steep_hit}/{len(steep_gt)} ({100*steep_hit/len(steep_gt):.1f}%) | mean_mask_IoU={np.mean(steep_ious):.4f}", flush=True)
print(f"STEEP 38-68 DEG 4-CORNER ERR (px @1280x720): median={np.median(steep_errs):.2f}px, mean={np.mean(steep_errs):.2f}px, p95={np.percentile(steep_errs,95):.2f}px", flush=True)

# 3. Render prediction visualization mosaics (showing outer red corners + inner cyan screen quad!)
def save_pred_grid(items, out_p, max_n=20, cols=4):
    sel = items if len(items) <= max_n else [items[i] for i in np.linspace(0, len(items)-1, max_n, dtype=int)]
    rows = (len(sel) + cols - 1) // cols
    tw, th = 420, 260
    canvas = Image.new("RGB", (tw * cols, th * rows), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    for k, (ip, det, label) in enumerate(sel):
        ri, ci = divmod(k, cols)
        im = cv2.imread(str(ip))
        if det is not None:
            ov = im.copy()
            ov[det.mask > 127] = (0, 210, 80)
            im = cv2.addWeighted(im, 0.72, ov, 0.28, 0)
            cv2.polylines(im, [det.outer_quad.astype(np.int32)], True, (0, 255, 255), 3)
            cv2.polylines(im, [det.screen_quad.astype(np.int32)], True, (255, 220, 0), 2)
            for idx_p, (px, py) in enumerate(det.outer_quad):
                cv2.circle(im, (int(round(px)), int(round(py))), 8, (0, 0, 255), -1)
        pim = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        pim.thumbnail((tw - 6, th - 24))
        canvas.paste(pim, (ci * tw + 3, ri * th + 20))
        draw.text((ci * tw + 4, ri * th + 3), f"{ip.stem} {label}"[:48], fill=(255, 255, 0))
    canvas.save(out_p, quality=86)
    print("Saved pred grid ->", out_p, flush=True)

save_pred_grid(vis_real, pathlib.Path("/mnt/ymtc2t/tablet_seg/pred_val_real_grid.jpg"), max_n=20, cols=4)
save_pred_grid(vis_steep, pathlib.Path("/mnt/ymtc2t/tablet_seg/pred_val_steep_grid.jpg"), max_n=16, cols=4)
print(f"=== TRAIN_AND_EVAL_ALL_DONE in {time.monotonic()-t0:.1f}s ===", flush=True)
