"""Auto-label real multi-camera tablet frames using Qwen3.5 VLM (guided JSON bbox) + SAM2.1-Base.

Outputs standard YOLO-seg dataset labels (`0 x1 y1 x2 y2 ...`) plus visual verification contact sheets
so Gemini can visually audit label quality before YOLO11s-seg training.
"""
from __future__ import annotations

import base64
import concurrent.futures
import io
import json
import pathlib
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw
import requests

RAW_ROOT = pathlib.Path("/mnt/ymtc2t/tablet_seg/raw_frames")
OUT_ROOT = pathlib.Path("/mnt/ymtc2t/tablet_seg/labeled_real")
WEIGHTS_DIR = pathlib.Path("/mnt/ymtc2t/tablet_seg/weights")

VLM_URL = "http://127.0.0.1:18809/v1/chat/completions"
VLM_MODEL = "mai-gym-9b-v3-s273"

BBOX_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "visible": {"type": "boolean"},
        "bbox": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 4,
            "maxItems": 4,
        },
    },
    "required": ["visible", "bbox"],
}

PROMPT_TEXT = (
    "请判断画面中是否能看到黑色矩形平板电脑（屏幕可能亮着桌面/音乐/白底网页，也可能熄屏或被机械臂局部遮挡，可能倾斜或被画面边缘截断）。\n"
    "若画面里能看到平板电脑，设 visible=true，并给出紧紧包住整块平板机身（含黑边框）的边界框 bbox=[x1, y1, x2, y2]（0到1000归一化整数坐标，左上到右下，不要把上方机械臂或桌布框进去）。\n"
    "若画面里看不到平板电脑（如只看到桌布、绿幕或机械臂），设 visible=false，bbox=[0,0,0,0]。"
)


def encode_jpg_b64(img_bgr: np.ndarray, max_w: int = 960) -> str:
    h, w = img_bgr.shape[:2]
    if w > max_w:
        nh = int(round(h * max_w / w))
        img_bgr = cv2.resize(img_bgr, (max_w, nh), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def query_vlm_bbox(img_path: pathlib.Path) -> tuple[pathlib.Path, dict[str, Any]]:
    img_bgr = cv2.imread(str(img_path))
    b64 = encode_jpg_b64(img_bgr)
    sess = requests.Session()
    sess.trust_env = False
    body = {
        "model": VLM_MODEL,
        "messages": [
            {"role": "system", "content": "你是精准的目标检测标注员，严格按 JSON Schema 输出。"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_TEXT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            },
        ],
        "temperature": 0.0,
        "max_tokens": 64,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "tablet_bbox", "strict": True, "schema": BBOX_SCHEMA},
        },
        "chat_template_kwargs": {"enable_thinking": False},
    }
    for _ in range(3):
        try:
            r = sess.post(VLM_URL, json=body, timeout=60)
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"]
            obj = json.loads(txt)
            return img_path, obj
        except Exception as e:
            last_err = e
            time.sleep(0.5)
    return img_path, {"visible": False, "bbox": [0, 0, 0, 0], "error": str(last_err)}


def box_iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - inter
    return inter / max(ua, 1e-6)


def mask_to_quad_corners(mask_u8: np.ndarray) -> np.ndarray | None:
    """Fit 4 crisp straight lines to the 4 edges of a tablet mask and intersect them."""
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 400:
        return None
    # Convex hull avoids small arm-occlusion bites when estimating the 4 straight sides
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    # Iteratively find a 4-vertex polygon approximation
    quad = None
    for eps_f in np.linspace(0.01, 0.08, 25):
        ap = cv2.approxPolyDP(hull, eps_f * peri, True)
        if len(ap) == 4:
            quad = ap.reshape(4, 2).astype(np.float64)
            break
    if quad is None:
        rect = cv2.minAreaRect(hull)
        quad = cv2.boxPoints(rect).astype(np.float64)

    # Order TL, TR, BR, BL
    s = quad.sum(axis=1)
    d = np.diff(quad, axis=1).ravel()
    ordered = np.array([
        quad[np.argmin(s)],
        quad[np.argmin(d)],
        quad[np.argmax(s)],
        quad[np.argmax(d)],
    ], dtype=np.float64)

    # Refine each of the 4 edges by Huber line fitting on nearby original contour points
    pts = cnt.reshape(-1, 2).astype(np.float64)
    lines = []
    for i in range(4):
        p1, p2 = ordered[i], ordered[(i + 1) % 4]
        v = p2 - p1
        elen = np.linalg.norm(v)
        if elen < 10:
            return ordered
        u = v / elen
        rel = pts - p1
        proj = rel @ u
        perp = np.abs(rel[:, 0] * (-u[1]) + rel[:, 1] * u[0])
        # Exclude rounded corner caps (keep middle 12%..88% of edge) & close points
        inliers = pts[(proj > 0.12 * elen) & (proj < 0.88 * elen) & (perp < max(6.0, 0.03 * elen))]
        if len(inliers) >= 12:
            vx, vy, x0, y0 = cv2.fitLine(inliers.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            # Line in normal form a*x + b*y + c = 0
            a, b = float(-vy), float(vx)
            c = -(a * float(x0) + b * float(y0))
        else:
            a, b = float(-u[1]), float(u[0])
            c = -(a * p1[0] + b * p1[1])
        lines.append((a, b, c))

    refined = []
    for i in range(4):
        l_prev = lines[(i - 1) % 4]
        l_curr = lines[i]
        det = l_prev[0] * l_curr[1] - l_curr[0] * l_prev[1]
        if abs(det) < 1e-6:
            refined.append(ordered[i])
        else:
            x = (l_prev[1] * l_curr[2] - l_curr[1] * l_prev[2]) / det
            y = (l_curr[0] * l_prev[2] - l_prev[0] * l_curr[2]) / det
            if np.linalg.norm(np.array([x, y]) - ordered[i]) > 60:
                refined.append(ordered[i])
            else:
                refined.append(np.array([x, y]))
    return np.array(refined, dtype=np.float64)


def draw_mosaic(records: list[dict[str, Any]], out_path: pathlib.Path, max_items: int = 24, cols: int = 4) -> None:
    if not records:
        return
    # Stratified sample across tablet/overview/wrist
    sel = records
    if len(records) > max_items:
        idxs = np.linspace(0, len(records) - 1, max_items, dtype=int)
        sel = [records[i] for i in idxs]
    rows = (len(sel) + cols - 1) // cols
    tw, th = 420, 260
    canvas = Image.new("RGB", (tw * cols, th * rows), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    for k, rec in enumerate(sel):
        r_i, c_i = divmod(k, cols)
        img = cv2.imread(rec["img_path"])
        h, w = img.shape[:2]
        vis = img.copy()
        if rec["has_mask"] and rec["poly_px"] is not None:
            overlay = vis.copy()
            poly_i = rec["poly_px"].astype(np.int32)
            cv2.fillPoly(overlay, [poly_i], (0, 220, 80))
            vis = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0)
            cv2.polylines(vis, [poly_i], True, (0, 255, 255), 3)
        if rec["vlm_box_px"] is not None:
            x1, y1, x2, y2 = [int(round(v)) for v in rec["vlm_box_px"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 160, 0), 2)
        if rec["quad_px"] is not None:
            qp = rec["quad_px"].astype(np.int32)
            for idx_p, (px, py) in enumerate(qp):
                cv2.circle(vis, (int(px), int(py)), 9, (0, 0, 255), -1)
                cv2.putText(vis, str(idx_p), (int(px) + 8, int(py) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        pil_im = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        pil_im.thumbnail((tw - 6, th - 24))
        canvas.paste(pil_im, (c_i * tw + 3, r_i * th + 20))
        tag = f"{pathlib.Path(rec['img_path']).stem} iou={rec['iou']:.2f} {rec['status']}"
        draw.text((c_i * tw + 4, r_i * th + 3), tag[:46], fill=(255, 255, 0))
    canvas.save(out_path, quality=85)
    print(f"Saved mosaic ({len(sel)} items) -> {out_path}")


def main() -> None:
    t0 = time.monotonic()
    all_imgs = []
    for split in ("train", "val"):
        for p in sorted((RAW_ROOT / split).glob("*.jpg")):
            all_imgs.append((split, p))
    print(f"[1/3] Running VLM bbox detection on {len(all_imgs)} real frames (8 workers)...", flush=True)

    vlm_res: dict[pathlib.Path, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(query_vlm_bbox, p) for _, p in all_imgs]
        for i, f in enumerate(concurrent.futures.as_completed(futs), 1):
            p, obj = f.result()
            vlm_res[p] = obj
            if i % 50 == 0 or i == len(all_imgs):
                print(f"  VLM {i}/{len(all_imgs)} ({time.monotonic()-t0:.1f}s)", flush=True)

    print(f"[2/3] Running SAM2.1-Base pixel segmentation on GPU...", flush=True)
    from ultralytics import SAM
    sam = SAM(str(WEIGHTS_DIR / "sam2.1_b.pt"))

    for split in ("train", "val"):
        (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    meta_rows = []
    split_records: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "suspect": []}

    for idx, (split, img_path) in enumerate(all_imgs, 1):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        obj = vlm_res[img_path]
        visible = bool(obj.get("visible"))
        bb = obj.get("bbox") or [0, 0, 0, 0]

        rec: dict[str, Any] = {
            "split": split,
            "img_path": str(img_path),
            "has_mask": False,
            "vlm_box_px": None,
            "poly_px": None,
            "quad_px": None,
            "iou": 0.0,
            "area_ratio": 0.0,
            "status": "neg",
        }

        if visible and len(bb) == 4:
            x1 = max(0.0, min(w - 1.0, bb[0] * w / 1000.0))
            y1 = max(0.0, min(h - 1.0, bb[1] * h / 1000.0))
            x2 = max(0.0, min(w - 1.0, bb[2] * w / 1000.0))
            y2 = max(0.0, min(h - 1.0, bb[3] * h / 1000.0))
            if x2 - x1 >= 25 and y2 - y1 >= 25:
                rec["vlm_box_px"] = [x1, y1, x2, y2]
                res = sam.predict(img, bboxes=[[x1, y1, x2, y2]], verbose=False)
                if res and res[0].masks is not None and len(res[0].masks.data) > 0:
                    m = (res[0].masks.data[0].cpu().numpy() > 0.5).astype(np.uint8)
                    if m.shape != (h, w):
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        cnt = max(cnts, key=cv2.contourArea)
                        area = float(cv2.contourArea(cnt))
                        area_ratio = area / float(w * h)
                        mx, my, mw, mh = cv2.boundingRect(cnt)
                        iou = box_iou([x1, y1, x2, y2], [mx, my, mx + mw, my + mh])
                        rec["iou"] = iou
                        rec["area_ratio"] = area_ratio
                        if area_ratio >= 0.008 and area_ratio <= 0.88 and iou >= 0.40:
                            peri = cv2.arcLength(cnt, True)
                            poly = cv2.approxPolyDP(cnt, 0.0022 * peri, True).reshape(-1, 2)
                            if len(poly) >= 4:
                                clean_m = np.zeros((h, w), dtype=np.uint8)
                                cv2.fillPoly(clean_m, [cnt], 255)
                                rec["has_mask"] = True
                                rec["poly_px"] = poly
                                rec["quad_px"] = mask_to_quad_corners(clean_m)
                                rec["status"] = "ok"
                        else:
                            rec["status"] = "suspect"
                else:
                    rec["status"] = "suspect"
            else:
                rec["status"] = "neg"

        # Copy image & write YOLO-seg label (only keep clean 'ok' or true 'neg'; skip 'suspect' from training)
        if rec["status"] in ("ok", "neg"):
            dst_img = OUT_ROOT / "images" / split / img_path.name
            dst_lbl = OUT_ROOT / "labels" / split / (img_path.stem + ".txt")
            if not dst_img.exists():
                cv2.imwrite(str(dst_img), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if rec["has_mask"] and rec["poly_px"] is not None:
                coords = []
                for px, py in rec["poly_px"]:
                    coords.append(f"{px / w:.6f}")
                    coords.append(f"{py / h:.6f}")
                dst_lbl.write_text("0 " + " ".join(coords) + "\n", encoding="utf-8")
            else:
                dst_lbl.write_text("", encoding="utf-8")
            split_records[split].append(rec)
        else:
            split_records["suspect"].append(rec)

        meta_rows.append({
            "split": split,
            "file": img_path.name,
            "status": rec["status"],
            "iou": round(rec["iou"], 3),
            "area_ratio": round(rec["area_ratio"], 4),
            "vlm_bbox": bb,
            "quad_px": rec["quad_px"].round(1).tolist() if rec["quad_px"] is not None else None,
        })
        if idx % 80 == 0 or idx == len(all_imgs):
            print(f"  SAM2 {idx}/{len(all_imgs)} ({time.monotonic()-t0:.1f}s)", flush=True)

    (OUT_ROOT / "autolabel_meta.json").write_text(json.dumps(meta_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    for k in ("train", "val"):
        n_ok = sum(1 for r in split_records[k] if r["status"] == "ok")
        n_neg = sum(1 for r in split_records[k] if r["status"] == "neg")
        print(f"SUMMARY {k:5s}: ok={n_ok}, neg={n_neg}")
    print(f"SUMMARY suspect(filtered): {len(split_records['suspect'])}")

    print("[3/3] Generating visual audit mosaics...", flush=True)
    draw_mosaic(split_records["train"], OUT_ROOT / "verify_train_grid.jpg", max_items=24, cols=4)
    draw_mosaic(split_records["val"], OUT_ROOT / "verify_val_grid.jpg", max_items=20, cols=4)
    if split_records["suspect"]:
        draw_mosaic(split_records["suspect"], OUT_ROOT / "verify_suspect_grid.jpg", max_items=16, cols=4)
    print(f"ALL DONE in {time.monotonic()-t0:.1f}s!")


if __name__ == "__main__":
    main()
