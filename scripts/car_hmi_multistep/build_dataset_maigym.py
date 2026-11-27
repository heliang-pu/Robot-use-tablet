"""Build L8 car HMI SFT dataset in exact Tongyi-MAI/MAI-Gym-9B format from dataset_v1_raw.

Reuses the exact same 1,500 random train screens and 150 val screens (holding out
settings:secure, settings:assistant, phone:recent, nav:route), near-test dedup,
camera-degradation + affine point warping, and control weighting as build_dataset.py,
formatted strictly to match run_bench_maigym.py / official MAIGymMobileUsePolicy:
  - System prompt: official MAI_GYM_SYSTEM_PROMPT
  - User turns: [text turn] + [image turn] (two user turns, matching official MAI-Gym)
  - Coordinate space: 0..999 (matching official MAI-Gym _scaled_point / 999)
  - Assistant answer:
      * think_mode="short": concise 1-line Chinese <think> + <tool_call> JSON
      * think_mode="empty": direct <tool_call> JSON (for enable_thinking=False)

Usage:
  python3 build_dataset_maigym.py /mnt/ymtc2t/datasets/carhmi_maigym_v1 \
    --train dataset_v1_raw/train --val dataset_v1_raw/val --per-screen 10 --think-mode short
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import sys
from typing import Any

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import build_dataset as bd  # noqa: E402
import run_bench as rb  # noqa: E402
import run_bench_maigym as mg  # noqa: E402
from tablet_tap.agent import by_steps as bs  # noqa: E402

W, H, FEED = rb.W, rb.H, rb.FEED
COORD_SCALE = 999.0


def make_sample(
    image_rel: str,
    ask: str,
    desc: str,
    cid: str,
    box: list[int],
    px: float,
    py: float,
    *,
    think_mode: str,
    with_note: bool,
    merged_user: bool,
) -> dict[str, Any]:
    instruction = f"{ask} ({bs._IN_APP})" if with_note else ask  # noqa: SLF001
    cx = int(round(min(max(px / FEED[0], 0.0), 1.0) * COORD_SCALE))
    cy = int(round(min(max(py / FEED[1], 0.0), 1.0) * COORD_SCALE))
    call = {
        "name": "mobile_use",
        "arguments": {"action_type": "click", "coordinate": [cx, cy]},
    }
    tool_str = f"<tool_call>\n{json.dumps(call, ensure_ascii=False)}\n</tool_call>"
    if think_mode == "short":
        loc = rb.where(box)
        think_str = f"需要点击{desc}，该控件位于{loc}，坐标约为 [{cx}, {cy}]。"
        assistant = f"<think>\n{think_str}\n</think>\n\n{tool_str}"
    else:
        assistant = tool_str

    if merged_user:
        messages = [
            {"role": "system", "content": mg.MAI_GYM_SYSTEM_PROMPT},
            {"role": "user", "content": f"{instruction}<image>"},
            {"role": "assistant", "content": assistant},
        ]
    else:
        messages = [
            {"role": "system", "content": mg.MAI_GYM_SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
            {"role": "user", "content": "<image>"},
            {"role": "assistant", "content": assistant},
        ]
    return {
        "messages": messages,
        "images": [image_rel],
        "meta": {"cid": cid, "desc": desc, "coord": [cx, cy]},
    }


def build_split(
    split: str,
    src: pathlib.Path,
    out: pathlib.Path,
    per: int,
    tests: list[np.ndarray],
    rng: random.Random,
    stats: dict[str, Any],
    *,
    think_mode: str,
    with_note: bool,
    merged_user: bool,
    reuse_images: pathlib.Path | None,
) -> None:
    meta = json.loads((src / "screens.json").read_text(encoding="utf-8"))
    (out / "images").mkdir(parents=True, exist_ok=True)
    with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
        for s in meta:
            img = cv2.imread(str(src / s["image"]))
            if min(float(np.mean(np.abs(bd.thumb(img) - t))) for t in tests) < 2.0:
                stats[split]["near_test_dropped"] += 1
                continue
            clean = cv2.resize(img, FEED, interpolation=cv2.INTER_AREA)
            cam, m = bd.camera(img, rng)
            names = {}
            for kind, pic in (("c", clean), ("k", cam)):
                rel = f"images/{s['screen']}_{kind}.png"
                dst = out / rel
                if kind == "c" and reuse_images is not None and (reuse_images / rel).exists():
                    if not dst.exists():
                        dst.symlink_to((reuse_images / rel).resolve())
                else:
                    cv2.imwrite(str(dst), pic)
                names[kind] = rel
            ctl = [c for c in s["controls"] if rb.describe(c[0], c[1])]
            weights = [
                0.35 if bd.GLOBAL.match(c[0]) else 3.5 if bd.ICON.match(c[0]) else 1.5
                for c in ctl
            ]
            chosen = []
            while ctl and len(chosen) < per:
                i = rng.choices(range(len(ctl)), weights)[0]
                chosen.append(ctl.pop(i))
                weights.pop(i)
            for cid, name, x, y, w, h in chosen:
                got = bd.instruction(rng, cid, name, [x, y, w, h])
                if got is None:
                    continue
                ask, desc = got
                # Center with tiny safe jitter (±4% of box) so arm targets safe interior
                jx = rng.uniform(-0.04, 0.04) * w
                jy = rng.uniform(-0.04, 0.04) * h
                cx, cy = x + w / 2 + jx, y + h / 2 + jy
                if rng.random() < 0.5:
                    kind, px, py = "c", cx * FEED[0] / W, cy * FEED[1] / H
                else:
                    kind = "k"
                    px, py = m @ np.array([cx, cy, 1.0])
                rec = make_sample(
                    names[kind],
                    ask,
                    desc,
                    cid,
                    [x, y, w, h],
                    px,
                    py,
                    think_mode=think_mode,
                    with_note=with_note,
                    merged_user=merged_user,
                )
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                st = stats[split]
                st["samples"] += 1
                st["icon"] += bool(bd.ICON.match(cid))
                st["camera"] += kind == "k"
                st["pages"][s["page"]["key"]] += 1
            stats[split]["screens"] += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--train", default=str(HERE / "dataset_v1_raw" / "train"))
    ap.add_argument("--val", default=str(HERE / "dataset_v1_raw" / "val"))
    ap.add_argument("--test", default=str(HERE / "screens"))
    ap.add_argument("--per-screen", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--think-mode", choices=("short", "empty"), default="short")
    ap.add_argument("--no-note", action="store_true")
    ap.add_argument("--merged-user", action="store_true")
    ap.add_argument("--reuse-images", default="")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tests = [bd.thumb(cv2.imread(str(p))) for p in sorted(pathlib.Path(a.test).glob("*.png"))]
    rng = random.Random(a.seed)
    reuse = pathlib.Path(a.reuse_images) if a.reuse_images else None
    stats = {
        k: {
            "screens": 0,
            "samples": 0,
            "icon": 0,
            "camera": 0,
            "near_test_dropped": 0,
            "pages": collections.Counter(),
        }
        for k in ("train", "val")
    }
    build_split(
        "train", pathlib.Path(a.train), out, a.per_screen, tests, rng, stats,
        think_mode=a.think_mode, with_note=not a.no_note, merged_user=a.merged_user, reuse_images=reuse,
    )
    build_split(
        "val", pathlib.Path(a.val), out, a.per_screen, tests, rng, stats,
        think_mode=a.think_mode, with_note=not a.no_note, merged_user=a.merged_user, reuse_images=reuse,
    )
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    for k, v in stats.items():
        print(k, {x: v[x] for x in ("screens", "samples", "icon", "camera", "near_test_dropped")})


if __name__ == "__main__":
    main()
