#!/usr/bin/env python3
"""CPU-only smoke test of the GUI-Owl-1.5 prompt -> model -> action-parsing chain, with camera-style degradation.

Runs in ``venvs/vlm`` (see ``scripts/setup_vlm_env.sh``).  It never touches the GPU: ``CUDA_VISIBLE_DEVICES`` is
emptied before torch is imported.

    venvs/vlm/bin/python scripts/smoke_vlm_cpu.py --make-screens   # render tests/data/screens (no model needed)
    venvs/vlm/bin/python scripts/smoke_vlm_cpu.py                  # all (screen x variant x instruction) steps
    venvs/vlm/bin/python scripts/smoke_vlm_cpu.py --screens home --variants clean --max-instructions 1  # one step
    venvs/vlm/bin/python scripts/smoke_vlm_cpu.py --check-vllm-render   # same tokens/pixels as vLLM's front end?
    venvs/vlm/bin/python scripts/smoke_vlm_cpu.py --report-only         # rebuild report.md from results.json

One step is built exactly as upstream (``third_party/mobile_agent_v35/mobile_use``) sends it to a vLLM server:
``build_messages`` -> the conversion of ``GUIOwlWrapper.convert_messages_format_to_openaiurl`` (``image_to_base64``'s
resize; see ``to_openai_messages`` for the one upstream bug worked around) -> the model's chat template -> the HF
processor with the pixel budget of ``scripts/serve_vlm.sh`` -> greedy generation -> ``parse_action`` ->
``rescale_coordinates``.

Variants of every screenshot: ``clean`` (the screenshot itself), ``cam_a`` and ``cam_b`` (the screenshot seen by a
simulated camera - oblique view, defocus, noise, colour shift, glare, JPEG - and rectified back to 1280 px wide).
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # the GPU is shared with a live robot run: never use it from here
THREADS = int(os.environ.get("SMOKE_VLM_THREADS", "12"))
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = str(THREADS)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # weights are local; no network needed

import argparse  # noqa: E402
import base64  # noqa: E402
import copy  # noqa: E402
import dataclasses  # noqa: E402
import datetime  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from typing import Any  # noqa: E402
import zoneinfo  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from PIL import ImageDraw  # noqa: E402
from PIL import ImageFont  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_DIR = ROOT / "third_party" / "mobile_agent_v35" / "mobile_use"
SCREENS_DIR = ROOT / "tests" / "data" / "screens"
DEFAULT_MODEL = ROOT / "weights" / "GUI-Owl-1.5-2B-Instruct"
DEFAULT_OUT = ROOT / "runs" / "smoke_vlm_cpu"

SCREEN_W, SCREEN_H = 1280, 800  # a 16:10 landscape tablet; also the size of the rectified camera view
# The pixel budget given to the server: scripts/serve_vlm.sh --mm-processor-kwargs (upstream README).
MM_PROCESSOR_SIZE = {"longest_edge": 3072000, "shortest_edge": 65536}
FONT_DIR = Path("/usr/share/fonts/opentype/noto")
Box = tuple[int, int, int, int]
Color = tuple[int, ...]

# --------------------------------------------------------------------------------------------------------------------
# Synthetic tablet screenshots
# --------------------------------------------------------------------------------------------------------------------


class Painter:
    """A supersampled PIL canvas addressed in screen pixels (drawn at ``scale`` x, then reduced)."""

    def __init__(self, width: int, height: int, scale: int = 2, background: np.ndarray | None = None) -> None:
        self.w, self.h, self.s = width, height, scale
        if background is None:
            self.img = Image.new("RGB", (width * scale, height * scale), (0, 0, 0))
        else:
            self.img = Image.fromarray(background).resize((width * scale, height * scale), Image.BILINEAR)
        self.d = ImageDraw.Draw(self.img, "RGBA")
        self._fonts: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}

    def _v(self, values: Any) -> list[float]:
        return [float(v) * self.s for v in values]

    def font(self, size: float, weight: str = "Regular") -> ImageFont.FreeTypeFont:
        """Noto Sans CJK SC at ``size`` screen pixels."""
        key = (weight, round(size * self.s))
        if key not in self._fonts:
            self._fonts[key] = ImageFont.truetype(str(FONT_DIR / f"NotoSansCJK-{weight}.ttc"), key[1], index=2)
        return self._fonts[key]

    def rect(self, box: Any, fill: Color | None, radius: float = 0, outline: Color | None = None, width: float = 0):
        self.d.rounded_rectangle(
            self._v(box), radius=radius * self.s, fill=fill, outline=outline, width=round(width * self.s)
        )

    def ellipse(self, box: Any, fill: Color | None, outline: Color | None = None, width: float = 0) -> None:
        self.d.ellipse(self._v(box), fill=fill, outline=outline, width=round(width * self.s))

    def circle(self, cx: float, cy: float, r: float, fill: Color | None, outline: Color | None = None, width=0.0):
        self.ellipse((cx - r, cy - r, cx + r, cy + r), fill, outline, width)

    def line(self, points: Any, fill: Color, width: float, rounded: bool = True) -> None:
        flat = self._v([c for p in points for c in p])
        self.d.line(flat, fill=fill, width=max(1, round(width * self.s)), joint="curve")
        if rounded:
            r = width * self.s / 2
            for x, y in zip(flat[0::2], flat[1::2]):
                self.d.ellipse((x - r, y - r, x + r, y + r), fill=fill)

    def poly(self, points: Any, fill: Color) -> None:
        self.d.polygon(self._v([c for p in points for c in p]), fill=fill)

    def arc(self, box: Any, start: float, end: float, fill: Color, width: float) -> None:
        self.d.arc(self._v(box), start, end, fill=fill, width=max(1, round(width * self.s)))

    def text(self, xy: Any, text: str, size: float, fill: Color, anchor: str = "la", weight: str = "Regular") -> None:
        self.d.text(self._v(xy), text, font=self.font(size, weight), fill=fill, anchor=anchor)

    def text_width(self, text: str, size: float, weight: str = "Regular") -> float:
        return self.font(size, weight).getlength(text) / self.s

    def finish(self) -> Image.Image:
        return self.img.resize((self.w, self.h), Image.LANCZOS)


def _gradient(width: int, height: int, stops: list[Color], direction: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    """A linear multi-stop gradient as an HxWx3 uint8 array."""
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    t = xs / width * direction[0] + ys / height * direction[1]
    t = (t - t.min()) / (t.max() - t.min())
    pos = np.linspace(0.0, 1.0, len(stops))
    out = np.stack([np.interp(t, pos, [s[c] for s in stops]) for c in range(3)], axis=-1)
    return out.astype(np.uint8)


def _add_blob(img: np.ndarray, cx: float, cy: float, sigma: float, color: Color, strength: float) -> np.ndarray:
    """Blend a soft Gaussian blob of ``color`` into ``img``."""
    ys, xs = np.mgrid[0 : img.shape[0], 0 : img.shape[1]].astype(np.float32)
    a = strength * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2))[..., None]
    return (img * (1 - a) + np.array(color, np.float32) * a).astype(np.uint8)


WHITE: Color = (255, 255, 255)


def _icon_bg(p: Painter, x: float, y: float, s: float, color: Color) -> None:
    p.rect((x, y, x + s, y + s), color, radius=s * 0.24)


def _glyph_music(p: Painter, x: float, y: float, s: float) -> None:
    _icon_bg(p, x, y, s, (250, 60, 92))
    for hx, hy, top in ((0.33, 0.69, 0.33), (0.66, 0.62, 0.26)):
        p.ellipse((x + s * (hx - 0.11), y + s * (hy - 0.085), x + s * (hx + 0.11), y + s * (hy + 0.085)), WHITE)
        p.line([(x + s * (hx + 0.085), y + s * hy), (x + s * (hx + 0.085), y + s * top)], WHITE, s * 0.05, False)
    p.poly(
        [
            (x + s * 0.39, y + s * 0.30),
            (x + s * 0.77, y + s * 0.22),
            (x + s * 0.77, y + s * 0.32),
            (x + s * 0.39, y + s * 0.40),
        ],
        WHITE,
    )


def _glyph_camera(p: Painter, x: float, y: float, s: float) -> None:
    _icon_bg(p, x, y, s, (62, 68, 80))
    p.rect((x + s * 0.36, y + s * 0.25, x + s * 0.64, y + s * 0.40), WHITE, radius=s * 0.04)
    p.rect((x + s * 0.17, y + s * 0.33, x + s * 0.83, y + s * 0.75), WHITE, radius=s * 0.08)
    p.circle(x + s * 0.5, y + s * 0.54, s * 0.155, (40, 44, 52))
    p.circle(x + s * 0.5, y + s * 0.54, s * 0.09, (80, 160, 245))
    p.circle(x + s * 0.74, y + s * 0.42, s * 0.03, (250, 190, 40))


def _glyph_settings(p: Painter, x: float, y: float, s: float) -> None:
    _icon_bg(p, x, y, s, (84, 110, 122))
    cx, cy = x + s * 0.5, y + s * 0.5
    for k in range(8):
        a = k * math.pi / 4
        ca, sa = math.cos(a), math.sin(a)
        pts = []
        for r, t in ((0.20, -0.075), (0.36, -0.06), (0.36, 0.06), (0.20, 0.075)):
            pts.append((cx + s * (r * ca - t * sa), cy + s * (r * sa + t * ca)))
        p.poly(pts, WHITE)
    p.circle(cx, cy, s * 0.27, WHITE)
    p.circle(cx, cy, s * 0.115, (84, 110, 122))


def _glyph_gallery(p: Painter, x: float, y: float, s: float) -> None:
    _icon_bg(p, x, y, s, (250, 250, 250))
    p.rect((x + s * 0.16, y + s * 0.2, x + s * 0.84, y + s * 0.8), (129, 212, 250), radius=s * 0.08)
    p.circle(x + s * 0.66, y + s * 0.36, s * 0.07, (255, 213, 79))
    p.poly([(x + s * 0.16, y + s * 0.74), (x + s * 0.38, y + s * 0.44), (x + s * 0.60, y + s * 0.74)], (67, 160, 71))
    p.poly([(x + s * 0.42, y + s * 0.74), (x + s * 0.64, y + s * 0.52), (x + s * 0.84, y + s * 0.74)], (46, 125, 50))
    p.rect((x + s * 0.16, y + s * 0.72, x + s * 0.84, y + s * 0.8), (46, 125, 50), radius=s * 0.04)


def _glyph_browser(p: Painter, x: float, y: float, s: float) -> None:
    _icon_bg(p, x, y, s, (250, 250, 250))
    cx, cy, r = x + s * 0.5, y + s * 0.5, s * 0.33
    p.circle(cx, cy, r, (30, 136, 229))
    p.ellipse((cx - r * 0.45, cy - r, cx + r * 0.45, cy + r), None, WHITE, s * 0.03)
    p.line([(cx - r, cy), (cx + r, cy)], WHITE, s * 0.03, False)
    p.line([(cx, cy - r), (cx, cy + r)], WHITE, s * 0.03, False)
    p.arc((cx - r * 0.9, cy - r * 1.6, cx + r * 0.9, cy - r * 0.2), 20, 160, WHITE, s * 0.03)
    p.arc((cx - r * 0.9, cy + r * 0.2, cx + r * 0.9, cy + r * 1.6), 200, 340, WHITE, s * 0.03)


def _glyph_calendar(p: Painter, x: float, y: float, s: float) -> None:
    _icon_bg(p, x, y, s, (229, 57, 53))
    p.rect((x, y + s * 0.3, x + s, y + s), (250, 250, 250), radius=s * 0.24)
    p.rect((x, y + s * 0.3, x + s, y + s * 0.6), (250, 250, 250))
    p.text((x + s * 0.5, y + s * 0.16), "11月", s * 0.17, WHITE, "mm", "Medium")
    p.text((x + s * 0.5, y + s * 0.62), "23", s * 0.42, (50, 50, 50), "mm", "Medium")


def _glyph_clock(p: Painter, x: float, y: float, s: float) -> None:
    _icon_bg(p, x, y, s, (38, 40, 56))
    cx, cy = x + s * 0.5, y + s * 0.5
    p.circle(cx, cy, s * 0.34, (245, 245, 245))
    p.line([(cx, cy), (cx, cy - s * 0.23)], (38, 40, 56), s * 0.045)
    p.line([(cx, cy), (cx + s * 0.16, cy + s * 0.07)], (38, 40, 56), s * 0.045)
    p.circle(cx, cy, s * 0.035, (229, 57, 53))


def _glyph_files(p: Painter, x: float, y: float, s: float) -> None:
    _icon_bg(p, x, y, s, (250, 250, 250))
    p.rect((x + s * 0.17, y + s * 0.26, x + s * 0.5, y + s * 0.5), (255, 160, 0), radius=s * 0.05)
    p.rect((x + s * 0.17, y + s * 0.34, x + s * 0.83, y + s * 0.76), (255, 193, 7), radius=s * 0.06)


def _glyph_calculator(p: Painter, x: float, y: float, s: float) -> None:
    _icon_bg(p, x, y, s, (55, 71, 79))
    for sym, gx, gy in (("+", 0.31, 0.30), ("−", 0.69, 0.30), ("×", 0.31, 0.68), ("=", 0.69, 0.68)):
        p.text((x + s * gx, y + s * gy), sym, s * 0.36, (255, 183, 77) if sym == "=" else WHITE, "mm", "Medium")


def _glyph_mail(p: Painter, x: float, y: float, s: float) -> None:
    _icon_bg(p, x, y, s, (30, 136, 229))
    p.rect((x + s * 0.17, y + s * 0.28, x + s * 0.83, y + s * 0.72), WHITE, radius=s * 0.05)
    p.line(
        [(x + s * 0.2, y + s * 0.32), (x + s * 0.5, y + s * 0.54), (x + s * 0.8, y + s * 0.32)],
        (30, 136, 229),
        s * 0.04,
    )


def _glyph_maps(p: Painter, x: float, y: float, s: float) -> None:
    _icon_bg(p, x, y, s, (200, 230, 201))
    p.line([(x + s * 0.05, y + s * 0.75), (x + s * 0.95, y + s * 0.45)], (255, 255, 255), s * 0.09, False)
    p.line([(x + s * 0.3, y + s * 0.05), (x + s * 0.4, y + s * 0.95)], (255, 255, 255), s * 0.07, False)
    p.circle(x + s * 0.56, y + s * 0.4, s * 0.18, (229, 57, 53))
    p.poly([(x + s * 0.40, y + s * 0.47), (x + s * 0.72, y + s * 0.47), (x + s * 0.56, y + s * 0.8)], (229, 57, 53))
    p.circle(x + s * 0.56, y + s * 0.4, s * 0.065, WHITE)


def _glyph_video(p: Painter, x: float, y: float, s: float) -> None:
    _icon_bg(p, x, y, s, (126, 87, 194))
    p.rect((x + s * 0.17, y + s * 0.26, x + s * 0.83, y + s * 0.74), None, s * 0.08, WHITE, s * 0.045)
    p.poly([(x + s * 0.43, y + s * 0.38), (x + s * 0.43, y + s * 0.62), (x + s * 0.62, y + s * 0.5)], WHITE)


def _glyph_messages(p: Painter, x: float, y: float, s: float) -> None:
    _icon_bg(p, x, y, s, (67, 160, 71))
    p.rect((x + s * 0.2, y + s * 0.25, x + s * 0.8, y + s * 0.66), WHITE, radius=s * 0.12)
    p.poly([(x + s * 0.3, y + s * 0.6), (x + s * 0.28, y + s * 0.8), (x + s * 0.48, y + s * 0.64)], WHITE)


def _glyph_contacts(p: Painter, x: float, y: float, s: float) -> None:
    _icon_bg(p, x, y, s, (251, 140, 0))
    p.circle(x + s * 0.5, y + s * 0.38, s * 0.14, WHITE)
    p.d.pieslice(p._v((x + s * 0.24, y + s * 0.56, x + s * 0.76, y + s * 1.0)), 180, 360, fill=WHITE)


def _glyph_store(p: Painter, x: float, y: float, s: float) -> None:
    _icon_bg(p, x, y, s, (250, 250, 250))
    p.rect((x + s * 0.22, y + s * 0.38, x + s * 0.78, y + s * 0.8), (0, 150, 136), radius=s * 0.06)
    p.arc((x + s * 0.36, y + s * 0.2, x + s * 0.64, y + s * 0.56), 180, 360, (0, 150, 136), s * 0.05)


def _glyph_notes(p: Painter, x: float, y: float, s: float) -> None:
    _icon_bg(p, x, y, s, (255, 213, 79))
    for k in range(3):
        yy = y + s * (0.34 + 0.16 * k)
        p.line([(x + s * 0.26, yy), (x + s * (0.74 if k < 2 else 0.56), yy)], (121, 85, 72), s * 0.05)


def _status_bar(p: Painter, color: Color) -> None:
    """Clock on the left; Wi-Fi and battery on the right."""
    p.text((24, 15), "10:24", 15, color, "lm", "Medium")
    wx, wy = 1178, 21
    for k, r in enumerate((16, 11, 6)):
        p.arc((wx - r, wy - r, wx + r, wy + r), 225, 315, color, 2.2 if k else 2.2)
    p.circle(wx, wy - 1, 1.8, color)
    p.rect((1204, 9, 1228, 21), None, 3, color, 1.5)
    p.rect((1206.5, 11.5, 1222, 18.5), color, 1)
    p.rect((1229, 12.5, 1231, 17.5), color, 1)
    p.text((1238, 15), "86%", 13, color, "lm")


def _nav_pill(p: Painter, color: Color) -> None:
    p.rect((SCREEN_W / 2 - 70, 787, SCREEN_W / 2 + 70, 792), color, radius=2.5)


def render_home() -> tuple[Image.Image, dict[str, Any]]:
    """Launcher: a 6x2 grid of labelled app icons, a search pill, a dock."""
    bg = _gradient(SCREEN_W // 4, SCREEN_H // 4, [(28, 52, 128), (92, 62, 150), (24, 124, 150)], (1.0, 0.7))
    bg = _add_blob(bg, 60, 40, 45, (255, 140, 120), 0.35)
    bg = _add_blob(bg, 250, 170, 60, (60, 200, 190), 0.30)
    p = Painter(SCREEN_W, SCREEN_H, background=bg)
    _status_bar(p, WHITE)
    p.text((70, 92), "10:24", 64, WHITE, "lm", "Light")
    p.text((72, 146), "11月23日 星期一", 18, (235, 235, 245), "lm")
    p.rect((470, 70, 1210, 122), (255, 255, 255, 235), radius=26)
    p.circle(503, 95, 8, None, (110, 110, 120), 2.4)
    p.line([(509, 101), (515, 107)], (110, 110, 120), 2.4)
    p.text((530, 95), "搜索应用、设置和网页", 17, (120, 120, 130), "lm")

    apps = [
        ("music", "音乐", _glyph_music),
        ("camera", "相机", _glyph_camera),
        ("settings", "设置", _glyph_settings),
        ("gallery", "图库", _glyph_gallery),
        ("browser", "浏览器", _glyph_browser),
        ("calendar", "日历", _glyph_calendar),
        ("clock", "时钟", _glyph_clock),
        ("files", "文件", _glyph_files),
        ("calculator", "计算器", _glyph_calculator),
        ("mail", "邮件", _glyph_mail),
        ("maps", "地图", _glyph_maps),
        ("video", "视频", _glyph_video),
    ]
    elements: dict[str, Any] = {
        "search": {"box": [470, 70, 1210, 122], "label": "搜索栏"},
    }
    size, cols, pitch_x, x0 = 88, 6, 186, 131
    for i, (name, label, glyph) in enumerate(apps):
        cx = x0 + pitch_x * (i % cols) + size / 2
        y = 228 + 196 * (i // cols)
        glyph(p, cx - size / 2, y, size)
        p.text((cx + 1, y + size + 25), label, 19, (0, 0, 0, 90), "mm")
        p.text((cx, y + size + 24), label, 19, WHITE, "mm")
        # tap target of a launcher cell: the icon and its label
        elements[name] = {"box": [round(cx - 60), y - 6, round(cx + 60), y + size + 40], "label": f"{label} 图标"}

    p.rect((420, 672, 860, 772), (255, 255, 255, 60), radius=34)
    for k, glyph in enumerate((_glyph_messages, _glyph_contacts, _glyph_store, _glyph_notes)):
        glyph(p, 452 + 104 * k, 688, 68)
    _nav_pill(p, (255, 255, 255, 230))
    meta = {
        "description": "合成的平板桌面（横屏 1280x800）：6x2 应用图标（中文标签）、搜索栏、底部停靠栏",
        "elements": elements,
        "instructions": [
            {"text": "打开音乐", "target": "music"},
            {"text": "打开设置", "target": "settings"},
            {"text": "点击相机图标", "target": "camera"},
            {"text": "打开图库", "target": "gallery"},
            {"text": "打开浏览器", "target": "browser"},
        ],
    }
    return p.finish(), meta


def _heart(p: Painter, cx: float, cy: float, r: float, color: Color, width: float) -> None:
    pts = []
    for k in range(41):
        t = 2 * math.pi * k / 40
        hx = 16 * math.sin(t) ** 3
        hy = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        pts.append((cx + hx * r / 16, cy - hy * r / 16))
    p.line(pts, color, width)


def _skip(p: Painter, cx: float, cy: float, r: float, color: Color, forward: bool) -> None:
    """The standard skip-next / skip-previous glyph: a triangle and a bar."""
    sgn = 1 if forward else -1
    p.poly([(cx - sgn * r, cy - r), (cx - sgn * r, cy + r), (cx + sgn * r * 0.55, cy)], color)
    bar0, bar1 = sorted((cx + sgn * r * 0.62, cx + sgn * r))
    p.rect((bar0, cy - r, bar1, cy + r), color, 1.5)


def render_player() -> tuple[Image.Image, dict[str, Any]]:
    """Music player (paused): album art, title, progress bar, shuffle / previous / play / next / repeat."""
    bg = _gradient(SCREEN_W // 4, SCREEN_H // 4, [(58, 36, 84), (30, 26, 52), (16, 16, 26)], (0.5, 1.0))
    p = Painter(SCREEN_W, SCREEN_H, background=bg)
    _status_bar(p, WHITE)
    p.line([(48, 70), (60, 82), (72, 70)], WHITE, 3)
    p.text((SCREEN_W / 2, 76), "正在播放", 20, WHITE, "mm", "Medium")
    for k in range(3):
        p.circle(1226, 64 + 12 * k, 2.6, WHITE)

    ax, ay, asz = 96, 150, 440
    art = _gradient(110, 110, [(20, 30, 90), (60, 40, 140), (250, 120, 90)], (0.3, 1.0))
    art = _add_blob(art, 78, 30, 9, (255, 250, 220), 0.95)
    art_img = Image.fromarray(art).resize((asz * p.s, asz * p.s), Image.BICUBIC)
    mask = Image.new("L", art_img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, art_img.size[0] - 1, art_img.size[1] - 1), 28 * p.s, fill=255)
    p.img.paste(art_img, (ax * p.s, ay * p.s), mask)
    p.poly([(ax, ay + asz - 28), (ax + 150, ay + 300), (ax + 300, ay + asz - 28)], (14, 16, 40, 255))
    p.poly(
        [(ax + 180, ay + asz - 28), (ax + 330, ay + 330), (ax + asz, ay + asz - 60), (ax + asz, ay + asz - 28)],
        (24, 22, 56),
    )
    p.rect((ax, ay + asz - 60, ax + asz, ay + asz), (14, 16, 40), radius=28)
    p.rect((ax, ay + asz - 60, ax + asz, ay + asz - 30), (14, 16, 40))
    p.text((ax + 28, ay + 44), "夜空", 34, (255, 255, 255, 230), "lm", "Medium")

    rx0, rx1 = 640, 1184
    p.text((rx0, 214), "夜空中最亮的星", 38, WHITE, "lm", "Medium")
    p.text((rx0, 268), "逃跑计划 · 世界", 20, (200, 196, 214), "lm")
    _heart(p, 1160, 222, 15, WHITE, 3)
    p.text((rx0, 336), "每当我找不到存在的意义", 18, (150, 146, 168), "lm")
    p.text((rx0, 368), "每当我迷失在黑夜里", 18, (235, 232, 245), "lm")

    bar_y = 452
    p.line([(rx0, bar_y), (rx1, bar_y)], (110, 104, 132), 5)
    p.line([(rx0, bar_y), (rx0 + 220, bar_y)], WHITE, 5)
    p.circle(rx0 + 220, bar_y, 9, WHITE)
    p.text((rx0, bar_y + 26), "1:42", 15, (200, 196, 214), "lm")
    p.text((rx1, bar_y + 26), "4:12", 15, (200, 196, 214), "rm")

    cy = 570
    centers = {"shuffle": 668, "prev": 790, "play": 912, "next": 1034, "repeat": 1156}
    grey: Color = (214, 210, 226)
    # shuffle: two crossing strokes with arrow heads
    sx = centers["shuffle"]
    p.line([(sx - 17, cy - 11), (sx - 6, cy - 11), (sx + 6, cy + 11), (sx + 15, cy + 11)], grey, 3)
    p.line([(sx - 17, cy + 11), (sx - 6, cy + 11), (sx + 6, cy - 11), (sx + 15, cy - 11)], grey, 3)
    for yy in (cy - 11, cy + 11):
        p.poly([(sx + 13, yy - 7), (sx + 13, yy + 7), (sx + 22, yy)], grey)
    _skip(p, centers["prev"], cy, 17, WHITE, forward=False)
    p.circle(centers["play"], cy, 46, WHITE)
    p.poly([(centers["play"] - 13, cy - 20), (centers["play"] - 13, cy + 20), (centers["play"] + 21, cy)], (30, 26, 52))
    _skip(p, centers["next"], cy, 17, WHITE, forward=True)
    # repeat: a rounded loop with two arrow heads
    qx = centers["repeat"]
    p.rect((qx - 18, cy - 11, qx + 18, cy + 11), None, 8, grey, 3)
    p.poly([(qx + 2, cy - 18), (qx + 2, cy - 4), (qx + 11, cy - 11)], grey)
    p.poly([(qx - 2, cy + 18), (qx - 2, cy + 4), (qx - 11, cy + 11)], grey)

    # volume row and queue
    vy = 690
    p.poly(
        [(rx0, vy - 5), (rx0 + 7, vy - 5), (rx0 + 15, vy - 12), (rx0 + 15, vy + 12), (rx0 + 7, vy + 5), (rx0, vy + 5)],
        grey,
    )
    p.arc((rx0 + 8, vy - 10, rx0 + 28, vy + 10), -50, 50, grey, 2.5)
    p.line([(rx0 + 46, vy), (rx0 + 330, vy)], (110, 104, 132), 4)
    p.line([(rx0 + 46, vy), (rx0 + 230, vy)], grey, 4)
    p.circle(rx0 + 230, vy, 7, WHITE)
    for k in range(3):
        p.line([(1140, vy - 9 + 9 * k), (1178, vy - 9 + 9 * k)], grey, 3)
    p.text((1066, vy), "词", 20, grey, "mm")
    p.rect((1048, vy - 17, 1084, vy + 17), None, 7, grey, 2)
    _nav_pill(p, (255, 255, 255, 230))

    def around(cx: float, half: int) -> list[int]:
        return [round(cx - half), cy - half, round(cx + half), cy + half]

    elements = {
        "play": {"box": around(centers["play"], 48), "label": "播放按钮（当前暂停，显示 ▶）"},
        "next": {"box": around(centers["next"], 36), "label": "下一首"},
        "prev": {"box": around(centers["prev"], 36), "label": "上一首"},
        "shuffle": {"box": around(centers["shuffle"], 36), "label": "随机播放"},
        "repeat": {"box": around(centers["repeat"], 36), "label": "循环播放"},
        "favorite": {"box": [1160 - 30, 222 - 30, 1160 + 30, 222 + 30], "label": "收藏（心形）"},
        "collapse": {"box": [30, 46, 90, 106], "label": "收起播放页（向下箭头）"},
    }
    meta = {
        "description": "合成的音乐播放页（横屏 1280x800，暂停状态）：封面、歌名、进度条、"
        "随机/上一首/播放/下一首/循环、音量条",
        "elements": elements,
        "instructions": [
            {"text": "点击播放按钮", "target": "play"},
            {"text": "下一首", "target": "next"},
            {"text": "上一首", "target": "prev"},
            {"text": "收藏这首歌", "target": "favorite"},
            {"text": "开启随机播放", "target": "shuffle"},
        ],
    }
    return p.finish(), meta


def _toggle(p: Painter, x: float, cy: float, on: bool) -> None:
    p.rect((x, cy - 14, x + 52, cy + 14), (26, 115, 232) if on else (189, 193, 198), radius=14)
    p.circle(x + (38 if on else 14), cy, 10.5, WHITE)


def _slider(p: Painter, x0: float, x1: float, cy: float, frac: float) -> None:
    xm = x0 + (x1 - x0) * frac
    p.line([(x0, cy), (x1, cy)], (210, 214, 220), 5)
    p.line([(x0, cy), (xm, cy)], (26, 115, 232), 5)
    p.circle(xm, cy, 10, (26, 115, 232))


def render_settings() -> tuple[Image.Image, dict[str, Any]]:
    """Two-pane Settings: the category list on the left, the "声音和振动" page on the right."""
    p = Painter(SCREEN_W, SCREEN_H, background=np.full((8, 8, 3), 242, np.uint8))
    ink: Color = (32, 33, 36)
    sub: Color = (95, 99, 104)
    _status_bar(p, ink)
    pane = 440
    p.text((40, 76), "设置", 34, ink, "lm", "Medium")
    p.rect((32, 112, pane - 24, 164), (226, 230, 236), radius=26)
    p.circle(64, 137, 8, None, sub, 2.4)
    p.line([(70, 143), (76, 149)], sub, 2.4)
    p.text((90, 138), "搜索设置", 17, sub, "lm")

    rows = [
        ("wlan", "WLAN", "Home-5G", (26, 115, 232)),
        ("bluetooth", "蓝牙", "已关闭", (26, 115, 232)),
        ("display", "显示和亮度", "深色主题、字体大小", (251, 140, 0)),
        ("sound", "声音和振动", "音量、勿扰模式", (0, 150, 136)),
        ("notifications", "通知", "通知历史记录、对话", (229, 57, 53)),
        ("battery", "电池", "86% · 大约还可使用 9 小时", (67, 160, 71)),
        ("storage", "存储", "已使用 41% · 剩余 75 GB", (123, 31, 162)),
        ("apps", "应用", "最近使用的应用、默认应用", (84, 110, 122)),
        ("about", "关于平板电脑", "Tab P11", (84, 110, 122)),
    ]
    elements: dict[str, Any] = {"search": {"box": [32, 112, pane - 24, 164], "label": "搜索设置"}}
    row_h, y = 66, 182
    for name, title, subtitle, color in rows:
        if name == "sound":
            p.rect((20, y + 2, pane - 12, y + row_h - 2), (210, 227, 252), radius=30)
        cx, cy = 62, y + row_h / 2
        p.circle(cx, cy, 19, color)
        if name == "wlan":
            for r in (13, 8):
                p.arc((cx - r, cy + 5 - r, cx + r, cy + 5 + r), 225, 315, WHITE, 2.4)
            p.circle(cx, cy + 5, 2.2, WHITE)
        elif name == "bluetooth":
            p.line(
                [(cx - 6, cy - 6), (cx + 6, cy + 6), (cx, cy + 12), (cx, cy - 12), (cx + 6, cy - 6), (cx - 6, cy + 6)],
                WHITE,
                2.2,
            )
        elif name == "display":
            p.circle(cx, cy, 5.5, WHITE)
            for k in range(8):
                a = k * math.pi / 4
                p.line(
                    [(cx + 9 * math.cos(a), cy + 9 * math.sin(a)), (cx + 12 * math.cos(a), cy + 12 * math.sin(a))],
                    WHITE,
                    2,
                )
        elif name == "sound":
            p.poly(
                [
                    (cx - 10, cy - 4),
                    (cx - 5, cy - 4),
                    (cx + 1, cy - 9),
                    (cx + 1, cy + 9),
                    (cx - 5, cy + 4),
                    (cx - 10, cy + 4),
                ],
                WHITE,
            )
            p.arc((cx - 4, cy - 8, cx + 10, cy + 8), -50, 50, WHITE, 2.2)
        elif name == "notifications":
            p.d.pieslice(p._v((cx - 8, cy - 10, cx + 8, cy + 8)), 180, 360, fill=WHITE)
            p.rect((cx - 8, cy - 1, cx + 8, cy + 6), WHITE)
            p.circle(cx, cy + 9, 2.5, WHITE)
        elif name == "battery":
            p.rect((cx - 5.5, cy - 9, cx + 5.5, cy + 10), WHITE, 2)
            p.rect((cx - 2.5, cy - 12, cx + 2.5, cy - 8), WHITE, 1)
        elif name == "storage":
            for k in range(3):
                p.rect((cx - 9, cy - 9 + 7 * k, cx + 9, cy - 4 + 7 * k), WHITE, 1.5)
        elif name == "apps":
            for gx in (-6, 6):
                for gy in (-6, 6):
                    p.rect((cx + gx - 4, cy + gy - 4, cx + gx + 4, cy + gy + 4), WHITE, 1.5)
        else:
            p.text((cx, cy), "i", 22, WHITE, "mm", "Bold")
        p.text((98, cy - 11), title, 19, ink, "lm")
        p.text((98, cy + 14), subtitle, 14, sub, "lm")
        elements[name] = {"box": [20, y + 2, pane - 12, y + row_h - 2], "label": f"左栏“{title}”一行"}
        y += row_h

    # right pane: a white card
    cx0, cx1 = pane + 16, SCREEN_W - 32
    p.rect((cx0, 48, cx1, 772), WHITE, radius=28)
    p.text((cx0 + 40, 100), "声音和振动", 28, ink, "lm", "Medium")
    sl0, sl1 = cx0 + 96, cx1 - 48
    for k, (name, title, frac) in enumerate(
        (("media_volume", "媒体音量", 0.45), ("ring_volume", "铃声和通知音量", 0.7), ("alarm_volume", "闹钟音量", 0.6))
    ):
        yy = 162 + 96 * k
        p.text((cx0 + 40, yy), title, 18, ink, "lm")
        vx, vy = cx0 + 54, yy + 40
        p.poly(
            [(vx - 11, vy - 4), (vx - 6, vy - 4), (vx, vy - 9), (vx, vy + 9), (vx - 6, vy + 4), (vx - 11, vy + 4)], sub
        )
        p.arc((vx - 5, vy - 8, vx + 9, vy + 8), -50, 50, sub, 2.2)
        _slider(p, sl0, sl1, vy, frac)
        elements[name] = {"box": [round(sl0) - 12, yy + 22, round(sl1) + 12, yy + 58], "label": f"{title} 滑块"}
    toggles = (
        ("dnd", "勿扰模式", "已关闭 · 只允许重要的人和应用打扰", False),
        ("touch_sounds", "触摸提示音", "点按屏幕时播放提示音", True),
        ("lock_sounds", "锁屏提示音", "锁定和解锁屏幕时播放提示音", True),
        ("vibrate", "响铃时振动", "来电时同时振动", False),
    )
    yy = 452
    for name, title, subtitle, on in toggles:
        p.line([(cx0 + 40, yy), (cx1 - 40, yy)], (232, 234, 237), 1.5, False)
        p.text((cx0 + 40, yy + 28), title, 19, ink, "lm")
        p.text((cx0 + 40, yy + 54), subtitle, 14, sub, "lm")
        _toggle(p, cx1 - 104, yy + 39, on)
        # tapping anywhere on a switch row toggles it, so the row is the target
        elements[name] = {"box": [cx0 + 24, yy + 4, cx1 - 24, yy + 74], "label": f"“{title}”一行（含开关）"}
        elements[name + "_switch"] = {"box": [cx1 - 112, yy + 17, cx1 - 44, yy + 61], "label": f"“{title}”的开关"}
        yy += 78
    _nav_pill(p, (60, 64, 67))
    meta = {
        "description": "合成的设置页（横屏 1280x800，双栏）：左栏分类列表，右栏“声音和振动”（三条音量滑块、四个开关）",
        "elements": elements,
        "instructions": [
            {"text": "打开蓝牙设置", "target": "bluetooth"},
            {"text": "打开勿扰模式", "target": "dnd"},
            {"text": "点击搜索设置", "target": "search"},
            {"text": "进入显示和亮度", "target": "display"},
            {"text": "点击WLAN", "target": "wlan"},
        ],
    }
    return p.finish(), meta


SCREEN_RENDERERS = {"home": render_home, "player": render_player, "settings": render_settings}

SOURCES_MD = """# 测试截图的来源

这里的三张“平板截图”都是**合成图**，由 `scripts/smoke_vlm_cpu.py --make-screens` 用 PIL 画出来
（字体：系统自带的 Noto Sans CJK SC，SIL OFL 许可）。没有任何图片来自网络，所以没有来源网址。

为什么没用真实截图：2026-11-23 在 Wikimedia Commons 上按关键词和分类找过可自由使用的横屏安卓平板截图
（"Android tablet screenshot home screen"、"Android 12L screenshot"、"Android 13 tablet screenshot"、
"Android music player tablet screenshot"、"Android tablet settings screenshot"，以及
Category:Android 12/13/14 screenshots 等分类），没找到同时满足“横屏平板、有音乐播放页/桌面/设置页、许可清楚”的图
（唯一命中的横屏图是 File:Elgato SAT)IP app on Nexus 7.png，是一个电视应用的界面，用不上），
之后 Commons 的接口开始限流，就改用合成图。

| 文件 | 内容 | 标注 |
|---|---|---|
| home.png | 桌面：6x2 个带中文标签的应用图标（音乐、相机、设置、图库、浏览器…）、搜索栏、停靠栏 | home.json |
| player.png | 音乐播放页（暂停）：封面、歌名、进度条、随机/上一首/播放/下一首/循环 | player.json |
| settings.png | 设置页（双栏）：左栏分类，右栏“声音和振动”的音量滑块和开关 | settings.json |

每个 json 里：`elements` 是各个可点元素的真值框 `[x0, y0, x1, y1]`（像素，1280x800，含边界），
`instructions` 是要问模型的中文指令和它对应的目标元素。

注意：合成图比真实截图干净、简单（没有真实壁纸、没有真实应用的图标细节），这里量到的命中率只能当作
链路是否通、以及画质退化影响大小的第一手参考，不能当作真实平板上的成功率。
"""


def make_screens(out_dir: Path) -> None:
    """Render the synthetic screenshots and their ground-truth JSON into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, renderer in SCREEN_RENDERERS.items():
        image, meta = renderer()
        image.save(out_dir / f"{name}.png", optimize=True)
        meta = {"image": f"{name}.png", "size": [SCREEN_W, SCREEN_H], "source": "synthetic (PIL)", **meta}
        (out_dir / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"已生成 {out_dir / (name + '.png')}（{(out_dir / (name + '.png')).stat().st_size / 1024:.0f} KB）")
    (out_dir / "SOURCES.md").write_text(SOURCES_MD, encoding="utf-8")


# --------------------------------------------------------------------------------------------------------------------
# Camera simulation and rectification
# --------------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CameraSpec:
    """One simulated camera view of the tablet."""

    name: str
    label: str
    frame: tuple[int, int]  # (width, height) of the camera frame
    span_px: float  # width of the screen's bounding box in the camera frame
    supersample: int  # the frame is rendered this many times larger, then area-averaged (pixel integration)
    blur_sigma: float  # defocus, camera pixels
    read_noise: float  # sensor noise std at black, 8-bit levels
    jpeg_quality: int


CAMERAS = {
    "cam_a": CameraSpec("cam_a", "相机(a)：1280x720 画面，屏幕宽约 900 px", (1280, 720), 900.0, 3, 0.9, 2.5, 85),
    "cam_b": CameraSpec("cam_b", "相机(b)：640x480 画面，屏幕宽约 450 px", (640, 480), 450.0, 4, 0.7, 3.5, 80),
}
VARIANTS = ("clean", "cam_a", "cam_b")
VARIANT_LABELS = {"clean": "原始截图", "cam_a": CAMERAS["cam_a"].label, "cam_b": CAMERAS["cam_b"].label}
# Per-screenshot camera pose (degrees): pitch = tilt away from looking straight down at the tablet, then yaw, roll.
POSES = [(30.0, 7.0, -4.0), (36.0, -9.0, 5.0), (24.0, 5.0, 3.0), (33.0, -5.0, -6.0)]
SCREEN_M = (0.236, 0.1475)  # an 11-inch 16:10 panel, metres
BEZEL_M = 0.009
HFOV_DEG = 70.0


def _rot(axis: str, deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _project(points_m: np.ndarray, pose: tuple[float, float, float], dist: float, focal: float, center: np.ndarray):
    """Pinhole projection of points on the tablet plane (metres, z = 0) into camera pixels."""
    pitch, yaw, roll = pose
    rot = _rot("z", roll) @ _rot("x", pitch) @ _rot("y", yaw)
    cam = points_m @ rot.T + np.array([0.0, 0.0, dist])
    return focal * cam[:, :2] / cam[:, 2:3] + center


def camera_geometry(spec: CameraSpec, pose_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (H, bezel): H maps screenshot pixels to camera pixels; bezel is the tablet body's quad in the frame."""
    pose = POSES[pose_index % len(POSES)]
    focal = spec.frame[0] / 2 / math.tan(math.radians(HFOV_DEG / 2))
    hw, hh = SCREEN_M[0] / 2, SCREEN_M[1] / 2
    screen = np.array([[-hw, -hh, 0], [hw, -hh, 0], [hw, hh, 0], [-hw, hh, 0]], dtype=np.float64)
    bezel = screen + np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]) * BEZEL_M
    lo, hi = 0.05, 5.0
    for _ in range(60):  # bisection on the distance until the screen spans span_px
        mid = (lo + hi) / 2
        q = _project(screen, pose, mid, focal, np.zeros(2))
        lo, hi = (mid, hi) if q[:, 0].max() - q[:, 0].min() > spec.span_px else (lo, mid)
    q = _project(screen, pose, hi, focal, np.zeros(2))
    # put the screen's bounding box near the middle of the frame, a little off-centre
    offset = np.array([0.03 * spec.frame[0], -0.02 * spec.frame[1]]) * (1 if pose_index % 2 == 0 else -1)
    center = np.array(spec.frame) / 2 - (q.min(0) + q.max(0)) / 2 + offset
    quad = _project(screen, pose, hi, focal, center).astype(np.float32)
    src = np.array([[0, 0], [SCREEN_W, 0], [SCREEN_W, SCREEN_H], [0, SCREEN_H]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, quad), _project(bezel, pose, hi, focal, center).astype(np.float32)


def simulate_camera(screen_rgb: np.ndarray, spec: CameraSpec, pose_index: int, seed: int) -> dict[str, Any]:
    """Photograph the screenshot with a simulated camera, then rectify it back to SCREEN_W x SCREEN_H.

    Returns the camera frame (after JPEG), the rectified image, the homography and the simulation parameters.
    """
    rng = np.random.default_rng(seed)
    cam_w, cam_h = spec.frame
    ss = spec.supersample
    h_mat, bezel = camera_geometry(spec, pose_index)
    big = (cam_w * ss, cam_h * ss)
    up = np.diag([ss, ss, 1.0])

    # a grey table: uneven lighting plus a faint texture
    ys, xs = np.mgrid[0:cam_h, 0:cam_w].astype(np.float32)
    light = 0.88 + 0.2 * (xs / cam_w) - 0.08 * (ys / cam_h)
    coarse = cv2.resize(rng.normal(0, 1, (cam_h // 16 + 1, cam_w // 16 + 1)).astype(np.float32), (cam_w, cam_h))
    fine = rng.normal(0, 1, (cam_h, cam_w)).astype(np.float32)
    table = (np.array([132.0, 130.0, 126.0], np.float32) * (light + 0.03 * coarse + 0.012 * fine)[..., None]).clip(
        0, 255
    )
    frame = cv2.resize(table, big, interpolation=cv2.INTER_LINEAR)

    # the tablet body (black bezel) and the emissive screen
    body = np.zeros((big[1], big[0]), np.float32)
    cv2.fillConvexPoly(body, np.round(bezel * ss * 16).astype(np.int32), 1.0, lineType=cv2.LINE_AA, shift=4)
    frame = frame * (1 - body[..., None]) + np.array([16.0, 16.0, 18.0], np.float32) * body[..., None]
    warped = cv2.warpPerspective(screen_rgb.astype(np.float32), up @ h_mat, big, flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.ones(screen_rgb.shape[:2], np.float32), up @ h_mat, big, flags=cv2.INTER_LINEAR)
    screen_gain = 0.92  # panel brightness relative to the exposure
    frame = frame * (1 - mask[..., None]) + warped * screen_gain * mask[..., None]
    frame = cv2.resize(frame, (cam_w, cam_h), interpolation=cv2.INTER_AREA)  # pixel integration
    body_small = cv2.resize(body, (cam_w, cam_h), interpolation=cv2.INTER_AREA)

    # reflections on the glass: a faint veil plus one soft specular blob somewhere on the screen
    u, v = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)
    gx, gy = cv2.perspectiveTransform(np.array([[[u * SCREEN_W, v * SCREEN_H]]], np.float32), h_mat)[0, 0]
    sig_u, sig_v = 0.11 * spec.span_px, 0.065 * spec.span_px
    ang = math.radians(rng.uniform(-30, 30))
    dx, dy = xs - gx, ys - gy
    ru, rv = dx * math.cos(ang) + dy * math.sin(ang), -dx * math.sin(ang) + dy * math.cos(ang)
    glare_peak = 105.0
    glare = glare_peak * np.exp(-0.5 * ((ru / sig_u) ** 2 + (rv / sig_v) ** 2))
    frame = frame + ((glare + 7.0) * body_small)[..., None] * np.array([1.0, 1.0, 0.96], np.float32)

    frame = cv2.GaussianBlur(frame, (0, 0), spec.blur_sigma)  # mild defocus
    wb_gain = np.array([1.05, 1.0, 0.90], np.float32) * 0.94  # warm indoor light, slightly under-exposed
    frame = frame * wb_gain + 4.0
    noise_std = np.sqrt(spec.read_noise**2 + 0.02 * frame.clip(0, 255))  # read noise + shot noise
    frame = frame + rng.normal(0, 1, frame.shape).astype(np.float32) * noise_std
    frame_u8 = frame.clip(0, 255).astype(np.uint8)

    ok, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, spec.jpeg_quality]
    )
    assert ok
    camera_rgb = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    rectified = cv2.warpPerspective(
        camera_rgb, h_mat, (SCREEN_W, SCREEN_H), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP
    )
    quad = cv2.perspectiveTransform(
        np.array([[[0, 0], [SCREEN_W, 0], [SCREEN_W, SCREEN_H], [0, SCREEN_H]]], np.float32), h_mat
    )[0]
    params = {
        "frame": list(spec.frame),
        "pose_deg_pitch_yaw_roll": list(POSES[pose_index % len(POSES)]),
        "screen_quad_px": [[round(float(x), 1), round(float(y), 1)] for x, y in quad],
        "screen_span_px": round(float(quad[:, 0].max() - quad[:, 0].min()), 1),
        "screen_height_px": round(float(quad[:, 1].max() - quad[:, 1].min()), 1),
        "supersample": ss,
        "blur_sigma_px": spec.blur_sigma,
        "read_noise_std": spec.read_noise,
        "shot_noise": "std^2 += 0.02 * level",
        "wb_gain_rgb": [round(float(g), 3) for g in wb_gain],
        "offset": 4.0,
        "screen_gain": screen_gain,
        "glare": {
            "center_px": [round(float(gx), 1), round(float(gy), 1)],
            "peak": glare_peak,
            "veil": 7.0,
            "sigma_px": [round(sig_u, 1), round(sig_v, 1)],
            "center_on_screen": [round(u, 3), round(v, 3)],
        },
        "jpeg_quality": spec.jpeg_quality,
        "jpeg_bytes": int(len(buf)),
        "rectify": "cv2.warpPerspective(INTER_CUBIC | WARP_INVERSE_MAP) with the exact homography",
        "seed": seed,
    }
    return {"camera": camera_rgb, "rectified": rectified, "homography": h_mat, "params": params}


def build_variants(
    screen_names: list[str], variants: list[str], out_dir: Path
) -> dict[tuple[str, str], dict[str, Any]]:
    """Create every requested (screen, variant) image file; return its path and simulation parameters."""
    var_dir = out_dir / "variants"
    var_dir.mkdir(parents=True, exist_ok=True)
    all_names = list(SCREEN_RENDERERS)
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for name in screen_names:
        src = SCREENS_DIR / f"{name}.png"
        screen = np.array(Image.open(src).convert("RGB"))
        pose_index = all_names.index(name) if name in all_names else 0
        for variant in variants:
            if variant == "clean":
                result[(name, variant)] = {"path": src, "params": None}
                continue
            spec = CAMERAS[variant]
            sim = simulate_camera(screen, spec, pose_index, seed=1000 * (pose_index + 1) + VARIANTS.index(variant))
            cam_path = var_dir / f"{name}__{variant}__camera.jpg"
            rect_path = var_dir / f"{name}__{variant}__rectified.png"
            Image.fromarray(sim["camera"]).save(cam_path, quality=95)
            Image.fromarray(sim["rectified"]).save(rect_path)
            diff = sim["rectified"].astype(np.float32) - screen.astype(np.float32)
            sim["params"]["psnr_vs_clean_db"] = round(float(10 * np.log10(255.0**2 / np.mean(diff**2))), 2)
            result[(name, variant)] = {"path": rect_path, "camera_path": cam_path, "params": sim["params"]}
    return result


# --------------------------------------------------------------------------------------------------------------------
# Model: upstream prompt -> HF processor -> generate -> upstream parser
# --------------------------------------------------------------------------------------------------------------------


def import_upstream() -> tuple[Any, Any]:
    """Import upstream ``utils`` and ``run_gui_owl_1_5_for_mobile`` (unchanged files in third_party)."""
    if str(UPSTREAM_DIR) not in sys.path:
        sys.path.insert(0, str(UPSTREAM_DIR))
    import run_gui_owl_1_5_for_mobile as up_run  # type: ignore[import-not-found]
    import utils as up_utils  # type: ignore[import-not-found]

    return up_utils, up_run


def check_weights_complete(model_dir: Path) -> None:
    """Refuse to load a model directory that ``hf download`` has not finished."""
    problems = []
    if not (model_dir / "config.json").is_file():
        problems.append("缺 config.json")
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        problems += [f"缺权重分片 {s}" for s in shards if not (model_dir / s).is_file()]
    elif not (model_dir / "model.safetensors").is_file():
        problems.append("缺 model.safetensors（也没有 model.safetensors.index.json）")
    partial = list((model_dir / ".cache").rglob("*.incomplete")) if (model_dir / ".cache").is_dir() else []
    if partial:
        problems.append(f"还有 {len(partial)} 个 *.incomplete 文件（下载没结束）")
    if problems:
        raise SystemExit(f"模型目录不完整，先等下载结束：{model_dir}\n  - " + "\n  - ".join(problems))


def to_openai_messages(dash_messages: list[dict[str, Any]], up_utils: Any) -> list[dict[str, Any]]:
    """Upstream's ``GUIOwlWrapper.convert_messages_format_to_openaiurl``, with its ``file://`` bug worked around.

    Upstream ``build_messages`` writes images as ``"file://" + path`` and the converter hands that string straight
    to ``image_to_base64`` -> ``PIL.Image.open``, which cannot open a ``file://`` URL (FileNotFoundError at
    upstream commit c4cd0a8).  The loop below is the upstream one line for line; the only change is that the
    prefix is removed before upstream's own ``image_to_base64`` (smart_resize factor 28, <= 10035200 px, PNG).
    """
    converted = []
    for message in dash_messages:
        new_content = []
        for item in message["content"]:
            if list(item.keys())[0] == "text":
                new_content.append({"type": "text", "text": item["text"]})
            elif list(item.keys())[0] == "image":
                url = up_utils.image_to_base64(item["image"].removeprefix("file://"))
                new_content.append({"type": "image_url", "image_url": {"url": url}})
        converted.append({"role": message["role"], "content": new_content})
    return converted


def to_hf_messages(openai_messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    """Turn the OpenAI-style payload upstream sends into HF chat-template messages plus the decoded images.

    Text parts are kept verbatim.  An ``image_url`` part carries the PNG that upstream's ``image_to_base64``
    produced (already resized); it is decoded back to the very same pixels.
    """
    messages, images = [], []
    for message in openai_messages:
        content = []
        for part in message["content"]:
            if part["type"] == "text":
                content.append({"type": "text", "text": part["text"]})
            elif part["type"] == "image_url":
                url = part["image_url"]["url"]
                assert url.startswith("data:image/png;base64,"), url[:40]
                image = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")
                images.append(image)
                content.append({"type": "image", "image": image})
            else:
                raise ValueError(f"unexpected content part: {part['type']}")
        messages.append({"role": message["role"], "content": content})
    return messages, images


class CpuAgentModel:
    """GUI-Owl-1.5 loaded with transformers on the CPU (float32)."""

    def __init__(self, model_dir: Path, threads: int) -> None:
        import torch
        from transformers import AutoModelForImageTextToText
        from transformers import AutoProcessor

        assert not torch.cuda.is_available(), "CUDA must stay hidden in this test"
        torch.set_num_threads(threads)
        torch.set_grad_enabled(False)
        self.torch = torch
        self.model_dir = model_dir
        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(str(model_dir))
        self.model = AutoModelForImageTextToText.from_pretrained(str(model_dir), dtype=torch.float32)
        self.model.eval()
        self.load_seconds = time.time() - t0
        self.image_token_id = self.model.config.image_token_id
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def step(self, openai_messages: list[dict[str, Any]], max_new_tokens: int) -> dict[str, Any]:
        """One agent step: the payload upstream would POST to vLLM, answered greedily."""
        torch = self.torch
        t0 = time.time()
        messages, images = to_hf_messages(openai_messages)
        # The same two stages a vLLM server runs: render the chat template to text, then call the HF processor
        # with --mm-processor-kwargs.
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=images, size=MM_PROCESSOR_SIZE, return_tensors="pt")
        t1 = time.time()
        n_prompt = int(inputs["input_ids"].shape[1])
        n_image = int((inputs["input_ids"] == self.image_token_id).sum())
        with torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None
            )
        t2 = time.time()
        new_tokens = out[0, n_prompt:]
        text = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        grid = inputs["image_grid_thw"].tolist() if "image_grid_thw" in inputs else None
        return {
            "raw_output": text,
            "prompt_text": prompt,
            "prompt_tokens": n_prompt,
            "image_tokens": n_image,
            "text_tokens": n_prompt - n_image,
            "output_tokens": int(new_tokens.shape[0]),
            "image_sizes_sent": [list(im.size) for im in images],
            "image_grid_thw": grid,
            "preprocess_seconds": round(t1 - t0, 3),
            "generate_seconds": round(t2 - t1, 3),
            "seconds": round(t2 - t0, 3),
        }


def check_vllm_render(model_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Render one upstream request with vLLM's own chat renderer (CPU only) and compare it with the HF path.

    This is the server's front half - chat template, image loading, the multi-modal processor with the
    ``serve_vlm.sh`` flags - without the engine, so no GPU is involved.  Equal token ids and equal pixel values mean
    the model in this test sees exactly what it would see behind ``vllm serve``.
    """
    import torch
    from transformers import AutoProcessor
    from vllm.config import ModelConfig
    from vllm.config import VllmConfig
    from vllm.renderers import renderer_from_config
    from vllm.renderers.params import ChatParams

    up_utils, _ = import_upstream()
    image = SCREENS_DIR / "home.png"
    dash_messages = up_utils.build_messages(str(image), "打开音乐", [], model_dir.name)
    openai_messages = to_openai_messages(dash_messages, up_utils)

    model_config = ModelConfig(
        model=str(model_dir),
        max_model_len=32768,
        limit_mm_per_prompt={"image": 5},
        mm_processor_kwargs={"size": MM_PROCESSOR_SIZE},
        allowed_local_media_path="/",
    )
    renderer = renderer_from_config(VllmConfig(model_config=model_config))
    chat_params = ChatParams(chat_template_kwargs={"add_generation_prompt": True})
    _, prompts = renderer.render_chat([openai_messages], chat_params)
    vllm_ids = list(prompts[0]["prompt_token_ids"])
    vllm_mm = prompts[0]["mm_kwargs"].get_data()

    processor = AutoProcessor.from_pretrained(str(model_dir))
    messages, images = to_hf_messages(openai_messages)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, size=MM_PROCESSOR_SIZE, return_tensors="pt")
    hf_ids = inputs["input_ids"][0].tolist()

    # vLLM keeps the patches as uint8 and normalizes on the device; the HF processor returns normalized floats
    vllm_pixels = vllm_mm["pixel_values"]
    if vllm_pixels.dtype == torch.uint8:
        vllm_pixels = (vllm_pixels.float() / 255.0 - 0.5) / 0.5
    hf_pixels = inputs["pixel_values"].float()
    same_shape = tuple(vllm_pixels.shape) == tuple(hf_pixels.shape)
    result = {
        "time": _beijing_now(),
        "request": {"image": str(image), "instruction": "打开音乐"},
        "vllm_prompt_tokens": len(vllm_ids),
        "hf_prompt_tokens": len(hf_ids),
        "token_ids_identical": vllm_ids == hf_ids,
        "vllm_image_grid_thw": vllm_mm["image_grid_thw"].tolist(),
        "hf_image_grid_thw": inputs["image_grid_thw"].tolist(),
        "pixel_values_shape": list(hf_pixels.shape),
        "vllm_pixel_dtype": str(vllm_mm["pixel_values"].dtype),
        "pixel_values_max_abs_diff": float((vllm_pixels - hf_pixels).abs().max()) if same_shape else None,
        "versions": _versions(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vllm_render_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def score_output(raw: str, image_size: tuple[int, int], box: list[int], up_utils: Any, up_run: Any) -> dict[str, Any]:
    """Parse the model output with upstream's parser, scale to pixels as upstream does, and test the target box."""
    result: dict[str, Any] = {"parsed_action": None, "pred_norm": None, "pred_px": None, "outcome": "parse_error"}
    try:
        action = up_run.parse_action(raw)
    except ValueError as exc:
        result["parse_error"] = str(exc)
        return result
    result["parsed_action"] = copy.deepcopy(action)
    args = copy.deepcopy(action["arguments"])
    coord = args.get("coordinate")
    if args.get("action") != "click" or not (isinstance(coord, list) and len(coord) == 2):
        result["outcome"] = "not_click"
        return result
    width, height = image_size
    resized_h, resized_w = up_utils.smart_resize(height, width, factor=16, min_pixels=3136, max_pixels=1003520 * 200)
    result["pred_norm"] = [float(coord[0]), float(coord[1])]
    px = up_run.rescale_coordinates(args, resized_w, resized_h)["coordinate"]
    # upstream taps (px) on a device whose screenshot is width x height; resized_* only rounds to a multiple of 16
    px = [px[0] * width / resized_w, px[1] * height / resized_h]
    result["pred_px"] = [round(px[0], 1), round(px[1], 1)]
    x0, y0, x1, y1 = box
    hit = x0 <= px[0] <= x1 and y0 <= px[1] <= y1
    result["outcome"] = "hit" if hit else "miss"
    result["center_error_px"] = round(math.hypot(px[0] - (x0 + x1) / 2, px[1] - (y0 + y1) / 2), 1)
    result["outside_box_px"] = round(math.hypot(max(x0 - px[0], 0, px[0] - x1), max(y0 - px[1], 0, px[1] - y1)), 1)
    return result


def annotate(image_path: Path, box: list[int], pred_px: list[float] | None, caption: str, out_path: Path) -> None:
    """Save the model's input image with the target box (green) and the predicted tap (red)."""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline=(0, 220, 0), width=3)
    if pred_px is not None:
        x, y = pred_px
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), outline=(255, 0, 0), width=3)
        draw.line((x - 18, y, x + 18, y), fill=(255, 0, 0), width=2)
        draw.line((x, y - 18, x, y + 18), fill=(255, 0, 0), width=2)
    font = ImageFont.truetype(str(FONT_DIR / "NotoSansCJK-Regular.ttc"), 20, index=2)
    tw = draw.textlength(caption, font=font)
    draw.rectangle((0, image.height - 30, tw + 16, image.height), fill=(0, 0, 0))
    draw.text((8, image.height - 29), caption, font=font, fill=(255, 255, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, quality=88)


# --------------------------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------------------------


def _rate(steps: list[dict[str, Any]]) -> str:
    hits = sum(s["outcome"] == "hit" for s in steps)
    return f"{hits}/{len(steps)}" if steps else "—"


def _median(values: list[float]) -> str:
    return f"{float(np.median(values)):.1f}" if values else "—"


def write_report(results: dict[str, Any], out_dir: Path) -> None:
    """Write report.md (Chinese) from the results dict."""
    steps = results["steps"]
    env, run = results["environment"], results["run"]
    screens = list(dict.fromkeys(s["screen"] for s in steps))
    variants = [v for v in VARIANTS if any(s["variant"] == v for s in steps)]
    lines = [
        "# GUI-Owl-1.5 CPU 冒烟测试报告",
        "",
        f"- 时间：{run['started']} → {run['finished']}（北京时间）",
        f"- 命令：`{run['command']}`",
        f"- 模型：`{run['model_dir']}`（{run['n_params'] / 1e9:.2f} B 参数，CPU float32，"
        f"{run['threads']} 线程，贪心解码，max_new_tokens={run['max_new_tokens']}）",
        f"- 环境：vllm {env['vllm']}（只安装，未启动）、transformers {env['transformers']}、torch {env['torch']}、"
        f"Python {env['python']}；CUDA_VISIBLE_DEVICES 为空，torch.cuda.is_available()={env['cuda_available']}",
        "- 提示词：上游 `build_messages` + `convert_messages_format_to_openaiurl` 的转换"
        "（含 `image_to_base64` 的缩放；上游这一步把 `file://` 前缀直接交给 PIL 打不开，"
        "这里去掉前缀后调用上游的 `image_to_base64`），"
        f"模型自带 chat template，处理器 size={json.dumps(MM_PROCESSOR_SIZE)}（与 serve_vlm.sh 一致）",
        f"- 步数：{len(steps)}（{len(screens)} 张截图 × {len(variants)} 种画面 × "
        f"每张 {run['instructions_per_screen']} 条指令）",
        "",
        "## 命中率（预测点落在目标框内的步数 / 总步数）",
        "",
        "| 画面 | 命中 | 未命中 | 不是 click | 解析失败 | 点到框中心的距离中位数 (px) | 每步秒数中位数 |",
        "|---|---|---|---|---|---|---|",
    ]
    for v in variants:
        sub = [s for s in steps if s["variant"] == v]
        count = {k: sum(s["outcome"] == k for s in sub) for k in ("miss", "not_click", "parse_error")}
        lines.append(
            f"| {VARIANT_LABELS[v]} | **{_rate(sub)}** | {count['miss']} | {count['not_click']} | "
            f"{count['parse_error']} | "
            f"{_median([s['center_error_px'] for s in sub if 'center_error_px' in s])} | "
            f"{_median([s['seconds'] for s in sub])} |"
        )
    lines += ["", "按截图分：", "", "| 截图 | " + " | ".join(VARIANT_LABELS[v] for v in variants) + " |"]
    lines.append("|---|" + "---|" * len(variants))
    for name in screens:
        cells = [_rate([s for s in steps if s["screen"] == name and s["variant"] == v]) for v in variants]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "相对原始截图的变化（同一张截图、同一条指令）："]
    outcome = {(s["screen"], s["variant"], s["instruction"]): s for s in steps}
    for v in variants:
        if v == "clean":
            continue
        lost, gained = [], []
        for (name, variant, ins), s in outcome.items():
            base = outcome.get((name, "clean", ins))
            if variant != v or base is None:
                continue
            if base["outcome"] == "hit" and s["outcome"] != "hit":
                lost.append(f"{name}“{ins}”")
            elif base["outcome"] != "hit" and s["outcome"] == "hit":
                gained.append(f"{name}“{ins}”")
        lines.append(
            f"- {v}：原图命中而这里没命中 {len(lost)} 步（{'、'.join(lost) or '无'}）；"
            f"原图没命中而这里命中 {len(gained)} 步（{'、'.join(gained) or '无'}）"
        )
    lines += [
        "",
        "注意：样本很少（每格只有十来步），截图是合成的，相机退化是仿真的（见文末“没有覆盖的东西”）。"
        "这张表只说明链路通不通、退化大致有多大影响，不是真实平板上的成功率。",
        "",
        "## 每一步",
        "",
        "| # | 截图 | 画面 | 指令 | 目标框 | 模型坐标 (0–1000) | 像素点 | 结果 | 离框中心 px | 秒 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    mark = {"hit": "命中", "miss": "未命中", "not_click": "不是 click", "parse_error": "解析失败"}
    for i, s in enumerate(steps):
        lines.append(
            f"| {i} | {s['screen']} | {s['variant']} | {s['instruction']} | {s['target']} {s['target_box']} | "
            f"{s['pred_norm']} | {s['pred_px']} | {mark[s['outcome']]} | {s.get('center_error_px', '—')} | "
            f"{s['seconds']} |"
        )
    lines += ["", "## 模型原始输出", ""]
    for i, s in enumerate(steps):
        lines += [
            f"**#{i} {s['screen']} / {s['variant']} / “{s['instruction']}”** → {mark[s['outcome']]}",
            "",
            "```",
            s["raw_output"],
            "```",
            "",
        ]
    secs = [s["seconds"] for s in steps]
    gen = [s["generate_seconds"] for s in steps]
    out_tok = [s["output_tokens"] for s in steps]
    first = steps[0]
    lines += [
        "## 耗时、内存、token",
        "",
        f"- 加载模型：{run['load_seconds']:.1f} s",
        f"- 每步（预处理 + 生成）：中位数 {np.median(secs):.1f} s，最小 {min(secs):.1f} s，最大 {max(secs):.1f} s；"
        f"其中生成 {np.median(gen):.1f} s（中位数），输出 {int(np.median(out_tok))} 个 token（中位数）",
        f"- 进程内存峰值（ru_maxrss）：{run['peak_rss_gb']:.2f} GB；加载完模型后 {run['rss_after_load_gb']:.2f} GB",
        f"- 一步的提示词：共 {first['prompt_tokens']} token = 图像 {first['image_tokens']} + "
        f"文字 {first['text_tokens']}（系统提示词占大头）；"
        f"送进处理器的图是 {first['image_sizes_sent'][0]}（上游 image_to_base64 缩放后），"
        f"处理器网格 (t, h, w) = {first['image_grid_thw'][0] if first['image_grid_thw'] else '—'}",
        "- 服务器预算怎么估：一张图的 token 数 = 网格 h×w / 4（16 px 的 patch，2×2 合并），"
        f"上限 = longest_edge / 1024 = {MM_PROCESSOR_SIZE['longest_edge'] // 1024} 个/图；"
        "1280×800 的矫正图是 1000 个/图。上游带 4 步历史时一次请求最多 5 张图。",
        "",
        "## 相机退化的参数",
        "",
        "| 截图 | 画面 | 屏幕在相机里的宽×高 (px) | 俯仰/偏航/滚转 (°) | 模糊 σ | 读出噪声 σ | JPEG 质量 | "
        "矫正图对原图的 PSNR (dB) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, info in results["variants"].items():
        prm = info["params"]
        if prm is None:
            continue
        name, variant = key.split("/")
        lines.append(
            f"| {name} | {variant} | {prm['screen_span_px']} × {prm['screen_height_px']} | "
            f"{prm['pose_deg_pitch_yaw_roll']} | {prm['blur_sigma_px']} | {prm['read_noise_std']} | "
            f"{prm['jpeg_quality']} | {prm['psnr_vs_clean_db']} |"
        )
    lines += [
        "",
        "做法：截图贴到灰色桌面上的黑边平板里，按针孔相机（水平视场 70°）斜着投影到相机画面（超采样后面积平均），"
        "加玻璃反光（一层薄雾 + 一个柔和的高光斑）、轻微失焦、偏暖的白平衡和略欠曝、读出噪声 + 散粒噪声、JPEG 压缩，"
        "最后用已知的单应矩阵（双三次插值，和 tablet_tap/vision/rectifier.py 一样）变回 1280×800。"
        "相机画面和矫正图在 `variants/`，带标注的图在 `annotated/`"
        "（绿框 = 目标，红十字 = 模型点的位置）。",
    ]
    check_path = out_dir / "vllm_render_check.json"
    if check_path.is_file():
        chk = json.loads(check_path.read_text(encoding="utf-8"))
        lines += [
            "",
            "## 和 vLLM 服务前半段的一致性检查（CPU，没有启动引擎）",
            "",
            f"`--check-vllm-render`（{chk['time']}）：同一个请求（home.png，“打开音乐”）分别走 "
            f"vLLM {chk['versions']['vllm']} 自己的聊天渲染器（chat template + 图像读取 + 多模态处理器，"
            "参数同 serve_vlm.sh）和本测试的 HF 路径：",
            "",
            f"- 提示词 token：vLLM {chk['vllm_prompt_tokens']} 个，HF {chk['hf_prompt_tokens']} 个，"
            f"逐个 id 相同：{chk['token_ids_identical']}",
            f"- 图像网格 (t, h, w)：vLLM {chk['vllm_image_grid_thw']}，HF {chk['hf_image_grid_thw']}",
            f"- 图像块 pixel_values（形状 {chk['pixel_values_shape']}；vLLM 存的是 {chk['vllm_pixel_dtype']}，"
            f"归一化后比较）：最大绝对差 {chk['pixel_values_max_abs_diff']}",
            "",
            "也就是说，本测试喂给模型的输入和 vLLM 服务喂给模型的输入是一样的；没验证的是服务后半段（GPU 上的引擎、"
            "bf16 精度、采样）。",
        ]
    lines += [
        "",
        "## 没有覆盖的东西",
        "",
        "- 没有用 GPU，没有启动 vLLM：8B 模型、vLLM 服务、bf16 精度下的结果都没有验证。"
        "这里是 2B 模型在 CPU 上用 float32 跑的。",
        "- 上游的运行脚本调用 vLLM 时不传 temperature，服务器会用模型 generation_config.json 里的采样参数"
        "（temperature 0.6、top_k 40、top_p 0.95）；上游 cookbook 传的是 temperature=0。"
        "这里用贪心解码（等价于 temperature=0），结果是确定的，和运行脚本的默认行为不完全一样。",
        "- 仿真里的矫正用的是精确的单应矩阵；真实系统里角点定位有误差，还会有摩尔纹、屏幕刷新条纹、镜头畸变、"
        "手和触控笔的遮挡、环境光变化，这些都没有仿真。",
        f"- 只测了单步（没有历史图像），每张截图只有 {run['instructions_per_screen']} 条指令。",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------------------------


def _beijing_now() -> str:
    return datetime.datetime.now(zoneinfo.ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")


def _rss_gb() -> float:
    with open("/proc/self/status", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024**2
    return float("nan")


def _peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def _versions() -> dict[str, Any]:
    import importlib.metadata as md
    import platform

    import torch

    out: dict[str, Any] = {"python": platform.python_version(), "cuda_available": bool(torch.cuda.is_available())}
    for name in ("vllm", "transformers", "torch", "tokenizers", "qwen-vl-utils", "pillow", "opencv-python-headless"):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    out["torch_cuda_build"] = torch.version.cuda
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="GUI-Owl-1.5 的 CPU 冒烟测试（不使用 GPU）")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="模型目录（CPU 上只建议 2B）")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="输出目录")
    parser.add_argument(
        "--make-screens", action="store_true", help="只生成 tests/data/screens 里的合成截图和标注，然后退出"
    )
    parser.add_argument("--variants-only", action="store_true", help="只生成相机退化图（不加载模型），然后退出")
    parser.add_argument(
        "--check-vllm-render",
        action="store_true",
        help="只做一致性检查：用 vLLM 自己的聊天渲染器（CPU，不启动引擎）渲染一个请求，"
        "和本测试的 HF 路径比 token 和像素",
    )
    parser.add_argument(
        "--report-only", action="store_true", help="不跑模型，只用已有的 results.json 重新生成 report.md"
    )
    parser.add_argument("--screens", nargs="+", default=None, help="只跑这些截图（默认全部）")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS, help="只跑这些画面")
    parser.add_argument("--max-instructions", type=int, default=None, help="每张截图最多跑几条指令（太慢时用它减量）")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="每步最多生成的 token 数")
    args = parser.parse_args()

    if args.make_screens:
        make_screens(SCREENS_DIR)
        return
    if args.report_only:
        write_report(json.loads((args.out / "results.json").read_text(encoding="utf-8")), args.out)
        print(f"报告：{args.out / 'report.md'}")
        return
    if args.check_vllm_render:
        check_weights_complete(args.model)
        chk = check_vllm_render(args.model, args.out)
        print(
            f"vLLM 渲染器 {chk['vllm_prompt_tokens']} token，HF 路径 {chk['hf_prompt_tokens']} token，"
            f"id 逐个相同：{chk['token_ids_identical']}；pixel_values 最大绝对差：{chk['pixel_values_max_abs_diff']}"
        )
        print(f"结果：{args.out / 'vllm_render_check.json'}")
        return
    metas = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(SCREENS_DIR.glob("*.json"))}
    if not metas:
        raise SystemExit(f"{SCREENS_DIR} 里没有截图，先运行：scripts/smoke_vlm_cpu.py --make-screens")
    order = [n for n in SCREEN_RENDERERS if n in metas] + [n for n in metas if n not in SCREEN_RENDERERS]
    screen_names = [n for n in order if args.screens is None or n in args.screens]
    args.out.mkdir(parents=True, exist_ok=True)
    variants = build_variants(screen_names, args.variants, args.out)
    for (name, variant), info in variants.items():
        if info["params"] is not None:
            prm = info["params"]
            print(
                f"相机退化图 {name}/{variant}: 屏宽 {prm['screen_span_px']} px, "
                f"PSNR {prm['psnr_vs_clean_db']} dB -> {info['path']}"
            )
    if args.variants_only:
        return

    check_weights_complete(args.model)
    up_utils, up_run = import_upstream()
    started = _beijing_now()
    print(f"加载模型 {args.model}（CPU，{THREADS} 线程）…", flush=True)
    agent = CpuAgentModel(args.model, THREADS)
    rss_after_load = _rss_gb()
    print(
        f"模型已加载：{agent.load_seconds:.1f} s，{agent.n_params / 1e9:.2f} B 参数，内存 {rss_after_load:.1f} GB",
        flush=True,
    )

    plan = []
    for name in screen_names:
        instructions = metas[name]["instructions"][: args.max_instructions]
        for variant in args.variants:
            plan += [(name, variant, ins) for ins in instructions]
    results: dict[str, Any] = {
        "environment": _versions(),
        "run": {
            "started": started,
            "command": " ".join([Path(sys.executable).name, *sys.argv]),
            "model_dir": str(args.model),
            "n_params": agent.n_params,
            "threads": THREADS,
            "dtype": "float32",
            "decoding": "greedy",
            "max_new_tokens": args.max_new_tokens,
            "mm_processor_size": MM_PROCESSOR_SIZE,
            "load_seconds": round(agent.load_seconds, 1),
            "rss_after_load_gb": round(rss_after_load, 2),
            "instructions_per_screen": args.max_instructions or max(len(m["instructions"]) for m in metas.values()),
        },
        "variants": {
            f"{n}/{v}": {"path": str(i["path"]), "camera_path": str(i.get("camera_path", "")), "params": i["params"]}
            for (n, v), i in variants.items()
        },
        "steps": [],
    }
    for k, (name, variant, ins) in enumerate(plan):
        info = variants[(name, variant)]
        box = metas[name]["elements"][ins["target"]]["box"]
        # exactly what upstream's agent loop does for the first step of a task (empty history)
        dash_messages = up_utils.build_messages(str(info["path"]), ins["text"], [], args.model.name)
        openai_messages = to_openai_messages(dash_messages, up_utils)
        step = agent.step(openai_messages, args.max_new_tokens)
        with Image.open(info["path"]) as im:
            image_size = im.size
        step.update(score_output(step["raw_output"], image_size, box, up_utils, up_run))
        step.update(screen=name, variant=variant, instruction=ins["text"], target=ins["target"], target_box=box)
        step["image_path"] = str(info["path"])
        step["user_text"] = dash_messages[-1]["content"][0]["text"]
        ann = args.out / "annotated" / f"{name}__{variant}__{ins['target']}.jpg"
        annotate(info["path"], box, step["pred_px"], f"{variant} | {ins['text']} | {step['outcome']}", ann)
        step["annotated_path"] = str(ann)
        if k > 0:
            step.pop("prompt_text")  # the system prompt is identical in every step; keep it once
        results["steps"].append(step)
        print(
            f"[{k + 1}/{len(plan)}] {name}/{variant} “{ins['text']}” -> {step['outcome']} {step['pred_px']} "
            f"框 {box}  {step['seconds']:.1f} s（生成 {step['output_tokens']} token）\n    {step['raw_output']!r}",
            flush=True,
        )
        if k == 0:
            print(
                f"  按第一步估计，全部 {len(plan)} 步大约需要 {step['seconds'] * len(plan) / 60:.1f} 分钟", flush=True
            )
        results["run"]["finished"] = _beijing_now()
        results["run"]["peak_rss_gb"] = round(_peak_rss_gb(), 2)
        (args.out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(results, args.out)
    print(f"报告：{args.out / 'report.md'}\n结果：{args.out / 'results.json'}")
    for v in args.variants:
        print(f"  {VARIANT_LABELS[v]}: 命中 {_rate([s for s in results['steps'] if s['variant'] == v])}")


if __name__ == "__main__":
    main()
