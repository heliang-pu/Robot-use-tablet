"""The printed tags that let the camera follow the tablet while an app covers the calibration page: an A4 page, 300 dpi.

    bin/py scripts/make_stand_tags.py            -> cad/out/stand_tags_A4.png and .pdf

Print at 100 %, cut along the thin lines, and fix two to four of them flat on the stand beside the screen, on the plane of
the glass as nearly as possible, so that they move with the tablet and the arm does not cover all of them."""

import pathlib
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tablet_tap.calibration.live_screen import TAG_IDS  # noqa: E402
from tablet_tap.vision.layout import marker_bits  # noqa: E402

DPI, SIDE_MM, QUIET_MM = 300, 36.0, 6.0
px = lambda mm: int(round(mm / 25.4 * DPI))  # noqa: E731
page = np.full((px(297), px(210)), 255, np.uint8)
for k, tag in enumerate(list(TAG_IDS)[:8]):
    x, y = px(30 + 90 * (k % 2)), px(20 + 66 * (k // 2))
    side, quiet = px(SIDE_MM), px(QUIET_MM)
    page[y : y + side, x : x + side] = cv2.resize(marker_bits(tag), (side, side), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(page, (x - quiet, y - quiet), (x + side + quiet, y + side + quiet), 170, 1)
    cv2.putText(page, f"{tag}  36 mm", (x, y + side + quiet + px(5)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 120, 2)
out = ROOT / "cad" / "out"
out.mkdir(parents=True, exist_ok=True)
Image.fromarray(page).save(out / "stand_tags_A4.png", dpi=(DPI, DPI))
Image.fromarray(page).save(out / "stand_tags_A4.pdf", resolution=DPI)
found = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)).detectMarkers(page)[1]
print("written", out / "stand_tags_A4.pdf", "- detected again:", sorted(found.ravel().tolist()))
