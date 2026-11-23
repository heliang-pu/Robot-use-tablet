from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from tablet_tap.app import AppContext
from tablet_tap.app import ArmUse
from tablet_tap.cli.commands.base import Command
from tablet_tap.cli.commands.base import register
from tablet_tap.errors import CalibrationError
from tablet_tap.vision.layout import ScreenLayout
from tablet_tap.vision.rectifier import ScreenRectifier
from tablet_tap.vision.rectifier import draw_overlay

MIN_SCREEN_PX = 700  # below this width in the camera frame small icons and text become unreadable for the model


@register
class View(Command):
    name = "view"
    help = "相机标定：拍标定页的 ArUco 码 → 相机到屏幕的单应"
    arm = ArmUse.NONE

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--out_width",
            type=int,
            default=1792,
            help="矫正后屏幕图的宽度（像素）；1792×1120 同时是 28 和 32 的倍数",
        )

    def run(self, ctx: AppContext, args: argparse.Namespace) -> None:
        camera = ctx.camera()
        page = ctx.page()
        page.set_mode("markers")
        page.clear()
        time.sleep(0.6)  # the page redraws
        frame = camera.read()
        camera.close()
        save_view(ctx, frame, page.layout(), int(args.out_width))
        ctx.save_setup()  # remembers --camera


def save_view(ctx: AppContext, frame: np.ndarray, layout: ScreenLayout, out_width: int) -> ScreenRectifier:
    """Calibrate camera -> screen from one frame of the calibration page and write it where ``--view`` says."""
    view_path = ctx.options.view_path
    view_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        rectifier = ScreenRectifier.from_marker_image(frame, layout, out_width=out_width)
    except CalibrationError as exc:
        debug = view_path.with_suffix(".failed.jpg")
        cv2.imwrite(str(debug), frame)
        raise CalibrationError(
            f"相机标定失败：{exc}\n相机画面已存到 {debug}，看看标定码是否被挡住、反光或太小。"
        ) from None
    rectifier.save(view_path)
    cv2.imwrite(str(view_path.with_suffix(".overlay.jpg")), draw_overlay(frame, rectifier))
    cv2.imwrite(str(view_path.with_suffix(".rectified.png")), rectifier.rectify(frame))
    quad = rectifier.screen_quad_cam()
    width_px = float(np.linalg.norm(quad[1] - quad[0]))
    print(f"已保存 {view_path}（重投影误差 {rectifier.reprojection_px:.2f} px；屏幕在相机画面里宽 {width_px:.0f} px）")
    if width_px < MIN_SCREEN_PX:
        print(
            f"⚠ 屏幕在画面里不到 {MIN_SCREEN_PX} px 宽，小图标和文字会看不清。相机已经是最高分辨率、又不能挪近时，"
            "把平板的“显示大小”和“字体大小”调到最大，尽量用大图标的界面。"
        )
    return rectifier
