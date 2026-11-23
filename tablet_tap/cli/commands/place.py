from __future__ import annotations

import argparse
import dataclasses
import time

import numpy as np

from tablet_tap.app import AppContext
from tablet_tap.app import ArmUse
from tablet_tap.calibration import hand_eye
from tablet_tap.calibration.procedures import fit_report
from tablet_tap.cli.commands.base import Command
from tablet_tap.cli.commands.base import register
from tablet_tap.cli.commands.view import save_view
from tablet_tap.errors import CalibrationError
from tablet_tap.vision.rectifier import detect_markers

CAMERA_SOURCE = "camera"  # setup.records["screen_map_source"]: the screen plane is the camera's estimate, not probed


@register
class Place(Command):
    name = "place"
    help = "平板换了位置：由相机估计屏幕平面（要先做过 board 手眼标定），之后用 probe 精修"
    arm = ArmUse.NONE

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--screen_mm", type=float, nargs=2, metavar=("宽", "高"), help="屏幕发光区域的宽和高（毫米）"
        )
        parser.add_argument("--out_width", type=int, default=1792, help="矫正后屏幕图的宽度（像素）")
        parser.add_argument("--tilt_deg", type=float, default=0.0, help="点按时笔向外倾多少度（远处够不到时用，例如 8）")
        parser.add_argument("--check", action="store_true", help="只和现有的屏幕平面比较，不保存")

    def run(self, ctx: AppContext, args: argparse.Namespace) -> None:
        setup = ctx.setup
        setup.require("camera_pose")
        setup.require("camera_model")
        if args.screen_mm:
            setup.screen_size_m = [args.screen_mm[0] * 1e-3, args.screen_mm[1] * 1e-3]
        elif setup.screen_size_m is None and setup.screen_map is not None:
            setup.screen_size_m = list(setup.screen_map.size_m)  # as the last touch calibration measured it
        setup.require("screen_size_m")

        camera, page = ctx.camera(), ctx.page()
        page.set_mode("markers")
        page.clear()
        time.sleep(0.6)  # the page redraws
        frame = camera.read()
        camera.close()
        if frame.shape[1::-1] != (setup.camera_model.width, setup.camera_model.height):
            raise CalibrationError(
                f"相机现在出的是 {frame.shape[1]}×{frame.shape[0]} 的图，手眼标定时是 "
                f"{setup.camera_model.width}×{setup.camera_model.height}；分辨率要一致。"
            )
        layout = page.layout()
        screen, rms, ratio = hand_eye.screen_from_camera(
            detect_markers(frame), layout, tuple(setup.screen_size_m), setup.camera_model, np.asarray(setup.camera_pose)
        )
        screen = dataclasses.replace(screen, tilt_deg=float(args.tilt_deg))
        print(f"由相机估计的屏幕平面：标定码重投影误差 {rms:.2f} px，倾角判别比 {ratio:.1f}")
        if ratio < 2.0:
            print("⚠ 倾角判别比小于 2：屏幕的倾角估计不可靠，probe 时请留意第一个点。")
        if setup.screen_map is not None:
            gaps = [np.linalg.norm(screen.point(u, v) - setup.screen_map.point(u, v)) * 1e3
                    for u, v in ((0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5))]  # fmt: skip
            print("和现有的屏幕平面相比（平板没动过的话，这就是手眼标定的误差）：四角 "
                  + " / ".join(f"{g:.1f}" for g in gaps[:4]) + f" mm，中心 {gaps[4]:.1f} mm")  # fmt: skip
        if args.check:
            return
        seed = hand_eye.seed_joints(ctx.pen, screen, setup.motion_limits())  # Unreachable: move the tablet closer
        setup.screen_map, setup.seed_q, setup.aspect = screen, seed.tolist(), layout.aspect
        setup.records["screen_map_source"] = CAMERA_SOURCE
        print("\n".join(fit_report(screen, ctx.pen, setup.motion_limits(), setup.seed_q)))
        save_view(ctx, frame, layout, int(args.out_width))
        ctx.save_setup()
        print("这是相机的估计，误差可能有几毫米：先运行 probe（它会从更高处、探得更深），再做 taptest。")
