from __future__ import annotations

import argparse
import dataclasses

import numpy as np

from tablet_tap.app import AppContext
from tablet_tap.calibration import procedures
from tablet_tap.cli.commands.base import Command
from tablet_tap.cli.commands.base import register
from tablet_tap.errors import CalibrationError
from tablet_tap.errors import Unreachable

TOUCH_DEPTH_MM = 6.0
CAMERA_DEPTH_MM = 3.0  # with a screen plane that is the camera's estimate
CAMERA_HOVER_M = 0.02  # the estimate is good to a few millimetres; higher costs reach (the wrist runs out)


def add_grid_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--nu", type=int, default=5, help="网格的列数")
    parser.add_argument("--nv", type=int, default=4, help="网格的行数")
    parser.add_argument("--margin", type=float, default=0.08, help="网格离屏幕边缘的距离（屏幕宽高的比例）")


@register
class Probe(Command):
    name = "probe"
    help = "自动探触网格 → 重新拟合屏幕平面"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        add_grid_options(parser)
        parser.add_argument(
            "--depth_mm",
            type=float,
            default=None,
            help=f"最多探到原平面以下多少毫米；默认 {TOUCH_DEPTH_MM:g}，屏幕平面是相机估计的（place）时 {CAMERA_DEPTH_MM:g}",
        )

    def run(self, ctx: AppContext, args: argparse.Namespace) -> None:
        from_camera = ctx.setup.records.get("screen_map_source") == "camera"
        depth_mm = args.depth_mm if args.depth_mm is not None else (CAMERA_DEPTH_MM if from_camera else TOUCH_DEPTH_MM)
        limits = ctx.setup.motion_limits()
        if from_camera:  # the plane may be off by millimetres: stay higher above it between the points
            limits = dataclasses.replace(limits, hover_m=max(limits.hover_m, CAMERA_HOVER_M))
            print(f"屏幕平面来自相机估计：悬停高度提到 {limits.hover_m * 1e3:.0f} mm，探触深度 {depth_mm:g} mm。")
        robot, page = ctx.robot(limits), ctx.page()
        points = procedures.grid_points(args.nu, args.nv, args.margin)
        ctx.confirm(f"自动探触 {len(points)} 个点")

        def say(index: int, at: tuple[float, float], point: procedures.ProbePoint | Unreachable | None) -> None:
            where = f"  {index:2d}/{len(points)}  u={at[0]:.3f} v={at[1]:.3f}  "
            if isinstance(point, Unreachable):
                print(where + "机械臂够不到，跳过")
            elif point is None:
                print(where + f"屏幕没有报告触摸（下压 {depth_mm:g} mm 仍未触发）")
            else:
                print(where + f"落点偏差 {point.error_mm:.2f} mm，接触面比原平面高 {point.height_mm:+.2f} mm")

        with ctx.interruptible():
            found = procedures.probe_grid(robot, page, points, depth_mm * 1e-3, on_point=say)
        if found:
            errors = np.array([p.error_mm for p in found])
            print(
                f"原标定：{len(found)} / {len(points)} 个点触发，"
                f"落点偏差 平均 {errors.mean():.2f} mm，最大 {errors.max():.2f} mm"
            )
        try:
            fitted = procedures.refit(found, ctx.setup.aspect, robot.screen.tilt_deg)
        except CalibrationError as exc:
            raise CalibrationError(f"重新拟合失败（原标定未改动）：{exc}") from None
        ctx.setup.screen_map, ctx.setup.screen_size_m = fitted, list(fitted.size_m)
        ctx.setup.records["probe_points"] = [dataclasses.asdict(p) for p in found]
        ctx.setup.records["screen_map_source"] = "touch"
        print("\n".join(procedures.fit_report(fitted, ctx.pen, ctx.setup.motion_limits(), ctx.setup.seed_q)))
        ctx.save_setup()
