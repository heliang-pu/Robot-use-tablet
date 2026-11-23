from __future__ import annotations

import argparse
import dataclasses
import datetime
import json

from tablet_tap import config
from tablet_tap.app import AppContext
from tablet_tap.calibration import procedures
from tablet_tap.cli.commands.base import Command
from tablet_tap.cli.commands.base import register
from tablet_tap.cli.commands.probe import add_grid_options


@register
class TapTest(Command):
    name = "taptest"
    help = "点按精度测试：标定页显示靶点，机械臂点，统计落点误差"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        add_grid_options(parser)
        parser.add_argument("--repeat", type=int, default=1, help="整个网格点几遍")

    def run(self, ctx: AppContext, args: argparse.Namespace) -> None:
        robot, page = ctx.robot(), ctx.page()
        points = procedures.grid_points(args.nu, args.nv, args.margin) * args.repeat
        ctx.confirm(f"点按精度测试，{len(points)} 次")

        def say(index: int, outcome: procedures.TapOutcome) -> None:
            if outcome.unreachable:
                result = "机械臂够不到，跳过"
            else:
                result = "没有触发" if outcome.error_mm is None else f"偏差 {outcome.error_mm:.2f} mm"
            print(f"  {index:3d}/{len(points)}  u={outcome.u:.3f} v={outcome.v:.3f}  {result}")

        with ctx.interruptible():
            outcomes = procedures.tap_test(robot, page, points, on_point=say)
        summary = procedures.summarize_taps(outcomes)
        out = config.runs_dir() / f"taptest_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
        out.parent.mkdir(exist_ok=True)
        record = {
            "summary": summary,
            "limits": dataclasses.asdict(robot.limits),
            "points": [dataclasses.asdict(o) for o in outcomes],
        }
        out.write_text(json.dumps(record, indent=2))
        text = f"触发 {summary['registered']} / {summary['n']} 次"
        if summary.get("unreachable"):
            text += f"（另有 {summary['unreachable']} 个点机械臂够不到，没有点）"
        if summary["registered"]:
            text += (
                f"；偏差 平均 {summary['mean_mm']:.2f}，中位 {summary['median_mm']:.2f}，"
                f"95 分位 {summary['p95_mm']:.2f}，最大 {summary['max_mm']:.2f} mm；"
                f"≤ 2 mm 的有 {summary['within_2mm']} / {summary['registered']} 次"
            )
        print(text)
        print(f"结果已保存 {out}（真机测试请同时写一份实验记录）")
