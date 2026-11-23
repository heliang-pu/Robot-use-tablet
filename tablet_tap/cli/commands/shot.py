from __future__ import annotations

import argparse
import pathlib

from tablet_tap import config
from tablet_tap.app import AppContext
from tablet_tap.cli.commands.base import Command
from tablet_tap.cli.commands.base import register


@register
class Shot(Command):
    name = "shot"
    help = "停靠后拍一张矫正的屏幕图（智能体会看到的图）"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--out", default=None, help="输出文件，默认 runs/shot.png")

    def run(self, ctx: AppContext, args: argparse.Namespace) -> None:
        device = ctx.device()
        ctx.confirm("回停靠位姿拍照")
        with ctx.interruptible():
            image = device.screenshot()
        out = pathlib.Path(args.out) if args.out else config.runs_dir() / "shot.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        image.save(out)
        print(f"已保存 {out}（{image.width} × {image.height}）")
