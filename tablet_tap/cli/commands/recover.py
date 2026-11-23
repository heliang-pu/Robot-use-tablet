from __future__ import annotations

import argparse
import runpy
import sys

import tablet_tap
from tablet_tap.app import AppContext
from tablet_tap.app import ArmUse
from tablet_tap.app import require_hardware
from tablet_tap.cli.commands.base import Command
from tablet_tap.cli.commands.base import register
from tablet_tap.errors import ArmBusyError
from tablet_tap.robot.piper import other_arm_programs


@register
class Recover(Command):
    """Runs RoboSynChallenge's reset_arms.py (third_party/robosyn), which talks to the arms itself."""

    name = "recover"
    help = "急停 / 碰撞 / 示教模式之后恢复并重新使能从臂（third_party/robosyn/reset_arms.py）"
    arm = ArmUse.NONE

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--arm", choices=("both", "left", "right"), default="both")
        parser.add_argument("--status_only", action="store_true", help="只打印两只从臂的状态和故障，不发任何指令")
        parser.add_argument("--fold", action="store_true", help="恢复后再把手臂低速收回零位（默认不动，只恢复和使能）")

    def run(self, ctx: AppContext, args: argparse.Namespace) -> None:
        argv = ["reset_arms.py", "--arm", args.arm]
        if args.status_only:
            argv.append("--status_only")
        elif not args.fold:
            argv.append("--no_move")
        if ctx.options.yes:
            argv.append("--yes")
        require_hardware("连接机械臂")
        running = other_arm_programs()
        if running and not args.status_only and not ctx.options.force:
            raise ArmBusyError("另一个程序正在控制机械臂，先停掉它再恢复：\n  " + "\n  ".join(running))
        sys.argv = argv
        runpy.run_path(str(tablet_tap.VENDORED / "reset_arms.py"), run_name="__main__")
