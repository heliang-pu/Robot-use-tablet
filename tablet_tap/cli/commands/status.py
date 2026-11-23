from __future__ import annotations

import argparse

import numpy as np

from tablet_tap.app import AppContext
from tablet_tap.app import ArmUse
from tablet_tap.app import require_hardware
from tablet_tap.cli.commands.base import Command
from tablet_tap.cli.commands.base import register
from tablet_tap.robot.factory import ArmMode
from tablet_tap.robot.piper import describe_bus
from tablet_tap.robot.piper import other_arm_programs


@register
class Status(Command):
    name = "status"
    help = "只读：CAN 反馈是否正常、关节角和笔尖位姿"
    arm = ArmUse.READS

    def run(self, ctx: AppContext, args: argparse.Namespace) -> None:
        if ctx.arm_mode is not ArmMode.FAKE and ctx.setup.arm.channel:
            require_hardware("监听机械臂的 CAN 通道")
            print(describe_bus(ctx.setup.arm.channel))
        q = ctx.arm.read()
        tip = ctx.pen.fk(q)
        print(f"关节角 (rad)：{np.round(q, 4).tolist()}")
        print(f"笔尖位置 (m)：{np.round(tip[:3, 3], 4).tolist()}，笔轴方向：{np.round(tip[:3, 2], 3).tolist()}")
        if ctx.setup.screen_map is not None:
            u, v, h = ctx.setup.screen_map.uv_of(tip[:3, 3])
            print(f"相对屏幕：u={u:.3f} v={v:.3f}，高 {h * 1e3:.1f} mm")
        running = other_arm_programs()
        if running:
            print("正在运行的其它机械臂程序：\n  " + "\n  ".join(line[:140] for line in running))
