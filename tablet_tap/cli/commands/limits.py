from __future__ import annotations

import argparse
import datetime
import json
import pathlib

import numpy as np
import requests

from tablet_tap.app import AppContext
from tablet_tap.app import ArmUse
from tablet_tap.app import require_hardware
from tablet_tap.cli.commands.base import Command
from tablet_tap.cli.commands.base import register
from tablet_tap.errors import SetupError
from tablet_tap.robot.piper import other_arm_programs

JOINT5_RANGE_DEG = (70.0, 75.0)  # the factory value, and what the left arm was moved to in free air on 2026-11-23


def saved_path(ctx: AppContext) -> pathlib.Path:
    return ctx.store.path.parent / f"firmware_limits_{ctx.setup.arm.channel}.json"


def remember(ctx: AppContext, limits: dict[int, tuple[float, float]]) -> bool:
    """Keep what the firmware says in the setup file, where planning and the driver's clamp take this arm's limits
    from. Only that row is written (not what the command line changed for this one run). True: the row is new or has
    changed."""
    channel = ctx.setup.arm.channel
    rows = [[round(float(np.degrees(v)), 1) for v in limits[joint]] for joint in range(1, 7)]
    if ctx.setup.arm.firmware_limits_deg.get(channel) == rows:
        return False
    ctx.setup.arm.firmware_limits_deg[channel] = rows
    on_disk = ctx.store.load()
    on_disk.arm.firmware_limits_deg[channel] = rows
    ctx.store.save(on_disk)
    return True


def console_holds_arm(url: str) -> bool:
    """Whether the browser console (another process) is connected to the arm; False when it does not answer."""
    session = requests.Session()
    session.trust_env = False
    try:
        return bool(session.get(f"{url.rstrip('/')}/api/state", timeout=2.0).json()["arm"]["connected"])
    except (requests.RequestException, ValueError, KeyError):
        return False


@register
class Limits(Command):
    name = "limits"
    help = "机械臂固件里的关节限位：查看；改 5 号关节的上限（笔要垂直够远处时用）；还原"
    arm = ArmUse.READS

    def configure(self, parser: argparse.ArgumentParser) -> None:
        low, high = JOINT5_RANGE_DEG
        parser.add_argument("--joint5_max", type=float, metavar="度", help=f"把 5 号关节的上限写成这个角度（{low:g}–{high:g}）")
        parser.add_argument("--restore", action="store_true", help="把 5 号关节的上限写回第一次改动之前的值")

    def run(self, ctx: AppContext, args: argparse.Namespace) -> None:
        arm = ctx.arm
        if not hasattr(arm, "firmware_joint_limits"):
            print("这只（模拟）手臂没有固件限位。")
            return
        require_hardware("读写机械臂的固件限位")
        now = arm.firmware_joint_limits()
        show(now)
        if remember(ctx, now):
            print(f"规划用的限位已经按它更新（记在 {ctx.store.path}）。")
        path = saved_path(ctx)
        saved = json.loads(path.read_text()) if path.exists() else None
        if saved is not None:
            print(f"第一次改动之前的值记在 {path}：5 号上限 {saved['limits_deg']['5'][1]:g}°")
        if args.restore and args.joint5_max is not None:
            raise SetupError("--restore 和 --joint5_max 只能给一个。")
        if args.restore:
            if saved is None:
                raise SetupError(f"没有 {path}：不知道原来的值，没有改动。")
            wanted = float(saved["limits_deg"]["5"][1])
        elif args.joint5_max is not None:
            wanted = float(args.joint5_max)
        else:
            return
        low, high = JOINT5_RANGE_DEG
        if not low <= wanted <= high:
            raise SetupError(f"5 号关节的上限只能写成 {low:g}–{high:g}°（再大没有试过，可能顶到机械限位）。")
        running = other_arm_programs()
        if running:
            raise SetupError("别的程序正在用机械臂，没有改动：\n  " + "\n  ".join(line[:120] for line in running))
        if console_holds_arm(ctx.options.calib_url):
            raise SetupError("网页控制台正连着机械臂：先在网页里点“断开”，再改限位。")
        if not ctx.options.yes:
            answer = input(f"把 5 号关节的上限写成 {wanted:g}°？这个值留在手臂的固件里，别的程序也按它走 [y/N] ")
            if answer.strip().lower() != "y":
                raise SystemExit("已取消。")
        if saved is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {"when": datetime.datetime.now().isoformat(timespec="seconds"), "channel": ctx.setup.arm.channel,
                      "limits_deg": {str(j): [round(float(np.degrees(v)), 2) for v in pair] for j, pair in now.items()}}  # fmt: skip
            path.write_text(json.dumps(record, indent=2))
        if not arm.set_firmware_joint_limit(5, highest=float(np.radians(wanted))):
            raise SetupError("手臂没有确认这次写入；用 limits 再看一下现在的值。")
        after = arm.firmware_joint_limits()
        others = [j for j in now if j != 5 and not np.allclose(now[j], after[j], atol=2e-3)]
        if abs(np.degrees(after[5][1]) - wanted) > 0.15 or others:
            raise SetupError(f"写入后读回来不对（5 号上限 {np.degrees(after[5][1]):.1f}°，别的关节变了：{others}）。")
        remember(ctx, after)
        print(f"5 号关节的上限现在是 {wanted:g}°，规划也按它走。手臂没有动。")


def show(limits: dict[int, tuple[float, float]]) -> None:
    print("固件里的关节限位（度）：" + "，".join(f"{j} 号 {np.degrees(a):.0f}…{np.degrees(b):.0f}" for j, (a, b) in limits.items()))
