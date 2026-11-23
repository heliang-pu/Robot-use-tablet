"""tablet_tap 的命令行：示教标定、点按、整条链路（语音 → GUI 智能体 → 机械臂点屏幕）。

    bin/py -m tablet_tap <子命令> [选项]          （bin/py -m tablet_tap <子命令> -h 看这条子命令的选项）

所有会动机械臂的子命令都要求人在急停旁边，并在第一次运动前确认；--dry_run 只读真实关节、在软件里走一遍，
--fake 完全不连硬件。详见 README.md。
"""

from __future__ import annotations

import argparse
import logging

from tablet_tap import config
from tablet_tap.app import AppContext
from tablet_tap.app import Options
from tablet_tap.cli import commands as command_registry
from tablet_tap.errors import ArmConnectionError
from tablet_tap.errors import CalibrationError
from tablet_tap.errors import PathError
from tablet_tap.errors import SetupError
from tablet_tap.errors import Stopped
from tablet_tap.errors import TabletTapError
from tablet_tap.errors import Unreachable

# what each kind of error means for the caller: (exit code, what is printed before the message)
EXITS: tuple[tuple[type[TabletTapError], int, str], ...] = (
    (Stopped, 130, "已停止："),
    (PathError, 2, "未执行："),
    (Unreachable, 2, "未执行："),
    (ArmConnectionError, 3, ""),
    (SetupError, 1, ""),
    (CalibrationError, 1, ""),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tablet_tap", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--setup", default=str(config.DEFAULT_SETUP_PATH), help="标定和配置文件")
    parser.add_argument("--view", default=str(config.DEFAULT_VIEW_PATH), help="相机到屏幕的标定文件")
    parser.add_argument(
        "--channel", help="拿笔那只机械臂的 CAN 通道（can_l_follower / can_r_follower），第一次要给，之后记在配置里"
    )
    parser.add_argument(
        "--camera",
        help="看屏幕的相机：realsense:<序列号>[:宽x高] | cv:<编号或地址> | file:<路径>，记在配置里；"
        f"不给也没记过时用台子的顶部相机 {config.BENCH_TOP_CAMERA}",
    )
    parser.add_argument("--calib_url", default="http://127.0.0.1:8801", help="标定页服务的地址")
    parser.add_argument(
        "--press_mm",
        type=float,
        help="点按时压到标定接触面以下多少毫米（记在配置里）。带簧片笔夹用 2.5；夹爪直接夹笔（没有弹性）用 0.5–0.8",
    )
    parser.add_argument("--tip_len_mm", type=float, help="笔尖超出指尖端面的长度（记在配置里），默认 65")
    parser.add_argument("--dry_run", action="store_true", help="只读真实关节角，运动只在软件里模拟，不发指令")
    parser.add_argument("--fake", action="store_true", help="不连任何硬件（软件机械臂）")
    parser.add_argument("--yes", action="store_true", help="跳过运动前的确认（确认过现场安全后再用）")
    parser.add_argument("--force", action="store_true", help="即使 run_real 等程序在运行也连接机械臂（不要常用）")
    sub = parser.add_subparsers(dest="command", required=True, metavar="<子命令>")
    for command in command_registry.commands():
        command.configure(sub.add_parser(command.name, help=command.help))
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    args = build_parser().parse_args(argv)
    command = next(c for c in command_registry.commands() if c.name == args.command)
    try:
        command.run(AppContext(Options.from_args(args), command.arm_use(args)), args)
    except TabletTapError as exc:
        for kind, code, prefix in EXITS:
            if isinstance(exc, kind):
                print(f"{prefix}{exc}")
                return code
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
