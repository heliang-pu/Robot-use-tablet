from __future__ import annotations

import argparse
import datetime

from tablet_tap import config
from tablet_tap.agent.gui_agent import GuiAgent
from tablet_tap.agent.gui_agent import add_info_for
from tablet_tap.agent.gui_agent import resolve_add_info
from tablet_tap.app import AppContext
from tablet_tap.cli.commands.base import Command
from tablet_tap.cli.commands.base import register
from tablet_tap.robot.tablet_robot import ArmPlace


@register
class Run(Command):
    name = "run"
    help = "整条链路：一句指令 → GUI 智能体一步步点屏幕"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--instruction", help="文字指令；不给就用麦克风")
        parser.add_argument("--loop", action="store_true", help="语音模式下一条做完接着听下一条")
        parser.add_argument("--vlm_url", default="http://127.0.0.1:8802/v1", help="GUI-Owl 服务的地址")
        parser.add_argument("--model", default="gui-owl")
        parser.add_argument("--asr_url", default="http://127.0.0.1:8803", help="语音识别服务的地址")
        parser.add_argument("--mic", default=None, help="arecord 的录音设备，例如 plughw:1,0")
        parser.add_argument("--max_steps", type=int, default=15)
        parser.add_argument(
            "--add_info",
            default=None,
            help="附加在指令后面、告诉模型本设备限制的说明：zh / en（2B 模型用它更准）/ none，或直接给一段文字；"
            "默认 zh，标定过屏幕键盘时用会打字的那一段",
        )
        parser.add_argument(
            "--settle_s", type=float, default=0.8, help="每个动作后等界面变化的秒数（回停靠位姿的时间另算）"
        )

    def run(self, ctx: AppContext, args: argparse.Namespace) -> None:
        device = ctx.device()
        ctx.confirm("按指令点屏幕")
        while True:
            instruction = args.instruction or self._listen(args)
            if not instruction.strip():
                continue
            ctx.stop.clear()
            agent = GuiAgent(
                device,
                base_url=args.vlm_url,
                model=args.model,
                max_steps=args.max_steps,
                settle_s=args.settle_s,
                add_info=add_info_for(typing=getattr(device, "keyboard", None) is not None)
                if args.add_info is None
                else resolve_add_info(args.add_info),
                should_stop=ctx.stop.is_set,
                run_dir=config.runs_dir() / f"{datetime.datetime.now():%Y%m%d_%H%M%S}",
            )
            with ctx.interruptible():
                try:
                    result = agent.run(instruction)
                finally:
                    if device.robot.where is ArmPlace.HOVER and not ctx.stop.is_set():
                        device.robot.to_park()
            print(
                f"结果：{result.status}，{result.steps} 步"
                + (f"，{result.answer}" if result.answer else "")
                + f"\n记录在 {result.run_dir}"
            )
            if args.instruction or not args.loop:
                break

    @staticmethod
    def _listen(args: argparse.Namespace) -> str:
        from tablet_tap.speech import voice  # noqa: PLC0415 - only the voice mode needs it

        instruction = voice.listen_once(url=args.asr_url, device=args.mic)
        print(f"听到：{instruction}")
        return instruction
