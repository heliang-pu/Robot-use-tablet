from __future__ import annotations

import argparse
import dataclasses

from tablet_tap.app import AppContext
from tablet_tap.app import ArmUse
from tablet_tap.calibration.procedures import fit_report
from tablet_tap.calibration.teach import MIN_TOUCHES
from tablet_tap.calibration.teach import TeachSession
from tablet_tap.calibration.teach import raw_keys
from tablet_tap.cli.commands.base import Command
from tablet_tap.cli.commands.base import register
from tablet_tap.errors import SetupError


@register
class Teach(Command):
    name = "teach"
    help = "键盘点动 + 触点示教 → 屏幕平面"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--read_only", action="store_true", help="不发指令，只记录（配合拖动示教或主臂）")
        parser.add_argument("--no_page", action="store_true", help="不连标定页（只点动、记停靠位姿）")

    def arm_use(self, args: argparse.Namespace) -> ArmUse:
        return ArmUse.READS if args.read_only else ArmUse.MOVES

    def run(self, ctx: AppContext, args: argparse.Namespace) -> None:
        setup = ctx.setup
        page, aspect = None, None
        if not args.no_page:
            page = ctx.page()
            screen = page.screen()
            if not screen:
                raise SetupError(f"标定页还没有连上：先在平板浏览器里打开 {ctx.options.calib_url}/calib 并进入全屏。")
            aspect = float(screen["inner_w"]) / float(screen["inner_h"])
            page.set_mode("markers")
        session = TeachSession(
            ctx.arm,
            ctx.pen,
            setup.motion_limits(),
            page=page,
            moves_arm=not args.read_only,
            should_stop=ctx.stop.is_set,
            realtime=ctx.realtime,
        )
        if not args.read_only:
            ctx.confirm("键盘点动")
        with raw_keys() as get_key, ctx.interruptible():
            session.run(get_key)

        if session.park_q is not None:
            setup.park_q = session.park_q
        fitted = session.fit(aspect)  # CalibrationError when the touches do not fit a screen
        if fitted is None:
            print(f"只有 {len(session.touches)} 个触点，不足 {MIN_TOUCHES} 个，不拟合。")
            if session.park_q is not None:
                ctx.save_setup()
            return
        setup.screen_map, setup.seed_q = fitted
        setup.aspect, setup.screen_size_m = aspect, list(setup.screen_map.size_m)
        setup.records["screen_map_source"] = "touch"
        setup.records["teach_touches"] = [dataclasses.asdict(touch) for touch in session.touches]
        print("\n".join(fit_report(setup.screen_map, ctx.pen, setup.motion_limits(), setup.seed_q)))
        ctx.save_setup()
