from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import time

from tablet_tap import android
from tablet_tap.app import AppContext
from tablet_tap.app import ArmUse
from tablet_tap.cli.commands.base import Command
from tablet_tap.cli.commands.base import register
from tablet_tap.errors import SetupError

NAV_KEYS = ["secure navigation_mode", *(f"{ns} {key}" for ns, key, _ in android.MAKER_NAV_SETTINGS.values())]


@register
class Tablet(Command):
    name = "tablet"
    help = "经 adb 把平板设置成适合标定和演示的样子（横屏、不息屏、界面放大、三键导航），可还原"
    arm = ArmUse.NONE

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--serial", help="adb 上有多台设备时指明平板的序列号")
        parser.add_argument("--show", action="store_true", help="只显示平板现在的状态和将要改什么，不改")
        parser.add_argument("--restore", metavar="文件", help="按 calib/tablet_before_*.json 把平板还原")
        parser.add_argument(
            "--page", action="store_true", help="设置完在平板上打开标定页（经 USB 转发，不需要平板联网）"
        )

    def run(self, ctx: AppContext, args: argparse.Namespace) -> None:
        adb = android.Adb(args.serial)
        if args.restore:
            before = android.Snapshot.from_json(json.loads(pathlib.Path(args.restore).read_text()))
            android.restore(adb, before)
            print(f"已按 {args.restore} 还原 {before.model} 的 {len(before.settings)} 项设置和显示大小。")
            return
        display = android.read_display(adb)
        wanted = android.wanted_settings(display)
        density = android.largest_density(display)
        keys = sorted({*wanted, *NAV_KEYS})
        before = android.snapshot(adb, keys)
        width_mm, height_mm = display.landscape_mm
        print(
            f"平板：{before.model}（{adb.serial}），Android {adb.prop('ro.build.version.release')}；"
            f"屏幕 {display.landscape_px[0]} × {display.landscape_px[1]} 像素，发光区域约 {width_mm:.1f} × {height_mm:.1f} mm"
        )
        changes = [
            (key, before.settings.get(key), value) for key, value in wanted.items() if before.settings.get(key) != value
        ]
        for key, old, new in changes:
            print(f"  {key}: {old} → {new}")
        if density != display.density:
            scale = density / display.factory_density
            print(
                f"  显示大小（density）: {display.density} → {density}（出厂 {display.factory_density} 的 {scale:.2f} 倍）"
            )
        if args.show:
            print("（--show：以上只是将要做的改动，没有执行。）")
            return

        saved = ctx.options.setup_path.parent / f"tablet_before_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_text(json.dumps(before.to_json(), indent=2, ensure_ascii=False))
        print(f"原来的设置已存到 {saved}（还原：bin/py -m tablet_tap tablet --restore {saved}）")
        for key, _, value in changes:
            adb.put_setting(*key.split(), value)
        if density != display.density:
            adb.shell(f"wm density {density}")
        in_three_button_mode = android.three_button_navigation(adb)
        time.sleep(1.5)  # the system bars redraw
        display = android.read_display(adb)
        wrong = [key for key, value in wanted.items() if adb.setting(*key.split()) != value]
        if wrong:
            print(f"⚠ 这些设置平板没有接受，需要在平板的设置里手动改：{wrong}")
        screen_px = display.landscape_px if display.is_landscape else display.landscape_px[::-1]
        buttons = android.nav_buttons(android.dump_window(adb), screen_px)

        setup = ctx.setup
        setup.screen_size_m = [round(width_mm * 1e-3, 5), round(height_mm * 1e-3, 5)]
        setup.aspect = display.landscape_px[0] / display.landscape_px[1]
        if {"back", "home"} <= set(buttons):
            setup.nav = {"back": buttons["back"], "home": buttons["home"]}
            print(f"三键导航已开：返回键在 (u, v) = {buttons['back']}，主页键在 {buttons['home']}，已记进配置。")
        else:
            setup.nav = None
            state = "系统说已经是三键模式，但" if in_three_button_mode else ""
            print(f"⚠ {state}屏幕上没有找到返回 / 主页按键：请在平板“设置 → 系统导航方式”里手动选三键导航，再运行一次本命令。"
                  "在那之前程序用边缘滑动手势代替。")  # fmt: skip
        setup.records["tablet"] = {"model": before.model, "serial": adb.serial, "density": display.density,
                                   "pixels": list(display.landscape_px), "before": str(saved)}  # fmt: skip
        ctx.save_setup()
        if not display.is_landscape:
            raise SetupError("平板现在还是竖屏：把它横过来放，或在平板上手动转一次屏。")
        if args.page:
            port = int(ctx.options.calib_url.rsplit(":", 1)[-1].split("/")[0])
            adb.reverse(port)
            adb.shell(f"am start -a android.intent.action.VIEW -d http://localhost:{port}/calib")
            print(
                f"已在平板上打开标定页 http://localhost:{port}/calib（经 USB 转发）。请在平板上点一下“点这里进入全屏”。"
            )
