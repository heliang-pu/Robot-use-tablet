"""Stand-alone camera tools (text for the operator is in Chinese):

    bin/py -m tablet_tap.vision.tools calibrate --camera realsense:<serial> --layout-url http://<pc>:8801/layout \
        --out calib/screen_view.json
    bin/py -m tablet_tap.vision.tools rectify --camera realsense:<serial> --calib calib/screen_view.json --out shot.png

The ``view`` and ``shot`` sub-commands of ``bin/py -m tablet_tap`` do the same with the project's setup file.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Sequence

import cv2

from tablet_tap.errors import CalibrationError
from tablet_tap.vision.camera import open_camera
from tablet_tap.vision.layout import ScreenLayout
from tablet_tap.vision.rectifier import ScreenRectifier
from tablet_tap.vision.rectifier import draw_overlay


def _fetch_layout(source: str, timeout_s: float = 5.0) -> tuple[ScreenLayout, dict[str, Any]]:
    """The layout from the calibration server's /layout (or from a JSON file), and the raw JSON."""
    if not source.startswith(("http://", "https://")):
        data = json.loads(pathlib.Path(source).read_text(encoding="utf-8"))
        return ScreenLayout.from_json(data), data
    import requests  # noqa: PLC0415 - only the CLI needs it

    try:
        response = requests.get(source, timeout=timeout_s, proxies={"http": None, "https": None})
    except requests.RequestException as exc:
        raise CalibrationError(f"连不上标定服务 {source}：{exc}") from exc
    try:
        data = response.json()
    except ValueError:
        data = {}
    if response.status_code != 200:
        raise CalibrationError(
            f"标定服务 {source} 返回 {response.status_code}：{data.get('error', response.text[:200])}"
        )
    return ScreenLayout.from_json(data), data


def _calibrate(args: argparse.Namespace) -> int:
    layout, raw = _fetch_layout(args.layout_url)
    if raw.get("covers_display") is False:
        print("警告：平板上的页面没有铺满整个屏幕（没进全屏），这样标定出来的坐标是错的。请先在平板上点“进入全屏”。")
    camera = open_camera(args.camera)
    try:
        frame = camera.read()
    finally:
        camera.close()
    out = pathlib.Path(args.out)
    debug = pathlib.Path(args.debug_image) if args.debug_image else out.with_name(out.stem + "_debug.jpg")
    debug.parent.mkdir(parents=True, exist_ok=True)
    try:
        rectifier = ScreenRectifier.from_marker_image(frame, layout, out_width=args.out_width)
    except CalibrationError as exc:
        cv2.imwrite(str(debug), frame)
        print(f"标定失败：{exc}")
        print(f"这一帧存在 {debug.resolve()}，可以看看相机拍到了什么。")
        return 1
    rectifier.save(out)
    cv2.imwrite(str(debug), draw_overlay(frame, rectifier))
    quad = rectifier.screen_quad_cam()
    print(
        f"标定完成：用了 {len(rectifier.marker_ids)}/{len(layout.markers)} 个标定码（编号 {list(rectifier.marker_ids)}），"
        f"重投影误差 RMS {rectifier.reprojection_px:.2f} 像素（相机图 {frame.shape[1]}x{frame.shape[0]}）。"
    )
    print("屏幕四角在相机图里（左上、右上、右下、左下）：" + "  ".join(f"({x:.0f}, {y:.0f})" for x, y in quad))
    print(f"标定文件：{out.resolve()}")
    print(f"检查图（绿框 = 标定码，黄框 = 屏幕，黄点 = 屏幕左上角）：{debug.resolve()}")
    if len(rectifier.marker_ids) < len(layout.markers):
        print("提示：有标定码没被用上；把挡住的东西（机械臂）移开再标一次会更准。")
    return 0


def _rectify(args: argparse.Namespace) -> int:
    rectifier = ScreenRectifier.load(args.calib)
    camera = open_camera(args.camera)
    try:
        frame = camera.read()
    finally:
        camera.close()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    screen = rectifier.rectify(frame)
    if not cv2.imwrite(str(out), screen):
        print(f"写不了 {out}")
        return 1
    print(f"屏幕图（{screen.shape[1]}x{screen.shape[0]}）：{out.resolve()}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把相机画面变成平板的正视全屏图：标定（calibrate）和取图（rectify）。")
    commands = parser.add_subparsers(dest="command", required=True)
    camera_help = "相机：realsense:<序列号>[:宽x高]、cv:<编号或URL>、file:<图片或目录>"

    calibrate = commands.add_parser("calibrate", help="平板显示标定页时拍一帧，算出相机到屏幕的映射")
    calibrate.add_argument("--camera", required=True, help=camera_help)
    calibrate.add_argument(
        "--layout-url",
        required=True,
        help="标定服务的 /layout 地址，例如 http://127.0.0.1:8801/layout（也可以是 JSON 文件）",
    )
    calibrate.add_argument("--out", required=True, help="标定结果（JSON），例如 calib/screen_view.json")
    calibrate.add_argument(
        "--debug-image", default=None, help="检查图：画出标定码和屏幕四边（默认在 --out 旁边，_debug.jpg）"
    )
    calibrate.add_argument("--out-width", type=int, default=1280, help="之后屏幕图的宽度（像素），默认 1280")
    calibrate.set_defaults(run=_calibrate)

    rectify = commands.add_parser("rectify", help="用已有的标定把一帧相机画面变成屏幕图")
    rectify.add_argument("--camera", required=True, help=camera_help)
    rectify.add_argument("--calib", required=True, help="calibrate 写出的 JSON")
    rectify.add_argument("--out", required=True, help="输出的屏幕图，例如 shot.png")
    rectify.set_defaults(run=_rectify)

    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except (CalibrationError, RuntimeError, ValueError, OSError) as exc:
        print(f"出错：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
