from __future__ import annotations

import argparse
import datetime
import json
import pathlib

import cv2
import numpy as np

from tablet_tap.app import AppContext
from tablet_tap.app import ArmUse
from tablet_tap.calibration import hand_eye
from tablet_tap.calibration.teach import TeachSession
from tablet_tap.calibration.teach import raw_keys
from tablet_tap.cli.commands.base import Command
from tablet_tap.cli.commands.base import register
from tablet_tap.core.pinhole import CameraModel
from tablet_tap.errors import CalibrationError
from tablet_tap.errors import SetupError
from tablet_tap.vision.board import BoardSpec
from tablet_tap.vision.board import draw_board
from tablet_tap.vision.board import find_chessboard
from tablet_tap.vision.board import numbered_points
from tablet_tap.vision.camera import camera_model

BENCH_BOARD_SQUARE_MM = 15.0  # the bench's glass board: 12 x 9 squares; the user gave the size on 2026-11-23

HOW = """
手眼标定：让相机和机械臂用同一块标定板对上坐标。
  1. 看 {picture}：图上用红圈和数字标了 {count} 个角点（棋盘格最外圈的内角点）。
  2. 按图上的编号顺序，把笔尖依次放到这些角点上，每放好一个按 m 记录；够不到的按 n 跳过（至少要记 3 个）。
     靠近板面时用小步长（按 1 或 2），眼睛平视笔尖，刚碰到就停。记错了按 b 删掉最后一个。
  3. {count} 个都处理完按 q。
点角点时尽量保持笔的朝向不变（只平移，不转动）：笔尖长度量得不准时，转动会把误差带进结果。
"""


@register
class Board(Command):
    name = "board"
    help = "手眼标定：相机看棋盘格标定板 + 笔尖点它的角点 → 相机在机械臂坐标系里的位姿"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--square_mm",
            type=float,
            default=BENCH_BOARD_SQUARE_MM,
            help=f"棋盘格一格的边长（毫米）；默认 {BENCH_BOARD_SQUARE_MM:g}，是台子上那块 12×9 格玻璃板的格距，换板要改",
        )
        parser.add_argument("--squares", help="格数，例如 12x9；不给就自动识别")
        parser.add_argument(
            "--points",
            type=int,
            choices=(4, 8),
            default=4,
            help="要点几个角点：4 个最外侧的角，或再加四条边的中点共 8 个（更准）",
        )
        parser.add_argument("--glass_mm", type=float, default=0.0, help="图案在玻璃背面时填玻璃厚度（笔点的是上表面）")
        parser.add_argument("--intrinsics", help="相机内参的 JSON 文件；RealSense 不用给，读出厂值")
        parser.add_argument("--frames", type=int, default=10, help="取多少帧的角点求平均")
        parser.add_argument("--read_only", action="store_true", help="不发指令：用拖动示教把笔带到角点上，再按 m")

    def arm_use(self, args: argparse.Namespace) -> ArmUse:
        return ArmUse.READS if args.read_only else ArmUse.MOVES

    def run(self, ctx: AppContext, args: argparse.Namespace) -> None:
        cols, rows = self._inner_corners(args.squares)
        if not ctx.options.yes and ctx.commands_arm:
            input("先让机械臂离开，不要挡住标定板，然后按回车拍照 > ")
        camera = ctx.camera()
        try:
            model = self._camera_model(camera, args.intrinsics)
            frames = [camera.read() for _ in range(max(args.frames, 1))]
        finally:
            camera.close()
        grid = np.mean([find_chessboard(frame, cols, rows) for frame in frames], axis=0)
        spec = BoardSpec(
            cols=grid.shape[1], rows=grid.shape[0], pitch_m=args.square_mm * 1e-3, glass_m=args.glass_mm * 1e-3
        )
        _, rms, ratio = hand_eye.board_in_camera(spec, grid, model)
        numbered = numbered_points(grid, args.points)
        picture = ctx.options.setup_path.parent / "board_corners.jpg"
        picture.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(picture), draw_board(frames[-1], grid, numbered))
        print(
            f"标定板：{spec.cols + 1} × {spec.rows + 1} 格（{spec.cols} × {spec.rows} 个内角点），格距 {args.square_mm:g} mm；"
            f"在画面里重投影误差 {rms:.2f} px，倾角判别比 {ratio:.1f}"
        )
        if ratio < 2.0:
            print("⚠ 倾角判别比小于 2：标定板在画面里太小或太正，它的倾角不可靠。让相机离近一些或换高分辨率再做。")

        session = TeachSession(
            ctx.arm,
            ctx.pen,
            ctx.setup.motion_limits(),
            page=None,
            moves_arm=not args.read_only,
            should_stop=ctx.stop.is_set,
            realtime=ctx.realtime,
        )
        print(HOW.format(picture=picture, count=len(numbered)))
        if not args.read_only:
            ctx.confirm("键盘点动去点标定板的角点")
        with raw_keys() as get_key, ctx.interruptible():
            session.run(get_key)

        marks = session.marks[: len(numbered)]
        joints = [(row, col, np.asarray(q)) for (row, col), q in zip(numbered, marks, strict=False) if q is not None]
        if len(joints) < hand_eye.MIN_TOUCHES:
            raise CalibrationError(f"只记录了 {len(joints)} 个角点，至少要 {hand_eye.MIN_TOUCHES} 个；没有保存。")
        touches = [hand_eye.BoardTouch(row, col, ctx.pen.fk(q)) for row, col, q in joints]
        result = hand_eye.solve_hand_eye(spec, grid, model, touches)
        spread = hand_eye.uncertainty(spec, grid, model, touches)
        tip = hand_eye.tip_length_check(ctx.pen, spec, joints)

        print(
            f"各触点到拟合板面的残差：{result.touch_residual_mm} mm；机械臂量出的角点间距和板的真实间距最多差 {result.span_error_mm:.2f} mm"
        )
        print(
            f"按人眼对点 ±{spread['touch_sigma_mm']:g} mm、角点 ±{spread['corner_sigma_px']:g} px 估计的不确定度（标准差）："
            f"板中心 {spread['centre_mm']:.1f} mm，板外 {spread['outside_m'] * 100:.0f} cm 处 {spread['outside_mm']:.1f} mm，"
            f"板面法线 {spread['normal_deg']:.2f}°"
        )
        if tip is not None and abs(tip["tip_len_mm"] - ctx.pen.tip_len * 1e3) > 1.5:
            print(
                f"⚠ 这些触点更符合笔尖长度 {tip['tip_len_mm']:.1f} mm（现在配置的是 {ctx.pen.tip_len * 1e3:.1f} mm，"
                f"残差 {tip['rms_now_mm']:.2f} → {tip['rms_mm']:.2f} mm）。量一下实际长度，用 --tip_len_mm 改正后重做。"
            )
        print(hand_eye.describe_camera(result.camera_in_base))
        if not ctx.options.yes and input("这和台子上相机的实际位置相符吗？相符才保存 [y/N] ").strip().lower() != "y":
            raise SetupError("没有保存。位置不对通常是角点编号对反了（转了 180°）：对照图片重新点一遍。")

        setup = ctx.setup
        setup.camera_model, setup.camera_pose = model, result.camera_in_base.tolist()
        setup.records["hand_eye"] = {
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "board": {"cols": spec.cols, "rows": spec.rows, "square_mm": args.square_mm, "glass_mm": args.glass_mm},
            "touched": [{"row": row, "col": col, "q": q.tolist()} for row, col, q in joints],
            "uncertainty": spread,
            **result.to_json(),
        }
        ctx.save_setup()
        print("下一步：把平板放好、打开标定页，运行 place 由相机估计屏幕位置，再运行 probe 精修。")

    @staticmethod
    def _inner_corners(squares: str | None) -> tuple[int | None, int | None]:
        if not squares:
            return None, None
        try:
            a, b = (int(n) for n in squares.lower().replace("×", "x").split("x"))
        except ValueError:
            raise SetupError(f"--squares 要写成 列数x行数，例如 12x9；收到的是 {squares!r}。") from None
        return a - 1, b - 1

    @staticmethod
    def _camera_model(camera, path: str | None) -> CameraModel:
        if path:
            return CameraModel.from_json(json.loads(pathlib.Path(path).read_text()))
        model = camera_model(camera)
        if model is None:
            raise SetupError(
                "这个相机读不到内参：用 --intrinsics 给一个含 width、height、fx、fy、cx、cy 的 JSON 文件。"
            )
        return model
