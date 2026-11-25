"""The wrist camera: plan where to look from, calibrate it, see the screen with it, run the agent with it.

Nothing here changes ``calib/setup.json`` or the top camera's calibration: the wrist camera has ``calib/wrist.json`` and
``calib/wrist_view.json``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import time

import cv2
import numpy as np

from tablet_tap import config
from tablet_tap.agent.robot_device import RobotTabletDevice
from tablet_tap.app import AppContext
from tablet_tap.app import ArmUse
from tablet_tap.calibration.eye_in_hand import edge_width_px
from tablet_tap.calibration.eye_in_hand import mount_error
from tablet_tap.calibration.eye_in_hand import solve_eye_in_hand
from tablet_tap.calibration.tap_correction import TapCorrection
from tablet_tap.calibration.wrist_procedures import collect_views
from tablet_tap.calibration.wrist_procedures import look_quality
from tablet_tap.calibration.wrist_procedures import still_picture
from tablet_tap.cli.commands.base import Command
from tablet_tap.cli.commands.base import register
from tablet_tap.cli.commands.moves import PARK_SPEED
from tablet_tap.cli.commands.run import Run
from tablet_tap.core.look import CALIBRATION_RULES
from tablet_tap.core.look import MOUNTS
from tablet_tap.core.look import camera_in_flange
from tablet_tap.core.motion import joint_move
from tablet_tap.core.screen_map import ScreenMap
from tablet_tap.errors import CalibrationError
from tablet_tap.errors import PathError
from tablet_tap.errors import SetupError
from tablet_tap.robot.executor import START_TOL_RAD
from tablet_tap.robot.executor import run_segments
from tablet_tap.robot.tablet_robot import ArmPlace
from tablet_tap.robot.wrist import LookRobot
from tablet_tap.robot.wrist import WristStore
from tablet_tap.vision.camera import camera_model
from tablet_tap.vision.layout import default_layout
from tablet_tap.vision.rectifier import ScreenRectifier
from tablet_tap.vision.rectifier import draw_overlay

GOOD_WIDTH_PX, GOOD_EDGE_PX = 900.0, 4.0
MOUNT_MM, MOUNT_DEG = 15.0, 6.0  # further than this from the drawing: the wrong camera, mount or joint convention


class WristContext(AppContext):
    """The same setup, seen with the wrist camera: the robot looks from the look pose."""

    def __init__(self, options, arm_use: ArmUse = ArmUse.MOVES) -> None:
        super().__init__(options, arm_use)
        self.wrist_store = WristStore(options.setup_path.with_name("wrist.json"))
        self.wrist = self.wrist_store.load()
        self.setup.camera = options.camera or self.wrist.camera
        self._device: RobotTabletDevice | None = None

    # ---- what is needed first ----------------------------------------------------------------------------------------
    def screen(self) -> ScreenMap:
        self.setup.require("screen_map", "park_q", "seed_q", hint="先用顶部相机或示教确定屏幕的位置（网页控制台，或 place / teach）。")
        return self.setup.screen_map

    def camera(self):
        camera = super().camera()
        model = camera_model(camera)
        if model is not None:
            self.wrist.model = model
        return camera

    def rectifier(self) -> ScreenRectifier:
        if not self.wrist_store.view_path.exists():
            raise SetupError(f"还没有 {self.wrist_store.view_path}；先运行 wrist view。")
        moved = self.wrist.screen_moved(self.screen())
        if moved:
            raise SetupError(f"{moved}：重新运行 wrist view。")
        return ScreenRectifier.load(self.wrist_store.view_path)

    def look_robot(self, look_q: np.ndarray | None = None) -> LookRobot:
        screen, setup = self.screen(), self.setup
        if look_q is None and self.wrist.look_q is None:
            raise SetupError("还没有看屏位；先运行 wrist view。")
        samples = setup.records.get("tap_correction") or []
        size = (setup.records.get("live_screen") or {}).get("size_m") or screen.size_m
        correction = TapCorrection(tuple(size), [tuple(x) for x in samples])
        return LookRobot(
            self.arm,
            self.pen,
            screen,
            np.asarray(self.wrist.look_q if look_q is None else look_q),
            np.asarray(setup.park_q),
            np.asarray(setup.seed_q),
            self.wrist.planner(self.pen),
            setup.motion_limits(),
            should_stop=self.stop.is_set,
            realtime=self.realtime,
            aim=correction.aim,
        )

    def robot(self, limits=None) -> LookRobot:
        return self.look_robot()

    def device(self) -> RobotTabletDevice:
        if self._device is None:
            rectifier = self.rectifier()  # asked first: it says when the tablet was moved
            self._device = RobotTabletDevice(
                self.robot(), self.camera(), rectifier, nav=self.setup.nav, keyboard=self.setup.on_screen_keyboard()
            )
        return self._device

    # ---- moving ------------------------------------------------------------------------------------------------------
    def move(self, q_goal: np.ndarray, name: str) -> None:
        """A slow joint move (the calibration's: the legs were checked when they were planned)."""
        limits = self.setup.motion_limits()
        run_segments(
            self.arm,
            [joint_move(name, self.arm.read(), q_goal, _slower(limits), settle=True)],
            hz=limits.hz,
            should_stop=self.stop.is_set,
            realtime=self.realtime,
        )

    def must_be_parked(self) -> None:
        gap = float(np.max(np.abs(self.arm.read() - np.asarray(self.setup.park_q))))
        if gap > START_TOL_RAD:
            raise PathError("机械臂不在停靠位姿；先用 park 命令（低速、确认路径无障碍）回到停靠位姿。")

    def save_wrist(self) -> None:
        self.wrist_store.save(self.wrist)
        print(f"已保存 {self.wrist_store.path}")


def _slower(limits):
    import dataclasses  # noqa: PLC0415

    return dataclasses.replace(limits, joint_move_speed=min(limits.joint_move_speed, PARK_SPEED))


@register
class Wrist(Command):
    name = "wrist"
    help = "腕部相机看屏幕：plan（只算）/ calib（手眼标定）/ view（去看屏位标定画面）/ shot / run"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        actions = parser.add_subparsers(dest="action", required=True, metavar="<动作>")
        plan = actions.add_parser("plan", help="只计算：看屏位和标定位姿，不连硬件")
        plan.add_argument("--screen_at", type=float, nargs=3, metavar=("前", "左", "高"), help="换成假想的平板中心位置（厘米，平放、下边朝基座）")
        calib = actions.add_parser("calib", help="手眼标定：平板显示标定页，手臂自动走一组位姿拍照（不碰屏幕）")
        calib.add_argument("--views", type=int, default=14)
        view = actions.add_parser("view", help="去看屏位，用标定页的六个码标定画面；平板挪动后重做")
        view.add_argument("--out_width", type=int, default=1792)
        view.add_argument("--keep_pose", action="store_true", help="沿用上次的看屏位，不重新找")
        shot = actions.add_parser("shot", help="在看屏位拍一张矫正的屏幕图")
        shot.add_argument("--out", default=None)
        run = actions.add_parser("run", help="整条链路，用腕部相机看屏幕（选项同 run）")
        Run().configure(run)
        for sub in (plan, calib, view):
            sub.add_argument("--mount", choices=sorted(MOUNTS), help="相机装在哪种支架上（记在 wrist.json）")

    def arm_use(self, args: argparse.Namespace) -> ArmUse:
        return ArmUse.NONE if args.action == "plan" else ArmUse.MOVES

    def run(self, ctx: AppContext, args: argparse.Namespace) -> None:
        wrist = WristContext(ctx.options, self.arm_use(args))
        if getattr(args, "mount", None) and args.mount != wrist.wrist.mount:
            wrist.wrist.mount = args.mount  # the calibration belonged to the other mount
            wrist.wrist.in_flange = wrist.wrist.look_q = wrist.wrist.look_screen = None
        getattr(self, f"_{args.action}")(wrist, args)

    # ---- plan --------------------------------------------------------------------------------------------------------
    def _plan(self, ctx: WristContext, args: argparse.Namespace) -> None:
        from tablet_tap.calibration.hand_eye import seed_joints  # noqa: PLC0415

        if args.screen_at:
            size = tuple(ctx.setup.screen_size_m or (0.2369, 0.1481))
            x, y, z = (v * 1e-2 for v in args.screen_at)
            right, up = np.array([0.0, -size[0], 0.0]), np.array([size[1], 0.0, 0.0])
            screen = ScreenMap(np.array([x, y, z]) - 0.5 * right + 0.5 * up, right, -up, np.array([1.0, 0.0, 0.0]), tilt_deg=8.0)
            seed = seed_joints(ctx.pen, screen, ctx.setup.motion_limits())
        else:
            screen, seed = ctx.screen(), np.asarray(ctx.setup.seed_q)
        park = np.asarray(ctx.setup.park_q)
        look = ctx.wrist.planner(ctx.pen).best(screen, seed, park)
        print(f"看屏位：{look.describe()}\n  关节角 {np.round(look.q, 3).tolist()}")
        boxes = list(default_layout(ctx.setup.aspect or 1.6).markers.values())
        views = ctx.wrist.planner(ctx.pen, CALIBRATION_RULES).calibration_views(screen, look, park, boxes)
        print(f"标定位姿 {len(views)} 个，每个离停靠位最大 {np.degrees(max(np.max(np.abs(v.q - park)) for v in views)):.0f}°")

    # ---- calibration -------------------------------------------------------------------------------------------------
    def _calib(self, ctx: WristContext, args: argparse.Namespace) -> None:
        screen, park = ctx.screen(), np.asarray(ctx.setup.park_q)
        page = ctx.page()
        around = ctx.wrist.planner(ctx.pen).best(screen, np.asarray(ctx.setup.seed_q), park)
        boxes = list(page.layout().markers.values())
        looks = ctx.wrist.planner(ctx.pen, CALIBRATION_RULES).calibration_views(screen, around, park, boxes, args.views)
        print(f"将低速走 {len(looks)} 个位姿拍标定页（笔不碰屏幕，最低离屏 {min(v.tip_height_m for v in looks) * 1e3:.0f} mm），最后回停靠位。")
        camera = ctx.camera()
        try:
            page.set_mode("markers")
            page.clear()
            ctx.must_be_parked()
            ctx.confirm("腕部相机标定")
            out = config.runs_dir() / f"wrist_calib_{datetime.datetime.now():%Y%m%d_%H%M%S}"
            out.mkdir(parents=True, exist_ok=True)
            with ctx.interruptible():
                views = collect_views(
                    looks, park, ctx.move, ctx.arm.read, camera, page.layout(),
                    settle_s=0.6 if ctx.realtime else 0.0, keep=lambda k, picture: cv2.imwrite(str(out / f"view_{k:02d}.png"), picture),
                )  # fmt: skip
        finally:
            camera.close()
        (out / "views.json").write_text(json.dumps([{"q": v.q.tolist(), "markers": {str(k): m.tolist() for k, m in v.markers.items()}} for v in views]))
        drawn = camera_in_flange(*MOUNTS[ctx.wrist.mount])
        model = ctx.wrist.wrist_camera().model
        found = solve_eye_in_hand(views, page.layout(), tuple(ctx.setup.screen_size_m or screen.size_m), model, ctx.pen, drawn)
        (out / "result.json").write_text(json.dumps(found.to_json(), indent=1))
        off_mm, off_deg = mount_error(found.camera_in_flange, drawn)
        seen = found.screen_map()
        gaps = [np.linalg.norm(seen.point(u, v) - screen.point(u, v)) * 1e3 for u, v in ((0, 0), (1, 0), (1, 1), (0, 1))]
        print(
            f"拟合残差 {found.rms_px:.2f} px，没参加拟合的位姿 {found.held_out_px:.2f} px，去掉 {len(found.left_out)} 个位姿；"
            f"相机位置的不确定度约 {found.position_sigma_mm:.1f} mm / {found.angle_sigma_deg:.2f}°\n"
            f"和图纸差 {off_mm:.1f} mm、{off_deg:.1f}°；屏幕尺寸 {found.screen_size_m[0] * 1e3:.1f} × {found.screen_size_m[1] * 1e3:.1f} mm；"
            f"和现有屏幕平面的四角差 {' / '.join(f'{g:.1f}' for g in gaps)} mm"
        )
        if off_mm > MOUNT_MM or off_deg > MOUNT_DEG:
            raise CalibrationError(
                f"标定结果离支架图纸太远（上限 {MOUNT_MM:.0f} mm、{MOUNT_DEG:.0f}°），没有保存：检查相机是不是左腕那台、支架选对没有（--mount）。"
            )
        ctx.wrist.in_flange = found.camera_in_flange.tolist()
        ctx.wrist.look_q = ctx.wrist.look_screen = None  # the look pose was planned with the old camera
        ctx.wrist.records["eye_in_hand"] = {k: v for k, v in found.to_json().items() if k != "camera_in_flange"} | {"folder": str(out)}
        ctx.save_wrist()
        print("接着运行 wrist view。")

    # ---- the look pose and its picture ---------------------------------------------------------------------------------
    def _view(self, ctx: WristContext, args: argparse.Namespace) -> None:
        screen, setup = ctx.screen(), ctx.setup
        planner = ctx.wrist.planner(ctx.pen)
        if args.keep_pose and ctx.wrist.look_q is not None:
            look_q = np.asarray(ctx.wrist.look_q)
        else:
            look = planner.best(screen, np.asarray(setup.seed_q), np.asarray(setup.park_q))
            print(f"看屏位：{look.describe()}")
            look_q = look.q
        robot = ctx.look_robot(look_q)
        if not ctx.wrist.calibrated:
            print("⚠ 腕部相机还没做手眼标定，看屏位按支架图纸算，画面可能偏几度。")
        camera, page = ctx.camera(), ctx.page()
        try:
            page.set_mode("markers")
            page.clear()
            ctx.confirm("去看屏位")
            with ctx.interruptible():
                robot.to_look()
                q, picture = still_picture(ctx.arm.read, camera, settle_s=0.6 if ctx.realtime else 0.0)
                robot.to_rest()
        finally:
            camera.close()
        path = ctx.wrist_store.view_path
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            rectifier = ScreenRectifier.from_marker_image(picture, page.layout(), out_width=int(args.out_width))
        except CalibrationError as exc:
            cv2.imwrite(str(path.with_suffix(".failed.jpg")), picture)
            raise CalibrationError(f"看屏位的画面标定失败：{exc}\n画面已存到 {path.with_suffix('.failed.jpg')}") from None
        rectifier.save(path)
        cv2.imwrite(str(path.with_suffix(".overlay.jpg")), draw_overlay(picture, rectifier))
        cv2.imwrite(str(path.with_suffix(".rectified.png")), rectifier.rectify(picture))
        ctx.wrist.look_q, ctx.wrist.look_screen = look_q.tolist(), screen.to_json()
        quality = look_quality(picture, rectifier)
        expected = ctx.wrist.predicted_homography(ctx.pen, q, screen)
        worst = 0.0
        for u, v in ((0.05, 0.05), (0.95, 0.05), (0.95, 0.95), (0.05, 0.95), (0.5, 0.5)):
            x, y, w = expected @ [*rectifier.norm_to_cam(np.array([u, v])), 1.0]
            worst = max(worst, screen.error_mm((u, v), (x / w, y / w)))
        ctx.wrist.records["view"] = quality | {"hand_eye_gap_mm": worst, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        ctx.save_wrist()
        edge = quality["edge_px"]
        print(
            f"已保存 {path}：屏幕在画面里 {quality['width_px']:.0f} × {quality['height_px']:.0f} px，码的重投影误差 {quality['reprojection_px']:.2f} px，"
            f"用到 {len(rectifier.marker_ids)} 个码，边缘过渡 {'量不出' if edge is None else f'{edge:.1f} px'}\n"
            f"手眼标定预测的画面和实际最大差 {worst:.1f} mm（只作检查，看屏和点按都不靠它）"
        )
        if quality["width_px"] < GOOD_WIDTH_PX:
            print(f"⚠ 屏幕不到 {GOOD_WIDTH_PX:.0f} px 宽，比预期小。")
        if edge is not None and edge > GOOD_EDGE_PX:
            print(f"⚠ 边缘过渡超过 {GOOD_EDGE_PX:.0f} px：离焦或抖动，离近了反而更糊时应该把看屏位放远。")

    # ---- using it ------------------------------------------------------------------------------------------------------
    def _shot(self, ctx: WristContext, args: argparse.Namespace) -> None:
        device = ctx.device()
        ctx.confirm("去看屏位拍照")
        with ctx.interruptible():
            device.robot.to_look()
            image = device.screenshot()
            device.robot.to_rest()
        out = pathlib.Path(args.out) if args.out else config.runs_dir() / "wrist_shot.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        image.save(out)
        sharp = edge_width_px(device.last_frame, {}) if device.last_frame is not None else None
        print(f"已保存 {out}（{image.width} × {image.height}）" + ("" if sharp is None else f"，边缘过渡 {sharp:.1f} px"))

    def _run(self, ctx: WristContext, args: argparse.Namespace) -> None:
        device = ctx.device()
        ctx.confirm("按指令点屏幕（腕部相机）")
        with ctx.interruptible():
            device.robot.to_look()
        try:
            Run().run(ctx, args)
        finally:
            if device.robot.where is ArmPlace.PARK and not ctx.stop.is_set():
                device.robot.to_rest()
