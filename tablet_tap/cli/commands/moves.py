"""Single moves: park, hover, and the gestures (tap, swipe, home, back)."""

from __future__ import annotations

import abc
import argparse
import dataclasses

import numpy as np

from tablet_tap.agent.robot_device import RobotTabletDevice
from tablet_tap.app import AppContext
from tablet_tap.cli.commands.base import Command
from tablet_tap.cli.commands.base import register
from tablet_tap.core.motion import joint_move
from tablet_tap.robot.executor import run_segments
from tablet_tap.robot.tablet_robot import ArmPlace
from tablet_tap.robot.tablet_robot import TabletRobot

PARK_SPEED = 0.25  # rad/s, average of the fastest joint: the way back from an unknown pose is not collision-checked


@register
class Park(Command):
    name = "park"
    help = "低速回停靠位姿（任意位姿出发，无碰撞检查）"

    def run(self, ctx: AppContext, args: argparse.Namespace) -> None:
        ctx.setup.require("park_q")
        limits = dataclasses.replace(ctx.setup.motion_limits(), joint_move_speed=PARK_SPEED)
        q, goal = ctx.arm.read(), np.asarray(ctx.setup.park_q)
        span = float(np.max(np.abs(goal - q)))
        if span < 1e-3:
            print("已经在停靠位姿。")
            return
        print(f"将从当前位姿沿关节空间直线低速回到停靠位姿（最大关节位移 {span:.2f} rad）。这条路径没有做碰撞检查。")
        ctx.confirm("回停靠位姿")
        with ctx.interruptible():
            run_segments(
                ctx.arm,
                [joint_move("回停靠位", q, goal, limits, settle=True)],
                hz=limits.hz,
                should_stop=ctx.stop.is_set,
                realtime=ctx.realtime,
            )
        print("已到停靠位姿。")


class Gesture(Command, abc.ABC):
    """A move over the calibrated screen. Subclasses say what to do; confirming, Ctrl-C and parking are shared."""

    parks_after: bool = True  # whether the command offers --park

    def configure(self, parser: argparse.ArgumentParser) -> None:
        if self.parks_after:
            parser.add_argument("--park", action="store_true", help="做完回停靠位姿")

    @abc.abstractmethod
    def gesture(self, robot: TabletRobot, device: RobotTabletDevice, args: argparse.Namespace) -> None: ...

    def run(self, ctx: AppContext, args: argparse.Namespace) -> None:
        robot = ctx.robot()
        device = RobotTabletDevice(robot, None, None, nav=ctx.setup.nav)  # touches only: no camera needed
        ctx.confirm(self.name)
        with ctx.interruptible():
            self.gesture(robot, device, args)
            if getattr(args, "park", False):
                robot.to_park()
        print(f"完成，机械臂在：{'停靠位姿' if robot.where is ArmPlace.PARK else '屏幕上方'}")


@register
class Hover(Gesture):
    name = "hover"
    help = "从停靠位姿到屏幕中心上方"
    parks_after = False

    def gesture(self, robot: TabletRobot, device: RobotTabletDevice, args: argparse.Namespace) -> None:
        robot.to_hover()


@register
class Tap(Gesture):
    name = "tap"
    help = "点一下 (u, v)"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("u", type=float)
        parser.add_argument("v", type=float)
        parser.add_argument("--hold", type=float, default=None, help="按住的秒数（长按）")
        super().configure(parser)

    def gesture(self, robot: TabletRobot, device: RobotTabletDevice, args: argparse.Namespace) -> None:
        robot.tap(args.u, args.v, hold_s=args.hold)


@register
class Swipe(Gesture):
    name = "swipe"
    help = "从 (u, v) 滑到 (u2, v2)"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        for name in ("u", "v", "u2", "v2"):
            parser.add_argument(name, type=float)
        parser.add_argument("--duration", type=float, default=0.5, help="滑动用的秒数")
        super().configure(parser)

    def gesture(self, robot: TabletRobot, device: RobotTabletDevice, args: argparse.Namespace) -> None:
        robot.swipe(args.u, args.v, args.u2, args.v2, args.duration)


@register
class Home(Gesture):
    name = "home"
    help = "回桌面"

    def gesture(self, robot: TabletRobot, device: RobotTabletDevice, args: argparse.Namespace) -> None:
        device.home()


@register
class Back(Gesture):
    name = "back"
    help = "返回"

    def gesture(self, robot: TabletRobot, device: RobotTabletDevice, args: argparse.Namespace) -> None:
        device.back()
