"""Choosing the arm: the real PiPER, a read-only connection to it, or a software arm."""

from __future__ import annotations

import enum
import logging

import numpy as np

from tablet_tap.config import ArmSetup
from tablet_tap.errors import SetupError
from tablet_tap.robot.arm import ArmIO
from tablet_tap.robot.arm import FakeArm

logger = logging.getLogger("tablet_tap.robot")

FAKE_START_Q = (0.0, 0.6, -0.5, 0.0, 0.3, 0.0)


class ArmMode(enum.Enum):
    COMMAND = "command"  # the motors follow this program
    READ_ONLY = "read_only"  # connected to the arm, nothing is ever sent to it
    DRY_RUN = "dry_run"  # the real joints are read once; a software arm then moves in their place
    FAKE = "fake"  # no hardware at all


def open_arm(
    arm: ArmSetup, mode: ArmMode, *, fake_start: np.ndarray | None = None, allow_other_programs: bool = False
) -> ArmIO:
    if mode is ArmMode.FAKE:
        return FakeArm(np.asarray(FAKE_START_Q if fake_start is None else fake_start, np.float64))
    if not arm.channel:
        raise SetupError("还没有指定拿笔的机械臂：加上 --channel can_l_follower（左臂）或 can_r_follower（右臂）。")
    from tablet_tap.robot.piper import PiperPenArm  # noqa: PLC0415 - needs the robot libraries

    if arm.channel not in arm.firmware_limits_deg:
        logger.warning("还没有读过 %s 的固件限位，先按 URDF 里的走：运行一次 limits 读出来记下。", arm.channel)
    real = PiperPenArm(
        arm.channel,
        send_commands=mode is ArmMode.COMMAND,
        joint_signs=tuple(arm.joint_signs),
        joint_offsets=tuple(arm.joint_offsets),
        firmware=arm.firmware,
        speed_percent=int(arm.speed_percent),
        grip_force_n=float(arm.grip_force_n),
        joint_limits_rad=arm.reading_limits_rad(),
        allow_other_programs=allow_other_programs,
    )
    if mode is ArmMode.DRY_RUN:
        print("【dry_run】只读取了真实关节角，之后的运动都在软件里模拟，不发任何指令。")
        return FakeArm(real.read())
    return real
