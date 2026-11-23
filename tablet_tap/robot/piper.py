"""The real arm: one PiPER on its CAN channel, through RoboSynChallenge's driver.

``PiperPenArm`` adapts ``real_robot.hardware.PiperArm`` (third_party/robosyn: same CAN driver, firmware profiles, fault
checks and ``SafetyFilter`` as the policy runner) to ``ArmIO``, with the gripper squeezing the pen holder. The helpers
around it only listen: which other programs drive the arms, and whether a CAN channel delivers the arm's frames.
"""

from __future__ import annotations

import logging
import subprocess
import time

import numpy as np

from tablet_tap.errors import ArmBusyError
from tablet_tap.errors import ArmConnectionError
from tablet_tap.errors import PathError

logger = logging.getLogger("tablet_tap.robot")


def other_arm_programs() -> list[str]:
    """Command lines of programs that drive the arms (the policy runner, the console, teleoperation)."""
    try:
        out = subprocess.run(
            ["pgrep", "-af", r"run_real\.py|reset_arms\.py|leader_|piper_teleop"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
    except OSError:
        return []
    return [line for line in out.splitlines() if "pgrep" not in line]


STATUS_FRAME_HZ = 200.0
# the feedback frames a healthy PiPER sends 200 times a second each: controller status, joints 1-2 / 3-4 / 5-6, gripper
FEEDBACK_FRAMES = {0x2A1: "状态", 0x2A5: "关节1-2", 0x2A6: "关节3-4", 0x2A7: "关节5-6", 0x2A8: "夹爪"}


def bus_rates(channel: str, seconds: float = 1.0, interface: str = "socketcan") -> dict[int, float] | None:
    """Frames per second of each kind of feedback on a CAN channel, listening only (nothing is sent). None: the channel
    cannot be opened. The status frames alone can look fine while others are being lost."""
    import can  # noqa: PLC0415 - comes with pyAgxArm

    filters = [{"can_id": frame_id, "can_mask": 0x7FF} for frame_id in FEEDBACK_FRAMES]
    try:
        bus = can.interface.Bus(channel=channel, interface=interface, can_filters=filters)
    except (OSError, can.CanError):
        return None
    counts = dict.fromkeys(FEEDBACK_FRAMES, 0)
    end = time.monotonic() + seconds
    try:
        while (left := end - time.monotonic()) > 0:
            message = bus.recv(timeout=left)
            if message is not None and message.arbitration_id in counts:
                counts[message.arbitration_id] += 1
    finally:
        bus.shutdown()
    return {frame_id: count / seconds for frame_id, count in counts.items()}


def describe_bus(channel: str) -> str:
    rates = bus_rates(channel)
    if rates is None:
        return f"CAN 通道 {channel} 打不开（接口不存在或没有拉起；bash scripts/setup_piper_can.sh status 可以查看）"
    if max(rates.values()) < 1:
        return f"CAN 通道 {channel} 上听不到机械臂的反馈：手臂没上电，或者 CAN 线 / 适配器没接好"
    slow = {FEEDBACK_FRAMES[i]: rate for i, rate in rates.items() if rate < 0.9 * STATUS_FRAME_HZ}
    if slow:
        detail = "、".join(f"{name} {rate:.0f}" for name, rate in slow.items())
        return (
            f"CAN 通道 {channel} 在丢帧：每秒 {detail}（正常各约 {STATUS_FRAME_HZ:.0f}）。反馈不可靠时不要让手臂运动。"
            "检查这只手臂的 USB-CAN 适配器、USB 集线器和 CAN 线，必要时拔插适配器、换一个集线器口，或给手臂断电重启"
        )
    return f"CAN 通道 {channel} 正常：各类反馈每秒约 {min(rates.values()):.0f} 帧"


def bus_rate(channel: str, seconds: float = 1.0, interface: str = "socketcan") -> float | None:
    """The slowest kind of feedback on the channel, in frames per second (None: the channel cannot be opened)."""
    rates = bus_rates(channel, seconds, interface)
    return None if rates is None else min(rates.values())


def hold_grip(arm, width_m: float, force_n: float, times: int = 5) -> None:
    """Send the gripper its target several times (a single frame may be lost)."""
    for _ in range(times):
        arm._gripper.move_gripper_m(value=float(width_m), force=float(force_n))
        time.sleep(0.02)


OPEN_WIDTH_M = 0.035  # wider than the pen holder (21 mm): the gripper holds nothing


def hold_width(arm, width_m: float) -> float:
    """The width to keep when taking control: ``width_m``, or the present one of a gripper that stands open."""
    try:
        present = float(arm.read()[1])
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return width_m
    return present if present > OPEN_WIDTH_M else width_m


class PiperPenArm:
    """One PiPER arm on its CAN channel. With ``send_commands`` False nothing is ever sent to the motors."""

    def __init__(
        self,
        channel: str,
        *,
        send_commands: bool,
        joint_signs: tuple[float, ...] = (1.0,) * 6,
        joint_offsets: tuple[float, ...] = (0.0,) * 6,
        firmware: str = "auto",
        interface: str = "socketcan",
        speed_percent: int = 30,
        grip_width_m: float = 0.0,
        grip_force_n: float = 5.0,
        max_joint_step_rad: float = 0.04,
        allow_other_programs: bool = False,
    ) -> None:
        running = other_arm_programs()
        if running and send_commands and not allow_other_programs:
            raise ArmBusyError(
                "另一个程序正在控制机械臂，两个程序同时发指令会互相打架。请先停掉它：\n  " + "\n  ".join(running)
            )
        if running:
            logger.warning("另一个控制机械臂的程序正在运行（本程序只读）：%s", running[0][:120])
        from real_robot import conversions as rc  # noqa: PLC0415 - needs the robot libraries
        from real_robot import hardware  # noqa: PLC0415

        self._rc = rc
        self._send = send_commands
        self._grip_width_m = float(grip_width_m)
        self._safety = rc.SafetyFilter(max_joint_step_rad=max_joint_step_rad)
        try:
            self._arm = self._connect(
                hardware,
                rc,
                channel,
                joint_signs,
                joint_offsets,
                firmware,
                interface,
                send_commands,
                speed_percent,
                grip_force_n,
                self._grip_width_m,
            )
        except (UnicodeDecodeError, TimeoutError, RuntimeError, OSError) as exc:
            raise ArmConnectionError(
                f"连不上 {channel} 上的机械臂（{type(exc).__name__}: {exc}）。\n{describe_bus(channel)}"
            ) from exc
        self._last: np.ndarray | None = None
        self._grip_width_m = float(getattr(self._arm, "held_width_m", self._grip_width_m))
        self.gripper_width_m = float("nan")

    @staticmethod
    def _connect(
        hardware,
        rc,
        channel,
        joint_signs,
        joint_offsets,
        firmware,
        interface,
        send_commands,
        speed_percent,
        grip_force_n,
        grip_width_m,
    ):
        config = hardware.ArmConfig(
            channel=channel,
            calibration=rc.ArmCalibration(signs=tuple(joint_signs), offsets_rad=tuple(joint_offsets)),
            firmware=firmware,
            interface=interface,
        )

        def connect(commands: bool):
            return hardware.PiperArm(
                config,
                send_commands=commands,
                speed_percent=speed_percent,
                gripper_force_n=grip_force_n,
                gripper_max_mm=100.0,
            )

        if not send_commands:
            return connect(False)
        # Taking control makes the gripper run to whatever target it was last given, an opened gripper included: on
        # 2026-11-23 it opened to 100 mm (run_real's last command) and dropped the pen holder. So the grip is set before
        # the mode changes and again right after it.
        # A gripper that stands wide open holds nothing: it stays as it is until it is told to close (fingers may be in it).
        probe = connect(False)
        grip_width_m = hold_width(probe, grip_width_m)
        hold_grip(probe, grip_width_m, grip_force_n)
        disconnect = getattr(probe._robot, "disconnect", None)
        if disconnect is not None:
            disconnect()
        arm = connect(True)
        hold_grip(arm, grip_width_m, grip_force_n)
        arm.held_width_m = grip_width_m
        return arm

    def read(self) -> np.ndarray:
        if self._send:
            self._arm.check_status()
        q, self.gripper_width_m = self._arm.read()
        return np.asarray(q, np.float64)

    def command(self, q: np.ndarray) -> None:
        q = np.asarray(q, np.float64)
        reference = self._last if self._last is not None else self.read()
        limited, largest = self._safety.limit_step(q, reference)
        if largest > self._safety.max_joint_step_rad + 1e-9:
            # never clip silently: a clipped target would leave the planned (collision-checked) path
            raise PathError(f"目标一步跳了 {largest:.3f} rad，超过上限 {self._safety.max_joint_step_rad} rad；未发送。")
        self._arm.command(limited, self._grip_width_m, self._safety)
        self._last = limited

    def set_grip_width(self, width_m: float) -> None:
        """Gripper target sent with every joint target from now on: 0 squeezes the pen holder, 0.06 lets it go."""
        self._grip_width_m = float(width_m)

    def firmware_joint_limits(self) -> dict[int, tuple[float, float]]:
        """What the arm's own controller allows: joint (1-6) -> (lowest, highest) of its readings, in rad."""
        limits = {}
        for joint in range(1, 7):
            answer = self._arm._robot.get_joint_angle_vel_limits(joint, timeout=1.0)
            if answer is None:
                raise ArmConnectionError(f"机械臂没有回答 {joint} 号关节的限位查询。")
            limits[joint] = (float(answer.msg.min_angle_limit), float(answer.msg.max_angle_limit))
        return limits

    def set_firmware_joint_limit(self, joint: int, *, lowest: float | None = None, highest: float | None = None) -> bool:
        """Write one joint's limit (rad; None leaves that side as it is) into the arm's controller, where it stays for
        every program. A configuration frame, not a motion: it is sent on a read-only connection too."""
        robot = self._arm._robot
        return bool(robot.set_joint_angle_vel_limits(joint, min_angle_limit=lowest, max_angle_limit=highest, timeout=2.0))

    def close(self) -> None:
        pass  # the arm stays enabled and holds its pose: disabling a PiPER lets its joints fall
