"""Bring the follower PiPER arms back after an emergency stop or a collision, then fold them to the zero pose.

    PYTHONPATH=policy/fmc3_no1 policy/fmc3_no1/.venv/bin/python scripts/sim2real/reset_arms.py [--arm right]
    PYTHONPATH=policy/fmc3_no1 policy/fmc3_no1/.venv/bin/python scripts/sim2real/reset_arms.py --status_only

Steps, per selected arm:
  1. connect to the arm's CAN bus (an arm whose power is off or whose CAN cable is loose sends nothing: the script
     reports it at once and goes on with the other arm; check its power and cable, or power-cycle it, and rerun);
  2. print the arm status and joint driver faults; if anything is wrong, clear every joint's error code first; for an
     emergency stop, a collision or a teaching mode also reset the controller -- once, after you hold the arm and
     press Enter, because the joints may go limp until the motors are enabled again; then make the measured pose the
     joint target and enable the motors, until the arm is normal (15 s at most);
  3. after you press Enter, fold the arm to the zero pose (grippers closed) with a slow, speed-limited joint
     path. Use --no_move to stop after step 2: the arm is then enabled and holds its pose.

The arms stay enabled at the end: a disabled PiPER drops its raised joints. This robot has no e-stop button (a
collision stops an arm): keep a hand near the arms. If the
arm is pressing on the table or the object, free it by hand or move it up with a small pose first; the path to
the zero pose is a straight line in joint space and does not avoid anything.
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from real_robot import conversions as rc
from real_robot import hardware as hw

# ctrl_mode values in which the arm does not follow CAN motion commands; leaving them needs the controller reset
# (piper_sdk demo piper_ctrl_reset.py: reset when switching from teach / MIT mode back to position-velocity control).
CTRL_MODE = {
    0: "待机",
    1: "CAN 指令控制",
    2: "示教模式",
    3: "以太网控制",
    4: "Wi-Fi 控制",
    5: "遥控器控制",
    6: "联动示教输入",
    7: "离线轨迹",
}
NEEDS_RESET_CTRL_MODES = {2, 6, 7}
# arm_status values that only the controller reset clears: emergency stop and collision.
NEEDS_RESET_ARM_STATUS = {1, 7}

ARM_STATUS = {
    0: "正常",
    1: "急停",
    2: "无解",
    3: "奇异点",
    4: "目标角度超限",
    5: "关节通信异常",
    6: "关节抱闸未打开",
    7: "发生碰撞",
    8: "示教拖动超速",
    9: "关节状态异常",
    10: "其他异常",
    11: "示教记录",
    12: "示教执行",
    13: "示教暂停",
    14: "主控 NTC 过温",
    15: "释放电阻 NTC 过温",
}


def unopenable(channel: str, e: Exception) -> str | None:
    """What to do about an arm that cannot be opened (this robot has no e-stop button); None for an error that is not
    about the arm, which the caller raises again."""
    if isinstance(e, ValueError):  # pyAgxArm: Device 'can_r_follower' does not exist
        if "does not exist" not in str(e):
            return None
        return (
            f"CAN 口 {channel} 不存在：这条臂的 USB-CAN 适配器没插好，或刚插上还没起来。插好后等几秒"
            "（bash scripts/sim2real/sim2real_4090.sh status 能看到它）再启动。"
        )
    return (
        f"没有回应（{e}）：臂没电，或适配器到臂底座的 CAN 线没插好。查电源和 CAN 线，或断电 10 秒再上电后重新启动"
        "（bash scripts/sim2real/sim2real_4090.sh status 看每条臂有没有报文）。"
    )


def connect(channel: str, firmware: str) -> hw.PiperArm | None:
    """Connect read-only. None if the arm is silent (powered off, its CAN cable or adapter out)."""
    try:
        return hw.PiperArm(hw.ArmConfig(channel=channel, firmware=firmware), send_commands=False)
    except (RuntimeError, ValueError) as e:
        why = unopenable(channel, e)
        if why is None:
            raise
        print(f"{channel}: {why}", flush=True)
        return None


def joint_faults(arm: hw.PiperArm) -> list[str]:
    """Joints whose driver reports an error or a collision-protection trip."""
    faults = []
    for j in range(1, 7):
        ds = arm._robot.get_driver_states(j)
        foc = getattr(getattr(ds, "msg", None), "foc_status", None)
        if foc is None:
            continue
        if foc.driver_error_status:
            faults.append(f"J{j}驱动错误")
        if foc.collision_status:
            faults.append(f"J{j}碰撞保护")
    return faults


def status_of(arm: hw.PiperArm) -> tuple[int, int, bool, list[str]]:
    robot = arm._robot
    msg = hw._wait_for(robot.get_arm_status, f"arm status on {arm._config.channel}", 3.0).msg
    time.sleep(0.3)  # the joint-enable list reads all False until its own feedback frame has arrived
    enabled = bool(all(robot.get_joints_enable_status_list()))
    return int(msg.arm_status), int(msg.ctrl_mode), enabled, joint_faults(arm)


def describe(name: str, arm: hw.PiperArm) -> bool:
    status, ctrl_mode, enabled, faults = status_of(arm)
    print(
        f"{name}: arm_status={status}（{ARM_STATUS.get(status, '未知')}）ctrl_mode={ctrl_mode}"
        f"（{CTRL_MODE.get(ctrl_mode, '未知')}）电机使能={enabled} 关节故障={'、'.join(faults) or '无'}",
        flush=True,
    )
    return status == 0 and enabled and not faults and ctrl_mode not in NEEDS_RESET_CTRL_MODES


def end_drag_teach(arm: hw.PiperArm) -> None:
    """End a drag teach (the arm's teach button: teach_status 1), which the controller reset alone does not leave: on
    10-18 the left arm stayed in the teach mode after the reset until its button was pressed again."""
    status = arm._robot.get_arm_status()
    if status is None or int(getattr(status.msg, "teach_status", 0)) != 1:
        return
    try:
        from pyAgxArm.protocols.can_protocol.msgs.piper.default.transmit.arm_motion_ctrl import (  # noqa: PLC0415
            ArmMsgMotionCtrl,
        )
    except ImportError:
        return
    arm._robot._send_msg(ArmMsgMotionCtrl(grag_teach_ctrl=0x02))  # 0x150 byte 2: end the teaching record
    time.sleep(0.3)


def hold_current_pose(arm: hw.PiperArm) -> None:
    """Make the measured pose the joint target, so that enabling does not move the arm toward an older target."""
    joints = arm._robot.get_joint_angles()
    if joints is not None:
        arm._robot.move_j([float(v) for v in joints.msg])


def recover(name: str, arm: hw.PiperArm, timeout_s: float = 15.0, confirm=input) -> None:
    """Clear the joint errors, reset the controller if it needs it (once, after the operator confirms), enable.

    The controller reset is only sent for an emergency stop, a collision or a teaching mode: the joints may go limp
    until the motors are enabled again, so the operator first holds the arm and presses Enter (``confirm``).
    """
    robot = arm._robot
    deadline = time.monotonic() + timeout_s
    reset_sent = False
    while True:
        status, ctrl_mode, enabled, faults = status_of(arm)
        if status == 0 and enabled and not faults and ctrl_mode not in NEEDS_RESET_CTRL_MODES:
            return
        robot.clear_joint_error(255)  # first clear the error code of every joint driver (0x475, 0xAE)
        time.sleep(0.3)
        if (status in NEEDS_RESET_ARM_STATUS or ctrl_mode in NEEDS_RESET_CTRL_MODES) and not reset_sent:
            # Leave the emergency stop / collision state or the teach mode of the motion controller (0x150, 0x02).
            confirm(f"{name}: 要复位控制器，复位时关节可能短暂没力——扶住这条臂，回车继续（Ctrl-C 取消）：")
            end_drag_teach(arm)
            robot.reset()
            reset_sent = True
            time.sleep(0.5)
        hold_current_pose(arm)
        robot.enable()
        time.sleep(0.5)
        if time.monotonic() > deadline:
            status, ctrl_mode, enabled, faults = status_of(arm)
            raise SystemExit(
                f"{name}: 恢复失败（arm_status={status} {ARM_STATUS.get(status, '')}，ctrl_mode={ctrl_mode}"
                f" {CTRL_MODE.get(ctrl_mode, '')}，使能={enabled}，"
                f"关节故障={'、'.join(faults) or '无'}）。不行就给这条臂断电 10 秒再上电。"
            )


def needs_recovery(status: int, ctrl_mode: int, faults: list[str]) -> bool:
    """What a run cannot start with: a status run_real stops on (hardware.FATAL_ARM_STATUS: an emergency stop, a
    collision, ...), a mode that ignores CAN commands, joint driver errors. A disabled arm (the run enables it) and a
    warning (no IK solution, singularity, a target beyond a limit) are fine."""
    return status in hw.FATAL_ARM_STATUS or ctrl_mode in NEEDS_RESET_CTRL_MODES or bool(faults)


def needs_fixing(arm: hw.PiperArm) -> tuple[bool, str]:
    """For an arm a run has open (its feedback streams, so nothing to wait for): whether it needs recover() before it
    moves -- its motors are off, or it reports what needs_recovery names -- and why."""
    robot = arm._robot
    status = robot.get_arm_status()
    if status is None:
        return True, "没有状态反馈"
    arm_status, ctrl_mode = int(status.msg.arm_status), int(status.msg.ctrl_mode)
    faults = joint_faults(arm)
    if needs_recovery(arm_status, ctrl_mode, faults):
        return True, (
            f"arm_status={arm_status}（{ARM_STATUS.get(arm_status, '未知')}）ctrl_mode={ctrl_mode}"
            f"（{CTRL_MODE.get(ctrl_mode, '未知')}）关节故障={'、'.join(faults) or '无'}"
        )
    if not all(robot.get_joints_enable_status_list() or [False]):
        return True, "电机没使能"
    return False, ""


def ensure_ready(name: str, channel: str, firmware: str = "auto", interface: str = "socketcan", confirm=input) -> bool:
    """run_real before it opens an arm: one that reports what needs_recovery names (e.g. the e-stop of the last run;
    enabling it would time out) is recovered -- errors cleared, the controller reset after ``confirm`` when it needs
    it, enabled where it is. False when nothing was needed; SystemExit when the arm does not answer or does not come
    back."""
    try:
        arm = hw.PiperArm(hw.ArmConfig(channel=channel, firmware=firmware, interface=interface), send_commands=False)
    except (RuntimeError, ValueError) as e:
        why = unopenable(channel, e)
        if why is None:
            raise
        raise SystemExit(f"{name}（{channel}）{why}") from None
    try:
        status, ctrl_mode, _, faults = status_of(arm)
        if not needs_recovery(status, ctrl_mode, faults):
            return False
        describe(name, arm)
        print(f"{name}: 清关节错误 → （急停 / 碰撞 / 示教模式时）复位控制器 → 使能…", flush=True)
        recover(name, arm, confirm=confirm)
        describe(name, arm)
        return True
    finally:
        disconnect = getattr(arm._robot, "disconnect", None)  # the run connects again with commands on
        if disconnect is not None:
            disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--left_channel", default="can_l_follower")
    parser.add_argument("--right_channel", default="can_r_follower")
    parser.add_argument("--firmware", default="auto")
    parser.add_argument("--arm", choices=("both", "left", "right"), default="both")
    parser.add_argument("--status_only", action="store_true", help="Only print the arm status; send nothing.")
    parser.add_argument("--no_move", action="store_true", help="Clear the fault and enable, but do not move.")
    parser.add_argument("--duration", type=float, default=8.0, help="Seconds for the move to the zero pose.")
    parser.add_argument("--speed_percent", type=int, default=15)
    parser.add_argument("--max_step", type=float, default=0.02, help="Largest joint change per 25 Hz step (rad).")
    parser.add_argument(
        "--yes", action="store_true", help="Do not wait for Enter before the controller reset and before moving."
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    names = {"left": args.left_channel, "right": args.right_channel}
    selected = ["left", "right"] if args.arm == "both" else [args.arm]
    arms = {n: a for n in selected if (a := connect(names[n], args.firmware)) is not None}
    silent = [n for n in selected if n not in arms]
    if not arms:
        raise SystemExit(1)

    healthy = {n: describe(n, a) for n, a in arms.items()}
    if args.status_only:
        raise SystemExit(1 if silent else 0)
    for n, a in arms.items():
        if not healthy[n]:
            print(f"{n}: 清关节错误 → （急停 / 碰撞 / 示教模式时）复位控制器 → 使能…", flush=True)
            recover(n, a, confirm=(lambda prompt: None) if args.yes else input)
            describe(n, a)
    if args.no_move:
        for a in arms.values():
            a._enable_gripper()
        print("已恢复并使能（含夹爪），臂保持当前姿态。", flush=True)
        raise SystemExit(1 if silent else 0)

    gripper = rc.GripperMapping()
    safety = rc.SafetyFilter(max_joint_step_rad=args.max_step)
    for a in arms.values():
        a._robot.set_speed_percent(int(np.clip(args.speed_percent, 1, 100)))
        a._send_commands = True
        a._enable_gripper()

    def read() -> dict[str, tuple[np.ndarray, float]]:
        return {n: (q, gripper.width_m_to_fraction(w)) for n, a in arms.items() for q, w in [a.read()]}

    start = read()
    for n, (q, g) in start.items():
        print(f"{n} 当前  关节={np.round(q, 3).tolist()} 夹爪={g:.2f}", flush=True)
    if not args.yes:
        input(f"按 Enter 开始回零（{args.duration:.0f} s，手放在急停上；Ctrl-C 取消）：")

    steps = max(1, int(args.duration * 25))
    current = {n: q.copy() for n, (q, _) in start.items()}
    for i in range(1, steps + 1):
        tick = time.monotonic()
        for n, a in arms.items():
            q0, g0 = start[n]
            frac = 1.0 - i / steps
            goal = q0 * frac  # straight line in joint space to the zero pose
            current[n], _ = safety.limit_step(goal, current[n])
            a.command(current[n], gripper.fraction_to_width_m(g0 * frac), safety)
        time.sleep(max(0.0, 0.04 - (time.monotonic() - tick)))
    time.sleep(1.0)
    for n, (q, g) in read().items():
        print(f"{n} 结果  关节={np.round(q, 3).tolist()} 夹爪={g:.2f}（目标全 0）", flush=True)
    raise SystemExit(1 if silent else 0)


if __name__ == "__main__":
    main()
