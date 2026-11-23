"""Hardware access for a dual-arm CobotMagic built from two PiPER arms and three D435 cameras.

The arms are driven through AgileX's pyAgxArm SDK
(https://github.com/agilexrobotics/pyAgxArm), which works with all PiPER
firmware versions. pyAgxArm and pyrealsense2 are imported lazily so that the
rest of the package, and its tests, run on machines without them.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import pathlib
import subprocess
import threading
import time

import numpy as np

from real_robot import conversions as rc

logger = logging.getLogger(__name__)

CAMERA_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")

# A follower streams its feedback (0x2A1-0x2A8) at ~200 Hz and a D435 delivers 30 frames/s. Data older than this means
# the arm left the CAN bus (emergency stop, power, adapter) or the camera stalled: stop instead of acting on it.
MAX_FEEDBACK_AGE_S = 0.25
# ... unless a fresh one comes within this long: right after an episode (its save running in this process, the leaders
# switching role, the reset move starting) the receive threads fell up to ~0.6 s behind on every bus at once (10-30),
# and the console showed a fault that 一键恢复 cleared at once. An arm really off the bus sends nothing in this time.
STALE_GRACE_S = 1.0
MAX_FRAME_AGE_S = 0.5
# arm_status values after which the arm no longer follows commands safely (PiPER protocol, see reset_arms.py).
FATAL_ARM_STATUS = {
    1: "an emergency stop",
    5: "a joint communication error",
    6: "a joint brake that is not released",
    7: "a collision",
    8: "a teaching-drag overspeed",
    9: "an abnormal joint state",
    10: "another error",
    11: "teaching record",
    12: "teaching execution",
    13: "teaching pause",
    14: "main controller over-temperature",
    15: "release resistor over-temperature",
}
# ctrl_mode values in which the arm ignores CAN motion commands.
IGNORING_CTRL_MODES = {2: "teaching", 6: "linkage teaching input", 7: "offline trajectory"}
# A fault stops the run once it has lasted this long, so a status reported for a moment (joint brakes still closed
# right after enabling, one odd frame) does not. An emergency stop or a collision latches until reset_arms clears it,
# so those stop it at once: no e-stop button here, a collision is the protective stop (and the leaders, mirroring the
# followers, must not run on past it).
FAULT_PERSIST_S = 0.2
LATCHED_ARM_STATUS = {1, 7}
# Signals read with every observation for the recordings only (PiperArm.read_extras), sizes per arm.
EXTRA_SIZES = {"joint_velocity": 6, "joint_current": 6, "joint_torque": 6, "gripper_force": 1, "flange_pose": 6}
# A follower whose feedback is older than this when it is checked before a move or recovered, or whose channel is a
# new interface (its adapter plugged out and back in), lost its CAN connection: PiperArm.reconnect opens it again.
LOST_FEEDBACK_S = 0.5
# pyAgxArm returns the joint list as soon as one of its three frames (0x2A5-0x2A7) has come, the joints of the others
# still 0 on a new connection or from long before: wait this long (they come at ~200 Hz) before acting on it.
JOINT_FRAMES_S = 0.05
SYS_CLASS_NET = pathlib.Path("/sys/class/net")
USBDEVFS_RESET = 21780  # _IO('U', 20): reset a USB device, as unplugging it and plugging it back would


def _age(message) -> float:
    stamp = getattr(message, "timestamp", None)
    return time.time() - float(stamp) if stamp else 0.0


def fresh(get, what: str, max_age_s: float = MAX_FEEDBACK_AGE_S, grace_s: float = STALE_GRACE_S):
    """``get()``'s message, once it is at most ``max_age_s`` old; a stale one is read again for up to ``grace_s``
    before check_fresh raises (a receive thread that fell behind catches up, an arm off the bus does not)."""
    message = get()
    deadline = time.monotonic() + grace_s
    while message is not None and _age(message) > max_age_s and time.monotonic() < deadline:
        time.sleep(0.01)
        message = get()
    if message is not None:
        check_fresh(message, what, max_age_s)
    return message


def check_fresh(message, what: str, max_age_s: float = MAX_FEEDBACK_AGE_S) -> None:
    """Raise HardwareFault when a pyAgxArm feedback message was received more than ``max_age_s`` ago.

    pyAgxArm keeps the last message of every kind and stamps it with the CAN receive time (time.time()); a message
    without a stamp is not checked.
    """
    stamp = getattr(message, "timestamp", None)
    if not stamp:
        return
    age = time.time() - float(stamp)
    if age > max_age_s:
        raise rc.HardwareFault(
            f"The {what} is {age:.2f} s old: the arm is off the CAN bus (emergency stop, power, USB-CAN adapter)."
        )


def link_id(channel: str) -> int | None:
    """The kernel's index of a network interface, None when there is none. An adapter plugged out and back in comes back
    as a new interface of the same name with a new index, which a connection opened before does not reach."""
    try:
        return int((SYS_CLASS_NET / channel / "ifindex").read_text())
    except (OSError, ValueError):
        return None


def _is_up(channel: str) -> bool:
    try:
        return bool(int((SYS_CLASS_NET / channel / "flags").read_text(), 16) & 0x1)  # IFF_UP
    except (OSError, ValueError):
        return False


def usb_replug(channel: str, timeout_s: float = 10.0) -> bool:
    """Reset the USB-CAN adapter of a CAN interface in software, as unplugging and plugging it back would (needs
    passwordless sudo): an adapter whose USB endpoint stalled (the kernel logs ``failed to xmit URB: -EPIPE`` and the
    arm's feedback stops) comes back as a new interface, which udev brings up again. False when that cannot be done
    here or the interface is not up again within ``timeout_s``."""
    try:
        device = (SYS_CLASS_NET / channel / "device").resolve().parent  # the adapter's USB device, e.g. .../1-5.4
        node = f"/dev/bus/usb/{int((device / 'busnum').read_text()):03d}/{int((device / 'devnum').read_text()):03d}"
    except (OSError, ValueError):
        return False
    old = link_id(channel)
    code = f"import fcntl, os; fcntl.ioctl(os.open({node!r}, os.O_WRONLY), {USBDEVFS_RESET})"
    try:
        done = subprocess.run(
            ["sudo", "-n", "/usr/bin/python3", "-c", code], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("Could not reset the USB-CAN adapter of %s: %s", channel, exc)
        return False
    if done.returncode:
        logger.warning("Could not reset the USB-CAN adapter of %s: %s", channel, done.stderr.strip()[-300:])
        return False
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        new = link_id(channel)
        if new is not None and new != old and _is_up(channel):
            return True
        time.sleep(0.2)
    return link_id(channel) is not None and _is_up(channel)  # reset without a new interface: let a reconnect tell


@dataclasses.dataclass(frozen=True)
class ArmConfig:
    # CAN channel, e.g. "can_left" with interface "socketcan".
    channel: str
    calibration: rc.ArmCalibration = dataclasses.field(default_factory=rc.ArmCalibration)
    # pyAgxArm PiperFW profile: "auto" reads the firmware version from the arm, as pyAgxArm's
    # detect_piper_series demo does; or "default" (<= S-V1.8-2), "v183", "v188", "v189" (>= S-V1.8-9).
    firmware: str = "auto"
    interface: str = "socketcan"


def _wait_for(read, what: str, timeout_s: float):
    deadline = time.monotonic() + timeout_s
    while True:
        value = read()
        if value is not None:
            return value
        if time.monotonic() > deadline:
            raise RuntimeError(f"Timed out waiting for {what}.")
        time.sleep(0.01)


def _resolve_profile(software_version: str) -> str | None:
    from pyAgxArm import ArmModel  # noqa: PLC0415 - optional hardware dependency
    from pyAgxArm import resolve_firmware_profile  # noqa: PLC0415 - optional hardware dependency

    try:
        return resolve_firmware_profile(ArmModel.PIPER, software_version)
    except Exception:  # Unknown versions are reported by the caller.
        return None


class PiperArm:
    """One PiPER arm with its AgxGripper, addressed through its CAN channel."""

    def __init__(
        self,
        config: ArmConfig,
        *,
        send_commands: bool,
        speed_percent: int = 100,
        gripper_force_n: float = 1.0,
        gripper_max_mm: float | None = None,
        sdk_config_overrides: dict | None = None,
        feedback_timeout_s: float = 3.0,
    ) -> None:
        self._config = config
        self._send_commands = send_commands
        self._gripper_force_n = float(gripper_force_n)
        self._feedback_timeout_s = feedback_timeout_s
        self._sdk_config_overrides = dict(sdk_config_overrides or {})
        self._warned_status: set[int] = set()
        self._fault_since: float | None = None
        self._extras_failed = False
        self._speed_percent = int(np.clip(speed_percent, 1, 100))  # set again on a new connection (reconnect)

        profile = config.firmware
        if profile == "auto":
            profile = self._detect_profile()
        self.firmware_profile = profile
        self._robot, self._gripper = self._connect(profile)
        self._link = link_id(config.channel)

        if config.firmware != "auto":
            firmware = self._robot.get_firmware()
            if firmware is not None:
                self._check_firmware(firmware.get("software_version"), profile)
        if gripper_max_mm is not None:
            self._check_gripper_range(gripper_max_mm)

        if send_commands:
            # Hold where the arm is before the motors take over, so it does not move toward an older target.
            joints = _wait_for(self._robot.get_joint_angles, f"joint feedback on {config.channel}", feedback_timeout_s)
            time.sleep(JOINT_FRAMES_S)  # every joint's frame, not only the first one to come
            joints = self._robot.get_joint_angles() or joints
            check_fresh(joints, f"joint feedback on {config.channel}")
            self._robot.move_j([float(v) for v in joints.msg])
            _wait_for(lambda: True if self._robot.enable() else None, f"enabling the arm on {config.channel}", 5.0)
            self._robot.set_speed_percent(self._speed_percent)
            self._enable_gripper()
        logger.info(
            "Connected PiPER on %s (firmware profile %s, commands %s).",
            config.channel,
            profile,
            "on" if send_commands else "off",
        )

    def _connect(self, profile: str):
        from pyAgxArm import AgxArmFactory  # noqa: PLC0415 - optional hardware dependency
        from pyAgxArm import ArmModel  # noqa: PLC0415 - optional hardware dependency
        from pyAgxArm import create_agx_arm_config  # noqa: PLC0415 - optional hardware dependency

        sdk_config = create_agx_arm_config(
            robot=ArmModel.PIPER,
            firmeware_version=profile,
            interface=self._config.interface,
            channel=self._config.channel,
            **self._sdk_config_overrides,
        )
        robot = AgxArmFactory.create_arm(sdk_config)
        # The effector has to be created before connecting.
        gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        robot.connect()
        return robot, gripper

    def _detect_profile(self, timeout_s: float = 5.0) -> str:
        """Read the firmware version with the baseline profile, then reconnect with the matching one."""
        from pyAgxArm import AgxArmFactory  # noqa: PLC0415 - optional hardware dependency
        from pyAgxArm import ArmModel  # noqa: PLC0415 - optional hardware dependency
        from pyAgxArm import create_agx_arm_config  # noqa: PLC0415 - optional hardware dependency

        probe = AgxArmFactory.create_arm(
            create_agx_arm_config(
                robot=ArmModel.PIPER,
                interface=self._config.interface,
                channel=self._config.channel,
                **self._sdk_config_overrides,
            )
        )
        probe.connect()
        try:
            firmware = _wait_for(probe.get_firmware, f"the firmware version on {self._config.channel}", timeout_s)
        finally:
            probe.disconnect()
        version = firmware.get("software_version")
        profile = _resolve_profile(version) if version else None
        if profile is None:
            raise RuntimeError(
                f"Could not map PiPER firmware {version!r} on {self._config.channel} to a pyAgxArm profile; "
                "set the firmware option explicitly."
            )
        logger.info("PiPER on %s runs firmware %s, using pyAgxArm profile %s.", self._config.channel, version, profile)
        return profile

    def _check_firmware(self, software_version: str | None, profile: str) -> None:
        if not software_version:
            return
        expected = _resolve_profile(software_version)
        if expected is None:
            logger.warning("Could not map PiPER firmware %s to a pyAgxArm profile.", software_version)
        elif expected != profile:
            logger.warning(
                "PiPER on %s runs firmware %s, which pyAgxArm maps to profile %r, but %r is configured. "
                "Set the firmware option to %r or to auto.",
                self._config.channel,
                software_version,
                expected,
                profile,
                expected,
            )

    def _enable_gripper(self, timeout_s: float = 1.0) -> None:
        """Bring up a gripper whose driver is disabled.

        move_gripper_m sends the plain enable code, which a gripper that came up with its driver disabled
        ignores: on a PiPER with firmware S-V1.8-8 the gripper stayed closed while the other arm's opened.
        The enable-and-clear-error code (3) brings it up at its current opening. pyAgxArm has no public
        call for that code, so its control message is sent directly.
        """
        try:
            status = _wait_for(
                self._gripper.get_gripper_status, f"gripper feedback on {self._config.channel}", timeout_s
            )
        except RuntimeError:
            return  # read() reports missing feedback
        foc = getattr(status.msg, "foc_status", None)
        if foc is None or foc.driver_enable_status or getattr(status.msg, "mode", "width") != "width":
            return
        try:
            from pyAgxArm.protocols.can_protocol.drivers.effector.agx_gripper.default.driver import (  # noqa: PLC0415
                ArmMsgGripperCtrl,
            )
        except ImportError:
            logger.warning("The gripper on %s is disabled and this pyAgxArm cannot enable it.", self._config.channel)
            return
        width_um, force_mn = round(abs(float(status.msg.value)) * 1e6), round(self._gripper_force_n * 1e3)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self._gripper._send_msg(ArmMsgGripperCtrl(value=width_um, force=force_mn, status_code=3))  # noqa: SLF001
            time.sleep(0.05)
            if self._gripper.get_gripper_status().msg.foc_status.driver_enable_status:
                logger.info("Enabled the disabled gripper driver on %s.", self._config.channel)
                return
        logger.warning("The gripper on %s stays disabled; it will not open or close.", self._config.channel)

    def _check_gripper_range(self, gripper_max_mm: float) -> None:
        """Warn when the gripper's configured stroke differs from the one the mapping assumes."""
        param = self._gripper.get_gripper_teaching_pendant_param()
        if param is None:
            logger.warning("Could not read the gripper stroke on %s.", self._config.channel)
            return
        configured_mm = float(param.msg.max_range_config) * rc.MM_PER_M
        if configured_mm > 0 and abs(configured_mm - gripper_max_mm) > 1.0:
            logger.warning(
                "The gripper on %s is configured for a %.0f mm stroke, but %.0f mm is assumed. Calibrate it "
                "(pyAgxArm: disable_gripper, set_gripper_teaching_pendant_param(max_range_config=%.3f), "
                "squeeze it closed, calibrate_gripper) or pass the matching gripper_max_mm.",
                self._config.channel,
                configured_mm,
                gripper_max_mm,
                gripper_max_mm / rc.MM_PER_M,
            )

    def read(self) -> tuple[np.ndarray, float]:
        """Return the arm joints in sim radians and the gripper opening width in metres."""
        joints = _wait_for(
            self._robot.get_joint_angles, f"joint feedback on {self._config.channel}", self._feedback_timeout_s
        )
        status = _wait_for(
            self._gripper.get_gripper_status, f"gripper feedback on {self._config.channel}", self._feedback_timeout_s
        )
        joints = fresh(self._robot.get_joint_angles, f"joint feedback on {self._config.channel}") or joints
        status = fresh(self._gripper.get_gripper_status, f"gripper feedback on {self._config.channel}") or status
        mode = getattr(status.msg, "mode", "width")
        if mode != "width":
            # In angle mode the value is in degrees, not metres.
            raise RuntimeError(
                f"The gripper on {self._config.channel} reports {mode!r} mode; switch it to width mode "
                "(any move_gripper_m command does) before reading its opening."
            )
        return self._config.calibration.real_to_sim(np.asarray(joints.msg, dtype=np.float64)), float(status.msg.value)

    def read_extras(self) -> dict[str, np.ndarray]:
        """Signals recorded with the rollouts but not used for control (rollout_v21's ``extra.*`` columns).

        Per joint the motor velocity (rad/s), current (A) and torque (N·m) in the sim sign convention, the gripper
        force (N), and the controller's flange pose [x, y, z (m), roll, pitch, yaw (rad)] in the arm base frame,
        which is where the organizers' ee_pose comes from. A message that has not arrived reads as zeros, and so does
        everything after a read error (logged once): recording must never stop the robot. ``extras_complete`` says
        whether this read had every message (rollout_v21 flags the episode otherwise).
        """
        self.extras_complete = False
        try:
            signs = np.asarray(self._config.calibration.signs, dtype=np.float64)
            motors = [self._robot.get_motor_states(joint) for joint in range(1, 7)]

            def per_joint(name: str) -> np.ndarray:
                return signs * np.array([0.0 if m is None else float(getattr(m.msg, name)) for m in motors])

            gripper = self._gripper.get_gripper_status()
            flange = self._robot.get_flange_pose()
            self.extras_complete = all(m is not None for m in motors) and gripper is not None and flange is not None
            return {
                "joint_velocity": per_joint("velocity"),
                "joint_current": per_joint("current"),
                "joint_torque": per_joint("torque"),
                "gripper_force": np.array([0.0 if gripper is None else float(getattr(gripper.msg, "force", 0.0))]),
                "flange_pose": np.zeros(6) if flange is None else np.asarray(flange.msg, dtype=np.float64)[:6],
            }
        except Exception as exc:  # the extra signals are optional
            if not self._extras_failed:
                self._extras_failed = True
                logger.warning(
                    "Could not read the extra signals of the arm on %s (%s); recording zeros.",
                    self._config.channel,
                    exc,
                )
            return {name: np.zeros(size) for name, size in EXTRA_SIZES.items()}

    def check_status(self) -> None:
        """Raise HardwareFault when the controller reports a fault, or a mode in which it ignores CAN commands, for
        FAULT_PERSIST_S.

        Warnings that do not stop the arm (no IK solution, singularity, target beyond a limit) are logged once.
        """
        status = fresh(self._robot.get_arm_status, f"status of the arm on {self._config.channel}")
        if status is None:
            raise rc.HardwareFault(f"The arm on {self._config.channel} reports no status.")
        arm_status, ctrl_mode = int(status.msg.arm_status), int(status.msg.ctrl_mode)
        if arm_status in FATAL_ARM_STATUS:
            fault = f"reports {FATAL_ARM_STATUS[arm_status]} (arm_status {arm_status})"
        elif ctrl_mode in IGNORING_CTRL_MODES:
            fault = f"is in {IGNORING_CTRL_MODES[ctrl_mode]} mode (ctrl_mode {ctrl_mode}) and ignores motion commands"
        else:
            fault = None
        if fault is not None:
            now = time.monotonic()
            if self._fault_since is None:
                self._fault_since = now
            if now - self._fault_since >= FAULT_PERSIST_S or arm_status in LATCHED_ARM_STATUS:
                raise rc.ArmFault(f"The arm on {self._config.channel} {fault}.", arm_status,
                                  latched=arm_status in LATCHED_ARM_STATUS)
            return
        self._fault_since = None
        if arm_status and arm_status not in self._warned_status:
            self._warned_status.add(arm_status)
            logger.warning("The arm on %s reports arm_status %d.", self._config.channel, arm_status)

    def command(self, sim_q: np.ndarray, width_m: float, safety: rc.SafetyFilter, *, fast: bool = False) -> np.ndarray:
        """Send one joint target (sim radians) and gripper width. Returns the real joint target."""
        real_q = safety.clamp_to_limits(self._config.calibration.sim_to_real(sim_q))
        if not self._send_commands:
            return real_q
        # Position-velocity joint control (MOVE J), smoothed by the arm controller. move_js skips the smoothing
        # (MOVE J + 0xAD, Evo-RL's "high follow"); pyAgxArm rates it extremely high risk, so only the operator
        # takeover uses it, on request (--takeover_high_follow), with targets rate-limited and kept within
        # HIGH_FOLLOW_MAX_LEAD_RAD of the measured joints (real_env.send_human).
        if fast:
            self._robot.move_js([float(v) for v in real_q])
        else:
            self._robot.move_j([float(v) for v in real_q])
        self._gripper.move_gripper_m(value=float(width_m), force=self._gripper_force_n)
        return real_q

    def set_speed_percent(self, percent: int) -> None:
        self._speed_percent = int(np.clip(percent, 1, 100))
        if self._send_commands:
            # pyAgxArm sends the speed with its cached motion mode: after the operator's move_js that is high follow,
            # which the speed change back to the policy's would set again (the reset after 直接采集 then went without
            # the controller's smoothing, 10-27). The normal joint mode (MOVE J, position-velocity) first.
            self._robot.set_motion_mode("j")
            self._robot.set_speed_percent(self._speed_percent)

    def feedback_age(self) -> float:
        """Seconds since the arm last sent its joints or its status, whichever is newer: either one coming means the
        connection is there (inf before any; 0 for messages without a receive time)."""
        ages = []
        for message in (self._robot.get_joint_angles(), self._robot.get_arm_status()):
            if message is not None:
                stamp = getattr(message, "timestamp", None)
                ages.append(time.time() - float(stamp) if stamp else 0.0)
        return min(ages, default=float("inf"))

    def connection_lost(self) -> bool:
        """Its adapter was plugged out (and maybe back in, as a new interface the old connection does not reach), or the
        arm's feedback stopped (a CAN cable out, a stalled adapter): reconnect() has to open the connection again."""
        return link_id(self._config.channel) != self._link or self.feedback_age() > LOST_FEEDBACK_S

    def reconnect(self) -> None:
        """Open the CAN connection again in place, without restarting the run: wait for fresh joint feedback and set the
        speed again (a new connection starts at 100 %). The caller enables the arm (reset_arms.recover).
        RuntimeError / ValueError as when the arm is opened: no such interface, no feedback."""
        with contextlib.suppress(Exception):  # the old socket, maybe of an interface that is gone
            self._robot.disconnect()
        self._robot, self._gripper = self._connect(self.firmware_profile)
        self._link = link_id(self._config.channel)
        self._fault_since = None
        joints = _wait_for(
            self._robot.get_joint_angles, f"joint feedback on {self._config.channel}", self._feedback_timeout_s
        )
        check_fresh(joints, f"joint feedback on {self._config.channel}")
        if self._send_commands:
            self._robot.set_speed_percent(self._speed_percent)
        logger.info("Reconnected the PiPER on %s.", self._config.channel)


# A camera whose reads keep failing (a stall, or off the USB bus) gets its stream restarted this often, so it comes
# back by itself once its hub is plugged in again; its failures are logged at most this often.
CAMERA_RETRY_S = 3.0
CAMERA_LOG_EVERY_S = 10.0
# How long opening a camera waits for its first frame. A D435 sometimes stays silent after its stream starts (10-25:
# a different camera each start, often right after a run that crashed); the reader's first read times out after 5 s
# and then restarts the stream, so the wait must cover that restart instead of giving up at 5 s.
CAMERA_START_S = 20.0
# close() waits this long for the reader, which stops the stream on its way out: longer than a stalled read (the SDK's
# wait_for_frames gives up after 5 s) or a restart in progress (a new device list and stream, slow after a USB hiccup).
CAMERA_CLOSE_S = 10.0
# A colour frame counts as garbled when this share of its pixels has red and blue near zero but green up (11-02, after a
# USB replug: the top camera's depth data came as its colour image, all green; the right wrist's frames came decoded
# wrong, green and magenta, the picture repeated) or is magenta. On 11,148 good frames of 10-30 to 11-02 (all three
# cameras, the green backdrop included) the first share was at most 3.4 % and the second 0; on the 256 bad ones the
# first was at least 43 %. Such frames are dropped, so they reach neither the policy nor the recording.
GARBLED_GREEN_SHARE = 0.25
GARBLED_MAGENTA_SHARE = 0.05
# The stream is started again only when garbled frames have kept coming for this long. A single one is just dropped:
# the next good frame is 33 ms away. Until 11-23 the first garbled frame restarted the stream at once (the retry clock
# ran from the reader's start), which takes 0.5-1 s, longer than MAX_FRAME_AGE_S: every single bad frame became a
# fault in the middle of an episode (three on the bench in two hours on 11-23). Longer than MAX_FRAME_AGE_S, so by
# the time of a restart the robot has stopped on the frame's age anyway.
GARBLED_RESTART_S = 1.0


def garbled(image: np.ndarray) -> bool:
    """True for a colour frame that is not a picture (see GARBLED_GREEN_SHARE); every 4th pixel is enough."""
    small = image[::4, ::4].astype(np.int16)
    r, g, b = small[..., 0], small[..., 1], small[..., 2]
    green = ((r < 25) & (b < 25) & (g > 60)).mean()
    magenta = ((r > g + 50) & (b > g + 50)).mean()
    return bool(green > GARBLED_GREEN_SHARE or magenta > GARBLED_MAGENTA_SHARE)


class WrongCameraStreamError(RuntimeError):
    """A RealSense stream opened, but of another camera, format or size than asked for."""


class RealSenseCamera:
    """Background reader that keeps the latest RGB frame of one RealSense camera."""

    def __init__(self, serial: str, width: int = 640, height: int = 480, fps: int = 30) -> None:
        import pyrealsense2 as rs  # noqa: PLC0415 - optional hardware dependency

        self._rs = rs
        self._serial = serial
        self._size = (width, height, fps)
        self._start()
        self._latest: np.ndarray | None = None
        self._stamp: float | None = None  # time.monotonic() of the latest frame
        self._trouble = ""  # why the latest frame is getting old (reads), for the fault the robot raises
        self._wrong_stream = ""  # the last restart opened another stream (restarts)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        _wait_for(self.latest, f"frames from RealSense camera {serial}", CAMERA_START_S)
        logger.info("RealSense %s streaming %dx%d RGB.", serial, width, height)

    def _start(self) -> None:
        """Open the colour stream on a device list read now. After a USB replug the camera has new video nodes; a
        restart on the pipeline opened before it gave the top camera's depth data as its colour image, and a wrist
        camera frames decoded wrong for two hours (11-02). So each start enumerates the cameras again, and checks it got
        this camera's RGB stream. RuntimeError when the camera is not on the bus, WrongCameraStreamError when the
        stream is not the one asked for."""
        rs = self._rs
        width, height, fps = self._size
        context = rs.context()
        present = {device.get_info(rs.camera_info.serial_number) for device in context.query_devices()}
        if self._serial not in present:
            raise RuntimeError(f"RealSense {self._serial} is not on the USB bus")
        pipeline = rs.pipeline(context)
        config = rs.config()
        config.enable_device(self._serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        profile = pipeline.start(config)
        got = profile.get_device().get_info(rs.camera_info.serial_number)
        stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        if (got, stream.format(), stream.width(), stream.height()) != (self._serial, rs.format.rgb8, width, height):
            with contextlib.suppress(RuntimeError):
                pipeline.stop()
            raise WrongCameraStreamError(
                f"RealSense {self._serial}: opened {got} {stream.format()} {stream.width()}x{stream.height()} instead"
            )
        self._pipeline = pipeline

    def _run(self) -> None:
        """The reader thread. It owns the stream and stops it on its way out; close() only asks it to. A stop from
        close() missed the stream that a restart was opening meanwhile, which then streamed on with no reader (found
        11-02 with a fake pyrealsense2)."""
        try:
            self._loop()
        finally:
            with contextlib.suppress(RuntimeError):  # stopped already, or its device went away
                self._pipeline.stop()

    def _loop(self) -> None:
        failures, bad, bad_since, reported, restarted = 0, 0, 0.0, 0.0, time.monotonic()
        width, height, _ = self._size
        while not self._stop.is_set():
            try:
                frames = self._pipeline.wait_for_frames()
            except RuntimeError as exc:  # no frame within 5 s after a USB hiccup, or off the bus: age() grows
                failures += 1
                self._trouble = f"USB dropped or stalled: {exc}"
                now = time.monotonic()
                if failures == 1 or now - reported >= CAMERA_LOG_EVERY_S:  # off the bus, every read fails at once
                    logger.warning("RealSense %s: %s (%d in a row).", self._serial, exc, failures)
                    reported = now
                if now - restarted >= CAMERA_RETRY_S:
                    restarted = now
                    self._restart()
                self._stop.wait(0.05)
                continue
            if failures:
                logger.info("RealSense %s: frames again after %d failed reads.", self._serial, failures)
            failures = 0
            color = frames.get_color_frame()
            if not color:
                continue
            image = np.asanyarray(color.get_data())
            if image.shape != (height, width, 3) or garbled(image):  # dropped: age() grows, the robot stops
                bad += 1
                self._trouble = "garbled colour frames (restart the console if it stays)"
                now = time.monotonic()
                if bad == 1:
                    bad_since = now
                if bad == 1 or now - reported >= CAMERA_LOG_EVERY_S:
                    logger.error("RealSense %s: garbled colour frame %s (%d in a row).", self._serial, image.shape, bad)
                    reported = now
                if now - bad_since >= GARBLED_RESTART_S and now - restarted >= CAMERA_RETRY_S:  # they keep coming
                    restarted = now
                    self._restart()
                continue
            if bad:
                logger.warning("RealSense %s: good frames again after %d garbled ones.", self._serial, bad)
            bad = 0
            image = image.copy()
            with self._lock:
                self._latest = image
                self._stamp = time.monotonic()
                self._trouble = self._wrong_stream = ""

    def _restart(self) -> None:
        """Start the stream again (the reader thread only): a camera back on the bus streams again. Not once close() has
        asked the reader to stop: it is on its way out."""
        if self._stop.is_set():
            return
        with contextlib.suppress(RuntimeError):  # stopped already when the device went away
            self._pipeline.stop()
        try:
            self._start()
        except WrongCameraStreamError as exc:
            self._wrong_stream = str(exc)
            logger.error("%s", exc)
        except RuntimeError as exc:  # still not on the bus
            logger.debug("RealSense %s: stream restart failed: %s", self._serial, exc)
        else:
            if self._stop.is_set():  # close() came while it opened: no stream outlives the reader
                with contextlib.suppress(RuntimeError):
                    self._pipeline.stop()
                return
            self._wrong_stream = ""
            logger.warning("RealSense %s: stream restarted.", self._serial)

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return self._latest

    def age(self) -> float:
        """Seconds since the latest frame arrived (inf before the first one)."""
        with self._lock:
            return float("inf") if self._stamp is None else time.monotonic() - self._stamp

    def trouble(self) -> str:
        """Why no frame came lately (empty while frames come)."""
        return self._wrong_stream or self._trouble

    def close(self) -> None:
        """Stop the reader, which stops the stream (_run); this waits up to CAMERA_CLOSE_S for a stalled read or a
        restart in progress."""
        self._stop.set()
        self._thread.join(timeout=CAMERA_CLOSE_S)
        if self._thread.is_alive():
            logger.warning(
                "RealSense %s: the reader has not stopped within %.0f s; its stream stops when it does.",
                self._serial,
                CAMERA_CLOSE_S,
            )


@dataclasses.dataclass(frozen=True)
class RobotConfig:
    left_arm: ArmConfig
    right_arm: ArmConfig
    gripper: rc.GripperMapping = dataclasses.field(default_factory=rc.GripperMapping)
    gripper_force_n: float = 1.0
    # RealSense serial numbers keyed by the simulator camera names.
    camera_serials: dict[str, str] = dataclasses.field(default_factory=dict)
    image_width: int = 640
    image_height: int = 480
    speed_percent: int = 100
    # Extra keyword arguments for pyAgxArm's create_agx_arm_config, e.g. for testing.
    sdk_config_overrides: dict = dataclasses.field(default_factory=dict)


class CobotMagicPiper:
    """Dual-arm robot read and commanded in sim units (see ``conversions``)."""

    def __init__(self, config: RobotConfig, *, send_commands: bool) -> None:
        missing = [name for name in CAMERA_NAMES if name not in config.camera_serials]
        if missing:
            raise ValueError(f"Missing RealSense serial numbers for {missing}.")
        self._gripper = config.gripper
        self._send_commands = send_commands
        self._left, self._right = (
            PiperArm(
                arm,
                send_commands=send_commands,
                speed_percent=config.speed_percent,
                gripper_force_n=config.gripper_force_n,
                gripper_max_mm=config.gripper.real_max_mm,
                sdk_config_overrides=config.sdk_config_overrides,
            )
            for arm in (config.left_arm, config.right_arm)
        )
        self._cameras = {
            name: RealSenseCamera(config.camera_serials[name], config.image_width, config.image_height)
            for name in CAMERA_NAMES
        }

    def read_state(self) -> np.ndarray:
        """14-dim state in sim units: joints in radians, grippers as opening fractions.

        Raises HardwareFault on stale feedback and, while commanding, on a controller fault (emergency stop,
        collision, teaching mode, ...).
        """
        if self._send_commands:
            self._left.check_status()
            self._right.check_status()
        left_q, left_w = self._left.read()
        right_q, right_w = self._right.read()
        return rc.join_state(
            left_q,
            self._gripper.width_m_to_fraction(left_w),
            right_q,
            self._gripper.width_m_to_fraction(right_w),
        )

    def read_extras(self) -> dict[str, np.ndarray]:
        """``PiperArm.read_extras`` of both arms, left then right; ``complete`` False when a message was missing."""
        left, right = self._left.read_extras(), self._right.read_extras()
        extras = {name: np.concatenate([left[name], right[name]]).astype(np.float32) for name in EXTRA_SIZES}
        extras["complete"] = bool(
            getattr(self._left, "extras_complete", True) and getattr(self._right, "extras_complete", True)
        )
        return extras

    def read_images(self) -> dict[str, np.ndarray]:
        images = {}
        for name, camera in self._cameras.items():
            image = camera.latest()
            if image is None:
                raise RuntimeError(f"Camera {name} has no frame.")
            age = camera.age()
            if age > MAX_FRAME_AGE_S:
                trouble = camera.trouble() or "USB dropped or stalled"
                raise rc.HardwareFault(f"Camera {name} has delivered no frame for {age:.1f} s ({trouble}).")
            images[name] = image
        return images

    def command(self, target: np.ndarray, safety: rc.SafetyFilter, *, fast: bool = False) -> None:
        """Send a 14-dim target: joints in sim radians, grippers as opening fractions."""
        left_q, left_f, right_q, right_f = rc.split_state(target)
        self._left.command(left_q, self._gripper.fraction_to_width_m(left_f), safety, fast=fast)
        self._right.command(right_q, self._gripper.fraction_to_width_m(right_f), safety, fast=fast)

    def set_speed_percent(self, percent: int) -> None:
        """Speed of the arm controllers' joint moves, e.g. raised while the operator drives. Both arms get it even when
        one fails (its adapter just knocked out): the other must not stay at the takeover speed."""
        failed = None
        for arm in (self._left, self._right):
            try:
                arm.set_speed_percent(percent)
            except Exception as exc:  # noqa: BLE001 - raised once both arms have had it
                failed = failed or exc
        if failed is not None:
            raise failed

    def follower_arms(self) -> dict[str, PiperArm]:
        """The two arms by side, for reset_arms.py's recovery (run_real.recover_followers, the console's 一键恢复)."""
        return {"left": self._left, "right": self._right}

    def close(self) -> None:
        # The arms are left enabled on purpose: disabling a PiPER releases its motors
        # and raised joints fall under gravity. Park the arms before powering them off.
        for camera in self._cameras.values():
            camera.close()
