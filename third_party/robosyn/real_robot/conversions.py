"""Unit conversions and safety filtering between the simulator convention and dual PiPER arms.

The fmc3_no1 checkpoints were trained and evaluated in EmbodiChain, so the real
robot is presented to them in exactly the simulator's units ("sim units"):

    index  0-5   left arm joints 1-6, radians
    index  6     left gripper
    index  7-12  right arm joints 1-6, radians
    index  13    right gripper

Gripper in the observation: the opening fraction of the simulated V100 gripper,
0 closed .. 1 fully open (each finger travels 0..0.05 m, 100 mm in total).

Gripper in the action: what ``deploy_policy._format_env_action`` hands to
``env.step``, i.e. after the task's ``gripper_action_scale``. The simulator
executes it as metres per finger and clamps it to 0..0.05 m, so the opening
fraction it produces is ``clip(value / 0.05, 0, 1)``.

Our PiPER grippers open 100 mm, like the simulated V100 (``real_max_mm`` = 100), so fraction 1 is
100 mm and widths match the simulator exactly. If a gripper with a shorter stroke is fitted, set
``--gripper_max_mm`` and pick one of two mappings:

* ``stretch``: fraction 1 is the full ``real_max_mm``. Open and closed look the same to
  the policy as in simulation; an object of width w reads w / real_max_mm instead of w / 100.
* ``physical``: fraction 1 is 100 mm, as in simulation, capped at ``real_max_mm``. Widths match, but
  a fully open real gripper reads real_max_mm / 100.
(With ``real_max_mm`` = 100 both mappings are the same.)

pyAgxArm reads and writes joint angles in radians and the gripper opening width
in metres. Everything in this module is pure numpy.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np

STATE_DIM = 14
LEFT_ARM = slice(0, 6)
LEFT_GRIPPER = 6
RIGHT_ARM = slice(7, 13)
RIGHT_GRIPPER = 13
GRIPPER_DIMS = (LEFT_GRIPPER, RIGHT_GRIPPER)
ARM_DIMS = tuple(range(6)) + tuple(range(7, 13))

# Simulated CobotMagic V100 gripper: each finger travels 0..0.05 m.
SIM_FINGER_TRAVEL_M = 0.05
SIM_FULL_OPEN_MM = 100.0
GRIPPER_MAPPINGS = ("stretch", "physical")

# PiPER joint limits in radians, as in agx_arm_urdf piper_description.urdf, the pyAgxArm defaults and the official
# simulation (EmbodiChain CobotMagicArm/CobotMagicWithGripperV100.urdf).
PIPER_JOINT_LIMITS_RAD = np.array(
    [
        [-2.6179938, 2.6179938],
        [0.0, 3.1415926],
        [-2.9670597, 0.0],
        [-1.7453292, 1.7453292],
        [-1.2217304, 1.2217304],
        [-2.0943951, 2.0943951],
    ],
    dtype=np.float64,
)

# What SafetyFilter clamps the commands to: the range of the organizers' real CobotMagic, the robot of the final
# evaluation. Its real demos (Real/cobotmagic_Real_*, both arms, every frame) turn the wrist past the URDF: joint 4 to
# -1.785, joint 5 to +-1.31, joint 6 from -2.58 to +2.77 rad (the URDF's +-120 deg is +-2.09), so joints 4-6 are
# widened to cover all of them (rounded up, the 0.02 margin inside); joints 1-3 stay the URDF's (their demos rest up to
# 0.03 rad past the zero stops of joints 2 and 3: calibration, not reach). 10-27: before, the bench held the follower's
# wrist at +-120 deg while the leader in the operator's hand turned on, and lagged behind it.
REAL_JOINT_LIMITS_RAD = PIPER_JOINT_LIMITS_RAD.copy()
REAL_JOINT_LIMITS_RAD[3:] = [[-1.81, 1.81], [-1.35, 1.35], [-2.82, 2.82]]

MM_PER_M = 1000.0


class HardwareFault(RuntimeError):  # noqa: N818 - a fault of the robot
    """The robot can no longer be trusted: stale camera frames or CAN feedback, or the arm controller reports a fault.

    run_real stops the run on it (the episode is not recorded); the hardware has to be checked before going on.
    """


class ArmFault(HardwareFault):
    """The arm controller reports a fault (``arm_status``, PiPER protocol). ``latched``: one that latches until it
    is cleared -- a collision or an emergency stop: what the policy did (the web console keeps such an episode as a
    failure while RLT runs), not what the hardware did."""

    def __init__(self, message: str, arm_status: int, *, latched: bool) -> None:
        super().__init__(message)
        self.arm_status, self.latched = int(arm_status), bool(latched)


@dataclasses.dataclass(frozen=True)
class ArmCalibration:
    """Per-joint mapping ``real = sign * sim + offset``.

    The simulated zero pose matches the real PiPER zero pose, so the offsets
    stay at zero. Flip a sign only if a single-joint jog moves the real joint
    the opposite way from the simulated one.
    """

    signs: tuple[float, ...] = (1.0,) * 6
    offsets_rad: tuple[float, ...] = (0.0,) * 6

    def __post_init__(self):
        if len(self.signs) != 6 or len(self.offsets_rad) != 6:
            raise ValueError("ArmCalibration needs exactly 6 signs and 6 offsets.")
        if any(s not in (1.0, -1.0) for s in self.signs):
            raise ValueError(f"Joint signs must be +1 or -1, got {self.signs}.")

    def sim_to_real(self, q: np.ndarray) -> np.ndarray:
        return np.asarray(self.signs) * np.asarray(q, dtype=np.float64) + np.asarray(self.offsets_rad)

    def real_to_sim(self, q: np.ndarray) -> np.ndarray:
        return (np.asarray(q, dtype=np.float64) - np.asarray(self.offsets_rad)) * np.asarray(self.signs)


@dataclasses.dataclass(frozen=True)
class GripperMapping:
    """Map between the simulator's gripper opening fraction and the real opening width."""

    mode: str = "stretch"
    # Largest opening the real gripper reaches (our PiPER grippers: 100 mm, same as the simulated V100).
    real_max_mm: float = 100.0

    def __post_init__(self):
        if self.mode not in GRIPPER_MAPPINGS:
            raise ValueError(f"Gripper mapping must be one of {GRIPPER_MAPPINGS}, got {self.mode!r}.")

    @property
    def fraction_full_mm(self) -> float:
        """Opening that the fraction 1.0 stands for."""
        return self.real_max_mm if self.mode == "stretch" else SIM_FULL_OPEN_MM

    def fraction_to_width_m(self, fraction: float) -> float:
        opening_mm = float(np.clip(fraction, 0.0, 1.0)) * self.fraction_full_mm
        return min(opening_mm, self.real_max_mm) / MM_PER_M

    def width_m_to_fraction(self, width_m: float) -> float:
        return float(np.clip(abs(float(width_m)) * MM_PER_M / self.fraction_full_mm, 0.0, 1.0))


def env_gripper_action_to_fraction(value: float) -> float:
    """Opening fraction the simulator reaches for an env gripper action (metres per finger, clamped)."""
    return float(np.clip(float(value) / SIM_FINGER_TRAVEL_M, 0.0, 1.0))


@dataclasses.dataclass
class SafetyFilter:
    """Clamp commands to the joint limits of the organizers' real robot (REAL_JOINT_LIMITS_RAD) and limit the per-step
    joint change.

    ``max_joint_step_rad`` is applied per control step. At 25 Hz the default
    0.1 rad corresponds to 2.5 rad/s, below the PiPER rated joint speeds.
    """

    max_joint_step_rad: float = 0.1
    joint_limit_margin_rad: float = 0.02
    joint_limits_rad: np.ndarray = dataclasses.field(default_factory=REAL_JOINT_LIMITS_RAD.copy)

    def clamp_to_limits(self, real_arm_q: np.ndarray) -> np.ndarray:
        low = self.joint_limits_rad[:, 0] + self.joint_limit_margin_rad
        high = self.joint_limits_rad[:, 1] - self.joint_limit_margin_rad
        return np.clip(real_arm_q, low, high)

    def limit_step(self, target: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, float]:
        """Move from ``current`` toward ``target`` by at most ``max_joint_step_rad`` per joint.

        Returns the limited target and the largest requested step.
        """
        delta = np.asarray(target, dtype=np.float64) - np.asarray(current, dtype=np.float64)
        largest = float(np.max(np.abs(delta))) if delta.size else 0.0
        limited = np.asarray(current) + np.clip(delta, -self.max_joint_step_rad, self.max_joint_step_rad)
        return limited, largest


def split_state(state: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, float]:
    state = np.asarray(state, dtype=np.float64).reshape(-1)
    if state.shape[0] < STATE_DIM:
        raise ValueError(f"Expected at least {STATE_DIM} values, got {state.shape[0]}.")
    return state[LEFT_ARM], float(state[LEFT_GRIPPER]), state[RIGHT_ARM], float(state[RIGHT_GRIPPER])


def join_state(left_arm: np.ndarray, left_gripper: float, right_arm: np.ndarray, right_gripper: float) -> np.ndarray:
    out = np.zeros(STATE_DIM, dtype=np.float32)
    out[LEFT_ARM] = left_arm
    out[LEFT_GRIPPER] = left_gripper
    out[RIGHT_ARM] = right_arm
    out[RIGHT_GRIPPER] = right_gripper
    return out


def interpolate_path(start: np.ndarray, goal: np.ndarray, num_steps: int) -> np.ndarray:
    """Linear path from ``start`` to ``goal`` with ``num_steps`` waypoints, excluding ``start``."""
    num_steps = max(1, int(num_steps))
    alphas = np.linspace(0.0, 1.0, num_steps + 1)[1:, None]
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    return start + alphas * (goal - start)


def warn_on_jump(largest_step: float, threshold: float, logger: logging.Logger) -> None:
    if largest_step > threshold:
        logger.warning(
            "Policy requested a %.3f rad joint step; limited to %.3f rad. "
            "Frequent warnings usually mean a state or unit mismatch.",
            largest_step,
            threshold,
        )
