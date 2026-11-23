"""One PiPER arm with the stylus: forward and inverse kinematics of the pen tip, with Pinocchio on the arm's URDF.

The model is ``assets/CobotMagicArm/CobotMagicNoGripper.urdf`` (the organizers' single-arm URDF; the real arm's joint
zero is the model's). The tool is an operational frame added to it: the fingertip TCP of the V100 gripper in ``link6``
moved ``tip_len`` along its z axis, so z points out of the gripper along the pen and the origin is the pen's rubber
tip. Everything is in the arm's own base frame (the URDF's root: x forward, z up, metres).

IK is closed-loop inverse kinematics on that frame: the pose error is Pinocchio's ``log6`` in the tool frame, the step
is damped least squares on the frame Jacobian, and the joints are kept inside the planning limits.
"""

from __future__ import annotations

import dataclasses
import functools
import pathlib
import threading

import numpy as np
import pinocchio as pin

from tablet_tap.errors import Unreachable

URDF_PATH = pathlib.Path(__file__).resolve().parents[2] / "assets" / "CobotMagicArm" / "CobotMagicNoGripper.urdf"
FLANGE_FRAME = "link6"
TIP_FRAME = "pen_tip"
# The fingertip TCP in link6 (EmbodiChain's cobotmagic OPWSolverCfg tcp, RoboSynChallenge's dock_switch.TCP): x and y
# flipped, 0.143 m out along z.
FINGERTIP_IN_FLANGE = pin.SE3(np.diag([-1.0, -1.0, 1.0]), np.array([0.0, 0.0, 0.143]))
DEFAULT_TIP_LEN_M = 0.065  # pen tip beyond the fingertip plane (holder frame Z_H = 65 mm); measure on the bench

# What the real arm's driver accepts (real_robot.conversions.REAL_JOINT_LIMITS_RAD; kept here so that planning needs
# no robot libraries, and checked against it by the tests).
DRIVER_JOINT_LIMITS_RAD = np.array(
    [[-2.618, 2.618], [0.0, 3.1416], [-2.9671, 0.0], [-1.81, 1.81], [-1.35, 1.35], [-2.82, 2.82]], dtype=np.float64
)
LIMIT_MARGIN_RAD = 0.05  # SafetyFilter clamps at 0.02; planning stays further inside so that it never clamps

DAMPING = 0.02
ROT_LENGTH = 0.15  # metres per radian: how a rotation error is weighed against a position error
ITERATIONS = 100
POS_TOL_M, ROT_TOL_RAD = 2e-5, 2e-4

_local = threading.local()  # pin.Data is scratch memory of one computation: one per thread and model


@functools.cache
def _arm_model(urdf_path: str) -> pin.Model:
    return pin.buildModelFromUrdf(urdf_path)


@functools.cache
def _pen_model(urdf_path: str, tip_len: float) -> tuple[pin.Model, int]:
    """The arm's model with the pen-tip frame added, and that frame's id."""
    model = pin.buildModelFromUrdf(urdf_path)  # its own model: the frame belongs to this pen length
    flange = model.frames[model.getFrameId(FLANGE_FRAME)]
    tip_in_flange = FINGERTIP_IN_FLANGE * pin.SE3(np.eye(3), np.array([0.0, 0.0, tip_len]))
    frame = pin.Frame(
        TIP_FRAME,
        flange.parent,
        model.getFrameId(FLANGE_FRAME),
        flange.placement * tip_in_flange,
        pin.FrameType.OP_FRAME,
    )
    return model, model.addFrame(frame)


def model_joint_limits(urdf_path: str | pathlib.Path = URDF_PATH) -> np.ndarray:
    """(6, 2) lower and upper joint limits written in the URDF."""
    model = _arm_model(str(urdf_path))
    return np.stack([model.lowerPositionLimit, model.upperPositionLimit], axis=1)


# Planning keeps to what the real arm's driver accepts, as before the URDF was the model: the URDF's own limits are
# the simulator's (narrower on joints 4 and 5, wider on joint 6) and are not what stops the real arm.
JOINT_LIMITS_RAD = DRIVER_JOINT_LIMITS_RAD


@dataclasses.dataclass(frozen=True)
class PenArm:
    """Kinematics of one arm holding the pen."""

    tip_len: float = DEFAULT_TIP_LEN_M
    limits: np.ndarray = dataclasses.field(default_factory=lambda: JOINT_LIMITS_RAD.copy())
    margin: float = LIMIT_MARGIN_RAD
    urdf_path: str = str(URDF_PATH)

    @property
    def lower(self) -> np.ndarray:
        return self.limits[:, 0] + self.margin

    @property
    def upper(self) -> np.ndarray:
        return self.limits[:, 1] - self.margin

    def _kinematics(self) -> tuple[pin.Model, pin.Data, int]:
        model, tip = _pen_model(self.urdf_path, float(self.tip_len))
        cache = _local.__dict__.setdefault("data", {})
        data = cache.get(id(model))
        if data is None:
            data = cache[id(model)] = model.createData()
        return model, data, tip

    def fk(self, q: np.ndarray) -> np.ndarray:
        """(..., 6) joints (rad) -> (..., 4, 4) pen-tip poses."""
        q = np.asarray(q, np.float64)
        model, data, tip = self._kinematics()
        flat = q.reshape(-1, model.nq)
        poses = np.empty((len(flat), 4, 4))
        for i, joints in enumerate(flat):
            pin.forwardKinematics(model, data, joints)
            poses[i] = pin.updateFramePlacement(model, data, tip).homogeneous
        return poses.reshape(*q.shape[:-1], 4, 4)

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """(6, 6) Jacobian of the pen tip in its own frame: rows are the linear then the angular velocity."""
        model, data, tip = self._kinematics()
        return pin.computeFrameJacobian(model, data, np.asarray(q, np.float64).reshape(model.nq), tip, pin.LOCAL)

    def ik(self, target: np.ndarray, q0: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Joints for the pen-tip pose ``target`` (4, 4), starting from ``q0``; and the remaining miss (m, rad)."""
        model, data, tip = self._kinematics()
        wanted = pin.SE3(np.asarray(target[:3, :3], np.float64), np.asarray(target[:3, 3], np.float64))
        q = np.clip(np.asarray(q0, np.float64).reshape(model.nq), self.lower, self.upper)
        weight = np.array([1.0, 1.0, 1.0, ROT_LENGTH, ROT_LENGTH, ROT_LENGTH])
        eye = np.eye(6)
        miss_pos = miss_rot = np.inf
        for i in range(ITERATIONS + 1):
            pin.forwardKinematics(model, data, q)
            to_target = pin.updateFramePlacement(model, data, tip).actInv(wanted)  # the target seen from the tip
            miss_pos = float(np.linalg.norm(to_target.translation))
            miss_rot = float(np.linalg.norm(pin.log3(to_target.rotation)))
            if i == ITERATIONS or (miss_pos < POS_TOL_M and miss_rot < ROT_TOL_RAD):
                break
            error = weight * pin.log6(to_target).vector
            jac = pin.computeFrameJacobian(model, data, q, tip, pin.LOCAL)
            jac = weight[:, None] * (pin.Jlog6(to_target.inverse()) @ jac)  # d(error) = -jac dq
            step = jac.T @ np.linalg.solve(jac @ jac.T + DAMPING**2 * eye, error)
            q = np.clip(pin.integrate(model, q, step), self.lower, self.upper)
        return q, miss_pos, miss_rot

    def solve(self, target: np.ndarray, q0: np.ndarray, *, pos_tol: float = 2e-4, rot_tol: float = 5e-3) -> np.ndarray:
        """``ik`` that raises ``Unreachable`` when the pose is missed by more than the tolerances."""
        q, miss_pos, miss_rot = self.ik(target, q0)
        if miss_pos > pos_tol or miss_rot > rot_tol:
            at_limit = [j + 1 for j in range(6) if q[j] <= self.lower[j] + 1e-6 or q[j] >= self.upper[j] - 1e-6]
            raise Unreachable(
                f"IK misses the pose by {miss_pos * 1e3:.2f} mm / {np.degrees(miss_rot):.2f} deg"
                + (f" (joints at their limit: {at_limit})" if at_limit else "")
            )
        return q


def pose(rot: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """(3, 3), (3,) -> (4, 4)."""
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = rot, pos
    return out
