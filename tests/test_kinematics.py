"""Pinocchio on the arm's URDF against the kinematics the bench has used so far, and against itself."""

import threading

import numpy as np
import pinocchio as pin

import tablet_tap  # noqa: F401 - puts third_party/robosyn on the path

from dock_switch import fk as bench_fk  # isort: skip - RoboSynChallenge's FK of the same arm (fingertip TCP)

from tablet_tap.core import kinematics
from tablet_tap.core.kinematics import PenArm

RNG = np.random.default_rng(7)


def random_joints(arm: PenArm, n: int) -> np.ndarray:
    return RNG.uniform(arm.lower, arm.upper, size=(n, 6))


def test_the_urdf_is_the_arm_the_bench_drives():
    assert kinematics.URDF_PATH.is_file()
    model = pin.buildModelFromUrdf(str(kinematics.URDF_PATH))
    assert model.nq == 6 and list(model.names)[1:] == [f"joint{i}" for i in range(1, 7)]
    from real_robot import conversions as rc

    assert np.allclose(kinematics.DRIVER_JOINT_LIMITS_RAD, rc.REAL_JOINT_LIMITS_RAD, atol=1e-4)
    assert kinematics.JOINT_LIMITS_RAD is kinematics.DRIVER_JOINT_LIMITS_RAD  # planning keeps to the driver's limits
    # the simulator's limits in the URDF are close to them but not the same (joints 4, 5 narrower, joint 6 wider)
    assert np.abs(kinematics.model_joint_limits() - kinematics.DRIVER_JOINT_LIMITS_RAD).max() < 0.2


def test_forward_kinematics_equals_the_bench_chain():
    for tip_len in (0.0, 0.04, 0.065, 0.1):
        arm = PenArm(tip_len=tip_len)
        q = random_joints(arm, 300)
        tool = np.eye(4)
        tool[2, 3] = tip_len
        assert np.abs(arm.fk(q) - bench_fk(q) @ tool).max() < 1e-12
    assert PenArm().fk(np.zeros(6)).shape == (4, 4) and PenArm().fk(np.zeros((2, 3, 6))).shape == (2, 3, 4, 4)


def test_jacobian_is_the_derivative_of_the_pose():
    arm = PenArm()
    for q in random_joints(arm, 5):
        here, numeric = arm.fk(q), np.zeros((6, 6))
        for j in range(6):
            step = np.zeros(6)
            step[j] = 1e-6
            numeric[:, j] = pin.log6(pin.SE3(np.linalg.inv(here) @ arm.fk(q + step))).vector / 1e-6
        assert np.abs(arm.jacobian(q) - numeric).max() < 1e-6


def test_inverse_kinematics_finds_the_pose_and_stays_inside_the_limits():
    arm = PenArm()
    joints = random_joints(arm, 400)
    solved = 0
    for q in joints:
        seed = np.clip(q + RNG.normal(0, 0.1, 6), arm.lower, arm.upper)
        found, miss_pos, miss_rot = arm.ik(arm.fk(q), seed)
        assert np.all(found >= arm.lower - 1e-12) and np.all(found <= arm.upper + 1e-12)
        if miss_pos < 2e-4 and miss_rot < 5e-3:
            solved += 1
            assert np.abs(arm.fk(found) - arm.fk(q))[:3, 3].max() < 2e-4
    assert solved >= 0.98 * len(joints)  # the few misses end on a joint limit from their seed


def test_two_threads_do_not_share_scratch_memory():
    arm = PenArm()
    joints = random_joints(arm, 200)
    expected = arm.fk(joints)
    results = {}

    def work(name: str, order: np.ndarray) -> None:
        results[name] = np.array([arm.ik(expected[i], joints[i] + 0.05)[1] for i in order])

    threads = [
        threading.Thread(target=work, args=(n, o)) for n, o in (("a", np.arange(200)), ("b", np.arange(200)[::-1]))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert np.allclose(results["a"], results["b"][::-1], atol=1e-9)
