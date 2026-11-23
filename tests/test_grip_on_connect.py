"""Taking control of the arm must not let the gripper run to an old target (it dropped the pen holder on 11-23)."""

import types

from tablet_tap.robot import piper


def test_the_grip_is_locked_before_and_after_the_arm_takes_commands(monkeypatch):
    log = []

    class Arm:
        def __init__(self, config, *, send_commands, **kwargs):
            log.append("connect" if send_commands else "look")
            self._gripper = types.SimpleNamespace(move_gripper_m=lambda value, force: log.append((value, force)))
            self._robot = types.SimpleNamespace(disconnect=lambda: log.append("disconnect"))

    hardware = types.SimpleNamespace(PiperArm=Arm, ArmConfig=lambda **k: None)
    rc = types.SimpleNamespace(ArmCalibration=lambda **k: None)
    monkeypatch.setattr(piper.time, "sleep", lambda s: None)
    connect = lambda commands: piper.PiperPenArm._connect(
        hardware, rc, "can", (1.0,) * 6, (0.0,) * 6, "auto", "socketcan", commands, 30, 5.0, 0.0
    )  # noqa: E731
    connect(True)
    assert log == ["look"] + [(0.0, 5.0)] * 5 + ["disconnect", "connect"] + [(0.0, 5.0)] * 5
    log.clear()
    connect(False)
    assert log == ["look"]  # a read-only connection never touches the gripper


def test_an_open_gripper_is_not_closed_by_connecting():
    arm = types.SimpleNamespace(read=lambda: (None, 0.074))
    assert piper.hold_width(arm, 0.0) == 0.074
    arm = types.SimpleNamespace(read=lambda: (None, 0.0212))
    assert piper.hold_width(arm, 0.0) == 0.0  # it holds the pen holder: keep squeezing
