"""The firmware joint limits: from the command line, on an arm that only remembers what it is told; and that planning
and the driver's clamp keep to what the arm's firmware holds."""

import json

import numpy as np
import pytest

from tablet_tap import app as cli_context
from tablet_tap.cli import main as cli
from tablet_tap.cli.commands import limits as limits_cmd
from tablet_tap.robot.arm import FakeArm


class LimitArm(FakeArm):
    def __init__(self):
        super().__init__(np.zeros(6))
        self.limits = {j: (np.radians(-70.0), np.radians(70.0)) for j in range(1, 7)}
        self.writes = []

    def firmware_joint_limits(self):
        return dict(self.limits)

    def set_firmware_joint_limit(self, joint, *, lowest=None, highest=None):
        self.writes.append((joint, lowest, highest))
        low, high = self.limits[joint]
        self.limits[joint] = (low if lowest is None else lowest, high if highest is None else highest)
        return True


@pytest.fixture
def bench(tmp_path, monkeypatch):
    arm = LimitArm()
    monkeypatch.setattr(cli_context, "open_arm", lambda *args, **kwargs: arm)
    monkeypatch.setattr(limits_cmd, "require_hardware", lambda what: None)
    monkeypatch.setattr(limits_cmd, "other_arm_programs", lambda: [])
    monkeypatch.setattr(limits_cmd, "console_holds_arm", lambda url: False)
    setup = tmp_path / "setup.json"
    setup.write_text(json.dumps({"arm": {"channel": "can_l_follower"}}))
    return arm, setup, ["--setup", str(setup), "--view", str(tmp_path / "view.json"), "--fake", "--yes"]


def test_show_change_and_restore(bench, capsys, monkeypatch):
    arm, setup, base = bench
    assert cli.main([*base, "limits"]) == 0
    out = capsys.readouterr().out
    assert "5 号 -70…70" in out and "规划用的限位已经按它更新" in out and arm.writes == []
    rows = lambda: json.loads(setup.read_text())["arm"]["firmware_limits_deg"]  # noqa: E731
    assert rows() == {"can_l_follower": [[-70.0, 70.0]] * 6}  # reading the arm is enough for planning to follow it
    assert cli.main([*base, "limits"]) == 0
    assert "已经按它更新" not in capsys.readouterr().out  # nothing new the second time
    assert cli.main([*base, "limits", "--joint5_max", "75"]) == 0
    assert np.degrees(arm.limits[5][1]) == pytest.approx(75.0) and arm.limits[5][0] == pytest.approx(np.radians(-70.0))
    assert rows()["can_l_follower"][4] == [-70.0, 75.0] and rows()["can_l_follower"][3] == [-70.0, 70.0]
    saved = json.loads((setup.parent / "firmware_limits_can_l_follower.json").read_text())
    assert saved["limits_deg"]["5"] == [-70.0, 70.0]  # what it was before the first change
    assert cli.main([*base, "limits", "--joint5_max", "74"]) == 0  # a second change keeps the first record
    assert json.loads((setup.parent / "firmware_limits_can_l_follower.json").read_text()) == saved
    assert cli.main([*base, "limits", "--restore"]) == 0
    assert np.degrees(arm.limits[5][1]) == pytest.approx(70.0)
    assert rows() == {"can_l_follower": [[-70.0, 70.0]] * 6}
    # the other arm gets its own row, and reading it does not make it the arm that holds the pen
    assert cli.main([*base, "--channel", "can_r_follower", "limits"]) == 0
    assert set(rows()) == {"can_l_follower", "can_r_follower"}
    assert json.loads(setup.read_text())["arm"]["channel"] == "can_l_follower"


def test_refusals_write_nothing(bench, capsys, monkeypatch):
    arm, _, base = bench
    assert cli.main([*base, "limits", "--joint5_max", "80"]) != 0  # never tried: may be the mechanical stop
    assert cli.main([*base, "limits", "--restore"]) != 0  # nothing was saved
    monkeypatch.setattr(limits_cmd, "other_arm_programs", lambda: ["123 run_real.py"])
    assert cli.main([*base, "limits", "--joint5_max", "75"]) != 0
    monkeypatch.setattr(limits_cmd, "other_arm_programs", lambda: [])
    monkeypatch.setattr(limits_cmd, "console_holds_arm", lambda url: True)
    assert cli.main([*base, "limits", "--joint5_max", "75"]) != 0
    assert "断开" in capsys.readouterr().out and arm.writes == []


def test_planning_and_the_drivers_clamp_keep_to_the_arms_firmware(monkeypatch):
    import types

    from real_robot import conversions as rc

    from tablet_tap.config import Setup
    from tablet_tap.robot import factory
    from tablet_tap.robot import piper

    monkeypatch.setattr(piper, "other_arm_programs", lambda: [])
    monkeypatch.setattr(piper.PiperPenArm, "_connect", staticmethod(lambda *args: types.SimpleNamespace()))
    setup = Setup()  # the bench: the left arm's joint 5 goes to 75 deg as in the URDF, the right arm's stops at 70
    setup.arm.firmware_limits_deg["can_r_follower"] = [[-150.0, 150.0], [0.0, 180.0], [-170.0, 0.0], [-100.0, 100.0],
                                                       [-70.0, 70.0], [-180.0, 180.0]]  # fmt: skip
    setup.arm.joint_offsets[4] = 0.044
    for channel, joint5 in (("can_l_follower", (-70.0, 75.0)), ("can_r_follower", (-70.0, 70.0))):
        setup.arm.channel = channel
        clamp = factory.open_arm(setup.arm, factory.ArmMode.READ_ONLY)._safety
        assert np.allclose(np.degrees(clamp.joint_limits_rad[4]), joint5)
        assert np.allclose(np.degrees(clamp.joint_limits_rad[[3, 5]]), [[-100.0, 100.0], [-180.0, 180.0]])
        pen = setup.pen_arm()
        calibration = rc.ArmCalibration(signs=tuple(setup.arm.joint_signs), offsets_rad=tuple(setup.arm.joint_offsets))
        # the furthest that planning goes is sent as it is, and is inside the firmware's limits
        for q in (pen.lower, pen.upper):
            reading = calibration.sim_to_real(q)
            assert np.array_equal(clamp.clamp_to_limits(reading), reading)
            assert np.all(reading > clamp.joint_limits_rad[:, 0]) and np.all(reading < clamp.joint_limits_rad[:, 1])
