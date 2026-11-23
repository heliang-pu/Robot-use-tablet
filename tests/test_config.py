"""The setup file: typed view, round trip, what is refused."""

import json

from helpers import true_screen
import numpy as np
import pytest

from tablet_tap.config import Setup
from tablet_tap.config import SetupStore
from tablet_tap.errors import SetupError


def test_defaults_and_round_trip(tmp_path):
    store = SetupStore(tmp_path / "calib" / "setup.json")
    setup = store.load()  # no file yet
    assert setup.arm.channel is None and setup.arm.speed_percent == 30 and setup.screen_map is None
    assert setup.motion_limits().press_m == 0.0025 and abs(setup.pen_arm().tip_len - 0.065) < 1e-12

    setup.arm.channel, setup.camera, setup.park_q = "can_r_follower", "cv:0", [0.0] * 6
    setup.screen_map, setup.limits["press_m"] = true_screen(), 0.0006
    setup.records["probe_points"] = [{"u": 0.5}]
    store.save(setup)
    assert not list(tmp_path.glob("calib/*.bak.json"))  # nothing was replaced
    again = store.load()
    assert again.arm.channel == "can_r_follower" and again.camera == "cv:0" and again.park_q == [0.0] * 6
    assert np.allclose(again.screen_map.origin, setup.screen_map.origin) and again.motion_limits().press_m == 0.0006
    assert again.records == {"probe_points": [{"u": 0.5}]}
    store.save(again)
    assert len(list(tmp_path.glob("calib/setup.*.bak.json"))) == 1  # the version it replaced is kept


def test_older_files_and_unknown_keys(tmp_path):
    path = tmp_path / "setup.json"
    path.write_text(json.dumps({"arm": {"channel": "can_l_follower"}, "park_q": None, "note": "from a later version"}))
    setup = SetupStore(path).load()
    assert setup.arm.channel == "can_l_follower" and setup.arm.firmware == "auto"  # missing arm keys get defaults
    assert setup.to_json()["note"] == "from a later version"  # unknown keys survive a load and save


def test_what_is_missing_is_said():
    setup = Setup()
    with pytest.raises(SetupError, match="--channel"):
        setup.require("arm.channel")
    with pytest.raises(SetupError, match="--camera"):
        setup.require("camera")
    with pytest.raises(SetupError, match=r"screen_map.*seed_q.*teach"):
        setup.require("screen_map", "park_q", "seed_q", hint="先运行 teach 完成示教标定。")
    setup.limits["pres_m"] = 0.001  # a misspelt limit must not be ignored silently
    with pytest.raises(SetupError, match="pres_m"):
        setup.motion_limits()


def test_planning_limits_follow_the_joint_offsets():
    from tablet_tap.config import Setup
    from tablet_tap.core.kinematics import JOINT_LIMITS_RAD

    top = np.radians(70.0)
    setup = Setup()
    assert np.allclose(setup.pen_arm().limits, JOINT_LIMITS_RAD)
    setup.arm.joint5_limit_deg = 70.0  # the firmware's limit, on the arm's readings
    assert np.allclose(setup.pen_arm().limits[4], [-top, top])
    setup.arm.joint_offsets = [0.0, 0.0, 0.0, 0.0, 0.044, 0.0]
    assert np.allclose(setup.pen_arm().limits[4], [-top - 0.044, top - 0.044])
    setup.arm.joint_signs[4] = -1.0
    assert np.allclose(setup.pen_arm().limits[4], [-top + 0.044, top + 0.044])
    assert Setup.from_json(setup.to_json()).arm.joint5_limit_deg == 70.0


def test_a_save_without_a_dated_copy(tmp_path):
    from tablet_tap.config import Setup
    from tablet_tap.config import SetupStore

    store = SetupStore(tmp_path / "setup.json")
    store.save(Setup())
    store.save(Setup(), backup=False)
    assert list(tmp_path.glob("*.bak.json")) == []
    store.save(Setup())
    assert len(list(tmp_path.glob("*.bak.json"))) == 1
