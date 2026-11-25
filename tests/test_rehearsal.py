"""The whole chain on the bench in software (``web.rehearsal``), and what the first rehearsal found."""

import dataclasses
import json

import numpy as np
import pytest

from tablet_tap import config
from tablet_tap.core.kinematics import PenArm
from tablet_tap.core.motion import MotionLimits
from tablet_tap.core.motion import TapPlanner
from tablet_tap.core.pinhole import CameraModel
from tablet_tap.web import rehearsal
from tablet_tap.web import sim

# the bench's top camera and park pose of 2026-11-23, rounded
CAMERA = [[-0.006, -0.9286, 0.3711, 0.024], [-0.9999, -0.0001, -0.0163, -0.235], [0.0151, -0.3711, -0.9285, 0.631], [0.0, 0.0, 0.0, 1.0]]  # fmt: skip
PARK = [0.7, 0.02, -0.13, -0.11, 0.16, 0.07]


def test_a_move_over_the_screen_turns_the_pen_gradually_when_the_screen_was_estimated_again():
    """One slow touch moves the estimate of the screen by millimetres, and with it the pen's orientation by a degree or
    two. The next move starts from the pose the pen has: taken at once, that turn was a jump of the wrist, which the
    speed check refused ("关节速度降不下来") and which ended the one-click calibration at its second point."""
    limits = MotionLimits(hover_m=0.03, max_joint_speed=0.6, travel_speed=0.05)
    pen, before = PenArm(), sim.flat_screen(tilt_deg=12.0)
    q = TapPlanner(pen, before, limits).hover_q(0.5, 0.5, np.array([0.2, 1.3, -0.9, 0.0, 1.0, 0.0]))
    turn = np.radians(1.7)  # the screen as the contact corrects it: 7 mm aside, the pen turned about its axis
    turned = np.cos(turn) * before.tool_x + np.sin(turn) * np.cross(before.normal, before.tool_x)
    after = dataclasses.replace(before, origin=before.origin + [0.0, 0.007, 0.002], tool_x=turned)
    planner = TapPlanner(pen, after, limits)
    segment = planner.to_point(q, 0.2, 0.3)
    steps = np.abs(np.diff(np.concatenate([q[None], segment.q]), axis=0))
    assert steps.max() * limits.hz <= limits.max_joint_speed + 1e-9
    end = pen.fk(segment.q[-1])
    assert np.allclose(end[:3, :3], planner.rot, atol=1e-3)
    assert np.linalg.norm(end[:3, 3] - after.point(0.2, 0.3, limits.hover_m)) < 5e-4


def test_the_whole_chain_from_connecting_to_carrying_out_instructions(tmp_path, monkeypatch):
    """Slanted stand, a camera calibration and an arm that are a little wrong, a stand-in for the model: connect, grip,
    find the tablet, calibrate, tap, leave the calibration page, get pushed, carry out instructions."""
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    setup = config.Setup(
        park_q=PARK, camera="realsense:0:1920x1080", camera_pose=CAMERA, screen_size_m=[0.2369, 0.1481],
        camera_model=CameraModel(1920, 1080, 1361.7, 1358.8, 975.8, 569.3), nav={"home": [0.5, 0.965], "back": [0.75, 0.965]},
        limits={"joint_move_speed": 0.25, "max_joint_speed": 0.6, "travel_speed": 0.05, "press_speed": 0.02},
    )  # fmt: skip
    setup.arm.tip_len = 0.085
    config.SetupStore(tmp_path / "bench.json").save(setup, backup=False)
    out = tmp_path / "rehearsal"
    code = rehearsal.main(["--setup", str(tmp_path / "bench.json"), "--out", str(out), "--llm", "scripted", "--quick",
                           "--slant", "12", "打开音乐，播放音乐", "回到桌面"])  # fmt: skip
    report = (out / "report.md").read_text()
    assert code == 0 and "全部通过" in report, report
    result = json.loads((out / "result.json").read_text())
    assert result["calibration"]["mean_mm"] < 0.8 and result["calibration"]["n"] == 12
    assert result["located"]["hidden"] < 0.02 and result["located"]["lean_deg"] == 0
    assert result["android"]["picture_error_mm"] < 1.0 and result["android"]["after_nudge_mm"] < 1.0
    first, second = result["runs"]
    assert first["status"] == "success" and first["reached"] and max(first["land_mm"]) < 2.0
    assert [e["hit"] for e in first["events"]] == ["music", "play"] and second["events"][0]["hit"] == "home"


@pytest.mark.parametrize("way", ["--wrist", "--top_moved"])
def test_the_chain_with_the_wrist_camera(tmp_path, monkeypatch, way):
    """The top camera places the tablet roughly; the wrist camera is calibrated on the flange from a set of looks, takes
    the look pose's picture, follows a small push of the tablet, and is what instructions are carried out with. The
    tablet lies 32 cm from the base: nearer, no pose of the arm shows the whole screen. With the top camera moved since
    its calibration (7 cm, 5 degrees), the wrist camera finds the tablet itself, and its calibration says where the top
    camera now is."""
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    setup = config.Setup(
        park_q=PARK, camera="realsense:0:1920x1080", camera_pose=CAMERA, screen_size_m=[0.2369, 0.1481],
        camera_model=CameraModel(1920, 1080, 1361.7, 1358.8, 975.8, 569.3), nav={"home": [0.5, 0.965], "back": [0.75, 0.965]},
        limits={"joint_move_speed": 0.25, "max_joint_speed": 0.6, "travel_speed": 0.05, "press_speed": 0.02},
    )  # fmt: skip
    setup.arm.tip_len = 0.085
    config.SetupStore(tmp_path / "bench.json").save(setup, backup=False)
    out = tmp_path / "rehearsal"
    code = rehearsal.main(["--setup", str(tmp_path / "bench.json"), "--out", str(out), "--llm", "scripted", "--quick", way,
                           "--slant", "12", "--dx_mm", "75", "打开音乐，播放音乐"])  # fmt: skip
    report = (out / "report.md").read_text()
    assert code == 0 and "全部通过" in report, report
    result = json.loads((out / "result.json").read_text())
    found = result["wrist_calibration"]
    assert found["views"] >= 12 and found["off_true_mm"] < 6.0 and found["off_true_deg"] < 1.0
    assert found["off_drawing_mm"] > 2.0  # the mount is not where the drawing has it, and the calibration says so
    assert abs(found["screen_size_mm"][0] - 236.9) < 0.05  # the display's size is not fitted
    view = result["wrist_views"][0]
    assert view["width_px"] > 900 and view["picture_error_mm"] < 0.5
    assert result["wrist_after_nudge_mm"] < 1.0  # the tablet was pushed 5 mm; the picture was not taken again
    run = result["runs"][0]
    assert run["status"] == "success" and run["reached"] and max(run["land_mm"]) < 2.0
    assert not (tmp_path / "calib" / "wrist.json").exists()  # the rehearsal keeps its files to itself
    assert (out / "calib" / "wrist.json").exists()
    saved = config.SetupStore(out / "calib" / "setup.json").load()
    if way == "--top_moved":
        assert result["wrist_find"]["worst_mm"] < 15 and result["top_recovered"]["off_true_mm"] < 10
        assert saved.records["camera_pose_source"].startswith("腕部相机手眼标定")
        assert np.allclose(saved.records["camera_pose_before_wrist"], CAMERA)
    else:  # the top camera was where its calibration has it: what the wrist says of it is offered, not taken
        assert np.allclose(saved.camera_pose, CAMERA) and "camera_pose_source" not in saved.records


@pytest.mark.parametrize("fullscreen", [True, False])
def test_the_tablet_is_followed_by_its_outline_when_there_are_no_markers(tmp_path, monkeypatch, fullscreen):
    """An app is in front and no tags are stuck on: the body's edges, learned while the calibration page was up, are
    followed, also when the page has reported that it left full screen; a screen full of light changes nothing."""
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    setup = config.Setup(
        park_q=PARK, camera="realsense:0:1920x1080", camera_pose=CAMERA, screen_size_m=[0.2369, 0.1481],
        camera_model=CameraModel(1920, 1080, 1361.7, 1358.8, 975.8, 569.3),
    )  # fmt: skip
    setup.arm.tip_len = 0.085
    config.SetupStore(tmp_path / "bench.json").save(setup, backup=False)
    seen = {}

    def run(self):
        self.connect()
        assert self.wait_for(lambda: self.con._located and "outline" in self.con.setup.records, 20)
        self.tablet.show("home")
        if not fullscreen:
            self.con.calib.set_screen({"inner_w": 1280, "inner_h": 700, "screen_w": 1280, "screen_h": 800, "fullscreen": False})
        self.wait_for(lambda: self.state()["tablet"].get("outline"), 8)
        seen["before"], seen["state"] = self.picture_error_mm(top=True), self.state()["tablet"]
        self.bench.move_tablet(0.012, -0.008, 2.0)
        seen["followed"] = self.wait_for(lambda: self.picture_error_mm(top=True) < 1.5, 10)
        seen["after"] = self.picture_error_mm(top=True)
        self.con.close()
        return 0

    monkeypatch.setattr(rehearsal.Rehearsal, "run", run)
    rehearsal.main(["--setup", str(tmp_path / "bench.json"), "--out", str(tmp_path / "r"), "--llm", "scripted", "--no_tags",
                    "--slant", "12", "--dx_mm", "75"])  # fmt: skip
    assert seen["state"]["outline"] and seen["state"]["seen"] == 0 and seen["before"] < 1.5, seen
    assert seen["followed"] and seen["after"] < 1.5, seen
