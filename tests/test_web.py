"""The calibration console through its HTTP interface, on the software bench (no hardware)."""

import json
import time

import numpy as np
import pytest

from tablet_tap import config
from tablet_tap.calibration import page_server
from tablet_tap.core.motion import TapPlanner
from tablet_tap.web import console as web
from tablet_tap.web import sim


@pytest.fixture
def bench(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    calib = page_server.CalibState()
    con = web.Console(
        config.SetupStore(tmp_path / "setup.json"), tmp_path / "view.json", calib, fake=True, realtime=False
    )
    client = web.create_app(con).test_client()
    yield con, client, tmp_path
    con.close()


def wait_idle(con, timeout=30.0):
    deadline = time.monotonic() + timeout
    while con.busy is not None:
        assert time.monotonic() < deadline, f"still busy with {con.busy}"
        time.sleep(0.01)


def cmd(con, client, expect=202, **body):
    response = client.post("/api/cmd", json=body)
    assert response.status_code == expect, response.get_json()
    wait_idle(con)
    return response


def log_text(client):
    return "\n".join(line["text"] for line in client.get("/api/state").get_json()["log"])


def hover_over(con, u, v, height=0.012):
    """Put the software arm right above a screen point, as the operator would by jogging."""
    arm = con.arm._arm
    planner = TapPlanner(con.setup.pen_arm(), arm.screen, con.setup.motion_limits())
    seed = np.array([np.arctan2(0.05, 0.27), 1.5, -1.1, 0.0, 1.0, 0.0])
    arm.q = con.setup.pen_arm().solve(planner.tip_pose(u, v, height), seed)


def test_pages_and_refusals(bench):
    con, client, _ = bench
    assert client.get("/console").status_code == 200 and "标定控制台" in client.get("/console/full").get_data(as_text=True)
    assert client.get("/calib").status_code == 200  # the tablet's page is served by the same app
    state = client.get("/api/state").get_json()
    assert state["fake"] and not state["arm"]["connected"] and state["busy"] is None
    assert client.post("/api/cmd", data="cmd=park", content_type="application/x-www-form-urlencoded").status_code == 415
    assert client.post("/api/cmd", json={"cmd": "nonsense"}).status_code == 400
    assert client.post("/api/cmd", json={"cmd": "key", "key": "!"}).status_code == 400
    assert client.post("/api/cmd", json={"cmd": "probe", "nu": 99}).status_code == 400
    assert client.get("/api/camera.jpg").status_code == 503
    cmd(con, client, cmd="key", key="w", step="3")  # accepted, then refused by the job: no arm yet
    assert "还没有连接机械臂" in log_text(client)


def test_teach_fit_view_probe_taptest(bench):
    con, client, tmp = bench
    cmd(con, client, cmd="connect", channel="", mode="command")
    state = client.get("/api/state").get_json()
    assert state["arm"]["connected"] and state["arm"]["mode"] == "fake" and len(state["arm"]["q"]) == 6
    assert state["page"]["screen"]["covers_display"]

    # jog: 5 mm forward moves the pen tip 5 mm in x, nothing else
    before = np.array(state["arm"]["tip_mm"])
    cmd(con, client, cmd="key", key="w", step="3")
    after = np.array(client.get("/api/state").get_json()["arm"]["tip_mm"])
    assert np.allclose(after - before, [5.0, 0.0, 0.0], atol=0.15)
    cmd(con, client, cmd="key", key="p")
    assert client.get("/api/state").get_json()["setup"]["park_set"]
    park = np.array(con.setup.park_q)

    # five guarded descents: the corners and the middle
    for u, v in ((0.15, 0.15), (0.85, 0.15), (0.85, 0.85), (0.15, 0.85), (0.5, 0.5)):
        hover_over(con, u, v)
        cmd(con, client, cmd="key", key="z")
    touches = client.get("/api/state").get_json()["touches"]
    assert len(touches) == 5 and abs(touches[0]["u"] - 0.15) < 0.003 and abs(touches[4]["v"] - 0.5) < 0.003
    cmd(con, client, cmd="key", key="x")
    assert len(client.get("/api/state").get_json()["touches"]) == 4
    hover_over(con, 0.5, 0.5)
    cmd(con, client, cmd="key", key="z")

    cmd(con, client, cmd="fit")
    fitted = client.get("/api/state").get_json()["setup"]["screen_map"]
    assert abs(fitted["width_mm"] - 250) < 1.0 and abs(fitted["height_mm"] - 160) < 1.0 and fitted["n_points"] == 5
    assert "可达性" in log_text(client)

    # the arm is still over the screen centre: back to the park pose (known place, so the checked path)
    cmd(con, client, cmd="park")
    assert np.allclose(con.arm.read(), park, atol=1e-6)
    assert client.get("/api/state").get_json()["place"]["where"] == "park"

    # camera: open, calibrate from the page's markers, pictures
    cmd(con, client, cmd="camera_open", spec="")
    deadline = time.monotonic() + 5
    while client.get("/api/camera.jpg").status_code != 200:
        assert time.monotonic() < deadline
        time.sleep(0.05)
    cmd(con, client, cmd="view")
    state = client.get("/api/state").get_json()
    assert state["camera"]["calibrated"] and state["results"]["view"]["reprojection_px"] < 1.0
    assert (tmp / "view.json").exists() and (tmp / "view.tags.json").exists()
    assert (
        client.get("/api/screen.jpg").data[:2] == b"\xff\xd8" and client.get("/api/camera.jpg").data[:2] == b"\xff\xd8"
    )

    # automatic probe refits; then the tap test measures what the screen itself reports
    real = con.arm._arm.screen
    cmd(con, client, cmd="probe", nu=4, nv=3)
    state = client.get("/api/state").get_json()
    assert state["results"]["probe"] == {"n": 12, "registered": 12}
    for u, v in ((0, 0), (1, 1), (0.5, 0.5)):
        assert np.linalg.norm(con.setup.screen_map.point(u, v) - real.point(u, v)) < 0.6e-3
    cmd(con, client, cmd="taptest", nu=3, nv=3, repeat=1)
    summary = client.get("/api/state").get_json()["results"]["taptest"]["summary"]
    assert summary["n"] == 9 and summary["registered"] == 9 and summary["max_mm"] < 0.5
    saved = json.loads(next((tmp / "runs").glob("taptest_*.json")).read_text())
    assert saved["summary"]["within_2mm"] == 9
    assert client.get("/api/state").get_json()["page"]["mode"] == "markers"

    # single moves; a tap reports where the screen saw it; touches used by procedures are not teaching touches
    cmd(con, client, cmd="tap", u=0.3, v=0.7, park=True)
    assert "偏差" in log_text(client) and np.allclose(con.arm.read(), park, atol=1e-6)
    cmd(con, client, cmd="home")
    cmd(con, client, cmd="hover")
    assert client.get("/api/state").get_json()["place"]["where"] == "hover"
    assert len(client.get("/api/state").get_json()["touches"]) == 5
    assert client.post("/api/cmd", json={"cmd": "tap", "u": 1.5, "v": 0.5}).status_code == 202
    wait_idle(con)
    assert "应在 0 到 1 之间" in log_text(client)

    # the setup on disk is what a later session (or the command line) starts from
    stored = config.SetupStore(tmp / "setup.json").load()
    assert stored.screen_map is not None and stored.park_q is not None and stored.seed_q is not None
    assert len(stored.records["probe_points"]) == 12


def test_busy_and_stop(bench):
    con, client, _ = bench
    cmd(con, client, cmd="connect", channel="", mode="command")
    cmd(con, client, cmd="key", key="p")
    for u, v in ((0.15, 0.15), (0.85, 0.15), (0.85, 0.85), (0.15, 0.85)):
        hover_over(con, u, v)
        cmd(con, client, cmd="key", key="z")
    cmd(con, client, cmd="fit")
    cmd(con, client, cmd="park")
    con.realtime = True  # a job that takes real time, so that it can be interrupted
    assert client.post("/api/cmd", json={"cmd": "taptest", "nu": 5, "nv": 4}).status_code == 202
    time.sleep(0.3)
    refused = client.post("/api/cmd", json={"cmd": "key", "key": "w"})
    assert refused.status_code == 409 and "不排队" in refused.get_json()["error"]
    sent = len(con.arm._arm.commands)
    assert client.post("/api/stop").status_code == 200
    wait_idle(con, timeout=5)
    assert "已停止" in log_text(client)
    time.sleep(0.2)
    assert len(con.arm._arm.commands) - sent < 20  # nothing more than the step in flight was sent after the stop
    assert client.get("/api/state").get_json()["place"]["where"] in ("unknown", "hover")
    con.realtime = False
    cmd(con, client, cmd="park")  # from wherever it stopped: the slow joint move
    assert client.get("/api/state").get_json()["place"]["where"] == "park"


def test_read_only_and_token(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    con = web.Console(config.SetupStore(tmp_path / "setup.json"), tmp_path / "view.json", page_server.CalibState(), fake=True,
                      realtime=False)  # fmt: skip
    try:
        client = web.create_app(con, token="s3cret").test_client()
        assert client.get("/api/state").status_code == 401 and client.get("/console").status_code == 401
        assert (
            client.post("/api/stop").status_code == 401
            and client.post("/api/cmd", json={"cmd": "fit"}).status_code == 401
        )
        assert client.get("/api/state?token=s3cret").status_code == 200
        assert client.get("/calib").status_code == 200  # the tablet's page needs no token
        # a real arm asked to move needs the operator's tick
        real = web.Console(
            config.SetupStore(tmp_path / "s2.json"), tmp_path / "v2.json", page_server.CalibState(), fake=False
        )
        try:
            with pytest.raises(ValueError, match="急停"):
                web._job(real, "connect", {"channel": "can_r_follower", "mode": "command"})
            with pytest.raises(ValueError, match="请选择"):
                web._job(real, "connect", {"channel": "can0", "mode": "read_only"})
        finally:
            real.close()
    finally:
        con.close()


def test_tag_drift():
    square = np.array([[0, 0], [50, 0], [50, 50], [0, 50]], float)
    reference = [{"id": 9, "corners": square.tolist()}, {"id": 9, "corners": (square + [400, 300]).tolist()},
                 {"id": 8, "corners": (square + [400, 0]).tolist()}]  # fmt: skip
    now = [{"id": 9, "corners": square + [401.5, 300]}, {"id": 9, "corners": square + [0.5, 0]}]
    drift = web.tag_drift(reference, now)
    assert drift["n"] == 3 and drift["missing"] == 1 and abs(drift["max_px"] - 1.5) < 1e-9
    assert web.tag_drift(reference, [])["max_px"] is None
    assert sim.SyntheticCamera(page_server.CalibState()).read().shape == (720, 1280, 3)


def taught(con, client):
    """Connect, park pose, four descents, fit, park: the state after the teaching step."""
    cmd(con, client, cmd="connect", channel="", mode="command")
    cmd(con, client, cmd="key", key="p")
    for u, v in ((0.15, 0.15), (0.85, 0.15), (0.85, 0.85), (0.15, 0.85)):
        hover_over(con, u, v)
        cmd(con, client, cmd="key", key="z")
    cmd(con, client, cmd="fit")
    cmd(con, client, cmd="park")


def test_plane_estimated_by_the_camera_is_probed_before_it_is_tapped(bench):
    con, client, tmp = bench
    taught(con, client)
    assert con.setup.records["screen_map_source"] == "touch" and con.setup.screen_size_m is not None
    real = con.arm._arm.screen
    # what the hand-eye calibration's "place" leaves: a plane a few millimetres off, with no residual
    shifted = dataclass_replace(real, origin=real.origin + [0.003, -0.002, 0.002])
    con.setup.screen_map, con.setup.records["screen_map_source"] = shifted, "camera"
    state = client.get("/api/state").get_json()  # NaN must not reach the page as invalid JSON
    assert state["setup"]["screen_map"]["source"] == "camera" and state["setup"]["screen_map"]["residual_mm"] is None
    sent = len(con.arm._arm.commands)
    cmd(con, client, cmd="tap", u=0.5, v=0.5)
    assert "先做一次“自动探触并重新拟合”" in log_text(client) and len(con.arm._arm.commands) == sent
    cmd(con, client, cmd="taptest", nu=2, nv=2)
    assert len(con.arm._arm.commands) == sent  # nothing moved
    cmd(con, client, cmd="probe", nu=4, nv=3, depth_mm=6)
    text = log_text(client)
    assert "悬停高度提到 30 mm，探触深度 3 mm" in text and con.setup.records["screen_map_source"] == "touch"
    probe = client.get("/api/state").get_json()["results"]["probe"]
    assert probe["n"] >= 10 and probe["registered"] == probe["n"]  # 30 mm up, a corner may be out of reach: skipped
    for u, v in ((0, 0), (1, 1), (0.5, 0.5)):
        assert np.linalg.norm(con.setup.screen_map.point(u, v) - real.point(u, v)) < 0.6e-3
    assert "hover_m" not in con.setup.limits  # the raised hover was for this probe only
    cmd(con, client, cmd="tap", u=0.5, v=0.5, park=True)
    assert "偏差" in log_text(client)


def test_points_out_of_reach_are_skipped(bench):
    con, client, _ = bench
    taught(con, client)
    real = con.arm._arm.screen
    # the tablet pushed away and aside: its far corners are beyond the pen's reach, the rest is not
    far = dataclass_replace(real, origin=real.origin + [0.05, 0.07, 0.0])
    con.arm._arm.screen = far
    con.setup.screen_map = far
    cmd(con, client, cmd="probe", nu=5, nv=4)
    assert "跳过" in log_text(client)
    probe = client.get("/api/state").get_json()["results"]["probe"]
    assert 4 <= probe["n"] < 20 and probe["registered"] == probe["n"]  # the reachable ones were all probed
    assert client.get("/api/state").get_json()["place"]["where"] == "park"
    # so far away that not even its centre is within reach: said in plain words, nothing moves
    gone = dataclass_replace(real, origin=real.origin + [0.25, 0.0, 0.0])
    con.setup.screen_map = gone
    sent = len(con.arm._arm.commands)
    cmd(con, client, cmd="probe", nu=5, nv=4)
    assert "把平板往拿笔的机械臂挪近" in log_text(client) and len(con.arm._arm.commands) == sent


def dataclass_replace(screen, **changes):
    import dataclasses

    return dataclasses.replace(screen, **changes)


def test_hardware_is_left_alone_while_it_is_locked_or_in_use(tmp_path, monkeypatch):
    """A console on the real bench refuses to open the arm or a camera while the lock is set or run_real is up."""
    from tablet_tap.errors import SetupError
    from tablet_tap.robot import piper

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    opened = []
    monkeypatch.setattr(web, "open_arm", lambda *a, **k: opened.append("arm"))
    monkeypatch.setattr(
        web, "open_camera", lambda spec: opened.append(spec) or sim.SyntheticCamera(page_server.CalibState())
    )
    con = web.Console(
        config.SetupStore(tmp_path / "setup.json"), tmp_path / "view.json", page_server.CalibState(), fake=False
    )
    try:
        for mode in ("read_only", "command"):  # the lock every test runs under
            with pytest.raises(SetupError, match="TABLET_TAP_NO_HARDWARE"):
                con.connect("can_r_follower", mode)
        with pytest.raises(SetupError, match="TABLET_TAP_NO_HARDWARE"):
            con.open_camera("realsense:262422077281:1920x1080")
        monkeypatch.delenv("TABLET_TAP_NO_HARDWARE")
        monkeypatch.setattr(piper, "other_arm_programs", lambda: ["123 python run_real.py drawer_open_place"])
        with pytest.raises(SetupError, match="先停掉它"):
            con.connect("can_r_follower", "read_only")
        with pytest.raises(SetupError, match="先停掉它"):
            con.open_camera("realsense:262422077281:1920x1080")
        assert opened == []  # nothing was opened in any of these
        con.open_camera(f"file:{tmp_path}")  # pictures on disk are not the bench
        assert opened == [f"file:{tmp_path}"]
        con.close_camera()
        monkeypatch.setattr(piper, "other_arm_programs", lambda: [])
        con.open_camera("realsense:262422077281:1920x1080")  # free bench, no lock: allowed
        assert opened[-1] == "realsense:262422077281:1920x1080"
    finally:
        con.close()


def test_click_goes_down_slowly_until_the_contacts_fix_the_plane_then_taps(bench):
    con, client, _ = bench
    taught(con, client)
    real = con.arm._arm.screen
    shifted = dataclass_replace(real, origin=real.origin + [0.003, -0.002, 0.002])
    con.setup.screen_map, con.setup.records["screen_map_source"] = shifted, "camera"
    cmd(con, client, cmd="point", on="camera", x=0.5, y=0.5)
    assert "先打开相机" in log_text(client)
    for x, y in ((0.3, 0.4), (0.7, 0.4), (0.7, 0.8), (0.3, 0.8)):
        cmd(con, client, cmd="point", on="screen", x=x, y=y)
    accuracy = client.get("/api/state").get_json()["accuracy"]
    assert accuracy["registered"] == 4 and accuracy["affine"] and con.setup.records["screen_map_source"] == "touch"
    assert accuracy["last"]["error_mm"] < 0.5 and accuracy["fast_n"] == 0  # already corrected by the earlier contacts
    assert np.linalg.norm(con.setup.screen_map.point(0.5, 0.6) - real.point(0.5, 0.6)) < 1e-3
    cmd(con, client, cmd="point", on="screen", x=0.5, y=0.6)
    accuracy = client.get("/api/state").get_json()["accuracy"]
    assert accuracy["fast_n"] == 1 and accuracy["fast_max"] < 0.7
    cmd(con, client, cmd="forget_contacts")
    assert con.setup.records["screen_map_source"] == "camera"
    assert client.post("/api/cmd", json={"cmd": "point", "on": "screen", "x": 1.2, "y": 0.5}).status_code == 400


def test_a_click_is_refused_until_the_camera_has_seen_the_tablet(bench):
    con, client, _ = bench
    taught(con, client)
    con.setup.camera_pose = np.eye(4).tolist()
    sent = len(con.arm._arm.commands)
    cmd(con, client, cmd="point", on="screen", x=0.5, y=0.5)
    assert "还没认出平板" in log_text(client) and len(con.arm._arm.commands) == sent


class OffPen:
    """The real arm's geometry: the model's with small joint zero errors."""

    def __init__(self, pen, degrees):
        self.pen, self.off = pen, np.radians(degrees)

    def fk(self, q):
        return self.pen.fk(np.asarray(q) + self.off)


@pytest.mark.parametrize("slant", [0.0, 15.0])
def test_one_click_calibration_reaches_a_millimetre_on_a_slanted_tablet(bench, slant):
    con, client, tmp = bench
    taught(con, client)
    arm = con.arm._arm
    real = sim.flat_screen(centre=(0.28, 0.02, 0.0), tilt_deg=slant)
    arm.screen, arm.pen = real, OffPen(con.setup.pen_arm(), [0.3, -0.3, 0.4, 0.4, -0.5, 0.0])
    # what the camera says of it: a few millimetres and half a degree off
    shifted = dataclass_replace(real, origin=real.origin + [0.003, -0.002, 0.0015], down=real.down + [0, 0, 0.0012])
    con.live = web.LiveScreen.from_map(shifted)
    con.correction = None
    con._apply_live()
    assert con._reach == ""
    cmd(con, client, cmd="calibrate_taps")
    result = con.results.get("calibration")
    assert result is not None, log_text(client)
    assert result["n"] == web.TEST_TAPS and result["mean_mm"] < 0.4 and result["max_mm"] < 1.0, result
    assert result["before_mean_mm"] > result["mean_mm"]
    saved = json.loads(next((tmp / "runs").glob("tapcal_*.json")).read_text())
    assert len(saved["tested"]) == web.TEST_TAPS
    cmd(con, client, cmd="point", on="screen", x=0.35, y=0.7)
    assert client.get("/api/state").get_json()["accuracy"]["last"]["error_mm"] < 0.8


def test_the_gripper_switch_forgets_what_was_learned_with_the_pen(bench):
    con, client, _ = bench
    taught(con, client)
    arm = con.arm._arm
    widths = []
    arm.set_grip_width = lambda w: (widths.append(w), setattr(arm, "gripper_width_m", max(w, 0.0212)))
    con.correction = web.TapCorrection((0.25, 0.16), [(0.5, 0.5, 1.0, 1.0)] * 3)
    cmd(con, client, cmd="grip", close=False)
    cmd(con, client, cmd="grip", close=True)
    assert widths == [web.GRIP_OPEN_M, 0.0] and con.correction.samples == []
    assert "夹住 21.2 mm" in log_text(client) and client.get("/api/state").get_json()["gripper_mm"] == 21.2


def test_the_console_follows_a_tablet_put_on_a_slanted_stand_and_then_its_tags(bench):
    from test_live_screen import CAMERA
    from test_live_screen import MODEL

    from tablet_tap.core.screen_map import ScreenMap
    from tablet_tap.vision.layout import default_layout

    con, client, _ = bench
    con.calib.set_screen({"inner_w": 1600, "inner_h": 1000, "screen_w": 1600, "screen_h": 1000, "fullscreen": True})
    layout, frame = default_layout(1.6), np.zeros((1080, 1920, 3), np.uint8)
    con.setup.camera_model, con.setup.camera_pose, con.setup.screen_size_m = MODEL, CAMERA.tolist(), [0.24, 0.15]
    quads = {40: ((-0.2, 0.3), (-0.08, 0.3), (-0.08, 0.5), (-0.2, 0.5)), 41: ((1.08, 0.3), (1.2, 0.3), (1.2, 0.5), (1.08, 0.5))}

    def seen(centre, slant, page=True):
        a = np.radians(slant)
        right, down = np.array([0.0, -1.0, 0.0]), -np.array([np.cos(a), 0.0, np.sin(a)])
        screen = ScreenMap(np.array(centre) - 0.12 * right - 0.075 * down, 0.24 * right, 0.15 * down, -down)
        boxes = {k: ((u0, v0), (u1, v0), (u1, v1), (u0, v1)) for k, (u0, v0, u1, v1) in layout.markers.items()} if page else {}
        inverse = np.linalg.inv(CAMERA)
        return screen, {k: MODEL.project(np.array([screen.point(u, v) for u, v in q]) @ inverse[:3, :3].T + inverse[:3, 3])
                        for k, q in (boxes | quads).items()}  # fmt: skip

    flat, found = seen((0.27, 0.0, 0.0), 0.0)
    for _ in range(3):
        con._follow(frame, found)
    assert con._located and np.linalg.norm(con.setup.screen_map.point(0.5, 0.5) - flat.point(0.5, 0.5)) < 1e-3
    con.live.add((0.5, 0.5), flat.point(0.5, 0.5))
    slanted, found = seen((0.28, 0.02, 0.05), 20.0)
    for _ in range(web.PLANE_LOOKS + 4):
        con._follow(frame, found)
    here = con.setup.screen_map
    assert "倾斜 20.0°" in log_text(client) and con.live.contacts == []  # learned on another plane
    assert np.linalg.norm(here.point(0.5, 0.5) - slanted.point(0.5, 0.5)) < 1.5e-3
    assert np.degrees(np.arccos(np.clip(here.normal @ slanted.normal, -1, 1))) < 0.3
    assert sorted(con.live.tags) == [40, 41]
    moved, found = seen((0.28, 0.0, 0.05), 20.0, page=False)  # another app is in front: only the printed tags are left
    for _ in range(3):
        con._follow(frame, found)
    assert np.linalg.norm(con.setup.screen_map.point(0.5, 0.5) - moved.point(0.5, 0.5)) < 2e-3
    assert con._tablet["state"] == "tracked" and con._tablet["seen"] == 0 and con._tablet["tags"] == 2


def test_a_job_that_touches_is_stopped_when_the_tablet_is_moved(bench):
    con, client, _ = bench
    taught(con, client)
    con.live = web.LiveScreen.from_map(con.setup.screen_map)
    con._located = True
    here = con.live.pose.copy()
    away = here.copy()
    away[:3, 3] += 0.008 * here[:3, 0]  # 8 mm along the screen
    for busy, seen, stops in (("点击位置", here, False), ("连接", away, False), ("点击位置", away, True)):
        con.stop.clear()
        con.busy, con._off_looks = busy, 0
        for _ in range(web.MOVED_LOOKS):
            assert not con.stop.is_set()  # one look alone never stops a job
            con._watch_job(con.live, seen)
        assert con.stop.is_set() is stops
    con.busy = None
    assert "平板被挪动了（8 mm" in log_text(client)


def test_https_certificate_is_made_once(tmp_path):
    import shutil

    if shutil.which("openssl") is None:
        pytest.skip("no openssl")
    pair = web.ensure_certificate(tmp_path, ["192.168.1.138"])
    assert pair is not None and all(p.exists() for p in pair) and (pair[1].stat().st_mode & 0o077) == 0
    stamp = pair[0].stat().st_mtime_ns
    assert web.ensure_certificate(tmp_path, ["192.168.1.138"]) == pair and pair[0].stat().st_mtime_ns == stamp
