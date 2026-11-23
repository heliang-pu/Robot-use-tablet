"""The command line end to end with no hardware: a software arm, a virtual tablet that reports the pen's touches to
the real calibration server, a camera frame from a file and a stub of the model server."""

import http.server
import json
import threading

import cv2
from helpers import ARM
from helpers import PARK
from helpers import SEED
from helpers import true_screen
import numpy as np
import pytest
from werkzeug.serving import make_server

from tablet_tap import app as cli_context
from tablet_tap import config
from tablet_tap.calibration import page_server
from tablet_tap.calibration import teach
from tablet_tap.calibration.page_client import CalibClient
from tablet_tap.cli import main as cli
from tablet_tap.core.motion import MotionLimits
from tablet_tap.core.motion import TapPlanner
from tablet_tap.core.screen_map import ScreenMap
from tablet_tap.robot.arm import FakeArm
from tablet_tap.vision import layout as screen_layout
from tablet_tap.vision import rectifier as screen_rectifier

ASPECT = 2560 / 1600


class VirtualTablet(FakeArm):
    """A software arm whose pen presses a real (hidden) screen: every crossing of the glass is a pointerdown."""

    def __init__(self, q, screen: ScreenMap, state: page_server.CalibState, trigger_mm: float = 0.3) -> None:
        super().__init__(q)
        self.screen, self.state, self.trigger = screen, state, trigger_mm * 1e-3
        self.down = False
        self.downs: list[tuple[float, float]] = []

    def command(self, q) -> None:
        super().command(q)
        u, v, h = self.screen.uv_of(ARM.fk(self.q)[:3, 3])
        pressed = h <= -self.trigger and 0 <= u <= 1 and 0 <= v <= 1  # a rubber tip needs a little pressure
        if pressed and not self.down:
            self.state.add_touch({"type": "pointerdown", "u": u, "v": v, "t_ms": 0})
            self.downs.append((u, v))
        elif self.down and not pressed:
            self.state.add_touch({"type": "pointerup", "u": u, "v": v, "t_ms": 0})
        self.down = pressed


@pytest.fixture
def bench(tmp_path, monkeypatch):
    state = page_server.CalibState()
    server = make_server("127.0.0.1", 0, page_server.create_app(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}"
    state.set_screen(
        {"inner_w": 2560, "inner_h": 1600, "screen_w": 2560, "screen_h": 1600, "dpr": 1, "fullscreen": True}
    )
    real = true_screen()
    # what a hand teach would give: 3 mm off sideways, 1.5 mm too high, turned by a degree
    turn = np.radians(1.0)
    rot = np.array([[np.cos(turn), -np.sin(turn), 0], [np.sin(turn), np.cos(turn), 0], [0, 0, 1]])
    rough = ScreenMap(real.origin + [0.003, -0.002, 0.0015], rot @ real.right, rot @ real.down, rot @ real.tool_x)
    setup = tmp_path / "setup.json"
    setup.write_text(
        json.dumps(
            {
                "park_q": PARK.tolist(),
                "seed_q": SEED.tolist(),
                "screen_map": rough.to_json(),
                "aspect": ASPECT,
                "camera": None,
            }
        )
    )
    arm = VirtualTablet(PARK, real, state)
    monkeypatch.setattr(cli_context, "open_arm", lambda *args, **kwargs: arm)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    base = ["--setup", str(setup), "--view", str(tmp_path / "view.json"), "--fake", "--calib_url", url]
    yield {"arm": arm, "real": real, "rough": rough, "setup": setup, "base": base, "state": state, "tmp": tmp_path}
    server.shutdown()


def test_probe_then_taptest(bench, capsys):
    arm, real = bench["arm"], bench["real"]
    assert cli.main([*bench["base"], "probe", "--nu", "4", "--nv", "3"]) == 0
    fitted = ScreenMap.from_json(json.loads(bench["setup"].read_text())["screen_map"])
    for u, v in ((0, 0), (1, 0), (0, 1), (1, 1), (0.5, 0.5)):
        before = np.linalg.norm(bench["rough"].point(u, v) - real.point(u, v)) * 1e3
        after = np.linalg.norm(fitted.point(u, v) - real.point(u, v)) * 1e3
        assert before > 2.0 and after < 0.6, (u, v, before, after)  # the trigger depth (0.3 mm) stays in the fit
    assert len(list(bench["tmp"].glob("setup.*.bak.json"))) == 1  # the old calibration is kept
    assert np.allclose(arm.q, PARK, atol=1e-6)
    out = capsys.readouterr().out
    assert "12 / 12 个点触发" in out and "可达性" in out

    arm.downs.clear()
    assert cli.main([*bench["base"], "taptest", "--nu", "3", "--nv", "3"]) == 0
    result = json.loads(next((bench["tmp"] / "runs").glob("taptest_*.json")).read_text())
    assert result["summary"]["n"] == 9 and result["summary"]["registered"] == 9
    assert result["summary"]["max_mm"] < 0.5 and result["summary"]["within_2mm"] == 9
    assert len(arm.downs) == 9
    assert bench["state"].snapshot()["mode"] == "markers"


def test_single_moves_and_refusals(bench, capsys):
    arm = bench["arm"]
    assert cli.main([*bench["base"], "status"]) == 0
    assert "笔尖位置" in capsys.readouterr().out
    assert cli.main([*bench["base"], "tap", "0.3", "0.4", "--park"]) == 0
    assert len(arm.downs) == 1 and np.allclose(arm.q, PARK, atol=1e-6)
    assert cli.main([*bench["base"], "swipe", "0.5", "0.8", "0.5", "0.3"]) == 0  # ends hovering
    assert cli.main([*bench["base"], "home"]) == 0
    assert cli.main([*bench["base"], "tap", "3.0", "0.5"]) == 2  # far outside the screen: refused, nothing sent
    assert "够不到" in capsys.readouterr().out
    arm.q = arm.q + 0.4  # somewhere unknown, e.g. after an emergency stop
    assert cli.main([*bench["base"], "tap", "0.5", "0.5"]) == 2
    assert "park" in capsys.readouterr().out
    assert cli.main([*bench["base"], "park"]) == 0
    assert np.allclose(arm.q, PARK, atol=1e-6)


class StubModel(http.server.BaseHTTPRequestHandler):
    """An OpenAI-compatible chat endpoint that answers with scripted GUI-Owl outputs."""

    outputs: list[str] = []
    requests: list[dict] = []

    def do_POST(self):  # noqa: N802 - http.server's name
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append(body)
        text = type(self).outputs.pop(0)
        payload = {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
        }
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def call(action: dict) -> str:
    return (
        "Action: 下一步。\n<tool_call>\n" + json.dumps({"name": "mobile_use", "arguments": action}) + "\n</tool_call>"
    )


def test_run_chain(bench, monkeypatch, tmp_path):
    """Camera frame -> rectified screenshot -> model -> tap by the arm at the point the model chose."""
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("http_proxy", raising=False)
    layout = screen_layout.default_layout(ASPECT)
    screen = screen_layout.render_layout(layout, 1280)
    corners = np.float32([[0, 0], [1280, 0], [1280, 800], [0, 800]])
    seen = np.float32([[260, 170], [1050, 140], [1120, 610], [200, 650]])  # the tablet seen obliquely
    frame = cv2.warpPerspective(
        screen, cv2.getPerspectiveTransform(corners, seen), (1280, 720), borderValue=(120, 120, 120)
    )
    cv2.imwrite(str(tmp_path / "frame.png"), frame)
    screen_rectifier.ScreenRectifier.from_marker_image(frame, layout, out_width=1792).save(tmp_path / "view.json")

    StubModel.outputs = [
        call({"action": "click", "coordinate": [250, 700]}),
        call({"action": "system_button", "button": "Home"}),
        call({"action": "terminate", "status": "success"}),
    ]
    StubModel.requests = []
    server = http.server.HTTPServer(("127.0.0.1", 0), StubModel)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    setup = json.loads(bench["setup"].read_text())
    setup["screen_map"] = bench["real"].to_json()
    bench["setup"].write_text(json.dumps(setup))
    code = cli.main(
        [
            *bench["base"],
            "--camera",
            f"file:{tmp_path / 'frame.png'}",
            "run",
            "--instruction",
            "打开音乐",
            "--vlm_url",
            f"http://127.0.0.1:{server.server_port}/v1",
            "--settle_s",
            "0",
        ]
    )
    server.shutdown()
    arm = bench["arm"]
    assert code == 0 and len(StubModel.requests) == 3
    assert bench["real"].error_mm((0.25, 0.70), arm.downs[0]) < 0.3  # the click
    assert arm.downs[1][1] > 0.98  # the home gesture starts at the bottom edge
    assert np.allclose(arm.q, PARK, atol=1e-6)  # parked at the end
    # the model was shown the rectified screen: the first request carries a 1792 x 1120 image
    content = StubModel.requests[0]["messages"][-1]["content"]
    assert any(part["type"] == "image_url" for part in content)
    shots = sorted(p for p in (tmp_path / "runs").rglob("step_*.png") if "anno" not in p.name)
    assert len(shots) == 3 and cv2.imread(str(shots[0])).shape[:2] == (1120, 1792)
    assert screen_rectifier.detect_markers(cv2.imread(str(shots[0]))).keys() == layout.markers.keys()


def test_guarded_descent_records_the_contact(bench):
    """The teach tool's z key: down along the pen until the page reports a touch, record it with the contact's joints."""
    arm, real = bench["arm"], bench["real"]
    limits = MotionLimits()
    client = CalibClient(bench["base"][-1])
    above = ARM.solve(TapPlanner(ARM, real, limits).tip_pose(0.3, 0.6, 0.012), SEED)  # 12 mm over the glass
    arm.q = above.copy()
    seq = client.last_down_seq()
    q_contact = teach.guarded_descent(
        arm, ARM, limits, lambda: client.last_down_seq() > seq, lambda: False, realtime=False
    )
    assert q_contact is not None
    u, v, h = real.uv_of(ARM.fk(q_contact)[:3, 3])
    assert real.error_mm((0.3, 0.6), (u, v)) < 0.2 and -0.6e-3 < h < 0  # stopped right at the trigger depth
    assert abs(real.uv_of(ARM.fk(arm.q)[:3, 3])[2] - (h + 0.015)) < 3e-4  # lifted 15 mm from the contact
    touches: list[teach.Touch] = []
    assert teach.record_touches(client, seq, q_contact, touches) > seq and len(touches) == 1
    assert np.allclose(touches[0].q, q_contact) and abs(touches[0].u - 0.3) < 1e-3
    # nothing under the pen: goes down 30 mm, comes back to where it started
    arm.q = ARM.solve(TapPlanner(ARM, real, limits).tip_pose(0.5, 0.5, 0.06), SEED)
    start = arm.q.copy()
    assert teach.guarded_descent(arm, ARM, limits, lambda: False, lambda: False, realtime=False) is None
    assert np.allclose(arm.q, start, atol=2e-4)


def test_grid_points_out_of_reach_are_skipped_not_fatal(bench, capsys):
    """A tablet lying too far away: the far corners cannot be reached. The probe and the tap test go on without them."""
    far = true_screen(centre=(0.42, 0.0, 0.03))
    setup = json.loads(bench["setup"].read_text())
    setup["screen_map"] = far.to_json()
    bench["setup"].write_text(json.dumps(setup))
    bench["arm"].screen = far
    assert cli.main([*bench["base"], "taptest", "--nu", "3", "--nv", "3", "--margin", "0.02"]) == 0
    out = capsys.readouterr().out
    result = json.loads(next((bench["tmp"] / "runs").glob("taptest_*.json")).read_text())
    skipped = result["summary"]["unreachable"]
    assert 0 < skipped < 9 and "够不到，跳过" in out and f"另有 {skipped} 个点" in out
    assert result["summary"]["n"] == 9 - skipped == result["summary"]["registered"]
    assert np.allclose(bench["arm"].q, PARK, atol=1e-6)  # parked at the end, not left hovering
    assert cli.main([*bench["base"], "probe", "--nu", "3", "--nv", "3", "--margin", "0.02"]) == 0
    assert "够不到，跳过" in capsys.readouterr().out
