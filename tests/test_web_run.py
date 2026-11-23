"""The run page through its HTTP interface: the software bench, a scripted model, stub services. No hardware, no GPU."""

import http.server
import json
import threading
import time

import cv2
import numpy as np
import pytest
from test_web import cmd
from test_web import taught
from test_web import wait_idle

from tablet_tap import config
from tablet_tap.calibration import page_server
from tablet_tap.speech import voice
from tablet_tap.web import console as web
from tablet_tap.web import run_page
from tablet_tap.web.run_page import RunOptions
from tablet_tap.web.run_page import RunSession
from tablet_tap.web.run_page import ServiceProbe


def call(action: dict, thought: str = "点一下") -> str:
    return (
        f"Action: {thought}\n<tool_call>\n" + json.dumps({"name": "mobile_use", "arguments": action}) + "\n</tool_call>"
    )


class ScriptedModel:
    """Answers with the next output; keeps what it was asked; can be held back to look at a run in the middle."""

    def __init__(self, outputs):
        self.outputs, self.asked, self.gate = list(outputs), [], threading.Event()
        self.gate.set()

    def predict_mm(self, messages):
        self.asked.append(messages)
        assert self.gate.wait(20)
        return self.outputs.pop(0), messages, None


class Services(http.server.BaseHTTPRequestHandler):
    """GET /v1/models, GET /health, POST /asr, as the real services answer."""

    def do_GET(self):
        self._send({"data": [{"id": "gui-owl"}]} if self.path.endswith("/models") else {"ok": True})

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self._send({"text": "打开音乐。", "raw": "", "seconds_audio": 1.0, "seconds_infer": 0.1})

    def _send(self, data):
        payload = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def services():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Services)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}"
    yield RunOptions(vlm_url=f"{url}/v1", asr_url=url)
    server.shutdown()


@pytest.fixture
def bench(tmp_path, monkeypatch, services):
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("tablet_tap.agent.robot_device.time.sleep", lambda s: None)
    con = web.Console(
        config.SetupStore(tmp_path / "setup.json"),
        tmp_path / "view.json",
        page_server.CalibState(),
        fake=True,
        realtime=False,
    )
    model = ScriptedModel([])
    session = RunSession(con, services, llm_factory=lambda: model)
    app = page_server.create_app(con.calib)
    web.register(app, con)
    run_page.register(app, con, session=session)
    yield con, app.test_client(), model, session, tmp_path
    con.close()


def ready(con, client):
    """Connected, taught, the camera open and calibrated: an instruction can be carried out."""
    taught(con, client)
    cmd(con, client, cmd="camera_open", camera="模拟相机")
    cmd(con, client, cmd="view")


def start(client, expect=202, **body):
    response = client.post("/api/run/start", json={"settle_s": 0, **body})
    assert response.status_code == expect, response.get_json()
    return response.get_json()


def wait_for(check, timeout=20.0):
    deadline = time.monotonic() + timeout
    while not check():
        assert time.monotonic() < deadline
        time.sleep(0.02)


def test_the_page_says_what_is_missing_and_refuses_to_start(bench):
    con, client, model, _, _ = bench
    page = client.get("/run")
    assert page.status_code == 200 and "平板点按 · 运行" in page.get_data(as_text=True)
    missing = client.get("/api/run/state").get_json()["missing"]
    assert any("机械臂没有连接" in m for m in missing) and any("相机没有打开" in m for m in missing)
    assert "还不能执行" in start(client, 400, instruction="打开音乐")["error"]
    assert "空的" in start(client, 400, instruction="   ")["error"]
    assert "太长" in start(client, 400, instruction="长" * 1100)["error"]
    assert start(client, 400, instruction="打开音乐", max_steps=1000)
    assert client.post("/api/run/start", data="instruction=x").status_code == 400  # a form is not accepted
    assert model.asked == []


def test_an_instruction_is_carried_out_and_every_step_is_shown(bench):
    con, client, model, _, tmp = bench
    ready(con, client)
    assert client.get("/api/run/state").get_json()["missing"] == []
    model.outputs = [
        call({"action": "click", "coordinate": [250, 700]}, "打开音乐应用"),
        call({"action": "terminate", "status": "success"}),
    ]
    model.gate.clear()
    started = start(client, instruction="  打开音乐，\n播放音乐 ", max_steps=5)
    assert started["instruction"] == "打开音乐，\n播放音乐" and started["status"] == "running"
    wait_for(lambda: len(model.asked) == 1)
    assert client.get("/api/run/state").get_json()["busy"] == "执行指令"
    assert start(client, 409, instruction="下一首")["error"].startswith("正在执行")
    assert client.post("/api/cmd", json={"cmd": "tap", "u": 0.5, "v": 0.5}).status_code == 409  # the arm is taken
    model.gate.set()
    wait_idle(con)

    state = client.get(f"/api/run/state?id={started['id']}&since=1").get_json()
    run = state["run"]
    assert (
        run["status"] == "success" and run["n_steps"] == 2 and [s["step"] for s in run["steps"]] == [2]
    )  # only what is new
    full = client.get(f"/api/run/{started['id']}").get_json()
    assert full["steps"][0]["thought"] == "打开音乐应用" and full["steps"][0]["coords"]["coordinate"] == [0.25, 0.7]
    # the arm really tapped there, and ended at the park pose
    assert any(con.arm._arm.screen.error_mm((0.25, 0.7), d) < 0.5 for d in con.arm._arm.downs)
    assert np.allclose(con.arm._arm.q, con.setup.park_q, atol=1e-3)
    # the model was asked with the instruction, the stylus note and a picture
    text = json.dumps(model.asked[0], ensure_ascii=False)
    assert "打开音乐，\\n播放音乐" in text and "step_01.png" in text

    picture = client.get(f"/api/run/{started['id']}/step_1.jpg?anno=1")
    image = cv2.imdecode(np.frombuffer(picture.data, np.uint8), cv2.IMREAD_COLOR)
    assert picture.mimetype == "image/jpeg" and image.shape[1] == 900  # the rectified screen, scaled for the page
    for bad in (
        "/api/run/nothing/step_1.jpg",
        f"/api/run/{started['id']}/step_9.jpg",
        "/api/run/..%2F..%2Fetc/step_1.jpg",
    ):
        assert client.get(bad).status_code == 404

    history = client.get("/api/run/history").get_json()["runs"]
    assert history[0]["id"] == started["id"] and history[0]["status"] == "success" and history[0]["steps"] == 2
    again = RunSession(con, session_options()).load(started["id"])  # a restarted console reads it from disk
    assert again["n_steps"] == 2 and again["instruction"] == "打开音乐，\n播放音乐"


def session_options():
    return RunOptions(vlm_url="http://127.0.0.1:1/v1", asr_url="http://127.0.0.1:1")


def test_stop_holds_the_arm_and_ends_the_run(bench):
    con, client, model, _, _ = bench
    ready(con, client)
    model.outputs = [call({"action": "click", "coordinate": [500, 500]})] * 9
    model.gate.clear()
    started = start(client, instruction="一直点", max_steps=9)
    wait_for(lambda: len(model.asked) == 1)
    client.post("/api/stop", json={})
    model.gate.set()
    wait_idle(con)
    run = client.get(f"/api/run/{started['id']}").get_json()
    assert run["status"] == "stopped" and run["n_steps"] <= 1
    downs = len(con.arm._arm.downs)
    model.outputs = [call({"action": "terminate", "status": "failure"})]
    start(client, instruction="再来")  # the stop does not linger
    wait_idle(con)
    assert client.get("/api/run/state").get_json()["run"]["status"] == "failure" and len(con.arm._arm.downs) == downs


@pytest.mark.parametrize(
    ("output", "status"),
    [
        (call({"action": "type", "text": "晴天"}), "unsupported"),
        (call({"action": "interact", "text": "请登录"}), "need_user"),
        (call({"action": "answer", "text": "在播放"}, "现在播放的是晴天"), "answer"),
    ],
)
def test_runs_that_end_by_asking_for_a_person(bench, output, status):
    con, client, model, _, _ = bench
    ready(con, client)
    model.outputs = [output]
    started = start(client, instruction="播放晴天")
    wait_idle(con)
    run = client.get(f"/api/run/{started['id']}").get_json()
    assert run["status"] == status and run["answer"]


def test_a_failing_model_or_camera_ends_the_run_with_an_explanation(bench):
    con, client, model, _, _ = bench
    ready(con, client)

    def broken(messages):
        raise ConnectionError("no route to the model")

    model.predict_mm = broken
    started = start(client, instruction="打开音乐")
    wait_idle(con)
    run = client.get(f"/api/run/{started['id']}").get_json()
    assert run["status"] == "llm_error" and con.busy is None
    cmd(con, client, cmd="camera_close")
    assert "相机没有打开" in start(client, 400, instruction="打开音乐")["error"]


def test_presets_are_kept_and_checked(bench):
    _, client, _, _, tmp = bench
    assert client.get("/api/run/presets").get_json()["presets"] == run_page.DEFAULT_PRESETS
    kept = client.post("/api/run/presets", json={"presets": ["打开音乐", " 播放  爵士乐 ", "打开音乐", ""]}).get_json()[
        "presets"
    ]
    assert kept == ["打开音乐", "播放 爵士乐"] and json.loads((tmp / "calib" / "run_presets.json").read_text()) == kept
    assert client.post("/api/run/presets", json={"presets": "打开音乐"}).status_code == 400
    assert client.post("/api/run/presets", json={"presets": ["x" * 1500]}).status_code == 400


def test_speech_becomes_text(bench, monkeypatch):
    _, client, _, _, _ = bench
    wav = voice.wav_header(3200) + b"\x01\x02" * 1600
    response = client.post("/api/run/transcribe", data=wav, content_type="audio/wav")
    assert response.status_code == 200 and response.get_json()["text"] == "打开音乐。"
    assert client.post("/api/run/transcribe", data=b"x", content_type="text/plain").status_code == 400

    def record(seconds=None, device=None, stop=None):  # this computer's microphone, until it is told to stop
        assert stop.wait(5)
        return wav

    monkeypatch.setattr(voice, "record_wav", record)
    assert client.post("/api/run/mic", json={"action": "start"}).status_code == 200
    assert client.get("/api/run/state").get_json()["recording"]
    assert client.post("/api/run/mic", json={"action": "start"}).status_code == 409
    assert client.post("/api/run/mic", json={"action": "stop"}).get_json()["text"] == "打开音乐。"
    assert (
        client.post("/api/run/mic", json={"action": "stop"}).status_code == 400
        and not client.get("/api/run/state").get_json()["recording"]
    )

    def no_microphone(seconds=None, device=None, stop=None):
        raise voice.VoiceError("没有找到麦克风")

    monkeypatch.setattr(voice, "record_wav", no_microphone)
    client.post("/api/run/mic", json={"action": "start"})
    answer = client.post("/api/run/mic", json={"action": "stop"})
    assert answer.status_code == 502 and "没有找到麦克风" in answer.get_json()["error"]


def test_services_are_probed(services):
    status = ServiceProbe(services).check()
    assert status["vlm"]["ok"] and status["asr"]["ok"]
    other = ServiceProbe(RunOptions(vlm_url=services.vlm_url, asr_url=services.asr_url, model="another")).check()
    assert not other["vlm"]["ok"] and "another" in other["vlm"]["detail"]
    off = ServiceProbe(session_options(), timeout_s=0.3)
    assert not off.status()["vlm"]["ok"]  # at once, from memory: the first check runs in the background
    assert "serve_vlm.sh" in off.check()["vlm"]["detail"] and "serve_asr.sh" in off.check()["asr"]["detail"]


def test_without_the_model_service_nothing_starts(bench):
    con, client, _, _, _ = bench
    ready(con, client)
    sent = len(con.arm._arm.commands)
    with pytest.raises(run_page.SetupError, match="serve_vlm.sh"):
        RunSession(con, session_options()).start("打开音乐")
    assert len(con.arm._arm.commands) == sent and con.busy is None


def test_the_console_serves_the_run_page_and_the_token_guards_it(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    con = web.Console(
        config.SetupStore(tmp_path / "s.json"), tmp_path / "v.json", page_server.CalibState(), fake=True, realtime=False
    )
    client = web.create_app(con, token="secret").test_client()
    assert client.get("/run").status_code == 401 == client.get("/api/run/state").status_code
    assert client.get("/run?token=secret").status_code == 200
    assert client.post("/api/run/start?token=secret", json={"instruction": "打开音乐"}).status_code == 400
    con.close()


KEYBOARD = {
    "q": [0.08, 0.60], "p": [0.92, 0.60], "a": [0.12, 0.71], "l": [0.88, 0.71], "z": [0.18, 0.82], "m": [0.74, 0.82],
    "space": [0.5, 0.93], "backspace": [0.90, 0.82],
}  # fmt: skip


def test_typing_on_the_calibrated_keyboard(bench):
    con, client, model, _, tmp = bench
    ready(con, client)
    assert client.get("/api/run/keyboard").get_json()["anchors"] is None
    for broken in (
        {**KEYBOARD, "q": [0.95, 0.60]},
        {**KEYBOARD, "space": [0.5, 0.5]},
        {**KEYBOARD, "backspace": [0.3, 0.82]},
    ):
        assert client.post("/api/run/keyboard", json={"anchors": broken}).status_code == 400
    partial = {k: v for k, v in KEYBOARD.items() if k != "m"}
    assert "缺少" in client.post("/api/run/keyboard", json={"anchors": partial}).get_json()["error"]
    saved = client.post("/api/run/keyboard", json={"anchors": KEYBOARD}).get_json()
    assert saved["anchors"] == KEYBOARD and con.store.load().keyboard == KEYBOARD  # kept in the setup file

    model.outputs = [
        call({"action": "type", "text": "晴天 lo"}, "输入歌名"),
        call({"action": "terminate", "status": "success"}),
    ]
    started = start(client, instruction="播放晴天")
    wait_idle(con)
    run = client.get(f"/api/run/{started['id']}").get_json()
    assert run["status"] == "success" and run["steps"][0]["did"] == "输入文字 '晴天 lo'"
    assert "拼音" in json.dumps(model.asked[0], ensure_ascii=False)  # the model is told what the device can do
    # pinyin of 晴天, and the Latin letters, each on its key
    from tablet_tap.core.keyboard import Keyboard
    from tablet_tap.core.keyboard import keystrokes

    keys = keystrokes("晴天 lo")
    assert "".join(k for k in keys if len(k) == 1) == "qingtianlo" and "space" not in keys  # no space inside Chinese
    board = Keyboard({k: tuple(v) for k, v in KEYBOARD.items()})
    wanted = [board.position(k) for k in keys]
    downs = con.arm._arm.downs[-len(keys) :]
    assert len(con.arm._arm.downs) >= len(keys)
    for want, got in zip(wanted, downs, strict=True):
        assert con.arm._arm.screen.error_mm(want, got) < 1.0, (want, got)
    client.post("/api/run/keyboard", json={"anchors": None})
    assert client.get("/api/run/keyboard").get_json()["anchors"] is None


def test_the_prompt_tab_sends_lines_and_its_own_note(bench):
    con, client, model, _, _ = bench
    ready(con, client)
    page = client.get("/console").get_data(as_text=True)
    assert 'data-tab="prompt"' in page and 'data-tab="tap"' in page
    default = client.get("/api/run/state").get_json()["add_info"]
    assert "机械臂" in default
    for body, want, absent in (
        ({"instruction": " 打开音乐 \n\n 播放第一首 "}, "打开音乐\n播放第一首 (" + default + ")", "XX"),
        ({"instruction": "打开音乐", "add_info": "只用右手边的图标"}, "打开音乐 (只用右手边的图标)", default),
        ({"instruction": "打开音乐", "add_info": ""}, "打开音乐", default),
    ):
        model.outputs, model.asked = [call({"action": "terminate", "status": "success"})], []
        start(client, **body)
        wait_idle(con)
        asked = json.dumps(model.asked, ensure_ascii=False, default=str)
        assert json.dumps(want, ensure_ascii=False)[1:-1] in asked and absent not in asked
    assert "附加说明太长" in start(client, 400, instruction="x", add_info="长" * 1100)["error"]
