"""Tests of the GUI agent loop with a scripted model and a FakeDevice (no hardware, no GPU, no model server)."""

from __future__ import annotations

import copy
import http.server
import importlib.util
import json
import socket
import sys
import threading
import time
import unittest

from PIL import Image
import pytest

from tablet_tap.agent import gui_agent
from tablet_tap.agent import upstream as up
from tablet_tap.agent.device import FakeDevice
from tablet_tap.agent.gui_agent import AgentResult
from tablet_tap.agent.gui_agent import GuiAgent
from tablet_tap.agent.gui_agent import GuiOwlLlm

WIDTH, HEIGHT = 1280, 800


def tc(action: str, thought: str = "Do it.", **arguments) -> str:
    """A model output in upstream's response format."""
    call = {"name": "mobile_use", "arguments": {"action": action, **arguments}}
    return f"Action: {thought}\n<tool_call>\n{json.dumps(call, ensure_ascii=False)}\n</tool_call>"


class ScriptedLlm:
    """Returns the scripted outputs in order (the last one repeats) and keeps the messages it was given."""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[list[dict]] = []

    def predict_mm(self, messages: list[dict]) -> tuple[str, None, None]:
        self.calls.append(copy.deepcopy(messages))
        return self.outputs[min(len(self.calls), len(self.outputs)) - 1], None, None


def screen(step: int) -> Image.Image:
    """A screenshot that differs per step."""
    return Image.new("RGB", (WIDTH, HEIGHT), (10 * step % 256, 80, 120))


@pytest.fixture(autouse=True)
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace time.sleep everywhere and record what was asked for."""
    asked: list[float] = []
    monkeypatch.setattr(time, "sleep", asked.append)
    return asked


def make_agent(tmp_path, outputs: list[str], device: FakeDevice | None = None, **kwargs):
    device = device if device is not None else FakeDevice(screen)
    llm = ScriptedLlm(outputs)
    kwargs.setdefault("settle_s", 0.0)
    agent = GuiAgent(device, llm=llm, run_dir=tmp_path / "run", **kwargs)
    return agent, device, llm


def read_records(run_dir) -> list[dict]:
    with open(f"{run_dir}/actions.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# -- coordinates ------------------------------------------------------------------------------------------------


def test_click_long_press_swipe_scroll_reach_the_device_as_uv(tmp_path):
    outputs = [
        tc("click", coordinate=[500, 600]),
        tc("long_press", coordinate=[250, 125], time=1.5),
        tc("long_press", coordinate=[0, 1000]),
        tc("swipe", coordinate=[100, 800], coordinate2=[900, 200]),
        tc("scroll", coordinate=[500, 750], coordinate2=[500, 250]),
        tc("terminate", status="success"),
    ]
    agent, device, _ = make_agent(tmp_path, outputs)
    result = agent.run("打开音乐")
    assert result.status == "success" and result.steps == 6
    assert device.actions == [
        {"action": "click", "u": 0.5, "v": 0.6},
        {"action": "long_press", "u": 0.25, "v": 0.125, "duration_s": 1.5},
        {"action": "long_press", "u": 0.0, "v": 1.0, "duration_s": 0.8},  # no time: the device's default
        {"action": "swipe", "u1": 0.1, "v1": 0.8, "u2": 0.9, "v2": 0.2, "duration_s": 0.8},
        {"action": "swipe", "u1": 0.5, "v1": 0.75, "u2": 0.5, "v2": 0.25, "duration_s": 0.8},
    ]


def test_uv_does_not_depend_on_the_image_size_and_matches_upstream_pixels(tmp_path):
    # Upstream sends int(x / 1000 * W') as the adb pixel; (u * W, v * H) must be that pixel up to its rounding.
    for width, height in ((1280, 800), (2560, 1600), (1080, 2400)):
        resized_h, resized_w = up.smart_resize(height, width, factor=16, min_pixels=3136, max_pixels=1003520 * 200)
        for xy in ([0, 0], [500, 600], [333, 777], [1000, 1000]):
            px, py = up.rescale_coordinates({"coordinate": list(xy)}, resized_w, resized_h)["coordinate"]
            u, v = gui_agent.to_uv(xy)
            assert (u, v) == (xy[0] / 1000, xy[1] / 1000)
            assert abs(u * width - px) <= 8 and abs(v * height - py) <= 8
    assert gui_agent.to_uv([-20, 1300]) == (0.0, 1.0)


# -- the other actions ------------------------------------------------------------------------------------------


def test_system_buttons_back_and_home(tmp_path):
    outputs = [
        tc("system_button", button="Back"),
        tc("system_button", button="Home"),
        tc("terminate", status="success"),
    ]
    agent, device, _ = make_agent(tmp_path, outputs)
    assert agent.run("返回").status == "success"
    assert device.actions == [{"action": "back"}, {"action": "home"}]


@pytest.mark.parametrize("button", ["Menu", "Enter"])
def test_system_button_menu_enter_are_unsupported(tmp_path, button):
    agent, device, _ = make_agent(tmp_path, [tc("system_button", button=button)])
    result = agent.run("x")
    assert result.status == "unsupported" and button in result.answer
    assert device.actions == []


def test_open_goes_home_and_the_run_continues(tmp_path):
    outputs = [tc("open", text="音乐"), tc("click", coordinate=[120, 340]), tc("terminate", status="success")]
    agent, device, llm = make_agent(tmp_path, outputs)
    result = agent.run("打开音乐")
    assert result.status == "success" and result.steps == 3
    assert device.actions == [{"action": "home"}, {"action": "click", "u": 0.12, "v": 0.34}]
    # the open step is part of the history the model sees next
    assert llm.calls[1][2] == {"role": "assistant", "content": [{"text": outputs[0]}]}
    assert read_records(result.run_dir)[0]["executed"] == {"call": "home", "reason": "open '音乐'"}


def test_type_on_a_device_that_cannot_type_is_unsupported(tmp_path):
    device = FakeDevice(screen, can_type=False)
    agent, _, _ = make_agent(tmp_path, [tc("type", text="周杰伦")], device=device)
    result = agent.run("搜索周杰伦")
    assert result.status == "unsupported" and "周杰伦" in result.answer
    assert device.actions == []


def test_type_on_a_device_that_can_type(tmp_path):
    agent, device, _ = make_agent(tmp_path, [tc("type", text="hello"), tc("terminate", status="success")])
    assert agent.run("x").status == "success"
    assert device.actions == [{"action": "type_text", "text": "hello"}]


def test_key_is_unsupported(tmp_path):
    agent, device, _ = make_agent(tmp_path, [tc("key", text="volume_up")])
    result = agent.run("调大音量")
    assert result.status == "unsupported" and "volume_up" in result.answer
    assert device.actions == []


def test_wait_sleeps_the_given_time_capped_at_ten_seconds(tmp_path, sleeps):
    outputs = [tc("wait", time=3), tc("wait"), tc("wait", time=600), tc("terminate", status="success")]
    agent, device, _ = make_agent(tmp_path, outputs)
    result = agent.run("x")
    assert result.status == "success" and device.actions == []
    assert [r["executed"] for r in read_records(result.run_dir)[:3]] == [
        {"call": "wait", "seconds": 3.0},
        {"call": "wait", "seconds": 2.0},
        {"call": "wait", "seconds": 10.0},
    ]
    assert sum(sleeps) == pytest.approx(15.0)


def test_settle_sleep_after_each_step(tmp_path, sleeps):
    agent, _, _ = make_agent(
        tmp_path, [tc("click", coordinate=[1, 1]), tc("terminate", status="success")], settle_s=2.0
    )
    assert agent.run("x").status == "success"
    assert sum(sleeps) == pytest.approx(2.0)  # once, after the click; not after terminate


def test_unknown_action_is_skipped_like_upstream(tmp_path):
    agent, device, _ = make_agent(tmp_path, [tc("double_click", coordinate=[5, 5]), tc("terminate", status="success")])
    result = agent.run("x")
    assert result.status == "success" and result.steps == 2 and device.actions == []


# -- the ways a run ends ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["success", "failure"])
def test_terminate_gives_the_models_status(tmp_path, status):
    agent, _, _ = make_agent(tmp_path, [tc("click", coordinate=[10, 10]), tc("terminate", status=status)])
    result = agent.run("x")
    assert result == AgentResult(status=status, steps=2, answer="", run_dir=str(tmp_path / "run"))


def test_answer_uses_upstreams_text_before_the_tool_call(tmp_path):
    agent, _, _ = make_agent(tmp_path, [tc("answer", thought="电量是 80%。", text="80%")])
    result = agent.run("电量多少")
    assert result.status == "answer" and result.answer == "Action: 电量是 80%。"
    # only a tool call: fall back to its text
    bare = '<tool_call>{"name": "mobile_use", "arguments": {"action": "answer", "text": "80%"}}</tool_call>'
    agent, _, _ = make_agent(tmp_path, [bare])
    assert agent.run("电量多少").answer == "80%"


@pytest.mark.parametrize("name", ["interact", "call_user", "calluser"])
def test_interact_needs_the_user(tmp_path, name):
    agent, device, _ = make_agent(tmp_path, [tc(name, text="请输入验证码")])
    result = agent.run("登录")
    assert result.status == "need_user" and result.answer == "请输入验证码" and result.steps == 1
    assert device.actions == []


def test_max_steps(tmp_path):
    outputs = [tc("click", coordinate=[100 * i, 100]) for i in range(1, 9)]
    agent, device, llm = make_agent(tmp_path, outputs, max_steps=4)
    result = agent.run("x")
    assert result.status == "max_steps" and result.steps == 4
    assert len(device.actions) == 4 and len(llm.calls) == 4


def test_should_stop_before_a_step(tmp_path):
    polls = iter([False, False, True])  # before step 1, after its settle wait, before step 2
    agent, device, llm = make_agent(
        tmp_path, [tc("click", coordinate=[100, 100])], should_stop=lambda: next(polls, True)
    )
    result = agent.run("x")
    assert result.status == "stopped" and result.steps == 1
    assert len(device.actions) == 1 and len(llm.calls) == 1
    assert json.load(open(tmp_path / "run" / "result.json", encoding="utf-8"))["status"] == "stopped"


def test_should_stop_at_once_runs_nothing(tmp_path):
    agent, device, llm = make_agent(tmp_path, [tc("click", coordinate=[1, 1])], should_stop=lambda: True)
    result = agent.run("x")
    assert (result.status, result.steps, device.step, llm.calls) == ("stopped", 0, 0, [])


def test_should_stop_during_a_wait(tmp_path, sleeps, monkeypatch):
    stop = {"now": False}

    def sleep(dt: float) -> None:
        sleeps.append(dt)
        stop["now"] = sum(sleeps) > 0.95

    monkeypatch.setattr(time, "sleep", sleep)
    agent, _, _ = make_agent(tmp_path, [tc("wait", time=8)], should_stop=lambda: stop["now"])
    result = agent.run("x")
    assert result.status == "stopped" and result.steps == 1
    assert sum(sleeps) == pytest.approx(1.0)  # not the full 8 s


def test_should_stop_during_the_settle_wait(tmp_path, sleeps, monkeypatch):
    stop = {"now": False}

    def sleep(dt: float) -> None:
        sleeps.append(dt)
        stop["now"] = True

    monkeypatch.setattr(time, "sleep", sleep)
    agent, device, llm = make_agent(
        tmp_path, [tc("click", coordinate=[1, 1])], should_stop=lambda: stop["now"], settle_s=5.0
    )
    result = agent.run("x")
    assert result.status == "stopped" and len(device.actions) == 1 and len(llm.calls) == 1
    assert len(sleeps) == 1


# -- model-side failures ----------------------------------------------------------------------------------------


def test_unparsable_output_is_retried_with_upstreams_hint(tmp_path):
    outputs = ["I will tap the button.", tc("click", coordinate=[500, 500]), tc("terminate", status="success")]
    agent, device, llm = make_agent(tmp_path, outputs)
    result = agent.run("x")
    assert result.status == "success" and result.steps == 2
    assert device.actions == [{"action": "click", "u": 0.5, "v": 0.5}]
    first, retry = llm.calls[0], llm.calls[1]
    assert {"text": up.FORMAT_RETRY_HINT} not in first[-1]["content"]
    assert retry[-1]["content"][-1] == {"text": up.FORMAT_RETRY_HINT}
    assert retry[-1]["content"][:-1] == first[-1]["content"]  # same screenshot, same prompt
    assert {"text": up.FORMAT_RETRY_HINT} not in llm.calls[2][-1]["content"]
    record = read_records(result.run_dir)[0]
    assert record["attempts"] == 2 and record["output"] == outputs[1]
    # the unparsable output is not part of the history
    assert llm.calls[2][2] == {"role": "assistant", "content": [{"text": outputs[1]}]}


def test_arguments_upstream_would_crash_on_are_retried(tmp_path):
    outputs = [
        tc("click"),
        tc("swipe", coordinate=[1, 2]),
        tc("click", coordinate="500,500"),
        tc("terminate", status="success"),
    ]
    agent, device, llm = make_agent(tmp_path, outputs)
    result = agent.run("x")
    assert result.status == "success" and result.steps == 1 and len(llm.calls) == 4 and device.actions == []


def test_all_outputs_unparsable_is_a_parse_error(tmp_path):
    agent, device, llm = make_agent(tmp_path, ["no tool call here"], max_parse_retries=3)
    result = agent.run("x")
    assert result.status == "parse_error" and result.steps == 1
    assert "No <tool_call> block" in result.answer
    assert len(llm.calls) == 4 and device.actions == []  # 1 + max_parse_retries, as upstream
    assert read_records(result.run_dir)[0]["status"] == "parse_error"


def test_error_calling_llm_string_is_an_llm_error(tmp_path):
    agent, device, llm = make_agent(tmp_path, [tc("click", coordinate=[1, 1]), up.ERROR_CALLING_LLM])
    result = agent.run("x")
    assert up.ERROR_CALLING_LLM == "Error calling LLM"
    assert result.status == "llm_error" and result.steps == 2 and result.answer
    assert len(llm.calls) == 2  # not retried as a parse failure
    assert len(device.actions) == 1


def test_llm_exception_is_an_llm_error(tmp_path):
    class Broken:
        def predict_mm(self, messages):
            raise ConnectionError("boom")

    agent = GuiAgent(FakeDevice(screen), llm=Broken(), run_dir=tmp_path / "run", settle_s=0)
    result = agent.run("x")
    assert result.status == "llm_error" and "boom" in result.answer


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_default_llm_with_a_dead_server_fails_fast(tmp_path):
    base_url = f"http://127.0.0.1:{_free_port()}/v1"
    agent = GuiAgent(
        FakeDevice(screen), base_url=base_url, run_dir=tmp_path / "run", settle_s=0, llm_max_retry=1, llm_retry_wait_s=0
    )
    assert isinstance(agent.llm, GuiOwlLlm) and agent.llm.max_retry == 1 and agent.llm.RETRY_WAITING_SECONDS == 0
    assert up.GUIOwlWrapper.RETRY_WAITING_SECONDS == 20  # upstream's class default is untouched
    started = time.monotonic()
    result = agent.run("x")
    assert result.status == "llm_error" and base_url in result.answer
    assert time.monotonic() - started < 5


def test_default_llm_keeps_upstreams_retry_defaults(tmp_path):
    agent = GuiAgent(FakeDevice(screen), run_dir=tmp_path / "run")
    assert agent.llm.max_retry == 10 and agent.llm.RETRY_WAITING_SECONDS == 20 and agent.llm.model == "gui-owl"
    assert str(agent.llm.bot.base_url) == "http://127.0.0.1:8802/v1/"


def test_default_llm_talks_to_an_openai_compatible_server(tmp_path):
    """The whole default path (build_messages -> GuiOwlLlm -> HTTP) against a canned local server."""
    requests: list[dict] = []
    replies = [tc("click", coordinate=[250, 750]), tc("terminate", status="success")]

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            message = {"role": "assistant", "content": replies[min(len(requests), len(replies)) - 1]}
            reply = {
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
            }
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        device = FakeDevice(screen)
        agent = GuiAgent(
            device,
            base_url=f"http://127.0.0.1:{server.server_address[1]}/v1",
            model="gui-owl",
            run_dir=tmp_path / "run",
            settle_s=0,
            llm_max_retry=1,
            llm_retry_wait_s=0,
        )
        result = agent.run("打开音乐")
    finally:
        server.shutdown()
        server.server_close()
    assert result.status == "success" and device.actions == [{"action": "click", "u": 0.25, "v": 0.75}]
    assert [r["model"] for r in requests] == ["gui-owl", "gui-owl"]
    first = requests[0]["messages"]
    assert first[0] == {"role": "system", "content": [{"type": "text", "text": up.SYSTEM_PROMPT}]}
    text, image = first[1]["content"]
    assert text["text"].endswith(f"打开音乐 ({gui_agent.DEFAULT_ADD_INFO})\n\nPrevious actions:\nNone")
    assert image["image_url"]["url"].startswith("data:image/png;base64,")
    # second request: step 1's screenshot and output, then step 2's screenshot
    assert [m["role"] for m in requests[1]["messages"]] == ["system", "user", "assistant", "user"]


# -- the stuck-click guard --------------------------------------------------------------------------------------


def test_same_click_three_times_in_a_row_stops_the_run(tmp_path):
    outputs = [
        tc("click", coordinate=[500, 600]),
        tc("click", coordinate=[510, 590]),
        tc("click", coordinate=[505, 600]),
    ]
    agent, device, _ = make_agent(tmp_path, outputs)
    result = agent.run("打开音乐")
    assert result.status == "failure" and result.steps == 3
    assert "卡住" in result.answer and "没有被屏幕识别" in result.answer
    assert len(device.actions) == 2  # the third tap is not made
    assert read_records(result.run_dir)[-1]["executed"] is None


def test_clicks_further_apart_or_interrupted_are_not_stuck(tmp_path):
    far = [tc("click", coordinate=[500, 600]), tc("click", coordinate=[500, 616]), tc("click", coordinate=[500, 632])]
    agent, device, _ = make_agent(tmp_path, [*far, tc("terminate", status="success")])
    assert agent.run("x").status == "success" and len(device.actions) == 3

    same = tc("click", coordinate=[500, 600])
    outputs = [same, same, tc("system_button", button="Back"), same, same, tc("terminate", status="success")]
    agent, device, _ = make_agent(tmp_path, outputs)
    assert agent.run("x").status == "success" and len(device.actions) == 5

    agent, device, _ = make_agent(tmp_path, [same, same, same, same, tc("terminate", status="success")], stuck_clicks=0)
    assert agent.run("x").status == "success" and len(device.actions) == 4


# -- history, prompt, logs --------------------------------------------------------------------------------------


def test_history_at_step_three_is_in_upstreams_format(tmp_path):
    outputs = [
        tc("system_button", thought="Go home.", button="Home"),
        tc("click", thought="Tap the music icon.", coordinate=[300, 400]),
        tc("terminate", status="success"),
    ]
    agent, _, llm = make_agent(tmp_path, outputs)
    result = agent.run("打开音乐播放音乐")
    run_dir = tmp_path / "run"
    prompt = f"打开音乐播放音乐 ({gui_agent.DEFAULT_ADD_INFO})"
    images = [str(run_dir / f"step_{i:02d}.png") for i in (1, 2, 3)]
    history = [{"output": outputs[0], "image": images[0]}, {"output": outputs[1], "image": images[1]}]

    messages = llm.calls[2]
    assert messages == up.build_messages(images[2], prompt, history, "gui-owl")
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert messages[0]["content"] == [{"text": up.SYSTEM_PROMPT}]
    instruction_text, first_image = messages[1]["content"]
    assert f"{prompt}\n\nPrevious actions:\nNone" in instruction_text["text"]
    assert first_image == {"image": "file://" + images[0]}
    assert messages[2]["content"] == [{"text": outputs[0]}]
    assert messages[3]["content"] == [{"image": "file://" + images[1]}]
    assert messages[4]["content"] == [{"text": outputs[1]}]
    assert messages[5]["content"] == [{"image": "file://" + images[2]}]
    assert result.status == "success"


def test_steps_older_than_upstreams_window_are_summarised_as_text(tmp_path):
    outputs = [tc("click", thought=f"Tap item {i}.", coordinate=[100 * i, 100]) for i in range(1, 7)]
    agent, _, llm = make_agent(tmp_path, [*outputs, tc("terminate", status="success")])
    assert agent.run("x").status == "success"
    messages = llm.calls[6]  # step 7: six steps of history, upstream keeps four of them as images
    assert sum("image" in item for m in messages for item in m["content"]) == 5
    assert "Previous actions:\nStep 1: Tap item 1.\nStep 2: Tap item 2." in messages[1]["content"][0]["text"]


def test_add_info_can_be_dropped(tmp_path):
    agent, _, llm = make_agent(tmp_path, [tc("terminate", status="success")], add_info="")
    agent.run("打开音乐")
    assert llm.calls[0][1]["content"][0]["text"].endswith("打开音乐\n\nPrevious actions:\nNone")
    assert all(word in gui_agent.DEFAULT_ADD_INFO for word in ("open", "type", "key", "terminate", "主屏幕"))


def test_run_dir_contents_and_on_step(tmp_path):
    outputs = [
        tc("click", coordinate=[500, 600]),
        tc("swipe", coordinate=[100, 500], coordinate2=[900, 500]),
        tc("system_button", button="Back"),
        tc("terminate", status="success"),
    ]
    seen: list[dict] = []
    agent, _, _ = make_agent(tmp_path, outputs, on_step=seen.append)
    result = agent.run("打开音乐")
    run_dir = tmp_path / "run"
    assert result.run_dir == str(run_dir)
    # upstream annotates clicks and swipes only
    assert sorted(p.name for p in run_dir.iterdir()) == [
        "actions.jsonl",
        "result.json",
        "step_01.png",
        "step_01_anno.png",
        "step_02.png",
        "step_02_anno.png",
        "step_03.png",
        "step_04.png",
    ]
    with Image.open(run_dir / "step_01.png") as raw, Image.open(run_dir / "step_01_anno.png") as anno:
        assert raw.size == anno.size == (WIDTH, HEIGHT)
        assert raw.getpixel((640, 480)) == (0, 80, 120)
        assert anno.getpixel((640, 480)) == (255, 0, 0)  # upstream's red dot at 500/1000, 600/1000 of the image
        assert anno.getpixel((100, 100)) == (0, 80, 120)
    with Image.open(run_dir / "step_02_anno.png") as anno:
        assert anno.getpixel((640, 400)) == (255, 0, 0)  # the swipe line

    records = read_records(run_dir)
    assert records == seen and [r["step"] for r in records] == [1, 2, 3, 4]
    first = records[0]
    assert first["output"] == outputs[0]
    assert first["action"] == {"action": "click", "coordinate": [500, 600]}  # the model's own units
    assert first["coords"] == {"coordinate": [0.5, 0.6]}
    assert first["executed"] == {"call": "click", "u": 0.5, "v": 0.6}
    assert first["image"] == "step_01.png" and first["annotated"] == "step_01_anno.png"
    assert first["image_size"] == [WIDTH, HEIGHT] and first["attempts"] == 1
    assert first["llm_s"] >= 0 and first["device_s"] >= 0 and first["time"] and first["status"] == ""
    assert records[1]["coords"] == {"coordinate": [0.1, 0.5], "coordinate2": [0.9, 0.5]}
    assert records[2]["executed"] == {"call": "back"} and records[2]["annotated"] == ""
    assert records[3]["status"] == "success" and records[3]["executed"] is None

    saved = json.load(open(run_dir / "result.json", encoding="utf-8"))
    assert saved["status"] == "success" and saved["steps"] == 4 and saved["run_dir"] == str(run_dir)
    assert saved["instruction"] == "打开音乐" and saved["model"] == "gui-owl" and saved["started"] and saved["finished"]


def test_default_run_dir_is_new_for_every_run(tmp_path, monkeypatch):
    monkeypatch.setattr(gui_agent, "PROJECT_DIR", tmp_path)
    agent = GuiAgent(FakeDevice(screen), llm=ScriptedLlm([tc("terminate", status="success")]), settle_s=0)
    first, second = agent.run("a"), agent.run("b")
    assert first.run_dir != second.run_dir
    for result in (first, second):
        assert result.run_dir.startswith(str(tmp_path / "runs")) and len(read_records(result.run_dir)) == 1


def test_device_fault_is_recorded_and_raised(tmp_path):
    class Faulty(FakeDevice):
        def click(self, u: float, v: float) -> None:
            raise RuntimeError("CAN feedback stale")

    agent, _, _ = make_agent(tmp_path, [tc("click", coordinate=[1, 1])], device=Faulty(screen))
    with pytest.raises(RuntimeError, match="CAN feedback stale"):
        agent.run("x")
    assert read_records(tmp_path / "run")[0]["status"] == "exception"
    assert json.load(open(tmp_path / "run" / "result.json", encoding="utf-8"))["status"] == "exception"


# -- FakeDevice and the CLI -------------------------------------------------------------------------------------


def test_fake_device_replays_a_list_and_repeats_the_last(tmp_path):
    path = tmp_path / "b.png"
    Image.new("RGB", (64, 40), (0, 255, 0)).save(path)
    device = FakeDevice([Image.new("RGBA", (64, 40), (255, 0, 0, 255)), path])
    shots = [device.screenshot() for _ in range(3)]
    assert [s.mode for s in shots] == ["RGB"] * 3
    assert [s.getpixel((0, 0)) for s in shots] == [(255, 0, 0), (0, 255, 0), (0, 255, 0)]
    with pytest.raises(ValueError):
        FakeDevice([])


def test_cli_runs_recorded_images_through_a_fake_device(tmp_path, monkeypatch, capsys):
    shots = tmp_path / "shots"
    shots.mkdir()
    for i in range(3):
        screen(i).save(shots / f"shot_{i}.png")
    screen(9).save(shots / "shot_0_anno.png")  # an annotated copy is not a screenshot
    llm = ScriptedLlm(
        [tc("click", coordinate=[500, 600]), tc("system_button", button="Back"), tc("click", coordinate=[1, 2])]
    )
    made: list[dict] = []

    def fake_llm(api_key, base_url, model, **kwargs):
        made.append({"api_key": api_key, "base_url": base_url, "model": model, **kwargs})
        return llm

    monkeypatch.setattr(gui_agent, "GuiOwlLlm", fake_llm)
    run_dir = tmp_path / "out"
    argv = ["--images", str(shots), "--instruction", "打开音乐", "--base_url", "http://h:1/v1", "--model", "m"]
    assert gui_agent.main([*argv, "--run_dir", str(run_dir)]) == 0
    assert made == [
        {"api_key": "EMPTY", "base_url": "http://h:1/v1", "model": "m", "max_retry": None, "retry_wait_s": None}
    ]
    assert len(llm.calls) == 3  # max_steps defaults to the number of images
    out = capsys.readouterr().out
    assert '"action": "click", "u": 0.5, "v": 0.6' in out and "max_steps" in out and str(run_dir) in out
    assert len(read_records(run_dir)) == 3


# -- upstream's own tests, through the import shim --------------------------------------------------------------


def test_import_shim_leaves_no_generic_module_names():
    for name in ("utils", "packages", "run_gui_owl_1_5_for_mobile"):
        module = sys.modules.get(name)
        assert module is None or up.UPSTREAM_DIR.as_posix() not in str(getattr(module, "__file__", ""))
    assert up.run_mobile.build_messages is up.utils.build_messages
    assert up.run_mobile.smart_resize is up.utils.smart_resize


def test_upstream_parse_action_tests_pass_through_the_shim(monkeypatch):
    # upstream's test file does `from run_gui_owl_1_5_for_mobile import parse_action`: give it the shim's module
    monkeypatch.setitem(sys.modules, "run_gui_owl_1_5_for_mobile", up.run_mobile)
    spec = importlib.util.spec_from_file_location(
        "_upstream_test_parse_action", up.UPSTREAM_DIR / "test_parse_action.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.parse_action is up.parse_action
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    outcome = unittest.TextTestRunner(verbosity=0).run(suite)
    assert outcome.testsRun == 12 and outcome.wasSuccessful(), outcome.failures + outcome.errors
