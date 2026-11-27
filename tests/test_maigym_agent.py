"""MAI-Gym-9B 原生端到端多步视觉代理（滑动窗口双图 + 内生状态账本）单元测试。"""

from __future__ import annotations

from PIL import Image

from tablet_tap.agent.car_reactive_agent import ReactiveCarAgent
from tablet_tap.agent.device import FakeDevice
from tablet_tap.agent.maigym_agent import MaiGymMultiStepAgent
from tablet_tap.agent.maigym_agent import build_multistep_messages
from tablet_tap.agent.maigym_agent import parse_maigym_action


def _blank(color: tuple[int, int, int] = (240, 240, 240)) -> Image.Image:
    return Image.new("RGB", (64, 40), color)


def test_parse_maigym_action_with_status_prefix() -> None:
    raw = (
        "状态：已完成[壁纸青山]；在空调页，点主驾温度调低\n"
        "<tool_call>\n"
        '{"name": "mobile_use", "arguments": {"action_type": "click", "coordinate": [250, 300]}}\n'
        "</tool_call>"
    )
    act = parse_maigym_action(raw)
    assert act["action_type"] == "click"
    assert act["coordinate"] == [250, 300]


def test_sliding_window_collapses_old_screenshots() -> None:
    imgs = ["b64_0", "b64_1", "b64_2", "b64_3"]
    turns = ["状态：步1\n<tool_call>{}</tool_call>", "状态：步2\n<tool_call>{}</tool_call>", "状态：步3\n<tool_call>{}</tool_call>"]
    msgs = build_multistep_messages("测试任务", imgs, turns, keep_last_images=2)

    img_msgs = [m for m in msgs if m["role"] == "user" and any(p.get("type") == "image_url" for p in m["content"])]
    collapsed = [m for m in msgs if m["role"] == "user" and m["content"] == [{"type": "text", "text": "[历史截图]"}]]
    asst_msgs = [m for m in msgs if m["role"] == "assistant"]

    assert len(img_msgs) == 2
    assert len(collapsed) == 2
    assert len(asst_msgs) == 3
    assert "状态：步1" in asst_msgs[0]["content"]


def test_agent_click_drag_and_terminate_success() -> None:
    dev = FakeDevice([_blank((200, 200, 200)), _blank((180, 180, 180)), _blank((160, 160, 160))], can_type=False)
    replies = iter(
        [
            '状态：在桌面，点底栏电话\n<tool_call>\n{"name": "mobile_use", "arguments": {"action_type": "click", "coordinate": [500, 999]}}\n</tool_call>',
            '状态：在通讯录，向上滑动找赵六\n<tool_call>\n{"name": "mobile_use", "arguments": {"action_type": "drag", "start_coordinate": [680, 780], "end_coordinate": [680, 280]}}\n</tool_call>',
            '状态：已拨打赵六，任务完成\n<tool_call>\n{"name": "mobile_use", "arguments": {"action_type": "terminate", "status": "success", "answer": "已呼叫赵六"}}\n</tool_call>',
        ]
    )
    agent = MaiGymMultiStepAgent(dev, chat_fn=lambda _m: next(replies), settle_s=0.0)
    res = agent.run("打开通讯录给赵六打电话")

    assert res.status == "success"
    assert res.steps == 3
    assert len(dev.actions) == 2
    assert dev.actions[0]["action"] == "click"
    assert abs(dev.actions[0]["u"] - 500 / 999) < 1e-4
    assert abs(dev.actions[0]["v"] - 1.0) < 1e-4
    assert dev.actions[1]["action"] == "swipe"


def test_agent_safety_guardrail_zero_clicks() -> None:
    dev = FakeDevice([_blank()], can_type=False)
    reply = '状态：高危不可逆操作，需人工确认\n<tool_call>\n{"name": "mobile_use", "arguments": {"action_type": "ask_user", "text": "恢复出厂设置属高危操作，请亲自确认"}}\n</tool_call>'
    agent = MaiGymMultiStepAgent(dev, chat_fn=lambda _m: reply, settle_s=0.0)
    res = agent.run("帮我恢复出厂设置")

    assert res.status == "need_user"
    assert res.steps == 1
    assert dev.actions == []


def test_agent_not_found_terminate_failed() -> None:
    dev = FakeDevice([_blank()], can_type=False)
    reply = '状态：已滑到底部未找到陈老王\n<tool_call>\n{"name": "mobile_use", "arguments": {"action_type": "terminate", "status": "failed", "answer": "未找到陈老王"}}\n</tool_call>'
    agent = MaiGymMultiStepAgent(dev, chat_fn=lambda _m: reply, settle_s=0.0)
    res = agent.run("打开通讯录给陈老王打电话")

    assert res.status == "failure"
    assert "陈老王" in res.answer
    assert dev.actions == []


def test_reactive_agent_safety_guardrail_zero_clicks(monkeypatch) -> None:
    dev = FakeDevice([_blank()], can_type=False)
    agent = ReactiveCarAgent(dev, grounder_url="http://mock/v1", grounder_model="mai-gym-9b-v3-s273", settle_s=0.0)
    monkeypatch.setattr(
        agent,
        "compile_goals",
        lambda _instr: {"safety_block": True, "safety_reason": "恢复出厂设置属不可逆高危操作", "goals": []},
    )
    log = agent.run("帮我恢复出厂设置")
    assert log.status == "need_user"
    assert log.clicks == 0
    assert dev.actions == []


def test_reactive_agent_toggle_idempotent_closed_loop(monkeypatch) -> None:
    dev = FakeDevice([_blank((240, 240, 240)), _blank((220, 220, 220))], can_type=False)
    agent = ReactiveCarAgent(dev, grounder_url="http://mock/v1", grounder_model="mai-gym-9b-v3-s273", settle_s=0.0)
    monkeypatch.setattr(
        agent,
        "compile_goals",
        lambda _instr: {
            "safety_block": False,
            "safety_reason": "",
            "goals": [{"domain": "climate", "kind": "toggle", "target": "“前除霜”按钮", "want_on": True}],
        },
    )
    monkeypatch.setattr(
        agent,
        "inspect_screen",
        lambda _img: {"page": "climate", "has_call_overlay": False, "settings_cat": "", "settings_sub": "", "sub_tab": ""},
    )
    bool_states = iter([False, True])
    monkeypatch.setattr(agent, "read_bool", lambda _q, _img: next(bool_states))
    monkeypatch.setattr(agent, "ground", lambda _t, _img: (0.51, 0.47))

    log = agent.run("打开前除霜")
    assert log.status == "success"
    assert log.clicks == 1
    assert len(dev.actions) == 1
    assert dev.actions[0]["action"] == "click"
