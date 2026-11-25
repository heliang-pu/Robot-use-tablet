"""An instruction carried out clause by clause: a clause counts as done when the screen says so, and what may cost money
is not pressed (both found in the runs of 2026-11-24)."""

import json
import time

import numpy as np
from PIL import Image
import pytest

from tablet_tap.agent import by_steps
from tablet_tap.agent import gui_agent
from tablet_tap.agent.device import FakeDevice


def tc(action: str, thought: str = "Do it.", **arguments) -> str:
    return f"Action: {thought}\n<tool_call>\n{json.dumps({'name': 'mobile_use', 'arguments': {'action': action, **arguments}})}\n</tool_call>"


CLICK = tc("click", "点击按钮", coordinate=[500, 870])
CLAIM = tc("answer", "已经做好了", text="好了")


class Tablet:
    """A screen of one colour per song, "desktop" or not, playing or not; the model's answers come in order."""

    def __init__(self, tmp_path, monkeypatch, instruction, outputs, song=lambda presses: 0, desktop=False, playing=lambda presses: False):  # noqa: ANN001, PLR0913
        monkeypatch.setattr(time, "sleep", lambda s: None)
        self.outputs, self.asked = outputs, []
        self.device = FakeDevice(lambda step: Image.fromarray(np.full((800, 1280, 3), 40 + 60 * song(self.presses()), np.uint8)))
        self.desktop, self.playing = desktop, playing
        clauses = by_steps.split(instruction)
        self.agent = by_steps.StepAgent(self.device, llm=self, run_dir=tmp_path / "run", settle_s=0.0, clauses=clauses, look=self.look)
        self.result = self.agent.run(instruction)

    def presses(self):
        return sum(a["action"] == "click" for a in self.device.actions)

    def predict_mm(self, messages):
        self.asked.append(json.dumps(messages, ensure_ascii=False))
        return self.outputs[min(len(self.asked), len(self.outputs)) - 1], None, None

    def look(self, picture, question):  # noqa: ARG002
        if question == by_steps.POPUP_QUESTION:
            return "没有"
        if question == by_steps.SCREEN_QUESTION:
            return "A" if self.desktop else "B"
        return by_steps.PLAYING if self.playing(self.presses()) else by_steps.PAUSED


def test_paused_music_is_not_playing_because_the_model_says_so(tmp_path, monkeypatch):
    run = Tablet(tmp_path, monkeypatch, "播放音乐", [CLAIM, CLICK], playing=lambda presses: presses > 0)
    assert run.result.status == "success" and run.presses() == 1
    assert "还没有做成" in run.asked[1]


def test_a_claim_that_the_screen_never_bears_out_is_no_success(tmp_path, monkeypatch):
    run = Tablet(tmp_path, monkeypatch, "播放音乐", [CLAIM])
    assert run.result.status == "failure" and "没有做成" in run.result.answer and run.presses() == 0


def test_a_claim_that_the_screen_bears_out_is_taken(tmp_path, monkeypatch):
    run = Tablet(tmp_path, monkeypatch, "播放音乐", [CLAIM], playing=lambda presses: True)
    assert run.result.status == "success" and "没有核对" not in run.result.answer


def test_an_app_is_not_open_while_the_desktop_is_shown(tmp_path, monkeypatch):
    assert Tablet(tmp_path, monkeypatch, "打开音乐", [CLAIM], desktop=True).result.status == "failure"


def test_an_unchecked_claim_is_told(tmp_path, monkeypatch):
    run = Tablet(tmp_path, monkeypatch, "点击设置", [CLAIM])
    assert run.result.status == "success" and "没有核对：点击设置" in run.result.answer


@pytest.mark.parametrize(("needed", "status", "presses"), [(1, "success", 1), (2, "success", 2), (99, "failure", by_steps.TRIES)])
def test_the_next_song_is_one_that_looks_different(tmp_path, monkeypatch, needed, status, presses):
    run = Tablet(tmp_path, monkeypatch, "切换下一首", [CLICK], song=lambda n: int(n >= needed))
    assert run.result.status == status and run.presses() == presses


def test_a_song_by_name_is_not_one_press():
    assert by_steps.split("打开音乐，播放晴天") is None and by_steps.split("播放周杰伦的歌") is None
    for whole in ("播放音乐", "播放第一首歌", "播放第一首", "播放下一首", "播放一首音乐", "播放"):
        assert by_steps.split(whole), whole
    assert by_steps.split("打开音乐", {"音乐": "汽水音乐"})[0].ask == "打开汽水音乐"


@pytest.mark.parametrize("said", [
    "点击弹窗中的绿色按钮以关闭广告",  # it was "去看视频"
    "点击弹窗底部的绿色按钮以关闭广告",
    "点击弹窗底部的“6元开通会员”文字选项。",
    "点击弹窗底部的“6元开通会员”以关闭弹窗",
    "点击“立即领取”，跳过广告",
    'Click on the "手机号登录" (phone number login) button to proceed with logging in.',
    "Click the green button to close the pop-up",
])  # fmt: skip
def test_what_may_cost_money_is_not_pressed_whatever_for(said):
    assert gui_agent.not_for_the_arm("Action: " + said)


@pytest.mark.parametrize("said", [
    "点击弹窗右上角的关闭按钮",
    "点击“6元开通会员”弹窗右上角的关闭按钮",
    "点击右上角的“关闭”按钮，关闭优惠券弹窗",
    "点击弹窗右上角的“×”按钮以关闭弹窗",
    "点击空白区域以关闭弹窗",
    "点击“跳过广告”按钮",
    "点击屏幕底部中央的播放按钮，开始播放内容。",
    "点击左上角箭头以返回主页",
    "Click the close button at the top right to dismiss the ad.",
])  # fmt: skip
def test_closing_and_ordinary_presses_are_made(said):
    assert gui_agent.not_for_the_arm("Action: " + said) == ""
