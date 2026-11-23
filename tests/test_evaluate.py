"""Tests of the offline grounding evaluation and of the instruction suffix (add_info) choices: a scripted model,
labelled pictures, no hardware, no model server."""

from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import threading
import types

from PIL import Image
import pytest

from tablet_tap.agent import evaluate
from tablet_tap.agent import gui_agent
from tablet_tap.agent import upstream as up
from tablet_tap.agent.gui_agent import AgentResult
from tablet_tap.cli.commands import run as run_command
from tablet_tap.errors import SetupError
from tablet_tap.errors import TabletTapError

SCREENS = pathlib.Path(__file__).resolve().parent / "data" / "screens"


def tc(action: str, **arguments) -> str:
    call = {"name": "mobile_use", "arguments": {"action": action, **arguments}}
    return f"Action: Do it.\n<tool_call>\n{json.dumps(call, ensure_ascii=False)}\n</tool_call>"


class ByInstruction:
    """A model that answers by which instruction it finds in the prompt; keeps the prompts it was given."""

    def __init__(self, answers: dict[str, str], default: str = "") -> None:
        self.answers, self.default = answers, default or tc("terminate", status="success")
        self.prompts: list[str] = []

    def predict_mm(self, messages: list[dict]) -> tuple[str, None, None]:
        text = messages[1]["content"][0]["text"]
        self.prompts.append(text)
        return next((out for key, out in self.answers.items() if key in text), self.default), None, None


def make_screen(folder: pathlib.Path, name: str = "demo", size=(200, 100)) -> pathlib.Path:
    """A labelled screen with one target in its right half and two instructions for it."""
    folder.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (30, 60, 90)).save(folder / f"{name}.png")
    label = {
        "size": list(size),
        "elements": {"ok": {"box": [size[0] // 2, 0, size[0], size[1] // 2]}},  # the top right quarter
        "instructions": [{"text": "点确定", "target": "ok"}, {"text": "再点确定", "target": "ok"}],
    }
    (folder / f"{name}.json").write_text(json.dumps(label, ensure_ascii=False), encoding="utf-8")
    return folder


def test_the_projects_screens_load_as_cases():
    cases = evaluate.load_cases(SCREENS)
    assert len(cases) == 15 and {c.variant for c in cases} == {"clean"}
    assert {c.screen for c in cases} == {"home", "player", "settings"}
    play = next(c for c in cases if c.target == "play")
    assert play.instruction == "点击播放按钮" and play.image == SCREENS / "player.png"
    assert play.box_uv == pytest.approx((864 / 1280, 522 / 800, 960 / 1280, 618 / 800))


def test_variants_are_other_pictures_of_the_same_screens(tmp_path):
    screens = make_screen(tmp_path / "screens")
    (tmp_path / "cam").mkdir()
    Image.new("RGB", (640, 320), (1, 2, 3)).save(tmp_path / "cam" / "demo__far.png")  # another size than the label's
    cases = evaluate.load_cases(screens, {"far": str(tmp_path / "cam" / "{screen}__far.png")})
    assert [(c.variant, c.image.name) for c in cases] == [("clean", "demo.png")] * 2 + [("far", "demo__far.png")] * 2
    assert {c.box_uv for c in cases} == {(0.5, 0.0, 1.0, 0.5)}  # fractions of the screen, whatever the picture's size
    with pytest.raises(SetupError, match="near"):
        evaluate.load_cases(screens, {"near": str(tmp_path / "cam" / "{screen}__near.png")})
    with pytest.raises(SetupError):
        evaluate.load_cases(tmp_path / "cam")  # pictures without labels


def test_outcomes_hit_miss_and_other(tmp_path):
    cases = evaluate.load_cases(make_screen(tmp_path / "screens"))
    # model coordinates are 0-1000 over the picture: (750, 250) is the middle of the top right quarter
    llm = ByInstruction({"再点确定": tc("click", coordinate=[250, 250]), "点确定": tc("click", coordinate=[750, 250])})
    rows = evaluate.evaluate(cases, {"none": ""}, tmp_path / "out", llm=llm)
    assert [r["outcome"] for r in rows] == ["hit", "miss"]
    assert rows[0]["did"] == {"action": "click", "u": 0.75, "v": 0.25} and rows[0]["add_info"] == "none"
    assert (pathlib.Path(rows[0]["run_dir"]) / "step_01_anno.png").is_file()  # where the model tapped, to look at

    # not a tap: what the device was told instead, or how the task was ended without telling it anything
    llm = ByInstruction({"再点确定": tc("system_button", button="Home")})
    rows = evaluate.evaluate(cases, {"none": ""}, tmp_path / "out2", llm=llm)
    assert [r["outcome"] for r in rows] == ["other", "other"]
    assert rows[0]["did"]["action"] == "success" and rows[1]["did"] == {"action": "home"}

    summary = evaluate.summarize(rows + evaluate.evaluate(cases, {"zh": "x"}, tmp_path / "out3", llm=llm))
    assert [(s["add_info"], s["variant"], s["n"], s["hit"], s["miss"], s["other"]) for s in summary] == [
        ("none", "clean", 2, 0, 0, 2),
        ("zh", "clean", 2, 0, 0, 2),
    ]


def test_a_model_server_that_does_not_answer_ends_the_evaluation(tmp_path):
    cases = evaluate.load_cases(make_screen(tmp_path / "screens"))
    with pytest.raises(TabletTapError, match="模型服务"):
        evaluate.evaluate(cases, {"none": ""}, tmp_path / "out", llm=ByInstruction({}, default=up.ERROR_CALLING_LLM))


def test_cli_compares_instruction_suffixes(tmp_path, capsys):
    screens = make_screen(tmp_path / "screens")
    llm = ByInstruction({"not available": tc("click", coordinate=[750, 250])}, default=tc("click", coordinate=[1, 1]))
    out = tmp_path / "out"
    argv = ["--screens", str(screens), "--out", str(out), "--add_info", "zh", "en", "none", "只许点击"]
    assert evaluate.main(argv, llm=llm) == 0
    record = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert record["add_info"] == {
        "zh": gui_agent.ADD_INFO_ZH,
        "en": gui_agent.ADD_INFO_EN,
        "none": "",
        "自定义1": "只许点击",
    }
    assert [(s["add_info"], s["hit"], s["miss"]) for s in record["summary"]] == [
        ("zh", 0, 2),
        ("en", 2, 0),  # the scripted model taps the target only when it reads the English text
        ("none", 0, 2),
        ("自定义1", 0, 2),
    ]
    # the suffix reaches the model the way upstream appends --add_info, and "none" appends nothing
    assert f"点确定 ({gui_agent.ADD_INFO_ZH})\n\nPrevious actions:" in llm.prompts[0]
    assert "点确定\n\nPrevious actions:" in llm.prompts[4]
    assert "点确定 (只许点击)\n\nPrevious actions:" in llm.prompts[6]
    printed = capsys.readouterr().out
    assert "命中  2/2" in printed and "没有命中的" in printed and "[zh] demo/clean “点确定”" in printed


def test_cli_with_variants_leaves_the_clean_pictures_out_unless_asked(tmp_path):
    screens = make_screen(tmp_path / "screens")
    Image.new("RGB", (64, 32), (9, 9, 9)).save(tmp_path / "demo_cam.png")
    llm = ByInstruction({}, default=tc("click", coordinate=[750, 250]))
    argv = ["--screens", str(screens), "--variant", f"cam={tmp_path}/{{screen}}_cam.png", "--add_info", "none"]
    assert evaluate.main([*argv, "--out", str(tmp_path / "a")], llm=llm) == 0
    assert evaluate.main([*argv, "--out", str(tmp_path / "b"), "--clean"], llm=llm) == 0
    variants = [
        [s["variant"] for s in json.loads((tmp_path / d / "results.json").read_text(encoding="utf-8"))["summary"]]
        for d in "ab"
    ]
    assert variants == [["cam"], ["clean", "cam"]]
    with pytest.raises(SystemExit):
        evaluate.main(["--screens", str(screens), "--variant", "cam=no_placeholder.png"], llm=llm)


# -- the instruction suffix ----------------------------------------------------------------------------------------


def test_add_info_presets():
    assert gui_agent.resolve_add_info("zh") == gui_agent.ADD_INFO_ZH == gui_agent.DEFAULT_ADD_INFO
    assert gui_agent.resolve_add_info("en") == gui_agent.ADD_INFO_EN
    assert gui_agent.resolve_add_info("none") == "" == gui_agent.resolve_add_info("")
    assert gui_agent.resolve_add_info("别点广告") == "别点广告"
    for text in (gui_agent.ADD_INFO_ZH, gui_agent.ADD_INFO_EN):
        assert all(word in text for word in ("open", "type", "key"))
        # the wording that sent the models home when the target was already on the screen must not come back
        assert "请先回到主屏幕" not in text and "first go" not in text.lower()


@pytest.mark.parametrize(
    ("argv", "expected"), [([], gui_agent.ADD_INFO_ZH), (["--add_info", "en"], gui_agent.ADD_INFO_EN)]
)
def test_run_command_passes_the_chosen_add_info_to_the_agent(monkeypatch, argv, expected):
    made: list[dict] = []

    class RecordingAgent:
        def __init__(self, device, **options) -> None:
            made.append(options)

        def run(self, instruction: str) -> AgentResult:
            return AgentResult("success", 1)

    class Ctx:
        stop = threading.Event()

        def device(self):
            return types.SimpleNamespace(robot=types.SimpleNamespace(where=None))

        def confirm(self, what: str) -> None:
            pass

        @contextlib.contextmanager
        def interruptible(self):
            yield

    monkeypatch.setattr(run_command, "GuiAgent", RecordingAgent)
    command = run_command.Run()
    parser = argparse.ArgumentParser()
    command.configure(parser)
    command.run(Ctx(), parser.parse_args(["--instruction", "打开音乐", *argv]))
    assert len(made) == 1 and made[0]["add_info"] == expected
