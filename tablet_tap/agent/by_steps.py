"""An instruction that names several things to do, done one at a time.

GUI-Owl finds one thing on the screen well and keeps poor track of a list: given the whole sentence it repeated what was
done, did not stop, and swiped instead of leaving the app. Here the program keeps the list. Each clause is asked for on
its own, as one action; leaving the app and going back are keys of the tablet and the model is not asked at all. Measured
on 2026-11-24 on 16 cases made from camera pictures of the bench's music app (the cases the wording was chosen on): 14
right, against 4 to 7 with the whole sentence. Two things made the rest: how a button is called ("下一首按钮" 14 of 14
pictures, "向下的箭头" 12), and seeing for itself whether a clause came out: an app is open when a click has changed the
picture, and whether music plays is read off its button, which is also asked about on its own (14 pictures, all read
right), before pressing it and after. Only sentences whose every clause is a simple one are taken apart.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
import dataclasses
import json
import pathlib
import re

import cv2
import numpy as np
from PIL import Image
import requests

from tablet_tap.agent import gui_agent
from tablet_tap.agent.gui_agent import AgentResult
from tablet_tap.agent.gui_agent import GuiAgent

_SPLIT = re.compile(r"[，,。；;、\n]|然后|最后|并且|接着|之后|再(?=[切播暂点打退返关换])")
_HOME = re.compile(r"^(请)?(退出|返回主页|返回桌面|返回主屏幕|回到?主页|回到?桌面|回到?主屏幕|关闭(音乐|应用)$)")
_BACK = re.compile(r"^(请)?(返回(上一页|上一级|上一层)?|后退)$")
_SIMPLE = re.compile(r"^(请)?(打开|播放|放|暂停|停止播放|继续|切换|切到|切歌|换一?首|换首|换歌|下一首|上一首|点击|点一下|点开|点|关掉|关闭)")
# a song or a singer by name has to be looked for, and typed: not one press
_NAMED = re.compile(r"(请)?(播放|放)(?!(第一首|一首|一下)?(音乐|歌曲?|播放)?$|(上|下)一(首|曲)|第.{1,2}(首|个))")
_ELSE = re.compile(r"如果|直到|结束|输入|搜索|找到|调")  # needs judgement or several actions: the whole sentence
# Turning pages and scrolling are strokes of the pen, without the model. "翻" and "看" say where to get to, "滑" and "划" how
# the finger moves. A slow pen has no swing when it lifts: less than half the screen's width and the page springs back.
_NEXT_PAGE, _PREVIOUS_PAGE = (0.85, 0.5, 0.15, 0.5), (0.15, 0.5, 0.85, 0.5)
_FINGER_UP, _FINGER_DOWN = (0.5, 0.75, 0.5, 0.25), (0.5, 0.25, 0.5, 0.75)
_STROKES = (
    (re.compile(r"(往|向)?(上一页|前翻|前一页|左翻)|翻(到|回)?(上|前)一?页|(向|往)?右(滑|划)"), _PREVIOUS_PAGE),
    (re.compile(r"(往|向)?(下一页|后翻|后一页|右翻)|翻(到)?(下|后)一?页|翻\S{0,2}页|(向|往)?左(滑|划)"), _NEXT_PAGE),
    (re.compile(r"(往|向)?下(翻|看)|(往|向)?上(滑|划)"), _FINGER_UP),
    (re.compile(r"(往|向)?上(翻|看)|(往|向)?下(滑|划)"), _FINGER_DOWN),
)
_TIMES = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5}
_POPUP = re.compile(r"广告|弹窗|跳过|关闭按钮")
TRIES = 4  # actions for one clause: what covers the screen is closed, a press that did not come out is made again
OPENED_DIFF = 12.0  # mean difference (of 255) between the pictures before and after an app came up
# The cover and the song's name in the music app: of the bench's pictures of 2026-11-24 the same song differed by 5.1 at
# most (11 pairs), another song by 17.3 at least (6 pairs).
SONG_BOX, SONG_DIFF = (0.10, 0.08, 0.42, 0.68), 10.0
SONG_LOADS_S = 2.0
PLAY_BUTTON = (0.497, 0.87)  # of the music app: where it is looked at when the model has not said where it is
PLAYING, PAUSED = "两条竖线", "三角形"
# Asked of camera pictures of the bench's tablet on 2026-11-24: 5 of 5 pop-ups seen, none in 40 pictures without one.
POPUP_QUESTION = "屏幕上是否有弹窗、广告或优惠券页面挡住了原来的界面？只回答“有”或“没有”。"
SCREEN_QUESTION = "这张平板截图属于哪一种？A 桌面（应用图标） B 应用的正常界面 C 弹窗盖在界面上 D 全屏广告或领券页面。只回答一个字母。"
CLOSE = "关闭弹窗或广告：点击它的关闭按钮（×）或“跳过”"
SPLASH_S = 5.0  # an app's opening advertisement goes by itself
POPUP_TRIES = 5

_TO_OPEN = gui_agent.ADD_INFO_ZH.split("任务完成或")[0]
_IN_APP = (
    "本设备由机械臂夹着电容笔点按屏幕来操作，没有 open、type、key 这三种动作。"
    "需要的应用已经打开，就在当前屏幕上操作，不要回主屏幕。"
)


def _stroke(text: str) -> tuple[tuple[float, float, float, float], int] | None:
    """The stroke (u1, v1, u2, v2) ``text`` asks for, and how many times."""
    if len(text) > 12 or not re.search(r"翻|滑|划|下一页|上一页|(往|向)[上下]看", text):
        return None
    for pattern, stroke in _STROKES:
        if pattern.search(text):
            count = re.search(r"([一两二三四五1-5])(次|下|页|遍)", text)
            number = 1 if count is None or re.search(r"[上下前后]一页", text) else _TIMES.get(count.group(1)) or int(count.group(1))
            return stroke, number
    return None


def _song_differs(before: Image.Image, after: Image.Image) -> bool:
    parts = []
    for image in (before, after):
        picture = cv2.GaussianBlur(np.asarray(image.convert("RGB").resize(before.size)), (0, 0), 2.0).astype(np.float32)
        h, w = picture.shape[:2]
        u1, v1, u2, v2 = SONG_BOX
        parts.append(picture[int(v1 * h) : int(v2 * h), int(u1 * w) : int(u2 * w)])
    return float(np.abs(parts[0] - parts[1]).mean()) > SONG_DIFF


def app_names(path: pathlib.Path) -> dict[str, str]:
    """What is said -> what the icon is called."""
    try:
        return {str(k): str(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    except (OSError, ValueError, AttributeError):
        return {}


@dataclasses.dataclass(frozen=True)
class Clause:
    text: str  # as it was said
    ask: str  # what the model is asked for
    key: str = ""  # "home" / "back" / "swipe": the tablet's key or a stroke of the pen, without the model
    note: str = ""
    stroke: tuple[float, float, float, float] | None = None
    kind: str = ""  # "open": done when the picture has changed; "play" / "pause": the state the button has to show; "song": another one


def _clause(text: str, in_app: bool, names: dict[str, str]) -> Clause:
    opened = re.match(r"(?:请)?打开(.+)$", text)
    if opened:
        name = opened.group(1).strip()
        return Clause(text, "打开" + names.get(name, name), "", _TO_OPEN, kind="open")
    if re.search(r"上一(首|曲)", text):
        return Clause(text, "点击屏幕右侧竖条里上面的“上一首”按钮", "", _IN_APP, kind="song")
    if re.search(r"下一(首|曲)|切换|切歌|切到|换一?首|换歌", text):
        return Clause(text, "点击屏幕右侧竖条里下面的“下一首”按钮", "", _IN_APP, kind="song")
    if re.match(r"(请)?(暂停|停止播放)", text):
        return Clause(text, "点击屏幕底部中间的暂停按钮，暂停播放", "", _IN_APP, kind="pause")
    if re.match(r"(请)?(播放|继续|放)(第一首|一首|一下)?(音乐|歌曲?|播放)?$", text):
        return Clause(text, "点击屏幕底部中间的播放按钮，开始播放", "", _IN_APP, kind="play")
    return Clause(text, text, "", _IN_APP if in_app else _TO_OPEN)


def split(instruction: str, names: dict[str, str] | None = None) -> list[Clause] | None:
    """The clauses of ``instruction``; None when it is not made of simple ones. ``names``: see ``app_names``."""
    texts = [x.strip() for x in _SPLIT.split(instruction) if x and x.strip()]
    texts = [part for text in texts for part in (re.match(r"(打开.+?)(播放(?!器).+)$", text) or re.match(r"(.+)", text)).groups()]  # two in one
    if not texts or _ELSE.search(instruction):
        return None
    clauses, in_app = [], False
    for text in texts:
        if _HOME.match(text):
            clauses.append(Clause(text, "", "home"))
            in_app = False
        elif (found := _stroke(text)) is not None:
            clauses += [Clause(text, "", "swipe", stroke=found[0])] * found[1]
        elif _BACK.match(text):
            clauses.append(Clause(text, "", "back"))
        elif _SIMPLE.match(text) and not _NAMED.match(text):
            clauses.append(_clause(text, in_app, names or {}))
            in_app = in_app or clauses[-1].kind == "open"
        else:
            return None
    return clauses


class StepAgent(GuiAgent):
    """``GuiAgent`` that goes through ``clauses``. ``look`` answers a question about a picture (tests); by default the
    model server does."""

    def __init__(self, *args, clauses: list[Clause], look: Callable[[np.ndarray, str], str] | None = None, **kwargs) -> None:  # noqa: ANN002, ANN003
        super().__init__(*args, **kwargs)
        self.clauses, self.look = clauses, look or self._ask_server

    def _ask_server(self, image_bgr: np.ndarray, question: str) -> str:
        picture = base64.b64encode(cv2.imencode(".png", image_bgr)[1]).decode()
        content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + picture}}, {"type": "text", "text": question}]
        with requests.Session() as session:
            session.trust_env = False  # the server is on this machine: no proxy
            reply = session.post(self.base_url.rstrip("/") + "/chat/completions", timeout=20, json={
                "model": self.model, "temperature": 0, "max_tokens": 40, "messages": [{"role": "user", "content": content}]})  # fmt: skip
        reply.raise_for_status()
        return str(reply.json()["choices"][0]["message"]["content"])

    def _button(self, image: Image.Image, at: tuple[float, float]) -> str:
        """What the play button at ``at`` shows: PLAYING, PAUSED or "" (cannot be told)."""
        bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        x, y = np.clip(at[0], 0.1, 0.9), np.clip(at[1], 0.085, 0.915)
        part = bgr[round((y - 0.085) * h) : round((y + 0.085) * h), round((x - 0.1) * w) : round((x + 0.1) * w)]
        part = cv2.resize(part, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        try:
            said = self.look(part, "图中央的圆形按钮，里面是“三角形”还是“两条竖线”？只回答其中一个。")
        except (requests.RequestException, KeyError, ValueError) as exc:
            gui_agent.log.warning("看播放键：%s", exc)
            return ""
        shown = [x for x in (PLAYING, PAUSED) if x in said]
        return shown[0] if len(shown) == 1 else ""

    def _question(self, image: Image.Image, question: str) -> str:
        try:
            return self.look(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR), question).strip()
        except (requests.RequestException, KeyError, ValueError) as exc:
            gui_agent.log.warning("看弹窗：%s", exc)
            return ""

    def _clear(self, opened: Clause | None) -> tuple[str, str]:
        """Get what covers the screen out of the way: (status, answer), status "" when the screen is free. First the Back
        key, which cannot buy anything; then the pop-up's own close button; then a wait. Where Back led out of the app
        it is opened again."""
        for attempt in range(POPUP_TRIES):
            image = self.device.screenshot().convert("RGB")
            if not self._question(image, POPUP_QUESTION).startswith("有"):
                if attempt and opened is not None and self._question(image, SCREEN_QUESTION).upper().startswith("A"):
                    status, answer, _ = self._step(opened, "关掉弹窗时退到了桌面，重新打开", 0)
                    if status:
                        return status, answer
                    continue
                return "", ""
            if self._stop_requested():
                return "stopped", ""
            if attempt % 3 == 0:
                clause = Clause("返回", "", "back")
            elif attempt % 3 == 1:
                clause = Clause("关闭弹窗", CLOSE, "", _IN_APP, kind="close")
            else:
                gui_agent.log.info("有弹窗或广告，等 %.0f 秒", SPLASH_S)
                if self._sleep(SPLASH_S):
                    return "stopped", ""
                continue
            status, answer, _ = self._step(clause, "有弹窗或广告挡着，先关掉", 0)
            if status in ("stopped", "llm_error", "max_steps"):
                return status, answer
        return "failure", "弹窗或广告关不掉，请手动关掉再执行。"

    def _step(self, clause: Clause, where: str, attempt: int) -> tuple[str, str, str]:
        if self._stop_requested():
            return "stopped", "", "again"
        if self._steps >= self.max_steps:
            return "max_steps", "步数用完。", "again"
        self._steps += 1
        return self._one(self._steps, clause, where, attempt)

    def _loop(self, prompt: str) -> AgentResult:  # noqa: ARG002 - the clauses' own prompts
        def done(status: str, answer: str = "") -> AgentResult:
            return AgentResult(status=status, steps=self._steps, answer=answer, run_dir=str(self.run_dir))

        unsure: list[str] = []
        opened: Clause | None = None  # the app the instruction is in
        for number, clause in enumerate(self.clauses, 1):
            where = f"第 {number}/{len(self.clauses)} 件“{clause.text}”"
            for attempt in range(TRIES):
                if clause.key != "home":  # which leaves whatever is in front
                    status, answer = self._clear(opened)
                    if status:
                        return done(status, answer if status == "stopped" else f"停在{where}：{answer}")
                status, answer, outcome = self._step(clause, where, attempt)
                if status in ("success", "answer"):
                    says_so = self._claimed(clause)
                    if says_so is False:
                        if self._sleep(self.settle_s):
                            return done("stopped")
                        continue
                    if says_so is None:
                        unsure.append(clause.text)
                    break
                if status == "stopped":
                    return done(status)
                if status:
                    return done(status, f"{where}没有做成。{answer}")
                if outcome != "again" and self._sleep(self.settle_s):
                    return done("stopped")
                if outcome == "done":
                    break
            else:
                return done("failure", f"{where}：试了 {TRIES} 次没有做成，后面的没有做。")
            opened = clause if clause.kind == "open" else None if clause.key == "home" else opened
        return done("success", f"{len(self.clauses)} 件事都做了：" + "、".join(c.text for c in self.clauses)
                    + (f"（没有核对：{'、'.join(unsure)}）" if unsure else ""))  # fmt: skip

    def _claimed(self, clause: Clause) -> bool | None:
        """The model says that ``clause`` is done and pressed nothing: what the screen says of it, None when that cannot
        be told. On 2026-11-24 it said of paused music that it was playing."""
        image = self.device.screenshot().convert("RGB")
        if clause.kind in ("play", "pause"):
            shown = self._button(image, PLAY_BUTTON)
            return None if shown == "" else shown == (PLAYING if clause.kind == "play" else PAUSED)
        if clause.kind == "open" and self._question(image, SCREEN_QUESTION).upper().startswith("A"):
            return False  # the desktop
        return False if clause.kind == "song" else None  # a song is not another one without a press

    def _came_out(self, clause: Clause, before: Image.Image, at: tuple[float, float]) -> bool:
        """After the press: whether the picture says that ``clause`` is done."""
        if self._sleep(0.8 if clause.kind == "open" else 0.0):
            return True
        after = self.device.screenshot().convert("RGB")
        if clause.kind == "open":
            return float(np.abs(np.asarray(after, np.float32) - np.asarray(before, np.float32)).mean()) > OPENED_DIFF
        if clause.kind == "song":
            if _song_differs(before, after):
                return True
            if self._sleep(SONG_LOADS_S):
                return True
            return _song_differs(before, self.device.screenshot().convert("RGB"))
        shown = self._button(after, at)
        return shown == "" or shown == (PLAYING if clause.kind == "play" else PAUSED)

    def _one(self, step: int, clause: Clause, where: str, attempt: int) -> tuple[str, str, str]:
        """One action for ``clause``: (status, answer, "done" or "again")."""
        image = self.device.screenshot().convert("RGB")
        image_path = self.run_dir / f"step_{step:02d}.png"
        image.save(image_path)
        record = {"step": step, "time": gui_agent._now(), "image": image_path.name, "image_size": list(image.size), "output": "",  # noqa: SLF001
                  "attempts": 0, "action": None, "coords": {}, "executed": None, "annotated": "", "llm_s": 0.0, "device_s": 0.0,
                  "status": "", "answer": "", "note": where}  # fmt: skip
        if clause.stroke is not None:
            u1, v1, u2, v2 = clause.stroke
            args = {"action": "swipe", "coordinate": [1000 * u1, 1000 * v1], "coordinate2": [1000 * u2, 1000 * v2]}
            record["output"] = f"Action: {where}：滑动"
        elif clause.key:
            args = {"action": "system_button", "button": "Home" if clause.key == "home" else "Back"}
            record["output"] = f"Action: {where}：按{'主页' if clause.key == 'home' else '返回'}键"
        else:
            again = "（还没有做成）" if attempt and clause.kind in ("play", "pause", "song") else ""
            action, status = self._ask_model(str(image_path), f"{clause.ask}{again} ({clause.note})", [], record)
            if action is None:
                record["status"], record["answer"] = status, "" if status == "stopped" else record["note"]
                self._emit(record)
                return status, record["answer"], "again"
            args = action["arguments"]
            record["note"] = where
        record["action"] = dict(args)
        record["coords"] = {k: list(gui_agent.to_uv(args[k])) for k in gui_agent.COORDINATE_KEYS if k in args}
        click = args["action"] == "click"
        if clause.kind == "close" and (not click or not gui_agent.CLOSES.search(gui_agent.thought(record["output"]))):
            record["note"] = f"{where}：模型没有找到关闭按钮"
            self._emit(record)
            return "", "", "again"
        popup = not clause.key and bool(_POPUP.search(gui_agent.thought(record["output"]))) and not _POPUP.search(clause.text)
        if clause.kind in ("play", "pause") and click and not popup:
            want = PLAYING if clause.kind == "play" else PAUSED
            if self._button(image, tuple(record["coords"]["coordinate"])) == want:
                record["note"] = f"{where}：播放键已经是{'正在播放' if clause.kind == 'play' else '暂停'}，不用点"
                self._emit(record)
                gui_agent.log.info("[步骤 %02d] %s", step, record["note"])
                return "", "", "done"
        started = gui_agent.time.monotonic()
        try:
            status, answer = self._dispatch(args, record["output"], record, [])
        except BaseException as exc:
            record["device_s"] = round(gui_agent.time.monotonic() - started, 3)
            record["status"], record["note"] = "exception", f"执行动作时出错：{type(exc).__name__}: {exc}"
            self._emit(record)
            raise
        record["device_s"] = round(gui_agent.time.monotonic() - started, 3)
        gui_agent.log.info("[步骤 %02d] %s：%s", step, where, gui_agent.thought(record["output"]))
        outcome = "done"
        if status:
            record["status"], record["answer"] = status, answer
        else:
            record["annotated"] = self._annotate(image_path, args, image.size)
            if popup or (clause.kind in ("open", "play", "pause", "song") and not click):
                outcome = "again"  # it cleared the way, or looked for the icon on another page
            elif clause.kind in ("open", "play", "pause", "song") and not self._came_out(clause, image, tuple(record["coords"]["coordinate"])):
                outcome = "again"
                record["note"] = f"{where}：点了没有反应，再试"
        self._emit(record)
        return status, answer, outcome
