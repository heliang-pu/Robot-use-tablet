"""The agent loop between a Device (screenshot + touches) and GUI-Owl-1.5.

The loop is upstream's ``run_gui_owl_1_5_for_mobile.main`` with adb replaced by a Device: upstream's prompt, message
format, history, action parsing, parse retries and annotation are used unchanged through ``tablet_tap.agent.upstream``.
What differs is listed in ``GuiAgent``.

Offline use on recorded screenshots, without a robot::

    bin/py -m tablet_tap.agent.gui_agent --images <dir-or-files> --instruction "打开音乐"
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import copy
import dataclasses
import datetime
import json
import logging
import os
import pathlib
import re
import time
from typing import Any, Protocol

from tablet_tap.agent import upstream as up
from tablet_tap.agent.device import Device
from tablet_tap.agent.device import FakeDevice
from tablet_tap.errors import UnsupportedAction

log = logging.getLogger(__name__)

# The text appended to the instruction (upstream's --add_info). Its wording changes what the model does, and the two
# model sizes react differently. Hits out of 45 single steps on tests/data/screens and their two simulated camera
# renderings, measured on 2026-11-23 the way `python -m tablet_tap.agent.evaluate` now does it (synthetic screens, so
# only a guide; README "模型和附加说明怎么选" has the numbers per picture quality):
#
#     text                              8B    2B
#     nothing                           27    38     the 8B then asks for `open` 9 times, which a pen cannot do
#     ADD_INFO_ZH                       37    30
#     ADD_INFO_EN                        ?    37     not measured on the 8B yet
#     the wording used until then       30    23     "要打开应用，请先回到主屏幕，再点击它的图标": it sent the models
#                                                    to the home screen when the target was already in view
ADD_INFO_ZH = (
    "本设备由机械臂夹着电容笔点按屏幕来操作，没有 open、type、key 这三种动作。"
    "要打开应用，就点击屏幕上它的图标；当前屏幕上没有这个图标时，再回主屏幕去找。"
    "任务完成或确认无法完成时，用 terminate 结束。"
)
ADD_INFO_EN = "The open, type and key actions are not available on this device. To open an app, click its icon."
# When the on-screen keyboard is calibrated the device can type (agent/keyboard.py): Chinese goes out as pinyin and the
# model picks the candidate. Not measured yet.
ADD_INFO_TYPING_ZH = (
    "本设备由机械臂夹着电容笔点按屏幕来操作，没有 open、key 这两种动作。"
    "要输入文字：先点输入框让键盘出来，再用 type 输入；中文会按拼音打出来，打完以后在键盘上方的候选词里点选正确的词，"
    "再点搜索或确定。没打好的字可以点退格。"
    "要打开应用，就点击屏幕上它的图标；当前屏幕上没有这个图标时，再回主屏幕去找。"
    "任务完成或确认无法完成时，用 terminate 结束。"
)
DEFAULT_ADD_INFO = ADD_INFO_ZH  # the default model of scripts/serve_vlm.sh is the 8B


def add_info_for(*, typing: bool) -> str:
    """The note that fits a device that can, or cannot, type."""
    return ADD_INFO_TYPING_ZH if typing else DEFAULT_ADD_INFO


ADD_INFO_PRESETS = {"zh": ADD_INFO_ZH, "en": ADD_INFO_EN, "none": ""}


def not_for_the_arm(said: str) -> str:
    """Why the arm does not press what the model says it is about to press; "" when it does. A button that closes or
    skips is pressed also when the words around it speak of money; that something else is pressed "in order to close"
    does not make it one."""
    said = said.removeprefix("Action:").strip()
    parts = [p for p in _PURPOSE.split(said) if p]
    target = (parts[0] if parts else said).strip()
    closes = bool(_CLOSE_CONTROL.search(target))
    if COSTS.search(said) and not closes:
        return "模型要点的东西可能要花钱、授权或下载"
    if len(parts) > 1 and _POPUP_OFF.search(said[len(target):]) and not closes and not CLOSES.search(target):
        return "模型为了关掉弹窗或广告，要点的却不是关闭键"
    return ""


def resolve_add_info(value: str) -> str:
    """``zh`` / ``en`` / ``none`` name the texts above; anything else is taken as the text itself."""
    return ADD_INFO_PRESETS.get(value, value)


PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]
COORDINATE_KEYS = ("coordinate", "coordinate1", "coordinate2")  # the keys upstream's rescale_coordinates touches
MODEL_COORD_RANGE = 1000.0  # the model's coordinates are 0-1000 over the image, whatever its aspect ratio
# What the model says it is about to press: never pressed by the arm, the operator does that (on 2026-11-24 it pressed
# "6元开通", a subscription, to get a pop-up out of the way).
COSTS = re.compile(
    r"开通|会员|续费|付款|付费|支付|购买|下单|充值|订阅|结算|转账|红包|提现|优惠券|立即使用|立即领取|立减|下载|安装|授权|验证码|密码|登录|登陆|手机号|注册|身份证|\d\s*元|[¥￥]"
    r"|\bVIP\b|subscri|purchas|\bpay|\bbuy\b|checkout|renew|\bmember|coupon|claim|download|install|password|log ?in|sign ?(in|up)|phone number|verif",
    re.IGNORECASE,
)
CLOSES = re.compile(r"关闭|跳过|取消|返回|拒绝|稍后|不再|\bclose|\bskip|\bcancel|\bdismiss", re.IGNORECASE)
# What is pressed and what for are told apart: on 2026-11-24 the model wanted "点击弹窗中的绿色按钮以关闭广告", and the green
# button was "去看视频".
_PURPOSE = re.compile(r"以便?(?=关|跳|取消|返回|退|尝试|解除|清除|移除)|，|,|来(?=关|跳)|从而|\bto (?=close|skip|dismiss|cancel|get|try|remove|clear)")
_POPUP_OFF = re.compile(
    r"(关闭|关掉|跳过|清除|移除|解除|消除).{0,12}(广告|弹窗|弹出|浮层|优惠|会员|遮挡|视频)|\b(close|skip|dismiss|remove|clear)\b.{0,30}\b(ad|ads|advert\w*|pop-?up|dialog|overlay|video)\b",
    re.IGNORECASE,
)
_CLOSE_CONTROL = re.compile(
    r"(关闭|跳过|取消|返回|拒绝|稍后|以后再说|不再提醒|叉号?|空白|箭头|[×✕xX])[”\"」』）)]?(的)?(按钮|图标|键|选项|文字|链接|区域|处|标志|符号)?([（(][^）)]*[）)])?[。.\s]*$"
    r"|\b(close|skip|cancel|dismiss|x)\b[\"”')]*\s*(button|icon|option|link|control|mark)\b|[\"“'](close|skip|cancel|dismiss|x|×|关闭|跳过|取消)[\"”']",
    re.IGNORECASE,
)
STUCK_TOLERANCE = 15.0  # two clicks closer than this (in 0-1000 units, both axes) count as the same click
MAX_WAIT_S = 10.0
DEFAULT_WAIT_S = 2.0  # upstream: action_parameter.get("time", 2)
POLL_S = 0.1
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

# Upstream's smart_resize arguments in the run script (step "4. Rescale coordinates").
_RESIZE_KWARGS = {"factor": 16, "min_pixels": 3136, "max_pixels": 1003520 * 200}


@dataclasses.dataclass
class AgentResult:
    """How a run ended.

    status: "success" | "failure" (the model terminated with that status, or the stuck-click guard) | "answer" |
    "need_user" | "unsupported" | "max_steps" | "parse_error" | "llm_error" | "stopped".
    """

    status: str
    steps: int
    answer: str = ""  # text for "answer" / "need_user", detail for the error statuses
    run_dir: str = ""


class Llm(Protocol):
    """Anything shaped like upstream's GUIOwlWrapper."""

    def predict_mm(self, messages: list[dict]) -> tuple[str, Any, Any]:
        """Return (text, payload, raw response) for upstream-format messages."""
        ...


class GuiOwlLlm(up.GUIOwlWrapper):
    """Upstream's GUIOwlWrapper, made to accept what upstream's build_messages produces.

    build_messages writes images as ``"file://" + path`` and the wrapper hands that string to ``Image.open``, which
    fails (FileNotFoundError, outside the wrapper's own try/except). The prefix is removed here before upstream's
    conversion runs. ``max_retry`` and ``retry_wait_s`` are upstream's knobs (default 10 tries, 20 s apart); the wait
    is set on the instance, the class default stays.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        *,
        max_retry: int | None = None,
        retry_wait_s: float | None = None,
    ) -> None:
        if max_retry is None:
            super().__init__(api_key, base_url, model_name)
        else:
            super().__init__(api_key, base_url, model_name, max_retry=max_retry)
        if retry_wait_s is not None:
            self.RETRY_WAITING_SECONDS = retry_wait_s
        # the same picture and the same words give the same action; upstream's requests leave it to the server to sample
        completions = getattr(getattr(getattr(self, "bot", None), "chat", None), "completions", None)
        if completions is not None:
            create = completions.create
            completions.create = lambda **kwargs: create(**{"temperature": 0, **kwargs})

    def convert_messages_format_to_openaiurl(self, messages: list[dict]) -> list[dict]:
        """Strip ``file://`` from the image items, then convert as upstream does."""
        local = []
        for message in messages:
            content = []
            for item in message["content"]:
                image = item.get("image")
                if isinstance(image, str) and image.startswith("file://"):
                    item = {**item, "image": image[len("file://") :]}
                content.append(item)
            local.append({**message, "content": content})
        return super().convert_messages_format_to_openaiurl(local)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_arguments(args: dict) -> None:
    """Raise ValueError when the arguments cannot be dispatched (upstream would hit a KeyError / TypeError here)."""
    action = args["action"]
    required: tuple[str, ...] = ()
    if action in ("click", "long_press"):
        required = ("coordinate",)
    elif action in ("swipe", "scroll"):
        required = ("coordinate", "coordinate2")
    for key in required:
        if key not in args:
            raise ValueError(f"Action '{action}' needs '{key}': {json.dumps(args, ensure_ascii=False)}")
    for key in COORDINATE_KEYS:
        if key in args:
            value = args[key]
            if not (isinstance(value, list) and len(value) == 2 and all(_is_number(x) for x in value)):
                raise ValueError(f"'{key}' must be [x, y] with two numbers: {json.dumps(args, ensure_ascii=False)}")
    if action == "system_button" and not isinstance(args.get("button"), str):
        raise ValueError(f"Action 'system_button' needs 'button': {json.dumps(args, ensure_ascii=False)}")
    if action == "type" and not isinstance(args.get("text"), str):
        raise ValueError(f"Action 'type' needs 'text': {json.dumps(args, ensure_ascii=False)}")


def to_uv(coordinate: list[float]) -> tuple[float, float]:
    """Convert a model coordinate (0-1000 in both axes) to normalized screen coordinates (u, v).

    Upstream computes ``px = int(x / 1000 * resized_w)``, ``py = int(y / 1000 * resized_h)`` (rescale_coordinates) with
    ``(resized_h, resized_w) = smart_resize(img.height, img.width, factor=16, ...)``, i.e. the screenshot's own size
    rounded to multiples of 16, and sends (px, py) to adb as pixels of a screen that has the screenshot's size. The
    fraction of the screen is therefore px / width = x / 1000 and py / height = y / 1000, apart from upstream's integer
    truncation and its rounding of the size to 16 (at most 8 px). So u = x / 1000, v = y / 1000, with no dependence on
    the image size or aspect ratio. Values outside the screen are clamped.
    """
    x, y = coordinate
    return min(max(x / MODEL_COORD_RANGE, 0.0), 1.0), min(max(y / MODEL_COORD_RANGE, 0.0), 1.0)


def thought(output: str) -> str:
    """The text before the tool call (upstream's "conclusion"), on one line."""
    return " ".join(output.split(up.TOOL_CALL_OPEN)[0].split())


def _seconds(value: object, default: float) -> float:
    """A positive duration from the model's ``time`` argument, capped at MAX_WAIT_S."""
    if not _is_number(value) or value <= 0:
        return default
    return min(float(value), MAX_WAIT_S)


class GuiAgent:
    """Runs one instruction on a Device with GUI-Owl-1.5, following upstream's loop.

    Differences from upstream's adb loop:

    - coordinates go to the device as normalized (u, v) (see ``to_uv``);
    - ``open`` cannot launch an app without adb: the device goes home and the model has to tap the icon;
    - ``type`` may be refused by the device, ``key`` and the Menu / Enter buttons do not exist: the run ends with
      "unsupported"; ``interact`` ends the run with "need_user" instead of waiting on stdin;
    - the same click three times in a row ends the run: a physical tap can miss and upstream would spin to max_steps;
    - upstream's ``'Error calling LLM'`` string ends the run with "llm_error" instead of being parsed as an action;
    - arguments that upstream would crash on (no coordinate, no button) are retried like an unparsable output.
    """

    def __init__(
        self,
        device: Device,
        *,
        base_url: str = "http://127.0.0.1:8802/v1",
        model: str = "gui-owl",
        api_key: str = "EMPTY",
        run_dir: str | os.PathLike | None = None,
        max_steps: int = 20,
        max_parse_retries: int = 3,
        settle_s: float = 2.0,
        add_info: str = DEFAULT_ADD_INFO,
        llm: Llm | None = None,
        should_stop: Callable[[], bool] | None = None,
        on_step: Callable[[dict], None] | None = None,
        llm_max_retry: int | None = None,
        llm_retry_wait_s: float | None = None,
        stuck_clicks: int = 3,
    ) -> None:
        """``run_dir`` defaults to a new ``runs/<YYYYmmdd_HHMMSS>`` in the project for every ``run()``.

        An explicit ``run_dir`` is reused by every ``run()`` of this agent, whose files replace the earlier ones.

        ``llm`` is anything with ``predict_mm(messages) -> (text, payload, raw)``; by default upstream's wrapper, for
        which ``llm_max_retry`` / ``llm_retry_wait_s`` override upstream's 10 tries, 20 s apart. ``stuck_clicks=0``
        turns the repeated-click guard off.
        """
        self.device = device
        self.base_url = base_url
        self.model = model
        self.max_steps = max_steps
        self.max_parse_retries = max_parse_retries
        self.settle_s = settle_s
        self.add_info = add_info
        self.should_stop = should_stop
        self.on_step = on_step
        self.stuck_clicks = stuck_clicks
        self._run_dir_arg = run_dir
        self.run_dir = pathlib.Path(run_dir).resolve() if run_dir is not None else PROJECT_DIR / "runs"
        self._steps = 0
        if llm is None:
            llm = GuiOwlLlm(api_key, base_url, model, max_retry=llm_max_retry, retry_wait_s=llm_retry_wait_s)
        self.llm = llm

    # -- helpers ------------------------------------------------------------------------------------------------

    def _new_run_dir(self) -> pathlib.Path:
        """Create and return the directory of the next run."""
        if self._run_dir_arg is not None:
            run_dir = pathlib.Path(self._run_dir_arg).resolve()
            run_dir.mkdir(parents=True, exist_ok=True)
            return run_dir
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        for suffix in ("", *(f"_{i}" for i in range(2, 100))):
            run_dir = PROJECT_DIR / "runs" / (stamp + suffix)
            try:
                run_dir.mkdir(parents=True)
            except FileExistsError:
                continue
            return run_dir
        raise FileExistsError(f"no free run directory for {stamp} under {PROJECT_DIR / 'runs'}")

    def _stop_requested(self) -> bool:
        return bool(self.should_stop is not None and self.should_stop())

    def _sleep(self, seconds: float) -> bool:
        """Sleep in short slices; return True as soon as a stop is requested."""
        remaining = float(seconds)
        while remaining > 1e-9:
            if self._stop_requested():
                return True
            dt = min(POLL_S, remaining)
            time.sleep(dt)
            remaining -= dt
        return self._stop_requested()

    def _ask_model(self, image_path: str, prompt: str, history: list[dict], record: dict) -> tuple[dict | None, str]:
        """Upstream's query-and-parse loop. Returns (action, "") or (None, status) with the detail in the record."""
        max_attempts = self.max_parse_retries + 1
        for attempt in range(1, max_attempts + 1):
            if attempt > 1 and self._stop_requested():
                return None, "stopped"
            messages = up.build_messages(image_path, prompt, history, self.model)
            if attempt > 1:
                messages[-1]["content"].append({"text": up.FORMAT_RETRY_HINT})
            started = time.monotonic()
            try:
                output, _, _ = self.llm.predict_mm(messages)
            except Exception as exc:  # the model side must not take the robot loop down with it
                record["llm_s"] = round(record["llm_s"] + time.monotonic() - started, 3)
                record["attempts"] = attempt
                record["note"] = f"调用模型出错：{type(exc).__name__}: {exc}"
                return None, "llm_error"
            record["llm_s"] = round(record["llm_s"] + time.monotonic() - started, 3)
            record["attempts"] = attempt
            record["output"] = output
            if output == up.ERROR_CALLING_LLM:
                record["note"] = f"模型服务 {self.base_url} 调用失败（上游重试后仍返回 {up.ERROR_CALLING_LLM!r}）"
                return None, "llm_error"
            try:
                action = up.parse_action(output)
                _check_arguments(action["arguments"])
            except ValueError as exc:
                record["note"] = str(exc)
                log.warning(
                    "[步骤 %02d] 模型输出无法解析（第 %d/%d 次）：%s", record["step"], attempt, max_attempts, exc
                )
                continue
            record["note"] = ""
            return action, ""
        return None, "parse_error"

    def _annotate(self, image_path: pathlib.Path, args: dict, size: tuple[int, int]) -> str:
        """Draw the action the way upstream does; returns the file name, or "" when upstream draws nothing."""
        width, height = size
        resized_h, resized_w = up.smart_resize(height, width, **_RESIZE_KWARGS)
        pixel_args = up.rescale_coordinates(copy.deepcopy(args), resized_w, resized_h)
        anno_path = image_path.with_name(image_path.stem + "_anno.png")
        return anno_path.name if up.annotate_screenshot(str(image_path), pixel_args, str(anno_path)) else ""

    def _dispatch(self, args: dict, output: str, record: dict, clicks: list[list[float]]) -> tuple[str, str]:
        """Execute one action. Returns (status, answer); status "" means the run goes on."""
        action = args["action"]
        device = self.device
        said = thought(output)
        if action in ("click", "long_press") and (why := not_for_the_arm(said)):
            return "need_user", f"{why}，没有点：{said}"
        if action != "click":
            clicks.clear()

        if action == "click":
            x, y = args["coordinate"]
            if clicks and (abs(x - clicks[-1][0]) > STUCK_TOLERANCE or abs(y - clicks[-1][1]) > STUCK_TOLERANCE):
                clicks.clear()
            clicks.append([x, y])
            if self.stuck_clicks and len(clicks) >= self.stuck_clicks:
                return "failure", (
                    f"卡住了：模型连续 {len(clicks)} 次要点击同一位置 ({x:g}, {y:g})/1000，"
                    "前面的点按很可能没有被屏幕识别，已停止。"
                )
            u, v = record["coords"]["coordinate"]
            record["executed"] = {"call": "click", "u": u, "v": v}
            device.click(u, v)

        elif action == "long_press":
            u, v = record["coords"]["coordinate"]
            if _is_number(args.get("time")) and args["time"] > 0:
                duration = _seconds(args["time"], 0.0)
                record["executed"] = {"call": "long_press", "u": u, "v": v, "duration_s": duration}
                device.long_press(u, v, duration)
            else:  # no time given: the device's own default (upstream: 800 ms)
                record["executed"] = {"call": "long_press", "u": u, "v": v}
                device.long_press(u, v)

        elif action in ("scroll", "swipe"):
            u1, v1 = record["coords"]["coordinate"]
            u2, v2 = record["coords"]["coordinate2"]
            record["executed"] = {"call": "swipe", "u1": u1, "v1": v1, "u2": u2, "v2": v2}
            device.swipe(u1, v1, u2, v2)

        elif action == "system_button":
            button = args["button"]
            if button == "Back":
                record["executed"] = {"call": "back"}
                device.back()
            elif button == "Home":
                record["executed"] = {"call": "home"}
                device.home()
            else:
                return "unsupported", f"模型想按系统键 {button}，本设备只有 Back 和 Home。"

        elif action == "wait":
            seconds = _seconds(args.get("time", DEFAULT_WAIT_S), DEFAULT_WAIT_S)
            record["executed"] = {"call": "wait", "seconds": seconds}
            if self._sleep(seconds):
                return "stopped", ""

        elif action == "terminate":
            status = args.get("status", "unknown")
            if status == "success":
                return "success", ""
            return "failure", "" if status == "failure" else f"模型以状态 {status!r} 结束了任务。"

        elif action == "open":
            # No adb, so no way to launch a package: go home and let the model find and tap the icon.
            record["executed"] = {"call": "home", "reason": f"open {args.get('text', '')!r}"}
            log.info(
                "[步骤 %02d] 模型想直接打开应用 %r：没有 adb，改为回到主屏幕", record["step"], args.get("text", "")
            )
            device.home()

        elif action == "answer":
            # Upstream's extraction: the text before the tool call, not arguments["text"].
            conclusion = output.split(up.TOOL_CALL_OPEN)[0].strip()
            return "answer", conclusion or str(args.get("text", ""))

        elif action in ("call_user", "calluser", "interact"):
            return "need_user", str(args.get("text") or output.split(up.TOOL_CALL_OPEN)[0].strip())

        elif action == "type":
            text = args["text"]
            try:
                device.type_text(text)
            except UnsupportedAction as exc:
                return "unsupported", f"模型想输入文字 {text!r}，本设备无法输入（{exc}）。"
            record["executed"] = {"call": "type_text", "text": text}

        elif action == "key":
            return "unsupported", f"模型想按键 {args.get('text', '')!r}，本设备没有 key 动作。"

        else:  # upstream: warn and go on
            log.warning("[步骤 %02d] 不认识的动作类型 %r，已跳过", record["step"], action)
            record["note"] = f"不认识的动作类型 {action!r}，未执行"

        return "", ""

    def _emit(self, record: dict) -> None:
        """Append the step to actions.jsonl and hand it to on_step."""
        with open(self.run_dir / "actions.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.on_step is not None:
            try:
                self.on_step(record)
            except Exception:
                log.exception("on_step 回调出错（已忽略）")

    def _write_result(self, result: AgentResult, info: dict) -> None:
        payload = {**dataclasses.asdict(result), **info, "finished": _now()}
        with open(self.run_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")

    # -- the loop -----------------------------------------------------------------------------------------------

    def run(self, instruction: str) -> AgentResult:
        """Carry out ``instruction``; every step is logged in ``run_dir``."""
        self.run_dir = self._new_run_dir()
        (self.run_dir / "actions.jsonl").write_text("", encoding="utf-8")
        self._steps = 0
        # Upstream appends --add_info this way.
        prompt = f"{instruction} ({self.add_info})" if self.add_info else instruction
        info = {
            "instruction": instruction,
            "prompt": prompt,
            "model": self.model,
            "base_url": self.base_url,
            "max_steps": self.max_steps,
            "started": _now(),
        }
        log.info("开始任务：%s（最多 %d 步，记录在 %s）", instruction, self.max_steps, self.run_dir)
        try:
            result = self._loop(prompt)
        except BaseException as exc:  # a device fault or Ctrl-C: leave a trace, then let the caller see it
            detail = f"{type(exc).__name__}: {exc}"
            self._write_result(AgentResult("exception", self._steps, detail, str(self.run_dir)), info)
            raise
        self._write_result(result, info)
        log.info("任务结束：%s，共 %d 步%s", result.status, result.steps, f"｜{result.answer}" if result.answer else "")
        return result

    def _loop(self, prompt: str) -> AgentResult:
        history: list[dict] = []
        clicks: list[list[float]] = []  # the current run of same-place clicks, in model units

        def done(status: str, answer: str = "") -> AgentResult:
            return AgentResult(status=status, steps=self._steps, answer=answer, run_dir=str(self.run_dir))

        for step in range(1, self.max_steps + 1):
            if self._stop_requested():
                return done("stopped")
            self._steps = step

            image = self.device.screenshot().convert("RGB")
            image_path = self.run_dir / f"step_{step:02d}.png"
            image.save(image_path)
            record: dict = {
                "step": step,
                "time": _now(),
                "image": image_path.name,
                "image_size": list(image.size),
                "output": "",
                "attempts": 0,
                "action": None,
                "coords": {},
                "executed": None,
                "annotated": "",
                "llm_s": 0.0,
                "device_s": 0.0,
                "status": "",
                "answer": "",
                "note": "",
            }

            action, status = self._ask_model(str(image_path), prompt, history, record)
            if action is None:
                record["status"] = status
                record["answer"] = "" if status == "stopped" else record["note"]
                self._emit(record)
                if status != "stopped":
                    log.error("[步骤 %02d] %s：%s", step, status, record["note"])
                return done(status, record["answer"])

            output = record["output"]
            args = action["arguments"]
            record["action"] = copy.deepcopy(args)
            record["coords"] = {key: list(to_uv(args[key])) for key in COORDINATE_KEYS if key in args}
            log.info("[步骤 %02d] 模型（%.1f s）：%s", step, record["llm_s"], thought(output) or "（没有说明）")

            started = time.monotonic()
            try:
                status, answer = self._dispatch(args, output, record, clicks)
            except BaseException as exc:
                record["device_s"] = round(time.monotonic() - started, 3)
                record["status"] = "exception"
                record["note"] = f"执行动作时出错：{type(exc).__name__}: {exc}"
                self._emit(record)
                raise
            record["device_s"] = round(time.monotonic() - started, 3)
            record["status"], record["answer"] = status, answer
            log.info("[步骤 %02d] %s（%.1f s）", step, describe(args, record), record["device_s"])

            if status:
                self._emit(record)
                return done(status, answer)

            # Upstream's step 6: history, then the annotated screenshot.
            history.append({"output": output, "image": str(image_path)})
            record["annotated"] = self._annotate(image_path, args, image.size)
            self._emit(record)
            if self._sleep(self.settle_s):
                return done("stopped")

        return done("max_steps", f"已到最大步数 {self.max_steps}，任务没有结束。")


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def describe(args: dict, record: dict) -> str:
    """One Chinese line saying what was done with the model's action."""
    action = args["action"]
    executed = record["executed"]
    if record["status"]:
        return f"动作 {action} → 结束：{record['status']}" + (f"｜{record['answer']}" if record["answer"] else "")
    if executed is None:
        return f"动作 {action} → 未执行"
    call = executed["call"]
    if call == "click":
        return f"点击 (u={executed['u']:.3f}, v={executed['v']:.3f})"
    if call == "long_press":
        held = f"，按住 {executed['duration_s']:g} s" if "duration_s" in executed else ""
        return f"长按 (u={executed['u']:.3f}, v={executed['v']:.3f}){held}"
    if call == "swipe":
        return (
            f"滑动 (u={executed['u1']:.3f}, v={executed['v1']:.3f}) → (u={executed['u2']:.3f}, v={executed['v2']:.3f})"
        )
    if call == "back":
        return "返回键"
    if call == "home":
        return "主屏幕键" + (f"（代替 {executed['reason']}）" if "reason" in executed else "")
    if call == "wait":
        return f"等待 {executed['seconds']:g} s"
    if call == "type_text":
        return f"输入文字 {executed['text']!r}"
    return f"动作 {action}"


# -- offline CLI ------------------------------------------------------------------------------------------------


def _collect_images(items: list[str]) -> list[str]:
    """Expand directories into their image files (sorted by name); keep files in the order given."""
    paths: list[str] = []
    for item in items:
        path = pathlib.Path(item)
        if path.is_dir():
            # the agent's own annotated copies are not screenshots
            found = (p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES and not p.stem.endswith("_anno"))
            paths.extend(str(p) for p in sorted(found))
        elif path.is_file():
            paths.append(str(path))
        else:
            raise SystemExit(f"找不到图片或目录：{item}")
    if not paths:
        raise SystemExit("没有找到任何图片")
    return paths


def main(argv: list[str] | None = None) -> int:
    """Run the agent on recorded screenshots with a FakeDevice and print what the model would do."""
    parser = argparse.ArgumentParser(
        description="离线查看 GUI-Owl 在录好的屏幕照片上的动作：用 FakeDevice 按顺序给图（最后一张重复），不动机械臂。"
    )
    parser.add_argument("--images", nargs="+", required=True, help="屏幕照片：一个目录（按文件名排序）或若干图片文件")
    parser.add_argument("--instruction", required=True, help="任务指令，例如 “打开音乐”")
    parser.add_argument("--base_url", default="http://127.0.0.1:8802/v1", help="vLLM 服务地址（OpenAI 接口）")
    parser.add_argument("--model", default="gui-owl", help="vLLM 里的模型名")
    parser.add_argument("--api_key", default="EMPTY", help="服务的 api key")
    parser.add_argument("--max_steps", type=int, default=None, help="最多几步；默认等于图片张数")
    parser.add_argument("--run_dir", default=None, help="记录目录；默认 runs/<年月日_时分秒>")
    parser.add_argument(
        "--add_info",
        default="zh",
        help="附加在指令后面的说明：zh（默认）/ en（2B 模型用它更准）/ none，或直接给一段文字；空字符串也是不加",
    )
    parser.add_argument("--settle_s", type=float, default=0.0, help="每步之后等待的秒数（离线默认 0）")
    parser.add_argument("--llm_max_retry", type=int, default=None, help="模型服务调用失败时的重试次数（上游默认 10）")
    parser.add_argument("--llm_retry_wait_s", type=float, default=None, help="两次重试之间等待的秒数（上游默认 20）")
    parser.add_argument(
        "--allow_type", action="store_true", help="让 FakeDevice 接受 type 动作（默认像电容笔一样拒绝）"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    images = _collect_images(args.images)
    device = FakeDevice(images, can_type=args.allow_type)
    agent = GuiAgent(
        device,
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        run_dir=args.run_dir,
        max_steps=args.max_steps if args.max_steps is not None else len(images),
        settle_s=args.settle_s,
        add_info=resolve_add_info(args.add_info),
        llm_max_retry=args.llm_max_retry,
        llm_retry_wait_s=args.llm_retry_wait_s,
    )
    result = agent.run(args.instruction)

    print(f"\n图片 {len(images)} 张，设备收到的动作 {len(device.actions)} 个：")
    for index, action in enumerate(device.actions, 1):
        print(f"  {index:2d}. {json.dumps(action, ensure_ascii=False)}")
    print(f"结果：{result.status}，{result.steps} 步" + (f"｜{result.answer}" if result.answer else ""))
    print(f"记录：{result.run_dir}（actions.jsonl、result.json、step_NN.png、step_NN_anno.png）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
