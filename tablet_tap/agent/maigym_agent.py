"""原生端到端多步视觉代理（基于 Tongyi-MAI/MAI-Gym-9B，兼容真实机械臂 Device 协议）。

采用「滑动窗口双图（上一帧 + 当前帧）+ 历史截图折叠 + 极简一行内生状态账本」范式：
  - 无需外部 Planner 或写死页面导航规则，模型每步看实时画面自主跨页决策；
  - 保留最近 2 张真实截图，天然具备物理漏点对比重试与列表滚底拒答能力；
  - 关闭长思考（enable_thinking=False），在 <tool_call> 前输出 15~35 字“状态：...”记忆行，
    避开 Qwen3.5 chat_template 剥离历史 <think> 的限制，兼顾低延迟与跨多页进度追踪；
  - 直接复用 tablet_tap.agent.device.Device 接口，可在仿真 Playwright 与真机 PiPER 机械臂间零改动切换。

离线回放示例：
    bin/py -m tablet_tap.agent.maigym_agent --images <目录或图片> --instruction "把主驾温度调到21度"
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Callable
import dataclasses
import io
import json
import logging
import pathlib
import re
import time
from typing import Any

from PIL import Image
import requests

from tablet_tap.agent.device import Device
from tablet_tap.agent.device import FakeDevice

log = logging.getLogger(__name__)


@dataclasses.dataclass
class AgentResult:
    """与 tablet_tap.agent.gui_agent.AgentResult 字段同构，但无需引入 upstream/openai 依赖。"""

    status: str
    steps: int
    answer: str = ""
    run_dir: str = ""

COORD_SCALE = 999.0

# 100% 沿用 Tongyi-MAI/MAI-Gym 官方 mobile_use 系统提示词，保证与基座预训练分布零漂移
MAI_GYM_SYSTEM_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
For each function call, return the thinking process in <think> </think> tags, and a json object with function name and arguments within <tool_call></tool_call> XML tags:
```
<think>
...
</think>
<tool_call>
{"name": "mobile_use", "arguments": <args-json-object>}
</tool_call>
```

## Action Space
{"action_type": "click", "coordinate": [x, y]}
{"action_type": "double_click", "coordinate": [x, y]}
{"action_type": "long_press", "coordinate": [x, y]}
{"action_type": "type", "text": ""}
{"action_type": "open", "text": "app_name"}
{"action_type": "drag", "start_coordinate": [x1, y1], "end_coordinate": [x2, y2]}
{"action_type": "system_button", "button": "button_name"}
{"action_type": "wait", "waiting_time": "xxx"}
{"action_type": "ask_user", "text": "xxx"}
{"action_type": "terminate", "answer": "xxx", "status": "success or failed"}
## Note
- Available Apps: `[]`.
- You should use the `open` action to open the app as possible as you can, because it is the fast way to open the app.
- You must follow the Action Space strictly, and return the correct json object within <think> </think> and <tool_call></tool_call> XML tags."""

# 机械臂纯视觉电容笔操作约束与多步内生状态记忆提示
MULTISTEP_NOTE = (
    "本设备由机械臂夹着电容笔操作理想 L8 车机屏幕，只可使用 click、drag、ask_user、terminate 动作，不要使用 open/type/system_button。"
    "请先用一行“状态：...”简述当前页面、已完成项与下一步，再输出 <tool_call> 执行一步操作；"
    "若任务含多个目标可自行跨页依次完成，全部达成后输出 terminate(status=\"success\")；"
    "若遇到恢复出厂设置、格式化、关闭AEB、解绑账号等不可逆高危操作，不要点击任何按钮，直接输出 ask_user；"
    "若列表滚动到底仍找不到目标联系人/条目，输出 terminate(status=\"failed\")。"
)

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_COORD_RE = re.compile(r'"coordinate"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]')


def img_to_b64(img: Image.Image) -> str:
    """将 PIL 图像编码为 PNG base64 字符串。"""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def to_uv(coord: list[float] | tuple[float, float]) -> tuple[float, float]:
    """将 MAI-Gym 0..999 坐标归一化到设备坐标 u, v ∈ [0, 1]。"""
    return (
        max(0.0, min(1.0, float(coord[0]) / COORD_SCALE)),
        max(0.0, min(1.0, float(coord[1]) / COORD_SCALE)),
    )


def parse_maigym_action(text: str) -> dict[str, Any]:
    """解析模型输出中的原生 MAI-Gym `<tool_call>` JSON 动作字典。"""
    match = _TOOL_CALL_RE.search(text)
    if match:
        raw = match.group(1).strip()
        try:
            obj = json.loads(raw)
            args = obj.get("arguments", obj) if isinstance(obj, dict) else {}
            if isinstance(args, str):
                args = json.loads(args)
            if isinstance(args, dict):
                return args
        except Exception:
            pass
    coord_m = _COORD_RE.search(text)
    if coord_m:
        return {"action_type": "click", "coordinate": [float(coord_m.group(1)), float(coord_m.group(2))]}
    return {}


def build_multistep_messages(
    instruction: str,
    screenshots_b64: list[str],
    assistant_turns: list[str],
    *,
    keep_last_images: int = 2,
    note: str = MULTISTEP_NOTE,
) -> list[dict[str, Any]]:
    """构造滑动窗口多轮多模态消息列表。

    仅保留最近 `keep_last_images` 帧真实图像（默认上一帧 + 当前帧），更早帧折叠为 `[历史截图]`，
    同时保留每轮助手输出中的 `状态：...` 与 `<tool_call>` 形成内生进度账本。
    """
    prompt_text = f"{instruction} ({note})" if note else instruction
    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": MAI_GYM_SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
    ]
    keep_from = max(0, len(screenshots_b64) - keep_last_images)
    for idx, b64 in enumerate(screenshots_b64):
        if idx >= keep_from:
            msgs.append(
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}],
                }
            )
        else:
            msgs.append({"role": "user", "content": [{"type": "text", "text": "[历史截图]"}]})
        if idx < len(assistant_turns):
            msgs.append({"role": "assistant", "content": assistant_turns[idx]})
    return msgs


@dataclasses.dataclass
class StepEvent:
    step: int
    raw: str
    action: dict[str, Any]


class MaiGymMultiStepAgent:
    """基于 MAI-Gym-9B（或微调版 carhmi-v2）的原生端到端多步视觉代理。"""

    def __init__(
        self,
        device: Device,
        *,
        base_url: str = "http://127.0.0.1:18809/v1",
        model: str = "mai-gym-9b-v2",
        api_key: str = "EMPTY",
        max_steps: int = 26,
        keep_last_images: int = 2,
        enable_thinking: bool = False,
        max_tokens: int = 256,
        settle_s: float = 0.5,
        note: str = MULTISTEP_NOTE,
        chat_fn: Callable[[list[dict[str, Any]]], str] | None = None,
        on_step: Callable[[StepEvent], None] | None = None,
    ) -> None:
        self.device = device
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_steps = max_steps
        self.keep_last_images = keep_last_images
        self.enable_thinking = enable_thinking
        self.max_tokens = max_tokens
        self.settle_s = settle_s
        self.note = note
        self._chat_fn = chat_fn
        self.on_step = on_step
        self._sess = requests.Session()
        self._sess.trust_env = False
        self.last_events: list[StepEvent] = []

    def _call_llm(self, messages: list[dict[str, Any]]) -> str:
        if self._chat_fn is not None:
            return self._chat_fn(messages)
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        last_err: Exception | None = None
        for _ in range(3):
            try:
                resp = self._sess.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=body,
                    timeout=120,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"].get("content") or ""
            except Exception as exc:
                last_err = exc
                time.sleep(1.0)
        raise RuntimeError(f"MAI-Gym VLM 调用失败: {last_err}")

    def run(self, instruction: str) -> AgentResult:
        """端到端闭环执行自然语言多步指令，返回兼容项目的 AgentResult。"""
        self.last_events = []
        screenshots_b64: list[str] = [img_to_b64(self.device.screenshot())]
        assistant_turns: list[str] = []

        for step in range(1, self.max_steps + 1):
            messages = build_multistep_messages(
                instruction,
                screenshots_b64,
                assistant_turns,
                keep_last_images=self.keep_last_images,
                note=self.note,
            )
            try:
                raw = self._call_llm(messages)
            except Exception as exc:
                return AgentResult(status="llm_error", steps=step, answer=str(exc))

            action = parse_maigym_action(raw)
            ev = StepEvent(step=step, raw=raw, action=action)
            self.last_events.append(ev)
            if self.on_step is not None:
                self.on_step(ev)

            atype = str(action.get("action_type") or action.get("action") or "").lower()
            if atype == "ask_user":
                return AgentResult(
                    status="need_user",
                    steps=step,
                    answer=str(action.get("text") or "该操作需人工确认"),
                )

            if atype == "terminate":
                st = str(action.get("status") or "success").lower()
                mapped = "failure" if st in ("failed", "failure") else "success"
                return AgentResult(status=mapped, steps=step, answer=str(action.get("answer") or ""))

            if atype in ("click", "double_click", "long_press"):
                coord = action.get("coordinate")
                if not (isinstance(coord, (list, tuple)) and len(coord) == 2):
                    return AgentResult(status="parse_error", steps=step, answer=f"无效点击坐标: {action}")
                u, v = to_uv(coord)
                if atype == "long_press":
                    self.device.long_press(u, v)
                else:
                    self.device.click(u, v)
                    if atype == "double_click":
                        self.device.click(u, v)

            elif atype in ("drag", "swipe"):
                sc = action.get("start_coordinate")
                ec = action.get("end_coordinate")
                if not (
                    isinstance(sc, (list, tuple))
                    and len(sc) == 2
                    and isinstance(ec, (list, tuple))
                    and len(ec) == 2
                ):
                    return AgentResult(status="parse_error", steps=step, answer=f"无效滑动坐标: {action}")
                u1, v1 = to_uv(sc)
                u2, v2 = to_uv(ec)
                self.device.swipe(u1, v1, u2, v2, duration_s=0.5)

            elif atype == "wait":
                time.sleep(min(float(action.get("waiting_time") or 0.5), 5.0))

            else:
                return AgentResult(
                    status="unsupported",
                    steps=step,
                    answer=f"不支持的动作类型: {atype or raw[:120]}",
                )

            if self.settle_s > 0:
                time.sleep(self.settle_s)
            assistant_turns.append(raw.strip())
            screenshots_b64.append(img_to_b64(self.device.screenshot()))

        return AgentResult(status="max_steps", steps=self.max_steps, answer=f"超过最大步数 {self.max_steps}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="离线测试 MAI-Gym-9B 原生端到端多步视觉代理。")
    parser.add_argument("--images", nargs="+", required=True, help="按顺序输入的截图文件")
    parser.add_argument("--instruction", required=True, help="自然语言多步指令")
    parser.add_argument("--base-url", default="http://127.0.0.1:18809/v1")
    parser.add_argument("--model", default="mai-gym-9b-v2")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    img_paths = [pathlib.Path(p) for p in args.images]
    device = FakeDevice(img_paths, can_type=False)
    agent = MaiGymMultiStepAgent(
        device,
        base_url=args.base_url,
        model=args.model,
        max_steps=args.max_steps,
        enable_thinking=args.enable_thinking,
        max_tokens=args.max_tokens,
        settle_s=0.0,
    )
    res = agent.run(args.instruction)
    print(json.dumps(dataclasses.asdict(res), ensure_ascii=False, indent=2))
    print("设备动作：", json.dumps(device.actions, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
