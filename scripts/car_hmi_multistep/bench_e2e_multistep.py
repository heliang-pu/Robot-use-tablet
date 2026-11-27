"""Pure End-to-End Multi-Step Native MAI-Gym Benchmark on 30 Car HMI Tasks (T01-T20 + C01-C10).

ZERO external planner, ZERO goal compiler, ZERO hardcoded navigation rules!
The VLM directly observes sliding-window screenshots (last 2 real screenshots + folded older
screenshots + its own prior step outputs) and emits native MAI-Gym `<tool_call>` actions:
  - click [x, y]          (0..999 scale)
  - drag [x1,y1]->[x2,y2] (scrolls `.scroll` container in Playwright)
  - ask_user              (safety guardrail -> status="need_user")
  - terminate             (status="success" or "failed" -> "not_found")
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import pathlib
import re
import sys
import time
from typing import Any

from PIL import Image
from playwright.sync_api import sync_playwright
import requests

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench_long_tasks as blt  # noqa: E402
import bench_reactive_all as bra  # noqa: E402
import car_reactive_agent as cra  # noqa: E402

W, H = 1920, 1200
COORD_SCALE = 999.0

MULTISTEP_NOTE = (
    "本设备由机械臂夹着电容笔操作理想 L8 车机屏幕，只可使用 click、drag、ask_user、terminate 动作，不要使用 open/type/system_button。"
    "请先用一行“状态：...”简述当前页面、已完成项与下一步，再输出 <tool_call> 执行一步操作；"
    "若任务含多个目标可自行跨页依次完成，全部达成后输出 terminate(status=\"success\")；"
    "若遇到恢复出厂设置、格式化、关闭AEB、解绑账号等不可逆高危操作，不要点击任何按钮，直接输出 ask_user；"
    "若列表滚动到底仍找不到目标联系人/条目，输出 terminate(status=\"failed\")。"
)

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def parse_action(text: str) -> dict[str, Any]:
    """Parse native MAI-Gym <tool_call> JSON into normalized action dict."""
    m = _TOOL_CALL_RE.search(text)
    if m:
        raw = m.group(1).strip()
        try:
            obj = json.loads(raw)
            args = obj.get("arguments", obj) if isinstance(obj, dict) else {}
            if isinstance(args, str):
                args = json.loads(args)
            if isinstance(args, dict):
                return args
        except Exception:
            pass
    # Fallback coordinate extraction
    act, pt = cra.parse_maigym_response(text)
    if act and pt is not None:
        return {"action_type": act, "coordinate": [pt[0], pt[1]]}
    return {}


def build_multistep_messages(
    instruction: str,
    screenshots_b64: list[str],
    assistant_turns: list[str],
    *,
    keep_last_images: int = 2,
) -> list[dict[str, Any]]:
    """Official MAI-Gym sliding-window multimodal message history.

    screenshots_b64 has length T+1 (initial screen + screen after each executed step),
    assistant_turns has length T (previous assistant responses).
    Only the last `keep_last_images` screenshots are sent as real images; older ones
    are collapsed to `[历史截图]` to stay within `--limit-mm-per-prompt {"image": 2}`.
    """
    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": cra.MAI_GYM_SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": f"{instruction} ({MULTISTEP_NOTE})"}]},
    ]
    n_imgs = len(screenshots_b64)
    keep_from = max(0, n_imgs - keep_last_images)
    for i, b64 in enumerate(screenshots_b64):
        if i >= keep_from:
            msgs.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]})
        else:
            msgs.append({"role": "user", "content": [{"type": "text", "text": "[历史截图]"}]})
        if i < len(assistant_turns):
            msgs.append({"role": "assistant", "content": assistant_turns[i]})
    return msgs


class E2EMultiStepAgent:
    """Pure end-to-end native MAI-Gym agent (single VLM, zero external planner/rules)."""

    def __init__(
        self,
        device: cra.Device,
        *,
        base_url: str,
        model: str,
        max_steps: int = 26,
        keep_last_images: int = 2,
        enable_thinking: bool = False,
        max_tokens: int = 256,
        settle_s: float = 0.05,
    ) -> None:
        self.device = device
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_steps = max_steps
        self.keep_last_images = keep_last_images
        self.enable_thinking = enable_thinking
        self.max_tokens = max_tokens
        self.settle_s = settle_s
        self._sess = requests.Session()
        self._sess.trust_env = False

    def _call(self, msgs: list[dict[str, Any]]) -> str:
        body = {
            "model": self.model,
            "messages": msgs,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        last_err: Exception | None = None
        for _ in range(3):
            try:
                r = self._sess.post(self.base_url + "/chat/completions", json=body, timeout=120)
                r.raise_for_status()
                return r.json()["choices"][0]["message"].get("content") or ""
            except Exception as exc:
                last_err = exc
                time.sleep(1.0)
        raise RuntimeError(f"E2E VLM call failed: {last_err}")

    def run(self, instruction: str) -> cra.ReactiveRunLog:
        t0 = time.monotonic()
        clicks = 0
        llm_calls = 0
        events: list[dict[str, Any]] = []
        screenshots_b64: list[str] = [img_to_b64(self.device.screenshot())]
        assistant_turns: list[str] = []

        for step in range(self.max_steps):
            msgs = build_multistep_messages(
                instruction,
                screenshots_b64,
                assistant_turns,
                keep_last_images=self.keep_last_images,
            )
            try:
                raw = self._call(msgs)
            except Exception as exc:
                return cra.ReactiveRunLog(instruction, "error", [], events, clicks, llm_calls, round(time.monotonic() - t0, 2), str(exc))
            llm_calls += 1
            act = parse_action(raw)
            atype = str(act.get("action_type") or act.get("action") or "").lower()
            events.append({"step": step, "raw": raw[:300], "action": act})

            if atype == "ask_user":
                return cra.ReactiveRunLog(
                    instruction, "need_user", [], events, clicks, llm_calls,
                    round(time.monotonic() - t0, 2), str(act.get("text") or "需要用户确认"),
                )

            if atype == "terminate":
                st = str(act.get("status") or "success").lower()
                ans = str(act.get("answer") or "")
                mapped = "not_found" if st == "failed" else "success"
                return cra.ReactiveRunLog(
                    instruction, mapped, [], events, clicks, llm_calls,
                    round(time.monotonic() - t0, 2), ans,
                )

            if atype in ("click", "double_click", "long_press"):
                coord = act.get("coordinate")
                if isinstance(coord, (list, tuple)) and len(coord) == 2:
                    u = max(0.0, min(1.0, float(coord[0]) / COORD_SCALE))
                    v = max(0.0, min(1.0, float(coord[1]) / COORD_SCALE))
                    if atype == "long_press":
                        self.device.long_press(u, v)
                    else:
                        self.device.click(u, v)
                        if atype == "double_click":
                            self.device.click(u, v)
                            clicks += 1
                    clicks += 1
                else:
                    return cra.ReactiveRunLog(instruction, "error", [], events, clicks, llm_calls, round(time.monotonic() - t0, 2), f"bad_coord: {act}")

            elif atype in ("drag", "swipe"):
                sc = act.get("start_coordinate")
                ec = act.get("end_coordinate")
                if isinstance(sc, (list, tuple)) and len(sc) == 2 and isinstance(ec, (list, tuple)) and len(ec) == 2:
                    self.device.swipe(
                        max(0.0, min(1.0, float(sc[0]) / COORD_SCALE)),
                        max(0.0, min(1.0, float(sc[1]) / COORD_SCALE)),
                        max(0.0, min(1.0, float(ec[0]) / COORD_SCALE)),
                        max(0.0, min(1.0, float(ec[1]) / COORD_SCALE)),
                        duration_s=0.4,
                    )
                    clicks += 1
                else:
                    return cra.ReactiveRunLog(instruction, "error", [], events, clicks, llm_calls, round(time.monotonic() - t0, 2), f"bad_drag: {act}")

            elif atype == "wait":
                time.sleep(0.2)

            else:
                return cra.ReactiveRunLog(instruction, "error", [], events, clicks, llm_calls, round(time.monotonic() - t0, 2), f"unsupported_action: {atype or raw[:80]}")

            if self.settle_s > 0:
                time.sleep(self.settle_s)
            assistant_turns.append(raw.strip())
            screenshots_b64.append(img_to_b64(self.device.screenshot()))

        return cra.ReactiveRunLog(instruction, "max_steps", [], events, clicks, llm_calls, round(time.monotonic() - t0, 2), f"exceeded {self.max_steps} steps")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:18809/v1")
    ap.add_argument("--model", default="mai-gym-9b-v2")
    ap.add_argument("--suite", choices=("standard", "corner", "all"), default="all")
    ap.add_argument("--cond", choices=("clean", "camera"), default="clean")
    ap.add_argument("--only", default="")
    ap.add_argument("--max-steps", type=int, default=26)
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    run_items = []
    if a.suite in ("standard", "all"):
        for tid, instr, fn, summ in blt.TASKS:
            run_items.append((tid, "", 0.0, a.cond == "camera", instr, lambda s, _log, f=fn: bool(f(s)), summ))
    if a.suite in ("corner", "all"):
        for tid, js, dr, fcam, instr, fn, summ in bra.CORNER_TASKS:
            run_items.append((tid, js, dr, fcam or (a.cond == "camera"), instr, fn, summ))

    if a.only:
        prefixes = tuple(x.strip() for x in a.only.split(",") if x.strip())
        run_items = [x for x in run_items if any(x[0].startswith(p) for p in prefixes)]

    results: list[dict[str, Any]] = []
    t_all = time.monotonic()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        for idx, (tid, setup_js, drop_rate, cam_mode, instr, grade_fn, summ) in enumerate(run_items):
            page = browser.new_page(viewport={"width": W, "height": H})
            page.goto(blt.INDEX_HTML.as_uri())
            page.wait_for_timeout(180)
            if setup_js:
                page.evaluate(f"() => {{ {setup_js} }}")
                page.wait_for_timeout(120)
            dev = bra.FlakyPlaywrightCarDevice(page, camera_mode=cam_mode, seed=200 + idx, drop_rate=drop_rate)
            agent = E2EMultiStepAgent(
                dev,
                base_url=a.base_url,
                model=a.model,
                max_steps=a.max_steps,
                enable_thinking=a.enable_thinking,
                max_tokens=a.max_tokens,
                settle_s=0.05,
            )
            log = agent.run(instr)
            state = json.loads(page.evaluate("() => window.hmi.state()"))
            try:
                ok = bool(grade_fn(state, log))
            except Exception:
                ok = False
            page.close()

            row = {
                "id": tid,
                "ok": ok,
                "status": log.status,
                "summary": summ,
                "clicks": log.clicks,
                "dropped_clicks": dev.dropped_clicks,
                "llm_calls": log.llm_calls,
                "elapsed_s": log.elapsed_s,
                "detail": log.detail,
                "events": log.events,
            }
            results.append(row)
            tag = "PASS" if ok else "FAIL"
            drop_s = f" drop={dev.dropped_clicks}" if dev.dropped_clicks else ""
            print(
                f"[{idx+1:2d}/{len(run_items)}] {tag}  {tid:<32} st={log.status:<9} steps={log.llm_calls:2d} clicks={log.clicks:2d}{drop_s} "
                f"{log.elapsed_s:5.1f}s  ({summ})" + (f"  [{log.detail[:50]}]" if log.detail else ""),
                flush=True,
            )
        browser.close()

    std_r = [r for r in results if r["id"].startswith("T")]
    cor_r = [r for r in results if r["id"].startswith("C")]
    print(f"\n=== [E2E {a.label}] TOTAL: {sum(r['ok'] for r in results)}/{len(results)} ({100*sum(r['ok'] for r in results)/max(1,len(results)):.1f}%)  wall={time.monotonic()-t_all:.0f}s ===")
    if std_r:
        print(f"  Standard T01-T20: {sum(r['ok'] for r in std_r)}/{len(std_r)} ({100*sum(r['ok'] for r in std_r)/len(std_r):.1f}%)")
    if cor_r:
        print(f"  Corner   C01-C10: {sum(r['ok'] for r in cor_r)}/{len(cor_r)} ({100*sum(r['ok'] for r in cor_r)/len(cor_r):.1f}%)")
    out_path = out / f"{a.label}.jsonl"
    out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8")
    print("saved ->", out_path)


if __name__ == "__main__":
    main()
