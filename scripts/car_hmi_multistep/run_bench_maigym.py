"""Zero-shot / fine-tuned locating test for Tongyi-MAI/MAI-Gym-9B on the L8 cockpit mock.

Uses the exact official MAI-Gym mobile_use system prompt, two-user-turn multimodal
format (text turn + image turn), 0-999 coordinate scaling, and <tool_call> parser
from Tongyi-MAI/MAI-Gym, while writing the exact same JSONL result format as
run_bench.py so analyze.py works unchanged.

Usage:
  bin/py run_bench_maigym.py --base-url http://127.0.0.1:18802/v1 --model mai-gym-9b --label maigym9b_zs
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import json
import pathlib
import re
import sys
import time
from typing import Any

import numpy as np
import requests

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import run_bench as rb  # noqa: E402
from tablet_tap.agent import by_steps as bs  # noqa: E402

W, H = rb.W, rb.H  # 1920, 1200

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

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_COORD_SCALE = 999.0


def build_maigym_messages(image_path: str, instruction: str, *, merged_user: bool = False) -> list[dict[str, Any]]:
    b64 = base64.b64encode(pathlib.Path(image_path).read_bytes()).decode("ascii")
    img_part = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
    txt_part = {"type": "text", "text": instruction}
    if merged_user:
        return [
            {"role": "system", "content": MAI_GYM_SYSTEM_PROMPT},
            {"role": "user", "content": [txt_part, img_part]},
        ]
    return [
        {"role": "system", "content": MAI_GYM_SYSTEM_PROMPT},
        {"role": "user", "content": [txt_part]},
        {"role": "user", "content": [img_part]},
    ]


def parse_maigym_response(text: str) -> tuple[str | None, tuple[float, float] | None]:
    """Parse official MAI-Gym <tool_call> JSON -> (action_type, (x_0_999, y_0_999))."""
    m = _TOOL_CALL_RE.search(text)
    if m:
        raw = m.group(1).strip()
        try:
            obj = json.loads(raw)
            args = obj.get("arguments", obj) if isinstance(obj, dict) else {}
            if isinstance(args, str):
                args = json.loads(args)
            action = args.get("action_type") or args.get("action")
            coord = args.get("coordinate")
            if isinstance(coord, (list, tuple)) and len(coord) == 2:
                return action, (float(coord[0]), float(coord[1]))
            return action, None
        except Exception:
            pass
    # Fallback regex for coordinate inside JSON-like text
    pt = rb.parse_point(text)
    if pt is not None:
        return "click", pt
    return None, None


def ask_maigym(
    base_url: str,
    model: str,
    image_path: str,
    instruction: str,
    *,
    enable_thinking: bool,
    max_tokens: int,
    merged_user: bool,
) -> tuple[str | None, tuple[float, float] | None, str, float]:
    msgs = build_maigym_messages(image_path, instruction, merged_user=merged_user)
    body: dict[str, Any] = {
        "model": model,
        "messages": msgs,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {
            "enable_thinking": enable_thinking,
            "preserve_thinking": True,
        },
    }
    t0, out = time.monotonic(), ""
    for _ in range(3):
        try:
            with requests.Session() as sess:
                sess.trust_env = False
                rep = sess.post(base_url.rstrip("/") + "/chat/completions", json=body, timeout=180)
            rep.raise_for_status()
            msg = rep.json()["choices"][0]["message"]
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
            if reasoning and "<think" not in content:
                out = f"<think>\n{reasoning}\n</think>\n{content}"
            else:
                out = content
            action, coord = parse_maigym_response(out)
            if action is not None:
                return action, coord, out, time.monotonic() - t0
        except Exception as exc:
            out = f"ERROR {exc}"
            time.sleep(2)
    action, coord = parse_maigym_response(out)
    return action, coord, out, time.monotonic() - t0


def eval_one(
    base_url: str,
    model: str,
    t: dict[str, Any],
    cond: str,
    image: str,
    *,
    enable_thinking: bool,
    max_tokens: int,
    with_note: bool,
    merged_user: bool,
) -> dict[str, Any]:
    desc = t["desc"]
    instruction = f"点击{desc} ({bs._IN_APP})" if with_note else f"点击{desc}"  # noqa: SLF001
    action, coord, output, secs = ask_maigym(
        base_url,
        model,
        image,
        instruction,
        enable_thinking=enable_thinking,
        max_tokens=max_tokens,
        merged_user=merged_user,
    )
    r: dict[str, Any] = {k: t[k] for k in ("screen", "id", "name", "box")} | {
        "desc": desc,
        "cond": cond,
        "secs": round(secs, 2),
        "output": output,
        "action": action,
    }
    if action not in ("click", "double_click", "long_press") or coord is None:
        r["hit"], r["px"], r["py"], r["landed"] = False, None, None, None
        return r
    px = min(max(coord[0] / _COORD_SCALE, 0.0), 1.0) * W
    py = min(max(coord[1] / _COORD_SCALE, 0.0), 1.0) * H
    r["px"], r["py"] = round(px), round(py)
    r["hit"] = any(rb.inside(px, py, b) for b in t["ok_boxes"])
    under = [c for c in t["controls"] if rb.inside(px, py, c[2:])]
    r["landed"] = min(under, key=lambda c: c[4] * c[5])[1] if under else ""
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="mai-gym-9b")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--conds", default="clean,camera")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--no-note", action="store_true")
    ap.add_argument("--merged-user", action="store_true")
    ap.add_argument("--screens", default="")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((HERE / "screens" / "screens.json").read_text())
    ts = rb.targets(meta)
    if a.screens:
        ts = [t for t in ts if t["screen"] in a.screens.split(",")]
    if a.limit:
        ts = ts[:: max(1, len(ts) // a.limit)][: a.limit]
    pics = rb.pictures(HERE / "screens", out, {t["image"] for t in ts})
    jobs = [(t, c) for c in a.conds.split(",") for t in ts]

    res: list[dict[str, Any]] = []
    t0 = time.monotonic()
    with cf.ThreadPoolExecutor(a.workers) as pool:
        futs = [
            pool.submit(
                eval_one,
                a.base_url,
                a.model,
                t,
                c,
                pics[c, t["image"]],
                enable_thinking=not a.no_think,
                max_tokens=a.max_tokens,
                with_note=not a.no_note,
                merged_user=a.merged_user,
            )
            for t, c in jobs
        ]
        for i, f in enumerate(cf.as_completed(futs), 1):
            res.append(f.result())
            if i % 50 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)}  {time.monotonic() - t0:.0f}s", flush=True)

    out_path = out / f"{a.label}.jsonl"
    out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in res), encoding="utf-8")

    holdout = {"set_secure_sentry", "set_assistant_timbre", "phone_recent", "nav_route"}
    for c in a.conds.split(","):
        rs = [r for r in res if r["cond"] == c]
        ho = [r for r in rs if r["screen"] in holdout]
        print(
            f"[{a.label}] {c}: {sum(r['hit'] for r in rs)}/{len(rs)} "
            f"({100 * sum(r['hit'] for r in rs) / max(1, len(rs)):.1f}%)  "
            f"holdout {sum(r['hit'] for r in ho)}/{len(ho)}  "
            f"mean {np.mean([r['secs'] for r in rs]):.2f}s"
        )
    print(f"wall {time.monotonic() - t0:.0f}s for {len(jobs)} asks -> {out_path}")


if __name__ == "__main__":
    main()
