"""How often the model picks the right thing on labelled screens: one step per instruction, no robot, no camera.

    bin/py -m tablet_tap.agent.evaluate                          # tests/data/screens, the model server on :8802
    bin/py -m tablet_tap.agent.evaluate --add_info zh en none    # compare the texts appended to the instruction
    bin/py -m tablet_tap.agent.evaluate --variant "cam_a=runs/smoke_vlm_cpu/variants/{screen}__cam_a__rectified.png"

A labelled screen is ``<name>.png`` with ``<name>.json`` beside it::

    {"size": [1280, 800],
     "elements": {"play": {"box": [864, 522, 960, 618]}},
     "instructions": [{"text": "点击播放按钮", "target": "play"}]}

Boxes are x0, y0, x1, y1 in pixels of ``size``. A ``--variant`` is another picture of the same screens, the camera's
rectified view of them for instance; it may have any size. To score the real tablet, save pictures taken with the
``shot`` command as screens and write the boxes of what should be tapped.

Every case goes through ``GuiAgent`` and a ``FakeDevice`` as a real step does (same prompt, same parsing), so what is
counted is the first thing the arm would have been told to do: a tap inside the box is a hit, a tap elsewhere a miss,
anything else (a swipe, the Home button, ending the task) is listed as "other".
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime
import json
import pathlib
import statistics
import time
from typing import Any

from tablet_tap.agent.device import FakeDevice
from tablet_tap.agent.gui_agent import ADD_INFO_PRESETS
from tablet_tap.agent.gui_agent import PROJECT_DIR
from tablet_tap.agent.gui_agent import GuiAgent
from tablet_tap.agent.gui_agent import Llm
from tablet_tap.agent.gui_agent import resolve_add_info
from tablet_tap.errors import SetupError
from tablet_tap.errors import TabletTapError

DEFAULT_SCREENS = PROJECT_DIR / "tests" / "data" / "screens"
CLEAN = "clean"  # the labelled pictures themselves
TAPS = ("click", "long_press")


@dataclasses.dataclass(frozen=True)
class Case:
    screen: str
    variant: str
    image: pathlib.Path
    instruction: str
    target: str
    box_uv: tuple[float, float, float, float]  # u0, v0, u1, v1: the target as fractions of the screen


def load_cases(screens_dir: pathlib.Path, variants: dict[str, str] | None = None) -> list[Case]:
    """The cases of every labelled screen in ``screens_dir``, once per variant.

    ``variants`` maps a name to a path with ``{screen}`` in it; without any, the labelled pictures are the only
    variant, named "clean". A relative path is looked up from the current directory, then from the project.
    """
    labels = sorted(pathlib.Path(screens_dir).glob("*.json"))
    cases = []
    for label in labels:
        meta = json.loads(label.read_text(encoding="utf-8"))
        if "instructions" not in meta or "elements" not in meta:
            continue  # some other json kept with the pictures
        width, height = meta["size"]
        pictures = {CLEAN: label.with_name(meta.get("image", label.stem + ".png"))}
        for name, pattern in (variants or {}).items():
            path = pathlib.Path(pattern.format(screen=label.stem))
            pictures[name] = path if path.is_absolute() or path.exists() else PROJECT_DIR / path
        for variant, image in pictures.items():
            if not image.is_file():
                raise SetupError(f"找不到 {label.stem} 的 {variant} 图片：{image}")
            for instruction in meta["instructions"]:
                x0, y0, x1, y1 = meta["elements"][instruction["target"]]["box"]
                box = (x0 / width, y0 / height, x1 / width, y1 / height)
                cases.append(Case(label.stem, variant, image, instruction["text"], instruction["target"], box))
    if not cases:
        raise SetupError(f"{screens_dir} 里没有带标注的截图（<名字>.png 和 <名字>.json）")
    return cases


def run_case(case: Case, add_info: str, run_dir: pathlib.Path, **agent_options: Any) -> dict[str, Any]:
    """One step of the agent on one case; ``agent_options`` go to ``GuiAgent`` (base_url, model, llm, ...)."""
    device = FakeDevice([case.image], can_type=False)
    agent = GuiAgent(device, max_steps=1, settle_s=0.0, add_info=add_info, run_dir=run_dir, **agent_options)
    started = time.monotonic()
    result = agent.run(case.instruction)
    seconds = time.monotonic() - started
    if result.status == "llm_error":
        raise TabletTapError(f"模型服务没有答复，评测中止：{result.answer}")
    first = device.actions[0] if device.actions else None
    row = {**dataclasses.asdict(case), "image": str(case.image), "seconds": round(seconds, 3), "run_dir": str(run_dir)}
    if first is not None and first["action"] in TAPS:
        u0, v0, u1, v1 = case.box_uv
        inside = u0 <= first["u"] <= u1 and v0 <= first["v"] <= v1
        return {**row, "outcome": "hit" if inside else "miss", "did": first}
    # no tap: what the device was told instead, or how the agent ended the task without telling it anything
    return {**row, "outcome": "other", "did": first or {"action": result.status, "text": result.answer}}


def evaluate(
    cases: list[Case], add_infos: dict[str, str], out_dir: pathlib.Path, **agent_options: Any
) -> list[dict[str, Any]]:
    """Every case with every text of ``add_infos`` (label -> text). Rows carry the label as ``add_info``."""
    rows = []
    for label, text in add_infos.items():
        for index, case in enumerate(cases):
            run_dir = out_dir / label / f"{index:03d}_{case.screen}_{case.variant}_{case.target}"
            rows.append({"add_info": label, **run_case(case, text, run_dir, **agent_options)})
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One line per (add_info, variant), in the order they first appear."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[row["add_info"], row["variant"]].append(row)
    lines = []
    for (label, variant), group in groups.items():
        counts = collections.Counter(row["outcome"] for row in group)
        lines.append({
            "add_info": label, "variant": variant, "n": len(group), "hit": counts["hit"], "miss": counts["miss"],
            "other": counts["other"], "median_s": round(statistics.median(row["seconds"] for row in group), 2),
        })  # fmt: skip
    return lines


def report(rows: list[dict[str, Any]]) -> str:
    """The table of ``summarize`` and every case that was not a hit, as text."""
    out = []
    for line in summarize(rows):
        out.append(
            f"说明 {line['add_info']:<6} 画面 {line['variant']:<8} 命中 {line['hit']:>2}/{line['n']:<3}"
            f" 点错 {line['miss']:>2}  其它 {line['other']:>2}  每步 {line['median_s']:.2f} s"
        )
    failed = [row for row in rows if row["outcome"] != "hit"]
    if failed:
        out.append("\n没有命中的：")
    for row in failed:
        did = row["did"]
        if did["action"] in TAPS:
            what = f"点在 u={did['u']:.3f} v={did['v']:.3f}"
        else:
            what = json.dumps(did, ensure_ascii=False)
        u0, v0, u1, v1 = row["box_uv"]
        out.append(
            f"  [{row['add_info']}] {row['screen']}/{row['variant']} “{row['instruction']}”"
            f"（目标 {row['target']}：u {u0:.3f}–{u1:.3f}，v {v0:.3f}–{v1:.3f}）→ {what}"
        )
    return "\n".join(out)


def _named(values: list[str]) -> dict[str, str]:
    """Labels for the --add_info values: the preset's name, or 自定义N for a text given in full."""
    named, custom = {}, 0
    for value in values:
        if value in ADD_INFO_PRESETS or value == "":
            named[value or "none"] = resolve_add_info(value)
        else:
            custom += 1
            named[f"自定义{custom}"] = value
    return named


def main(argv: list[str] | None = None, llm: Llm | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="在带标注的截图上统计模型第一步点得对不对（不动机械臂、不开相机；要先起好模型服务）。"
    )
    parser.add_argument("--screens", type=pathlib.Path, default=DEFAULT_SCREENS, help="带标注截图的目录")
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        metavar="名字=路径",
        help="同一批截图的另一种画面，路径里用 {screen} 代表截图名，可以给多次；给了就不再评测原图，除非加 --clean",
    )
    parser.add_argument("--clean", action="store_true", help="给了 --variant 时也评测原图")
    parser.add_argument(
        "--add_info", nargs="+", default=["zh"], help="要比较的附加说明：zh / en / none，或直接给一段文字；可以给多个"
    )
    parser.add_argument("--base_url", default="http://127.0.0.1:8802/v1", help="模型服务地址（OpenAI 接口）")
    parser.add_argument("--model", default="gui-owl", help="服务里的模型名")
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument("--out", type=pathlib.Path, default=None, help="记录目录；默认 runs/eval_<年月日_时分秒>")
    args = parser.parse_args(argv)

    variants = {}
    for item in args.variant:
        name, sep, pattern = item.partition("=")
        if not sep or not name or "{screen}" not in pattern:
            parser.error(f"--variant 要写成 名字=路径，路径里带 {{screen}}：{item!r}")
        variants[name] = pattern
    cases = load_cases(args.screens, variants)
    if variants and not args.clean:
        cases = [case for case in cases if case.variant != CLEAN]
    out_dir = args.out or PROJECT_DIR / "runs" / f"eval_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    options: dict[str, Any] = {"base_url": args.base_url, "model": args.model, "api_key": args.api_key}
    if llm is not None:
        options["llm"] = llm
    else:
        options.update(llm_max_retry=1, llm_retry_wait_s=0)  # a server that is down should end the run at once
    add_infos = _named(args.add_info)
    print(f"{len(cases)} 个用例 × {len(add_infos)} 种说明，模型 {args.model}（{args.base_url}）……", flush=True)
    rows = evaluate(cases, add_infos, out_dir, **options)
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {"model": args.model, "base_url": args.base_url, "add_info": add_infos, "summary": summarize(rows)}
    record["rows"] = rows
    (out_dir / "results.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    print(report(rows))
    print(f"\n记录：{out_dir}（results.json；每个用例一个目录，里面有标了落点的 step_01_anno.png）")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TabletTapError as exc:
        raise SystemExit(f"✗ {exc}") from None
