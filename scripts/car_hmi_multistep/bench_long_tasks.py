"""Closed-loop Playwright benchmark of 20 multi-step long-horizon tasks on the L8 Car HMI.

Pure-vision interaction:
  - Renders android/car_hmi/assets/www/index.html in headless Chromium at 1920x1200.
  - Feeds 1280x800 clean or simulated wrist-camera screenshots to `CarTaskRunner`.
  - Executes REAL mouse clicks at `(u * 1920, v * 1200)` — zero DOM cheating!
  - After the task finishes, reads `window.hmi.S` ONLY for objective pass/fail grading.

Usage:
  ~/workspace/tablet_tap/bin/py bench_long_tasks.py \
    --base-url http://127.0.0.1:18809/v1 --model mai-gym-9b \
    --label long_maigym9b_ft --cond clean
"""
from __future__ import annotations

import argparse
import io
import json
import pathlib
import sys
import time
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import car_long_agent as cla  # noqa: E402

FEED = (1280, 800)

def sim_camera(img: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    small = cv2.resize(img, (1200, 750), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 0.9)
    small = np.clip(small.astype(np.float32) * 0.82 + 22 + rng.normal(0, 3.0, small.shape), 0, 255).astype(np.uint8)
    _, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return cv2.resize(cv2.imdecode(jpg, cv2.IMREAD_COLOR), FEED, interpolation=cv2.INTER_CUBIC)

INDEX_HTML = REPO_ROOT / "android" / "car_hmi" / "assets" / "www" / "index.html"
W, H = 1920, 1200


class PlaywrightCarDevice:
    """Implements tablet_tap.agent.device.Device over live Playwright Chromium L8 mock."""

    def __init__(self, page: Any, *, camera_mode: bool = False, seed: int = 0) -> None:
        self.page = page
        self.camera_mode = camera_mode
        self._seed = seed
        self._shot_i = 0

    def screenshot(self) -> Image.Image:
        png = self.page.screenshot(type="png")
        arr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        if self.camera_mode:
            pic = sim_camera(arr, self._seed * 100 + self._shot_i)
        else:
            pic = cv2.resize(arr, FEED, interpolation=cv2.INTER_AREA)
        self._shot_i += 1
        return Image.fromarray(cv2.cvtColor(pic, cv2.COLOR_BGR2RGB))

    def click(self, u: float, v: float) -> None:
        x, y = u * W, v * H
        self.page.mouse.move(x, y)
        self.page.mouse.down()
        self.page.wait_for_timeout(40)
        self.page.mouse.up()
        self.page.wait_for_timeout(120)

    def long_press(self, u: float, v: float, duration_s: float = 0.8) -> None:
        x, y = u * W, v * H
        self.page.mouse.move(x, y)
        self.page.mouse.down()
        self.page.wait_for_timeout(int(duration_s * 1000))
        self.page.mouse.up()
        self.page.wait_for_timeout(120)

    def swipe(self, u1: float, v1: float, u2: float, v2: float, duration_s: float = 0.8) -> None:
        self.page.mouse.move(u1 * W, v1 * H)
        self.page.mouse.down()
        self.page.mouse.move(u2 * W, v2 * H, steps=10)
        self.page.mouse.up()
        self.page.wait_for_timeout(150)

    def back(self) -> None:
        self.page.evaluate("window.hmiBack && window.hmiBack()")
        self.page.wait_for_timeout(120)

    def home(self) -> None:
        self.page.evaluate("window.hmi.go('home'); window.hmi.render()")
        self.page.wait_for_timeout(120)

    def type_text(self, text: str) -> None:
        raise NotImplementedError("Stylus-only car HMI benchmark")


# 20 diverse, multi-step (4-16 clicks each) realistic car cockpit tasks, including holdout pages!
TASKS: list[tuple[str, str, Callable[[dict[str, Any]], bool], str]] = [
    (
        "T01_climate_driver_temp_heat",
        "打开空调页，把主驾温度调到21度，把主驾座椅加热调到2档，并开启前除霜。",
        lambda s: abs(s["c"]["tL"] - 21.0) < 0.01 and s["c"]["seat"]["fl"][0] == 2 and s["c"]["fdef"] is True,
        "tL==21.0 & fl.heat==2 & fdef==True",
    ),
    (
        "T02_climate_rear_comfort",
        "进入空调页，打开后排空调，把后排温度调到26度，后排风量调到4档，并打开香氛和负离子。",
        lambda s: s["c"]["ron"] is True and abs(s["c"]["rt"] - 26.0) < 0.01 and s["c"]["rfan"] == 4 and s["c"]["scent"] is True and s["c"]["ion"] is True,
        "ron & rt==26 & rfan==4 & scent & ion",
    ),
    (
        "T03_climate_seats_multi",
        "进空调页，把主驾座椅通风关掉，把副驾座椅按摩开到1档，把二排左座椅加热开到3档，再打开方向盘加热。",
        lambda s: s["c"]["seat"]["fl"][1] == 0 and s["c"]["seat"]["fr"][2] == 1 and s["c"]["seat"]["ml"][0] == 3 and s["v"]["wheel"] is True,
        "fl.vent==0 & fr.massage==1 & ml.heat==3 & wheel",
    ),
    (
        "T04_climate_dual_zone_flow",
        "打开空调页，把副驾温度调到22.5度，把前排风量调到5档，把前排吹风模式切到吹脚，并开启内循环。",
        lambda s: abs(s["c"]["tR"] - 22.5) < 0.01 and s["c"]["fan"] == 5 and s["c"]["mode"] == 2 and s["c"]["rec"] is True,
        "tR==22.5 & fan==5 & mode==2(foot) & rec",
    ),
    (
        "T05_settings_ambient_light",
        "进入车辆设置的灯光页面，把氛围灯颜色改成紫色，氛围灯模式切到多色模式，把前大灯设为近光灯。",
        lambda s: s["v"]["amb_color"] == 6 and s["v"]["amb_mode"] == 1 and s["v"]["headlamp"] == 2,
        "amb_color==6(purple) & amb_mode==1 & headlamp==2",
    ),
    (
        "T06_settings_display_dark_font",
        "打开车辆设置里的显示屏幕，把外观切换到深色模式，把字体大小改成大，再打开屏保时钟。",
        lambda s: s["v"]["theme"] == 1 and s["v"]["font"] == 1 and s["v"]["screensaver"] == 0,
        "theme==1(dark) & font==1(big) & screensaver==0(on)",
    ),
    (
        "T07_settings_drive_sport",
        "进入车辆设置的驾驶页面，把加速模式改成运动，能量回收改成强，转向助力改成运动，悬架高度改成低。",
        lambda s: s["v"]["accel"] == 2 and s["v"]["regen"] == 2 and s["v"]["steer"] == 2 and s["v"]["susp"] == 0,
        "accel==2 & regen==2 & steer==2 & susp==0",
    ),
    (
        "T08_settings_door_mirror",
        "打开车辆设置的门窗页面，把车窗设为通风、解锁方式设为仅主驾，然后切到后视镜页面打开倒车自动下翻和后视镜加热。",
        lambda s: s["v"]["win_all"] == 1 and s["v"]["unlock_way"] == 1 and s["v"]["mir_tilt"] == 0 and s["v"]["mir_heat"] == 0,
        "win_all==1 & unlock_way==1 & mir_tilt==on & mir_heat==on",
    ),
    (
        "T09_settings_adas_parking",
        "进入智能驾驶的辅助驾驶，把跟车距离设为远、变道风格设为高效，再进入泊车打开遥控泊车并把雷达提示音调成高。",
        lambda s: s["v"]["follow"] == 2 and s["v"]["lane_style"] == 2 and s["v"]["rpa"] == 0 and s["v"]["radar_vol"] == 2,
        "follow==2 & lane_style==2 & rpa==on & radar_vol==2",
    ),
    (
        "T10_settings_sound_cinema",
        "打开车辆设置里的声音音效，把音效模式改成影院、声场位置改成前排，然后切到提示音把提示音量调为低。",
        lambda s: s["v"]["eq"] == 1 and s["v"]["field"] == 2 and s["v"]["tone_vol"] == 0,
        "eq==1(cinema) & field==2(front) & tone_vol==0(low)",
    ),
    (
        "T11_nav_search_route_go",
        "打开导航，搜索目的地选择高铁站，路线选最快，然后开始导航。",
        lambda s: s["nav"]["st"] == "guide" and s["nav"]["dest"] == 6 and s["nav"]["opt"] == 1,
        "nav.st==guide & dest==6(高铁站) & opt==1(最快)",
    ),
    (
        "T12_nav_company_lowtoll",
        "进入导航页，点击去公司，路线偏好选少收费，开始导航，并切换成3D视角。",
        lambda s: s["nav"]["st"] == "guide" and s["nav"]["dest"] == 1 and s["nav"]["opt"] == 2 and s["nav"]["view"] == 1,
        "nav.st==guide & dest==1(公司) & opt==2(少收费) & view==1(3D)",
    ),
    (
        "T13_music_play_next_fav",
        "打开音乐页面，切换到本地音乐标签，选择播放公路旅行歌单，并把播放模式改成单曲循环，顺便点亮收藏。",
        lambda s: s["page"] == "music" and s["m"]["tab"] == 3 and s["m"]["i"] == 1 and s["m"]["play"] is True and s["m"]["mode"] == 1 and bool(s["m"]["fav"].get("1")),
        "music tab==3 & track==1 & play & mode==1 & fav[1]",
    ),
    (
        "T14_phone_contact_call",
        "打开电话，切到通讯录，给妈妈打电话。",
        lambda s: s["phone"]["call"] is not None and s["phone"]["call"].get("name") == "妈妈",
        "call.name==妈妈",
    ),
    (
        "T15_phone_dial_number",
        "打开电话的拨号键盘，输入号码 138001 并拨打。",
        lambda s: s["phone"]["call"] is not None and s["phone"]["call"].get("num") == "138001",
        "call.num==138001",
    ),
    (
        "T16_holdout_security_assistant",
        "进车辆设置的安全哨兵页，打开哨兵模式并把灵敏度调成高，再切到理想同学的音色页把音色改成温柔女声、语速改成快。",
        lambda s: s["v"]["sentry_on"] == 0 and s["v"]["sentry_sens"] == 2 and s["v"]["tts_voice"] == 1 and s["v"]["tts_speed"] == 2,
        "[HOLDOUT] sentry==on & sens==high & voice==温柔女声 & speed==快",
    ),
    (
        "T17_slider_ambient_charge",
        "打开车辆设置的灯光页，把氛围灯亮度调到70%左右，再切到充电页把充电上限调到80%左右并开启预约充电。",
        lambda s: abs(s["v"]["amb_bright"] - 70) <= 10 and abs(s["v"]["chg_limit"] - 80) <= 10 and s["v"]["chg_sched"] == 0,
        "amb_bright~=70 & chg_limit~=80 & chg_sched==on",
    ),
    (
        "T18_cross_page_morning_commute",
        "先在桌面打开露营模式，然后进入音乐播放并切到下一首，接着进导航去公司选推荐路线并开始导航。",
        lambda s: bool(s["scn"].get("camp")) and s["m"]["i"] == 1 and s["m"]["play"] is True and s["nav"]["st"] == "guide" and s["nav"]["dest"] == 1,
        "camp==on & music track==1 playing & nav guiding to 公司",
    ),
    (
        "T19_cross_page_dark_then_climate",
        "先去显示屏幕开启深色模式，再去空调页把主驾温度调到22度、打开后除霜并把三排左座椅放倒。",
        lambda s: s["v"]["theme"] == 1 and abs(s["c"]["tL"] - 22.0) < 0.01 and s["v"]["rdef"] is True and s["c"]["seat"]["rl"][2] == 1,
        "dark mode + tL==22 + rdef + rl.fold==1",
    ),
    (
        "T20_mega_cross_4pages",
        "完成出发准备：打开车辆设置把壁纸换成青山，去空调把主驾座椅加热开到1档、主驾温度调到23度，再打开导航选回家并开始导航，最后回到桌面。",
        lambda s: s["v"]["wall"] == 2 and s["c"]["seat"]["fl"][0] == 1 and abs(s["c"]["tL"] - 23.0) < 0.01 and s["nav"]["st"] == "guide" and s["nav"]["dest"] == 0 and s["page"] == "home",
        "wall==青山 & fl.heat==1 & tL==23 & nav guiding home & page==home",
    ),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:18809/v1")
    ap.add_argument("--model", default="mai-gym-9b")
    ap.add_argument("--planner-model", default="")
    ap.add_argument("--label", required=True)
    ap.add_argument("--cond", choices=("clean", "camera"), default="clean")
    ap.add_argument("--only", default="", help="comma task ids prefix filter")
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tasks = TASKS
    if a.only:
        prefixes = tuple(a.only.split(","))
        tasks = [t for t in tasks if t[0].startswith(prefixes)]

    results: list[dict[str, Any]] = []
    t_all = time.monotonic()

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        for idx, (tid, instr, check_fn, summary) in enumerate(tasks):
            page = browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
            page.goto(INDEX_HTML.as_uri())
            page.wait_for_function("window.hmi !== undefined")
            page.add_style_tag(content="*{transition:none!important;animation:none!important}")
            page.wait_for_timeout(120)

            dev = PlaywrightCarDevice(page, camera_mode=(a.cond == "camera"), seed=100 + idx)
            runner = cla.CarTaskRunner(
                dev,
                base_url=a.base_url,
                grounder_model=a.model,
                planner_model=a.planner_model or a.model,
                settle_s=0.05,
            )
            log = runner.run(instr)
            state = json.loads(page.evaluate("() => window.hmi.state()"))
            ok = False
            try:
                ok = bool(check_fn(state))
            except Exception:
                ok = False
            page.close()

            row = {
                "id": tid,
                "cond": a.cond,
                "ok": ok,
                "summary": summary,
                "clicks": log.clicks,
                "llm_calls": log.llm_calls,
                "elapsed_s": log.elapsed_s,
                "error": log.error,
                "plan_len": len(log.plan),
                "plan": log.plan,
                "events": log.events,
            }
            results.append(row)
            status = "PASS" if ok else "FAIL"
            print(
                f"[{idx+1:2d}/{len(tasks)}] {status}  {tid:<32} clicks={log.clicks:2d} llm={log.llm_calls:2d} "
                f"{log.elapsed_s:5.1f}s  ({summary})" + (f"  ERR={log.error}" if log.error else ""),
                flush=True,
            )
        browser.close()

    passed = sum(r["ok"] for r in results)
    avg_clicks = float(np.mean([r["clicks"] for r in results])) if results else 0.0
    avg_secs = float(np.mean([r["elapsed_s"] for r in results])) if results else 0.0
    print(f"\n=== [{a.label}] {a.cond}: {passed}/{len(results)} ({100*passed/max(1,len(results)):.1f}%)  avg_clicks={avg_clicks:.1f}  avg_time={avg_secs:.1f}s  wall={time.monotonic()-t_all:.0f}s ===")
    out_path = out / f"{a.label}_{a.cond}.jsonl"
    out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8")
    print("saved ->", out_path)


if __name__ == "__main__":
    main()
