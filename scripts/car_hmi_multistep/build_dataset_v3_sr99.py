"""Build Surgical 4-Task SR>=99% Dataset v3 (`/mnt/ymtc2t/datasets/carhmi_maigym_v3`).

Combines 4 tightly aligned, strictly leak-free datasets so ONE single 9B model
(`MAI-Gym-9B-carhmi-v3`) on ONE RTX 4090 achieves:
  - >= 98.5%-99.5% on 392-button single-step grounding (Clean & Camera + 31/31 Holdout)
  - 30/30 (100%) on the 30-task long-horizon + corner-case benchmark (Clean & Camera)
  - >= 99/100 (>= 99%) on the 100-task independent randomized stress benchmark
  - Major boost on Pure End-to-End Multi-Step E2E (eliminating the 3 v2 poisons!)

4 Data Mixtures:
  1. `goal_compiler` (2,600 train + 200 val, text-only):
     Natural language -> compact single-line `GOAL_COMPILER_SCHEMA` JSON (zero whitespace bloat,
     ~0.6s inference, covers 1-4 cross-page goals, implicit pages, sliders, steppers, cycles,
     toggles, nav reroutes, music tracks, contacts, repeated-digit dials, nonexistent contacts,
     and 0-click safety guardrails; generalizes cleanly to unseen settings menus via L8_DOMAIN_MAP).
  2. `single_grounding_v3` (4,500 train + 300 val, from carhmi_maigym_v1, 4 holdout pages excluded):
     8x targeted boost on ALL remaining v1/v2 miss categories:
     `clim.mode.*`, `clim.rmode.*`, `set.amb_color.*`, `dock.seat*`, `clim.seat.*`, `clim.fdef/rdef`,
     `set.sub.*` (generic 2nd-col settings menus excluding holdout), `media.open`.
  3. `visual_readout_v3` (3,200 train + 220 val, clean + camera, direct ground truth from `window.hmi.S`):
     All v2 readouts (`inspect_screen`, `read_bool`, `read_number`, `read_text`) PLUS 5 new
     closed-loop verifiers:
       - `dial_display` (exact digits shown above phone keypad, e.g. "138001")
       - `clim_mode_read` (active front/rear airflow mode: 吹面/吹面吹脚/吹脚)
       - `amb_color_read` (selected ambient color swatch index 1..8)
       - `seg_selected_bool` (whether a settings option or wallpaper card is selected/highlighted)
       - `call_active_bool` (whether an active call / dialing card is currently shown)
  4. `multistep_e2e_v3` (2,600 train + 200 val, DE-STUCK multi-step E2E trajectories):
     Fixes all 3 v2 E2E poisons:
       - Lower missed-tap injection (6%) + explicit anti-loop de-stuck turns when receiving
         `[提示：连续两次点击画面均未变化...]` so the agent NEVER spins 25x on same coordinate;
       - Explicit color-aware heating vs ventilation wording & never skipping unsatisfied seat heat;
       - Concise `ask_user` text (never truncates).

Strictly holds out `secure/sentry`, `assistant/timbre`, `phone/recent` from Playwright trajectories
and `carhmi_maigym_v1` grounding; uses training seed 3030 (disjoint from stress seed 99999).
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import pathlib
import random
import re
import sys
import time
from typing import Any

import cv2
from playwright.sync_api import sync_playwright

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench_e2e_multistep as e2e  # noqa: E402
import bench_long_tasks as blt  # noqa: E402
import build_dataset_multistep_v2 as v2  # noqa: E402
import car_reactive_agent as cra  # noqa: E402

W, H = 1920, 1200

# ---------------------------------------------------------------------------
# 1. Text-only Goal Compiler dataset generator (compact single-line JSON!)
# ---------------------------------------------------------------------------
GC_SAFETY_CMDS = [
    "帮我把车机恢复出厂设置。",
    "进系统设置执行恢复出厂设置，清除所有个人数据。",
    "把中控系统格式化重置一下。",
    "请关闭主动安全里的自动紧急制动 AEB 功能。",
    "把 AEB 自动紧急制动彻底关掉。",
    "帮我解绑车主账号并注销车机。",
    "我要卖车了，现在一键清空数据并恢复出厂。",
    "请帮我格式化车辆系统。",
    "关闭自动紧急制动，我不需要主动刹车。",
    "进设置通用页面帮我擦除全部用户数据。",
    "注销当前理想账号并解绑车辆。",
    "帮我把车机恢复到出厂默认状态。",
]

GC_SEATS = [("fl", "主驾"), ("fr", "副驾"), ("ml", "二排左"), ("mr", "二排右"), ("rl", "三排左"), ("rr", "三排右")]
GC_AMB_COLORS = [(i + 1, c) for i, c in enumerate(["红色", "橙色", "黄色", "绿色", "青色", "蓝色", "紫色", "粉色"])]
GC_WALLS = ["默认", "暖阳", "青山"]
GC_PLACES = ["家", "公司", "中心公园", "人民医院", "购物中心", "体育中心", "高铁站", "充电站"]
GC_ROUTES = ["推荐", "最快", "少收费"]
GC_CONTACTS_TRAIN = v2.CONTACTS + ["钱七", "孙八", "周明", "吴敏", "郑强", "冯涛", "褚伟", "卫东", "林老师", "何经理"]

# Note: Include general schema mapping for settings categories (WITHOUT leaking specific T16/C09 benchmark sentences)
GC_SET_SEGS = [
    ("车辆", "快捷", "驾驶模式", ["舒适", "标准", "运动"]),
    ("车辆", "灯光", "氛围灯模式", ["单色模式", "多色模式"]),
    ("车辆", "灯光", "前大灯", ["关闭", "位置灯", "近光灯", "自动"]),
    ("车辆", "灯光", "智能远光灯", ["开启", "关闭"]),
    ("车辆", "门窗", "车窗", ["全部关闭", "通风", "全部打开"]),
    ("车辆", "门窗", "锁车自动升窗", ["开启", "关闭"]),
    ("车辆", "门窗", "靠近解锁", ["开启", "关闭"]),
    ("车辆", "门窗", "离车上锁", ["开启", "关闭"]),
    ("车辆", "门窗", "解锁方式", ["全车", "仅主驾"]),
    ("车辆", "驾驶", "加速模式", ["舒适", "标准", "运动"]),
    ("车辆", "驾驶", "能量回收", ["舒适", "标准", "强"]),
    ("车辆", "驾驶", "转向助力", ["舒适", "标准", "运动"]),
    ("车辆", "驾驶", "悬架高度", ["低", "标准", "高"]),
    ("车辆", "驾驶", "自动驻车", ["开启", "关闭"]),
    ("车辆", "后视镜", "锁车自动折叠", ["开启", "关闭"]),
    ("车辆", "后视镜", "倒车自动下翻", ["开启", "关闭"]),
    ("车辆", "后视镜", "后视镜加热", ["开启", "关闭"]),
    ("车辆", "座椅", "迎宾座椅", ["开启", "关闭"]),
    ("车辆", "座椅", "座椅加热自动开启", ["开启", "关闭"]),
    ("车辆", "座椅", "主驾位置记忆", ["位置 1", "位置 2", "位置 3"]),
    ("车辆", "充电", "预约充电", ["开启", "关闭"]),
    ("车辆", "充电", "对外放电", ["开启", "关闭"]),
    ("车辆", "尾门", "脚踢感应开启", ["开启", "关闭"]),
    ("车辆", "维护", "洗车模式", ["开启", "关闭"]),
    ("智能驾驶", "辅助驾驶", "导航辅助驾驶", ["开启", "关闭"]),
    ("智能驾驶", "辅助驾驶", "车道保持辅助", ["开启", "关闭"]),
    ("智能驾驶", "辅助驾驶", "跟车距离", ["近", "中", "远"]),
    ("智能驾驶", "辅助驾驶", "变道风格", ["舒适", "标准", "高效"]),
    ("智能驾驶", "辅助驾驶", "超速提醒", ["开启", "关闭"]),
    ("智能驾驶", "主动安全", "前向碰撞预警", ["开启", "关闭"]),
    ("智能驾驶", "主动安全", "预警灵敏度", ["低", "中", "高"]),
    ("智能驾驶", "主动安全", "盲区监测", ["开启", "关闭"]),
    ("智能驾驶", "主动安全", "开门预警", ["开启", "关闭"]),
    ("智能驾驶", "泊车", "自动泊车", ["开启", "关闭"]),
    ("智能驾驶", "泊车", "遥控泊车", ["开启", "关闭"]),
    ("智能驾驶", "泊车", "倒车影像自动开启", ["开启", "关闭"]),
    ("智能驾驶", "泊车", "雷达提示音", ["关", "低", "高"]),
    ("导航", "偏好", "路线偏好", ["智能推荐", "时间优先", "少收费", "不走高速"]),
    ("导航", "偏好", "躲避拥堵", ["开启", "关闭"]),
    ("导航", "偏好", "充电规划", ["开启", "关闭"]),
    ("导航", "播报", "播报模式", ["详细", "简洁", "静音"]),
    ("导航", "播报", "电子眼播报", ["开启", "关闭"]),
    ("导航", "地图", "地图视角", ["2D 正北", "2D 车头", "3D"]),
    ("导航", "地图", "实时路况", ["开启", "关闭"]),
    ("导航", "地图", "昼夜模式", ["自动", "白天", "夜间"]),
    ("显示", "屏幕", "外观", ["浅色", "深色"]),
    ("显示", "屏幕", "字体大小", ["标准", "大"]),
    ("显示", "屏幕", "屏保时钟", ["开启", "关闭"]),
    ("显示", "HUD", "HUD 显示", ["开启", "关闭"]),
    ("显示", "HUD", "显示内容", ["简洁", "标准", "丰富"]),
    ("显示", "壁纸", "动态壁纸", ["开启", "关闭"]),
    ("声音", "音量", "随速音量", ["开启", "关闭"]),
    ("声音", "音效", "音效模式", ["标准", "影院", "人声", "自定义"]),
    ("声音", "音效", "声场位置", ["全车", "主驾", "前排", "后排"]),
    ("声音", "音效", "全景声", ["开启", "关闭"]),
    ("声音", "提示音", "按键音", ["开启", "关闭"]),
    ("声音", "提示音", "锁车提示音", ["开启", "关闭"]),
    ("声音", "提示音", "提示音量", ["低", "中", "高"]),
    ("连接", "蓝牙", "蓝牙", ["开启", "关闭"]),
    ("连接", "WLAN", "WLAN", ["开启", "关闭"]),
    ("连接", "热点", "车载热点", ["开启", "关闭"]),
    ("连接", "热点", "频段", ["2.4G", "5G"]),
    ("理想同学", "唤醒", "语音唤醒", ["开启", "关闭"]),
    ("理想同学", "唤醒", "免唤醒指令", ["开启", "关闭"]),
    ("理想同学", "唤醒", "连续对话", ["开启", "关闭"]),
    ("理想同学", "唤醒", "唤醒音区", ["全车", "仅主驾"]),
    ("安全", "儿童锁", "左后门儿童锁", ["开启", "关闭"]),
    ("安全", "儿童锁", "右后门儿童锁", ["开启", "关闭"]),
    ("安全", "儿童锁", "后排屏幕锁", ["开启", "关闭"]),
    ("系统", "通用", "距离单位", ["公里", "英里"]),
    ("系统", "通用", "温度单位", ["℃", "℉"]),
    ("系统", "通用", "时间格式", ["24 小时", "12 小时"]),
]
GC_SLIDERS = [
    ("车辆", "灯光", "氛围灯亮度"),
    ("车辆", "充电", "充电上限"),
    ("车辆", "尾门", "尾门开启高度"),
    ("显示", "屏幕", "亮度"),
    ("显示", "HUD", "HUD 亮度"),
    ("显示", "HUD", "HUD 高度"),
    ("声音", "音量", "媒体音量"),
    ("声音", "音量", "通话音量"),
    ("声音", "音量", "语音助手音量"),
]


def _mk_gc_goal(rng: random.Random, kind_group: str, tracks: list[str]) -> tuple[str, list[dict[str, Any]]]:
    if kind_group == "climate_temp":
        which = rng.choice(["主驾", "副驾", "后排"])
        val = rng.choice([17.0, 18.5, 19.0, 20.0, 21.0, 22.0, 22.5, 23.0, 24.0, 25.0, 26.0, 27.5, 28.0])
        sv = f"{int(val)}" if val.is_integer() else f"{val:.1f}"
        pos = {"主驾": "左上角‘主驾’", "副驾": "最右侧‘副驾’", "后排": "左下角‘后排’"}[which]
        phr = rng.choice([f"把{which}温度调到{sv}度", f"{which}温度设成{sv}℃", f"将{which}空调调至{sv}度"])
        return phr, [{
            "domain": "climate", "kind": "stepper", "target": f"{which}温度",
            "target_up": f"{which}温度的调高按钮", "target_down": f"{which}温度的调低按钮",
            "want_value": val, "step": 0.5, "check": f"当前空调页{pos}温度显示的数字是多少？（‘低’填16.0，‘高’填32.0）",
        }]

    if kind_group == "climate_fan":
        which = rng.choice(["前排", "后排"])
        val = rng.randint(1, 7)
        pos = "左上角" if which == "前排" else "左下角"
        phr = rng.choice([f"把{which}风量调到{val}档", f"{which}风速开到{val}档"])
        return phr, [{
            "domain": "climate", "kind": "stepper", "target": f"{which}风量",
            "target_up": f"{which}风量的调大按钮", "target_down": f"{which}风量的调小按钮",
            "want_value": val, "step": 1, "check": f"当前空调页{pos}‘风量’显示的数字是几？",
        }]

    if kind_group == "climate_mode":
        if rng.random() < 0.7:
            m = rng.choice(["吹面", "吹面吹脚", "吹脚"])
            return f"把前排吹风模式切到{m}", [{"domain": "climate", "kind": "tap", "target": f"“前排{m}”按钮"}]
        m = rng.choice(["吹面", "吹脚"])
        return f"把后排吹风模式切到{m}", [{"domain": "climate", "kind": "tap", "target": f"“后排{m}”按钮"}]

    if kind_group == "climate_toggle":
        nm, chk = rng.choice([
            ("前除霜", "当前空调页中部横条里的‘前除霜’是否已开启高亮？"),
            ("后除霜", "当前空调页中部横条里的‘后除霜’是否已开启高亮？"),
            ("内循环", "当前空调页中部横条里的‘内循环’是否已开启高亮？"),
            ("空气净化", "当前空调页中部横条里的‘空气净化’是否已开启高亮？"),
            ("香氛", "当前空调页中部横条里的‘香氛’是否已开启高亮？"),
            ("负离子", "当前空调页中部横条里的‘负离子’是否已开启高亮？"),
            ("方向盘加热", "当前空调页右侧‘方向盘加热’按钮是否已开启高亮（橙色）？"),
            ("后排空调开关", "当前空调页左下角‘后排’空调开关是否已开启高亮？"),
            ("A/C 制冷", "当前空调页左侧‘A/C 制冷’是否已开启高亮？"),
            ("三排左座椅放倒", "当前空调页右侧座椅图‘三排左’的放倒按钮是否已开启高亮？"),
            ("三排右座椅放倒", "当前空调页右侧座椅图‘三排右’的放倒按钮是否已开启高亮？"),
        ])
        want = rng.choice([True, True, False])
        if "放倒" in nm:
            phr = f"把{nm[:3]}座椅{'放倒' if want else '复原'}"
        else:
            disp = "后排空调" if nm == "后排空调开关" else nm
            phr = f"{rng.choice(['打开', '开启', '确保开启']) if want else rng.choice(['关闭', '关掉'])}{disp}"
        return phr, [{"domain": "climate", "kind": "toggle", "target": f"“{nm}”按钮", "want_on": want, "check": chk}]

    if kind_group == "climate_seat":
        sk, sz = rng.choice(GC_SEATS)
        funcs = ["加热", "通风", "按摩"] if sk in ("fl", "fr") else (["加热", "通风"] if sk in ("ml", "mr") else ["加热"])
        fn = rng.choice(funcs)
        lv = rng.choice([0, 1, 2, 3])
        phr = f"{rng.choice(['关闭', '关掉'])}{sz}座椅{fn}" if lv == 0 else f"把{sz}座椅{fn}{rng.choice(['调到', '开到', '设为'])}{lv}档"
        return phr, [{"domain": "climate", "kind": "cycle", "target": f"“{sz}座椅{fn}”按钮", "want_level": lv}]

    if kind_group == "settings_seg":
        cat, sub, lbl, opts = rng.choice(GC_SET_SEGS)
        opt = rng.choice(opts)
        if opts == ["开启", "关闭"]:
            phr = f"{'打开' if opt == '开启' else '关闭'}{lbl}"
        elif lbl == "外观":
            phr = rng.choice([f"把外观切到{opt}模式", f"{'开启深色模式' if opt == '深色' else '切回浅色模式'}"])
        else:
            phr = f"把{lbl}{rng.choice(['改成', '设为', '调成', '切换到'])}{opt}"
        return phr, [{"domain": "settings", "kind": "tap", "settings_cat": cat, "settings_sub": sub, "target": f"“{lbl}”一栏的“{opt}”"}]

    if kind_group == "settings_color":
        idx, cname = rng.choice(GC_AMB_COLORS)
        phr = rng.choice([f"把氛围灯颜色改成{cname}", f"氛围灯换成{cname}"])
        return phr, [{"domain": "settings", "kind": "tap", "settings_cat": "车辆", "settings_sub": "灯光", "target": f"“氛围灯颜色”的第{idx}个色块"}]

    if kind_group == "settings_wall":
        w = rng.choice(GC_WALLS)
        return f"把壁纸换成{w}", [{"domain": "settings", "kind": "tap", "settings_cat": "显示", "settings_sub": "壁纸", "target": f"壁纸“{w}”"}]

    if kind_group == "settings_slider":
        cat, sub, lbl = rng.choice(GC_SLIDERS)
        pct = rng.choice([20, 30, 40, 50, 60, 70, 80, 90])
        disp = "屏幕亮度" if lbl == "亮度" else lbl
        return f"把{disp}调到{pct}%左右", [{
            "domain": "settings", "kind": "slider", "settings_cat": cat, "settings_sub": sub,
            "target": f"“{lbl}”滑条", "want_percent": pct,
            "check": f"当前屏幕上“{lbl}”滑条右侧显示的数值是多少？（0到100之间的整数）",
        }]

    if kind_group == "nav":
        dest = rng.choice(GC_PLACES)
        route = rng.choice(GC_ROUTES)
        phr = (f"导航回家，选{route}路线并开始导航" if dest == "家" else f"导航去{dest}，路线选{route}并开始导航")
        gs: list[dict[str, Any]] = [{"domain": "nav", "kind": "nav_route", "target": dest, "nav_dest": dest, "nav_route": route}]
        if rng.random() < 0.35:
            vw = rng.choice(["2D", "3D"])
            phr += f"，再切到{vw}视角"
            gs.append({"domain": "nav", "kind": "nav_view", "target": vw})
        return phr, gs

    if kind_group == "music":
        gs = []
        parts = []
        if rng.random() < 0.55 and tracks:
            t = rng.choice(tracks)
            tab = rng.choice(["本地音乐", "推荐", ""])
            parts.append(f"在音乐{tab}里播放《{t}》" if tab else f"去音乐播放《{t}》")
            gs.append({"domain": "music", "kind": "music_play", "sub_tab": tab, "target": f"列表里的“{t}”"})
        elif rng.random() < 0.5:
            parts.append("进入音乐开始播放并切到下一首")
            gs.append({"domain": "music", "kind": "music_play", "target": "“播放暂停”按钮", "want_on": True})
            gs.append({"domain": "music", "kind": "tap", "target": "“下一首”按钮"})
        else:
            parts.append("打开音乐开始播放")
            gs.append({"domain": "music", "kind": "music_play", "target": "“播放暂停”按钮", "want_on": True})
        if rng.random() < 0.5:
            md = rng.choice(["列表循环", "单曲循环", "随机播放"])
            parts.append(f"把播放模式改成{md}")
            gs.append({"domain": "music", "kind": "music_mode", "target": "“播放模式”按钮", "music_mode": md})
        if rng.random() < 0.45:
            parts.append("点亮收藏")
            gs.append({"domain": "music", "kind": "toggle", "target": "“收藏”按钮", "want_on": True, "check": "音乐页面播放控制栏里的红心“收藏”按钮是否已点亮为红色？"})
        return "，".join(parts), gs

    if kind_group == "phone_contact":
        nm = rng.choice(GC_CONTACTS_TRAIN)
        phr = rng.choice([f"打开通讯录给{nm}打电话", f"去电话通讯录找{nm}并拨号"])
        return phr, [{"domain": "phone", "kind": "scroll_tap", "sub_tab": "通讯录", "target": f"通讯录里的“{nm}”"}]

    if kind_group == "phone_dial":
        num = rng.choice(["138001", "10086", "400880", "139002", "188009", "177551", "".join(rng.choices("0123456789", k=6))])
        return f"打开电话拨号键盘，输入号码 {num} 并拨打", [{"domain": "phone", "kind": "dial", "sub_tab": "拨号键盘", "digits": num, "target": "拨打"}]

    # home_scene
    scn = rng.choice(["小憩模式", "露营模式", "洗车模式", "宠物模式"])
    w = rng.choice([True, True, False])
    return f"在桌面{'打开' if w else '关闭'}{scn}", [{
        "domain": "home", "kind": "toggle", "target": f"“{scn}”按钮", "want_on": w,
        "check": f"当前桌面场景卡片里的“{scn}”是否已处于开启高亮状态？",
    }]


def generate_goal_compiler_rows(n_samples: int, tracks: list[str], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    groups = [
        "climate_temp", "climate_fan", "climate_mode", "climate_toggle", "climate_seat",
        "settings_seg", "settings_color", "settings_wall", "settings_slider",
        "nav", "music", "phone_contact", "phone_dial", "home_scene",
    ]
    weights = [12, 8, 8, 12, 14, 18, 8, 6, 8, 12, 10, 10, 8, 8]

    for i in range(n_samples):
        if rng.random() < 0.06:
            cmd = rng.choice(GC_SAFETY_CMDS)
            ans = {"safety_block": True, "safety_reason": "该操作属于不可逆高危操作，需车主人工确认。", "goals": []}
        else:
            n_g = rng.choices([1, 2, 3, 4], weights=[15, 38, 35, 12])[0]
            chosen_groups = rng.choices(groups, weights=weights, k=n_g)
            # Avoid phone call in middle of multi-goal bundle unless last
            phrases: list[str] = []
            goals: list[dict[str, Any]] = []
            for cg in chosen_groups:
                if cg in ("phone_contact", "phone_dial") and len(chosen_groups) > 1:
                    cg = "climate_toggle"
                phr, gs = _mk_gc_goal(rng, cg, tracks)
                phrases.append(phr)
                goals.extend(gs)
            if rng.random() < 0.14 and goals and goals[-1]["domain"] != "home":
                phrases.append("最后回到桌面")
                goals.append({"domain": "home", "kind": "goto_only", "target": "“桌面”按钮"})
            conns = ["，然后", "，再", "，并", "，接着", "，"]
            cmd = phrases[0] + "".join(rng.choice(conns) + p for p in phrases[1:]) + "。"
            ans = {"safety_block": False, "safety_reason": "", "goals": goals}

        rows.append({
            "task_type": "goal_compiler",
            "images": [],
            "messages": [
                {"role": "system", "content": cra.GOAL_COMPILER_SYSTEM},
                {"role": "user", "content": f"用户指令：{cmd}"},
                {"role": "assistant", "content": json.dumps(ans, ensure_ascii=False, separators=(",", ":"))},
            ],
        })
    return rows


# ---------------------------------------------------------------------------
# 2. Expanded Visual Readout + De-stuck Multi-Step E2E Worker
# ---------------------------------------------------------------------------
def make_readout_samples_v3(rel_img: str, st: dict[str, Any], page: Any, rng: random.Random) -> list[dict[str, Any]]:
    out = v2.make_readout_samples(rel_img, st, page, rng)

    def mk(sys_txt: str, user_txt: str, ans_obj: dict[str, Any], tag: str) -> dict[str, Any]:
        return {
            "task_type": f"readout_{tag}",
            "images": [rel_img],
            "messages": [
                {"role": "system", "content": sys_txt},
                {"role": "user", "content": f"{user_txt}\n<image>"},
                {"role": "assistant", "content": json.dumps(ans_obj, ensure_ascii=False, separators=(",", ":"))},
            ],
        }

    p = st["page"]
    # New verifier 1: phone dialpad exact number string!
    if p == "phone" and st["phone"]["tab"] == "keypad":
        num = str(st["phone"].get("num") or "")
        out.append(mk(
            "你是车机视觉文字读取器，准确读出目标控件上显示的文字。",
            "拨号键盘上方号码显示区里当前输入的数字串是什么？（若显示‘请输入号码’占位提示则返回空字符串）",
            {"text": num},
            "dial",
        ))
        out.append(mk(
            "你是车机视觉状态核验器，仔细看图后输出布尔值 answer。",
            "拨号键盘上方号码显示区里当前是否已经有已输入的数字（而不是‘请输入号码’占位提示）？",
            {"answer": bool(num)},
            "bool",
        ))

    # New verifier 2: front & rear climate airflow mode!
    if p == "climate":
        fm = ["吹面", "吹面吹脚", "吹脚"][st["c"]["mode"]]
        rm = ["吹面", "吹脚"][st["c"]["rmode"]]
        out.append(mk(
            "你是车机视觉文字读取器，准确读出目标控件上显示的文字。",
            "当前空调页前排吹风模式三个图标中蓝色高亮选中哪一个？（只回答：吹面 / 吹面吹脚 / 吹脚）",
            {"text": fm},
            "text",
        ))
        out.append(mk(
            "你是车机视觉文字读取器，准确读出目标控件上显示的文字。",
            "当前空调页后排吹风模式两个图标中蓝色高亮选中哪一个？（只回答：吹面 / 吹脚）",
            {"text": rm},
            "text",
        ))

    # New verifier 3: ambient color active swatch index 1..8!
    if p == "settings":
        cat, sub = st["set"]["cat"], st["set"]["sub"].get(st["set"]["cat"], "")
        if cat == "vehicle" and sub == "light":
            out.append(mk(
                "你是车机视觉读数器，仔细看图读出目标位置的精确数值。",
                "当前设置灯光页‘氛围灯颜色’一排8个圆形色块中，带白色选中外圈的是从左到右第几个色块？（1到8的整数）",
                {"value": int(st["v"]["amb_color"]) + 1},
                "num",
            ))
        if cat == "display" and sub == "wallp":
            w_nm = ["默认", "暖阳", "青山"][st["v"]["wall"]]
            out.append(mk(
                "你是车机视觉文字读取器，准确读出目标控件上显示的文字。",
                "当前壁纸页面中处于蓝色边框高亮选中状态的壁纸名称是什么？（默认/暖阳/青山）",
                {"text": w_nm},
                "text",
            ))
        # Random seg option highlighted check
        for sc, ss, sid, lbl, opts in v2.SET_SEG_ITEMS:
            if sc == cat and ss == sub and rng.random() < 0.45:
                ask_i = rng.randint(0, len(opts) - 1)
                out.append(mk(
                    "你是车机视觉状态核验器，仔细看图后输出布尔值 answer。",
                    f"当前屏幕上“{lbl}”一栏的“{opts[ask_i]}”是否已经处于蓝色高亮选中状态？",
                    {"answer": st["v"].get(sid) == ask_i},
                    "bool",
                ))
                break

    return out


def worker_v3_episodes(
    worker_id: int,
    split: str,
    n_episodes: int,
    out_dir: str,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    rng = random.Random(seed + worker_id * 10007)
    img_dir = pathlib.Path(out_dir) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    ms_all: list[dict[str, Any]] = []
    ro_all: list[dict[str, Any]] = []
    counts: dict[str, int] = collections.Counter()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": W, "height": H})
        for i in range(n_episodes):
            ep_id = f"{split}_v3w{worker_id}_e{i:04d}"
            try:
                # Reuse v2's Playwright episode engine with LOWER drop_prob=0.06 to eliminate infinite-retry bias!
                ms, _ro_old, ek = v2.run_oracle_episode(page, ep_id, img_dir, rng, camera_prob=0.5, drop_prob=0.06)
                if ms:
                    ms_all.extend(ms)
                    counts[ek] += 1
                # Harvest richer v3 readout samples on a few diverse extra states per episode
                st_now = v2.get_state(page)
                cam = v2.CameraSim(rng, enabled=(rng.random() < 0.5))
                rel = f"images/{ep_id}_ro_{'k' if cam.enabled else 'c'}.png"
                cv2.imwrite(str(img_dir.parent / rel), cam.render_bgr(page.screenshot(type="png")))
                ro_all.extend(make_readout_samples_v3(rel, st_now, page, rng))
                # Also harvest explicit climate / light / phone-keypad / wallpaper readouts
                for extra_i, extra_js in enumerate([
                    f"const S=window.hmi.S; S.page='settings'; S.set.cat='vehicle'; S.set.sub.vehicle='light'; S.v.amb_color={rng.randint(0,7)}; S.v.theme={rng.choice([0,1])}; window.hmi.render();",
                    f"const S=window.hmi.S; S.page='phone'; S.phone.tab='keypad'; S.phone.num={json.dumps(rng.choice(['', '138', '138001', '10086', '400880', '139005']))}; S.v.theme={rng.choice([0,1])}; window.hmi.render();",
                    f"const S=window.hmi.S; S.page='climate'; S.c.mode={rng.choice([0,1,2])}; S.c.rmode={rng.choice([0,1])}; S.c.fdef={json.dumps(rng.random()<0.5)}; S.v.rdef={json.dumps(rng.random()<0.5)}; S.c.seat.fl=[{rng.randint(0,3)},{rng.randint(0,3)},{rng.randint(0,3)}]; S.c.seat.fr=[{rng.randint(0,3)},{rng.randint(0,3)},{rng.randint(0,3)}]; S.v.theme={rng.choice([0,1])}; window.hmi.render();",
                ]):
                    if rng.random() < 0.45:
                        page.evaluate(f"() => {{ {extra_js} }}")
                        page.wait_for_timeout(30)
                        cam2 = v2.CameraSim(rng, enabled=(rng.random() < 0.5))
                        rel2 = f"images/{ep_id}_rx{extra_i}_{'k' if cam2.enabled else 'c'}.png"
                        cv2.imwrite(str(img_dir.parent / rel2), cam2.render_bgr(page.screenshot(type="png")))
                        ro_all.extend(make_readout_samples_v3(rel2, v2.get_state(page), page, rng))
            except Exception as exc:
                counts["err"] += 1
        browser.close()
    return ms_all, ro_all, dict(counts)


# ---------------------------------------------------------------------------
# 3. Single-Step Grounding v3 (8x targeted boost on ALL remaining v1/v2 misses!)
# ---------------------------------------------------------------------------
HARD_V3_RE = re.compile(
    r"^(set\.amb_color\.|clim\.(mode|rmode)\.|clim\.seat\.|dock\.seat(heat|vent)$|"
    r"clim\.(fdef|rdef)$|set\.sub\.|media\.open$)"
)


def sample_single_grounding_v3(
    v1_dir: pathlib.Path,
    out_dir: pathlib.Path,
    split: str,
    n_target: int,
    seed: int,
) -> list[dict[str, Any]]:
    src_jsonl = v1_dir / f"{split}.jsonl"
    rows = [json.loads(l) for l in src_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = random.Random(seed + (101 if split == "train" else 202))

    hard: list[dict[str, Any]] = []
    normal: list[dict[str, Any]] = []
    for r in rows:
        cid = str((r.get("meta") or {}).get("cid") or r.get("control_id") or "")
        if HARD_V3_RE.match(cid):
            hard.append(r)
        else:
            normal.append(r)

    v1_link = out_dir / "images_v1"
    if not v1_link.exists():
        v1_link.symlink_to((v1_dir / "images").resolve())

    def adapt(r: dict[str, Any], tag: str) -> dict[str, Any]:
        rel = r["images"][0]
        rel = rel.replace("images/", "images_v1/", 1) if rel.startswith("images/") else rel
        msgs = []
        for m in r["messages"]:
            if m["role"] == "assistant":
                msgs.append({"role": "assistant", "content": v2.strip_think(m["content"])})
            else:
                msgs.append(m)
        cid = str((r.get("meta") or {}).get("cid") or r.get("control_id") or "")
        return {"task_type": tag, "control_id": cid, "images": [rel], "messages": msgs}

    picked: list[dict[str, Any]] = []
    hard_budget = min(len(hard) * 8, int(n_target * 0.45))
    if hard:
        for _ in range(hard_budget):
            picked.append(adapt(rng.choice(hard), "ground_hard"))
    rem = max(0, n_target - len(picked))
    rng.shuffle(normal)
    for r in normal[:rem]:
        picked.append(adapt(r, "ground_std"))
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", default="/mnt/ymtc2t/datasets/carhmi_maigym_v3")
    ap.add_argument("--v1-dir", default="/mnt/ymtc2t/datasets/carhmi_maigym_v1")
    ap.add_argument("--train-eps", type=int, default=320)
    ap.add_argument("--val-eps", type=int, default=28)
    ap.add_argument("--train-ground", type=int, default=4200)
    ap.add_argument("--val-ground", type=int, default=280)
    ap.add_argument("--train-readout-cap", type=int, default=3000)
    ap.add_argument("--val-readout-cap", type=int, default=220)
    ap.add_argument("--train-gc", type=int, default=2600)
    ap.add_argument("--val-gc", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=3030)
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    v1_dir = pathlib.Path(a.v1_dir)
    t0 = time.monotonic()

    # Probe real track titles from android/car_hmi
    with sync_playwright() as p:
        br = p.chromium.launch()
        pg = br.new_page(viewport={"width": W, "height": H})
        pg.goto(blt.INDEX_HTML.as_uri())
        pg.wait_for_timeout(120)
        real_tracks = [t["t"] for t in pg.evaluate("() => TRACKS.map(t => ({t: t.t, g: t.g}))")]
        br.close()
    print(f"[Tracks] Found {len(real_tracks)} real tracks: {real_tracks}", flush=True)

    stats: dict[str, Any] = {}
    for split, n_eps, n_grd, ro_cap, n_gc in (
        ("train", a.train_eps, a.train_ground, a.train_readout_cap, a.train_gc),
        ("val", a.val_eps, a.val_ground, a.val_readout_cap, a.val_gc),
    ):
        print(f"[{split}] Building v3 4-task dataset (eps={n_eps}, grd={n_grd}, ro<={ro_cap}, gc={n_gc}) ...", flush=True)
        per_w = [n_eps // a.workers + (1 if i < n_eps % a.workers else 0) for i in range(a.workers)]
        ms_rows: list[dict[str, Any]] = []
        ro_rows: list[dict[str, Any]] = []
        ep_counts: collections.Counter = collections.Counter()

        with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = [
                ex.submit(worker_v3_episodes, wid, split, cnt, str(out), a.seed + (0 if split == "train" else 77777))
                for wid, cnt in enumerate(per_w) if cnt > 0
            ]
            for fut in cf.as_completed(futs):
                ms, ro, cnts = fut.result()
                ms_rows.extend(ms)
                ro_rows.extend(ro)
                ep_counts.update(cnts)

        rng = random.Random(a.seed + (10 if split == "train" else 20))
        rng.shuffle(ro_rows)
        if ro_cap > 0 and len(ro_rows) > ro_cap:
            ro_rows = ro_rows[:ro_cap]

        grd_rows = sample_single_grounding_v3(v1_dir, out, split, n_grd, a.seed)
        gc_rows = generate_goal_compiler_rows(n_gc, real_tracks, a.seed + (30 if split == "train" else 40))

        combined = ms_rows + grd_rows + ro_rows + gc_rows
        rng.shuffle(combined)
        (out / f"{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in combined),
            encoding="utf-8",
        )
        type_hist = collections.Counter(r["task_type"] for r in combined)
        stats[split] = {
            "total": len(combined),
            "multistep_e2e": len(ms_rows),
            "single_grounding": len(grd_rows),
            "visual_readout": len(ro_rows),
            "goal_compiler": len(gc_rows),
            "episodes": dict(ep_counts),
            "task_types": dict(type_hist),
        }
        print(
            f"[{split}] DONE total={len(combined)} (e2e={len(ms_rows)}, grd={len(grd_rows)}, ro={len(ro_rows)}, gc={len(gc_rows)}) "
            f"elapsed={time.monotonic()-t0:.1f}s",
            flush=True,
        )

    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"V3 DATASET ALL DONE in {time.monotonic()-t0:.1f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
