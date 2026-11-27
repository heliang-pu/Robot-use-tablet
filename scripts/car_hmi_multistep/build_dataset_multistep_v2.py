"""Build Multi-Step Joint Fine-Tuning Dataset v2 (`/mnt/ymtc2t/datasets/carhmi_maigym_v2`).

Generates 3 synergistic, leak-free data mixtures for Tongyi-MAI/MAI-Gym-9B:
  1. `multistep_e2e`    : Real Playwright multi-step reactive trajectories with sliding-window
                          history (last 2 screenshots real, older collapsed to `[历史截图]`),
                          arbitrary start pages, dark mode, active call overlays, toggle
                          idempotency, 14% injected physical missed-taps (self-healing retry),
                          scroll-to-find contacts, nonexistent-item refusal (`terminate:failed`),
                          and safety guardrails (`ask_user` at 0 clicks).
                          Strictly HOLDS OUT `secure/sentry`, `assistant/timbre`, `phone/recent`
                          so T16 / C09 remain genuine unseen-page multi-step generalization tests!
  2. `single_grounding` : Sampled from carhmi_maigym_v1 (same 4 holdout pages excluded) with
                          5x targeted oversampling on v1 remaining hard icons (`amb_color`,
                          `clim.mode`, `clim.seat.*.vent/fold`, `media.open`, `fdef/rdef`).
  3. `visual_readout`   : Ground-truth structured JSON screen/value readout directly from
                          `window.hmi.S` (`inspect_screen`, `read_bool`, `read_number`, `read_text`)
                          covering thin-stroke icons like `前除霜` and orange-heat vs blue-vent
                          seat badges so a SINGLE v2 model excels at BOTH E2E and ReactiveAgent!

All 3 mixtures use 50% clean + 50% simulated wrist-camera degradation with exact affine
coordinate projection. Parallelized across Playwright worker processes for ~2 min total build.
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
import numpy as np
from playwright.sync_api import sync_playwright

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench_e2e_multistep as e2e  # noqa: E402
import bench_long_tasks as blt  # noqa: E402
import car_reactive_agent as cra  # noqa: E402

W, H = 1920, 1200
FEED = (1280, 800)
COORD_SCALE = 999.0

# 严格保留与单步测试一致的未见子页，确保 T16 / C09 为真正的未见子页多步泛化测试
HOLDOUT_SET_SUBS = {("secure", "sentry"), ("assistant", "timbre")}

AMB_COLORS = ["红色", "橙色", "黄色", "绿色", "青色", "蓝色", "紫色", "粉色"]
WALLS = ["默认", "暖阳", "青山"]
PLACES = ["家", "公司", "中心公园", "人民医院", "购物中心", "体育中心", "高铁站", "充电站"]
ROUTES = ["推荐", "最快", "少收费"]
MUSIC_TABS = ["推荐", "电台", "我的收藏", "本地音乐"]
MUSIC_MODES = ["列表循环", "单曲循环", "随机播放"]
CONTACTS = ["Aaron", "阿姨", "爸爸", "Bella", "道路救援", "公司前台", "客服中心", "李四", "妈妈", "王五", "张三", "赵六"]
FAKE_CONTACTS = ["钱七", "孙八", "周明", "吴敏", "郑强", "冯涛", "褚伟", "卫东", "林老师", "何经理"]

SET_CAT_ZH = {
    "vehicle": "车辆", "adas": "智能驾驶", "navset": "导航", "display": "显示",
    "sound": "声音", "conn": "连接", "assistant": "理想同学", "secure": "安全", "system": "系统",
}
SET_SUB_ZH = {
    ("vehicle", "quick"): "快捷", ("vehicle", "light"): "灯光", ("vehicle", "door"): "门窗",
    ("vehicle", "drive"): "驾驶", ("vehicle", "mirr"): "后视镜", ("vehicle", "seats"): "座椅",
    ("vehicle", "charge"): "充电", ("vehicle", "tail"): "尾门", ("vehicle", "maint"): "维护",
    ("adas", "assist"): "辅助驾驶", ("adas", "safety"): "主动安全", ("adas", "parking"): "泊车",
    ("navset", "pref"): "偏好", ("navset", "voice"): "播报", ("navset", "mapset"): "地图",
    ("display", "screen"): "屏幕", ("display", "hud"): "HUD", ("display", "wallp"): "壁纸",
    ("sound", "volume"): "音量", ("sound", "effect"): "音效", ("sound", "tone"): "提示音",
    ("conn", "bt"): "蓝牙", ("conn", "wlan"): "WLAN", ("conn", "hotspot"): "热点",
    ("assistant", "wake"): "唤醒", ("assistant", "timbre"): "音色",
    ("secure", "sentry"): "哨兵", ("secure", "child"): "儿童锁",
    ("system", "general"): "通用", ("system", "about"): "关于本车",
}
PHONE_TAB_ZH = {"recent": "最近通话", "contacts": "通讯录", "keypad": "拨号键盘", "bt": "蓝牙设置"}

SET_SEG_ITEMS: list[tuple[str, str, str, str, list[str]]] = [
    ("vehicle", "quick", "drive_mode", "驾驶模式", ["舒适", "标准", "运动"]),
    ("vehicle", "light", "amb_mode", "氛围灯模式", ["单色模式", "多色模式"]),
    ("vehicle", "light", "headlamp", "前大灯", ["关闭", "位置灯", "近光灯", "自动"]),
    ("vehicle", "light", "auto_high", "智能远光灯", ["开启", "关闭"]),
    ("vehicle", "door", "win_all", "车窗", ["全部关闭", "通风", "全部打开"]),
    ("vehicle", "door", "lock_winup", "锁车自动升窗", ["开启", "关闭"]),
    ("vehicle", "door", "near_unlock", "靠近解锁", ["开启", "关闭"]),
    ("vehicle", "door", "leave_lock", "离车上锁", ["开启", "关闭"]),
    ("vehicle", "door", "unlock_way", "解锁方式", ["全车", "仅主驾"]),
    ("vehicle", "drive", "accel", "加速模式", ["舒适", "标准", "运动"]),
    ("vehicle", "drive", "regen", "能量回收", ["舒适", "标准", "强"]),
    ("vehicle", "drive", "steer", "转向助力", ["舒适", "标准", "运动"]),
    ("vehicle", "drive", "susp", "悬架高度", ["低", "标准", "高"]),
    ("vehicle", "drive", "auto_hold", "自动驻车", ["开启", "关闭"]),
    ("vehicle", "mirr", "mir_autofold", "锁车自动折叠", ["开启", "关闭"]),
    ("vehicle", "mirr", "mir_tilt", "倒车自动下翻", ["开启", "关闭"]),
    ("vehicle", "mirr", "mir_heat", "后视镜加热", ["开启", "关闭"]),
    ("vehicle", "seats", "seat_welcome", "迎宾座椅", ["开启", "关闭"]),
    ("vehicle", "seats", "seat_autoheat", "座椅加热自动开启", ["开启", "关闭"]),
    ("vehicle", "seats", "seat_mem", "主驾位置记忆", ["位置 1", "位置 2", "位置 3"]),
    ("vehicle", "charge", "chg_sched", "预约充电", ["开启", "关闭"]),
    ("vehicle", "charge", "v2l", "对外放电", ["开启", "关闭"]),
    ("vehicle", "charge", "chg_capauto", "充电口盖自动关闭", ["开启", "关闭"]),
    ("vehicle", "tail", "tail_kick", "脚踢感应开启", ["开启", "关闭"]),
    ("vehicle", "maint", "wash_mode", "洗车模式", ["开启", "关闭"]),
    ("adas", "assist", "noa", "导航辅助驾驶", ["开启", "关闭"]),
    ("adas", "assist", "lka", "车道保持辅助", ["开启", "关闭"]),
    ("adas", "assist", "follow", "跟车距离", ["近", "中", "远"]),
    ("adas", "assist", "lane_style", "变道风格", ["舒适", "标准", "高效"]),
    ("adas", "assist", "speed_warn", "超速提醒", ["开启", "关闭"]),
    ("adas", "safety", "fcw", "前向碰撞预警", ["开启", "关闭"]),
    ("adas", "safety", "fcw_sens", "预警灵敏度", ["低", "中", "高"]),
    ("adas", "safety", "bsd", "盲区监测", ["开启", "关闭"]),
    ("adas", "safety", "dow", "开门预警", ["开启", "关闭"]),
    ("adas", "parking", "apa", "自动泊车", ["开启", "关闭"]),
    ("adas", "parking", "rpa", "遥控泊车", ["开启", "关闭"]),
    ("adas", "parking", "rvc_auto", "倒车影像自动开启", ["开启", "关闭"]),
    ("adas", "parking", "radar_vol", "雷达提示音", ["关", "低", "高"]),
    ("navset", "pref", "route_pref", "路线偏好", ["智能推荐", "时间优先", "少收费", "不走高速"]),
    ("navset", "pref", "avoid_jam", "躲避拥堵", ["开启", "关闭"]),
    ("navset", "pref", "chg_plan", "充电规划", ["开启", "关闭"]),
    ("navset", "voice", "guide_mode", "播报模式", ["详细", "简洁", "静音"]),
    ("navset", "voice", "cam_warn", "电子眼播报", ["开启", "关闭"]),
    ("navset", "mapset", "map_view", "地图视角", ["2D 正北", "2D 车头", "3D"]),
    ("navset", "mapset", "traffic", "实时路况", ["开启", "关闭"]),
    ("navset", "mapset", "map_day", "昼夜模式", ["自动", "白天", "夜间"]),
    ("display", "screen", "theme", "外观", ["浅色", "深色"]),
    ("display", "screen", "font", "字体大小", ["标准", "大"]),
    ("display", "screen", "screensaver", "屏保时钟", ["开启", "关闭"]),
    ("display", "hud", "hud_on", "HUD 显示", ["开启", "关闭"]),
    ("display", "hud", "hud_mode", "显示内容", ["简洁", "标准", "丰富"]),
    ("display", "wallp", "wall_dyn", "动态壁纸", ["开启", "关闭"]),
    ("sound", "volume", "speed_vol", "随速音量", ["开启", "关闭"]),
    ("sound", "effect", "eq", "音效模式", ["标准", "影院", "人声", "自定义"]),
    ("sound", "effect", "field", "声场位置", ["全车", "主驾", "前排", "后排"]),
    ("sound", "effect", "surround", "全景声", ["开启", "关闭"]),
    ("sound", "tone", "key_tone", "按键音", ["开启", "关闭"]),
    ("sound", "tone", "lock_tone", "锁车提示音", ["开启", "关闭"]),
    ("sound", "tone", "tone_vol", "提示音量", ["低", "中", "高"]),
    ("conn", "bt", "bt_on", "蓝牙", ["开启", "关闭"]),
    ("conn", "wlan", "wlan_on", "WLAN", ["开启", "关闭"]),
    ("conn", "hotspot", "hot_on", "车载热点", ["开启", "关闭"]),
    ("conn", "hotspot", "hot_band", "频段", ["2.4G", "5G"]),
    ("assistant", "wake", "wake_on", "语音唤醒", ["开启", "关闭"]),
    ("assistant", "wake", "wake_free", "免唤醒指令", ["开启", "关闭"]),
    ("assistant", "wake", "cont_talk", "连续对话", ["开启", "关闭"]),
    ("assistant", "wake", "wake_zone", "唤醒音区", ["全车", "仅主驾"]),
    ("secure", "child", "child_l", "左后门儿童锁", ["开启", "关闭"]),
    ("secure", "child", "child_r", "右后门儿童锁", ["开启", "关闭"]),
    ("secure", "child", "rear_lock", "后排屏幕锁", ["开启", "关闭"]),
    ("system", "general", "unit_dist", "距离单位", ["公里", "英里"]),
    ("system", "general", "unit_temp", "温度单位", ["℃", "℉"]),
    ("system", "general", "time_fmt", "时间格式", ["24 小时", "12 小时"]),
]

SET_SLIDER_ITEMS: list[tuple[str, str, str, str]] = [
    ("vehicle", "light", "amb_bright", "氛围灯亮度"),
    ("vehicle", "charge", "chg_limit", "充电上限"),
    ("vehicle", "tail", "tail_height", "尾门开启高度"),
    ("display", "screen", "bright", "屏幕亮度"),
    ("display", "hud", "hud_bright", "HUD 亮度"),
    ("display", "hud", "hud_height", "HUD 高度"),
    ("sound", "volume", "vol_media", "媒体音量"),
    ("sound", "volume", "vol_call", "通话音量"),
    ("sound", "volume", "vol_voice", "语音助手音量"),
]

SAFETY_INSTRUCTIONS = [
    "帮我把车机恢复出厂设置，清空所有数据。",
    "进系统设置执行恢复出厂设置。",
    "把车辆系统格式化重置一下。",
    "请帮我关闭自动紧急制动 AEB 功能。",
    "把主动安全里的 AEB 自动紧急制动彻底关掉。",
    "帮我解绑当前车主账号并注销车机登录。",
    "清除车机全部用户数据并恢复初始出厂状态。",
    "我要卖车了，现在帮我一键恢复出厂设置。",
    "把自动紧急制动关掉，我不需要主动刹车。",
    "进入车辆设置把账号解绑并清空个人记录。",
]

_THINK_STRIP_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def strip_think(text: str) -> str:
    return _THINK_STRIP_RE.sub("", text).strip()


class CameraSim:
    """腕部相机仿射与光照退化仿真，同一轨迹保持固定机位并带轻微帧间噪声。"""

    def __init__(self, rng: random.Random, enabled: bool) -> None:
        self.enabled = enabled
        self.rng = rng
        if not enabled:
            self.M = np.array([[FEED[0] / W, 0, 0], [0, FEED[1] / H, 0]], dtype=np.float64)
            return
        ang = rng.uniform(-2.0, 2.0)
        sc = rng.uniform(0.95, 1.03)
        tx = rng.uniform(-16.0, 16.0)
        ty = rng.uniform(-10.0, 10.0)
        base_scale = np.array([[FEED[0] / W, 0, 0], [0, FEED[1] / H, 0], [0, 0, 1]], dtype=np.float64)
        rot = cv2.getRotationMatrix2D((FEED[0] / 2, FEED[1] / 2), ang, sc)
        rot[0, 2] += tx
        rot[1, 2] += ty
        rot3 = np.vstack([rot, [0, 0, 1]])
        self.M = (rot3 @ base_scale)[:2, :]
        self.alpha = rng.uniform(0.84, 1.15)
        self.beta = rng.uniform(-18.0, 18.0)
        self.wb = np.array([rng.uniform(0.93, 1.07), rng.uniform(0.95, 1.05), rng.uniform(0.93, 1.07)], dtype=np.float32)
        self.blur_k = rng.choice([0, 0, 3])
        self.jpeg_q = rng.randint(58, 86)

    def render_bgr(self, raw_png_bytes: bytes) -> np.ndarray:
        arr = np.frombuffer(raw_png_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if not self.enabled:
            return cv2.resize(bgr, FEED, interpolation=cv2.INTER_AREA)
        warped = cv2.warpAffine(bgr, self.M, FEED, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(18, 18, 22))
        f = warped.astype(np.float32) * self.alpha + self.beta
        f *= self.wb[None, None, :]
        noise = np.random.RandomState(self.rng.randint(0, 2**31 - 1)).normal(0, 2.2, f.shape).astype(np.float32)
        f = np.clip(f + noise, 0, 255).astype(np.uint8)
        if self.blur_k >= 3:
            f = cv2.GaussianBlur(f, (self.blur_k, self.blur_k), 0)
        ok, enc = cv2.imencode(".jpg", f, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_q])
        if ok:
            f = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return f

    def stage_to_norm999(self, x: float, y: float) -> tuple[int, int]:
        px, py = self.M @ np.array([x, y, 1.0], dtype=np.float64)
        nx = int(round(max(0.0, min(COORD_SCALE, px / FEED[0] * COORD_SCALE))))
        ny = int(round(max(0.0, min(COORD_SCALE, py / FEED[1] * COORD_SCALE))))
        return nx, ny


def fmt_tool_call(args: dict[str, Any], status_line: str = "") -> str:
    payload = json.dumps({"name": "mobile_use", "arguments": args}, ensure_ascii=False)
    tc = f"<tool_call>\n{payload}\n</tool_call>"
    return f"状态：{status_line}\n{tc}" if status_line else tc


def get_state(page: Any) -> dict[str, Any]:
    return json.loads(page.evaluate("() => window.hmi.state()"))


def get_box(page: Any, cid: str) -> tuple[float, float, float, float] | None:
    """仅当控件在滚动视口内完全可见时返回其在 1920x1200 stage 上的边界框。"""
    res = page.evaluate(
        """(cid) => {
          const st = document.getElementById('stage');
          const sr = st.getBoundingClientRect();
          const sc = Math.min(window.innerWidth / 1920, window.innerHeight / 1200);
          const el = st.querySelector('[data-id="' + cid + '"]');
          if (!el) return null;
          const q = el.getBoundingClientRect();
          if (q.width < 2 || q.height < 2) return null;
          const scrollParent = el.closest('.scroll');
          if (scrollParent) {
            const sp = scrollParent.getBoundingClientRect();
            if (q.top < sp.top + 4 || q.bottom > sp.bottom - 4) return null;
          }
          const x = (q.left - sr.left) / sc, y = (q.top - sr.top) / sc;
          const w = q.width / sc, h = q.height / sc;
          if (y + h < 64 || y > 1200) return null;
          return [x, y, w, h];
        }""",
        cid,
    )
    return tuple(res) if res else None


def page_label_zh(st: dict[str, Any]) -> str:
    p = st["page"]
    if p == "home":
        return "桌面"
    if p == "climate":
        return "空调页"
    if p == "settings":
        cat = st["set"]["cat"]
        sub = st["set"]["sub"].get(cat, "")
        return f"设置/{SET_CAT_ZH.get(cat, cat)}/{SET_SUB_ZH.get((cat, sub), sub)}"
    if p == "nav":
        ns = st["nav"]["st"]
        return {"idle": "导航首页", "search": "导航搜索页", "route": "导航路线页", "guide": "导航引导中"}.get(ns, "导航页")
    if p == "music":
        return f"音乐/{MUSIC_TABS[st['m']['tab']]}"
    if p == "phone":
        return f"电话/{PHONE_TAB_ZH.get(st['phone']['tab'], '')}"
    if p == "apps":
        return "应用中心"
    return "应用演示页"


def screen_state_json(st: dict[str, Any]) -> dict[str, Any]:
    p = st["page"]
    page_out = f"nav_{st['nav']['st']}" if p == "nav" else p
    has_call = bool(st["phone"]["call"] is not None and not st["phone"]["hide"] and p != "climate")
    cat_zh = SET_CAT_ZH.get(st["set"]["cat"], "") if p == "settings" else ""
    sub_key = st["set"]["sub"].get(st["set"]["cat"], "") if p == "settings" else ""
    sub_zh = SET_SUB_ZH.get((st["set"]["cat"], sub_key), "") if p == "settings" else ""
    sub_tab = ""
    if p == "phone":
        sub_tab = PHONE_TAB_ZH.get(st["phone"]["tab"], "")
    elif p == "music":
        sub_tab = MUSIC_TABS[st["m"]["tab"]]
    return {
        "page": page_out,
        "has_call_overlay": has_call,
        "settings_cat": cat_zh,
        "settings_sub": sub_zh,
        "sub_tab": sub_tab,
    }


def make_readout_samples(
    rel_img: str,
    st: dict[str, Any],
    page: Any,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """根据当前页面真实状态 window.hmi.S 生成结构化 JSON 视觉读屏训练样本。"""
    out: list[dict[str, Any]] = []

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

    ss_prompt = (
        "观察当前理想 L8 车机屏幕截图，判断：\n"
        "1. page（当前主页面）：\n"
        "   - home：右侧有导航预览卡片+天气+续航+胎压+场景卡片\n"
        "   - climate：全屏空调与6座座椅控制页（左侧前/后排温度风量，右侧6个座位图，无左侧P档黑车卡片）\n"
        "   - settings：右侧分为两列灰蓝设置目录（车辆/智能驾驶/导航/显示/声音/连接/理想同学/安全/系统）和右边设置项\n"
        "   - nav_idle：大地图，左上角有‘搜索目的地’框和‘回家/去公司/充电站’筹码\n"
        "   - nav_search：大地图左上角弹出‘选择目的地’列表卡片\n"
        "   - nav_route：大地图左上角弹出路线选择卡片（有‘推荐/最快/少收费’和蓝色‘开始导航’按钮）\n"
        "   - nav_guide：大地图左上角是蓝色转向引导卡（如‘300 米’），底部有红色‘退出导航’按钮\n"
        "   - music：右侧标题为‘音乐’，有封面、播放进度条、曲目列表\n"
        "   - phone：右侧标题为‘电话’，左侧菜单有最近通话/通讯录/拨号键盘/蓝牙设置\n"
        "   - apps：右侧标题为‘应用中心’，6x3彩色应用图标网格\n"
        "   - stub：单个应用占位演示界面，左上角有‘返回’按钮\n"
        "2. has_call_overlay：左侧面板是否正在显示通话界面（有红色挂断圆钮和数字拨号键盘，盖住了P档车体）？\n"
        "3. settings_cat / settings_sub：若 page==settings，填左侧第一列和第二列中白色高亮选中的中文菜单名；否则填空字符串。\n"
        "4. sub_tab：若 page==phone 或 page==music，填左侧白色高亮选中的标签名（如‘通讯录’/‘拨号键盘’/‘最近通话’/‘本地音乐’/‘推荐’）；否则填空字符串。"
    )
    out.append(mk(
        "你是车机界面状态识别器，严格输出紧凑单行 JSON，不要多余换行。",
        ss_prompt,
        screen_state_json(st),
        "screen",
    ))

    p = st["page"]
    if p == "climate":
        c = st["c"]
        bool_pool = [
            ("当前空调页中部横条里的‘前除霜’是否已开启高亮？", bool(c["fdef"])),
            ("当前空调页中部横条里的‘后除霜’是否已开启高亮？", bool(st["v"]["rdef"])),
            ("当前空调页中部横条里的‘内循环’是否已开启高亮？", bool(c["rec"])),
            ("当前空调页中部横条里的‘空气净化’是否已开启高亮？", bool(c["pure"])),
            ("当前空调页中部横条里的‘香氛’是否已开启高亮？", bool(c["scent"])),
            ("当前空调页中部横条里的‘负离子’是否已开启高亮？", bool(c["ion"])),
            ("当前空调页右侧‘方向盘加热’按钮是否已开启高亮（橙色）？", bool(st["v"]["wheel"])),
            ("当前空调页左下角‘后排’空调开关是否已开启高亮？", bool(c["ron"])),
            ("当前空调页左侧‘A/C 制冷’是否已开启高亮？", bool(c["ac"])),
            ("当前空调页右侧座椅图‘三排左’的放倒按钮是否已开启高亮？", bool(c["seat"]["rl"][2])),
            ("当前空调页右侧座椅图‘三排右’的放倒按钮是否已开启高亮？", bool(c["seat"]["rr"][2])),
        ]
        for q, v in rng.sample(bool_pool, k=min(4, len(bool_pool))):
            out.append(mk("你是车机视觉状态核验器，仔细看图后输出布尔值 answer。", q, {"answer": v}, "bool"))

        num_pool: list[tuple[str, float | int]] = [
            ("当前空调页左上角‘主驾’温度显示的数字是多少？（‘低’填16.0，‘高’填32.0）", float(c["tL"])),
            ("当前空调页最右侧‘副驾’温度显示的数字是多少？（‘低’填16.0，‘高’填32.0）", float(c["tR"])),
            ("当前空调页左下角‘后排’温度显示的数字是多少？（‘低’填16.0，‘高’填32.0）", float(c["rt"])),
            ("当前空调页左上角‘风量’显示的数字是几？", int(c["fan"])),
            ("当前空调页左下角‘风量’显示的数字是几？", int(c["rfan"])),
        ]
        for sk, sz in (("fl", "主驾"), ("fr", "副驾"), ("ml", "二排左"), ("mr", "二排右")):
            num_pool.append((
                f"空调页右侧座椅图‘{sz}’格内从上往下第1个【座椅加热】按钮当前是几档？（未亮橙色返回0；若亮橙色且右上角有数字角标则返回该数字1、2或3；不要看下面的蓝色通风按钮）",
                int(c["seat"][sk][0]),
            ))
            num_pool.append((
                f"空调页右侧座椅图‘{sz}’格内从上往下第2个【座椅通风】按钮当前是几档？（未亮蓝色返回0；若亮蓝色且右上角有数字角标则返回该数字1、2或3；不要看上面的橙色加热按钮）",
                int(c["seat"][sk][1]),
            ))
        for sk, sz in (("fl", "主驾"), ("fr", "副驾")):
            num_pool.append((
                f"空调页右侧座椅图‘{sz}’格内从上往下第3个【座椅按摩】按钮当前是几档？（未亮紫色返回0；若亮紫色且右上角有数字角标则返回该数字1、2或3）",
                int(c["seat"][sk][2]),
            ))
        for q, v in rng.sample(num_pool, k=min(4, len(num_pool))):
            out.append(mk("你是车机视觉读数器，仔细看图读出目标位置的精确数值。", q, {"value": v}, "num"))

    elif p == "home":
        scns = [
            ("小憩模式", bool(st["v"]["nap"])),
            ("露营模式", bool(st["scn"].get("camp"))),
            ("洗车模式", bool(st["scn"].get("wash"))),
            ("宠物模式", bool(st["scn"].get("pet"))),
        ]
        name, val = rng.choice(scns)
        out.append(mk(
            "你是车机视觉状态核验器，仔细看图后输出布尔值 answer。",
            f"当前桌面场景卡片里的“{name}”是否已处于开启高亮状态？",
            {"answer": val},
            "bool",
        ))

    elif p == "music":
        out.append(mk(
            "你是车机视觉状态核验器，仔细看图后输出布尔值 answer。",
            "音乐主播放按钮（蓝色圆形大按钮）里显示的是两条竖线（暂停图标，表示正在播放）吗？",
            {"answer": bool(st["m"]["play"])},
            "bool",
        ))
        out.append(mk(
            "你是车机视觉文字读取器，准确读出目标控件上显示的文字。",
            "音乐页面播放控制栏中播放模式按钮上当前显示的文字是什么（列表循环/单曲循环/随机播放）？",
            {"text": MUSIC_MODES[st["m"]["mode"]]},
            "text",
        ))
        out.append(mk(
            "你是车机视觉状态核验器，仔细看图后输出布尔值 answer。",
            "音乐页面播放控制栏里的红心“收藏”按钮是否已点亮为红色？",
            {"answer": bool(st["m"]["fav"].get(str(st["m"]["i"])))},
            "bool",
        ))

    elif p == "nav":
        out.append(mk(
            "你是车机视觉文字读取器，准确读出目标控件上显示的文字。",
            "地图右侧悬浮按钮上当前显示的是‘2D’还是‘3D’？",
            {"text": "3D" if st["nav"]["view"] else "2D"},
            "text",
        ))
        if st["nav"]["st"] == "guide" and st["nav"]["dest"] is not None:
            actual = PLACES[st["nav"]["dest"]]
            ask_dest = actual if rng.random() < 0.5 else rng.choice(PLACES)
            out.append(mk(
                "你是车机视觉状态核验器，仔细看图后输出布尔值 answer。",
                f"底部导航信息栏是否显示正在前往“{ask_dest}”？",
                {"answer": ask_dest == actual},
                "bool",
            ))

    elif p == "phone" and st["phone"]["tab"] == "contacts":
        cand_name = rng.choice(CONTACTS + FAKE_CONTACTS[:4])
        vis = False
        if cand_name in CONTACTS:
            idx = CONTACTS.index(cand_name)
            vis = get_box(page, f"phone.contact.{idx}") is not None
        out.append(mk(
            "你是车机视觉状态核验器，仔细看图后输出布尔值 answer。",
            f"当前屏幕列表中是否清晰可见名为“{cand_name}”的条目？",
            {"answer": vis},
            "bool",
        ))

    elif p == "settings":
        cat, sub = st["set"]["cat"], st["set"]["sub"].get(st["set"]["cat"], "")
        for sc, ss, sid, slabel in SET_SLIDER_ITEMS:
            if sc == cat and ss == sub:
                out.append(mk(
                    "你是车机视觉读数器，仔细看图读出目标位置的精确数值。",
                    f"当前屏幕上“{slabel}”滑条右侧显示的数值是多少？（0到100之间的整数）",
                    {"value": int(st["v"][sid])},
                    "num",
                ))
                break

    return out


def randomize_initial_state(page: Any, rng: random.Random) -> None:
    """将车机模拟页置于多样化的真实初始状态（任意起点、深浅模式、来电浮窗、导航中）。"""
    start_page = rng.choices(
        ["home", "climate", "settings", "nav", "music", "phone", "apps", "stub"],
        weights=[30, 16, 16, 14, 9, 8, 5, 2],
    )[0]
    theme = 1 if rng.random() < 0.22 else 0
    wall = rng.choice([0, 1, 2])
    tL = rng.choice([18.0, 19.5, 20.0, 21.5, 22.0, 23.0, 24.0, 25.0, 26.5, 28.0])
    tR = rng.choice([19.0, 21.0, 22.0, 24.0, 25.5, 27.0])
    rt = rng.choice([20.0, 22.0, 24.0, 26.0, 27.5])
    fan = rng.randint(1, 7)
    rfan = rng.randint(1, 7)
    ron = rng.random() < 0.45
    fdef = rng.random() < 0.35
    rdef = rng.random() < 0.35
    rec = rng.random() < 0.40
    pure = rng.random() < 0.60
    scent = rng.random() < 0.35
    ion = rng.random() < 0.35
    wheel = rng.random() < 0.35
    mode = rng.choice([0, 1, 2])
    rmode = rng.choice([0, 1])
    seat_fl = [rng.choice([0, 0, 1, 2, 3]), rng.choice([0, 1, 2, 3]), rng.choice([0, 0, 1, 2, 3])]
    seat_fr = [rng.choice([0, 0, 1, 2, 3]), rng.choice([0, 1, 2, 3]), rng.choice([0, 0, 1, 2, 3])]
    seat_ml = [rng.choice([0, 0, 1, 2, 3]), rng.choice([0, 0, 1, 2, 3]), 0]
    seat_mr = [rng.choice([0, 0, 1, 2, 3]), rng.choice([0, 0, 1, 2, 3]), 0]
    seat_rl = [rng.choice([0, 0, 1, 2]), 0, rng.choice([0, 1])]
    seat_rr = [rng.choice([0, 0, 1, 2]), 0, rng.choice([0, 1])]

    safe_subs = [k for k in SET_SUB_ZH.keys() if k not in HOLDOUT_SET_SUBS]
    scat, ssub = rng.choice(safe_subs)
    nav_st = rng.choices(["idle", "guide", "route", "search"], weights=[45, 35, 10, 10])[0]
    nav_dest = rng.randint(0, 7) if nav_st in ("guide", "route") else None
    nav_opt = rng.choice([0, 1, 2])
    nav_view = rng.choice([0, 1])
    m_tab = rng.choice([0, 1, 2, 3])
    m_i = rng.randint(0, 4)
    m_play = rng.random() < 0.5
    m_mode = rng.choice([0, 1, 2])
    p_tab = rng.choice(["contacts", "keypad", "bt"])
    p_num = rng.choice(["", "", "13900"])
    has_call = rng.random() < 0.14 and start_page != "climate"

    js = f"""
    const S = window.hmi.S;
    S.page = {json.dumps(start_page)};
    S.prev = 'home';
    S.v.theme = {theme}; S.v.wall = {wall};
    S.v.rdef = {json.dumps(rdef)}; S.v.wheel = {json.dumps(wheel)};
    S.v.nap = {json.dumps(rng.random() < 0.3)};
    S.scn = {{ camp: {json.dumps(rng.random() < 0.25)}, wash: {json.dumps(rng.random() < 0.2)}, pet: {json.dumps(rng.random() < 0.2)} }};
    S.c.tL = {tL}; S.c.tR = {tR}; S.c.rt = {rt};
    S.c.fan = {fan}; S.c.rfan = {rfan}; S.c.ron = {json.dumps(ron)};
    S.c.fdef = {json.dumps(fdef)}; S.c.rec = {json.dumps(rec)}; S.c.pure = {json.dumps(pure)};
    S.c.scent = {json.dumps(scent)}; S.c.ion = {json.dumps(ion)};
    S.c.mode = {mode}; S.c.rmode = {rmode};
    S.c.seat = {{ fl: {seat_fl}, fr: {seat_fr}, ml: {seat_ml}, mr: {seat_mr}, rl: {seat_rl}, rr: {seat_rr} }};
    S.set.cat = {json.dumps(scat)}; S.set.sub[{json.dumps(scat)}] = {json.dumps(ssub)};
    S.nav.st = {json.dumps(nav_st)}; S.nav.dest = {json.dumps(nav_dest)};
    S.nav.opt = {nav_opt}; S.nav.view = {nav_view};
    S.m.tab = {m_tab}; S.m.i = {m_i}; S.m.play = {json.dumps(m_play)}; S.m.mode = {m_mode};
    S.phone.tab = {json.dumps(p_tab)}; S.phone.num = {json.dumps(p_num)};
    if ({json.dumps(has_call)}) {{
      S.phone.call = {{ num: '13800000103', name: '爸爸', st: 'talk', t0: Date.now() - 25000, dtmf: '' }};
      S.phone.hide = false;
    }} else {{
      S.phone.call = null; S.phone.hide = false;
    }}
    window.hmi.render();
    """
    page.evaluate(f"() => {{ {js} }}")
    page.wait_for_timeout(40)


def sample_goal_bundle(rng: random.Random, st0: dict[str, Any], tracks: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]], str]:
    """随机生成自然语言多步指令与对应结构化目标列表。"""
    ep_kind = rng.choices(
        ["climate", "settings", "nav", "music", "phone_contact", "phone_dial", "cross_multi", "not_found", "safety"],
        weights=[22, 22, 12, 8, 8, 5, 22, 6, 6],
    )[0]

    if ep_kind == "safety":
        return rng.choice(SAFETY_INSTRUCTIONS), [{"kind": "safety"}], "safety"

    if ep_kind == "not_found":
        fake = rng.choice(FAKE_CONTACTS)
        phr = rng.choice([
            f"打开通讯录给{fake}打电话。",
            f"去电话通讯录里找{fake}并拨号。",
            f"帮我在联系人里给{fake}拨个电话。",
        ])
        return phr, [{"kind": "phone_not_found", "name": fake}], "not_found"

    def gen_climate_goals(k: int) -> list[dict[str, Any]]:
        pool = ["tL", "tR", "rt", "fan", "rfan", "mode", "fdef", "rdef", "rec", "pure", "scent", "ion", "wheel", "ron", "seat_heat", "seat_vent", "seat_massage", "seat_fold"]
        weights = [10, 7, 6, 7, 5, 10, 10, 8, 6, 7, 6, 6, 6, 6, 12, 14, 7, 9]
        chosen = []
        while pool and len(chosen) < k:
            i = rng.choices(range(len(pool)), weights)[0]
            chosen.append(pool.pop(i))
            weights.pop(i)
        gs: list[dict[str, Any]] = []
        c = st0["c"]
        for item in chosen:
            if item == "tL":
                delta = rng.choice([-2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5])
                want = max(17.0, min(30.0, round((c["tL"] + delta) * 2) / 2))
                if abs(want - c["tL"]) < 0.1:
                    want = 22.0 if c["tL"] != 22.0 else 24.0
                s_val = f"{int(want)}" if want.is_integer() else f"{want:.1f}"
                gs.append({"domain": "climate", "kind": "temp", "key": "tL", "up": "clim.tl.up", "down": "clim.tl.down", "want": want, "phrase": f"把主驾温度调到{s_val}度", "done_lbl": f"主驾温度{s_val}度"})
            elif item == "tR":
                delta = rng.choice([-2.0, -1.5, -1.0, 1.0, 1.5, 2.0])
                want = max(17.0, min(30.0, round((c["tR"] + delta) * 2) / 2))
                s_val = f"{int(want)}" if want.is_integer() else f"{want:.1f}"
                gs.append({"domain": "climate", "kind": "temp", "key": "tR", "up": "clim.tr.up", "down": "clim.tr.down", "want": want, "phrase": f"副驾温度调到{s_val}度", "done_lbl": f"副驾温度{s_val}度"})
            elif item == "rt":
                delta = rng.choice([-2.0, -1.0, 1.0, 2.0])
                want = max(18.0, min(29.0, round((c["rt"] + delta) * 2) / 2))
                s_val = f"{int(want)}" if want.is_integer() else f"{want:.1f}"
                gs.append({"domain": "climate", "kind": "temp", "key": "rt", "up": "clim.rt.up", "down": "clim.rt.down", "want": want, "phrase": f"后排温度调到{s_val}度", "done_lbl": f"后排温度{s_val}度"})
            elif item == "fan":
                want = rng.choice([x for x in range(1, 8) if abs(x - c["fan"]) in (1, 2, 3)])
                gs.append({"domain": "climate", "kind": "fan", "key": "fan", "up": "clim.fan.up", "down": "clim.fan.down", "want": want, "phrase": f"前排风量调到{want}档", "done_lbl": f"前排风量{want}档"})
            elif item == "rfan":
                want = rng.choice([x for x in range(1, 8) if abs(x - c["rfan"]) in (1, 2, 3)])
                gs.append({"domain": "climate", "kind": "fan", "key": "rfan", "up": "clim.rfan.up", "down": "clim.rfan.down", "want": want, "phrase": f"后排风量调到{want}档", "done_lbl": f"后排风量{want}档"})
            elif item == "mode":
                m_idx = rng.choice([0, 1, 2])
                m_nm = ["吹面", "吹面吹脚", "吹脚"][m_idx]
                gs.append({"domain": "climate", "kind": "clim_mode", "want": m_idx, "cid": f"clim.mode.{m_idx}", "phrase": f"把前排吹风模式切到{m_nm}", "done_lbl": f"前排{m_nm}"})
            elif item in ("fdef", "rdef", "rec", "pure", "scent", "ion", "wheel", "ron"):
                meta = {
                    "fdef": ("clim.fdef", "前除霜", "c", "fdef"),
                    "rdef": ("clim.rdef", "后除霜", "v", "rdef"),
                    "rec": ("clim.rec", "内循环", "c", "rec"),
                    "pure": ("clim.pure", "空气净化", "c", "pure"),
                    "scent": ("clim.scent", "香氛", "c", "scent"),
                    "ion": ("clim.ion", "负离子", "c", "ion"),
                    "wheel": ("clim.wheel", "方向盘加热", "v", "wheel"),
                    "ron": ("clim.rpower", "后排空调", "c", "ron"),
                }[item]
                want_on = rng.choice([True, True, False])
                verb = "打开" if want_on else "关闭"
                gs.append({"domain": "climate", "kind": "toggle", "cid": meta[0], "name": meta[1], "bucket": meta[2], "key": meta[3], "want": want_on, "phrase": f"{verb}{meta[1]}", "done_lbl": f"{meta[1]}{'已开' if want_on else '已关'}"})
            elif item == "seat_heat":
                sk, sz = rng.choice([("fl", "主驾"), ("fr", "副驾"), ("ml", "二排左"), ("mr", "二排右"), ("rl", "三排左"), ("rr", "三排右")])
                want = rng.choice([0, 1, 2, 3])
                phr = f"把{sz}座椅加热关掉" if want == 0 else f"把{sz}座椅加热调到{want}档"
                gs.append({"domain": "climate", "kind": "seat_cycle", "seat": sk, "slot": 0, "cid": f"clim.seat.{sk}.heat", "name": f"{sz}座椅加热", "want": want, "phrase": phr, "done_lbl": f"{sz}加热{want}档"})
            elif item == "seat_vent":
                sk, sz = rng.choice([("fl", "主驾"), ("fr", "副驾"), ("ml", "二排左"), ("mr", "二排右")])
                want = rng.choice([0, 0, 1, 2, 3])
                phr = f"把{sz}座椅通风关掉" if want == 0 else f"把{sz}座椅通风开到{want}档"
                gs.append({"domain": "climate", "kind": "seat_cycle", "seat": sk, "slot": 1, "cid": f"clim.seat.{sk}.vent", "name": f"{sz}座椅通风", "want": want, "phrase": phr, "done_lbl": f"{sz}通风{want}档"})
            elif item == "seat_massage":
                sk, sz = rng.choice([("fl", "主驾"), ("fr", "副驾")])
                want = rng.choice([0, 1, 2, 3])
                phr = f"关闭{sz}座椅按摩" if want == 0 else f"把{sz}座椅按摩开到{want}档"
                gs.append({"domain": "climate", "kind": "seat_cycle", "seat": sk, "slot": 2, "cid": f"clim.seat.{sk}.massage", "name": f"{sz}座椅按摩", "want": want, "phrase": phr, "done_lbl": f"{sz}按摩{want}档"})
            elif item == "seat_fold":
                sk, sz = rng.choice([("rl", "三排左"), ("rr", "三排右")])
                want = rng.choice([1, 1, 0])
                phr = f"把{sz}座椅放倒" if want == 1 else f"把{sz}座椅复原"
                gs.append({"domain": "climate", "kind": "seat_fold", "seat": sk, "cid": f"clim.seat.{sk}.fold", "name": f"{sz}座椅放倒", "want": want, "phrase": phr, "done_lbl": f"{sz}{'已放倒' if want else '已复原'}"})
        return gs

    def gen_settings_goals(k: int) -> list[dict[str, Any]]:
        gs: list[dict[str, Any]] = []
        base_cat = rng.choice([c for c in SET_CAT_ZH.keys() if c != "system"])
        used_ids: set[str] = set()
        for _ in range(k):
            kind = rng.choices(["seg", "amb_color", "wall", "slider"], weights=[62, 16, 10, 12])[0]
            if kind == "amb_color" and "amb_color" not in used_ids:
                used_ids.add("amb_color")
                ci = rng.randint(0, 7)
                gs.append({"domain": "settings", "kind": "set_color", "cat": "vehicle", "sub": "light", "want": ci, "cid": f"set.amb_color.{ci}", "phrase": f"把氛围灯颜色改成{AMB_COLORS[ci]}", "done_lbl": f"氛围灯{AMB_COLORS[ci]}"})
            elif kind == "wall" and "wall" not in used_ids:
                used_ids.add("wall")
                wi = rng.randint(0, 2)
                gs.append({"domain": "settings", "kind": "set_wall", "cat": "display", "sub": "wallp", "want": wi, "cid": f"set.wall.{wi}", "phrase": f"把壁纸换成{WALLS[wi]}", "done_lbl": f"壁纸{WALLS[wi]}"})
            elif kind == "slider":
                cands = [x for x in SET_SLIDER_ITEMS if x[2] not in used_ids]
                if not cands:
                    continue
                sc, ss, sid, slabel = rng.choice(cands)
                used_ids.add(sid)
                pct = rng.choice([20, 30, 40, 50, 60, 70, 80, 90])
                gs.append({"domain": "settings", "kind": "set_slider", "cat": sc, "sub": ss, "id": sid, "label": slabel, "want": pct, "phrase": f"把{slabel}调到{pct}%左右", "done_lbl": f"{slabel}{pct}%"})
            else:
                same_cat = [x for x in SET_SEG_ITEMS if x[0] == base_cat and x[2] not in used_ids]
                cands = same_cat if (same_cat and rng.random() < 0.65) else [x for x in SET_SEG_ITEMS if x[2] not in used_ids]
                if not cands:
                    continue
                sc, ss, sid, slabel, opts = rng.choice(cands)
                used_ids.add(sid)
                oi = rng.randint(0, len(opts) - 1)
                opt_zh = opts[oi]
                phr = f"{'打开' if oi == 0 else '关闭'}{slabel}" if opts == ["开启", "关闭"] else f"把{slabel}改成{opt_zh}"
                gs.append({"domain": "settings", "kind": "set_seg", "cat": sc, "sub": ss, "id": sid, "label": slabel, "want": oi, "opt_zh": opt_zh, "cid": f"set.{sid}.{oi}", "phrase": phr, "done_lbl": f"{slabel}{opt_zh}"})
        return gs

    def gen_nav_goal() -> list[dict[str, Any]]:
        di = rng.randint(0, 7)
        ri = rng.choice([0, 1, 2])
        dest_nm = PLACES[di]
        route_nm = ROUTES[ri]
        add_view = rng.random() < 0.35
        want_view = rng.choice([0, 1])
        phr = f"导航去{dest_nm}，选{route_nm}路线并开始导航" if dest_nm != "家" else f"导航回家，选{route_nm}路线开始导航"
        gs: list[dict[str, Any]] = [{"domain": "nav", "kind": "nav_route", "dest": di, "dest_nm": dest_nm, "opt": ri, "opt_nm": route_nm, "phrase": phr, "done_lbl": f"已导航{dest_nm}({route_nm})"}]
        if add_view:
            v_nm = "3D" if want_view else "2D"
            gs.append({"domain": "nav", "kind": "nav_view", "want": want_view, "phrase": f"把地图视角切换成{v_nm}", "done_lbl": f"地图{v_nm}视角"})
        return gs

    def gen_music_goals() -> list[dict[str, Any]]:
        gs: list[dict[str, Any]] = []
        if rng.random() < 0.65 and tracks:
            tab_i = rng.choice([0, 1, 3])
            if tab_i == 0:
                valid = list(range(min(6, len(tracks))))
            elif tab_i == 1:
                valid = [i for i, t in enumerate(tracks) if t.get("g") == 1]
            else:
                valid = [i for i, t in enumerate(tracks) if t.get("g") == 0][:5]
            if not valid:
                valid = [0]
            ti = rng.choice(valid)
            t_title = tracks[ti]["t"]
            gs.append({"domain": "music", "kind": "music_track", "tab": tab_i, "track": ti, "title": t_title, "phrase": f"去音乐的{MUSIC_TABS[tab_i]}播放{t_title}", "done_lbl": f"正在播放{t_title}"})
        else:
            gs.append({"domain": "music", "kind": "music_next", "phrase": "打开音乐播放并切到下一首", "done_lbl": "音乐已切下一首并播放"})
        if rng.random() < 0.55:
            mi = rng.choice([0, 1, 2])
            gs.append({"domain": "music", "kind": "music_mode", "want": mi, "phrase": f"把播放模式改成{MUSIC_MODES[mi]}", "done_lbl": f"播放模式{MUSIC_MODES[mi]}"})
        if rng.random() < 0.45:
            gs.append({"domain": "music", "kind": "music_fav", "want": True, "phrase": "点亮收藏", "done_lbl": "已收藏当前曲目"})
        return gs

    def gen_phone_contact() -> list[dict[str, Any]]:
        ci = rng.choices(range(len(CONTACTS)), weights=[1, 1, 1, 1, 2, 2, 2, 3, 4, 5, 6, 7])[0]
        nm = CONTACTS[ci]
        return [{"domain": "phone", "kind": "phone_contact", "idx": ci, "name": nm, "phrase": f"打开通讯录给{nm}打电话", "done_lbl": f"已呼叫{nm}"}]

    def gen_phone_dial() -> list[dict[str, Any]]:
        # 包含连续重复数字（如 00/88）锻炼模型核对已输入号码，防止漏按重复键
        if rng.random() < 0.5:
            num = rng.choice(["139002", "10086", "400880", "136005", "188009", "135501", "177008"])
        else:
            num = "".join(rng.choices("0123456789", k=rng.choice([4, 5, 6])))
        return [{"domain": "phone", "kind": "phone_dial", "digits": num, "phrase": f"打开电话拨号键盘输入号码 {num} 并拨打", "done_lbl": f"已拨打{num}"}]

    def gen_home_scene() -> dict[str, Any]:
        sk, sz = rng.choice([("nap", "小憩模式"), ("camp", "露营模式"), ("wash", "洗车模式"), ("pet", "宠物模式")])
        want = rng.choice([True, True, False])
        return {"domain": "home", "kind": "home_scene", "key": sk, "name": sz, "want": want, "phrase": f"在桌面{'打开' if want else '关闭'}{sz}", "done_lbl": f"{sz}{'已开' if want else '已关'}"}

    if ep_kind == "climate":
        gs = gen_climate_goals(rng.choice([2, 3]))
    elif ep_kind == "settings":
        gs = gen_settings_goals(rng.choice([2, 3]))
    elif ep_kind == "nav":
        gs = gen_nav_goal()
    elif ep_kind == "music":
        gs = gen_music_goals()
    elif ep_kind == "phone_contact":
        gs = gen_phone_contact()
    elif ep_kind == "phone_dial":
        gs = gen_phone_dial()
    else:
        doms = rng.sample(["climate", "settings", "nav", "music", "home_scene"], k=rng.choice([2, 2, 3]))
        gs = []
        for d in doms:
            if d == "climate":
                gs.extend(gen_climate_goals(rng.choice([1, 2])))
            elif d == "settings":
                gs.extend(gen_settings_goals(1))
            elif d == "nav":
                gs.extend(gen_nav_goal()[:1])
            elif d == "music":
                gs.extend(gen_music_goals()[:1])
            elif d == "home_scene":
                gs.append(gen_home_scene())
        if rng.random() < 0.25 and gs[-1]["domain"] != "home":
            gs.append({"domain": "home", "kind": "goto_home", "phrase": "最后回到桌面", "done_lbl": "已回桌面"})

    connectors = ["，然后", "，再", "，并", "，接着", "，"]
    parts = [gs[0]["phrase"]]
    for g in gs[1:]:
        parts.append(rng.choice(connectors) + g["phrase"])
    instr = "".join(parts) + "。"
    return instr, gs, ep_kind


def run_oracle_episode(
    page: Any,
    ep_id: str,
    img_dir: pathlib.Path,
    rng: random.Random,
    *,
    camera_prob: float = 0.5,
    drop_prob: float = 0.14,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """在真实 Chromium 中执行一条完整随机多步轨迹并展开为滑动窗口步级样本。"""
    page.goto(blt.INDEX_HTML.as_uri())
    page.wait_for_timeout(80)
    tracks = page.evaluate("() => TRACKS.map(t => ({t: t.t, g: t.g}))")
    randomize_initial_state(page, rng)
    st0 = get_state(page)
    instr, goals, ep_kind = sample_goal_bundle(rng, st0, tracks)

    cam = CameraSim(rng, enabled=(rng.random() < camera_prob))
    frame_paths: list[str] = []
    assistant_turns: list[str] = []
    multistep_rows: list[dict[str, Any]] = []
    readout_rows: list[dict[str, Any]] = []
    done_notes: list[str] = []

    def capture_frame(step_idx: int, sample_readout: bool = False) -> str:
        raw_png = page.screenshot(type="png")
        bgr = cam.render_bgr(raw_png)
        rel = f"images/{ep_id}_s{step_idx:02d}_{'k' if cam.enabled else 'c'}.png"
        cv2.imwrite(str(img_dir.parent / rel), bgr)
        if sample_readout:
            readout_rows.extend(make_readout_samples(rel, get_state(page), page, rng))
        return rel

    def emit_step(status_line: str, args: dict[str, Any], *, is_retry: bool = False) -> None:
        ans_text = fmt_tool_call(args, status_line=status_line)
        n = len(frame_paths)
        keep_from = max(0, n - 2)
        msgs: list[dict[str, Any]] = [
            {"role": "system", "content": cra.MAI_GYM_SYSTEM_PROMPT},
            {"role": "user", "content": f"{instr} ({e2e.MULTISTEP_NOTE})"},
        ]
        real_imgs: list[str] = []
        for i, rel in enumerate(frame_paths):
            if i >= keep_from:
                msgs.append({"role": "user", "content": "<image>"})
                real_imgs.append(rel)
            else:
                msgs.append({"role": "user", "content": "[历史截图]"})
            if i < len(assistant_turns):
                msgs.append({"role": "assistant", "content": assistant_turns[i]})
        msgs.append({"role": "assistant", "content": ans_text})
        multistep_rows.append({
            "task_type": f"e2e_{ep_kind}{'_retry' if is_retry else ''}",
            "ep_id": ep_id,
            "step": len(assistant_turns),
            "camera": cam.enabled,
            "images": real_imgs,
            "messages": msgs,
        })
        assistant_turns.append(ans_text)

    def prefix_done() -> str:
        if not done_notes:
            return ""
        return "已完成[" + "、".join(done_notes[-3:]) + "]；"

    def do_click_box(box: tuple[float, float, float, float], desc: str, *, allow_drop: bool = True, x_frac: float = 0.5) -> bool:
        x, y, w, h = box
        jx = rng.uniform(-0.04, 0.04) * w if x_frac == 0.5 else 0.0
        jy = rng.uniform(-0.04, 0.04) * h
        cx = x + w * x_frac + jx
        cy = y + h * 0.5 + jy
        nx, ny = cam.stage_to_norm999(cx, cy)
        args = {"action_type": "click", "coordinate": [nx, ny]}
        p_lbl = page_label_zh(get_state(page))

        if allow_drop and len(assistant_turns) < 20 and rng.random() < drop_prob:
            emit_step(f"{prefix_done()}在{p_lbl}，点{desc}", args)
            frame_paths.append(capture_frame(len(frame_paths), sample_readout=False))
            emit_step(f"{prefix_done()}上次点{desc}未生效，重新点击{desc}", args, is_retry=True)
        else:
            emit_step(f"{prefix_done()}在{p_lbl}，点{desc}", args)

        page.mouse.move(cx, cy)
        page.mouse.down()
        page.wait_for_timeout(20)
        page.mouse.up()
        page.wait_for_timeout(45)
        frame_paths.append(capture_frame(len(frame_paths), sample_readout=(rng.random() < 0.28)))
        return True

    def click_id(cid: str, desc: str, *, allow_drop: bool = True, x_frac: float = 0.5) -> bool:
        box = get_box(page, cid)
        if box is None:
            return False
        return do_click_box(box, desc, allow_drop=allow_drop, x_frac=x_frac)

    def do_drag_contacts(desc: str) -> bool:
        sx, sy = 0.68 * W, 0.78 * H
        ex, ey = 0.68 * W, 0.28 * H
        n1 = cam.stage_to_norm999(sx, sy)
        n2 = cam.stage_to_norm999(ex, ey)
        args = {"action_type": "drag", "start_coordinate": [n1[0], n1[1]], "end_coordinate": [n2[0], n2[1]]}
        p_lbl = page_label_zh(get_state(page))
        emit_step(f"{prefix_done()}在{p_lbl}，{desc}", args)
        scrolled = page.evaluate(
            """([x, y, dy]) => {
              const el = document.elementFromPoint(x, y);
              const sc = el ? el.closest('.scroll') : document.querySelector('#stage .scroll');
              if (!sc) return false;
              const before = sc.scrollTop;
              sc.scrollTop += dy;
              return Math.abs(sc.scrollTop - before) > 1;
            }""",
            [sx, sy, sy - ey],
        )
        page.wait_for_timeout(50)
        frame_paths.append(capture_frame(len(frame_paths), sample_readout=(rng.random() < 0.35)))
        return bool(scrolled)

    def ensure_page(target_domain: str, *, cat: str = "", sub: str = "", sub_tab_id: str = "", sub_tab_zh: str = "") -> bool:
        st = get_state(page)
        if st["phone"]["call"] is not None and not st["phone"]["hide"] and st["page"] != "climate" and target_domain != "phone":
            if not click_id("call.hide", "隐藏通话"):
                return False
            st = get_state(page)

        if st["page"] != target_domain:
            dock_zh = {"home": "底栏桌面", "climate": "底栏空调", "settings": "底栏车辆设置", "nav": "底栏导航", "music": "底栏音乐", "phone": "底栏电话"}[target_domain]
            if not click_id(f"dock.{target_domain}", dock_zh):
                return False
            st = get_state(page)

        if target_domain == "settings" and cat:
            if st["set"]["cat"] != cat:
                if not click_id(f"set.cat.{cat}", f"设置大类{SET_CAT_ZH[cat]}"):
                    return False
                st = get_state(page)
            if sub and st["set"]["sub"].get(cat) != sub:
                if not click_id(f"set.sub.{cat}.{sub}", f"设置子页{SET_SUB_ZH[(cat, sub)]}"):
                    return False
        if target_domain == "phone" and sub_tab_id:
            if st["phone"]["tab"] != sub_tab_id:
                if not click_id(f"phone.tab.{sub_tab_id}", f"电话标签{sub_tab_zh}"):
                    return False
        if target_domain == "music" and sub_tab_id != "":
            if str(st["m"]["tab"]) != sub_tab_id:
                if not click_id(f"music.tab.{sub_tab_id}", f"音乐标签{sub_tab_zh}"):
                    return False
        return True

    frame_paths.append(capture_frame(0, sample_readout=True))

    if ep_kind == "safety":
        p_lbl = page_label_zh(get_state(page))
        emit_step(
            f"当前在{p_lbl}，指令涉及不可逆高危操作，需人工确认",
            {"action_type": "ask_user", "text": "该操作属于不可逆高危操作，存在安全或数据丢失风险，请您亲自确认后再执行。"},
        )
        return multistep_rows, readout_rows, ep_kind

    if ep_kind == "not_found":
        fake_name = goals[0]["name"]
        if not ensure_page("phone", sub_tab_id="contacts", sub_tab_zh="通讯录"):
            return [], readout_rows, ep_kind
        for _ in range(5):
            moved = do_drag_contacts(f"未见{fake_name}，向上滑动通讯录继续查找")
            if not moved:
                break
        emit_step(
            f"通讯录已滑到底部仍未找到{fake_name}，终止任务",
            {"action_type": "terminate", "status": "failed", "answer": f"通讯录中未找到联系人“{fake_name}”"},
        )
        return multistep_rows, readout_rows, ep_kind

    for g in goals:
        if len(assistant_turns) >= 24:
            return [], readout_rows, ep_kind
        kind = g["kind"]
        dom = g["domain"]

        if kind == "goto_home":
            if not ensure_page("home"):
                return [], readout_rows, ep_kind
            done_notes.append(g["done_lbl"])
            continue

        if kind == "home_scene":
            if not ensure_page("home"):
                return [], readout_rows, ep_kind
            st = get_state(page)
            cur = bool(st["v"]["nap"]) if g["key"] == "nap" else bool(st["scn"].get(g["key"]))
            if cur != g["want"]:
                if not click_id(f"home.scene.{g['key']}", g["name"]):
                    return [], readout_rows, ep_kind
            done_notes.append(g["done_lbl"])
            continue

        if dom == "climate":
            if not ensure_page("climate"):
                return [], readout_rows, ep_kind
            if kind == "temp":
                for _ in range(8):
                    cur = float(get_state(page)["c"][g["key"]])
                    if abs(cur - g["want"]) < 0.05:
                        break
                    cid = g["up"] if g["want"] > cur else g["down"]
                    lbl = f"{g['phrase'][:4]}{'调高' if g['want'] > cur else '调低'}(现{cur:.1f})"
                    if not click_id(cid, lbl):
                        return [], readout_rows, ep_kind
            elif kind == "fan":
                for _ in range(7):
                    cur = int(get_state(page)["c"][g["key"]])
                    if cur == g["want"]:
                        break
                    cid = g["up"] if g["want"] > cur else g["down"]
                    lbl = f"{'前排' if g['key']=='fan' else '后排'}风量{'调大' if g['want'] > cur else '调小'}(现{cur})"
                    if not click_id(cid, lbl):
                        return [], readout_rows, ep_kind
            elif kind == "clim_mode":
                if get_state(page)["c"]["mode"] != g["want"]:
                    if not click_id(g["cid"], g["done_lbl"]):
                        return [], readout_rows, ep_kind
            elif kind == "toggle":
                st = get_state(page)
                cur = bool(st[g["bucket"]][g["key"]])
                if cur != g["want"]:
                    if not click_id(g["cid"], g["name"]):
                        return [], readout_rows, ep_kind
            elif kind == "seat_cycle":
                for _ in range(4):
                    cur = int(get_state(page)["c"]["seat"][g["seat"]][g["slot"]])
                    if cur == g["want"]:
                        break
                    if not click_id(g["cid"], f"{g['name']}(现{cur}档->目标{g['want']}档)"):
                        return [], readout_rows, ep_kind
            elif kind == "seat_fold":
                cur = int(get_state(page)["c"]["seat"][g["seat"]][2])
                if cur != g["want"]:
                    if not click_id(g["cid"], g["name"]):
                        return [], readout_rows, ep_kind
            done_notes.append(g["done_lbl"])
            continue

        if dom == "settings":
            if not ensure_page("settings", cat=g["cat"], sub=g["sub"]):
                return [], readout_rows, ep_kind
            st = get_state(page)
            if kind == "set_seg":
                if st["v"].get(g["id"]) != g["want"]:
                    if not click_id(g["cid"], f"{g['label']}:{g['opt_zh']}"):
                        return [], readout_rows, ep_kind
            elif kind == "set_color":
                if st["v"].get("amb_color") != g["want"]:
                    if not click_id(g["cid"], f"氛围灯{AMB_COLORS[g['want']]}色块"):
                        return [], readout_rows, ep_kind
            elif kind == "set_wall":
                if st["v"].get("wall") != g["want"]:
                    if not click_id(g["cid"], f"壁纸{WALLS[g['want']]}"):
                        return [], readout_rows, ep_kind
            elif kind == "set_slider":
                cur = int(st["v"].get(g["id"], 50))
                if abs(cur - g["want"]) > 5:
                    frac = max(0.05, min(0.95, g["want"] / 100.0))
                    if not click_id(f"set.{g['id']}", f"{g['label']}滑条至{g['want']}%", x_frac=frac):
                        return [], readout_rows, ep_kind
            done_notes.append(g["done_lbl"])
            continue

        if dom == "nav":
            if not ensure_page("nav"):
                return [], readout_rows, ep_kind
            if kind == "nav_route":
                st = get_state(page)
                if st["nav"]["st"] == "guide":
                    if not (st["nav"]["dest"] == g["dest"] and st["nav"]["opt"] == g["opt"]):
                        if not click_id("nav.exit", "退出当前导航"):
                            return [], readout_rows, ep_kind
                        st = get_state(page)
                if st["nav"]["st"] == "route" and st["nav"]["dest"] != g["dest"]:
                    if not click_id("nav.cancel", "取消旧路线"):
                        return [], readout_rows, ep_kind
                    st = get_state(page)
                if st["nav"]["st"] == "idle":
                    if g["dest"] in (0, 1, 7):
                        chip_lbl = {0: "回家", 1: "去公司", 7: "充电站"}[g["dest"]]
                        if not click_id(f"nav.chip.{g['dest']}", f"快捷按钮{chip_lbl}"):
                            return [], readout_rows, ep_kind
                    else:
                        if not click_id("nav.search", "搜索目的地"):
                            return [], readout_rows, ep_kind
                    st = get_state(page)
                if st["nav"]["st"] == "search":
                    if not click_id(f"nav.place.{g['dest']}", f"目的地{g['dest_nm']}"):
                        return [], readout_rows, ep_kind
                    st = get_state(page)
                if st["nav"]["st"] == "route":
                    if st["nav"]["opt"] != g["opt"]:
                        if not click_id(f"nav.opt.{g['opt']}", f"路线{g['opt_nm']}"):
                            return [], readout_rows, ep_kind
                    if not click_id("nav.start", "开始导航"):
                        return [], readout_rows, ep_kind
            elif kind == "nav_view":
                if get_state(page)["nav"]["view"] != g["want"]:
                    if not click_id("nav.view", f"切换{'3D' if g['want'] else '2D'}视角"):
                        return [], readout_rows, ep_kind
            done_notes.append(g["done_lbl"])
            continue

        if dom == "music":
            if kind == "music_track":
                if not ensure_page("music", sub_tab_id=str(g["tab"]), sub_tab_zh=MUSIC_TABS[g["tab"]]):
                    return [], readout_rows, ep_kind
                st = get_state(page)
                if not (st["m"]["i"] == g["track"] and st["m"]["play"]):
                    if not click_id(f"music.track.{g['track']}", f"曲目{g['title']}"):
                        return [], readout_rows, ep_kind
            elif kind == "music_next":
                if not ensure_page("music"):
                    return [], readout_rows, ep_kind
                if not get_state(page)["m"]["play"]:
                    if not click_id("music.play", "播放音乐"):
                        return [], readout_rows, ep_kind
                if not click_id("music.next", "下一首"):
                    return [], readout_rows, ep_kind
            elif kind == "music_mode":
                if not ensure_page("music"):
                    return [], readout_rows, ep_kind
                for _ in range(3):
                    if get_state(page)["m"]["mode"] == g["want"]:
                        break
                    if not click_id("music.mode", f"播放模式切换至{MUSIC_MODES[g['want']]}"):
                        return [], readout_rows, ep_kind
            elif kind == "music_fav":
                if not ensure_page("music"):
                    return [], readout_rows, ep_kind
                st = get_state(page)
                if bool(st["m"]["fav"].get(str(st["m"]["i"]))) != g["want"]:
                    if not click_id("music.fav", "收藏按钮"):
                        return [], readout_rows, ep_kind
            done_notes.append(g["done_lbl"])
            continue

        if dom == "phone":
            if kind == "phone_contact":
                if not ensure_page("phone", sub_tab_id="contacts", sub_tab_zh="通讯录"):
                    return [], readout_rows, ep_kind
                cid = f"phone.contact.{g['idx']}"
                found = False
                for _ in range(5):
                    if get_box(page, cid) is not None:
                        found = True
                        break
                    moved = do_drag_contacts(f"未见{g['name']}，向上滑动通讯录")
                    if not moved:
                        break
                if not found or not click_id(cid, f"联系人{g['name']}"):
                    return [], readout_rows, ep_kind
            elif kind == "phone_dial":
                if not ensure_page("phone", sub_tab_id="keypad", sub_tab_zh="拨号键盘"):
                    return [], readout_rows, ep_kind
                if get_state(page)["phone"]["num"]:
                    if not click_id("phone.clear", "清空残留号码"):
                        return [], readout_rows, ep_kind
                for ch in g["digits"]:
                    if not click_id(f"phone.key.{ch}", f"拨号数字{ch}", allow_drop=(rng.random() < 0.35)):
                        return [], readout_rows, ep_kind
                if not click_id("phone.call", "拨打按钮"):
                    return [], readout_rows, ep_kind
            done_notes.append(g["done_lbl"])
            continue

    p_lbl = page_label_zh(get_state(page))
    emit_step(
        f"{prefix_done()}在{p_lbl}，全部目标已达成，结束任务",
        {"action_type": "terminate", "status": "success", "answer": "已完成全部操作"},
    )
    return multistep_rows, readout_rows, ep_kind


def worker_generate_episodes(
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
            ep_id = f"{split}_w{worker_id}_e{i:04d}"
            try:
                ms, ro, ek = run_oracle_episode(page, ep_id, img_dir, rng)
                if ms:
                    ms_all.extend(ms)
                    ro_all.extend(ro)
                    counts[ek] += 1
            except Exception as exc:
                counts["err"] += 1
                print(f"[w{worker_id} {ep_id}] err: {exc}", flush=True)
        browser.close()
    return ms_all, ro_all, dict(counts)


HARD_GROUNDING_RE = re.compile(
    r"^(set\.amb_color\.|clim\.mode\.|clim\.seat\..*\.(vent|fold|heat|massage)|clim\.(fdef|rdef)|media\.open$)"
)


def sample_single_grounding(
    v1_dir: pathlib.Path,
    out_dir: pathlib.Path,
    split: str,
    n_target: int,
    seed: int,
) -> list[dict[str, Any]]:
    src_jsonl = v1_dir / f"{split}.jsonl"
    rows = [json.loads(l) for l in src_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = random.Random(seed + (1 if split == "train" else 2))

    hard: list[dict[str, Any]] = []
    normal: list[dict[str, Any]] = []
    for r in rows:
        cid = str((r.get("meta") or {}).get("cid") or r.get("control_id") or "")
        if HARD_GROUNDING_RE.match(cid):
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
                msgs.append({"role": "assistant", "content": strip_think(m["content"])})
            else:
                msgs.append(m)
        cid = str((r.get("meta") or {}).get("cid") or r.get("control_id") or "")
        return {
            "task_type": tag,
            "control_id": cid,
            "images": [rel],
            "messages": msgs,
        }

    picked: list[dict[str, Any]] = []
    hard_budget = min(len(hard) * 5, int(n_target * 0.35))
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
    ap.add_argument("out", default="/mnt/ymtc2t/datasets/carhmi_maigym_v2")
    ap.add_argument("--v1-dir", default="/mnt/ymtc2t/datasets/carhmi_maigym_v1")
    ap.add_argument("--train-eps", type=int, default=480)
    ap.add_argument("--val-eps", type=int, default=36)
    ap.add_argument("--train-ground", type=int, default=2800)
    ap.add_argument("--val-ground", type=int, default=200)
    ap.add_argument("--train-readout-cap", type=int, default=2400)
    ap.add_argument("--val-readout-cap", type=int, default=180)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    v1_dir = pathlib.Path(a.v1_dir)
    t0 = time.monotonic()

    stats: dict[str, Any] = {}
    for split, n_eps, n_grd, ro_cap in (
        ("train", a.train_eps, a.train_ground, a.train_readout_cap),
        ("val", a.val_eps, a.val_ground, a.val_readout_cap),
    ):
        print(f"[{split}] Generating {n_eps} multi-step Playwright episodes across {a.workers} workers ...", flush=True)
        per_w = [n_eps // a.workers + (1 if i < n_eps % a.workers else 0) for i in range(a.workers)]
        ms_rows: list[dict[str, Any]] = []
        ro_rows: list[dict[str, Any]] = []
        ep_counts: collections.Counter = collections.Counter()

        with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = [
                ex.submit(worker_generate_episodes, wid, split, cnt, str(out), a.seed + (0 if split == "train" else 99999))
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

        grd_rows = sample_single_grounding(v1_dir, out, split, n_grd, a.seed)

        combined = ms_rows + grd_rows + ro_rows
        rng.shuffle(combined)
        (out / f"{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in combined),
            encoding="utf-8",
        )
        type_hist = collections.Counter(r["task_type"] for r in combined)
        stats[split] = {
            "total": len(combined),
            "multistep_steps": len(ms_rows),
            "single_grounding": len(grd_rows),
            "visual_readout": len(ro_rows),
            "episodes": dict(ep_counts),
            "task_types": dict(type_hist),
        }
        print(
            f"[{split}] DONE total={len(combined)} (e2e_steps={len(ms_rows)}, ground={len(grd_rows)}, readout={len(ro_rows)}) "
            f"eps={dict(ep_counts)} elapsed={time.monotonic()-t0:.1f}s",
            flush=True,
        )

    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ALL DONE in {time.monotonic()-t0:.1f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
