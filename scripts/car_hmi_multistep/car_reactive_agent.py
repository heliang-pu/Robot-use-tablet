"""Reactive Goal-Ledger Closed-Loop Long-Horizon Car HMI Agent (v2).

Designed for real, messy cockpit environments over the pure-vision `Device` protocol
(no ADB, no DOM coordinates during execution):
  1. Goal Compiler (1 structured-JSON call): compiles ANY complex natural-language command
     into a Goal Ledger (desired end states only — NO hardcoded start page or open-loop path).
  2. Per-step Reactive Loop (`Observe -> Interrupt Guard -> Decide -> Ground -> Verify`):
     - Re-observes the LIVE screenshot before every decision (arbitrary start page, already
       on climate, already navigating, active call overlay, physical missed-taps all self-heal).
     - Uses vLLM guided JSON (`response_format: json_schema`, `enable_thinking=False`) for
       fast (~0.6-1.5s), 100%-valid structured outputs (zero JSON parse errors, zero rambling).
     - Uses fine-tuned `MAI-Gym-9B-carhmi-v1` (97.4% camera accuracy) as the pure-vision
       Grounder for every physical tap.
     - Supports scroll-to-find in long lists, toggle idempotency checks, cycle/stepper/slider
       local closed-loop control, physical missed-tap retry, safety guardrails, and loop guard.
"""
from __future__ import annotations

import base64
import dataclasses
import io
import json
import pathlib
import re
import time
from typing import Any, Protocol

import numpy as np
from PIL import Image
import requests


class Device(Protocol):
    def screenshot(self) -> Image.Image: ...
    def click(self, u: float, v: float) -> None: ...
    def long_press(self, u: float, v: float, duration_s: float = 0.8) -> None: ...
    def swipe(self, u1: float, v1: float, u2: float, v2: float, duration_s: float = 0.8) -> None: ...
    def back(self) -> None: ...
    def home(self) -> None: ...
    def type_text(self, text: str) -> None: ...


IN_APP_NOTE = (
    "本设备由机械臂夹着电容笔点按屏幕来操作，没有 open、type、key 这三种动作。"
    "需要的应用已经打开，就在当前屏幕上操作，不要回主屏幕。"
)

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
_NUM_RE = r"-?\d+(?:\.\d+)?"


def parse_maigym_response(text: str) -> tuple[str | None, tuple[float, float] | None]:
    m = _TOOL_CALL_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(1).strip())
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
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S).split("</think>")[-1]
    m2 = re.search(rf"coordinate\"?\s*[:=]?\s*\[\s*({_NUM_RE})\s*,\s*({_NUM_RE})\s*\]", cleaned)
    if m2:
        return "click", (float(m2.group(1)), float(m2.group(2)))
    return None, None


L8_DOMAIN_MAP = """# 理想 L8 车机功能域与标准控件名地图

## 各页面（page）与底栏入口
- home（“桌面”按钮）：场景卡片（“小憩模式”按钮、“露营模式”按钮、“洗车模式”按钮、“宠物模式”按钮）
- climate（“空调”按钮）：
  - 前排：“前排空调开关”按钮、“前排自动”按钮、“A/C 制冷”按钮、“最大制冷”按钮、主驾温度的调高按钮/主驾温度的调低按钮(16-32步长0.5)、前排风量的调大按钮/前排风量的调小按钮(1-7步长1)、“前排吹面”按钮、“前排吹面吹脚”按钮、“前排吹脚”按钮
  - 横条：“前除霜”按钮、“后除霜”按钮、“内循环”按钮、“空气净化”按钮、“香氛”按钮、“负离子”按钮
  - 后排：“后排空调开关”按钮、“后排锁定”按钮、“后排自动”按钮、“后排同步”按钮、后排温度的调高按钮/后排温度的调低按钮、后排风量的调大按钮/后排风量的调小按钮、“后排吹面”按钮、“后排吹脚”按钮
  - 座椅（档位循环0->3->2->1->0）：“方向盘加热”按钮；主驾/副驾（“<座>座椅加热”按钮、“<座>座椅通风”按钮、“<座>座椅按摩”按钮）；二排左/二排右（加热、通风）；三排左/三排右（“<座>座椅加热”按钮、“<座>座椅放倒”按钮[两态开关]）
  - 右侧：副驾温度的调高按钮/副驾温度的调低按钮、“温区同步”按钮
- settings（“车辆设置”按钮，左侧两列目录：settings_cat / settings_sub）：
  - 车辆 / 快捷：中控锁/车窗通风/尾门/后视镜折叠/遮阳帘/儿童锁/雨刮维修/加油小门/充电口盖 瓷砖；“驾驶模式”一栏的“舒适/标准/运动”
  - 车辆 / 灯光：“氛围灯颜色”的第1个色块(红)/第2个色块(橙)/第3个色块(黄)/第4个色块(绿)/第5个色块(青)/第6个色块(蓝)/第7个色块(紫)/第8个色块(粉)；“氛围灯亮度”滑条；“氛围灯模式”一栏的“单色模式/多色模式”；“前大灯”一栏的“关闭/位置灯/近光灯/自动”；“智能远光灯”一栏的“开启/关闭”
  - 车辆 / 门窗：“车窗”一栏的“全部关闭/通风/全部打开”；“锁车自动升窗”/“靠近解锁”/“离车上锁”一栏的“开启/关闭”；“解锁方式”一栏的“全车/仅主驾”
  - 车辆 / 驾驶：“加速模式”/“能量回收”/“转向助力”/“悬架高度”/“自动驻车”一栏的对应选项
  - 车辆 / 后视镜：“锁车自动折叠”/“倒车自动下翻”/“后视镜加热”一栏的“开启/关闭”
  - 车辆 / 座椅：“迎宾座椅”/“座椅加热自动开启”；“主驾位置记忆”一栏的“位置 1/位置 2/位置 3”
  - 车辆 / 充电：“充电上限”滑条；“预约充电”/“对外放电”/“充电口盖自动关闭”一栏的“开启/关闭”
  - 车辆 / 尾门：“尾门开启高度”滑条；“脚踢感应开启”一栏的“开启/关闭”；“尾门”一栏的“打开尾门”
  - 车辆 / 维护：“洗车模式”/“拖车模式”一栏的“开启/关闭”
  - 智能驾驶 / 辅助驾驶：“导航辅助驾驶”/“车道保持辅助”/“超速提醒”一栏的“开启/关闭”；“跟车距离”一栏的“近/中/远”；“变道风格”一栏的“舒适/标准/高效”
  - 智能驾驶 / 主动安全：“前向碰撞预警”/“自动紧急制动”/“盲区监测”/“开门预警”；“预警灵敏度”一栏的“低/中/高”
  - 智能驾驶 / 泊车：“自动泊车”/“遥控泊车”/“倒车影像自动开启”；“雷达提示音”一栏的“关/低/高”
  - 导航 / 偏好：“路线偏好”一栏的“智能推荐/时间优先/少收费/不走高速”；“躲避拥堵”/“充电规划”
  - 导航 / 播报：“播报模式”一栏的“详细/简洁/静音”；“导航音量”滑条；“电子眼播报”
  - 导航 / 地图：“地图视角”一栏的“2D 正北/2D 车头/3D”；“实时路况”；“昼夜模式”一栏的“自动/白天/夜间”
  - 显示 / 屏幕：“亮度”滑条；“外观”一栏的“浅色/深色”；“字体大小”一栏的“标准/大”；“屏保时钟”一栏的“开启/关闭”
  - 显示 / HUD：“HUD 显示”；“HUD 亮度”滑条；“HUD 高度”滑条；“显示内容”一栏的“简洁/标准/丰富”
  - 显示 / 壁纸：壁纸“默认”/壁纸“暖阳”/壁纸“青山”；“动态壁纸”一栏的“开启/关闭”
  - 声音 / 音量：“媒体音量”/“通话音量”/“语音助手音量”滑条；“随速音量”
  - 声音 / 音效：“音效模式”一栏的“标准/影院/人声/自定义”；“声场位置”一栏的“全车/主驾/前排/后排”；“全景声”
  - 声音 / 提示音：“按键音”/“锁车提示音”；“提示音量”一栏的“低/中/高”
  - 连接 / 蓝牙：“蓝牙”一栏的“开启/关闭”
  - 连接 / WLAN：“WLAN”一栏的“开启/关闭”
  - 连接 / 热点：“车载热点”一栏的“开启/关闭”；“频段”一栏的“2.4G/5G”
  - 理想同学 / 唤醒：“语音唤醒”/“免唤醒指令”/“连续对话”；“唤醒音区”一栏的“全车/仅主驾”
  - 理想同学 / 音色：“音色”一栏的“标准女声/温柔女声/标准男声”；“语速”一栏的“慢/标准/快”；“语言”一栏的“普通话/English/Deutsch”
  - 安全 / 哨兵：“哨兵模式”一栏的“开启/关闭”；“灵敏度”一栏的“低/中/高”；“行车记录仪”
  - 安全 / 儿童锁：“左后门儿童锁”/“右后门儿童锁”/“后排屏幕锁”
  - 系统 / 通用：“距离单位”一栏的“公里/英里”；“温度单位”一栏的“℃/℉”；“时间格式”一栏的“24 小时/12 小时”；“恢复出厂设置”【高危操作，必须标记 safety_risk=true！】
- nav（“导航”按钮）：目的地（家/公司/中心公园/人民医院/购物中心/体育中心/高铁站/充电站），路线选项（推荐/最快/少收费），地图控件（“路况”按钮、“视角”按钮[2D/3D切换]、“定位”按钮）
- music（“音乐”按钮）：左侧标签 sub_tab（推荐/电台/我的收藏/本地音乐）；曲目（列表里的“轻音乐精选/公路旅行歌单/钢琴时光/交通广播/晨间新闻”）；控制（“上一首”按钮、“播放暂停”按钮、“下一首”按钮、“收藏”按钮、“播放模式”按钮[列表循环->单曲循环->随机播放]、“歌词开关”按钮）
- phone（“电话”按钮）：左侧标签 sub_tab（最近通话/通讯录/拨号键盘/蓝牙设置）；联系人（通讯录里的“名字”）；最近通话（最近通话里的“名字”）；拨号（dial）
- apps（“应用中心”按钮）：“<应用名>”应用图标
"""

SETTINGS_SUBS: dict[str, list[str]] = {
    "车辆": ["快捷", "灯光", "门窗", "驾驶", "后视镜", "座椅", "充电", "尾门", "维护"],
    "智能驾驶": ["辅助驾驶", "主动安全", "泊车"],
    "导航": ["偏好", "播报", "地图"],
    "显示": ["屏幕", "HUD", "壁纸"],
    "声音": ["音量", "音效", "提示音"],
    "连接": ["蓝牙", "WLAN", "热点"],
    "理想同学": ["唤醒", "音色"],
    "安全": ["哨兵", "儿童锁"],
    "系统": ["通用", "关于本车"],
}

GOAL_COMPILER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "safety_block": {"type": "boolean"},
        "safety_reason": {"type": "string"},
        "goals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "domain": {"type": "string", "enum": ["home", "climate", "settings", "nav", "music", "phone", "apps"]},
                    "kind": {
                        "type": "string",
                        "enum": [
                            "goto_only",
                            "tap",
                            "toggle",
                            "cycle",
                            "stepper",
                            "slider",
                            "nav_route",
                            "nav_view",
                            "music_play",
                            "music_mode",
                            "scroll_tap",
                            "dial",
                        ],
                    },
                    "settings_cat": {"type": "string"},
                    "settings_sub": {"type": "string"},
                    "sub_tab": {"type": "string"},
                    "target": {"type": "string"},
                    "target_up": {"type": "string"},
                    "target_down": {"type": "string"},
                    "want_on": {"type": "boolean"},
                    "want_level": {"type": "integer"},
                    "want_value": {"type": "number"},
                    "step": {"type": "number"},
                    "want_percent": {"type": "integer"},
                    "nav_dest": {"type": "string"},
                    "nav_route": {"type": "string"},
                    "music_mode": {"type": "string"},
                    "digits": {"type": "string"},
                    "check": {"type": "string"},
                },
                "required": ["domain", "kind", "target"],
            },
        },
    },
    "required": ["safety_block", "safety_reason", "goals"],
}

GOAL_COMPILER_SYSTEM = f"""你是理想 L8 车机目标编译器。把用户的自然语言指令编译为一份结构化的“目标账本”（goals 数组）。
你**不需要**考虑当前停在哪一页，也**不要**自己写切页步骤（执行引擎会自动看实时画面并跨页导航），只需要列出用户要求达成的各个最终目标！

{L8_DOMAIN_MAP}

## 各种 kind 怎么填（严格遵守，不相关字段可留空或默认0/false）
1. `tap`: 点击确定性单选/分段选项/色块/应用图标/普通动作按钮。
   - 设置项开关（如哨兵模式开启、外观深色、跟车距离远）全部用 `tap`，并填好 `settings_cat` 和 `settings_sub`！
   - 例：外观切深色/浅色（如“开启深色模式/切回浅色模式/浅色主题”）-> domain="settings", kind="tap", settings_cat="显示", settings_sub="屏幕", target="“外观”一栏的“深色”" 或 "“外观”一栏的“浅色”"（注意：切浅色/深色模式一定在 显示/屏幕 的“外观”一栏；只有明确说“地图昼夜模式/地图白天/地图夜间”时才去 导航/地图！）
   - 例：理想同学音色/语速/语言 -> 都在 settings_cat="理想同学", settings_sub="音色"（不要误写到“唤醒”！）
   - 例：氛围灯紫色 -> domain="settings", kind="tap", settings_cat="车辆", settings_sub="灯光", target="“氛围灯颜色”的第7个色块"
   - 例：壁纸青山 -> domain="settings", kind="tap", settings_cat="显示", settings_sub="壁纸", target="壁纸“青山”"
   - 例：音乐下一首 -> domain="music", kind="tap", target="“下一首”按钮"
2. `toggle`: 二态图标开关（按一下开再按一下关，必须先看状态防止误反转）。
   - 用于空调页的 前排空调开关/后排空调开关/A/C 制冷/前除霜/后除霜/内循环/空气净化/香氛/负离子/方向盘加热/三排左座椅放倒/三排右座椅放倒，以及桌面的 小憩模式/露营模式/洗车模式/宠物模式，或音乐的 收藏/播放暂停。
   - 填 `want_on` (true/false) 和简洁 `check`（例如“当前空调页中部横条里的‘前除霜’是否已开启高亮？”）。
3. `cycle`: 座椅加热/通风/按摩循环档位（0=关闭, 1/2/3=对应档位）。
   - 例：主驾座椅加热2档 -> domain="climate", kind="cycle", target="“主驾座椅加热”按钮", want_level=2, check="空调页右侧座椅图‘主驾’格内从上往下第1个座椅加热按钮当前是几档？(未亮橙色为0，亮橙色带角标为1/2/3)"
   - 例：关闭主驾座椅通风 -> domain="climate", kind="cycle", target="“主驾座椅通风”按钮", want_level=0, check="空调页右侧座椅图‘主驾’格内从上往下第2个座椅通风按钮当前是几档？(未亮蓝色为0，亮蓝色带角标为1/2/3)"
4. `stepper`: 温度(step=0.5)或风量(step=1)步进调节。
   - 例：主驾温度21度 -> domain="climate", kind="stepper", target="主驾温度", target_up="主驾温度的调高按钮", target_down="主驾温度的调低按钮", want_value=21.0, step=0.5, check="当前空调页左上角‘主驾’温度显示的数字是多少？"
   - 例：副驾温度22.5度 -> target_up="副驾温度的调高按钮", target_down="副驾温度的调低按钮", want_value=22.5, step=0.5, check="当前空调页最右侧‘副驾’温度显示的数字是多少？"
   - 例：后排温度26度 -> target_up="后排温度的调高按钮", target_down="后排温度的调低按钮", want_value=26.0, step=0.5, check="当前空调页左下角‘后排’温度显示的数字是多少？"
   - 例：前排风量5档 -> target_up="前排风量的调大按钮", target_down="前排风量的调小按钮", want_value=5, step=1, check="当前空调页左上角‘风量’显示的数字是几？"
   - 例：后排风量4档 -> target_up="后排风量的调大按钮", target_down="后排风量的调小按钮", want_value=4, step=1, check="当前空调页左下角‘风量’显示的数字是几？"
5. `slider`: 百分比滑条（亮度、氛围灯亮度、充电上限、尾门开启高度、音量）。填 `want_percent`(0-100) 和 `check`。
6. `nav_route`: 发起导航到某目的地。填 `domain="nav"`, `kind="nav_route"`, `nav_dest`（如"高铁站"/"公司"/"家"/"充电站"）, `nav_route`（"推荐"/"最快"/"少收费"，未指定则填"推荐"）, `target=nav_dest`。
7. `nav_view`: 切换导航地图2D/3D视角。填 `domain="nav"`, `kind="nav_view"`, `target="3D"` 或 `"2D"`。
8. `music_play`: 播放某首曲目或切播放状态。如果指定了歌名（如公路旅行歌单），填 `domain="music"`, `kind="music_play"`, `sub_tab`（如"本地音乐"或""）, `target="列表里的“公路旅行歌单”"`；如果只是“开始播放音乐”，填 `target="“播放暂停”按钮"` 且 `want_on=true`。
9. `music_mode`: 切换音乐播放模式。填 `domain="music"`, `kind="music_mode"`, `music_mode`（"列表循环"/"单曲循环"/"随机播放"）, `target="“播放模式”按钮"`。
10. `scroll_tap`: 在可能很长的滚动列表中查找并点击某项（例如通讯录找联系人、最近通话找人）。
    - 例：通讯录给妈妈打电话 -> domain="phone", kind="scroll_tap", sub_tab="通讯录", target="通讯录里的“妈妈”"
    - 例：通讯录给赵六打电话 -> domain="phone", kind="scroll_tap", sub_tab="通讯录", target="通讯录里的“赵六”"
11. `dial`: 电话拨号键盘拨号。填 `domain="phone"`, `kind="dial"`, `sub_tab="拨号键盘"`, `digits="138001"`, `target="拨打"`。
12. `goto_only`: 仅当用户最后明确说“最后回到桌面”等纯切页要求时使用（如 domain="home", kind="goto_only", target="“桌面”按钮"）。

## 安全护栏
如果用户要求执行不可逆高危操作（如“恢复出厂设置”、“格式化”、“解绑车主账号”、“关闭自动紧急制动 AEB”），必须设置 `safety_block=true` 并写明 `safety_reason`，`goals` 留空！

## 目标排序优化
- 如果用户要求既切深色模式又去空调页，保持用户指定顺序（或尽量把同一 domain 的目标排在一起，避免来回切页；但若用户明确说了“先……再……最后回到桌面”，尊重先后顺序）。
"""

SCREEN_STATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "page": {
            "type": "string",
            "enum": ["home", "climate", "settings", "nav_idle", "nav_search", "nav_route", "nav_guide", "music", "phone", "apps", "stub"],
        },
        "has_call_overlay": {"type": "boolean"},
        "settings_cat": {"type": "string"},
        "settings_sub": {"type": "string"},
        "sub_tab": {"type": "string"},
    },
    "required": ["page", "has_call_overlay", "settings_cat", "settings_sub", "sub_tab"],
}


def _img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _img_diff(a: Image.Image, b: Image.Image) -> float:
    aa = np.asarray(a.convert("RGB"), dtype=np.float32)
    bb = np.asarray(b.convert("RGB").resize(a.size), dtype=np.float32)
    return float(np.mean(np.abs(aa - bb)))


def _roi_diff(a: Image.Image, b: Image.Image, u: float, v: float, radius: int = 80) -> float:
    """Local mean pixel diff around tap point (u, v) so thin icon color toggles register cleanly."""
    w, h = a.size
    cx, cy = int(round(u * w)), int(round(v * h))
    x0, y0 = max(0, cx - radius), max(0, cy - radius)
    x1, y1 = min(w, cx + radius), min(h, cy + radius)
    if x1 <= x0 or y1 <= y0:
        return _img_diff(a, b)
    aa = np.asarray(a.convert("RGB").crop((x0, y0, x1, y1)), dtype=np.float32)
    bb = np.asarray(b.convert("RGB").resize(a.size).crop((x0, y0, x1, y1)), dtype=np.float32)
    return float(np.mean(np.abs(aa - bb)))


def _canonical_cycle_check(target: str, fallback: str) -> str:
    """Always generate a neutral level-reading question without leaking want_level."""
    seat_pos = ""
    for s in ("主驾", "副驾", "二排左", "二排右", "三排左", "三排右"):
        if s in target:
            seat_pos = s
            break
    if "加热" in target:
        return f"空调页右侧座椅图‘{seat_pos}’格内从上往下第1个【座椅加热】按钮当前是几档？（未亮橙色返回0；若亮橙色且右上角有数字角标则返回该数字1、2或3；不要看下面的蓝色通风按钮）"
    if "通风" in target:
        return f"空调页右侧座椅图‘{seat_pos}’格内从上往下第2个【座椅通风】按钮当前是几档？（未亮蓝色返回0；若亮蓝色且右上角有数字角标则返回该数字1、2或3；不要看上面的橙色加热按钮）"
    if "按摩" in target:
        return f"空调页右侧座椅图‘{seat_pos}’格内从上往下第3个【座椅按摩】按钮当前是几档？（未亮蓝色返回0；若亮蓝色且右上角有数字角标则返回该数字1、2或3）"
    return fallback


def _canonical_toggle_check(target: str, fallback: str) -> str:
    """Map known cockpit toggle targets to exact trained unambiguous visual verification prompts."""
    mapping = [
        ("前除霜", "当前空调页中部横条里的‘前除霜’是否已开启高亮？"),
        ("后除霜", "当前空调页中部横条里的‘后除霜’是否已开启高亮？"),
        ("内循环", "当前空调页中部横条里的‘内循环’是否已开启高亮？"),
        ("空气净化", "当前空调页中部横条里的‘空气净化’是否已开启高亮？"),
        ("香氛", "当前空调页中部横条里的‘香氛’是否已开启高亮？"),
        ("负离子", "当前空调页中部横条里的‘负离子’是否已开启高亮？"),
        ("方向盘加热", "当前空调页右侧‘方向盘加热’按钮是否已开启高亮（橙色）？"),
        ("后排空调", "当前空调页左下角‘后排’空调开关是否已开启高亮？"),
        ("A/C", "当前空调页左侧‘A/C 制冷’是否已开启高亮？"),
        ("三排左", "当前空调页右侧座椅图‘三排左’的放倒按钮是否已开启高亮？"),
        ("三排右", "当前空调页右侧座椅图‘三排右’的放倒按钮是否已开启高亮？"),
        ("小憩模式", "当前桌面场景卡片里的“小憩模式”是否已处于开启高亮状态？"),
        ("露营模式", "当前桌面场景卡片里的“露营模式”是否已处于开启高亮状态？"),
        ("洗车模式", "当前桌面场景卡片里的“洗车模式”是否已处于开启高亮状态？"),
        ("宠物模式", "当前桌面场景卡片里的“宠物模式”是否已处于开启高亮状态？"),
        ("收藏", "音乐页面播放控制栏里的红心“收藏”按钮是否已点亮为红色？"),
    ]
    for k, q in mapping:
        if k in target:
            return q
    return fallback


@dataclasses.dataclass
class ReactiveRunLog:
    instruction: str
    status: str  # "success" | "need_user" | "not_found" | "error"
    goals: list[dict[str, Any]]
    events: list[dict[str, Any]]
    clicks: int
    llm_calls: int
    elapsed_s: float
    detail: str = ""


class ReactiveCarAgent:
    """Reactive Goal-Ledger closed-loop car HMI agent with real-world corner-case resilience."""

    def __init__(
        self,
        device: Device,
        *,
        grounder_url: str = "http://127.0.0.1:18809/v1",
        grounder_model: str = "mai-gym-9b-ft",
        policy_url: str | None = None,
        policy_model: str | None = None,
        settle_s: float = 0.08,
        max_total_clicks: int = 65,
    ) -> None:
        self.device = device
        self.grounder_url = grounder_url.rstrip("/")
        self.grounder_model = grounder_model
        self.policy_url = (policy_url or grounder_url).rstrip("/")
        self.policy_model = policy_model or grounder_model
        self.settle_s = settle_s
        self.max_total_clicks = max_total_clicks
        self._sess = requests.Session()
        self._sess.trust_env = False
        self._clicks = 0
        self._llm_calls = 0

    def _chat_json(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        schema_name: str,
        *,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        self._llm_calls += 1
        body = {
            "model": self.policy_model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            },
        }
        last_err: Exception | None = None
        for _ in range(3):
            try:
                rep = self._sess.post(self.policy_url + "/chat/completions", json=body, timeout=120)
                rep.raise_for_status()
                content = rep.json()["choices"][0]["message"].get("content") or "{}"
                return json.loads(content)
            except Exception as exc:
                last_err = exc
                time.sleep(1.0)
        raise RuntimeError(f"Policy JSON call failed: {last_err}")

    def compile_goals(self, instruction: str) -> dict[str, Any]:
        msgs = [
            {"role": "system", "content": GOAL_COMPILER_SYSTEM},
            {"role": "user", "content": f"用户指令：{instruction}"},
        ]
        return self._chat_json(msgs, GOAL_COMPILER_SCHEMA, "goal_ledger", max_tokens=1536)

    def inspect_screen(self, screen: Image.Image) -> dict[str, Any]:
        """Observe LIVE screenshot to determine exact current page, overlays, and selected menus."""
        b64 = _img_to_b64(screen)
        prompt = (
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
        msgs = [
            {"role": "system", "content": "你是车机界面状态识别器，严格输出紧凑单行 JSON，不要多余换行。"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            },
        ]
        return self._chat_json(msgs, SCREEN_STATE_SCHEMA, "screen_state", max_tokens=320)

    def read_bool(self, question: str, screen: Image.Image) -> bool:
        b64 = _img_to_b64(screen)
        schema = {"type": "object", "properties": {"answer": {"type": "boolean"}}, "required": ["answer"]}
        msgs = [
            {"role": "system", "content": "你是车机视觉状态核验器，仔细看图后输出布尔值 answer。"},
            {"role": "user", "content": [{"type": "text", "text": question}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]},
        ]
        return bool(self._chat_json(msgs, schema, "bool_check", max_tokens=48).get("answer", False))

    def read_number(self, question: str, screen: Image.Image) -> float | None:
        b64 = _img_to_b64(screen)
        schema = {"type": "object", "properties": {"value": {"type": "number"}}, "required": ["value"]}
        msgs = [
            {"role": "system", "content": "你是车机视觉读数器，仔细看图读出目标位置的精确数值。"},
            {"role": "user", "content": [{"type": "text", "text": question}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]},
        ]
        val = self._chat_json(msgs, schema, "num_check", max_tokens=48).get("value")
        return float(val) if isinstance(val, (int, float)) else None

    def read_text(self, question: str, screen: Image.Image) -> str:
        b64 = _img_to_b64(screen)
        schema = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
        msgs = [
            {"role": "system", "content": "你是车机视觉文字读取器，准确读出目标控件上显示的文字。"},
            {"role": "user", "content": [{"type": "text", "text": question}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]},
        ]
        return str(self._chat_json(msgs, schema, "text_check", max_tokens=64).get("text", "")).strip()

    def ground(self, target_desc: str, screen: Image.Image) -> tuple[float, float] | None:
        """Pure-vision coordinate localization via fine-tuned MAI-Gym-9B-carhmi-v1."""
        self._llm_calls += 1
        b64 = _img_to_b64(screen)
        instruction = f"点击{target_desc} ({IN_APP_NOTE})"
        msgs = [
            {"role": "system", "content": MAI_GYM_SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]},
        ]
        body = {
            "model": self.grounder_model,
            "messages": msgs,
            "temperature": 0.0,
            "max_tokens": 128,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        for _ in range(3):
            try:
                rep = self._sess.post(self.grounder_url + "/chat/completions", json=body, timeout=120)
                rep.raise_for_status()
                out = rep.json()["choices"][0]["message"].get("content") or ""
                action, coord = parse_maigym_response(out)
                if action in ("click", "double_click", "long_press") and coord is not None:
                    return min(max(coord[0] / 999.0, 0.0), 1.0), min(max(coord[1] / 999.0, 0.0), 1.0)
            except Exception:
                time.sleep(1.0)
        return None

    def _click(self, u: float, v: float) -> Image.Image:
        if self._clicks >= self.max_total_clicks:
            raise RuntimeError(f"Exceeded max_total_clicks={self.max_total_clicks}")
        self.device.click(u, v)
        self._clicks += 1
        if self.settle_s > 0:
            time.sleep(self.settle_s)
        return self.device.screenshot()

    def _tap_verified(
        self,
        target: str,
        screen: Image.Image,
        events: list[dict[str, Any]],
        *,
        expect_change: bool = True,
        retries: int = 3,
    ) -> tuple[Image.Image, bool]:
        """Click target and self-heal on physical missed taps (L0 diff check)."""
        uv = self.ground(target, screen)
        if uv is None:
            events.append({"act": "tap_miss_ground", "target": target})
            return screen, False
        for attempt in range(retries):
            # Subtle jitter on retry if physical stylus missed the contact point
            du = (0.0, 0.004, -0.004)[attempt % 3]
            dv = (0.0, 0.004, -0.004)[attempt % 3]
            tu, tv = min(max(uv[0] + du, 0.01), 0.99), min(max(uv[1] + dv, 0.01), 0.99)
            nxt = self._click(tu, tv)
            diff = _img_diff(screen, nxt)
            rdiff = _roi_diff(screen, nxt, tu, tv, radius=80)
            events.append({"act": "tap", "target": target, "uv": [round(uv[0], 4), round(uv[1], 4)], "diff": round(diff, 2), "roi_diff": round(rdiff, 2), "try": attempt})
            screen = nxt
            if not expect_change or diff >= 0.35 or rdiff >= 1.2:
                return screen, True
        return screen, False

    def _ensure_page(self, domain: str, screen: Image.Image, state: dict[str, Any], events: list[dict[str, Any]], *, cat: str = "", sub: str = "", sub_tab: str = "") -> tuple[Image.Image, dict[str, Any]]:
        """Reactively navigate from ANY current screen state to the required domain / sub-menu!"""
        dock_map = {
            "home": "“桌面”按钮",
            "climate": "“空调”按钮",
            "settings": "“车辆设置”按钮",
            "nav": "“导航”按钮",
            "music": "“音乐”按钮",
            "phone": "“电话”按钮",
            "apps": "“应用中心”按钮",
        }
        # 1. If a call overlay covers the left panel and we might need unobstructed view, hide it cleanly when switching pages
        if state.get("has_call_overlay") and domain != "phone":
            screen, _ = self._tap_verified("“隐藏通话”按钮", screen, events, expect_change=True, retries=2)
            state = self.inspect_screen(screen)
            events.append({"obs": "after_hide_call", "state": state})

        # 2. Ensure main domain page (NEVER click climate again if already on climate!)
        def on_domain(st: dict[str, Any]) -> bool:
            p = st.get("page", "")
            return p.startswith("nav_") if domain == "nav" else p == domain

        for _ in range(5):
            if on_domain(state):
                break
            screen, _ = self._tap_verified(dock_map[domain], screen, events, expect_change=True, retries=2)
            state = self.inspect_screen(screen)
            events.append({"obs": f"ensure_{domain}", "state": state})

        # 3. Settings 2-level menu navigation (verified against live screenshot after each jump!)
        if domain == "settings" and on_domain(state):
            if cat:
                for _ in range(5):
                    if state.get("settings_cat") == cat:
                        break
                    screen, _ = self._tap_verified(f"设置左侧菜单里的“{cat}”", screen, events, expect_change=True, retries=2)
                    state = self.inspect_screen(screen)
                    events.append({"obs": f"settings_cat_{cat}", "state": state})
            if sub:
                for _ in range(5):
                    if state.get("settings_sub") == sub:
                        break
                    screen, _ = self._tap_verified(f"设置左侧菜单里的“{sub}”", screen, events, expect_change=True, retries=2)
                    state = self.inspect_screen(screen)
                    events.append({"obs": f"settings_sub_{sub}", "state": state})

        # 4. Phone / Music sub-tab navigation
        if domain in ("phone", "music") and sub_tab and on_domain(state):
            page_zh = "电话" if domain == "phone" else "音乐"
            for _ in range(5):
                if state.get("sub_tab") == sub_tab:
                    break
                screen, _ = self._tap_verified(f"{page_zh}页面的“{sub_tab}”标签", screen, events, expect_change=True, retries=2)
                state = self.inspect_screen(screen)
                events.append({"obs": f"{domain}_tab_{sub_tab}", "state": state})

        return screen, state

    def _exec_goal(self, g: dict[str, Any], screen: Image.Image, state: dict[str, Any], events: list[dict[str, Any]]) -> tuple[Image.Image, dict[str, Any], str]:
        domain = g["domain"]
        kind = g["kind"]
        screen, state = self._ensure_page(
            domain, screen, state, events,
            cat=g.get("settings_cat", ""),
            sub=g.get("settings_sub", ""),
            sub_tab=g.get("sub_tab", ""),
        )

        if kind == "goto_only":
            return screen, state, "done"

        if kind == "tap":
            target = g["target"]
            # Closed-loop ambient color swatch verification (1..8) — self-corrects 1-swatch offsets & drops!
            m_amb = re.search(r"氛围灯颜色.*第\s*([1-8])\s*个色块", target)
            if m_amb:
                want_idx = int(m_amb.group(1))
                uv = self.ground(target, screen)
                if uv is not None:
                    u_cur, v_cur = uv
                    for attempt in range(5):
                        cur_v = self.read_number("当前设置灯光页‘氛围灯颜色’一排8个圆形色块中，带白色选中外圈的是从左到右第几个色块？（1到8的整数）", screen)
                        cur_idx = int(round(cur_v)) if cur_v is not None else -1
                        events.append({"verify_amb_color": cur_idx, "want": want_idx, "try": attempt})
                        if cur_idx == want_idx:
                            return screen, state, "done"
                        if attempt > 0 and 1 <= cur_idx <= 8:
                            # Pure-vision proportional shift along the 8-swatch row (~92px/1920px pitch)
                            u_cur = min(max(u_cur + (want_idx - cur_idx) * (92.0 / 1920.0), 0.35), 0.98)
                        screen = self._click(u_cur, v_cur)
                return screen, state, "done"

            # Closed-loop climate airflow mode verification (front/rear)
            m_mode = re.search(r"(前排|后排)(吹面吹脚|吹面|吹脚)", target)
            if domain == "climate" and m_mode:
                zone, want_m = m_mode.group(1), m_mode.group(2)
                q = (
                    "当前空调页前排吹风模式三个图标中蓝色高亮选中哪一个？（只回答：吹面 / 吹面吹脚 / 吹脚）"
                    if zone == "前排"
                    else "当前空调页后排吹风模式两个图标中蓝色高亮选中哪一个？（只回答：吹面 / 吹脚）"
                )
                for attempt in range(4):
                    cur_m = self.read_text(q, screen)
                    events.append({"verify_clim_mode": cur_m, "want": want_m, "try": attempt})
                    if cur_m == want_m:
                        return screen, state, "done"
                    screen, _ = self._tap_verified(target, screen, events, expect_change=True, retries=1)
                return screen, state, "done"

            # Closed-loop wallpaper selection verification
            m_wall = re.search(r'壁纸[“"](默认|暖阳|青山)[”"]', target)
            if domain == "settings" and m_wall:
                want_w = m_wall.group(1)
                for attempt in range(4):
                    cur_w = self.read_text("当前壁纸页面中处于蓝色边框高亮选中状态的壁纸名称是什么？（默认/暖阳/青山）", screen)
                    events.append({"verify_wall": cur_w, "want": want_w, "try": attempt})
                    if want_w in cur_w:
                        return screen, state, "done"
                    screen, _ = self._tap_verified(target, screen, events, expect_change=True, retries=1)
                return screen, state, "done"

            # Closed-loop settings segmented option verification (100% immune to physical drops, idempotent,
            # plus pure-vision sibling sub-tab self-recovery for unseen holdout sub-tabs!)
            if domain == "settings" and "一栏的" in target:
                q_seg = f"当前屏幕上{target}是否已经处于蓝色高亮选中状态？"
                cur_cat = g.get("settings_cat") or state.get("settings_cat", "")
                tried_subs: set[str] = {state.get("settings_sub", ""), g.get("settings_sub", "")}
                for attempt in range(4):
                    already = self.read_bool(q_seg, screen)
                    events.append({"verify_seg": target, "selected": already, "try": attempt})
                    if already:
                        return screen, state, "done"
                    if attempt == 2 and cur_cat in SETTINGS_SUBS:
                        # Pure-vision recovery: if 2 taps on current sub-tab failed to highlight target,
                        # check sibling sub-tabs under the same category (fixes unseen holdout sub-tab routing!)
                        siblings = [s for s in SETTINGS_SUBS[cur_cat] if s not in tried_subs]
                        for sib in siblings[:3]:
                            tried_subs.add(sib)
                            screen, _ = self._tap_verified(f"设置左侧菜单里的“{sib}”", screen, events, expect_change=True, retries=2)
                            state = self.inspect_screen(screen)
                            events.append({"obs": f"settings_sub_recover_{sib}", "state": state})
                            for r_try in range(3):
                                ok_sib = self.read_bool(q_seg, screen)
                                events.append({"verify_seg_recover": target, "sub": sib, "selected": ok_sib, "try": r_try})
                                if ok_sib:
                                    return screen, state, "done"
                                if r_try == 0:
                                    m_row = re.match(r'[“"]([^”"]+)[”"]一栏的', target)
                                    if m_row:
                                        has_row = self.read_bool(f"当前右侧设置列表里是否能看到名为“{m_row.group(1)}”的这一行设置项？", screen)
                                        events.append({"verify_row_visible": m_row.group(1), "sub": sib, "visible": has_row})
                                        if not has_row:
                                            break
                                screen, _ = self._tap_verified(target, screen, events, expect_change=False, retries=1)
                    screen, _ = self._tap_verified(target, screen, events, expect_change=False, retries=1)
                return screen, state, "done"

            screen, changed = self._tap_verified(target, screen, events, expect_change=True, retries=2)
            state = self.inspect_screen(screen)
            return screen, state, "done"

        if kind == "toggle":
            target = g["target"]
            want_on = bool(g.get("want_on", True))
            check = _canonical_toggle_check(target, g.get("check") or f"当前屏幕上{target}是否已处于开启高亮状态（蓝色或橙色高亮/红心/暂停双竖线）？")
            for attempt in range(5):
                cur = self.read_bool(check, screen)
                events.append({"verify_toggle": target, "cur": cur, "want": want_on, "try": attempt})
                if cur == want_on:
                    return screen, state, "done"
                screen, _ = self._tap_verified(target, screen, events, expect_change=False, retries=1)
            return screen, state, "done"

        if kind == "cycle":
            target = g["target"]
            want = int(g.get("want_level", 0))
            check = _canonical_cycle_check(target, f"当前屏幕上{target}处于几档？（未亮为0，亮色带角标数字则为1、2或3）")
            order = [0, 3, 2, 1]
            cached_uv: tuple[float, float] | None = None
            color_kind = "orange" if "加热" in target else ("blue" if ("通风" in target or "按摩" in target) else "")

            def _fovea_color_p95(img: Image.Image, uv_pt: tuple[float, float], ck: str) -> float:
                arr = np.asarray(img.convert("RGB"), dtype=np.float32)
                h_im, w_im = arr.shape[:2]
                cx, cy = int(round(uv_pt[0] * w_im)), int(round(uv_pt[1] * h_im))
                rx, ry = max(12, int(w_im * 0.015)), max(12, int(h_im * 0.024))
                patch = arr[max(0, cy - ry):min(h_im, cy + ry), max(0, cx - rx):min(w_im, cx + rx)]
                if patch.size == 0:
                    return 60.0
                diff_ch = (patch[:, :, 0] - patch[:, :, 2]) if ck == "orange" else (patch[:, :, 2] - patch[:, :, 0])
                return float(np.percentile(diff_ch, 95))

            for attempt in range(8):
                val = self.read_number(check, screen)
                cur = int(round(val)) if val is not None else -1
                # Pure-vision foveated color consistency check around VLM-grounded (u, v):
                # prevents blurry camera frames from confusing adjacent heat(0) vs vent(2)!
                if color_kind and cur == want:
                    if cached_uv is None:
                        cached_uv = self.ground(target, screen)
                    if cached_uv is not None:
                        p95 = _fovea_color_p95(screen, cached_uv, color_kind)
                        if want > 0 and p95 < 42.0:
                            events.append({"verify_cycle_fovea": target, "raw_cur": cur, "color_p95": round(p95, 1), "override": 0, "try": attempt})
                            cur = 0
                        elif want == 0 and p95 > 85.0:
                            events.append({"verify_cycle_fovea": target, "raw_cur": cur, "color_p95": round(p95, 1), "override": 1, "try": attempt})
                            cur = 1
                events.append({"verify_cycle": target, "cur": cur, "want": want, "try": attempt})
                if cur == want:
                    return screen, state, "done"
                if cached_uv is None:
                    cached_uv = self.ground(target, screen)
                    if cached_uv is None:
                        break
                clicks = (order.index(want) - order.index(cur)) % 4 if cur in order and want in order else 1
                clicks = max(1, min(clicks, 3))
                for _ in range(clicks):
                    screen = self._click(*cached_uv)
                events.append({"act": "cycle_clicks", "target": target, "n": clicks, "uv": [round(cached_uv[0], 4), round(cached_uv[1], 4)]})
            return screen, state, "done"

        if kind == "stepper":
            want = float(g.get("want_value", 24.0))
            inc = float(g.get("step", 0.5)) or 0.5
            t_up, t_dn = g.get("target_up", ""), g.get("target_down", "")
            check = g.get("check") or f"当前屏幕上{g['target']}显示的数值是多少？（‘低’填16.0，‘高’填32.0）"
            cache: dict[str, tuple[float, float]] = {}
            for attempt in range(10):
                cur = self.read_number(check, screen)
                events.append({"verify_stepper": g["target"], "cur": cur, "want": want, "try": attempt})
                if cur is None:
                    break
                diff = want - cur
                if abs(diff) < inc * 0.4:
                    return screen, state, "done"
                direction = "up" if diff > 0 else "down"
                tgt = t_up if direction == "up" else t_dn
                if direction not in cache:
                    uv = self.ground(tgt, screen)
                    if uv is None:
                        break
                    cache[direction] = uv
                n_press = max(1, min(int(round(abs(diff) / inc)), 6))
                for _ in range(n_press):
                    screen = self._click(*cache[direction])
                events.append({"act": "stepper_clicks", "target": tgt, "n": n_press, "uv": [round(cache[direction][0], 4), round(cache[direction][1], 4)]})
            return screen, state, "done"

        if kind == "slider":
            target = g["target"]
            want = float(g.get("want_percent", 50))
            check = g.get("check") or f"当前屏幕上{target}右侧显示的数值是多少？（0到100之间的整数）"
            uv = self.ground(target, screen)
            if uv is None:
                return screen, state, "error"
            bar_w = 520.0 / 1920.0
            # Read current value first — if already within tolerance, no click needed!
            cur0 = self.read_number(check, screen)
            events.append({"verify_slider_init": target, "cur": cur0, "want": want})
            if cur0 is not None and abs(cur0 - want) <= 6.0:
                return screen, state, "done"
            u_try = uv[0]
            for attempt in range(6):
                u_try = min(max(u_try, 0.02), 0.98)
                screen = self._click(u_try, uv[1])
                cur = self.read_number(check, screen)
                events.append({"verify_slider": target, "u": round(u_try, 4), "cur": cur, "want": want, "try": attempt})
                if cur is None or abs(cur - want) <= 6.0:
                    return screen, state, "done"
                u_try = u_try + ((want - cur) / 100.0) * bar_w
            return screen, state, "done"

        if kind == "nav_route":
            dest = g.get("nav_dest") or g["target"]
            route = g.get("nav_route") or "推荐"
            quick_chips = {"家": "“回家”按钮", "回家": "“回家”按钮", "公司": "“去公司”按钮", "去公司": "“去公司”按钮", "充电站": "“充电站”按钮", "超级充电站": "“充电站”按钮"}
            for _ in range(10):
                p = state.get("page", "")
                if p == "nav_guide":
                    # Check if we are ALREADY navigating to the requested destination!
                    dest_short = dest.replace("回", "").replace("去", "")
                    already = self.read_bool(f"底部导航信息栏是否显示正在前往“{dest_short}”？", screen)
                    events.append({"verify_nav_guide": dest, "already": already})
                    if already:
                        return screen, state, "done"
                    # Corner case: currently navigating somewhere ELSE -> exit navigation FIRST!
                    screen, _ = self._tap_verified("“退出导航”按钮", screen, events, expect_change=True, retries=3)
                    state = self.inspect_screen(screen)
                elif p == "nav_idle":
                    if dest in quick_chips:
                        screen, _ = self._tap_verified(quick_chips[dest], screen, events, expect_change=True, retries=3)
                    else:
                        screen, _ = self._tap_verified("“搜索目的地”按钮", screen, events, expect_change=True, retries=3)
                    state = self.inspect_screen(screen)
                elif p == "nav_search":
                    clean_dest = dest.replace("回家", "家").replace("去公司", "公司")
                    screen, _ = self._tap_verified(f"目的地列表里的“{clean_dest}”", screen, events, expect_change=True, retries=3)
                    state = self.inspect_screen(screen)
                elif p == "nav_route":
                    if route in ("推荐", "最快", "少收费"):
                        # Radio tabs on unseen holdout nav_route are 100% idempotent:
                        # always tap non-default route at least once, then verify to catch physical drops!
                        for _r in range(3):
                            if _r > 0 or route == "推荐":
                                r_ok = self.read_bool(f"当前屏幕上路线选项“{route}”是否已经处于蓝色高亮选中状态？", screen)
                                events.append({"verify_nav_route": route, "selected": r_ok, "try": _r})
                                if r_ok:
                                    break
                            screen, _ = self._tap_verified(f"路线选项“{route}”", screen, events, expect_change=False, retries=1)
                    screen, _ = self._tap_verified("“开始导航”按钮", screen, events, expect_change=True, retries=3)
                    state = self.inspect_screen(screen)
                    if state.get("page") == "nav_guide":
                        return screen, state, "done"
                else:
                    screen, state = self._ensure_page("nav", screen, state, events)
            return screen, state, "done"

        if kind == "nav_view":
            want_view = "3D" if "3" in str(g.get("target", "3D")) else "2D"
            for _ in range(5):
                cur = self.read_text("地图右侧悬浮按钮上当前显示的是‘2D’还是‘3D’？", screen)
                events.append({"verify_nav_view": cur, "want": want_view})
                if want_view in cur.upper():
                    return screen, state, "done"
                screen, _ = self._tap_verified("“视角”按钮", screen, events, expect_change=False, retries=1)
            return screen, state, "done"

        if kind == "music_play":
            target = g["target"]
            if "播放暂停" in target:
                want_on = bool(g.get("want_on", True))
                for _ in range(5):
                    is_playing = self.read_bool("音乐主播放按钮（蓝色圆形大按钮）里显示的是两条竖线（暂停图标，表示正在播放）吗？", screen)
                    events.append({"verify_music_playing": is_playing, "want": want_on})
                    if is_playing == want_on:
                        return screen, state, "done"
                    screen, _ = self._tap_verified("“播放暂停”按钮", screen, events, expect_change=False, retries=1)
            else:
                screen, _ = self._tap_verified(target, screen, events, expect_change=True, retries=3)
            return screen, state, "done"

        if kind == "music_mode":
            want = g.get("music_mode") or "单曲循环"
            for _ in range(6):
                cur = self.read_text("音乐页面播放控制栏中播放模式按钮上当前显示的文字是什么（列表循环/单曲循环/随机播放）？", screen)
                events.append({"verify_music_mode": cur, "want": want})
                if want in cur:
                    return screen, state, "done"
                screen, _ = self._tap_verified("“播放模式”按钮", screen, events, expect_change=False, retries=1)
            return screen, state, "done"

        if kind == "scroll_tap":
            target = g["target"]
            # Extract quoted item name for visibility check
            m = re.search(r"[“\"]([^”\"]+)[”\"]", target)
            item_name = m.group(1) if m else target
            for scroll_i in range(5):
                visible = self.read_bool(f"当前屏幕列表中是否清晰可见名为“{item_name}”的条目？", screen)
                events.append({"verify_visible": item_name, "visible": visible, "scroll_i": scroll_i})
                if visible:
                    screen, _ = self._tap_verified(target, screen, events, expect_change=True, retries=3)
                    state = self.inspect_screen(screen)
                    return screen, state, "done"
                # Scroll down the right-pane list
                self.device.swipe(0.68, 0.78, 0.68, 0.28, duration_s=0.4)
                if self.settle_s > 0:
                    time.sleep(self.settle_s)
                nxt = self.device.screenshot()
                diff = _img_diff(screen, nxt)
                events.append({"act": "scroll_down", "scroll_i": scroll_i, "diff": round(diff, 2)})
                screen = nxt
                if diff < 0.3:
                    # Hit bottom of list without finding target -> do NOT click a random entry!
                    return screen, state, "not_found"
            return screen, state, "not_found"

        if kind == "dial":
            digits = re.sub(r"[^0-9*#]", "", str(g.get("digits", "")))
            key_cache: dict[str, tuple[float, float]] = {}
            q_num = "拨号键盘上方号码显示区里当前输入的数字串是什么？（若显示‘请输入号码’占位提示则返回空字符串）"
            for dial_round in range(3):
                shown = re.sub(r"[^0-9*#]", "", self.read_text(q_num, screen))
                events.append({"verify_dial_pre": shown, "want": digits, "round": dial_round})
                if shown == digits:
                    break
                if shown and not digits.startswith(shown):
                    screen, _ = self._tap_verified("“清空”按钮", screen, events, expect_change=True, retries=2)
                    shown = ""
                rem = digits[len(shown):]
                for ch in rem:
                    if ch not in key_cache:
                        uv = self.ground(f"拨号键盘上的“{ch}”键", screen)
                        if uv is None:
                            return screen, state, "error"
                        key_cache[ch] = uv
                    for _att in range(4):
                        nxt = self._click(*key_cache[ch])
                        # Check ROI diff around display area (top) or full diff
                        if _img_diff(screen, nxt) >= 0.02:
                            screen = nxt
                            break
                        screen = nxt
            events.append({"act": "dial_digits", "digits": digits})
            screen, _ = self._tap_verified("“拨打”按钮", screen, events, expect_change=True, retries=3)
            state = self.inspect_screen(screen)
            return screen, state, "done"

        return screen, state, "done"

    def run(self, instruction: str) -> ReactiveRunLog:
        self._clicks = 0
        self._llm_calls = 0
        t0 = time.monotonic()
        events: list[dict[str, Any]] = []
        try:
            compiled = self.compile_goals(instruction)
        except Exception as exc:
            return ReactiveRunLog(instruction, "error", [], events, self._clicks, self._llm_calls, round(time.monotonic() - t0, 2), f"compile_error: {exc}")

        if compiled.get("safety_block"):
            return ReactiveRunLog(
                instruction,
                "need_user",
                [],
                events,
                self._clicks,
                self._llm_calls,
                round(time.monotonic() - t0, 2),
                compiled.get("safety_reason", "高危操作需人工确认"),
            )

        goals = compiled.get("goals") or []
        screen = self.device.screenshot()
        state = self.inspect_screen(screen)
        events.append({"obs": "initial", "state": state})

        try:
            for idx, g in enumerate(goals):
                events.append({"goal_idx": idx, "goal": g})
                screen, state, outcome = self._exec_goal(g, screen, state, events)
                if outcome == "not_found":
                    return ReactiveRunLog(
                        instruction,
                        "not_found",
                        goals,
                        events,
                        self._clicks,
                        self._llm_calls,
                        round(time.monotonic() - t0, 2),
                        f"目标未找到: {g.get('target')}",
                    )
        except Exception as exc:
            return ReactiveRunLog(instruction, "error", goals, events, self._clicks, self._llm_calls, round(time.monotonic() - t0, 2), str(exc))

        return ReactiveRunLog(instruction, "success", goals, events, self._clicks, self._llm_calls, round(time.monotonic() - t0, 2))
