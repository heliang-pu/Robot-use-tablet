"""Setting up the Android tablet over adb, once, before calibrating: orientation, sleep, the size of what it shows,
navigation buttons. Nothing here is used while the robot works: at run time the tablet is only seen by the camera.

Every value that is changed is read first and kept (``Snapshot``), so that the tablet can be put back as it was.
"""

from __future__ import annotations

import dataclasses
import re
import subprocess
from typing import Any
import xml.etree.ElementTree as ET  # noqa: N817 - the usual alias

from tablet_tap.errors import SetupError

MIN_TABLET_WIDTH_DP = 600  # below this smallest width Android apps switch to their phone layouts
MAX_UI_SCALE = 1.5  # how much larger than the factory setting the interface is made at most
FONT_SCALE = 1.3  # the largest step of Android's own font size setting
SCREEN_TIMEOUT_MS = 30 * 60 * 1000
NAV_BUTTON_IDS = {"back": "com.android.systemui:id/back", "home": "com.android.systemui:id/home",
                  "recent": "com.android.systemui:id/recent_apps"}  # fmt: skip
# How the three-button navigation is switched on, by maker; the plain Android way comes first.
THREE_BUTTON_OVERLAY = "com.android.internal.systemui.navbar.threebutton"
MAKER_NAV_SETTINGS = {
    "xiaomi": ("global", "force_fsg_nav_bar", "0"),
    "redmi": ("global", "force_fsg_nav_bar", "0"),
    "huawei": ("secure", "secure_gesture_navigation", "0"),
    "honor": ("secure", "secure_gesture_navigation", "0"),
    "oppo": ("secure", "hide_navigationbar_enable", "0"),
    "oneplus": ("secure", "hide_navigationbar_enable", "0"),
    "realme": ("secure", "hide_navigationbar_enable", "0"),
    "vivo": ("secure", "navigation_gesture_on", "0"),
}


class AdbError(SetupError):
    """adb is missing, no tablet answers, or it has not authorised this computer."""


class Adb:
    """One device behind the ``adb`` command."""

    def __init__(self, serial: str | None = None, program: str = "adb", timeout_s: float = 20.0) -> None:
        self.program, self.timeout_s = program, timeout_s
        self.serial = serial or self._only_device()

    def _run(self, *args: str) -> str:
        try:
            done = subprocess.run(
                [self.program, *args], capture_output=True, text=True, timeout=self.timeout_s, check=False
            )
        except FileNotFoundError:
            raise AdbError("这台电脑上没有 adb 命令。") from None
        except subprocess.TimeoutExpired:
            raise AdbError(f"adb {' '.join(args[:3])} 超过 {self.timeout_s:.0f} 秒没有返回。") from None
        if done.returncode != 0:
            raise AdbError(f"adb {' '.join(args[:4])} 失败：{(done.stderr or done.stdout).strip()[:300]}")
        return done.stdout.replace("\r", "")

    def _only_device(self) -> str:
        devices = parse_devices(self._run("devices"))
        if not devices:
            raise AdbError("adb 没有看到任何设备：检查 USB 线，以及平板的“开发者选项 → USB 调试”是否打开。")
        waiting = [serial for serial, state in devices if state == "unauthorized"]
        ready = [serial for serial, state in devices if state == "device"]
        if not ready and waiting:
            raise AdbError(
                f"平板 {waiting[0]} 还没有授权这台电脑：看平板屏幕上的“允许 USB 调试吗？”对话框，勾选“一律允许”，点“允许”。"
            )
        if len(ready) != 1:
            raise AdbError(f"adb 上有 {len(ready)} 台可用设备 {ready}；用 --serial 指明是哪一台。")
        return ready[0]

    def shell(self, command: str) -> str:
        return self._run("-s", self.serial, "shell", command)

    def reverse(self, port: int) -> None:
        """Let the tablet reach this computer's ``port`` as its own localhost:``port``, through the USB cable."""
        self._run("-s", self.serial, "reverse", f"tcp:{port}", f"tcp:{port}")

    def prop(self, name: str) -> str:
        return self.shell(f"getprop {name}").strip()

    def setting(self, namespace: str, key: str) -> str | None:
        value = self.shell(f"settings get {namespace} {key}").strip()
        return None if value in ("null", "") else value

    def put_setting(self, namespace: str, key: str, value: str | None) -> None:
        if value is None:
            self.shell(f"settings delete {namespace} {key}")
        else:
            self.shell(f"settings put {namespace} {key} {value}")


def parse_devices(text: str) -> list[tuple[str, str]]:
    """``adb devices`` -> [(serial, state)]."""
    rows = [line.split() for line in text.splitlines()[1:] if line.strip() and not line.startswith("*")]
    return [(row[0], row[1]) for row in rows if len(row) >= 2]


@dataclasses.dataclass(frozen=True)
class Display:
    width_px: int  # of the display in its natural orientation
    height_px: int
    xdpi: float  # physical pixels per inch
    ydpi: float
    factory_density: int  # Android's "density" in dpi: what the interface is scaled by
    density: int  # the one in force (an override, or the factory one)
    rotation: int  # 0..3 quarter turns from the natural orientation

    @property
    def size_mm(self) -> tuple[float, float]:
        """Width and height of the lit area in the natural orientation."""
        return self.width_px / self.xdpi * 25.4, self.height_px / self.ydpi * 25.4

    @property
    def landscape_px(self) -> tuple[int, int]:
        return max(self.width_px, self.height_px), min(self.width_px, self.height_px)

    @property
    def landscape_mm(self) -> tuple[float, float]:
        return max(self.size_mm), min(self.size_mm)

    @property
    def is_landscape(self) -> bool:
        return (self.width_px >= self.height_px) == (self.rotation % 2 == 0)


def parse_display(wm_size: str, wm_density: str, dumpsys_display: str) -> Display:
    """What ``wm size``, ``wm density`` and ``dumpsys display`` say about the built-in display."""
    size = re.search(r"Physical size:\s*(\d+)x(\d+)", wm_size)
    factory = re.search(r"Physical density:\s*(\d+)", wm_density)
    override = re.search(r"Override density:\s*(\d+)", wm_density)
    info = re.search(r"mBaseDisplayInfo=DisplayInfo\{.*", dumpsys_display)
    dpi = re.search(r"density \d+ \(([\d.]+) x ([\d.]+)\) dpi", info.group(0)) if info else None
    rotation = re.search(r"rotation (\d)", info.group(0)) if info else None
    if not (size and factory and dpi):
        raise AdbError("读不懂平板的显示参数（wm size / wm density / dumpsys display 的输出和预期的格式不一样）。")
    return Display(
        width_px=int(size.group(1)),
        height_px=int(size.group(2)),
        xdpi=float(dpi.group(1)),
        ydpi=float(dpi.group(2)),
        factory_density=int(factory.group(1)),
        density=int(override.group(1)) if override else int(factory.group(1)),
        rotation=int(rotation.group(1)) if rotation else 0,
    )


def largest_density(display: Display) -> int:
    """The density that makes the interface as large as it can be while apps still use their tablet layouts."""
    short_side = min(display.width_px, display.height_px)
    keeps_tablet_layout = int(short_side * 160 / MIN_TABLET_WIDTH_DP)
    return max(display.factory_density, min(keeps_tablet_layout, int(display.factory_density * MAX_UI_SCALE)))


def nav_buttons(window_dump_xml: str, screen_px: tuple[int, int]) -> dict[str, list[float]]:
    """Centres of the navigation bar's buttons in a ``uiautomator dump``, as normalized (u, v) of a screen
    ``screen_px`` wide and high in its present orientation. Empty when the bar has no buttons (gesture navigation)."""
    found = {}
    try:
        nodes = list(ET.fromstring(window_dump_xml).iter("node"))
    except ET.ParseError:
        return found
    for name, resource_id in NAV_BUTTON_IDS.items():
        for node in nodes:
            bounds = re.fullmatch(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", node.get("bounds", ""))
            if node.get("resource-id") == resource_id and bounds:
                x0, y0, x1, y1 = (int(v) for v in bounds.groups())
                found[name] = [round((x0 + x1) / 2 / screen_px[0], 4), round((y0 + y1) / 2 / screen_px[1], 4)]
                break
    return found


@dataclasses.dataclass
class Snapshot:
    """What the tablet was like before it was set up: enough to put it back."""

    serial: str
    model: str
    settings: dict[str, str | None]  # "namespace key" -> value, None: was not set
    density_override: int | None  # None: the factory density was in force

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Snapshot:
        return cls(**data)


def wanted_settings(display: Display) -> dict[str, str]:
    """The settings a tablet on the stand should have. The orientation is kept when it is already landscape."""
    landscape_turn = display.rotation if display.is_landscape else (0 if display.width_px >= display.height_px else 1)
    return {
        "system accelerometer_rotation": "0",  # no auto-rotate
        "system user_rotation": str(landscape_turn),
        "system screen_off_timeout": str(SCREEN_TIMEOUT_MS),
        "global stay_on_while_plugged_in": "7",  # awake on any charger, and the USB cable is one
        "system screen_brightness_mode": "0",  # no adaptive brightness: the camera's view must not change by itself
        "system font_scale": str(FONT_SCALE),
    }


def read_display(adb: Adb) -> Display:
    return parse_display(adb.shell("wm size"), adb.shell("wm density"), adb.shell("dumpsys display"))


def snapshot(adb: Adb, keys: list[str]) -> Snapshot:
    override = re.search(r"Override density:\s*(\d+)", adb.shell("wm density"))
    return Snapshot(
        serial=adb.serial,
        model=f"{adb.prop('ro.product.manufacturer')} {adb.prop('ro.product.model')}",
        settings={key: adb.setting(*key.split()) for key in keys},
        density_override=int(override.group(1)) if override else None,
    )


def restore(adb: Adb, before: Snapshot) -> None:
    for key, value in before.settings.items():
        adb.put_setting(*key.split(), value)
    adb.shell("wm density reset" if before.density_override is None else f"wm density {before.density_override}")


def three_button_navigation(adb: Adb) -> bool:
    """Switch the navigation bar to its three buttons; True when the tablet then says it is in that mode."""
    maker = adb.prop("ro.product.manufacturer").lower()
    try:
        adb.shell(f"cmd overlay enable-exclusive --category {THREE_BUTTON_OVERLAY}")
    except AdbError:
        pass  # this maker's Android has no such overlay: its own setting below
    if maker in MAKER_NAV_SETTINGS:
        adb.put_setting(*MAKER_NAV_SETTINGS[maker])
    return adb.setting("secure", "navigation_mode") == "0"


def dump_window(adb: Adb) -> str:
    adb.shell("uiautomator dump /sdcard/tablet_tap_window.xml")
    text = adb.shell("cat /sdcard/tablet_tap_window.xml")
    adb.shell("rm -f /sdcard/tablet_tap_window.xml")
    return text
