"""Setting the tablet up over adb, against canned adb output and a pretend device: no tablet needed."""

import json

import pytest

from tablet_tap import android
from tablet_tap.cli import main as cli
from tablet_tap.cli.commands import tablet as tablet_command
from tablet_tap.config import SetupStore

WM_SIZE = "Physical size: 2560x1600\n"
WM_DENSITY = "Physical density: 320\n"
DUMPSYS = (
    "Display Devices: size=1\n"
    '  mBaseDisplayInfo=DisplayInfo{"内置屏幕", displayId 0, FLAG_SECURE, real 2560 x 1600, largest app 2560 x 2560, '
    "smallest app 1600 x 1600, appVsyncOff 1000000, presDeadline 16666666, mode 1, defaultModeId 1, rotation 0, "
    "density 320 (274.571 x 274.571) dpi, layerStack 0, type INTERNAL}\n"
)
WINDOW = """<?xml version='1.0' encoding='UTF-8' standalone='yes' ?><hierarchy rotation="0">
<node index="0" text="" resource-id="com.android.systemui:id/navigation_bar_frame" bounds="[0,1504][2560,1600]">
<node index="0" resource-id="com.android.systemui:id/back" class="android.widget.ImageView" bounds="[1000,1504][1160,1600]" />
<node index="1" resource-id="com.android.systemui:id/home" class="android.widget.ImageView" bounds="[1200,1504][1360,1600]" />
<node index="2" resource-id="com.android.systemui:id/recent_apps" class="android.widget.ImageView" bounds="[1400,1504][1560,1600]" />
</node></hierarchy>"""


def test_reading_what_adb_says():
    listing = "List of devices attached\nfdc517c1\tunauthorized\nemulator-5554\tdevice\n\n"
    assert android.parse_devices(listing) == [("fdc517c1", "unauthorized"), ("emulator-5554", "device")]
    display = android.parse_display(WM_SIZE, WM_DENSITY, DUMPSYS)
    assert (display.width_px, display.height_px, display.factory_density, display.density, display.rotation) == (
        2560,
        1600,
        320,
        320,
        0,
    )
    width_mm, height_mm = display.landscape_mm
    assert abs(width_mm - 236.8) < 0.1 and abs(height_mm - 148.0) < 0.1 and display.is_landscape
    overridden = android.parse_display(WM_SIZE, WM_DENSITY + "Override density: 400\n", DUMPSYS)
    assert overridden.density == 400 and overridden.factory_density == 320
    portrait = android.parse_display(
        "Physical size: 1600x2560\n", WM_DENSITY, DUMPSYS.replace("rotation 0", "rotation 1")
    )
    assert portrait.is_landscape and portrait.landscape_px == (2560, 1600)  # a portrait panel turned a quarter
    with pytest.raises(android.AdbError, match="读不懂"):
        android.parse_display("?", "?", "?")


def test_the_interface_is_enlarged_but_stays_a_tablet():
    display = android.parse_display(WM_SIZE, WM_DENSITY, DUMPSYS)
    density = android.largest_density(display)
    assert density == 426 and 1600 * 160 / density >= 600  # the short side stays 600 dp wide: tablet layouts
    small = android.parse_display("Physical size: 1920x1200\n", "Physical density: 240\n", DUMPSYS)
    assert android.largest_density(small) == 320  # 1200 px / 600 dp
    dense = android.parse_display("Physical size: 2560x1600\n", "Physical density: 200\n", DUMPSYS)
    assert android.largest_density(dense) == 300  # never more than 1.5 times the factory setting
    wanted = android.wanted_settings(display)
    assert wanted["system accelerometer_rotation"] == "0" and wanted["system user_rotation"] == "0"
    upright = android.parse_display("Physical size: 1600x2560\n", WM_DENSITY, DUMPSYS)  # a portrait panel, not turned
    assert android.wanted_settings(upright)["system user_rotation"] == "1"


def test_navigation_buttons_are_found_in_the_window_dump():
    buttons = android.nav_buttons(WINDOW, (2560, 1600))
    assert buttons == {"back": [0.4219, 0.97], "home": [0.5, 0.97], "recent": [0.5781, 0.97]}
    assert android.nav_buttons("<hierarchy/>", (2560, 1600)) == {} == android.nav_buttons("not xml", (2560, 1600))


class PretendTablet:
    """Answers the shell commands the set-up sends, like a tablet with gesture navigation and auto-rotate on."""

    serial = "fdc517c1"

    def __init__(self, three_button_works=True):
        self.settings = {"system accelerometer_rotation": "1", "system user_rotation": "0", "system font_scale": "1.0",
                         "system screen_off_timeout": "60000", "system screen_brightness_mode": "1",
                         "secure navigation_mode": "2"}  # fmt: skip
        self.density_override, self.three_button_works, self.commands, self.reversed = None, three_button_works, [], []

    def shell(self, command: str) -> str:
        self.commands.append(command)
        if command == "wm size":
            return WM_SIZE
        if command == "wm density":
            return WM_DENSITY + (f"Override density: {self.density_override}\n" if self.density_override else "")
        if command == "wm density reset":
            self.density_override = None
        elif command.startswith("wm density "):
            self.density_override = int(command.split()[-1])
        elif command == "dumpsys display":
            return DUMPSYS
        elif command.startswith("cmd overlay enable-exclusive") and self.three_button_works:
            self.settings["secure navigation_mode"] = "0"
        elif command.startswith("cat /sdcard/"):
            return WINDOW if self.settings["secure navigation_mode"] == "0" else "<hierarchy/>"
        return ""

    def prop(self, name: str) -> str:
        return {"ro.product.manufacturer": "Lenovo", "ro.product.model": "TB-J716F", "ro.build.version.release": "13"}[
            name
        ]

    def setting(self, namespace: str, key: str):
        return self.settings.get(f"{namespace} {key}")

    def put_setting(self, namespace: str, key: str, value) -> None:
        if value is None:
            self.settings.pop(f"{namespace} {key}", None)
        else:
            self.settings[f"{namespace} {key}"] = value

    def reverse(self, port: int) -> None:
        self.reversed.append(port)


def test_the_tablet_command_sets_up_records_and_restores(tmp_path, monkeypatch, capsys):
    device = PretendTablet()
    monkeypatch.setattr(android, "Adb", lambda serial=None: device)
    monkeypatch.setattr(tablet_command.time, "sleep", lambda s: None)
    before = dict(device.settings)
    setup = tmp_path / "calib" / "setup.json"
    base = ["--setup", str(setup), "--calib_url", "http://127.0.0.1:8801"]

    assert cli.main([*base, "tablet", "--show"]) == 0
    assert "没有执行" in capsys.readouterr().out and device.settings == before and not setup.exists()

    assert cli.main([*base, "tablet", "--page"]) == 0
    out = capsys.readouterr().out
    assert device.settings["system accelerometer_rotation"] == "0" and device.settings["system font_scale"] == "1.3"
    assert device.settings["global stay_on_while_plugged_in"] == "7" and device.density_override == 426
    saved = SetupStore(setup).load()
    assert saved.nav == {"back": [0.4219, 0.97], "home": [0.5, 0.97]}
    assert abs(saved.screen_size_m[0] - 0.2368) < 1e-4 and abs(saved.aspect - 1.6) < 1e-9
    assert device.reversed == [8801] and any("am start" in c and "localhost:8801/calib" in c for c in device.commands)
    assert "三键导航已开" in out and "Lenovo TB-J716F" in out

    snapshot = next((tmp_path / "calib").glob("tablet_before_*.json"))
    assert json.loads(snapshot.read_text())["settings"]["system accelerometer_rotation"] == "1"
    assert cli.main([*base, "tablet", "--restore", str(snapshot)]) == 0
    assert device.settings["system accelerometer_rotation"] == "1" and device.settings["system font_scale"] == "1.0"
    assert "global stay_on_while_plugged_in" not in device.settings and device.density_override is None


def test_a_tablet_that_keeps_its_gestures_is_reported(tmp_path, monkeypatch, capsys):
    device = PretendTablet(three_button_works=False)
    monkeypatch.setattr(android, "Adb", lambda serial=None: device)
    monkeypatch.setattr(tablet_command.time, "sleep", lambda s: None)
    assert cli.main(["--setup", str(tmp_path / "s.json"), "tablet"]) == 0
    assert "手动选三键导航" in capsys.readouterr().out and SetupStore(tmp_path / "s.json").load().nav is None


def test_an_unauthorised_tablet_is_explained(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(android.Adb, "_run", lambda self, *args: "List of devices attached\nfdc517c1\tunauthorized\n")
    assert cli.main(["--setup", str(tmp_path / "s.json"), "tablet"]) == 1
    assert "允许 USB 调试" in capsys.readouterr().out
