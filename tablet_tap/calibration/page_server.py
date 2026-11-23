"""The calibration page's server: the tablet's browser opens http://<pc>:8801/calib.

The page draws ArUco markers at the positions of `vision.layout.default_layout` (camera -> screen calibration), reports
every touch back (where did the stylus really land), and draws what the server asks for:

- `mode = "markers"` (default): the markers;
- `mode = "target"`: a crosshair at `target = [u, v]` (the tap-accuracy test);
- `mode = "blank"`: the dark background only.

In every mode the page draws a magenta dot at the last pointerdown (`vision.rectifier.find_touch_dot` finds it in a rectified
camera frame). State lives in memory; restarting the server forgets it (the page reports its screen again by itself).

    bin/py -m tablet_tap.calibration.page_server --host 0.0.0.0 --port 8801

`calibration/page_client.CalibClient` is what the other modules use to talk to it.
"""

from __future__ import annotations

import argparse
import collections
import logging
import math
import pathlib
import socket
import threading
import time
from typing import Any

import flask

from tablet_tap.vision import layout as screen_layout

logger = logging.getLogger(__name__)

STATIC_DIR = pathlib.Path(__file__).resolve().parent / "static"
MODES = ("markers", "target", "blank")
TOUCH_TYPES = ("pointerdown", "pointermove", "pointerup", "pointercancel")
MAX_TOUCHES = 5000
OWNER_TIMEOUT_S = 5.0  # a page that has not polled for this long gives the calibration up
# The page covers the display when its viewport and the screen agree within this many CSS pixels.
COVER_TOLERANCE_PX = 1.5


def page_style() -> dict[str, Any]:
    """The sizes and colours the page draws with (fractions of the screen height): one source, screen_layout."""
    return {
        "background": screen_layout.BACKGROUND_HEX,
        "dot_color": screen_layout.DOT_HEX,
        "dot_diameter": screen_layout.DOT_DIAMETER,
        "dot_ring": screen_layout.DOT_RING,
        "target_arm": screen_layout.TARGET_ARM,
        "target_ring": screen_layout.TARGET_RING,
        "target_line": screen_layout.TARGET_LINE,
        "marker_cells": screen_layout.MARKER_CELLS,
    }


def _number(value: Any) -> float | None:
    """A finite float, or None for anything else (bool included)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _uv(value: Any) -> tuple[float, float] | None:
    """(u, v) inside the screen, or None."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    u, v = _number(value[0]), _number(value[1])
    if u is None or v is None or not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        return None
    return u, v


class CalibState:
    """Everything the server remembers, behind one lock."""

    def __init__(self, max_touches: int = MAX_TOUCHES) -> None:
        self._lock = threading.Lock()
        self.screen: dict[str, Any] | None = None
        self.mode = "markers"
        self.target: tuple[float, float] | None = None
        self.clear_seq = 0
        self.touches: collections.deque[dict[str, Any]] = collections.deque(maxlen=max_touches)
        self.seq = 0  # of the newest touch event; the first one gets 1
        self.last_down: dict[str, Any] | None = None
        self.page_seen: float | None = None  # time.time() of the page's last poll
        self.owner: str | None = None  # address of the device whose screen report is kept
        self.owner_seen: float | None = None

    def taken_by_another(self, address: str | None) -> bool:
        """Another device shows the calibration page in full screen right now: a second open page (the operator's own
        browser) must not mix its size and touches into the tablet's."""
        fresh = self.owner_seen is not None and time.time() - self.owner_seen < OWNER_TIMEOUT_S
        covers = self.screen is not None and self.screen["covers_display"]
        return fresh and covers and self.owner is not None and address != self.owner

    def set_screen(self, info: dict[str, Any]) -> dict[str, Any]:
        """Store what the page reports about itself; ValueError (Chinese) when it is not usable."""
        sizes = {key: _number(info.get(key)) for key in ("inner_w", "inner_h", "screen_w", "screen_h")}
        if any(value is None or value <= 0 for value in sizes.values()):
            raise ValueError("需要正数的 inner_w、inner_h、screen_w、screen_h")
        inner, display = (sizes["inner_w"], sizes["inner_h"]), (sizes["screen_w"], sizes["screen_h"])
        # Some browsers keep screen.width/height in the device's natural orientation: the swapped pair counts too.
        covers = any(
            abs(inner[0] - w) <= COVER_TOLERANCE_PX and abs(inner[1] - h) <= COVER_TOLERANCE_PX
            for w, h in (display, display[::-1])
        )
        screen = {
            **sizes,
            "dpr": _number(info.get("dpr")) or 1.0,
            "fullscreen": bool(info.get("fullscreen")),
            "user_agent": str(info.get("user_agent", ""))[:300],
            "aspect": inner[0] / inner[1],
            "covers_display": covers,
            "t_server": time.time(),
        }
        with self._lock:
            self.screen = screen
        return screen

    def add_touch(self, event: dict[str, Any]) -> dict[str, Any]:
        kind, u, v = event.get("type"), _number(event.get("u")), _number(event.get("v"))
        if kind not in TOUCH_TYPES or u is None or v is None:
            raise ValueError(f"触摸事件需要 type（{'、'.join(TOUCH_TYPES)}）和数值 u、v")
        touch = {
            "type": kind,
            "u": u,
            "v": v,
            "t_ms": _number(event.get("t_ms")),
            "pointer_type": str(event.get("pointer_type", ""))[:20],
            "pointer_id": _number(event.get("pointer_id")),
            "t_server": time.time(),
        }
        with self._lock:
            self.seq += 1
            touch["seq"] = self.seq
            self.touches.append(touch)
            if kind == "pointerdown":
                self.last_down = {key: touch[key] for key in ("u", "v", "t_server", "seq")}
        return touch

    def set_mode(self, mode: Any, target: Any) -> None:
        if mode not in MODES:
            raise ValueError(f"mode 应为 {'、'.join(MODES)} 之一")
        point = None if target is None else _uv(target)
        if target is not None and point is None:
            raise ValueError("target 应为 [u, v]，u、v 都在 0 到 1 之间")
        if mode == "target" and point is None:
            raise ValueError("mode 为 target 时要给出 target: [u, v]")
        with self._lock:
            self.mode, self.target = mode, point

    def clear(self) -> int:
        """Ask the page to remove its dot. The touch history and `last_down` stay (so sequence numbers keep going)."""
        with self._lock:
            self.clear_seq += 1
            return self.clear_seq

    def touches_after(self, since_seq: int, kind: str | None = None) -> tuple[list[dict[str, Any]], int]:
        with self._lock:
            events = [t for t in self.touches if t["seq"] > since_seq and (kind is None or t["type"] == kind)]
            return events, self.seq

    def snapshot(self, from_page: bool = False) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            if from_page:
                self.page_seen = now
            return {
                "screen": self.screen,
                "mode": self.mode,
                "target": None if self.target is None else list(self.target),
                "clear_seq": self.clear_seq,
                "last_down": self.last_down,
                "n_touches": self.seq,
                "last_seq": self.seq,
                "page_age_s": None if self.page_seen is None else round(now - self.page_seen, 3),
                "t_server": now,
            }


def _layout_response(screen: dict[str, Any] | None) -> tuple[dict[str, Any], int]:
    """The /layout body for the screen last reported, or an error body, with the HTTP status."""
    if screen is None:
        return {"error": "还不知道屏幕大小：先在平板的浏览器里打开 /calib 页面"}, 409
    try:
        layout = screen_layout.default_layout(screen["aspect"])
    except ValueError as exc:
        return {"error": str(exc)}, 409
    body = layout.to_json() | {"covers_display": screen["covers_display"], "screen": screen, "style": page_style()}
    return body, 200


def create_app(state: CalibState | None = None) -> flask.Flask:
    """The Flask app; its state is `app.config["CALIB_STATE"]`."""
    app = flask.Flask(__name__, static_folder=None)
    state = state or CalibState()
    app.config["CALIB_STATE"] = state

    def body() -> dict[str, Any]:
        data = flask.request.get_json(force=True, silent=True)  # sendBeacon may send JSON as text/plain
        return data if isinstance(data, dict) else {}

    def error(message: str, status: int = 400) -> tuple[flask.Response, int]:
        return flask.jsonify({"error": message}), status

    @app.after_request
    def no_cache(response: flask.Response) -> flask.Response:
        if flask.request.endpoint != "marker":
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index() -> Any:
        return flask.redirect("/calib")

    @app.get("/calib")
    def calib() -> Any:
        return flask.send_from_directory(STATIC_DIR, "calib.html")

    @app.get("/marker/<int:marker_id>.png")
    def marker(marker_id: int) -> Any:
        try:
            png = screen_layout.marker_png(marker_id, flask.request.args.get("cell", default=40, type=int))
        except ValueError as exc:
            return error(str(exc), 404)
        response = flask.Response(png, mimetype="image/png")
        response.headers["Cache-Control"] = "public, max-age=86400"
        return response

    @app.get("/layout")
    def layout() -> Any:
        data, status = _layout_response(state.snapshot()["screen"])
        return flask.jsonify(data), status

    @app.post("/screen")
    def screen() -> Any:
        """Stores the page's report; answers with the layout for it (or why there is none), saving the page a
        round trip."""
        try:
            if state.taken_by_another(flask.request.remote_addr):
                return flask.jsonify({"error": "平板正在用标定页；这台设备的上报不算，请关掉这个页面。"}), 409
            stored = state.set_screen(body())
        except ValueError as exc:
            return error(str(exc))
        state.owner = flask.request.remote_addr
        data, status = _layout_response(stored)
        if status != 200:
            return flask.jsonify({"ok": True, "screen": stored, "layout": None, "error": data["error"]})
        return flask.jsonify({"ok": True, "screen": stored, "layout": data})

    @app.post("/touch")
    def touch() -> Any:
        try:
            if state.taken_by_another(flask.request.remote_addr):
                return flask.jsonify({"error": "平板正在用标定页；这台设备的触摸不算。"}), 409
            stored = state.add_touch(body())
        except ValueError as exc:
            return error(str(exc))
        return flask.jsonify({"ok": True, "seq": stored["seq"]})

    @app.get("/state")
    def get_state() -> Any:
        from_page = flask.request.args.get("page") == "1" and not state.taken_by_another(flask.request.remote_addr)
        if from_page:
            state.owner_seen = time.time()
        return flask.jsonify(state.snapshot(from_page=from_page))

    @app.post("/mode")
    def mode() -> Any:
        data = body()
        try:
            state.set_mode(data.get("mode"), data.get("target"))
        except ValueError as exc:
            return error(str(exc))
        return flask.jsonify({"ok": True, **{k: state.snapshot()[k] for k in ("mode", "target")}})

    @app.post("/clear")
    def clear() -> Any:
        return flask.jsonify({"ok": True, "clear_seq": state.clear()})

    @app.get("/touches")
    def touches() -> Any:
        since = flask.request.args.get("since_seq", default=0, type=int)
        kind = flask.request.args.get("type")
        if kind is not None and kind not in TOUCH_TYPES:
            return error(f"type 应为 {'、'.join(TOUCH_TYPES)} 之一")
        events, last = state.touches_after(since, kind)
        return flask.jsonify({"touches": events, "last_seq": last})

    return app


def _local_addresses() -> list[str]:
    """This machine's IPv4 addresses a tablet could reach (no packet is sent)."""
    addresses: list[str] = []
    for probe in ("192.168.1.1", "10.0.0.1", "8.8.8.8"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect((probe, 9))
                address = sock.getsockname()[0]
        except OSError:
            continue
        if address not in addresses and not address.startswith("127."):
            addresses.append(address)
    return addresses


local_addresses = _local_addresses


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="平板标定页面的服务：平板浏览器打开 http://<本机地址>:8801/calib")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址，默认 0.0.0.0（局域网里的平板可以访问）")
    parser.add_argument("--port", type=int, default=8801, help="端口，默认 8801")
    parser.add_argument("--verbose", action="store_true", help="打印每一个请求（触摸和轮询很多，默认不打印）")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if not args.verbose:
        logging.getLogger("werkzeug").setLevel(logging.WARNING)
    hosts = _local_addresses() if args.host in ("0.0.0.0", "::") else [args.host]
    for host in hosts or ["<本机地址>"]:
        print(f"在平板的浏览器（Chrome）里打开：http://{host}:{args.port}/calib ，然后点“点这里进入全屏”。", flush=True)
    print("平板请横屏放置，关掉自动旋转和自动息屏，亮度不要开到最大（相机里标定码不能过曝）。", flush=True)
    create_app().run(host=args.host, port=args.port, threaded=True, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
