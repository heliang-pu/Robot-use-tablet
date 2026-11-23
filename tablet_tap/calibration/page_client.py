"""Client of the calibration page's server: where did the pen really touch, and what the page should show."""

from __future__ import annotations

import time
from typing import Any, Protocol

import requests

from tablet_tap.vision.layout import ScreenLayout


class TouchPage(Protocol):
    """What the calibration steps need from the page on the tablet (``CalibClient`` is the real one)."""

    def screen(self) -> dict[str, Any] | None: ...  # what the page reported about its display, None before it has

    def layout(self) -> ScreenLayout: ...  # the markers it draws

    def set_mode(self, mode: str, target: tuple[float, float] | None = None) -> None: ...  # markers / target / blank

    def clear(self) -> None: ...  # remove the touch dot

    def last_down_seq(self) -> int: ...  # sequence number of the newest pointerdown

    def touches(self, since_seq: int = 0, kind: str | None = None) -> list[dict[str, Any]]: ...

    def wait_down(self, after_seq: int, timeout_s: float = 5.0) -> tuple[float, float] | None: ...


class CalibClient:
    """What the other modules need from a running calibration server."""

    POLL_S = 0.05

    def __init__(self, base_url: str = "http://127.0.0.1:8801", timeout_s: float = 3.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._session = requests.Session()
        self._session.trust_env = False  # never through the machine's proxy

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        url = self.base_url + path
        try:
            response = self._session.request(
                method, url, timeout=self.timeout_s, proxies={"http": None, "https": None}, **kwargs
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"连不上标定服务 {url}：{exc}") from exc
        try:
            data = response.json()
        except ValueError:
            data = {}
        if response.status_code != 200:
            message = data.get("error") if isinstance(data, dict) else None
            raise RuntimeError(f"标定服务 {url} 返回 {response.status_code}：{message or response.text[:200]}")
        return data

    def state(self) -> dict[str, Any]:
        return self._request("GET", "/state")

    def screen(self) -> dict[str, Any] | None:
        """What the page last reported about itself (plus `aspect`, `covers_display`), None before it has."""
        return self.state()["screen"]

    def layout(self) -> ScreenLayout:
        """The layout the page draws; RuntimeError (with the server's message) while the page has not reported."""
        return ScreenLayout.from_json(self._request("GET", "/layout"))

    def set_mode(self, mode: str, target: tuple[float, float] | None = None) -> None:
        payload: dict[str, Any] = {"mode": mode}
        if target is not None:
            payload["target"] = [float(target[0]), float(target[1])]
        self._request("POST", "/mode", json=payload)

    def clear(self) -> None:
        """Remove the page's touch dot (the page obeys within about 0.25 s)."""
        self._request("POST", "/clear")

    def last_down_seq(self) -> int:
        """Sequence number of the newest pointerdown, 0 when there has been none. Take it before a tap and pass it
        to wait_down()."""
        last = self.state()["last_down"]
        return 0 if last is None else int(last["seq"])

    def touches(self, since_seq: int = 0, kind: str | None = None) -> list[dict[str, Any]]:
        """The touch events with seq > since_seq, oldest first (optionally of one type)."""
        params: dict[str, Any] = {"since_seq": int(since_seq)}
        if kind is not None:
            params["type"] = kind
        return self._request("GET", "/touches", params=params)["touches"]

    def wait_down(self, after_seq: int, timeout_s: float = 5.0) -> tuple[float, float] | None:
        """(u, v) of the first pointerdown with seq > after_seq; None when none comes within timeout_s."""
        deadline = time.monotonic() + timeout_s
        while True:
            downs = self.touches(after_seq, kind="pointerdown")
            if downs:
                return float(downs[0]["u"]), float(downs[0]["v"])
            if time.monotonic() >= deadline:
                return None
            time.sleep(self.POLL_S)
