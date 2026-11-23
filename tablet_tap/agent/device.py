"""The device the GUI agent drives: something that shows a screenshot and performs touches.

Coordinates are normalized over the full physical display in landscape: ``u`` in [0, 1] from left to right, ``v`` in
[0, 1] from top to bottom.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import os
from typing import Protocol

from PIL import Image

from tablet_tap.errors import UnsupportedAction


class Device(Protocol):
    """What the agent loop needs from a device; the robot-backed implementation follows this."""

    def screenshot(self) -> Image.Image:
        """Return the full screen, upright, RGB."""
        ...

    def click(self, u: float, v: float) -> None:
        """Tap at (u, v)."""
        ...

    def long_press(self, u: float, v: float, duration_s: float = 0.8) -> None:
        """Press at (u, v) and hold for ``duration_s``."""
        ...

    def swipe(self, u1: float, v1: float, u2: float, v2: float, duration_s: float = 0.8) -> None:
        """Drag from (u1, v1) to (u2, v2) in ``duration_s``."""
        ...

    def back(self) -> None:
        """Go back one screen."""
        ...

    def home(self) -> None:
        """Go to the home screen."""
        ...

    def type_text(self, text: str) -> None:
        """Enter text into the focused field; may raise UnsupportedAction."""
        ...


Screen = Image.Image | str | os.PathLike


class FakeDevice:
    """Replays screenshots (a list of images/paths, or a callable step -> image) and records every action.

    For tests and offline runs. With a list, screenshot number ``i`` (from 0) returns item ``i`` and the last item
    repeats. ``can_type=False`` makes ``type_text`` raise UnsupportedAction, like a device that only has a stylus.
    """

    def __init__(self, screens: Sequence[Screen] | Callable[[int], Screen], *, can_type: bool = True) -> None:
        if not callable(screens):
            screens = list(screens)
            if not screens:
                raise ValueError("FakeDevice needs at least one screenshot")
        self._screens = screens
        self.can_type = can_type
        self.step = 0  # number of screenshots handed out so far
        self.actions: list[dict] = []

    def screenshot(self) -> Image.Image:
        """Return the screenshot of the current step and move on to the next one."""
        index = self.step
        self.step += 1
        screen = self._screens(index) if callable(self._screens) else self._screens[min(index, len(self._screens) - 1)]
        if isinstance(screen, Image.Image):
            return screen.convert("RGB")
        with Image.open(os.fspath(screen)) as image:
            return image.convert("RGB")

    def click(self, u: float, v: float) -> None:
        self.actions.append({"action": "click", "u": u, "v": v})

    def long_press(self, u: float, v: float, duration_s: float = 0.8) -> None:
        self.actions.append({"action": "long_press", "u": u, "v": v, "duration_s": duration_s})

    def swipe(self, u1: float, v1: float, u2: float, v2: float, duration_s: float = 0.8) -> None:
        self.actions.append({"action": "swipe", "u1": u1, "v1": v1, "u2": u2, "v2": v2, "duration_s": duration_s})

    def back(self) -> None:
        self.actions.append({"action": "back"})

    def home(self) -> None:
        self.actions.append({"action": "home"})

    def type_text(self, text: str) -> None:
        if not self.can_type:
            raise UnsupportedAction(f"FakeDevice(can_type=False) cannot type {text!r}")
        self.actions.append({"action": "type_text", "text": text})
