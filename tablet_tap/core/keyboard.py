"""Typing with the pen on the tablet's on-screen QWERTY keyboard.

The keyboard's keys are found once, by the operator pointing at a few of them in the rectified picture of the screen
(``ANCHORS``); the others lie on straight, evenly spaced rows between them. Chinese text is typed as pinyin, and the
word is then picked from the input method's candidates by the model, which sees them in the next picture.
"""

from __future__ import annotations

import dataclasses
import re

from tablet_tap.errors import SetupError
from tablet_tap.errors import UnsupportedAction

ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")
ANCHORS = ("q", "p", "a", "l", "z", "m", "space", "backspace")  # the first and last key of every row, and two more
MAX_KEYSTROKES = 60
ROW_TOLERANCE = 0.02  # of the screen height: how level a row must be

_CHINESE = re.compile(r"[一-鿿]+")


@dataclasses.dataclass(frozen=True)
class Keyboard:
    """Where the keys are, in normalized screen coordinates."""

    anchors: dict[str, tuple[float, float]]

    def __post_init__(self) -> None:
        missing = [key for key in ANCHORS if key not in self.anchors]
        if missing:
            raise SetupError(f"键盘标定缺少这些键：{missing}")
        a = {key: tuple(map(float, value)) for key, value in self.anchors.items()}
        if not all(0.0 <= c <= 1.0 for key in ANCHORS for c in a[key]):
            raise SetupError("键盘标定的点超出了屏幕")
        for first, last in (("q", "p"), ("a", "l"), ("z", "m")):
            if a[last][0] - a[first][0] < 0.1 or abs(a[last][1] - a[first][1]) > ROW_TOLERANCE:
                raise SetupError(f"键盘标定里 {first} 和 {last} 应在同一行、{first} 在左：请重新点这两个键。")
        rows = [a["q"][1], a["a"][1], a["z"][1], a["space"][1]]
        if any(below - above < 0.015 for above, below in zip(rows, rows[1:], strict=False)):
            raise SetupError("键盘标定里各行的上下顺序不对（应从上到下是 q、a、z、空格）：请重新点一遍。")
        if a["backspace"][0] < a["m"][0] or abs(a["backspace"][1] - a["z"][1]) > 2 * ROW_TOLERANCE:
            raise SetupError("退格键应在 z 那一行的最右边：请重新点退格键。")

    def position(self, key: str) -> tuple[float, float]:
        """Where a letter, ``space`` or ``backspace`` is."""
        if key in self.anchors:
            return tuple(self.anchors[key])
        for row in ROWS:
            if key in row:
                first, last = self.anchors[row[0]], self.anchors[row[-1]]
                along = row.index(key) / (len(row) - 1)
                return first[0] + along * (last[0] - first[0]), first[1] + along * (last[1] - first[1])
        raise KeyError(key)


def keystrokes(text: str) -> list[str]:
    """The keys to press for ``text``: letters and spaces, Chinese as pinyin. ``UnsupportedAction`` for anything else
    (digits and punctuation sit on other layers of the keyboard)."""
    text = text.strip()
    if not text:
        raise UnsupportedAction("要输入的文字是空的")
    unsupported = sorted({c for c in text if not (c.isascii() and c.isalpha()) and c != " " and not _CHINESE.match(c)})
    if unsupported:
        raise UnsupportedAction(f"键盘上还不会打这些字符：{''.join(unsupported)}（只会字母、空格和中文）")
    if _CHINESE.search(text):
        from pypinyin import lazy_pinyin  # noqa: PLC0415 - only when there is Chinese to type

        keys = [c for word in lazy_pinyin(text.replace(" ", "")) for c in word.lower()]
    else:
        keys = ["space" if c == " " else c.lower() for c in text]
    if len(keys) > MAX_KEYSTROKES:
        raise UnsupportedAction(f"文字太长，要按 {len(keys)} 个键（最多 {MAX_KEYSTROKES}）：{text!r}")
    return keys


def typing_note(text: str) -> str:
    """What the model is told once text has been typed, for the step log."""
    return "已按拼音打出，待点选候选词" if _CHINESE.search(text) else "已打出"
