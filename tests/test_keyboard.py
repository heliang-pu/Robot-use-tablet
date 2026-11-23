"""The on-screen keyboard: key positions from a few points, and what the arm presses for a text."""

import pytest

from tablet_tap.core.keyboard import Keyboard
from tablet_tap.core.keyboard import keystrokes
from tablet_tap.errors import SetupError
from tablet_tap.errors import UnsupportedAction

ANCHORS = {
    "q": (0.05, 0.60), "p": (0.95, 0.60), "a": (0.10, 0.71), "l": (0.90, 0.71), "z": (0.20, 0.82), "m": (0.80, 0.82),
    "space": (0.5, 0.93), "backspace": (0.95, 0.82),
}  # fmt: skip


def test_letters_lie_evenly_between_the_row_ends():
    board = Keyboard(ANCHORS)
    u, v = board.position("y")  # q w e r t [y] : 5 of 9 steps along the top row
    assert abs(u - (0.05 + 0.9 * 5 / 9)) < 1e-9 and v == pytest.approx(0.60)
    assert board.position("f") == pytest.approx((0.10 + 0.8 * 3 / 8, 0.71))
    assert board.position("b") == pytest.approx((0.20 + 0.6 * 4 / 6, 0.82))
    assert board.position("space") == (0.5, 0.93)
    with pytest.raises(KeyError):
        board.position("1")


@pytest.mark.parametrize(
    ("change", "said"),
    [
        ({"p": (0.05, 0.60)}, "q 和 p"),
        ({"a": (0.1, 0.5)}, "行"),
        ({"space": (0.5, 0.5)}, "上下顺序"),
        ({"backspace": (0.3, 0.82)}, "退格"),
        ({"q": (1.4, 0.60)}, "超出"),
    ],
)
def test_a_wrong_calibration_is_refused(change, said):
    with pytest.raises(SetupError, match=said):
        Keyboard({**ANCHORS, **change})
    with pytest.raises(SetupError, match="缺少"):
        Keyboard({k: v for k, v in ANCHORS.items() if k != "m"})


def test_text_becomes_keystrokes():
    assert keystrokes("晴天") == list("qingtian")
    assert keystrokes("周杰伦 晴天") == list(
        "zhoujielunqingtian"
    )  # no space inside Chinese: it would commit a candidate
    assert keystrokes("重庆") == list("chongqing")  # read as a word, not character by character
    assert keystrokes("Let it be") == list("letit")[:3] + ["space"] + list("it") + ["space"] + list("be")
    assert keystrokes("女儿情")[:2] == ["n", "v"]  # ü is typed as v
    for bad in ("晴天2024", "晴天，", "", "   ", "a" * 80):
        with pytest.raises(UnsupportedAction):
            keystrokes(bad)
