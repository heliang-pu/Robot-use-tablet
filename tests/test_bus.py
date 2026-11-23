"""The listen-only check of an arm's CAN channel, against a pretend bus."""

import types

import pytest

from tablet_tap.robot import piper


class PretendBus:
    def __init__(self, rates, seconds):
        self.queue = [
            types.SimpleNamespace(arbitration_id=i) for i, hz in rates.items() for _ in range(int(hz * seconds))
        ]

    def recv(self, timeout):
        return self.queue.pop() if self.queue else None

    def shutdown(self):
        pass


HEALTHY = dict.fromkeys(piper.FEEDBACK_FRAMES, 200)


@pytest.fixture
def fake_can(monkeypatch):
    can = types.SimpleNamespace(CanError=OSError, interface=types.SimpleNamespace(Bus=None))
    monkeypatch.setitem(__import__("sys").modules, "can", can)
    return can


@pytest.mark.parametrize(
    ("rates", "said"),
    [
        (HEALTHY, "正常"),
        ({**HEALTHY, 0x2A6: 96, 0x2A8: 87}, "在丢帧：每秒 关节3-4 96、夹爪 87"),  # the status frames alone look fine
        (dict.fromkeys(piper.FEEDBACK_FRAMES, 0), "听不到"),
    ],
)
def test_every_kind_of_feedback_counts(fake_can, monkeypatch, rates, said):
    fake_can.interface.Bus = lambda **kwargs: PretendBus(rates, 1.0)
    monkeypatch.setattr(piper.time, "monotonic", iter(x * 0.0005 for x in range(10**6)).__next__)
    assert said in piper.describe_bus("can_x")


def test_a_channel_that_cannot_be_opened(fake_can):
    def broken(**kwargs):
        raise OSError("no such device")

    fake_can.interface.Bus = broken
    assert "打不开" in piper.describe_bus("can_x") and piper.bus_rate("can_x") is None
