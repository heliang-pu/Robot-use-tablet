"""Sending a planned joint path to an arm, one target per control period."""

from __future__ import annotations

from collections.abc import Callable
import time

import numpy as np

from tablet_tap.core.motion import Segment
from tablet_tap.errors import PathError
from tablet_tap.errors import Stopped
from tablet_tap.robot.arm import ArmIO

START_TOL_RAD = 0.06  # a path must begin where the arm is
SETTLE_TOL_RAD = 0.006
SETTLE_TIMEOUT_S = 3.0


def run_segments(
    arm: ArmIO,
    segments: list[Segment],
    *,
    hz: float,
    should_stop: Callable[[], bool] | None = None,
    stop_last_when: Callable[[], bool] | None = None,
    realtime: bool = True,
) -> np.ndarray:
    """Send the segments one target per 1 / hz seconds. Returns the last target sent.

    ``stop_last_when`` is polled during the last segment and ends it early (the touch probe). ``realtime`` False sends
    without sleeping (tests with a FakeArm).
    """
    if not segments:
        return arm.read()
    period = 1.0 / hz
    measured = arm.read()
    gap = float(np.max(np.abs(segments[0].q[0] - measured)))
    if gap > START_TOL_RAD:
        raise PathError(f"路径起点离机械臂当前位置 {gap:.3f} rad（上限 {START_TOL_RAD}）；未执行。")
    sent = measured
    next_tick = time.monotonic()

    def tick(q: np.ndarray) -> None:
        nonlocal sent, next_tick
        if should_stop is not None and should_stop():
            raise Stopped("已停止，机械臂保持在当前位置。")
        arm.command(q)
        sent = q
        if realtime:
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()  # fell behind: do not burst to catch up

    for index, segment in enumerate(segments):
        last = index == len(segments) - 1
        for i, q in enumerate(segment.q):
            tick(q)
            if i % 5 == 4:
                arm.read()  # fault check
            if last and stop_last_when is not None and stop_last_when():
                return sent
        if segment.settle:
            deadline = time.monotonic() + SETTLE_TIMEOUT_S
            steps = 0
            while float(np.max(np.abs(arm.read() - sent))) > SETTLE_TOL_RAD:
                steps += 1
                if (time.monotonic() > deadline) if realtime else (steps > 500):
                    raise PathError(
                        f"机械臂在“{segment.name}”后 {SETTLE_TIMEOUT_S:.0f} s 内没有到位（被挡住或速度太低）。"
                    )
                tick(sent)
        for _ in range(int(round(segment.hold_s * hz))):
            tick(sent)
    return sent
