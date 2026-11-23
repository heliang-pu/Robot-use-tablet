"""The port every arm implements, and the software arm for dry runs and tests."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


class ArmIO(Protocol):
    def read(self) -> np.ndarray: ...  # (6,) measured joints (rad, simulator convention); raises on a fault

    def command(self, q: np.ndarray) -> None: ...  # one joint target

    def close(self) -> None: ...


class FakeArm:
    """An arm that moves toward its target by a fraction per ``command`` (1.0: at once)."""

    def __init__(self, q: np.ndarray, follow: float = 1.0) -> None:
        self.q = np.asarray(q, np.float64).copy()
        self.follow = follow
        self.commands: list[np.ndarray] = []

    def read(self) -> np.ndarray:
        return self.q.copy()

    def command(self, q: np.ndarray) -> None:
        q = np.asarray(q, np.float64)
        self.commands.append(q.copy())
        self.q = self.q + self.follow * (q - self.q)

    def close(self) -> None:
        pass


@runtime_checkable
class GripsPen(Protocol):
    """An arm whose gripper this program controls."""

    def set_grip_width(self, width_m: float) -> None: ...  # sent with every joint target from now on
