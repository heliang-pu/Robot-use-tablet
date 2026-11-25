"""A bench in software for ``--fake`` and the tests: an arm whose pen presses a hidden tablet, and a camera that sees
that tablet's calibration page. Nothing here touches hardware."""

from __future__ import annotations

import cv2
import numpy as np

from tablet_tap.calibration.page_server import CalibState
from tablet_tap.core.kinematics import PenArm
from tablet_tap.core.screen_map import ScreenMap
from tablet_tap.robot.arm import FakeArm
from tablet_tap.vision import layout as screen_layout

SCREEN_PX = (2560, 1600)  # the simulated tablet's display
SCREEN_M = (0.250, 0.160)


def flat_screen(centre: tuple[float, float, float] = (0.26, 0.05, 0.003), tilt_deg: float = 0.0) -> ScreenMap:
    """A 250 x 160 mm screen lying in front of the arm, its bottom edge toward the robot: where the upright pen reaches
    all of it, bezel included, within the firmware's limits (joint 5 at its 75 deg over the far edge)."""
    tilt = np.radians(tilt_deg)
    right = np.array([0.0, -1.0, 0.0])
    up_slope = np.array([np.cos(tilt), 0.0, np.sin(tilt)])
    down = -up_slope
    origin = np.asarray(centre, np.float64) - 0.5 * SCREEN_M[0] * right - 0.5 * SCREEN_M[1] * down
    return ScreenMap(origin, SCREEN_M[0] * right, SCREEN_M[1] * down, tool_x=up_slope)


class VirtualTablet(FakeArm):
    """A software arm whose pen presses a real (hidden) screen: every crossing of the glass is a pointerdown that the
    calibration page's state receives, as from the tablet's browser."""

    def __init__(
        self, q: np.ndarray, pen: PenArm, screen: ScreenMap, state: CalibState, trigger_mm: float = 0.3
    ) -> None:
        super().__init__(q)
        self.pen, self.screen, self.state, self.trigger = pen, screen, state, trigger_mm * 1e-3
        self.down = False
        self.downs: list[tuple[float, float]] = []
        state.set_screen(
            {"inner_w": SCREEN_PX[0], "inner_h": SCREEN_PX[1], "screen_w": SCREEN_PX[0], "screen_h": SCREEN_PX[1],
             "dpr": 1, "fullscreen": True, "user_agent": "tablet_tap simulated tablet"}
        )  # fmt: skip

    def command(self, q: np.ndarray) -> None:
        super().command(q)
        u, v, h = self.screen.uv_of(self.pen.fk(self.q)[:3, 3])
        pressed = h <= -self.trigger and 0 <= u <= 1 and 0 <= v <= 1  # a rubber tip needs a little pressure
        if pressed and not self.down:
            self.state.add_touch({"type": "pointerdown", "u": u, "v": v, "t_ms": 0, "pointer_type": "sim"})
            self.downs.append((u, v))
        elif self.down and not pressed:
            self.state.add_touch({"type": "pointerup", "u": u, "v": v, "t_ms": 0, "pointer_type": "sim"})
        self.down = pressed


class SyntheticCamera:
    """A camera that looks obliquely at the simulated tablet and sees what its calibration page shows now."""

    SIZE = (1280, 720)
    QUAD = np.float32([[330, 190], [960, 170], [1020, 560], [280, 590]])  # the screen's TL, TR, BR, BL in the frame

    def __init__(self, state: CalibState) -> None:
        self.state = state

    def read(self) -> np.ndarray:
        snap = self.state.snapshot()
        aspect = SCREEN_PX[0] / SCREEN_PX[1]
        last = snap["last_down"]
        screen = screen_layout.render_layout(
            screen_layout.default_layout(aspect),
            1280,
            mode=snap["mode"],
            target=None if snap["target"] is None else tuple(snap["target"]),
            dot=None if last is None else (last["u"], last["v"]),
        )
        height = screen.shape[0]
        corners = np.float32([[0, 0], [1280, 0], [1280, height], [0, height]])
        warp = cv2.getPerspectiveTransform(corners, self.QUAD)
        return cv2.warpPerspective(screen, warp, self.SIZE, borderValue=(150, 150, 150))

    def close(self) -> None:
        pass
