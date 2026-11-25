"""The tablet as the GUI agent sees it (an adapter of ``agent.device.Device``): screenshots from the camera, touches by
the arm.

A screenshot is taken with the arm out of the camera's view of the screen; the camera frame is rectified to the upright full
screen by ``vision.rectifier.ScreenRectifier``. Back and Home are taps on the navigation bar's buttons when their positions
are configured (Android's three-button navigation), else the edge swipes of gesture navigation.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
import time

import cv2
import numpy as np
from PIL import Image

from tablet_tap.core.keyboard import Keyboard
from tablet_tap.core.keyboard import keystrokes
from tablet_tap.errors import UnsupportedAction
from tablet_tap.robot.tablet_robot import ArmPlace
from tablet_tap.robot.tablet_robot import TabletRobot

logger = logging.getLogger("tablet_tap.device")

EDGE = 0.012  # taps are kept this far inside the screen: the rubber tip is about 6 mm wide


def _inside(value: float) -> float:
    return float(np.clip(value, EDGE, 1.0 - EDGE))


class RobotTabletDevice:
    def __init__(
        self,
        robot: TabletRobot,
        camera,
        rectifier,
        *,
        nav: dict | None = None,
        keyboard: Keyboard | None = None,
        view_settle_s: float = 0.5,
        keep_frames: int = 3,
        keep_up: Callable[[RobotTabletDevice], None] | None = None,
    ) -> None:
        """``keep_up`` is called before every look and every touch made from out of the picture: the place to notice
        that the tablet was moved and to bring ``robot`` and ``rectifier`` up to date."""
        self.robot, self.camera, self.rectifier = robot, camera, rectifier
        self.nav = nav or {}
        self.keyboard = keyboard  # where the on-screen keyboard's keys are; None: it was not calibrated
        self.view_settle_s = view_settle_s
        self.keep_frames = keep_frames
        self.keep_up = keep_up
        self.last_frame: np.ndarray | None = None  # the camera frame of the last screenshot (BGR), for logs

    def _keep_up(self) -> None:
        if self.keep_up is not None and self.robot.where in (ArmPlace.PARK, ArmPlace.VIEW):
            self.keep_up(self)

    # ---- seeing ---------------------------------------------------------------------------------------------------
    def screenshot(self) -> Image.Image:
        self.robot.to_view()
        time.sleep(self.view_settle_s)  # the arm's shadow and the camera's exposure settle
        self._keep_up()
        frames = [self.camera.read() for _ in range(max(self.keep_frames, 1))]
        self.last_frame = frames[-1]
        # the median of a few frames removes sensor noise and the display's refresh flicker
        frame = np.median(np.stack(frames), axis=0).astype(np.uint8) if len(frames) > 2 else frames[-1]
        screen = self.rectifier.rectify(frame)
        return Image.fromarray(cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))

    # ---- touching -------------------------------------------------------------------------------------------------
    def click(self, u: float, v: float) -> None:
        self._keep_up()
        self.robot.tap(_inside(u), _inside(v))

    def long_press(self, u: float, v: float, duration_s: float = 0.8) -> None:
        self._keep_up()
        self.robot.tap(_inside(u), _inside(v), hold_s=duration_s)

    def swipe(self, u1: float, v1: float, u2: float, v2: float, duration_s: float = 0.8) -> None:
        self._keep_up()
        self.robot.swipe(_inside(u1), _inside(v1), _inside(u2), _inside(v2), duration_s)

    def back(self) -> None:
        self._keep_up()
        if "back" in self.nav:
            self.robot.tap(*self.nav["back"])
        else:  # gesture navigation: from the left bezel inward
            self.robot.swipe(-0.012, 0.5, 0.22, 0.5, 0.25)

    def home(self) -> None:
        self._keep_up()
        if "home" in self.nav:
            self.robot.tap(*self.nav["home"])
        else:  # gesture navigation: from the bottom bezel upward
            self.robot.swipe(0.5, 1.012, 0.5, 0.55, 0.25)

    def type_text(self, text: str) -> None:
        """Tap ``text`` on the on-screen keyboard, one key at a time. The keyboard must be showing: the model taps the
        input field first, as the prompt asks it to."""
        if self.keyboard is None:
            raise UnsupportedAction("还没有标定屏幕键盘（网页“运行”页里的“键盘标定”）")
        for key in keystrokes(text):
            u, v = self.keyboard.position(key)
            self.robot.tap(_inside(u), _inside(v))
