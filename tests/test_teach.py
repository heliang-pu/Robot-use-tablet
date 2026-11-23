"""The teaching session, key by key, with a software arm and a page that reports touches of a synthetic screen."""

from helpers import ARM
from helpers import SEED
from helpers import true_screen
import numpy as np
import pytest

from tablet_tap.calibration.teach import TeachSession
from tablet_tap.core.motion import MotionLimits
from tablet_tap.core.motion import TapPlanner
from tablet_tap.errors import CalibrationError
from tablet_tap.robot.arm import FakeArm


class FakePage:
    """Reports a pointerdown whenever the arm's pen is on (or under) the synthetic screen."""

    def __init__(self, arm, screen):
        self.arm, self.screen, self.events, self.down = arm, screen, [], False

    def _look(self):
        u, v, h = self.screen.uv_of(ARM.fk(self.arm.q)[:3, 3])
        pressed = h <= -0.0003
        if pressed and not self.down:
            self.events.append({"seq": len(self.events) + 1, "type": "pointerdown", "u": u, "v": v})
        self.down = pressed

    def last_down_seq(self):
        self._look()
        return len(self.events)

    def touches(self, since_seq=0, kind=None):
        self._look()
        return [e for e in self.events if e["seq"] > since_seq]


def session(u=0.5, v=0.5, height=0.012, **kwargs):
    screen = true_screen()
    arm = FakeArm(ARM.solve(TapPlanner(ARM, screen).tip_pose(u, v, height), SEED))
    said = []
    page = FakePage(arm, screen)
    teach = TeachSession(arm, ARM, MotionLimits(), page=page, realtime=False, say=said.append, **kwargs)
    return teach, arm, screen, said


def tip(arm):
    return ARM.fk(arm.q)[:3, 3]


def test_jog_keys_move_the_pen_tip():
    teach, arm, _, said = session()
    start = tip(arm)
    assert teach.press("w") and np.allclose(tip(arm) - start, [0.005, 0, 0], atol=2e-4)  # 5 mm forward
    teach.press("2")  # 1 mm steps
    teach.press("r")
    assert np.allclose(tip(arm) - start, [0.005, 0, 0.001], atol=2e-4) and "1 mm" in said[-1]
    axis = ARM.fk(arm.q)[:3, 2].copy()
    teach.press("i")  # a turn about the tip: the tip stays, the pen's direction changes by 1 degree
    assert np.allclose(tip(arm) - start, [0.005, 0, 0.001], atol=3e-4)
    assert abs(np.degrees(np.arccos(ARM.fk(arm.q)[:3, 2] @ axis)) - 1.0) < 0.05
    teach.press("4")
    for _ in range(40):  # 20 mm steps away: at some point out of reach, said and not done
        teach.press("w")
    assert any("到不了" in line for line in said) and np.linalg.norm(tip(arm)) < 0.9
    assert teach.press("?") and not teach.press("q")  # an unknown key is ignored; q ends the session


def test_park_probe_and_fit():
    teach, arm, screen, said = session()
    teach.press("p")
    assert np.allclose(teach.park_q, arm.q)
    assert teach.fit(aspect=None) is None  # no touches yet
    planner = TapPlanner(ARM, screen)
    for u, v in ((0.15, 0.2), (0.85, 0.2), (0.85, 0.8), (0.15, 0.8), (0.5, 0.5)):
        arm.q = ARM.solve(planner.tip_pose(u, v, 0.012), SEED)
        teach._target = ARM.fk(arm.q)
        teach.press("z")  # down along the pen until the page reports the touch, then up again
        assert abs(teach.touches[-1].u - u) < 2e-3 and abs(teach.touches[-1].v - v) < 2e-3
        assert screen.uv_of(tip(arm))[2] > 0.01  # lifted
    assert len(teach.touches) == 5
    teach.press("x")
    assert len(teach.touches) == 4
    fitted, seed_q = teach.fit(aspect=0.25 / 0.16)
    for u, v in ((0, 0), (1, 1), (0.5, 0.5)):
        assert np.linalg.norm(fitted.point(u, v) - screen.point(u, v)) < 1e-3
    assert np.allclose(
        seed_q, teach.touches[int(np.argmin([abs(t.u - 0.5) + abs(t.v - 0.5) for t in teach.touches]))].q
    )
    with pytest.raises(CalibrationError, match="宽高比"):
        teach.fit(aspect=1.0)


def test_read_only_session_records_but_never_moves():
    teach, arm, screen, _ = session(moves_arm=False)
    before = arm.q.copy()
    for key in "wsadrfecijkluogGz":
        teach.press(key)
    assert np.array_equal(arm.q, before) and not arm.commands
    arm.q = ARM.solve(TapPlanner(ARM, screen).tip_pose(0.3, 0.4, -0.001), SEED)  # the operator's hand puts the pen down
    teach.poll()
    assert len(teach.touches) == 1 and abs(teach.touches[0].u - 0.3) < 1e-3 and np.allclose(teach.touches[0].q, arm.q)
    keys = iter(["p", None, "q"])
    teach.run(lambda timeout: next(keys))
    assert teach.park_q is not None
