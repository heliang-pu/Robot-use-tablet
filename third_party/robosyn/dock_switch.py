"""item_assembly: when the base policy (Phoenix) hands the docking to the RL policy.

The rule, on the joint state alone (what the organizers' robot gives a policy as well; their recordings' end-effector
poses are not given at evaluation, and the two arms' bases are not calibrated against each other):

    both grippers hold a tube (empty_below <= opening < closed_below, at two checks in a row),
    then the hands moved towards each other (the left TCP minus the right one changed faster than ``fast`` m per
    step over the last check),
    then that relative motion and each hand slowed below ``slow``  ->  hand over (and stay handed over).

Checked every ``every`` steps like the runtime's gates; speeds are displacements since the previous check over the
steps between, of each TCP in its own arm's base frame (``tcp_positions``: numpy FK of the CobotMagic arm, no
EmbodiChain needed) and of their difference: the two hands lifting their tubes together barely change it (10-27 a
handover right after the lift, with each hand's own speed), the approach does; the bases' spacing, which differs
between our bench and the organizers' robot, drops out. A loose grip for a moment (Phoenix's regrasps read 0.5-0.6)
does not forget the approach; a gripper closed on nothing (~0) does.

Replayed 10-27 (held 0.15-0.5, fast 4 mm, slow 1.5 mm per step at 10 Hz): the organizers' 60 item_assembly
recordings fire in 58, the right TCP then 4.1 cm (median; 8.7 cm at 90 %) from where it ends docked -- the tip in
front of the socket; of Phoenix's takeovers (real10 e3 / e2) 19 / 21 and 8 / 9 fire before the operator took over
(Phoenix stops in front of the socket and the operator waits); the first HIL episodes' two misfires (both grippers
empty; right after the lift) do not fire. pi0.5 (r0) never makes the fast approach and does not fire: the rule
follows Phoenix's motion. A gripper threshold of 0.6 fires in all 60 but 10 % of them 20 cm away.
"""

from __future__ import annotations

import dataclasses

import numpy as np

# CobotMagicNoGripper.urdf (EmbodiChain's CobotMagicArm asset): joint1..joint6 (origin xyz, origin rpy), all about z
_CHAIN = (
    ((0.0, 0.0, 0.123), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (1.5707963267948966, -0.1358999999, -3.141592653589793)),
    ((0.28503, 0.0, 0.0), (0.0, 0.0, -1.79389999999)),
    ((-0.021984, -0.25075, 0.0), (1.5707963267948966, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (-1.5707963267948966, 0.0, 0.0)),
    ((0.0, -0.091, 0.0), (1.5707963267948966, 0.0, 0.0)),
)
# the TCP in link6 (EmbodiChain's cobotmagic OPWSolverCfg tcp, as policy/openwam/eef_kinematics.TCP)
TCP = np.array([[-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.143], [0.0, 0.0, 0.0, 1.0]])
LEFT_ARM, RIGHT_ARM, LEFT_GRIPPER, RIGHT_GRIPPER = slice(0, 6), slice(7, 13), 6, 13


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


_ORIGINS = []
for _xyz, _rpy_angles in _CHAIN:
    _m = np.eye(4)
    _m[:3, :3], _m[:3, 3] = _rpy(*_rpy_angles), _xyz
    _ORIGINS.append(_m)


def fk(q6: np.ndarray) -> np.ndarray:
    """(N, 6) arm joints (rad) -> (N, 4, 4) TCP poses in the arm's base frame."""
    q = np.asarray(q6, np.float64).reshape(-1, 6)
    out = np.tile(np.eye(4), (len(q), 1, 1))
    for j, origin in enumerate(_ORIGINS):
        c, s = np.cos(q[:, j]), np.sin(q[:, j])
        rot = np.zeros((len(q), 4, 4))
        rot[:, 0, 0], rot[:, 0, 1], rot[:, 1, 0], rot[:, 1, 1] = c, -s, s, c
        rot[:, 2, 2] = rot[:, 3, 3] = 1.0
        out = out @ origin @ rot
    return out @ TCP


def tcp_positions(q14: np.ndarray) -> np.ndarray:
    """(N, 14) states -> (N, 2, 3): the left and the right TCP position (m), each in its own arm's base frame."""
    q = np.asarray(q14, np.float64).reshape(-1, 14)
    return np.stack([fk(q[:, LEFT_ARM])[:, :3, 3], fk(q[:, RIGHT_ARM])[:, :3, 3]], axis=1)


@dataclasses.dataclass(frozen=True)
class SwitchRule:
    closed_below: float = 0.5  # gripper opening (state) below which a gripper holds its tube ...
    # ... and above which it does: closed on nothing reads ~0 (a missed grasp; 10-27 a handover with both grippers
    # empty), a tube 0.25-0.38 on our bench and 0.41 in the organizers' recordings
    empty_below: float = 0.15
    fast: float = 0.004  # m per step: the approach
    slow: float = 0.0015  # m per step: settled in front of the socket
    every: int = 10  # steps between checks


class SwitchState:
    """The rule over one episode: ``check`` at each check step (the state of that step); latches once it fires."""

    def __init__(self, rule: SwitchRule | None = None) -> None:
        self.rule = rule or SwitchRule()
        self.reset()

    def reset(self) -> None:
        self.prev, self.prev_step, self.moved, self.fired_at, self.speed = None, None, False, None, None
        self.hand_speed = None
        self.held_before = False  # both held at the previous check too (a gripper on its way shut passes the band)

    @property
    def fired(self) -> bool:
        return self.fired_at is not None

    def check(self, state: np.ndarray, step: int) -> bool:
        if self.fired:
            return True
        state = np.asarray(state, np.float64)
        tcp = tcp_positions(state)[0]
        prev, prev_step = self.prev, self.prev_step
        self.prev, self.prev_step = tcp, step
        if prev is None or step <= prev_step:
            return False
        dt = step - prev_step
        self.hand_speed = float(np.linalg.norm(tcp - prev, axis=1).max() / dt)
        self.speed = float(np.linalg.norm((tcp[0] - tcp[1]) - (prev[0] - prev[1])) / dt)  # towards / away from each other
        r = self.rule
        grips = (state[LEFT_GRIPPER], state[RIGHT_GRIPPER])
        if min(grips) < r.empty_below:  # a gripper closed on nothing: no tube to dock, the approach so far is void
            self.moved = self.held_before = False
            return False
        if max(grips) >= r.closed_below:
            self.held_before = False
            return False  # a loose grip for a moment does not forget the approach
        if not self.held_before:  # held at two checks in a row
            self.held_before = True
            return False
        if self.speed > r.fast:
            self.moved = True
        elif self.moved and self.speed < r.slow and self.hand_speed < r.slow:
            self.fired_at = step
        return self.fired


def replay(states: np.ndarray, rule: SwitchRule | None = None, upto: int | None = None) -> int | None:
    """The step the rule fires at on recorded (T, 14) ``states`` (checked every rule.every steps up to ``upto``)."""
    sw = SwitchState(rule)
    end = len(states) - 1 if upto is None else min(upto, len(states) - 1)
    for t in range(0, end + 1, sw.rule.every):
        if sw.check(states[t], t):
            return t
    return None
