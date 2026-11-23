"""Every error this program raises on purpose, in one hierarchy.

The command line maps them to exit codes (cli/main.py); library code raises them and never calls ``sys.exit``.
"""

from __future__ import annotations


class TabletTapError(RuntimeError):
    """Base of the errors below."""


class SetupError(TabletTapError):
    """Something the operator has to do or give first: a missing option, a calibration step not yet run."""


class CalibrationError(TabletTapError):
    """A calibration could not be computed: the screen was not found in the camera frame, or the touches do not fit."""


class Unreachable(TabletTapError):  # noqa: N818 - a property of the target
    """The IK did not reach a pose (out of the workspace or at a joint limit)."""


class PathError(TabletTapError):
    """A path that must not be sent: it does not start at the arm, jumps, or passes too low over the screen."""


class Stopped(TabletTapError):  # noqa: N818 - the operator's stop, not an error of the code
    """The operator asked to stop; the arm holds its last target."""


class ArmConnectionError(TabletTapError):
    """The arm cannot be reached (CAN channel down, frames lost, arm off)."""


class ArmBusyError(ArmConnectionError):
    """Another program is driving the arms; two programs sending targets would fight."""


class UnsupportedAction(TabletTapError):  # noqa: N818 - upstream's name for it
    """The device cannot perform the requested action (e.g. typing with a stylus)."""
