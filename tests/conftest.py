"""Tests never touch the bench: the lock makes the command line refuse the arm, its CAN channel and real cameras."""

import os

os.environ["TABLET_TAP_NO_HARDWARE"] = "1"
