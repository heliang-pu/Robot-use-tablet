"""tablet_tap: a PiPER arm with a capacitive pen operates a tablet from a spoken instruction."""

import pathlib
import sys

# The PiPER driver is RoboSynChallenge's, copied unchanged into third_party/robosyn (see SOURCE.md there). Its files
# import each other as the top-level package ``real_robot``, so their folder goes on the path.
VENDORED = pathlib.Path(__file__).resolve().parent.parent / "third_party" / "robosyn"
if str(VENDORED) not in sys.path:
    sys.path.insert(0, str(VENDORED))
