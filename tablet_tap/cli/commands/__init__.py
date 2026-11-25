"""The sub-commands. Importing this package registers them, in the order of the help text."""

from tablet_tap.cli.commands import status  # noqa: F401, I001 - the order is the order of the help
from tablet_tap.cli.commands import recover  # noqa: F401
from tablet_tap.cli.commands import limits  # noqa: F401
from tablet_tap.cli.commands import tablet  # noqa: F401
from tablet_tap.cli.commands import teach  # noqa: F401
from tablet_tap.cli.commands import view  # noqa: F401
from tablet_tap.cli.commands import board  # noqa: F401
from tablet_tap.cli.commands import place  # noqa: F401
from tablet_tap.cli.commands import probe  # noqa: F401
from tablet_tap.cli.commands import taptest  # noqa: F401
from tablet_tap.cli.commands import moves  # noqa: F401
from tablet_tap.cli.commands import shot  # noqa: F401
from tablet_tap.cli.commands import run  # noqa: F401
from tablet_tap.cli.commands import wrist  # noqa: F401
from tablet_tap.cli.commands.base import Command
from tablet_tap.cli.commands.base import commands

__all__ = ["Command", "commands"]
