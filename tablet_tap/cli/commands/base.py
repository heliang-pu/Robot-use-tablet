"""The sub-command interface and the registry the parser is built from."""

from __future__ import annotations

import abc
import argparse
from typing import ClassVar

from tablet_tap.app import AppContext
from tablet_tap.app import ArmUse


class Command(abc.ABC):
    """One sub-command of ``bin/py -m tablet_tap``."""

    name: ClassVar[str]
    help: ClassVar[str]
    arm: ClassVar[ArmUse] = ArmUse.MOVES

    def configure(self, parser: argparse.ArgumentParser) -> None:  # noqa: B027 - most commands have no options
        """Add this command's own options."""

    def arm_use(self, args: argparse.Namespace) -> ArmUse:
        """What this invocation does with the arm (an option may make a moving command read-only)."""
        return self.arm

    @abc.abstractmethod
    def run(self, ctx: AppContext, args: argparse.Namespace) -> None: ...


_REGISTRY: dict[str, Command] = {}


def register(cls: type[Command]) -> type[Command]:
    """Class decorator: make the command available under its ``name``."""
    if cls.name in _REGISTRY:
        raise ValueError(f"two commands are called {cls.name!r}")
    _REGISTRY[cls.name] = cls()
    return cls


def commands() -> list[Command]:
    """The registered commands, in the order their modules were imported."""
    return list(_REGISTRY.values())
