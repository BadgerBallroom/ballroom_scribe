"""Base class for groups of arguments."""

import argparse
from abc import ABC, abstractmethod
from typing import TypedDict


class GroupArgs(TypedDict):
    pass


class Group[Resolved: GroupArgs](ABC):
    @abstractmethod
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Adds one or more arguments to the parser."""

    @abstractmethod
    def resolve(self, args: argparse.Namespace) -> Resolved:
        """Converts the arguments from the given `Namespace` into a dictionary of values."""
