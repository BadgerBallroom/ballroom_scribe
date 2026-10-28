"""Group of arguments controlling diagnostic log output."""

import argparse
from typing import override

from .group_abc import Group, GroupArgs


class LoggingArgs(GroupArgs):
    verbose: bool


class LoggingGroup(Group[LoggingArgs]):
    """Group of arguments related to logging."""

    @override
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        group = parser.add_argument_group("Logging", "Controlling diagnostic output")
        group.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="Enable verbose debug logging, including diagnostics from the underlying libraries",
        )

    @override
    def resolve(self, args: argparse.Namespace) -> LoggingArgs:
        return LoggingArgs(verbose=args.verbose)
