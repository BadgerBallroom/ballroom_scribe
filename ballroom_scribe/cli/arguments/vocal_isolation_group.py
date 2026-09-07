import argparse
from typing import override

from .group_abc import Group, GroupArgs


class VocalIsolationArgs(GroupArgs):
    demucs_name: str
    demucs_device: str


class VocalIsolationGroup(Group[VocalIsolationArgs]):
    """Group of arguments related to vocal isolation."""

    @override
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        group = parser.add_argument_group("Vocal Isolation", "Isolating human speech from other sounds")
        group.add_argument(
            "--demucs-name",
            default="htdemucs",
            help="The value of the --name flag for Demucs (default: %(default)s)",
        )
        group.add_argument(
            "--demucs-device",
            default="cuda",
            help="The value of the --device flag for Demucs (default: %(default)s)",
        )

    @override
    def resolve(self, args: argparse.Namespace) -> VocalIsolationArgs:
        return VocalIsolationArgs(
            demucs_name=args.demucs_name,
            demucs_device=args.demucs_device,
        )
