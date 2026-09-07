import argparse
from typing import override

from .group_abc import Group, GroupArgs


class TranscriptChunkingArgs(GroupArgs):
    split_on_silence_threshold: float


class TranscriptChunkingGroup(Group[TranscriptChunkingArgs]):
    """Group of arguments related to transcript chunking."""

    @override
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--split-on-silence-threshold",
            type=float,
            default=10.0,
            help=(
                "The number of seconds that should pass in the audio without human vocals to trigger a new Ollama chat"
                " session (default: %(default)s)"
            ),
        )

    @override
    def resolve(self, args: argparse.Namespace) -> TranscriptChunkingArgs:
        return TranscriptChunkingArgs(split_on_silence_threshold=args.split_on_silence_threshold)
