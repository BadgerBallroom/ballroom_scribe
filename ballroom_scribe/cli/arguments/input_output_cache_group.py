import argparse
import pathlib
from typing import override

from .group_abc import Group, GroupArgs


class InputCacheArgs(GroupArgs):
    input_video: pathlib.Path
    cache_dir: pathlib.Path


class InputOutputCacheArgs(InputCacheArgs):
    output_csv: pathlib.Path


class _InputCacheGroupBase[Resolved: InputCacheArgs](Group[Resolved]):
    """Shared base class for input video and cache directory arguments."""

    @override
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "input_video",
            type=pathlib.Path,
            help="Path to input video file",
        )
        parser.add_argument(
            "-c",
            "--cache-dir",
            type=pathlib.Path,
            help=(
                "Parent directory for a directory for caching intermediate results (default: same parent directory as"
                " input video, so the cache directory is in the same directory as the input video)"
            ),
        )

    def _resolve_input_and_cache(self, args: argparse.Namespace) -> InputCacheArgs:
        return InputCacheArgs(
            input_video=args.input_video,
            cache_dir=args.cache_dir or args.input_video.parent,
        )


class InputCacheGroup(_InputCacheGroupBase[InputCacheArgs]):
    """Group of arguments related to input video and cache directory."""

    @override
    def resolve(self, args: argparse.Namespace) -> InputCacheArgs:
        return self._resolve_input_and_cache(args)


class InputOutputCacheGroup(_InputCacheGroupBase[InputOutputCacheArgs]):
    """Group of arguments related to input video, cache directory, and output CSV file."""

    @override
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        super().add_arguments(parser)
        parser.add_argument(
            "-o",
            "--output-csv",
            type=pathlib.Path,
            help="Path to output CSV file (default: same as input video with .csv extension)",
        )

    @override
    def resolve(self, args: argparse.Namespace) -> InputOutputCacheArgs:
        return InputOutputCacheArgs(
            **self._resolve_input_and_cache(args),
            output_csv=args.output_csv or args.input_video.with_suffix(".csv"),
        )
