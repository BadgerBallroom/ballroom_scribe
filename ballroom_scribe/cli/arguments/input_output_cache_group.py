"""Argument groups for input videos, their audio channels and cache directories, and CSV files."""

import argparse
import pathlib
from collections.abc import Sequence
from typing import Any, override

from .group_abc import Group, GroupArgs


class InputCacheArgs(GroupArgs):
    input_video: pathlib.Path
    input_channel: int
    cache_dir: pathlib.Path


class InputOutputCacheArgs(GroupArgs):
    inputs: list[InputCacheArgs]
    output_csv: pathlib.Path


class InputCsvCacheArgs(GroupArgs):
    inputs: list[InputCacheArgs]
    csv: pathlib.Path | None


class _CacheDirAction(argparse.Action):
    """Attaches each `-c/--cache-dir` to the most recent `-i/--input`.

    `argparse` invokes actions in command-line order, so when this action runs, `namespace.input` holds exactly the
    `-i/--input` entries that preceded this `-c/--cache-dir`. The directory is stored under that entry's index in a
    `dict[int, pathlib.Path]` so that videos without their own `-c/--cache-dir` can fall back to their default.
    """

    @override
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        inputs = getattr(namespace, "input", None)
        if not inputs:
            raise argparse.ArgumentError(self, "Must come after the -i/--input to which it applies")
        index = len(inputs) - 1
        cache_dirs = getattr(namespace, self.dest, None) or {}
        if index in cache_dirs:
            raise argparse.ArgumentError(self, f"Given more than once for {inputs[index][0]}")
        cache_dirs[index] = values
        setattr(namespace, self.dest, cache_dirs)


def _parse_channel(input_action: argparse.Action | None, channel: str) -> int:
    """Converts the `CHANNEL` half of `-i/--input VIDEO CHANNEL`, failing the same way `type=int` would."""
    try:
        return int(channel)
    except ValueError:
        raise argparse.ArgumentError(input_action, f"invalid int value: {channel!r}") from None


class _InputCacheGroupBase[Resolved: GroupArgs](Group[Resolved]):
    """Shared base class for the `-i/--input VIDEO CHANNEL` and `-c/--cache-dir` arguments."""

    def __init__(self, max_inputs: int | None = None, *, required: bool = True) -> None:
        """Initializes the instance.

        Args:
            max_inputs: The maximum number of `-i/--input` videos to accept, or `None` for no limit.
            required: Whether at least one `-i/--input` must be provided.

        Raises:
            ValueError: If `max_inputs` is less than 1.
        """
        if max_inputs is not None and max_inputs < 1:
            raise ValueError(f"max_inputs must be at least 1, got {max_inputs}")
        self._max_inputs = max_inputs
        self._required = required
        self._input_action: argparse.Action | None = None

    @override
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        input_help = (
            "Path to an input video file followed by the zero-based index of the audio channel to extract from it"
        )
        if self._max_inputs is None:
            input_help += " (repeat for each video)"
        elif self._max_inputs > 1:
            input_help += f" (repeat for up to {self._max_inputs} videos)"
        self._input_action = parser.add_argument(
            "-i",
            "--input",
            nargs=2,
            metavar=("VIDEO", "CHANNEL"),
            action="append",
            required=self._required,
            help=input_help,
        )
        parser.add_argument(
            "-c",
            "--cache-dir",
            action=_CacheDirAction,
            type=pathlib.Path,
            help=(
                "Parent directory for a directory for caching intermediate results of the preceding -i/--input"
                " (default: same parent directory as that input video, so the cache directory is in the same directory"
                " as the input video)"
            ),
        )

    def _resolve_inputs(self, args: argparse.Namespace) -> list[InputCacheArgs]:
        raw_inputs = args.input or []
        if self._max_inputs is not None and len(raw_inputs) > self._max_inputs:
            raise argparse.ArgumentError(
                self._input_action,
                f"Expected at most {self._max_inputs} video(s), got {len(raw_inputs)}",
            )
        cache_dirs: dict[int, pathlib.Path] = args.cache_dir or {}
        inputs: list[InputCacheArgs] = []
        for index, (video, channel) in enumerate(raw_inputs):
            input_video = pathlib.Path(video)
            inputs.append(
                InputCacheArgs(
                    input_video=input_video,
                    input_channel=_parse_channel(self._input_action, channel),
                    cache_dir=cache_dirs.get(index, input_video.parent),
                )
            )
        return inputs


class InputCacheGroup(_InputCacheGroupBase[InputCacheArgs]):
    """Group of arguments for a single input video, its audio channel, and its cache directory."""

    def __init__(self) -> None:
        super().__init__(max_inputs=1)

    @override
    def resolve(self, args: argparse.Namespace) -> InputCacheArgs:
        # `-i/--input` is required and `max_inputs` is 1, so exactly one input is guaranteed here.
        (video_input,) = self._resolve_inputs(args)
        return video_input


class InputOutputCacheGroup(_InputCacheGroupBase[InputOutputCacheArgs]):
    """Group of arguments for one or more input videos, their audio channels and cache directories, and the output CSV."""

    def __init__(self, max_inputs: int | None = None) -> None:
        """Initializes the instance.

        Args:
            max_inputs: The maximum number of `-i/--input` videos to accept, or `None` for no limit.
        """
        super().__init__(max_inputs=max_inputs)
        self._output_csv_action: argparse.Action | None = None

    @override
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        super().add_arguments(parser)
        self._output_csv_action = parser.add_argument(
            "-o",
            "--output-csv",
            type=pathlib.Path,
            help=(
                "Path to output CSV file (default: same as input video with .csv extension; required when more than"
                " one -i/--input is given)"
            ),
        )

    @override
    def resolve(self, args: argparse.Namespace) -> InputOutputCacheArgs:
        inputs = self._resolve_inputs(args)
        if args.output_csv is None and len(inputs) > 1:
            raise argparse.ArgumentError(self._output_csv_action, "Required when more than one -i/--input is given")
        return InputOutputCacheArgs(
            inputs=inputs,
            output_csv=args.output_csv or inputs[0]["input_video"].with_suffix(".csv"),
        )


class InputCsvCacheGroup(_InputCacheGroupBase[InputCsvCacheArgs]):
    """Group of arguments for optional input videos, their cache directories, and an optional events CSV file."""

    def __init__(self, max_inputs: int | None = None) -> None:
        """Initializes the instance.

        Args:
            max_inputs: The maximum number of `-i/--input` videos to accept, or `None` for no limit.
        """
        super().__init__(max_inputs=max_inputs, required=False)

    @override
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "csv",
            nargs="?",
            type=pathlib.Path,
            metavar="EVENTS_CSV",
            help="Optional path to an existing events CSV file to open",
        )
        super().add_arguments(parser)

    @override
    def resolve(self, args: argparse.Namespace) -> InputCsvCacheArgs:
        return InputCsvCacheArgs(
            inputs=self._resolve_inputs(args),
            csv=args.csv,
        )
