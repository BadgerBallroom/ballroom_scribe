import argparse
from typing import override

from .group_abc import Group, GroupArgs


class MusicValleyDetectionArgs(GroupArgs):
    min_dance_seconds: float
    max_song_dip_seconds: float
    max_fade_speech_seconds: float
    max_dip_speech_seconds: float


class MusicValleyDetectionGroup(Group[MusicValleyDetectionArgs]):
    """Group of arguments related to music valley detection."""

    @override
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        group = parser.add_argument_group(
            "Music Valley Detection",
            "Detecting transitions between dances and extracting speech segments",
        )
        group.add_argument(
            "--min-dance-seconds",
            type=float,
            default=35.0,
            help=(
                "Minimum cumulative duration in seconds required for a music run to be classified as a competition"
                " dance (default: %(default)s)"
            ),
        )
        group.add_argument(
            "--max-song-dip-seconds",
            type=float,
            default=4.5,
            help=(
                "Maximum cumulative duration in seconds of a non-music gap that can be bridged into an active music"
                " run if music resumes (default: %(default)s)"
            ),
        )
        group.add_argument(
            "--max-fade-speech-seconds",
            type=float,
            default=12.0,
            help=(
                "Maximum cumulative duration in seconds of overlapping speech and music at the leading or trailing"
                " edge of a dance to peel off and include in the adjacent music valley (default: %(default)s)"
            ),
        )
        group.add_argument(
            "--max-dip-speech-seconds",
            type=float,
            default=2.0,
            help=(
                "Duration in seconds of speech within a non-music gap at or above which the gap is no longer"
                " considered a mid-song dip (default: %(default)s)"
            ),
        )

    @override
    def resolve(self, args: argparse.Namespace) -> MusicValleyDetectionArgs:
        return MusicValleyDetectionArgs(
            min_dance_seconds=args.min_dance_seconds,
            max_song_dip_seconds=args.max_song_dip_seconds,
            max_fade_speech_seconds=args.max_fade_speech_seconds,
            max_dip_speech_seconds=args.max_dip_speech_seconds,
        )
