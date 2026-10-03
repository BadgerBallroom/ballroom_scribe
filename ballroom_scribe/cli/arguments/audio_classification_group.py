import argparse
from typing import override

import ballroom_scribe
import ballroom_scribe.models

from .group_abc import Group, GroupArgs


class AudioClassificationArgs(GroupArgs):
    pann_device: str
    pann_chunk_minutes: float
    pann_chunk_overlap_seconds: float
    pann_thresholds: ballroom_scribe.AudioClassifier.ThresholdMapping
    pann_min_chunk_seconds: float


class AudioClassificationGroup(Group[AudioClassificationArgs]):
    def __init__(self) -> None:
        self._chunk_overlap_action: argparse.Action | None = None

    @override
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        group = parser.add_argument_group("Audio Classification", "Detecting speech and music in the audio")
        group.add_argument(
            "--pann-device",
            default="cuda",
            help="The device on which to run audio classification (default: %(default)s)",
        )
        group.add_argument(
            "--pann-chunk-minutes",
            type=float,
            default=10.0,
            help="How many minutes of audio to classify at a time (increases VRAM usage) (default: %(default)s)",
        )
        self._chunk_overlap_action = group.add_argument(
            "--pann-chunk-overlap-seconds",
            type=float,
            default=2.0,
            help=(
                "How many seconds to extend each audio chunk at both the start and end for better classification"
                " accuracy (increases VRAM usage) (minimum: double of --pann-min-chunk-seconds) (default: %(default)s)"
            ),
        )
        group.add_argument(
            "--pann-threshold-speech",
            type=float,
            default=0.18,
            help="Confidence threshold for speech detection (default: %(default)s)",
        )
        group.add_argument(
            "--pann-threshold-music",
            type=float,
            default=0.45,
            help="Confidence threshold for music detection (default: %(default)s)",
        )
        group.add_argument(
            "--pann-min-chunk-seconds",
            type=float,
            default=1.0,
            help="Minimum chunk duration in seconds to absorb micro-pauses (default: %(default)s)",
        )

    @override
    def resolve(self, args: argparse.Namespace) -> AudioClassificationArgs:
        min_overlap_seconds = 2 * args.pann_min_chunk_seconds
        if args.pann_chunk_overlap_seconds < min_overlap_seconds:
            raise argparse.ArgumentError(
                self._chunk_overlap_action,
                f"Must be at least double --pann-min-chunk-seconds ({min_overlap_seconds}), got {args.pann_chunk_overlap_seconds}",
            )

        return AudioClassificationArgs(
            pann_device=args.pann_device,
            pann_chunk_minutes=args.pann_chunk_minutes,
            pann_chunk_overlap_seconds=args.pann_chunk_overlap_seconds,
            pann_thresholds={
                ballroom_scribe.models.AudioChunk.Classification.SPEECH: args.pann_threshold_speech,
                ballroom_scribe.models.AudioChunk.Classification.MUSIC: args.pann_threshold_music,
            },
            pann_min_chunk_seconds=args.pann_min_chunk_seconds,
        )
