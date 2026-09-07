import argparse
from typing import override

from .group_abc import Group, GroupArgs
from .json_loads_type import json_loads_dict


class SpeechTranscriptionArgs(GroupArgs):
    whisper_init_kwargs: dict
    whisper_beam_size: int


class SpeechTranscriptionGroup(Group[SpeechTranscriptionArgs]):
    """Group of arguments related to speech transcription."""

    @override
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        group = parser.add_argument_group("Speech Transcription", "Transcribing spoken audio into text")
        group.add_argument(
            "--whisper-init-kwargs",
            type=json_loads_dict,
            default={
                "model_size_or_path": "large-v3",
                "device": "cuda",
                "compute_type": "float16",
            },
            help="A JSON dictionary of keyword arguments to pass to WhisperModel.__init__ (default: %(default)s)",
        )
        group.add_argument(
            "--whisper-beam-size",
            type=int,
            default=5,
            help="The beam size to use for decoding by Faster Whisper (default: %(default)s)",
        )

    @override
    def resolve(self, args: argparse.Namespace) -> SpeechTranscriptionArgs:
        return SpeechTranscriptionArgs(
            whisper_init_kwargs=args.whisper_init_kwargs,
            whisper_beam_size=args.whisper_beam_size,
        )
