import argparse
import os
from collections.abc import Mapping
from typing import override

import ballroom_scribe.utils

from .group_abc import Group, GroupArgs
from .json_loads_type import json_loads_dict

_WHISPER_DEVICES = ("auto", "cuda", "cpu")
# Keyword arguments of `WhisperModel.__init__` that have their own flags and so must not appear in
# `--whisper-init-kwargs`.
_DEDICATED_INIT_KWARGS = {
    "model_size_or_path": "--whisper-model",
    "device": "--whisper-device",
    "compute_type": "--whisper-compute-type",
    "cpu_threads": "--whisper-cpu-threads",
}


def _resolve_for_device(value: str | Mapping[str, str], device: str) -> str:
    """Returns `value` itself, or its entry for `device` when it maps device names to values, as the defaults do."""
    return value[device] if isinstance(value, Mapping) else value


class SpeechTranscriptionArgs(GroupArgs):
    whisper_device: str
    whisper_model: str
    whisper_compute_type: str
    whisper_cpu_threads: int
    whisper_beam_size: int
    whisper_init_kwargs: dict


class SpeechTranscriptionGroup(Group[SpeechTranscriptionArgs]):
    """Group of arguments related to speech transcription.

    `resolve` turns `--whisper-device auto` into the concrete device (CUDA when it is available, the CPU otherwise) and
    picks that device's entry from the per-device defaults of `--whisper-model` and `--whisper-compute-type`, so the
    resolved arguments hold plain strings.
    """

    def __init__(self) -> None:
        self._device_action: argparse.Action | None = None
        self._init_kwargs_action: argparse.Action | None = None

    @override
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        group = parser.add_argument_group("Speech Transcription", "Transcribing spoken audio into text")
        self._device_action = group.add_argument(
            "--whisper-device",
            choices=_WHISPER_DEVICES,
            default="auto",
            help=(
                "The device on which to run Faster Whisper; 'auto' uses CUDA when it is available and the CPU"
                " otherwise (default: %(default)s)"
            ),
        )
        # argparse applies `type` only to string defaults, so these mapping defaults are used as-is while values given
        # on the command line stay plain strings.
        group.add_argument(
            "--whisper-model",
            default={"cuda": "large-v3", "cpu": "turbo"},
            help=(
                "The Faster Whisper model to load: a size such as large-v3 or turbo, a HuggingFace repository, or a"
                " local directory (default: per device, %(default)s)"
            ),
        )
        group.add_argument(
            "--whisper-compute-type",
            default={"cuda": "float16", "cpu": "int8"},
            help=(
                "The CTranslate2 compute type, such as float16, int8_float16, or int8"
                " (default: per device, %(default)s)"
            ),
        )
        group.add_argument(
            "--whisper-cpu-threads",
            type=int,
            # `process_cpu_count` honors the CPU affinity of this process and is `None` when the count is unknown.
            default=os.process_cpu_count() or 1,
            help=(
                "The number of threads that Faster Whisper uses when it runs on the CPU; 0 falls back to Faster"
                " Whisper's own default of 4 (default: %(default)s, every logical processor available to this process)"
            ),
        )
        group.add_argument(
            "--whisper-beam-size",
            type=int,
            default=5,
            help="The beam size to use for decoding by Faster Whisper (default: %(default)s)",
        )
        self._init_kwargs_action = group.add_argument(
            "--whisper-init-kwargs",
            type=json_loads_dict,
            default={},
            help=(
                "A JSON dictionary of additional keyword arguments to pass unchanged to WhisperModel.__init__, such"
                " as device_index or local_files_only (default: %(default)s)"
            ),
        )

    @override
    def resolve(self, args: argparse.Namespace) -> SpeechTranscriptionArgs:
        for key, flag in _DEDICATED_INIT_KWARGS.items():
            if key in args.whisper_init_kwargs:
                raise argparse.ArgumentError(self._init_kwargs_action, f"Use {flag} instead of the {key!r} key")

        try:
            device = ballroom_scribe.utils.resolve_torch_device(args.whisper_device)
        except ValueError as err:
            raise argparse.ArgumentError(self._device_action, str(err)) from err

        return SpeechTranscriptionArgs(
            whisper_device=device,
            whisper_model=_resolve_for_device(args.whisper_model, device),
            whisper_compute_type=_resolve_for_device(args.whisper_compute_type, device),
            whisper_cpu_threads=args.whisper_cpu_threads,
            whisper_beam_size=args.whisper_beam_size,
            whisper_init_kwargs=args.whisper_init_kwargs,
        )
