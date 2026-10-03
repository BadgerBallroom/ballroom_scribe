"""Utilities to construct and parse command-line arguments."""

import argparse
from collections.abc import Iterable, Sequence
from typing import Any, overload

from .audio_classification_group import AudioClassificationArgs, AudioClassificationGroup
from .group_abc import Group, GroupArgs
from .input_output_cache_group import InputCacheArgs, InputCacheGroup, InputOutputCacheArgs, InputOutputCacheGroup
from .language_processing_group import LanguageProcessingArgs, LanguageProcessingGroup
from .music_valley_detection_group import MusicValleyDetectionArgs, MusicValleyDetectionGroup
from .speech_transcription_group import SpeechTranscriptionArgs, SpeechTranscriptionGroup

__all__ = [
    "AudioClassificationArgs",
    "AudioClassificationGroup",
    "InputCacheArgs",
    "InputCacheGroup",
    "InputOutputCacheArgs",
    "InputOutputCacheGroup",
    "LanguageProcessingArgs",
    "LanguageProcessingGroup",
    "MusicValleyDetectionArgs",
    "MusicValleyDetectionGroup",
    "SpeechTranscriptionArgs",
    "SpeechTranscriptionGroup",
    "parse_args",
]


@overload
def parse_args[T1: GroupArgs](
    groups: tuple[Group[T1]],
    argv: Iterable[str] | None = None,
    **kwargs: Any,
) -> tuple[T1]: ...


@overload
def parse_args[T1: GroupArgs, T2: GroupArgs](
    groups: tuple[Group[T1], Group[T2]],
    argv: Iterable[str] | None = None,
    **kwargs: Any,
) -> tuple[T1, T2]: ...


@overload
def parse_args[T1: GroupArgs, T2: GroupArgs, T3: GroupArgs](
    groups: tuple[Group[T1], Group[T2], Group[T3]],
    argv: Iterable[str] | None = None,
    **kwargs: Any,
) -> tuple[T1, T2, T3]: ...


@overload
def parse_args[T1: GroupArgs, T2: GroupArgs, T3: GroupArgs, T4: GroupArgs](
    groups: tuple[Group[T1], Group[T2], Group[T3], Group[T4]],
    argv: Iterable[str] | None = None,
    **kwargs: Any,
) -> tuple[T1, T2, T3, T4]: ...


@overload
def parse_args[T1: GroupArgs, T2: GroupArgs, T3: GroupArgs, T4: GroupArgs, T5: GroupArgs](
    groups: tuple[Group[T1], Group[T2], Group[T3], Group[T4], Group[T5]],
    argv: Iterable[str] | None = None,
    **kwargs: Any,
) -> tuple[T1, T2, T3, T4, T5]: ...


@overload
def parse_args[T1: GroupArgs, T2: GroupArgs, T3: GroupArgs, T4: GroupArgs, T5: GroupArgs, T6: GroupArgs](
    groups: tuple[Group[T1], Group[T2], Group[T3], Group[T4], Group[T5], Group[T6]],
    argv: Iterable[str] | None = None,
    **kwargs: Any,
) -> tuple[T1, T2, T3, T4, T5, T6]: ...


@overload
def parse_args(
    groups: Sequence[Group[Any]],
    argv: Iterable[str] | None = None,
    **kwargs: Any,
) -> Sequence[GroupArgs]: ...


def parse_args(
    groups: Sequence[Group[Any]],
    argv: Iterable[str] | None = None,
    **kwargs: Any,
) -> Sequence[GroupArgs]:
    """Parses arguments for CLIs in this repository."""
    parser = argparse.ArgumentParser(**kwargs)
    for group in groups:
        group.add_arguments(parser)
    parsed_args = parser.parse_args(argv)
    try:
        return tuple(group.resolve(parsed_args) for group in groups)
    except argparse.ArgumentError as err:
        if parser.exit_on_error:
            parser.error(str(err))
        raise
