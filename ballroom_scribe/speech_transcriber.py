"""Speech-to-text transcription service using Faster Whisper."""

import functools
import inspect
import os
import pathlib
import sys
from collections.abc import Generator, Iterable
from typing import Any

if sys.platform == "win32":
    import importlib.util

    def _fix_windows_dll_search_path():
        """Add NVIDIA DLL directories to the search path on Windows.

        On Windows, Python 3.8+ restricts DLL search paths. This registers the NVIDIA pip packages (cublas, cudnn) so
        that Faster Whisper can find the required DLLs.

        Both ``os.add_dll_directory`` and ``PATH`` are needed: the former covers ``LoadLibraryExW`` with
        ``LOAD_LIBRARY_SEARCH_DEFAULT_DIRS`` (used by PyTorch), while the latter covers plain ``LoadLibraryW`` (used by
        CTranslate2).
        """
        spec = importlib.util.find_spec("nvidia")
        if not spec or not spec.submodule_search_locations:
            return

        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        for search_location in spec.submodule_search_locations:
            search_location = pathlib.Path(search_location)
            for package in ("cublas", "cudnn"):
                bin_dir = search_location / package / "bin"
                if not bin_dir.is_dir():
                    continue

                bin_dir_str = str(bin_dir)
                os.add_dll_directory(bin_dir_str)
                if bin_dir_str not in path_parts:
                    path_parts.insert(0, bin_dir_str)
        os.environ["PATH"] = os.pathsep.join(path_parts)

    _fix_windows_dll_search_path()

import dataclass_csv
import faster_whisper
import faster_whisper.transcribe

import ballroom_scribe.models
import ballroom_scribe.utils

_CTRANSLATE2_EXTRA_KWARGS = frozenset({"max_queued_batches", "flash_attention", "tensor_parallel"})


def _flatten_segments_to_words(
    segments: Iterable[faster_whisper.transcribe.Segment],
) -> Generator[ballroom_scribe.models.Word]:
    """Flattens Faster Whisper segments into a generator of `Word` objects.

    Args:
        segments: The segments that Faster Whisper transcribed with word-level timestamps.

    Yields:
        The words in order.
    """
    for segment in segments:
        if not segment.words:
            continue

        for word in segment.words:
            yield ballroom_scribe.models.Word(
                word=word.word.strip(),
                start=word.start,
                end=word.end,
                probability=word.probability,
            )


def _save_words_to_file(words: Iterable[ballroom_scribe.models.Word], output_path: pathlib.Path) -> None:
    """Saves the transcribed words to a CSV file."""
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = dataclass_csv.DataclassWriter(f, words, ballroom_scribe.models.Word)
        writer.write()


def _read_words_from_file(input_path: pathlib.Path) -> Iterable[ballroom_scribe.models.Word]:
    """Reads transcribed words from a CSV file."""
    with open(input_path, "r", encoding="utf-8") as f:
        reader = dataclass_csv.DataclassReader(f, ballroom_scribe.models.Word)
        yield from reader


class SpeechTranscriber:
    """Transcribes audio files into word-level timestamps using Faster Whisper."""

    def __init__(
        self,
        whisper_init_kwargs: dict[str, Any],
        whisper_beam_size: int,
    ):
        """Initializes the instance.

        Args:
            whisper_init_kwargs: Keyword arguments to pass to the Faster Whisper model initializer.
            whisper_beam_size: The beam size to use for the Faster Whisper model.
        """
        # Validate the arguments to `WhisperModel.__init__` now because it is lazily called.
        whisper_model_signature = inspect.signature(faster_whisper.WhisperModel)
        whisper_model_signature.bind(**whisper_init_kwargs)
        valid_keys = (whisper_model_signature.parameters.keys() - {"model_kwargs"}) | _CTRANSLATE2_EXTRA_KWARGS
        if unexpected := whisper_init_kwargs.keys() - valid_keys:
            raise TypeError(f"WhisperModel will get unexpected keyword argument(s): {', '.join(sorted(unexpected))}")

        self._whisper_beam_size = whisper_beam_size
        self._whisper_init_kwargs = whisper_init_kwargs

    @functools.cached_property
    def _whisper_model(self) -> faster_whisper.WhisperModel:
        """Initializes the Faster Whisper model."""
        return faster_whisper.WhisperModel(**self._whisper_init_kwargs)

    def transcribe(
        self, input_audio: pathlib.Path, output_text: pathlib.Path
    ) -> Generator[ballroom_scribe.models.Word]:
        """Transcribes the speech in an audio file into word-level timestamps.

        Args:
            input_audio: The audio file to transcribe.
            output_text: The path to the CSV file to save the words to.

        Yields:
            The transcribed words in order.
        """
        if output_text.exists():
            print("Transcription already exists. Not transcribing again.")
        else:
            self._call_whisper(input_audio, output_text)

        # Read the words back from the file.
        yield from _read_words_from_file(output_text)

    def _call_whisper(self, input_audio: pathlib.Path, output_text: pathlib.Path) -> None:
        """Transcribes audio file with Faster Whisper and saves the words to a CSV file."""
        segments, _ = self._whisper_model.transcribe(
            str(input_audio),
            beam_size=self._whisper_beam_size,
            word_timestamps=True,
            vad_filter=True,  # Use the Voice Activity Detection filter to remove non-speech segments.
            condition_on_previous_text=False,  # Don't loop on past mistakes.
        )

        # Buffer the words directly to a file to avoid using more RAM than necessary for long videos.
        with ballroom_scribe.utils.atomic_path(output_text) as partial_text:
            _save_words_to_file(_flatten_segments_to_words(segments), partial_text)
