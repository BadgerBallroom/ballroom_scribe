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
    offset_seconds: float,
) -> Generator[ballroom_scribe.models.Word]:
    """Flattens Faster Whisper segments into a generator of `Word` objects.

    Args:
        segments: The segments that Faster Whisper transcribed with word-level timestamps.
        offset_seconds: The start time of the transcribed audio, in seconds since the beginning of the original audio
            file. Faster-Whisper reports timestamps relative to the audio it was given, so this is added to every
            word's timestamps to make them relative to the original audio file instead.

    Yields:
        The words in order, with timestamps relative to the beginning of the original audio file.
    """
    for segment in segments:
        if not segment.words:
            continue

        for word in segment.words:
            yield ballroom_scribe.models.Word(
                word=word.word.strip(),
                start=offset_seconds + word.start,
                end=offset_seconds + word.end,
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
    """Transcribes speech chunks into word-level timestamps using Faster Whisper."""

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

    @property
    def _whisper_sample_rate(self) -> int:
        """The sample rate, in Hz, at which the Faster-Whisper model expects its input audio.

        `WhisperModel.transcribe` assumes that a NumPy array is already at this sample rate. It is read from the model's
        `preprocessor_config.json`, so accessing this property loads the model.
        """
        return self._whisper_model.feature_extractor.sampling_rate

    def transcribe(
        self,
        music_valley: ballroom_scribe.models.MusicValley,
        channel: int,
        cache_dir: pathlib.Path,
    ) -> ballroom_scribe.models.TranscriptChunk:
        """Transcribes the speech in a music valley into word-level timestamps.

        Each speech chunk in the music valley is transcribed separately, and its words are cached in a CSV file in
        `cache_dir` so that re-running the pipeline does not transcribe the same audio twice.

        Args:
            music_valley: The music valley whose `speech_chunks` will be transcribed.
            channel: The audio channel from which the speech chunks were loaded.
            cache_dir: The directory in which transcriptions will be cached.

        Returns:
            A transcript chunk spanning the whole music valley, containing the words of all of its speech chunks in
            chronological order. Word timestamps are in seconds since the beginning of the original audio file.
        """
        words = tuple(
            word
            for audio_chunk in music_valley.speech_chunks
            for word in self._load_or_transcribe_chunk(audio_chunk=audio_chunk, channel=channel, cache_dir=cache_dir)
        )
        return ballroom_scribe.models.TranscriptChunk(
            start_samples=music_valley.start_samples,
            sample_rate=music_valley.sample_rate,
            duration_samples=music_valley.duration_samples,
            words=words,
        )

    def _load_or_transcribe_chunk(
        self,
        audio_chunk: ballroom_scribe.models.AudioChunk,
        channel: int,
        cache_dir: pathlib.Path,
    ) -> Iterable[ballroom_scribe.models.Word]:
        """Loads the words of one speech chunk from the cache, or transcribes the chunk if it is not cached.

        Args:
            audio_chunk: The speech chunk to transcribe.
            channel: The audio channel from which `audio_chunk` was loaded.
            cache_dir: The directory in which transcriptions will be cached.

        Returns:
            The transcribed words.
        """
        cache_file = cache_dir / f"{audio_chunk.get_filename_stem()}-{channel}.csv"
        if cache_file.exists():
            return _read_words_from_file(cache_file)

        words = list(self._call_whisper(audio_chunk))
        with ballroom_scribe.utils.atomic_path(cache_file) as partial_cache_file:
            _save_words_to_file(words, partial_cache_file)
        return words

    def _call_whisper(
        self,
        audio_chunk: ballroom_scribe.models.AudioChunk,
    ) -> Generator[ballroom_scribe.models.Word]:
        """Transcribes one speech chunk with Faster Whisper."""
        # `WhisperModel.transcribe` assumes that a NumPy array is already at the model's sample rate.
        resampled_chunk = audio_chunk.resample(new_sample_rate=self._whisper_sample_rate)
        segments, _ = self._whisper_model.transcribe(
            resampled_chunk.data,
            beam_size=self._whisper_beam_size,
            word_timestamps=True,
            condition_on_previous_text=False,  # Don't loop on past mistakes.
        )
        return _flatten_segments_to_words(segments, offset_seconds=audio_chunk.start_seconds)
