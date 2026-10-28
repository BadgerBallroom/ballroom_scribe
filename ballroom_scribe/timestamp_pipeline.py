"""Orchestration pipeline for audio extraction, transcription, and event timestamping."""

import contextlib
import dataclasses
import itertools
import pathlib
from collections.abc import Generator, Iterable
from typing import Literal, TypedDict

import ballroom_scribe
import ballroom_scribe.cli.arguments
import ballroom_scribe.models
import ballroom_scribe.utils


@dataclasses.dataclass(frozen=True)
class _Video:
    """An input video paired with its own cache directory.

    Attributes:
        input_video: The path to the input video file.
        input_channel: The audio channel to extract from `input_video`.
        cache_dir: The path to the video's own cache directory. Inside it, there will be:
            - `Audio-Channel-*.wav`: the extracted audio from the input video
            - `Waveform-Channel-*.npz`: min/max waveform peaks cached by the GUI editor
            - `*.npy`: audio classification inferences
            - `*.csv`: the transcribed words of each speech chunk
            - `*.json`: the detected events from the transcript
            - `*.md`: LLM thinking output
            Except for the `*.md` files, each file is first written to a temporary `*.partial.*` file and renamed once
            complete, so an existing file is never partially written.
    """

    input_video: pathlib.Path
    input_channel: int
    cache_dir: pathlib.Path


def _resolve_video(input_video: pathlib.Path, input_channel: int, cache_dir: pathlib.Path) -> _Video:
    """Pairs an input video with its own cache directory inside the given parent cache directory.

    Args:
        input_video: The path to the input video file.
        input_channel: The audio channel to extract from the input video.
        cache_dir: The parent directory in which the video's cache directory lives.
    """
    return _Video(
        input_video=input_video,
        input_channel=input_channel,
        cache_dir=ballroom_scribe.utils.cache_dir_for_video(parent_cache_dir=cache_dir, input_video=input_video),
    )


def _require_distinct_cache_dirs(videos: Iterable[_Video]) -> None:
    """Raises a `ValueError` if two input videos resolve to the same cache directory.

    Cache files are named after sample positions and channels, not after the video, so two videos sharing a cache
    directory (the same file given twice, or two files with the same name sharing a parent cache directory) would
    silently reuse each other's extracted audio, inferences, and LLM responses. Directories are compared by their
    resolved absolute paths, so relative and absolute spellings of one directory, or symlinks to it, count as the
    same directory.
    """
    videos_by_cache_dir: dict[pathlib.Path, _Video] = {}
    for video in videos:
        cache_dir = video.cache_dir.resolve()
        if (other := videos_by_cache_dir.get(cache_dir)) is not None:
            raise ValueError(
                f"{video.input_video} and {other.input_video} would share the cache directory {video.cache_dir}."
                " Give one of them a different parent cache directory."
            )
        videos_by_cache_dir[cache_dir] = video


def _extract_audio(video: _Video) -> pathlib.Path:
    """Creates the video's cache directory and extracts its audio channel as a 32,000 Hz mono WAV file."""
    video.cache_dir.mkdir(parents=True, exist_ok=True)
    # `extract_audio` makes sure that the output audio is a 32,000 Hz mono WAV.
    # `AudioLoader` requires that the input audio is a 32,000 Hz mono WAV.
    return ballroom_scribe.utils.extract_audio(
        input_video=video.input_video,
        input_channel=video.input_channel,
        cache_dir=video.cache_dir,
    )


def _transcribe_speech(
    speech_transcriber: ballroom_scribe.SpeechTranscriber,
    video: _Video,
    music_valleys: Iterable[ballroom_scribe.models.MusicValley],
) -> list[ballroom_scribe.models.TranscriptChunk]:
    """Transcribes speech in each music valley of a video and prints the detected text.

    Args:
        speech_transcriber: The transcriber used to convert valley speech chunks into words.
        video: The video whose music valleys are being transcribed.
        music_valleys: The music valleys whose speech chunks should be transcribed.

    Returns:
        A list of transcript chunks corresponding to `music_valleys`, including empty chunks for silent valleys.
    """
    transcript_chunks: list[ballroom_scribe.models.TranscriptChunk] = []
    for music_valley in music_valleys:
        print(music_valley.format_time_range())
        transcript_chunk = speech_transcriber.transcribe(
            music_valley=music_valley,
            channel=video.input_channel,
            cache_dir=video.cache_dir,
        )
        # Keep music valleys in which nothing was said so that every music valley produces an event.
        # `EventDetector` will not ask the LLM about empty text.
        print(transcript_chunk.get_text() or "(no speech detected)")
        transcript_chunks.append(transcript_chunk)
    return transcript_chunks


def _detect_events(
    event_detector: ballroom_scribe.EventDetector,
    video: _Video,
    transcript_chunks: Iterable[ballroom_scribe.models.TranscriptChunk],
) -> Generator[tuple[pathlib.Path, ballroom_scribe.models.Event]]:
    """Detects the events in a video's transcript and pairs each with the video's resolved path for the CSV's `File` column.

    Yields:
        `(input_video, event)` pairs, lazily, so that the LLM is queried while the CSV is being written.
    """
    print(f"\nDetecting events in {video.input_video}...")
    resolved_input_video = video.input_video.resolve()
    for event in event_detector.detect(chunks_of_words=transcript_chunks, cache_dir=video.cache_dir):
        yield resolved_input_video, event


type _VRAMConsumersKey = Literal[
    "audio_classifier",
    "speech_transcriber",
]


class _VRAMConsumers(TypedDict, total=False):
    audio_classifier: ballroom_scribe.AudioClassifier
    speech_transcriber: ballroom_scribe.SpeechTranscriber


class TimestampPipeline:
    """Orchestrates the ballroom competition timestamp extraction pipeline."""

    def __init__(
        self,
        audio_classification_args: ballroom_scribe.cli.arguments.AudioClassificationArgs,
        music_valley_detection_args: ballroom_scribe.cli.arguments.MusicValleyDetectionArgs,
        speech_transcription_args: ballroom_scribe.cli.arguments.SpeechTranscriptionArgs,
        language_processing_args: ballroom_scribe.cli.arguments.LanguageProcessingArgs,
    ):
        """Initializes the instance.

        Args:
            audio_classification_args: Configuration for audio loading and classification.
            music_valley_detection_args: Configuration for music valley detection.
            speech_transcription_args: Configuration for speech transcription.
            language_processing_args: Configuration for language processing.
        """
        self._audio_classification_args = audio_classification_args
        self._speech_transcription_args = speech_transcription_args

        # These modules do not use the GPU in a way that Python can release the VRAM.
        self._audio_loader = ballroom_scribe.AudioLoader(
            chunk_minutes=audio_classification_args["pann_chunk_minutes"],
            chunk_overlap_seconds=audio_classification_args["pann_chunk_overlap_seconds"],
        )
        self._music_valley_detector = ballroom_scribe.MusicValleyDetector(**music_valley_detection_args)
        self._event_detector = ballroom_scribe.EventDetector(**language_processing_args)
        self._event_recorder = ballroom_scribe.EventRecorder()

    def run(
        self,
        inputs: Iterable[ballroom_scribe.cli.arguments.InputCacheArgs],
        output_csv: pathlib.Path,
    ) -> None:
        """Runs the complete timestamp pipeline end-to-end on one or more videos.

        Every input is checked before any slow work begins, so a mistake in the last video fails the run immediately.
        Then every video passes through each VRAM stage before the next stage begins, so each model is loaded into
        VRAM only once per run rather than once per video. The price is RAM: between the audio classification and
        speech transcription stages, the speech chunks of every video are held in memory at once.

        Args:
            inputs: For each video: its path, the audio channel to extract, and the parent directory of its cache
                directory.
            output_csv: The path to the output CSV file, which receives the events of every video.

        Raises:
            ValueError: If two inputs resolve to the same cache directory, or an audio channel index is out of range.
        """
        # All parts of the pipeline should lazy-load anything that uses a significant amount of VRAM.
        # All parts are initialized here so that if there is a bad configuration, the pipeline will fail fast.
        managed_vram_consumers: _VRAMConsumers = {
            "audio_classifier": ballroom_scribe.AudioClassifier(
                device=self._audio_classification_args["pann_device"],
                thresholds=self._audio_classification_args["pann_thresholds"],
                min_segment_seconds=self._audio_classification_args["pann_min_chunk_seconds"],
            ),
            "speech_transcriber": ballroom_scribe.SpeechTranscriber(**self._speech_transcription_args),
        }

        @contextlib.contextmanager
        def free_vram_afterward(key: _VRAMConsumersKey) -> Generator[None]:
            try:
                yield
            finally:
                del managed_vram_consumers[key]
                ballroom_scribe.utils.free_vram()

        print("Checking input videos...")
        videos = [_resolve_video(**video_input) for video_input in inputs]
        _require_distinct_cache_dirs(videos)
        for video in videos:
            ballroom_scribe.utils.validate_input_channel(
                input_video=video.input_video,
                input_channel=video.input_channel,
            )

        extracted_audio: dict[_Video, pathlib.Path] = {}
        for video in videos:
            print(f"\nExtracting audio from {video.input_video}...")
            extracted_audio[video] = _extract_audio(video)

        # Each VRAM-consuming stage runs in its own helper function and receives its consumer directly from
        # `managed_vram_consumers` so that paused generators and local variables do not pin the consumer in memory
        # after `free_vram_afterward` exits.
        music_valleys: dict[_Video, list[ballroom_scribe.models.MusicValley]] = {}
        with free_vram_afterward("audio_classifier"):
            for video in videos:
                print(f"\nReading and classifying audio segments of {video.input_video}...")
                music_valleys[video] = self._detect_music_valleys(
                    audio_classifier=managed_vram_consumers["audio_classifier"],
                    video=video,
                    extracted_audio=extracted_audio[video],
                )

        transcript_chunks: dict[_Video, list[ballroom_scribe.models.TranscriptChunk]] = {}
        with free_vram_afterward("speech_transcriber"):
            for video in videos:
                print(f"\nTranscribing speech in {video.input_video}...")
                transcript_chunks[video] = _transcribe_speech(
                    speech_transcriber=managed_vram_consumers["speech_transcriber"],
                    video=video,
                    music_valleys=music_valleys[video],
                )

        print("\nDetecting events and exporting to CSV...")
        events = itertools.chain.from_iterable(
            _detect_events(self._event_detector, video, transcript_chunks[video]) for video in videos
        )
        self._event_recorder.record(events=events, target=output_csv)
        # Ollama will unload the LLM from VRAM when the `keep_alive` time expires, so we don't need to do it explicitly.

        print(f"\nPipeline complete! Results saved to {output_csv}")

    def _detect_music_valleys(
        self,
        audio_classifier: ballroom_scribe.AudioClassifier,
        video: _Video,
        extracted_audio: pathlib.Path,
    ) -> list[ballroom_scribe.models.MusicValley]:
        """Loads and classifies audio chunks and detects all music valleys in a video.

        Consumes the entire detection stream into a list so that all PANN inference completes before `audio_classifier`
        is freed from VRAM, while retaining only detached speech-chunk copies so multi-minute audio buffers from
        `AudioLoader` are not pinned in memory.

        Args:
            audio_classifier: The classifier used to segment audio into speech, music, and other sounds.
            video: The video whose extracted audio is being classified.
            extracted_audio: The path to the video's extracted 32,000 Hz mono WAV file.

        Returns:
            The detected music valleys across the video's audio stream.
        """
        unclassified_audio_chunks = self._audio_loader.load(input_audio=extracted_audio)
        classified_audio_chunks = audio_classifier.classify(
            audio_chunks=unclassified_audio_chunks,
            channel=video.input_channel,
            cache_dir=video.cache_dir,
        )
        music_valleys = self._music_valley_detector.detect(
            classified_chunks=classified_audio_chunks,
        )

        # Accumulate and store all the music valleys in memory.
        # This essentially waits for all PANN inferencing to complete before proceeding.
        # This allows the VRAM that the PANN was using to be freed before Faster Whisper is loaded.
        # Each music valley holds only detached copies of its speech chunks, so this does not pin the
        # multi-minute audio buffers that `AudioLoader` produced.
        return list(music_valleys)
