"""Orchestration pipeline for audio extraction, transcription, and event timestamping."""

import contextlib
import pathlib
from collections.abc import Generator, Iterable
from typing import Literal, TypedDict

import ballroom_scribe
import ballroom_scribe.cli.arguments
import ballroom_scribe.models
import ballroom_scribe.utils


def _transcribe_speech(
    speech_transcriber: ballroom_scribe.SpeechTranscriber,
    music_valleys: Iterable[ballroom_scribe.models.MusicValley],
    input_channel: int,
    cache_dir: pathlib.Path,
) -> list[ballroom_scribe.models.TranscriptChunk]:
    """Transcribes speech in each music valley and prints the detected text.

    Args:
        speech_transcriber: The transcriber used to convert valley speech chunks into words.
        music_valleys: The music valleys whose speech chunks should be transcribed.
        input_channel: The audio channel from which the audio was extracted.
        cache_dir: The path to the video's cache directory.

    Returns:
        A list of transcript chunks corresponding to `music_valleys`, including empty chunks for silent valleys.
    """
    transcript_chunks: list[ballroom_scribe.models.TranscriptChunk] = []
    for music_valley in music_valleys:
        print(music_valley.format_time_range())
        transcript_chunk = speech_transcriber.transcribe(
            music_valley=music_valley,
            channel=input_channel,
            cache_dir=cache_dir,
        )
        # Keep music valleys in which nothing was said so that every music valley produces an event.
        # `EventDetector` will not ask the LLM about empty text.
        print(transcript_chunk.get_text() or "(no speech detected)")
        transcript_chunks.append(transcript_chunk)
    return transcript_chunks


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
        input_video: pathlib.Path,
        input_channel: int,
        output_csv: pathlib.Path,
        cache_dir: pathlib.Path,
    ) -> None:
        """Runs the complete timestamp pipeline end-to-end.

        Args:
            input_video: The path to the input video file.
            input_channel: The audio channel to extract from the input video.
            output_csv: The path to the output CSV file.
            cache_dir: The path to the cache directory for intermediate results.
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

        # Inside the cache directory, there will be:
        # - Audio-Channel-*.wav: the extracted audio from the input video
        # - *.npy: audio classification inferences
        # - *.csv: the transcribed words of each speech chunk
        # - *.json: the detected events from the transcript
        # - *.md: LLM thinking output
        # Except for the *.md files, each file is first written to a temporary `*.partial.*` file and renamed once
        # complete, so an existing file is never partially written.
        cache_dir = ballroom_scribe.utils.cache_dir_for_video(
            parent_cache_dir=cache_dir,
            input_video=input_video,
        )
        cache_dir.mkdir(parents=True, exist_ok=True)

        print("Extracting audio...")
        # `extract_audio` writes a 32,000 Hz mono WAV into `cache_dir`, which is exactly what `AudioLoader` requires.
        extracted_audio = ballroom_scribe.utils.extract_audio(
            input_video=input_video,
            input_channel=input_channel,
            cache_dir=cache_dir,
        )

        # Each VRAM-consuming stage runs in its own helper function and receives its consumer directly from
        # `managed_vram_consumers` so that paused generators and local variables do not pin the consumer in memory
        # after `free_vram_afterward` exits.
        print("\nReading and classifying audio segments...")
        with free_vram_afterward("audio_classifier"):
            music_valleys = self._detect_music_valleys(
                audio_classifier=managed_vram_consumers["audio_classifier"],
                extracted_audio=extracted_audio,
                input_channel=input_channel,
                cache_dir=cache_dir,
            )

        print("\nTranscribing speech...")
        with free_vram_afterward("speech_transcriber"):
            transcript_chunks = _transcribe_speech(
                speech_transcriber=managed_vram_consumers["speech_transcriber"],
                music_valleys=music_valleys,
                input_channel=input_channel,
                cache_dir=cache_dir,
            )

        print("\nDetecting events and exporting to CSV...")
        extracted_events = self._event_detector.detect(chunks_of_words=transcript_chunks, cache_dir=cache_dir)
        self._event_recorder.record(input_video=input_video.resolve(), events=extracted_events, target=output_csv)
        # Ollama will unload the LLM from VRAM when the `keep_alive` time expires, so we don't need to do it explicitly.

        print(f"\nPipeline complete! Results saved to {output_csv}")

    def _detect_music_valleys(
        self,
        audio_classifier: ballroom_scribe.AudioClassifier,
        extracted_audio: pathlib.Path,
        input_channel: int,
        cache_dir: pathlib.Path,
    ) -> list[ballroom_scribe.models.MusicValley]:
        """Loads and classifies audio chunks and detects all music valleys.

        Consumes the entire detection stream into a list so that all PANN inference completes before `audio_classifier`
        is freed from VRAM, while retaining only detached speech-chunk copies so multi-minute audio buffers from
        `AudioLoader` are not pinned in memory.

        Args:
            audio_classifier: The classifier used to segment audio into speech, music, and other sounds.
            extracted_audio: The path to the extracted 32,000 Hz mono WAV file.
            input_channel: The audio channel from which `extracted_audio` was extracted.
            cache_dir: The path to the video's cache directory.

        Returns:
            The detected music valleys across the audio stream.
        """
        unclassified_audio_chunks = self._audio_loader.load(input_audio=extracted_audio)
        classified_audio_chunks = audio_classifier.classify(
            audio_chunks=unclassified_audio_chunks,
            channel=input_channel,
            cache_dir=cache_dir,
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
