"""Orchestration pipeline for audio extraction, transcription, and event timestamping."""

import contextlib
import gc
import pathlib
from collections.abc import Generator
from typing import Literal, TypedDict

import torch

import ballroom_scribe
import ballroom_scribe.cli.arguments
import ballroom_scribe.models
import ballroom_scribe.utils


def _free_vram() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


type _VRAMConsumersKey = Literal[
    "vocal_isolator",
    "speech_transcriber",
]


class _VRAMConsumers(TypedDict, total=False):
    vocal_isolator: ballroom_scribe.VocalIsolator
    speech_transcriber: ballroom_scribe.SpeechTranscriber


class TimestampPipeline:
    """Orchestrates the ballroom competition timestamp extraction pipeline."""

    def __init__(
        self,
        transcript_chunking_args: ballroom_scribe.cli.arguments.TranscriptChunkingArgs,
        vocal_isolation_args: ballroom_scribe.cli.arguments.VocalIsolationArgs,
        speech_transcription_args: ballroom_scribe.cli.arguments.SpeechTranscriptionArgs,
        language_processing_args: ballroom_scribe.cli.arguments.LanguageProcessingArgs,
    ):
        """Initializes the instance.

        Args:
            transcript_chunking_args: Configuration for transcript chunking.
            vocal_isolation_args: Configuration for vocal isolation.
            speech_transcription_args: Configuration for speech transcription.
            language_processing_args: Configuration for language processing.
        """
        self._vocal_isolation_args = vocal_isolation_args
        self._speech_transcription_args = speech_transcription_args

        # These modules do not use the GPU in a way that Python can release the VRAM.
        self._transcript_chunker = ballroom_scribe.TranscriptChunker(**transcript_chunking_args)
        self._event_detector = ballroom_scribe.EventDetector(**language_processing_args)
        self._event_recorder = ballroom_scribe.EventRecorder()

    def run(
        self,
        input_video: pathlib.Path,
        output_csv: pathlib.Path,
        cache_dir: pathlib.Path,
    ) -> None:
        """Runs the complete timestamp pipeline end-to-end.

        Args:
            input_video: The path to the input video file.
            output_csv: The path to the output CSV file.
            cache_dir: The path to the cache directory for intermediate results.
        """
        # All parts of the pipeline should lazy-load anything that uses a significant amount of VRAM.
        # All parts are initialized here so that if there is a bad configuration, the pipeline will fail fast.
        managed_vram_consumers: _VRAMConsumers = {
            "vocal_isolator": ballroom_scribe.VocalIsolator(**self._vocal_isolation_args),
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
        # - Audio.wav: the extracted audio from the input video
        # - Vocals.wav: the isolated vocals from the extracted audio
        # - Speech.csv: the transcribed words from the isolated vocals
        # - *.json: the detected events from the transcript
        # - *.md: LLM thinking output
        # Except for the *.md files, each file is first written to a temporary `*.partial.*` file and renamed once
        # complete, so an existing file is never partially written.
        cache_dir = ballroom_scribe.utils.cache_dir_for_video(parent_cache_dir=cache_dir, input_video=input_video)
        cache_dir.mkdir(parents=True, exist_ok=True)
        extracted_audio = cache_dir / "Audio.wav"
        isolated_vocals = cache_dir / "Vocals.wav"
        transcribed_speech = cache_dir / "Speech.csv"

        print("Extracting audio...")
        ballroom_scribe.utils.extract_audio(
            input_video=input_video,
            output_audio=extracted_audio,
        )

        print("\nIsolating vocals...")
        with free_vram_afterward("vocal_isolator"):
            managed_vram_consumers["vocal_isolator"].separate(
                input_audio=extracted_audio,
                output_vocals=isolated_vocals,
            )

        print("\nTranscribing and chunking speech...")
        with free_vram_afterward("speech_transcriber"):
            chunks_of_words = self._transcribe_and_chunk_speech(
                managed_vram_consumers["speech_transcriber"],
                input_audio=isolated_vocals,
                output_text=transcribed_speech,
            )

        print("\nDetecting events and exporting to CSV...")
        extracted_events = self._event_detector.detect(chunks_of_words=chunks_of_words, cache_dir=cache_dir)
        self._event_recorder.record(input_video=input_video.resolve(), events=extracted_events, target=output_csv)
        # Ollama will unload the LLM from VRAM when the `keep_alive` time expires, so we don't need to do it explicitly.

        print(f"\nPipeline complete! Results saved to {output_csv}")

    def _transcribe_and_chunk_speech(
        self,
        speech_transcriber: ballroom_scribe.SpeechTranscriber,
        input_audio: pathlib.Path,
        output_text: pathlib.Path,
    ) -> list[ballroom_scribe.models.TranscriptChunk]:
        """Transcribes speech and chunks the transcript."""
        # This is its own function to ensure that `words` is deleted and doesn't pin `speech_transcriber` in memory.
        words = speech_transcriber.transcribe(input_audio=input_audio, output_text=output_text)
        return list(self._transcript_chunker.chunk(words=words))
