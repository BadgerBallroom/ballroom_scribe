"""Orchestration pipeline for audio extraction, transcription, and event timestamping."""

import itertools
import pathlib
from collections.abc import Generator, Iterable

import ballroom_scribe
import ballroom_scribe.cli.arguments
import ballroom_scribe.models
import ballroom_scribe.utils


def _detect_events(
    event_detector: ballroom_scribe.EventDetector,
    video_input: ballroom_scribe.cli.arguments.InputCacheArgs,
    transcript_chunks: Iterable[ballroom_scribe.models.TranscriptChunk],
) -> Generator[tuple[pathlib.Path, ballroom_scribe.models.Event]]:
    """Detects the events in a video's transcript and pairs each with the video's resolved path for the CSV's `File` column.

    Args:
        event_detector: The detector used to find the events that the transcript announces.
        video_input: The video's path, its audio channel, and the parent directory of its cache directory.
        transcript_chunks: The transcript chunk of each music valley of the video.

    Yields:
        `(input_video, event)` pairs, lazily, so that the LLM is queried while the CSV is being written.
    """
    input_video = video_input["input_video"]
    print(f"\nDetecting events in {input_video}...")
    # This is the directory in which `TranscriptionPipeline` cached the video's audio, inferences, and transcripts.
    cache_dir = ballroom_scribe.utils.cache_dir_for_video(
        parent_cache_dir=video_input["parent_cache_dir"],
        input_video=input_video,
    )
    resolved_input_video = input_video.resolve()
    for event in event_detector.detect(chunks_of_words=transcript_chunks, cache_dir=cache_dir):
        yield resolved_input_video, event


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
        self._transcription_pipeline = ballroom_scribe.TranscriptionPipeline(
            audio_classification_args=audio_classification_args,
            music_valley_detection_args=music_valley_detection_args,
            speech_transcription_args=speech_transcription_args,
        )
        self._event_detector = ballroom_scribe.EventDetector(**language_processing_args)
        self._event_recorder = ballroom_scribe.EventRecorder()

    def run(
        self,
        inputs: Iterable[ballroom_scribe.cli.arguments.InputCacheArgs],
        output_csv: pathlib.Path,
    ) -> None:
        """Runs the complete timestamp pipeline end-to-end on one or more videos.

        `TranscriptionPipeline.run` first transcribes the speech of every video. It checks every input before any slow
        work begins, so a mistake in the last video fails the run immediately, and it takes every video through each
        VRAM stage before the next stage begins, so each model is loaded into VRAM only once per run rather than once
        per video. The events of each video are then detected while the CSV is being written.

        Args:
            inputs: For each video: its path, the audio channel to extract, and the parent directory of its cache
                directory.
            output_csv: The path to the output CSV file, which receives the events of every video.

        Raises:
            ValueError: If two inputs resolve to the same cache directory, or an audio channel index is out of range.
        """
        # `inputs` is read twice: by `TranscriptionPipeline.run`, and to pair each video with its transcript.
        inputs = list(inputs)
        transcripts = self._transcription_pipeline.run(inputs)

        print("\nDetecting events and exporting to CSV...")
        events = itertools.chain.from_iterable(
            _detect_events(self._event_detector, video_input, transcript_chunks)
            for video_input, transcript_chunks in zip(inputs, transcripts, strict=True)
        )
        self._event_recorder.record(events=events, target=output_csv)
        # Ollama will unload the LLM from VRAM when the `keep_alive` time expires, so we don't need to do it explicitly.

        print(f"\nPipeline complete! Results saved to {output_csv}")
