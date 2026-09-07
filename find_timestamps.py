"""CLI for the timestamp extraction pipeline."""

import ballroom_scribe
import ballroom_scribe.cli.arguments


def run_pipeline():
    """Runs the timestamp extraction pipeline using command-line arguments."""
    (
        input_output_cache_args,
        transcript_chunking_args,
        vocal_isolation_args,
        speech_transcription_args,
        language_processing_args,
    ) = ballroom_scribe.cli.arguments.parse_args(
        (
            ballroom_scribe.cli.arguments.InputOutputCacheGroup(),
            ballroom_scribe.cli.arguments.TranscriptChunkingGroup(),
            ballroom_scribe.cli.arguments.VocalIsolationGroup(),
            ballroom_scribe.cli.arguments.SpeechTranscriptionGroup(),
            ballroom_scribe.cli.arguments.LanguageProcessingGroup(),
        ),
        description="Extract timestamps for ballroom competition events from a video file.",
    )
    pipeline = ballroom_scribe.TimestampPipeline(
        transcript_chunking_args=transcript_chunking_args,
        vocal_isolation_args=vocal_isolation_args,
        speech_transcription_args=speech_transcription_args,
        language_processing_args=language_processing_args,
    )
    return pipeline.run(**input_output_cache_args)


if __name__ == "__main__":
    run_pipeline()
