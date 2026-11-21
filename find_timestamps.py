"""CLI for the timestamp extraction pipeline."""

import ballroom_scribe
import ballroom_scribe.cli.arguments
import ballroom_scribe.utils


def run_pipeline():
    """Runs the timestamp extraction pipeline using command-line arguments.

    The output starts with the version of Ballroom Scribe and the software that it runs on, for anyone who reads the
    output to find out why a run failed.
    """
    (
        input_output_cache_args,
        audio_classification_args,
        music_valley_detection_args,
        speech_transcription_args,
        language_processing_args,
    ) = ballroom_scribe.cli.arguments.parse_args(
        (
            ballroom_scribe.cli.arguments.InputOutputCacheGroup(),
            ballroom_scribe.cli.arguments.AudioClassificationGroup(),
            ballroom_scribe.cli.arguments.MusicValleyDetectionGroup(),
            ballroom_scribe.cli.arguments.SpeechTranscriptionGroup(),
            ballroom_scribe.cli.arguments.LanguageProcessingGroup(),
        ),
        description="Extract timestamps for ballroom competition events from one or more video files.",
    )
    print(ballroom_scribe.utils.describe_environment())
    print(ballroom_scribe.utils.describe_torch_devices())
    pipeline = ballroom_scribe.TimestampPipeline(
        audio_classification_args=audio_classification_args,
        music_valley_detection_args=music_valley_detection_args,
        speech_transcription_args=speech_transcription_args,
        language_processing_args=language_processing_args,
    )
    return pipeline.run(**input_output_cache_args)


if __name__ == "__main__":
    run_pipeline()
