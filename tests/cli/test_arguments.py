import argparse
import pathlib
import re
import unittest

import ollama
from parameterized import param, parameterized

import ballroom_scribe.cli.arguments
import ballroom_scribe.models


class TestParseArgs(unittest.TestCase):
    """Unit tests for the `parse_args` function."""

    @parameterized.expand(
        [
            param(
                "input_output_cache_defaults",
                args=["path/to/input.mp4", "1"],
                expected=ballroom_scribe.cli.arguments.InputOutputCacheArgs(
                    input_video=pathlib.Path("path/to/input.mp4"),
                    input_channel=1,
                    cache_dir=pathlib.Path("path/to"),
                    output_csv=pathlib.Path("path/to/input.csv"),
                ),
            ),
            param(
                "input_output_cache_custom",
                args=[
                    "path/to/input.mp4",
                    "2",
                    "--cache-dir",
                    "custom_cache_dir",
                    "--output-csv",
                    "custom_output.csv",
                ],
                expected=ballroom_scribe.cli.arguments.InputOutputCacheArgs(
                    input_video=pathlib.Path("path/to/input.mp4"),
                    input_channel=2,
                    cache_dir=pathlib.Path("custom_cache_dir"),
                    output_csv=pathlib.Path("custom_output.csv"),
                ),
            ),
            param(
                "audio_classification",
                args=[
                    "path/to/input.mp4",
                    "1",
                    "--pann-device",
                    "fake-device",
                    "--pann-chunk-minutes",
                    "9.87",
                    "--pann-chunk-overlap-seconds",
                    "4.68",
                    "--pann-threshold-speech",
                    "0.123",
                    "--pann-threshold-music",
                    "0.234",
                    "--pann-min-chunk-seconds",
                    "2.34",
                ],
                expected=ballroom_scribe.cli.arguments.AudioClassificationArgs(
                    pann_device="fake-device",
                    pann_chunk_minutes=9.87,
                    pann_chunk_overlap_seconds=4.68,
                    pann_thresholds={
                        ballroom_scribe.models.AudioChunk.Classification.SPEECH: 0.123,
                        ballroom_scribe.models.AudioChunk.Classification.MUSIC: 0.234,
                    },
                    pann_min_chunk_seconds=2.34,
                ),
            ),
            param(
                "music_valley_detection_defaults",
                args=["path/to/input.mp4", "1"],
                expected=ballroom_scribe.cli.arguments.MusicValleyDetectionArgs(
                    min_dance_seconds=35.0,
                    max_song_dip_seconds=4.5,
                    max_fade_speech_seconds=12.0,
                    max_dip_speech_seconds=2.0,
                ),
            ),
            param(
                "music_valley_detection",
                args=[
                    "path/to/input.mp4",
                    "1",
                    "--min-dance-seconds",
                    "40.5",
                    "--max-song-dip-seconds",
                    "5.5",
                    "--max-fade-speech-seconds",
                    "10.5",
                    "--max-dip-speech-seconds",
                    "1.5",
                ],
                expected=ballroom_scribe.cli.arguments.MusicValleyDetectionArgs(
                    min_dance_seconds=40.5,
                    max_song_dip_seconds=5.5,
                    max_fade_speech_seconds=10.5,
                    max_dip_speech_seconds=1.5,
                ),
            ),
            param(
                "speech_transcription",
                args=[
                    "path/to/input.mp4",
                    "1",
                    "--whisper-init-kwargs",
                    '{"model_size_or_path": "fake-size"}',
                    "--whisper-beam-size",
                    "321",
                ],
                expected=ballroom_scribe.cli.arguments.SpeechTranscriptionArgs(
                    whisper_init_kwargs={"model_size_or_path": "fake-size"},
                    whisper_beam_size=321,
                ),
            ),
            param(
                "ollama_options",
                args=[
                    "path/to/input.mp4",
                    "1",
                    "--ollama-model",
                    "fake-model",
                    "--ollama-options",
                    '{"temperature": 0.123}',
                ],
                expected=ballroom_scribe.cli.arguments.LanguageProcessingArgs(
                    ollama_model="fake-model",
                    ollama_options=ollama.Options(temperature=0.123),
                ),
            ),
        ]
    )
    def test_valid_arguments(self, _, args, expected):
        parsed_args = ballroom_scribe.cli.arguments.parse_args(
            (
                ballroom_scribe.cli.arguments.InputOutputCacheGroup(),
                ballroom_scribe.cli.arguments.AudioClassificationGroup(),
                ballroom_scribe.cli.arguments.MusicValleyDetectionGroup(),
                ballroom_scribe.cli.arguments.SpeechTranscriptionGroup(),
                ballroom_scribe.cli.arguments.LanguageProcessingGroup(),
            ),
            args,
            exit_on_error=False,
        )
        self.assertIn(expected, parsed_args)

    @parameterized.expand(
        [
            param(
                "input_missing",
                args=["--cache-dir", "custom_cache_dir", "--output-csv", "custom_output.csv"],
                expected_message="the following arguments are required: input_video, input_channel",
            ),
            param(
                "input_channel_not_a_number",
                args=["path/to/input.mp4", "foo"],
                expected_message="argument input_channel: invalid int value: 'foo'",
            ),
            param(
                "audio_classification_pann_chunk_minutes_not_a_number",
                args=["path/to/input.mp4", "1", "--pann-chunk-minutes", "foo"],
                expected_message="argument --pann-chunk-minutes: invalid float value: 'foo'",
            ),
            param(
                "audio_classification_pann_chunk_overlap_seconds_not_a_number",
                args=["path/to/input.mp4", "1", "--pann-chunk-overlap-seconds", "foo"],
                expected_message="argument --pann-chunk-overlap-seconds: invalid float value: 'foo'",
            ),
            param(
                "audio_classification_pann_chunk_overlap_seconds_too_small",
                args=[
                    "path/to/input.mp4",
                    "1",
                    "--pann-chunk-overlap-seconds",
                    "1.99",
                    "--pann-min-chunk-seconds",
                    "1.0",
                ],
                expected_message=(
                    "argument --pann-chunk-overlap-seconds: Must be at least double --pann-min-chunk-seconds (2.0),"
                    " got 1.99"
                ),
            ),
            param(
                "audio_classification_pann_threshold_speech_not_a_number",
                args=["path/to/input.mp4", "1", "--pann-threshold-speech", "foo"],
                expected_message="argument --pann-threshold-speech: invalid float value: 'foo'",
            ),
            param(
                "audio_classification_pann_threshold_music_not_a_number",
                args=["path/to/input.mp4", "1", "--pann-threshold-music", "foo"],
                expected_message="argument --pann-threshold-music: invalid float value: 'foo'",
            ),
            param(
                "audio_classification_pann_min_chunk_seconds_not_a_number",
                args=["path/to/input.mp4", "1", "--pann-min-chunk-seconds", "foo"],
                expected_message="argument --pann-min-chunk-seconds: invalid float value: 'foo'",
            ),
            param(
                "music_valley_detection_min_dance_seconds_not_a_number",
                args=["path/to/input.mp4", "1", "--min-dance-seconds", "foo"],
                expected_message="argument --min-dance-seconds: invalid float value: 'foo'",
            ),
            param(
                "music_valley_detection_max_song_dip_seconds_not_a_number",
                args=["path/to/input.mp4", "1", "--max-song-dip-seconds", "foo"],
                expected_message="argument --max-song-dip-seconds: invalid float value: 'foo'",
            ),
            param(
                "music_valley_detection_max_fade_speech_seconds_not_a_number",
                args=["path/to/input.mp4", "1", "--max-fade-speech-seconds", "foo"],
                expected_message="argument --max-fade-speech-seconds: invalid float value: 'foo'",
            ),
            param(
                "music_valley_detection_max_dip_speech_seconds_not_a_number",
                args=["path/to/input.mp4", "1", "--max-dip-speech-seconds", "foo"],
                expected_message="argument --max-dip-speech-seconds: invalid float value: 'foo'",
            ),
            param(
                "speech_transcription_whisper_init_kwargs_not_a_dict",
                args=["path/to/input.mp4", "1", "--whisper-init-kwargs", '["not", "a", "dict"]'],
                expected_message="argument --whisper-init-kwargs: Expected a JSON dictionary, got <class 'list'>",
            ),
            param(
                "speech_transcription_whisper_beam_size_not_a_number",
                args=["path/to/input.mp4", "1", "--whisper-beam-size", "foo"],
                expected_message="argument --whisper-beam-size: invalid int value: 'foo'",
            ),
            param(
                "language_processing_ollama_options_not_a_dict",
                args=["path/to/input.mp4", "1", "--ollama-options", '["not", "a", "dict"]'],
                expected_message="argument --ollama-options: Expected a JSON dictionary, got <class 'list'>",
            ),
        ]
    )
    def test_invalid_arguments(self, _, args, expected_message):
        with self.assertRaisesRegex(argparse.ArgumentError, re.escape(expected_message)):
            ballroom_scribe.cli.arguments.parse_args(
                (
                    ballroom_scribe.cli.arguments.InputOutputCacheGroup(),
                    ballroom_scribe.cli.arguments.AudioClassificationGroup(),
                    ballroom_scribe.cli.arguments.MusicValleyDetectionGroup(),
                    ballroom_scribe.cli.arguments.SpeechTranscriptionGroup(),
                    ballroom_scribe.cli.arguments.LanguageProcessingGroup(),
                ),
                args,
                exit_on_error=False,
            )


if __name__ == "__main__":
    unittest.main()
