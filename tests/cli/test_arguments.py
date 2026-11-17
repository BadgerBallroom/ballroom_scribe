import argparse
import os
import pathlib
import re
import unittest
from unittest.mock import patch

import ollama
from parameterized import param, parameterized

import ballroom_scribe.cli.arguments
import ballroom_scribe.models
import ballroom_scribe.utils

# The number of logical processors that the mocked `os.process_cpu_count` reports; `--whisper-cpu-threads` defaults
# to it.
_CPU_THREADS = 12


class TestParseArgs(unittest.TestCase):
    """Unit tests for the `parse_args` function."""

    def setUp(self):
        self.mock_process_cpu_count = self.enterContext(patch.object(os, "process_cpu_count", autospec=True))
        self.mock_process_cpu_count.return_value = _CPU_THREADS

        # `SpeechTranscriptionGroup.resolve` asks PyTorch which device `--whisper-device auto` means.
        self.mock_resolve_torch_device = self.enterContext(
            patch.object(ballroom_scribe.utils, "resolve_torch_device", autospec=True)
        )
        self.mock_resolve_torch_device.return_value = "cpu"

    @parameterized.expand(
        [
            param(
                "input_output_cache_defaults",
                args=["-i", "path/to/input.mp4", "1"],
                expected=ballroom_scribe.cli.arguments.InputOutputCacheArgs(
                    inputs=[
                        ballroom_scribe.cli.arguments.InputCacheArgs(
                            input_video=pathlib.Path("path/to/input.mp4"),
                            input_channel=1,
                            parent_cache_dir=pathlib.Path("path/to"),
                        ),
                    ],
                    output_csv=pathlib.Path("path/to/input.csv"),
                ),
            ),
            param(
                "input_output_cache_custom",
                args=[
                    "-i",
                    "path/to/input.mp4",
                    "2",
                    "--cache-dir",
                    "custom_cache_dir",
                    "--output-csv",
                    "custom_output.csv",
                ],
                expected=ballroom_scribe.cli.arguments.InputOutputCacheArgs(
                    inputs=[
                        ballroom_scribe.cli.arguments.InputCacheArgs(
                            input_video=pathlib.Path("path/to/input.mp4"),
                            input_channel=2,
                            parent_cache_dir=pathlib.Path("custom_cache_dir"),
                        ),
                    ],
                    output_csv=pathlib.Path("custom_output.csv"),
                ),
            ),
            param(
                "multiple_inputs",
                args=[
                    "-i",
                    "path/to/a.mp4",
                    "1",
                    "--input",
                    "other/b.mp4",
                    "0",
                    "-o",
                    "combined.csv",
                ],
                expected=ballroom_scribe.cli.arguments.InputOutputCacheArgs(
                    inputs=[
                        ballroom_scribe.cli.arguments.InputCacheArgs(
                            input_video=pathlib.Path("path/to/a.mp4"),
                            input_channel=1,
                            parent_cache_dir=pathlib.Path("path/to"),
                        ),
                        ballroom_scribe.cli.arguments.InputCacheArgs(
                            input_video=pathlib.Path("other/b.mp4"),
                            input_channel=0,
                            parent_cache_dir=pathlib.Path("other"),
                        ),
                    ],
                    output_csv=pathlib.Path("combined.csv"),
                ),
            ),
            param(
                "multiple_inputs_cache_dirs",
                args=[
                    "-i",
                    "path/to/a.mp4",
                    "1",
                    "--cache-dir",
                    "cache_a",
                    "-i",
                    "other/b.mp4",
                    "0",
                    "-c",
                    "cache_b",
                    "-o",
                    "combined.csv",
                ],
                expected=ballroom_scribe.cli.arguments.InputOutputCacheArgs(
                    inputs=[
                        ballroom_scribe.cli.arguments.InputCacheArgs(
                            input_video=pathlib.Path("path/to/a.mp4"),
                            input_channel=1,
                            parent_cache_dir=pathlib.Path("cache_a"),
                        ),
                        ballroom_scribe.cli.arguments.InputCacheArgs(
                            input_video=pathlib.Path("other/b.mp4"),
                            input_channel=0,
                            parent_cache_dir=pathlib.Path("cache_b"),
                        ),
                    ],
                    output_csv=pathlib.Path("combined.csv"),
                ),
            ),
            param(
                "cache_dir_for_second_input_only",
                args=[
                    "-i",
                    "path/to/a.mp4",
                    "1",
                    "-i",
                    "other/b.mp4",
                    "0",
                    "--cache-dir",
                    "cache_b",
                    "-o",
                    "combined.csv",
                ],
                expected=ballroom_scribe.cli.arguments.InputOutputCacheArgs(
                    inputs=[
                        ballroom_scribe.cli.arguments.InputCacheArgs(
                            input_video=pathlib.Path("path/to/a.mp4"),
                            input_channel=1,
                            parent_cache_dir=pathlib.Path("path/to"),
                        ),
                        ballroom_scribe.cli.arguments.InputCacheArgs(
                            input_video=pathlib.Path("other/b.mp4"),
                            input_channel=0,
                            parent_cache_dir=pathlib.Path("cache_b"),
                        ),
                    ],
                    output_csv=pathlib.Path("combined.csv"),
                ),
            ),
            param(
                "audio_classification_defaults",
                args=["-i", "path/to/input.mp4", "1"],
                expected=ballroom_scribe.cli.arguments.AudioClassificationArgs(
                    pann_device="auto",
                    pann_chunk_minutes=10.0,
                    pann_chunk_overlap_seconds=2.0,
                    pann_thresholds={
                        ballroom_scribe.models.AudioChunk.Classification.SPEECH: 0.18,
                        ballroom_scribe.models.AudioChunk.Classification.MUSIC: 0.45,
                    },
                    pann_min_chunk_seconds=1.0,
                ),
            ),
            param(
                "audio_classification",
                args=[
                    "-i",
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
                args=["-i", "path/to/input.mp4", "1"],
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
                    "-i",
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
                "speech_transcription_defaults",
                args=["-i", "path/to/input.mp4", "1"],
                expected=ballroom_scribe.cli.arguments.SpeechTranscriptionArgs(
                    whisper_device="cpu",
                    whisper_model="turbo",
                    whisper_compute_type="int8",
                    whisper_cpu_threads=_CPU_THREADS,
                    whisper_beam_size=5,
                    whisper_init_kwargs={},
                ),
            ),
            param(
                "speech_transcription",
                args=[
                    "-i",
                    "path/to/input.mp4",
                    "1",
                    "--whisper-device",
                    "cpu",
                    "--whisper-model",
                    "fake-size",
                    "--whisper-compute-type",
                    "int8",
                    "--whisper-cpu-threads",
                    "3",
                    "--whisper-beam-size",
                    "321",
                    "--whisper-init-kwargs",
                    '{"local_files_only": true}',
                ],
                expected=ballroom_scribe.cli.arguments.SpeechTranscriptionArgs(
                    whisper_device="cpu",
                    whisper_model="fake-size",
                    whisper_compute_type="int8",
                    whisper_cpu_threads=3,
                    whisper_beam_size=321,
                    whisper_init_kwargs={"local_files_only": True},
                ),
            ),
            param(
                "ollama_options",
                args=[
                    "-i",
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
            param(
                "logging_defaults",
                args=["-i", "path/to/input.mp4", "1"],
                expected=ballroom_scribe.cli.arguments.LoggingArgs(verbose=False),
            ),
            param(
                "logging_short_flag",
                args=["-i", "path/to/input.mp4", "1", "-v"],
                expected=ballroom_scribe.cli.arguments.LoggingArgs(verbose=True),
            ),
            param(
                "logging_long_flag",
                args=["-i", "path/to/input.mp4", "1", "--verbose"],
                expected=ballroom_scribe.cli.arguments.LoggingArgs(verbose=True),
            ),
        ]
    )
    def test_valid_arguments(self, _, args, expected):
        parsed_args = ballroom_scribe.cli.arguments.parse_args(
            (
                ballroom_scribe.cli.arguments.InputOutputCacheGroup(max_inputs=2),
                ballroom_scribe.cli.arguments.AudioClassificationGroup(),
                ballroom_scribe.cli.arguments.MusicValleyDetectionGroup(),
                ballroom_scribe.cli.arguments.SpeechTranscriptionGroup(),
                ballroom_scribe.cli.arguments.LanguageProcessingGroup(),
                ballroom_scribe.cli.arguments.LoggingGroup(),
            ),
            args,
            exit_on_error=False,
        )
        self.assertIn(expected, parsed_args)

    @parameterized.expand(
        [
            param(
                "defaults",
                args=["-i", "path/to/input.mp4", "1"],
                expected=ballroom_scribe.cli.arguments.InputCacheArgs(
                    input_video=pathlib.Path("path/to/input.mp4"),
                    input_channel=1,
                    parent_cache_dir=pathlib.Path("path/to"),
                ),
            ),
            param(
                "custom_cache_dir",
                args=["-i", "path/to/input.mp4", "2", "--cache-dir", "custom_cache_dir"],
                expected=ballroom_scribe.cli.arguments.InputCacheArgs(
                    input_video=pathlib.Path("path/to/input.mp4"),
                    input_channel=2,
                    parent_cache_dir=pathlib.Path("custom_cache_dir"),
                ),
            ),
        ]
    )
    def test_valid_input_cache_arguments(self, _, args, expected):
        parsed_args = ballroom_scribe.cli.arguments.parse_args(
            (
                ballroom_scribe.cli.arguments.InputCacheGroup(),
                ballroom_scribe.cli.arguments.AudioClassificationGroup(),
            ),
            args,
            exit_on_error=False,
        )
        self.assertIn(expected, parsed_args)

    @parameterized.expand(
        [
            param(
                "empty",
                args=[],
                expected=ballroom_scribe.cli.arguments.InputCsvCacheArgs(
                    inputs=[],
                    csv=None,
                ),
            ),
            param(
                "csv_only",
                args=["path/to/events.csv"],
                expected=ballroom_scribe.cli.arguments.InputCsvCacheArgs(
                    inputs=[],
                    csv=pathlib.Path("path/to/events.csv"),
                ),
            ),
            param(
                "videos_only_with_per_input_cache_dir",
                args=[
                    "-i",
                    "path/to/a.mp4",
                    "0",
                    "-c",
                    "cache_a",
                    "-i",
                    "other/b.mp4",
                    "1",
                ],
                expected=ballroom_scribe.cli.arguments.InputCsvCacheArgs(
                    inputs=[
                        ballroom_scribe.cli.arguments.InputCacheArgs(
                            input_video=pathlib.Path("path/to/a.mp4"),
                            input_channel=0,
                            parent_cache_dir=pathlib.Path("cache_a"),
                        ),
                        ballroom_scribe.cli.arguments.InputCacheArgs(
                            input_video=pathlib.Path("other/b.mp4"),
                            input_channel=1,
                            parent_cache_dir=pathlib.Path("other"),
                        ),
                    ],
                    csv=None,
                ),
            ),
            param(
                "csv_and_video",
                args=[
                    "path/to/events.csv",
                    "-i",
                    "path/to/c.mp4",
                    "0",
                ],
                expected=ballroom_scribe.cli.arguments.InputCsvCacheArgs(
                    inputs=[
                        ballroom_scribe.cli.arguments.InputCacheArgs(
                            input_video=pathlib.Path("path/to/c.mp4"),
                            input_channel=0,
                            parent_cache_dir=pathlib.Path("path/to"),
                        ),
                    ],
                    csv=pathlib.Path("path/to/events.csv"),
                ),
            ),
        ]
    )
    def test_valid_input_csv_cache_arguments(self, _, args, expected):
        parsed_args = ballroom_scribe.cli.arguments.parse_args(
            (
                ballroom_scribe.cli.arguments.InputCsvCacheGroup(),
                ballroom_scribe.cli.arguments.LoggingGroup(),
            ),
            args,
            exit_on_error=False,
        )
        self.assertIn(expected, parsed_args)

    @parameterized.expand(
        [
            param(
                "input_missing",
                args=["--output-csv", "custom_output.csv"],
                expected_message="the following arguments are required: -i/--input",
            ),
            param(
                "input_channel_missing",
                args=["-i", "path/to/input.mp4"],
                expected_message="argument -i/--input: expected 2 arguments",
            ),
            param(
                "input_channel_not_a_number",
                args=["-i", "path/to/input.mp4", "foo"],
                expected_message="argument -i/--input: invalid int value: 'foo'",
            ),
            param(
                "input_output_cache_more_than_max_inputs",
                args=["-i", "a.mp4", "1", "-i", "b.mp4", "0", "-i", "c.mp4", "1", "-o", "out.csv"],
                expected_message="argument -i/--input: Expected at most 2 video(s), got 3",
            ),
            param(
                "output_csv_missing_for_multiple_inputs",
                args=["-i", "a.mp4", "1", "-i", "b.mp4", "0"],
                expected_message="argument -o/--output-csv: Required when more than one -i/--input is given",
            ),
            param(
                "cache_dir_before_input",
                args=["--cache-dir", "custom_cache_dir", "-i", "path/to/input.mp4", "1"],
                expected_message="argument -c/--cache-dir: Must come after the -i/--input to which it applies",
            ),
            param(
                "cache_dir_twice_for_same_input",
                args=["-i", "path/to/input.mp4", "1", "--cache-dir", "dir1", "--cache-dir", "dir2"],
                expected_message="argument -c/--cache-dir: Given more than once for path/to/input.mp4",
            ),
            param(
                "audio_classification_pann_chunk_minutes_not_a_number",
                args=["-i", "path/to/input.mp4", "1", "--pann-chunk-minutes", "foo"],
                expected_message="argument --pann-chunk-minutes: invalid float value: 'foo'",
            ),
            param(
                "audio_classification_pann_chunk_overlap_seconds_not_a_number",
                args=["-i", "path/to/input.mp4", "1", "--pann-chunk-overlap-seconds", "foo"],
                expected_message="argument --pann-chunk-overlap-seconds: invalid float value: 'foo'",
            ),
            param(
                "audio_classification_pann_chunk_overlap_seconds_too_small",
                args=[
                    "-i",
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
                args=["-i", "path/to/input.mp4", "1", "--pann-threshold-speech", "foo"],
                expected_message="argument --pann-threshold-speech: invalid float value: 'foo'",
            ),
            param(
                "audio_classification_pann_threshold_music_not_a_number",
                args=["-i", "path/to/input.mp4", "1", "--pann-threshold-music", "foo"],
                expected_message="argument --pann-threshold-music: invalid float value: 'foo'",
            ),
            param(
                "audio_classification_pann_min_chunk_seconds_not_a_number",
                args=["-i", "path/to/input.mp4", "1", "--pann-min-chunk-seconds", "foo"],
                expected_message="argument --pann-min-chunk-seconds: invalid float value: 'foo'",
            ),
            param(
                "music_valley_detection_min_dance_seconds_not_a_number",
                args=["-i", "path/to/input.mp4", "1", "--min-dance-seconds", "foo"],
                expected_message="argument --min-dance-seconds: invalid float value: 'foo'",
            ),
            param(
                "music_valley_detection_max_song_dip_seconds_not_a_number",
                args=["-i", "path/to/input.mp4", "1", "--max-song-dip-seconds", "foo"],
                expected_message="argument --max-song-dip-seconds: invalid float value: 'foo'",
            ),
            param(
                "music_valley_detection_max_fade_speech_seconds_not_a_number",
                args=["-i", "path/to/input.mp4", "1", "--max-fade-speech-seconds", "foo"],
                expected_message="argument --max-fade-speech-seconds: invalid float value: 'foo'",
            ),
            param(
                "music_valley_detection_max_dip_speech_seconds_not_a_number",
                args=["-i", "path/to/input.mp4", "1", "--max-dip-speech-seconds", "foo"],
                expected_message="argument --max-dip-speech-seconds: invalid float value: 'foo'",
            ),
            param(
                "speech_transcription_whisper_device_invalid_choice",
                args=["-i", "path/to/input.mp4", "1", "--whisper-device", "gpu"],
                expected_message="argument --whisper-device: invalid choice: 'gpu'",
            ),
            param(
                "speech_transcription_whisper_init_kwargs_not_a_dict",
                args=["-i", "path/to/input.mp4", "1", "--whisper-init-kwargs", '["not", "a", "dict"]'],
                expected_message="argument --whisper-init-kwargs: Expected a JSON dictionary, got <class 'list'>",
            ),
            param(
                "speech_transcription_whisper_init_kwargs_model_key",
                args=["-i", "path/to/input.mp4", "1", "--whisper-init-kwargs", '{"model_size_or_path": "turbo"}'],
                expected_message="argument --whisper-init-kwargs: Use --whisper-model instead of the 'model_size_or_path' key",
            ),
            param(
                "speech_transcription_whisper_init_kwargs_device_key",
                args=["-i", "path/to/input.mp4", "1", "--whisper-init-kwargs", '{"device": "cpu"}'],
                expected_message="argument --whisper-init-kwargs: Use --whisper-device instead of the 'device' key",
            ),
            param(
                "speech_transcription_whisper_init_kwargs_compute_type_key",
                args=["-i", "path/to/input.mp4", "1", "--whisper-init-kwargs", '{"compute_type": "int8"}'],
                expected_message=(
                    "argument --whisper-init-kwargs: Use --whisper-compute-type instead of the 'compute_type' key"
                ),
            ),
            param(
                "speech_transcription_whisper_init_kwargs_cpu_threads_key",
                args=["-i", "path/to/input.mp4", "1", "--whisper-init-kwargs", '{"cpu_threads": 8}'],
                expected_message="argument --whisper-init-kwargs: Use --whisper-cpu-threads instead of the 'cpu_threads' key",
            ),
            param(
                "speech_transcription_whisper_cpu_threads_not_a_number",
                args=["-i", "path/to/input.mp4", "1", "--whisper-cpu-threads", "foo"],
                expected_message="argument --whisper-cpu-threads: invalid int value: 'foo'",
            ),
            param(
                "speech_transcription_whisper_beam_size_not_a_number",
                args=["-i", "path/to/input.mp4", "1", "--whisper-beam-size", "foo"],
                expected_message="argument --whisper-beam-size: invalid int value: 'foo'",
            ),
            param(
                "language_processing_ollama_options_not_a_dict",
                args=["-i", "path/to/input.mp4", "1", "--ollama-options", '["not", "a", "dict"]'],
                expected_message="argument --ollama-options: Expected a JSON dictionary, got <class 'list'>",
            ),
        ]
    )
    def test_invalid_arguments(self, _, args, expected_message):
        with self.assertRaisesRegex(argparse.ArgumentError, re.escape(expected_message)):
            ballroom_scribe.cli.arguments.parse_args(
                (
                    ballroom_scribe.cli.arguments.InputOutputCacheGroup(max_inputs=2),
                    ballroom_scribe.cli.arguments.AudioClassificationGroup(),
                    ballroom_scribe.cli.arguments.MusicValleyDetectionGroup(),
                    ballroom_scribe.cli.arguments.SpeechTranscriptionGroup(),
                    ballroom_scribe.cli.arguments.LanguageProcessingGroup(),
                    ballroom_scribe.cli.arguments.LoggingGroup(),
                ),
                args,
                exit_on_error=False,
            )

    def test_invalid_input_cache_arguments(self):
        with self.assertRaisesRegex(
            argparse.ArgumentError,
            re.escape("argument -i/--input: Expected at most 1 video(s), got 2"),
        ):
            ballroom_scribe.cli.arguments.parse_args(
                (
                    ballroom_scribe.cli.arguments.InputCacheGroup(),
                    ballroom_scribe.cli.arguments.AudioClassificationGroup(),
                ),
                ["-i", "a.mp4", "1", "-i", "b.mp4", "0"],
                exit_on_error=False,
            )

    @parameterized.expand(
        [
            param("zero", max_inputs=0),
            param("negative", max_inputs=-1),
        ]
    )
    def test_rejects_non_positive_max_inputs(self, _, max_inputs):
        with self.assertRaisesRegex(ValueError, f"max_inputs must be at least 1, got {max_inputs}"):
            ballroom_scribe.cli.arguments.InputOutputCacheGroup(max_inputs=max_inputs)

    @parameterized.expand(
        [
            param("known", process_cpu_count=12, expected=12),
            param("unknown", process_cpu_count=None, expected=1),
        ]
    )
    def test_whisper_cpu_threads_defaults_to_available_processors(self, _, process_cpu_count, expected):
        self.mock_process_cpu_count.return_value = process_cpu_count

        (parsed_args,) = ballroom_scribe.cli.arguments.parse_args(
            (ballroom_scribe.cli.arguments.SpeechTranscriptionGroup(),),
            [],
            exit_on_error=False,
        )

        self.assertEqual(parsed_args["whisper_cpu_threads"], expected)

    @parameterized.expand(
        [
            param(
                "auto_resolves_to_cuda",
                args=[],
                resolved_device="cuda",
                expected_requested="auto",
                expected_model="large-v3",
                expected_compute_type="float16",
            ),
            param(
                "auto_resolves_to_cpu",
                args=[],
                resolved_device="cpu",
                expected_requested="auto",
                expected_model="turbo",
                expected_compute_type="int8",
            ),
            param(
                "explicit_device_still_validated",
                args=["--whisper-device", "cuda"],
                resolved_device="cuda",
                expected_requested="cuda",
                expected_model="large-v3",
                expected_compute_type="float16",
            ),
            param(
                "explicit_model_applies_to_every_device",
                args=["--whisper-model", "medium"],
                resolved_device="cpu",
                expected_requested="auto",
                expected_model="medium",
                expected_compute_type="int8",
            ),
        ]
    )
    def test_whisper_defaults_follow_resolved_device(
        self, _, args, resolved_device, expected_requested, expected_model, expected_compute_type
    ):
        self.mock_resolve_torch_device.return_value = resolved_device

        (parsed_args,) = ballroom_scribe.cli.arguments.parse_args(
            (ballroom_scribe.cli.arguments.SpeechTranscriptionGroup(),),
            args,
            exit_on_error=False,
        )

        self.mock_resolve_torch_device.assert_called_once_with(expected_requested)
        self.assertEqual(parsed_args["whisper_device"], resolved_device)
        self.assertEqual(parsed_args["whisper_model"], expected_model)
        self.assertEqual(parsed_args["whisper_compute_type"], expected_compute_type)

    def test_whisper_device_unavailable_is_an_argument_error(self):
        self.mock_resolve_torch_device.side_effect = ValueError("The device is set to CUDA, but CUDA is not available.")

        with self.assertRaisesRegex(
            argparse.ArgumentError,
            re.escape("argument --whisper-device: The device is set to CUDA, but CUDA is not available."),
        ):
            ballroom_scribe.cli.arguments.parse_args(
                (ballroom_scribe.cli.arguments.SpeechTranscriptionGroup(),),
                ["--whisper-device", "cuda"],
                exit_on_error=False,
            )


if __name__ == "__main__":
    unittest.main()
