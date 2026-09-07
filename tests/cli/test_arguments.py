import argparse
import pathlib
import re
import unittest

import ollama
from parameterized import param, parameterized

import ballroom_scribe.cli.arguments


class TestParseArgs(unittest.TestCase):
    """Unit tests for the `parse_args` function."""

    @parameterized.expand(
        [
            param(
                "input_output_cache_defaults",
                args=["path/to/input.mp4"],
                expected=ballroom_scribe.cli.arguments.InputOutputCacheArgs(
                    input_video=pathlib.Path("path/to/input.mp4"),
                    cache_dir=pathlib.Path("path/to"),
                    output_csv=pathlib.Path("path/to/input.csv"),
                ),
            ),
            param(
                "input_output_cache_custom",
                args=["path/to/input.mp4", "--cache-dir", "custom_cache_dir", "--output-csv", "custom_output.csv"],
                expected=ballroom_scribe.cli.arguments.InputOutputCacheArgs(
                    input_video=pathlib.Path("path/to/input.mp4"),
                    cache_dir=pathlib.Path("custom_cache_dir"),
                    output_csv=pathlib.Path("custom_output.csv"),
                ),
            ),
            param(
                "transcript_chunking",
                args=["path/to/input.mp4", "--split-on-silence-threshold", "0.123"],
                expected=ballroom_scribe.cli.arguments.TranscriptChunkingArgs(
                    split_on_silence_threshold=0.123,
                ),
            ),
            param(
                "vocal_isolation",
                args=["path/to/input.mp4", "--demucs-name", "fake-model", "--demucs-device", "fake-device"],
                expected=ballroom_scribe.cli.arguments.VocalIsolationArgs(
                    demucs_name="fake-model",
                    demucs_device="fake-device",
                ),
            ),
            param(
                "speech_transcription",
                args=[
                    "path/to/input.mp4",
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
                ballroom_scribe.cli.arguments.TranscriptChunkingGroup(),
                ballroom_scribe.cli.arguments.VocalIsolationGroup(),
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
                expected_message="the following arguments are required: input_video",
            ),
            param(
                "speech_transcription_whisper_init_kwargs_not_a_dict",
                args=["path/to/input.mp4", "--whisper-init-kwargs", '["not", "a", "dict"]'],
                expected_message="argument --whisper-init-kwargs: Expected a JSON dictionary, got <class 'list'>",
            ),
            param(
                "speech_transcription_whisper_beam_size_not_a_number",
                args=["path/to/input.mp4", "--whisper-beam-size", "foo"],
                expected_message="argument --whisper-beam-size: invalid int value: 'foo'",
            ),
            param(
                "language_processing_ollama_options_not_a_dict",
                args=["path/to/input.mp4", "--ollama-options", '["not", "a", "dict"]'],
                expected_message="argument --ollama-options: Expected a JSON dictionary, got <class 'list'>",
            ),
        ]
    )
    def test_invalid_arguments(self, _, args, expected_message):
        with self.assertRaisesRegex(argparse.ArgumentError, re.escape(expected_message)):
            ballroom_scribe.cli.arguments.parse_args(
                (
                    ballroom_scribe.cli.arguments.InputOutputCacheGroup(),
                    ballroom_scribe.cli.arguments.TranscriptChunkingGroup(),
                    ballroom_scribe.cli.arguments.VocalIsolationGroup(),
                    ballroom_scribe.cli.arguments.SpeechTranscriptionGroup(),
                    ballroom_scribe.cli.arguments.LanguageProcessingGroup(),
                ),
                args,
                exit_on_error=False,
            )


if __name__ == "__main__":
    unittest.main()
