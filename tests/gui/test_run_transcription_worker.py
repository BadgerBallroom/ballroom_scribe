import io
import os
import pathlib
import sys
import unittest
from unittest.mock import ANY, MagicMock, call, patch

import ballroom_scribe
import ballroom_scribe.cli.arguments
import ballroom_scribe.gui
import ballroom_scribe.gui.utils
import ballroom_scribe.models
import ballroom_scribe.utils

_ARGV = ["-i", "video.mp4", "1", "-c", "cache"]
_STANDARD_OUTPUT_COPY = 7
_ENVIRONMENT = "Ballroom Scribe 0911b61b64dc\nPython 3.14.3 (python) on Windows-11\nPackages: torch 2.14.0"
_TORCH_DEVICES = "PyTorch: CUDA 13.2 is available (1 device)"


def _make_transcript_chunk(start_samples: int, text: str) -> ballroom_scribe.models.TranscriptChunk:
    """Helper to create a transcript chunk with one word."""
    return ballroom_scribe.models.TranscriptChunk(
        start_samples=start_samples,
        sample_rate=32_000,
        duration_samples=16_000,
        channel=1,
        words=(ballroom_scribe.models.Word(word=text, start=start_samples / 32_000, end=start_samples / 32_000 + 0.5),),
    )


class TestRunTranscriptionWorker(unittest.TestCase):
    """Unit tests for `run_transcription_worker`."""

    def setUp(self):
        self.mock_dup = self.enterContext(patch.object(os, "dup", autospec=True))
        self.mock_dup.return_value = _STANDARD_OUTPUT_COPY
        self.mock_dup2 = self.enterContext(patch.object(os, "dup2", autospec=True))
        self.mock_fdopen = self.enterContext(patch.object(os, "fdopen", autospec=True))
        self.mock_transcript_stream = self.mock_fdopen.return_value
        # The worker replaces `sys.stdout`; the patch puts the original back after the test.
        self.enterContext(patch.object(sys, "stdout", sys.stdout))
        self.mock_stderr = self.enterContext(patch.object(sys, "stderr", new_callable=io.StringIO))
        self.mock_describe_environment = self.enterContext(
            patch.object(ballroom_scribe.utils, "describe_environment", autospec=True, return_value=_ENVIRONMENT)
        )
        self.mock_describe_torch_devices = self.enterContext(
            patch.object(ballroom_scribe.utils, "describe_torch_devices", autospec=True, return_value=_TORCH_DEVICES)
        )

        self.input_cache_args = ballroom_scribe.cli.arguments.InputCacheArgs(
            input_video=pathlib.Path("video.mp4"),
            input_channel=1,
            parent_cache_dir=pathlib.Path("cache"),
        )
        self.audio_classification_args = MagicMock()
        self.music_valley_detection_args = MagicMock()
        self.speech_transcription_args = MagicMock()
        self.mock_parse_args = self.enterContext(
            patch.object(ballroom_scribe.cli.arguments, "parse_args", autospec=True)
        )
        self.mock_parse_args.return_value = (
            self.input_cache_args,
            self.audio_classification_args,
            self.music_valley_detection_args,
            self.speech_transcription_args,
        )

        self.mock_pipeline_cls = self.enterContext(
            patch.object(ballroom_scribe, "TranscriptionPipeline", autospec=True)
        )
        self.mock_pipeline = self.mock_pipeline_cls.return_value

    def test_runs_the_transcription_pipeline_on_the_video(self):
        exit_status = ballroom_scribe.gui.run_transcription_worker(_ARGV)

        self.assertEqual(exit_status, 0)
        # The same argument groups as `find_timestamps.py`, without the output CSV and the language model.
        (groups, argv), _ = self.mock_parse_args.call_args
        self.assertEqual(
            [type(group) for group in groups],
            [
                ballroom_scribe.cli.arguments.InputCacheGroup,
                ballroom_scribe.cli.arguments.AudioClassificationGroup,
                ballroom_scribe.cli.arguments.MusicValleyDetectionGroup,
                ballroom_scribe.cli.arguments.SpeechTranscriptionGroup,
            ],
        )
        self.assertEqual(argv, _ARGV)
        self.mock_pipeline_cls.assert_called_once_with(
            audio_classification_args=self.audio_classification_args,
            music_valley_detection_args=self.music_valley_detection_args,
            speech_transcription_args=self.speech_transcription_args,
        )
        self.mock_pipeline.run.assert_called_once_with([self.input_cache_args], on_speech_transcribed=ANY)

    def test_writes_each_speech_chunk_as_a_line_before_the_pipeline_continues(self):
        first_chunk = _make_transcript_chunk(0, "Welcome")
        second_chunk = _make_transcript_chunk(64_000, "back")
        first_line = ballroom_scribe.gui.utils.encode_transcript_chunk(first_chunk) + "\n"
        second_line = ballroom_scribe.gui.utils.encode_transcript_chunk(second_chunk) + "\n"
        written_after_first_chunk = []

        def run(inputs, on_speech_transcribed):
            on_speech_transcribed(inputs[0], first_chunk)
            written_after_first_chunk.extend(self.mock_transcript_stream.mock_calls)
            on_speech_transcribed(inputs[0], second_chunk)
            return [[first_chunk, second_chunk]]

        self.mock_pipeline.run.side_effect = run

        ballroom_scribe.gui.run_transcription_worker(_ARGV)

        # The first chunk has been written and flushed by the time the pipeline goes on to the second one.
        self.assertEqual(written_after_first_chunk, [call.write(first_line), call.flush()])
        # Nothing but the chunks is written to the transcript stream.
        self.assertEqual(
            self.mock_transcript_stream.mock_calls,
            [call.write(first_line), call.flush(), call.write(second_line), call.flush()],
        )

    def test_keeps_standard_output_for_the_transcript(self):
        standard_output_during_run = []
        self.mock_pipeline.run.side_effect = lambda inputs, on_speech_transcribed: standard_output_during_run.append(
            sys.stdout
        )
        manager = MagicMock()
        manager.attach_mock(self.mock_dup, "dup")
        manager.attach_mock(self.mock_dup2, "dup2")
        manager.attach_mock(self.mock_fdopen, "fdopen")
        manager.attach_mock(self.mock_parse_args, "parse_args")

        ballroom_scribe.gui.run_transcription_worker(_ARGV)

        # The transcript stream is a copy of standard output, made before file descriptor 1 is pointed at standard
        # error and before anything can print.
        self.assertEqual(
            manager.mock_calls,
            [
                call.dup(1),
                call.fdopen(_STANDARD_OUTPUT_COPY, "w", encoding="utf-8", newline="\n"),
                call.dup2(2, 1),
                call.parse_args(ANY, _ARGV, description=ANY),
            ],
        )
        # What the pipeline prints goes to standard error.
        self.assertEqual(standard_output_during_run, [sys.stderr])

    def test_names_the_environment_on_standard_error_before_the_pipeline_runs(self):
        printed_before_the_run = []
        self.mock_pipeline.run.side_effect = lambda inputs, on_speech_transcribed: printed_before_the_run.append(
            self.mock_stderr.getvalue()
        )

        ballroom_scribe.gui.run_transcription_worker(_ARGV)

        self.assertEqual(printed_before_the_run, [f"{_ENVIRONMENT}\n{_TORCH_DEVICES}\n"])
        # None of it reaches the transcript, which the editor reads from standard output.
        self.mock_transcript_stream.write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
