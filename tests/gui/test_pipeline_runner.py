import logging
import pathlib
import signal
import sys
import unittest
from unittest.mock import MagicMock, patch

from parameterized import param, parameterized

try:
    from PySide6 import QtCore
except ImportError as err:
    raise unittest.SkipTest("PySide6 is not installed.") from err

import ballroom_scribe.cli.arguments
import ballroom_scribe.gui
import ballroom_scribe.gui.pipeline_runner

# Expected command arguments are derived from these same `Path` objects via `str(...)`, so the test is agnostic to
# the platform's path separator.
_VIDEO_A = pathlib.Path("/videos/a.mp4")
_CACHE_A = pathlib.Path("/cache_a")
_VIDEO_B = pathlib.Path("/videos/b.mp4")
_CACHE_B = pathlib.Path("/videos")
_OUTPUT_CSV = pathlib.Path("/out/events.csv")


class TestBuildCommand(unittest.TestCase):
    """Unit tests for `_build_command`."""

    def test_builds_multi_video_command_with_per_input_cache_dirs_and_extra_args(self):
        inputs = [
            ballroom_scribe.cli.arguments.InputCacheArgs(
                input_video=_VIDEO_A,
                input_channel=0,
                parent_cache_dir=_CACHE_A,
            ),
            ballroom_scribe.cli.arguments.InputCacheArgs(
                input_video=_VIDEO_B,
                input_channel=2,
                parent_cache_dir=_CACHE_B,
            ),
        ]

        command = ballroom_scribe.gui.pipeline_runner._build_command(
            inputs=inputs,
            output_csv=_OUTPUT_CSV,
            extra_args=["--pann-threshold-speech", "0.25"],
        )

        self.assertEqual(
            command,
            [
                sys.executable,
                "-u",
                str(ballroom_scribe.gui.pipeline_runner._FIND_TIMESTAMPS_SCRIPT),
                "-i",
                str(_VIDEO_A),
                "0",
                "-c",
                str(_CACHE_A),
                "-i",
                str(_VIDEO_B),
                "2",
                "-c",
                str(_CACHE_B),
                "--output-csv",
                str(_OUTPUT_CSV),
                "--pann-threshold-speech",
                "0.25",
            ],
        )

    def test_rejects_empty_inputs(self):
        with self.assertRaises(ValueError):
            ballroom_scribe.gui.pipeline_runner._build_command(
                inputs=[],
                output_csv=pathlib.Path("out.csv"),
            )


class TestBuildTranscriptionCommand(unittest.TestCase):
    """Unit tests for `_build_transcription_command`."""

    def test_appends_the_video_to_the_command_of_the_worker(self):
        video_input = ballroom_scribe.cli.arguments.InputCacheArgs(
            input_video=_VIDEO_A,
            input_channel=1,
            parent_cache_dir=_CACHE_A,
        )

        command = ballroom_scribe.gui.pipeline_runner._build_transcription_command(
            video_input,
            ("python", "-u", "-m", "worker"),
        )

        self.assertEqual(
            command,
            ["python", "-u", "-m", "worker", "-i", str(_VIDEO_A), "1", "-c", str(_CACHE_A)],
        )

    def test_rejects_an_empty_command(self):
        video_input = ballroom_scribe.cli.arguments.InputCacheArgs(
            input_video=_VIDEO_A,
            input_channel=1,
            parent_cache_dir=_CACHE_A,
        )

        with self.assertRaises(ValueError):
            ballroom_scribe.gui.pipeline_runner._build_transcription_command(video_input, ())


def _make_byte_array(data: bytes) -> MagicMock:
    """Helper to create a stand-in for the `QByteArray` that `QProcess` returns."""
    byte_array = MagicMock()
    byte_array.data.return_value = data
    return byte_array


class TestPipelineRunner(unittest.TestCase):
    """Unit tests for `PipelineRunner` with a mocked `QProcess`."""

    def setUp(self):
        self.mock_qprocess_cls = self.enterContext(patch.object(QtCore, "QProcess", autospec=True))
        self.mock_process = self.mock_qprocess_cls.return_value
        self.mock_process.readyReadStandardOutput = MagicMock()
        self.mock_process.readyReadStandardError = MagicMock()
        self.mock_process.finished = MagicMock()
        self.mock_process.errorOccurred = MagicMock()
        self.mock_process.state.return_value = QtCore.QProcess.ProcessState.NotRunning
        self.mock_process.readAllStandardOutput.return_value = _make_byte_array(b"Extracting audio...\n")

    def test_start_cancel_and_signals(self):
        runner = ballroom_scribe.gui.PipelineRunner()
        outputs: list[str] = []
        exit_codes: list[int] = []
        runner.output.connect(outputs.append)
        runner.finished.connect(exit_codes.append)
        self.mock_process.setProcessChannelMode.assert_called_once_with(
            QtCore.QProcess.ProcessChannelMode.MergedChannels
        )

        inputs = [
            ballroom_scribe.cli.arguments.InputCacheArgs(
                input_video=pathlib.Path("clip.mp4"),
                input_channel=1,
                parent_cache_dir=pathlib.Path("."),
            )
        ]
        runner.start(inputs=inputs, output_csv=pathlib.Path("clip.csv"))
        self.mock_process.start.assert_called_once()

        self.mock_process.state.return_value = QtCore.QProcess.ProcessState.Running
        self.assertTrue(runner.is_running())
        runner.cancel()
        self.mock_process.kill.assert_called_once()

        runner._on_finished(0, QtCore.QProcess.ExitStatus.NormalExit)
        self.assertEqual(outputs, ["Extracting audio...\n"])
        self.assertEqual(exit_codes, [0])

    def test_reports_a_killed_process_as_failed(self):
        runner = ballroom_scribe.gui.PipelineRunner()
        exit_codes: list[int] = []
        runner.finished.connect(exit_codes.append)

        runner._on_finished(9, QtCore.QProcess.ExitStatus.CrashExit)

        self.assertEqual(exit_codes, [-1])

    def test_reports_a_program_that_could_not_be_started_as_failed(self):
        runner = ballroom_scribe.gui.PipelineRunner()
        exit_codes: list[int] = []
        runner.finished.connect(exit_codes.append)

        # `QProcess` reports a crash through `finished` as well, so only a failed start is reported here.
        runner._on_error_occurred(QtCore.QProcess.ProcessError.Crashed)
        self.assertEqual(exit_codes, [])
        runner._on_error_occurred(QtCore.QProcess.ProcessError.FailedToStart)
        self.assertEqual(exit_codes, [-1])

    def test_children_report_fatal_errors_and_print_utf8(self):
        ballroom_scribe.gui.PipelineRunner()

        (environment,), _ = self.mock_process.setProcessEnvironment.call_args
        self.assertEqual(environment.value("PYTHONUTF8"), "1")
        # A child that dies in native code then writes the Python stack of every thread to its standard error.
        self.assertEqual(environment.value("PYTHONFAULTHANDLER"), "1")

    def test_logs_the_command_that_it_starts(self):
        runner = ballroom_scribe.gui.PipelineRunner()
        inputs = [
            ballroom_scribe.cli.arguments.InputCacheArgs(
                input_video=pathlib.Path("clip.mp4"),
                input_channel=1,
                parent_cache_dir=pathlib.Path("."),
            )
        ]

        with self.assertLogs(ballroom_scribe.gui.pipeline_runner.__name__, level=logging.INFO) as logs:
            runner.start(
                inputs=inputs,
                output_csv=pathlib.Path("clip.csv"),
                extra_args=["--ollama-model", "a model"],
            )

        self.assertEqual(len(logs.output), 1)
        self.assertIn("find_timestamps.py -i clip.mp4 1 -c . --output-csv clip.csv", logs.output[0])
        # An argument with a space in it is quoted, so that the logged command can be run as it is.
        self.assertIn("--ollama-model 'a model'", logs.output[0])


class TestExitDescription(unittest.TestCase):
    """Unit tests for `PipelineRunner.exit_description` with a mocked `QProcess`."""

    def setUp(self):
        self.mock_qprocess_cls = self.enterContext(patch.object(QtCore, "QProcess", autospec=True))
        self.mock_process = self.mock_qprocess_cls.return_value
        self.mock_process.readyReadStandardOutput = MagicMock()
        self.mock_process.readyReadStandardError = MagicMock()
        self.mock_process.finished = MagicMock()
        self.mock_process.errorOccurred = MagicMock()
        self.mock_process.state.return_value = QtCore.QProcess.ProcessState.NotRunning
        self.mock_process.readAllStandardOutput.return_value = _make_byte_array(b"")

        self.runner = ballroom_scribe.gui.PipelineRunner()
        self.exit_codes: list[int] = []
        self.runner.finished.connect(self.exit_codes.append)

    def _patch_exit_code_of_a_crash_is_a_signal(self, value: bool) -> None:
        self.enterContext(patch.object(ballroom_scribe.gui.pipeline_runner, "_EXIT_CODE_OF_A_CRASH_IS_A_SIGNAL", value))

    def test_is_empty_until_a_child_has_ended(self):
        self.assertEqual(self.runner.exit_description, "")

    def test_child_that_exited_by_itself(self):
        self.runner._on_finished(1, QtCore.QProcess.ExitStatus.NormalExit)

        self.assertEqual(self.runner.exit_description, "exited with status 1")
        self.assertEqual(self.exit_codes, [1])

    def test_child_that_a_signal_ended(self):
        self._patch_exit_code_of_a_crash_is_a_signal(True)
        segmentation_fault = int(signal.SIGSEGV)

        self.runner._on_finished(segmentation_fault, QtCore.QProcess.ExitStatus.CrashExit)

        self.assertEqual(self.runner.exit_description, f"was ended by signal {segmentation_fault} (SIGSEGV)")
        self.assertEqual(self.exit_codes, [-1])

    @parameterized.expand(
        [
            # `QProcess.exitCode` is a signed 32-bit number, so it reports `0xC0000005` as a negative number.
            param(
                "access_violation",
                exit_code=-1073741819,
                expected="crashed with status 0xC0000005 (access violation)",
            ),
            # The exit status that `QProcess.kill` gives a child on Windows.
            param("killed_by_the_runner", exit_code=0xF291, expected="exited with status 62097"),
        ]
    )
    def test_child_that_crashed_where_there_are_no_signals(self, _, exit_code, expected):
        self._patch_exit_code_of_a_crash_is_a_signal(False)

        self.runner._on_finished(exit_code, QtCore.QProcess.ExitStatus.CrashExit)

        self.assertEqual(self.runner.exit_description, expected)
        self.assertEqual(self.exit_codes, [-1])

    def test_child_that_could_not_be_started(self):
        self.mock_process.errorString.return_value = "execve: No such file or directory"

        self.runner._on_error_occurred(QtCore.QProcess.ProcessError.FailedToStart)

        self.assertEqual(self.runner.exit_description, "could not be started: execve: No such file or directory")
        self.assertEqual(self.exit_codes, [-1])

    def test_is_set_before_finished_is_emitted(self):
        descriptions_when_finished: list[str] = []
        self.runner.finished.connect(lambda _exit_code: descriptions_when_finished.append(self.runner.exit_description))

        self.runner._on_finished(3, QtCore.QProcess.ExitStatus.NormalExit)

        self.assertEqual(descriptions_when_finished, ["exited with status 3"])

    def test_is_empty_again_once_the_next_child_is_started(self):
        self.runner._on_finished(3, QtCore.QProcess.ExitStatus.NormalExit)

        self.runner.start_transcription(
            ballroom_scribe.cli.arguments.InputCacheArgs(
                input_video=_VIDEO_A,
                input_channel=1,
                parent_cache_dir=_CACHE_A,
            ),
            ("python", "-m", "worker"),
        )

        self.assertEqual(self.runner.exit_description, "")


class TestTranscriptionRunner(unittest.TestCase):
    """Unit tests for `PipelineRunner` running the transcription worker, with a mocked `QProcess`."""

    def setUp(self):
        self.mock_qprocess_cls = self.enterContext(patch.object(QtCore, "QProcess", autospec=True))
        self.mock_process = self.mock_qprocess_cls.return_value
        self.mock_process.readyReadStandardOutput = MagicMock()
        self.mock_process.readyReadStandardError = MagicMock()
        self.mock_process.finished = MagicMock()
        self.mock_process.errorOccurred = MagicMock()
        self.mock_process.state.return_value = QtCore.QProcess.ProcessState.NotRunning
        self.mock_process.canReadLine.return_value = False
        self.mock_process.readAllStandardOutput.return_value = _make_byte_array(b"")
        self.mock_process.readAllStandardError.return_value = _make_byte_array(b"")

        self.runner = ballroom_scribe.gui.PipelineRunner(merge_output=False)
        self.outputs: list[str] = []
        self.error_outputs: list[str] = []
        self.exit_codes: list[int] = []
        self.runner.output.connect(self.outputs.append)
        self.runner.error_output.connect(self.error_outputs.append)
        self.runner.finished.connect(self.exit_codes.append)

        self.video_input = ballroom_scribe.cli.arguments.InputCacheArgs(
            input_video=_VIDEO_A,
            input_channel=1,
            parent_cache_dir=_CACHE_A,
        )

    def test_keeps_standard_output_apart_from_standard_error(self):
        self.mock_process.setProcessChannelMode.assert_called_once_with(
            QtCore.QProcess.ProcessChannelMode.SeparateChannels
        )

    def test_start_transcription_runs_the_worker_on_the_video(self):
        self.runner.start_transcription(self.video_input, ("python", "-m", "worker"))

        self.mock_process.start.assert_called_once_with(
            "python",
            ["-m", "worker", "-i", str(_VIDEO_A), "1", "-c", str(_CACHE_A)],
        )

    def test_start_transcription_refuses_while_a_worker_is_running(self):
        self.mock_process.state.return_value = QtCore.QProcess.ProcessState.Running

        with self.assertRaises(RuntimeError):
            self.runner.start_transcription(self.video_input, ("python", "-m", "worker"))

        self.mock_process.start.assert_not_called()

    def test_standard_output_arrives_one_complete_line_at_a_time(self):
        # Two complete lines have arrived, one of them with a Windows line ending.
        self.mock_process.canReadLine.side_effect = [True, True, False]
        self.mock_process.readLine.side_effect = [_make_byte_array(b'{"n": 1}\n'), _make_byte_array(b'{"n": 2}\r\n')]

        self.runner._on_ready_read()

        self.assertEqual(self.outputs, ['{"n": 1}', '{"n": 2}'])
        # Whatever follows the last line ending stays in the process until its line is complete.
        self.mock_process.readAllStandardOutput.assert_not_called()

    def test_unfinished_line_is_dropped_when_the_worker_exits(self):
        self.mock_process.canReadLine.side_effect = [True, False]
        self.mock_process.readLine.side_effect = [_make_byte_array(b'{"n": 1}\n')]
        self.mock_process.readAllStandardOutput.return_value = _make_byte_array(b'{"n": 2')

        self.runner._on_finished(9, QtCore.QProcess.ExitStatus.CrashExit)

        self.assertEqual(self.outputs, ['{"n": 1}'])
        self.mock_process.readAllStandardOutput.assert_called_once_with()
        self.assertEqual(self.exit_codes, [-1])

    def test_standard_error_arrives_separately(self):
        self.mock_process.readAllStandardError.return_value = _make_byte_array(b"Checking input videos...\n")

        self.runner._on_ready_read_error()

        self.assertEqual(self.error_outputs, ["Checking input videos...\n"])
        self.assertEqual(self.outputs, [])

    def test_remaining_standard_error_is_delivered_before_the_exit_status(self):
        self.mock_process.readAllStandardError.return_value = _make_byte_array(b"OSError: download failed\n")
        events: list[str | int] = []
        self.runner.error_output.connect(events.append)
        self.runner.finished.connect(events.append)

        self.runner._on_finished(1, QtCore.QProcess.ExitStatus.NormalExit)

        self.assertEqual(events, ["OSError: download failed\n", 1])

    def test_shutdown_kills_the_worker_and_waits_for_it_to_exit(self):
        self.mock_process.state.return_value = QtCore.QProcess.ProcessState.Running
        self.mock_process.waitForFinished.return_value = True

        self.assertTrue(self.runner.shutdown(2.5))

        self.mock_process.kill.assert_called_once_with()
        self.mock_process.waitForFinished.assert_called_once_with(2500)

    def test_shutdown_reports_a_worker_that_does_not_exit_in_time(self):
        self.mock_process.state.return_value = QtCore.QProcess.ProcessState.Running
        self.mock_process.waitForFinished.return_value = False

        self.assertFalse(self.runner.shutdown(0.5))

    def test_shutdown_without_a_running_worker_does_nothing(self):
        self.assertTrue(self.runner.shutdown(2.5))

        self.mock_process.kill.assert_not_called()
        self.mock_process.waitForFinished.assert_not_called()


if __name__ == "__main__":
    unittest.main()
