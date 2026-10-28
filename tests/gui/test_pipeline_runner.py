import pathlib
import sys
import unittest
from unittest.mock import MagicMock, patch

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
                cache_dir=_CACHE_A,
            ),
            ballroom_scribe.cli.arguments.InputCacheArgs(
                input_video=_VIDEO_B,
                input_channel=2,
                cache_dir=_CACHE_B,
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


class TestPipelineRunner(unittest.TestCase):
    """Unit tests for `PipelineRunner` with a mocked `QProcess`."""

    def setUp(self):
        self.mock_qprocess_cls = self.enterContext(patch.object(QtCore, "QProcess", autospec=True))
        self.mock_process = self.mock_qprocess_cls.return_value
        self.mock_process.readyReadStandardOutput = MagicMock()
        self.mock_process.finished = MagicMock()
        self.mock_process.state.return_value = QtCore.QProcess.ProcessState.NotRunning
        fake_byte_array = MagicMock()
        fake_byte_array.data.return_value = b"Extracting audio...\n"
        self.mock_process.readAllStandardOutput.return_value = fake_byte_array

    def test_start_cancel_and_signals(self):
        runner = ballroom_scribe.gui.PipelineRunner()
        outputs: list[str] = []
        exit_codes: list[int] = []
        runner.output.connect(outputs.append)
        runner.finished.connect(exit_codes.append)

        inputs = [
            ballroom_scribe.cli.arguments.InputCacheArgs(
                input_video=pathlib.Path("clip.mp4"),
                input_channel=1,
                cache_dir=pathlib.Path("."),
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


if __name__ == "__main__":
    unittest.main()
