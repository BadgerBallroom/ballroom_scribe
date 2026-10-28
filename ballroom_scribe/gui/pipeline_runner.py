"""Out-of-process runner for `find_timestamps.py` using `QProcess`."""

import pathlib
import sys
from collections.abc import Sequence

from PySide6 import QtCore

import ballroom_scribe.cli.arguments

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_FIND_TIMESTAMPS_SCRIPT = _REPO_ROOT / "find_timestamps.py"


def _build_command(
    inputs: Sequence[ballroom_scribe.cli.arguments.InputCacheArgs],
    output_csv: pathlib.Path,
    extra_args: Sequence[str] = (),
) -> list[str]:
    """Constructs the command line to invoke `find_timestamps.py` in a child Python process."""
    if not inputs:
        raise ValueError("At least one input video is required to run the pipeline.")

    command = [sys.executable, "-u", str(_FIND_TIMESTAMPS_SCRIPT)]
    for video_input in inputs:
        command.extend(
            (
                "-i",
                str(video_input["input_video"]),
                str(video_input["input_channel"]),
                "-c",
                str(video_input["cache_dir"]),
            )
        )
    command.extend(("--output-csv", str(output_csv)))
    command.extend(extra_args)
    return command


class PipelineRunner(QtCore.QObject):
    """Executes `find_timestamps.py` asynchronously via `QProcess` and streams merged output."""

    output = QtCore.Signal(str)
    finished = QtCore.Signal(int)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._process = QtCore.QProcess(self)
        self._process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        self._process.setWorkingDirectory(str(_REPO_ROOT))

        environment = QtCore.QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUTF8", "1")
        self._process.setProcessEnvironment(environment)

        self._process.readyReadStandardOutput.connect(self._on_ready_read)
        self._process.finished.connect(self._on_finished)

    def is_running(self) -> bool:
        """Returns whether the child pipeline process is currently running."""
        return self._process.state() != QtCore.QProcess.ProcessState.NotRunning

    def start(
        self,
        inputs: Sequence[ballroom_scribe.cli.arguments.InputCacheArgs],
        output_csv: pathlib.Path,
        extra_args: Sequence[str] = (),
    ) -> None:
        """Starts `find_timestamps.py` with the given video inputs, output CSV, and extra CLI flags."""
        if self.is_running():
            raise RuntimeError("Pipeline is already running.")
        command = _build_command(inputs=inputs, output_csv=output_csv, extra_args=extra_args)
        self._process.start(command[0], command[1:])

    def cancel(self) -> None:
        """Terminates the running child pipeline process immediately."""
        if self.is_running():
            self._process.kill()

    def _on_ready_read(self) -> None:
        raw_data = self._process.readAllStandardOutput().data()
        if raw_data:
            self.output.emit(raw_data.decode("utf-8", errors="replace"))

    def _on_finished(self, exit_code: int, exit_status: QtCore.QProcess.ExitStatus) -> None:
        self._on_ready_read()
        effective_code = exit_code if exit_status == QtCore.QProcess.ExitStatus.NormalExit else -1
        self.finished.emit(effective_code)
