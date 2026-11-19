"""Out-of-process runner for `find_timestamps.py` and the transcription worker using `QProcess`."""

import pathlib
import sys
from collections.abc import Sequence

from PySide6 import QtCore

import ballroom_scribe.cli.arguments

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_FIND_TIMESTAMPS_SCRIPT = _REPO_ROOT / "find_timestamps.py"
_MILLISECONDS_PER_SECOND = 1000


def _input_arguments(video_input: ballroom_scribe.cli.arguments.InputCacheArgs) -> list[str]:
    """Returns the `-i VIDEO CHANNEL -c DIR` command-line arguments that name one input video."""
    return [
        "-i",
        str(video_input["input_video"]),
        str(video_input["input_channel"]),
        "-c",
        str(video_input["parent_cache_dir"]),
    ]


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
        command.extend(_input_arguments(video_input))
    command.extend(("--output-csv", str(output_csv)))
    command.extend(extra_args)
    return command


def _build_transcription_command(
    video_input: ballroom_scribe.cli.arguments.InputCacheArgs,
    command: Sequence[str],
) -> list[str]:
    """Constructs the command line that runs the transcription worker `command` on one video."""
    if not command:
        raise ValueError("The command of the transcription worker must name a program.")
    return [*command, *_input_arguments(video_input)]


class PipelineRunner(QtCore.QObject):
    """Runs `find_timestamps.py` or the transcription worker in a child process via `QProcess` and streams its output.

    By default the child's standard output and standard error are merged, and `output` carries them as they arrive,
    which suits showing the progress of `find_timestamps.py`. With `merge_output=False`, `output` carries standard
    output one complete line at a time, without its line ending, and `error_output` carries standard error as it
    arrives. That suits the transcription worker, whose standard output holds one transcript chunk per line.

    `finished` carries the child's exit status, or `-1` if it was killed, crashed, or could not be started.
    """

    output = QtCore.Signal(str)
    error_output = QtCore.Signal(str)
    finished = QtCore.Signal(int)

    def __init__(self, parent: QtCore.QObject | None = None, *, merge_output: bool = True) -> None:
        """Initializes the instance.

        Args:
            parent: The Qt parent object.
            merge_output: Whether `output` carries standard output and standard error merged, as they arrive. If
                `False`, `output` carries the complete lines of standard output and `error_output` carries standard
                error.
        """
        super().__init__(parent)
        self._merge_output = merge_output
        self._process = QtCore.QProcess(self)
        self._process.setProcessChannelMode(
            QtCore.QProcess.ProcessChannelMode.MergedChannels
            if merge_output
            else QtCore.QProcess.ProcessChannelMode.SeparateChannels
        )
        self._process.setWorkingDirectory(str(_REPO_ROOT))

        environment = QtCore.QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUTF8", "1")
        self._process.setProcessEnvironment(environment)

        self._process.readyReadStandardOutput.connect(self._on_ready_read)
        self._process.readyReadStandardError.connect(self._on_ready_read_error)
        self._process.finished.connect(self._on_finished)
        self._process.errorOccurred.connect(self._on_error_occurred)

    def is_running(self) -> bool:
        """Returns whether the child process is currently running."""
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

    def start_transcription(
        self,
        video_input: ballroom_scribe.cli.arguments.InputCacheArgs,
        command: Sequence[str],
    ) -> None:
        """Starts the transcription worker on one video.

        Args:
            video_input: The video's path, its audio channel, and the parent directory of its cache directory. They
                are appended to `command` as `-i VIDEO CHANNEL -c DIR`.
            command: The program that runs the worker, followed by its leading arguments.
        """
        if self.is_running():
            raise RuntimeError("The transcription worker is already running.")
        full_command = _build_transcription_command(video_input, command)
        self._process.start(full_command[0], full_command[1:])

    def cancel(self) -> None:
        """Terminates the running child process immediately, without waiting for it to exit."""
        if self.is_running():
            self._process.kill()

    def shutdown(self, timeout_seconds: float) -> bool:
        """Kills the child process, if it is running, and waits for it to exit.

        `finished` is emitted before this returns. Use this instead of `cancel` when something else is about to use
        what the child was using, such as its cache files.

        Args:
            timeout_seconds: The longest time to wait for the child to exit.

        Returns:
            Whether no child process is running any more.
        """
        if not self.is_running():
            return True
        self._process.kill()
        return self._process.waitForFinished(round(timeout_seconds * _MILLISECONDS_PER_SECOND))

    def _on_ready_read(self) -> None:
        if self._merge_output:
            raw_data = self._process.readAllStandardOutput().data()
            if raw_data:
                self.output.emit(raw_data.decode("utf-8", errors="replace"))
            return
        while self._process.canReadLine():
            line = self._process.readLine().data().decode("utf-8", errors="replace")
            self.output.emit(line.rstrip("\r\n"))

    def _on_ready_read_error(self) -> None:
        raw_data = self._process.readAllStandardError().data()
        if raw_data:
            self.error_output.emit(raw_data.decode("utf-8", errors="replace"))

    def _on_finished(self, exit_code: int, exit_status: QtCore.QProcess.ExitStatus) -> None:
        self._on_ready_read()
        if not self._merge_output:
            # A line that the child did not finish writing, because it was killed or it crashed, is not delivered.
            self._process.readAllStandardOutput()
            self._on_ready_read_error()
        effective_code = exit_code if exit_status == QtCore.QProcess.ExitStatus.NormalExit else -1
        self.finished.emit(effective_code)

    def _on_error_occurred(self, error: QtCore.QProcess.ProcessError) -> None:
        # `QProcess` emits no `finished` for a program that could not be started, so report that failure here.
        if error == QtCore.QProcess.ProcessError.FailedToStart:
            self.finished.emit(-1)
