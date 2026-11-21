"""Supervisor of the timestamp editor: runs the editor as a child process and reports it if it crashes.

A process that has crashed cannot tell anyone. `edit_timestamps.py` therefore runs this supervisor, which starts the
editor itself (`python -m ballroom_scribe.gui.run_editor`) as a child process and outlives it.
"""

import codecs
import contextlib
import ctypes
import io
import os
import pathlib
import signal
import subprocess
import sys
import threading
import traceback
from collections.abc import Callable, Generator, Sequence
from typing import BinaryIO

import ballroom_scribe.gui
import ballroom_scribe.gui.models
import ballroom_scribe.gui.utils

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
# The editor keeps the folder that the user runs it in, so that paths on its command line mean what the user meant.
# `-P` keeps that folder off the editor's import path, which gets the package through `PYTHONPATH` instead (see
# `_editor_environment`). `-u` makes what the editor prints arrive at once and in order.
_EDITOR_COMMAND = (sys.executable, "-P", "-u", "-m", "ballroom_scribe.gui.run_editor")
_READ_BYTES = 65536
# How long to go on reading what the editor wrote after it has exited. A program that the editor had started, such as
# `ffmpeg`, may outlive it and keep the pipe open, which must not hold up the report of a crash for long.
_DRAIN_SECONDS = 2.0
# Requests to end this process, which are passed on to the editor so that it never runs without its supervisor.
_FORWARDED_SIGNALS = tuple(getattr(signal, name) for name in ("SIGTERM", "SIGHUP") if hasattr(signal, name))
# Signals that end the editor because the user or the system asked for it, which is not a crash.
_REQUESTED_SIGNALS = frozenset(
    int(getattr(signal, name)) for name in ("SIGINT", "SIGTERM", "SIGHUP") if hasattr(signal, name)
)
# The exit status of a Windows process that was ended with Ctrl+C.
_WINDOWS_CTRL_C_EXIT = 0xC000013A
_MAX_EXIT_STATUS = 255
# A shell reports a process that signal `n` ended with the exit status `128 + n`, and so does the supervisor.
_SIGNAL_EXIT_STATUS_BASE = 128
_CRASH_HEADLINE = "Ballroom Scribe closed unexpectedly"
_CRASH_MESSAGE_WITH_REPORT = (
    "{headline}\n\n{summary}\n\nA report file has been saved:\n{report_file}\n\nTo report the problem, open a GitHub"
    " issue at {issue_url} and attach the file, or send the file to the maintainer. The file contains file names and"
    " folder paths from this computer, and it may contain speech that was transcribed from your videos, so look"
    " through it before you share it."
)
_CRASH_MESSAGE_WITHOUT_REPORT = "{headline}\n\n{summary}\n\nThe logs of the editor are in:\n{log_directory}"
_MB_ICONERROR = 0x10


def _print(message: str) -> None:
    """Prints to this process's console, if it has one (`pythonw` has none) and it still works.

    The console may be gone by the time the editor ends, a closed terminal for instance. That must not keep the
    supervisor from reporting a crash.
    """
    if sys.stderr is not None:
        with contextlib.suppress(OSError, ValueError):
            print(message, file=sys.stderr, flush=True)


def _editor_environment(session_logs: ballroom_scribe.gui.models.SessionLogs) -> dict[str, str]:
    """Returns the environment of the editor's process: that of this process and what the editor needs to be told."""
    environment = dict(os.environ)
    environment[ballroom_scribe.gui.models.SessionLogs.ENVIRONMENT_VARIABLE] = str(session_logs.directory)
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (str(_REPO_ROOT), environment.get("PYTHONPATH"))))
    # The editor's console is a pipe, for which Python would otherwise choose the encoding of the locale and fail on
    # what that cannot express.
    environment["PYTHONIOENCODING"] = "utf-8:backslashreplace"
    return environment


def _copy_output(output: io.BufferedIOBase, console_log: BinaryIO) -> None:
    """Copies what the editor writes to its console into the session's console log and on to this process's console.

    Runs on a thread of its own until the editor and every program that it started have closed their end of the
    pipe, or until the console log has been closed.
    """
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    console = sys.stderr
    while True:
        try:
            chunk = output.read1(_READ_BYTES)
            if not chunk:
                return
            console_log.write(chunk)
            console_log.flush()
        except OSError, ValueError:
            # The supervisor has stopped waiting for a program that outlived the editor and has closed the log.
            return
        if console is not None:
            try:
                console.write(decoder.decode(chunk))
                console.flush()
            except OSError, ValueError:
                console = None  # This process's console is gone; the log still gets everything.


@contextlib.contextmanager
def _forwarding_termination(process: subprocess.Popen[bytes]) -> Generator[None]:
    """Passes requests to end this process on to the editor for as long as the `with` block runs.

    The editor writes to a pipe that this process reads. If this process went first, the editor would be left with
    a console that fails whatever is written to it.
    """

    def forward(signal_number: int, _frame: object) -> None:
        with contextlib.suppress(OSError):
            process.send_signal(signal_number)

    previous_handlers = {signal_number: signal.signal(signal_number, forward) for signal_number in _FORWARDED_SIGNALS}
    try:
        yield
    finally:
        for signal_number, previous_handler in previous_handlers.items():
            # A handler that was not installed from Python is reported as `None` and cannot be put back as such.
            signal.signal(signal_number, signal.SIG_DFL if previous_handler is None else previous_handler)


def _wait_for(process: subprocess.Popen[bytes]) -> int:
    """Waits until the editor has exited, however often this process is interrupted in the meantime.

    Ctrl+C in the console reaches the editor as well as this process. What it means is for the editor to decide.
    """
    while True:
        try:
            return process.wait()
        except KeyboardInterrupt:
            continue


def _describe_exit(exit_status: int) -> str:
    """Returns how the editor ended, as a phrase that completes "The editor …".

    Args:
        exit_status: What `subprocess.Popen.wait` returned: an exit status, or the negative number of the signal
            that ended the process.
    """
    if exit_status < 0:
        return ballroom_scribe.gui.utils.describe_process_exit(0, signal_number=-exit_status)
    return ballroom_scribe.gui.utils.describe_process_exit(exit_status)


def _was_requested(exit_status: int) -> bool:
    """Returns whether the editor ended because the user or the system asked it to, which is not a crash."""
    if exit_status < 0:
        return -exit_status in _REQUESTED_SIGNALS
    return exit_status == _WINDOWS_CTRL_C_EXIT


def _own_exit_status(exit_status: int) -> int:
    """Returns the exit status with which this process passes on how the editor ended."""
    if exit_status < 0:
        return _SIGNAL_EXIT_STATUS_BASE - exit_status
    return exit_status if exit_status <= _MAX_EXIT_STATUS else 1


def _report_without_dialog(session_logs: ballroom_scribe.gui.models.SessionLogs, summary: str) -> None:
    """Says on the console, and on Windows in a message box, that the editor crashed and where the report file is.

    This is what is left when Qt cannot be loaded, which may be the very reason why the editor did not run.
    """
    try:
        message = _CRASH_MESSAGE_WITH_REPORT.format(
            headline=_CRASH_HEADLINE,
            summary=summary,
            report_file=ballroom_scribe.gui.utils.write_problem_report(session_logs, summary),
            issue_url=ballroom_scribe.gui.utils.build_issue_url("", ""),
        )
    except OSError:
        message = _CRASH_MESSAGE_WITHOUT_REPORT.format(
            headline=_CRASH_HEADLINE,
            summary=summary,
            log_directory=session_logs.directory,
        )
    _print(message)
    if sys.platform == "win32":
        # Without a console (`pythonw`), nothing else would tell the user that anything happened.
        ctypes.windll.user32.MessageBoxW(None, message, _CRASH_HEADLINE, _MB_ICONERROR)


def _show_crash_dialog(session_logs: ballroom_scribe.gui.models.SessionLogs, summary: str) -> None:
    """Tells the user that the editor crashed and offers to report it, in `ProblemReportDialog` if Qt can be loaded.

    Qt is loaded here and nowhere else in the supervisor, so the supervisor stays small while the editor runs.
    """
    try:
        # Qt needs the application object for as long as the dialog is shown, so it is kept until this returns.
        _app = ballroom_scribe.gui.utils.create_application(sys.argv[:1])
        dialog = ballroom_scribe.gui.ProblemReportDialog(session_logs, headline=_CRASH_HEADLINE, summary=summary)
    except Exception:  # noqa: BLE001  # Whatever keeps the dialog from being shown, the user has to be told.
        _print(traceback.format_exc())
        _report_without_dialog(session_logs, summary)
        return
    # The editor's window has just vanished, so the dialog goes where the user is looking.
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    dialog.exec()


def supervise_editor(
    argv: Sequence[str] | None = None,
    *,
    session_logs: ballroom_scribe.gui.models.SessionLogs | None = None,
    editor_command: Sequence[str] = _EDITOR_COMMAND,
    on_crash: Callable[[ballroom_scribe.gui.models.SessionLogs, str], None] = _show_crash_dialog,
) -> int:
    """Runs the editor as a child process, keeps what it writes to its console, and reports it if it crashes.

    Everything that the editor's process writes to its standard output and standard error is copied into the
    session's console log and on to the console of this process, if it has one. That includes what never passes
    through Python's `logging`: Qt's own messages, the output of `ffmpeg`, and tracebacks.

    If the editor ends in any other way than by exiting with status `0` or by a request of the user or the system
    (Ctrl+C, a request to terminate, a closed terminal), `on_crash` is called. An editor that dies after its window
    was closed in good order is not reported, because the user lost nothing; the console log says what happened.

    Nothing here imports Qt unless the editor has crashed.

    Args:
        argv: The command-line arguments of the editor, or `None` for those of the process. `--help` and mistakes
            in them end the process here, before the editor is started.
        session_logs: Where the session keeps its logs, or `None` to make a folder for them (see
            `current_session_logs`). The editor is told to use the same folder.
        editor_command: The program that runs the editor, followed by its leading arguments. `argv` is appended.
        on_crash: Called with `session_logs` and a sentence that says how the editor ended, once the console log
            is complete. By default it shows `ProblemReportDialog`.

    Returns:
        The exit status for this process: that of the editor, or `128 + n` if signal `n` ended the editor, or `1`
        if the editor's exit status is larger than an exit status can be everywhere.
    """
    arguments = sys.argv[1:] if argv is None else list(argv)
    ballroom_scribe.gui.parse_editor_args(arguments)
    if session_logs is None:
        session_logs = ballroom_scribe.gui.utils.current_session_logs()
    session_logs.directory.mkdir(parents=True, exist_ok=True)

    with open(session_logs.console_log, "ab") as console_log:
        process = subprocess.Popen(
            [*editor_command, *arguments],
            stdin=subprocess.DEVNULL,
            # Both streams go down one pipe, so that what the editor and its libraries write stays in order.
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=_editor_environment(session_logs),
        )
        copier = threading.Thread(
            target=_copy_output,
            args=(process.stdout, console_log),
            name="editor-console",
            daemon=True,
        )
        copier.start()
        with _forwarding_termination(process):
            exit_status = _wait_for(process)
        copier.join(_DRAIN_SECONDS)
        if exit_status == 0:
            return 0
        closed_normally = session_logs.closed_normally_marker.exists()
        description = _describe_exit(exit_status)
        note = f"The editor {description}{' after its window had been closed' if closed_normally else ''}."
        console_log.write(f"\n{note}\n".encode())
        _print(note)

    # The console log is complete and closed by now, so a report can take it in.
    if not closed_normally and not _was_requested(exit_status):
        on_crash(session_logs, f"The editor {description}.")
    return _own_exit_status(exit_status)
