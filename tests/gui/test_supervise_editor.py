import ctypes
import importlib
import importlib.util
import io
import os
import pathlib
import signal
import subprocess
import sys
import threading
import unittest
from unittest.mock import ANY, MagicMock, call, mock_open, patch

from parameterized import param, parameterized

import ballroom_scribe
import ballroom_scribe.gui
import ballroom_scribe.gui.models
import ballroom_scribe.gui.utils

_REPO_ROOT = pathlib.Path(ballroom_scribe.__file__).resolve().parents[1]
_SESSION_LOGS = ballroom_scribe.gui.models.SessionLogs(pathlib.Path("/logs/20261121-091645-4242"))
_EDITOR_COMMAND = ("python", "-m", "stand_in_editor")
_CRASH_HEADLINE = "Ballroom Scribe closed unexpectedly"
# The exit statuses with which Windows marks a process that was ended with Ctrl+C and one that crashed.
_WINDOWS_CTRL_C_EXIT = 0xC000013A
_WINDOWS_ACCESS_VIOLATION = 0xC0000005


def _signal_cases() -> list[param]:
    """Returns the editors that a signal ended, for the signals that this platform has."""
    cases = [
        param(
            "segmentation_fault",
            exit_status=-int(signal.SIGSEGV),
            reported=f"The editor was ended by signal {int(signal.SIGSEGV)} (SIGSEGV).",
            expected=128 + int(signal.SIGSEGV),
        ),
        param(
            "aborted",
            exit_status=-int(signal.SIGABRT),
            reported=f"The editor was ended by signal {int(signal.SIGABRT)} (SIGABRT).",
            expected=128 + int(signal.SIGABRT),
        ),
        # The user or the system asked for these, so they are not crashes.
        param("interrupted", exit_status=-int(signal.SIGINT), reported=None, expected=128 + int(signal.SIGINT)),
        param("terminated", exit_status=-int(signal.SIGTERM), reported=None, expected=128 + int(signal.SIGTERM)),
    ]
    if hasattr(signal, "SIGKILL"):
        # Also what the system does to a process when memory runs out, and what Force Quit does.
        cases.append(
            param(
                "killed",
                exit_status=-int(signal.SIGKILL),
                reported=f"The editor was ended by signal {int(signal.SIGKILL)} (SIGKILL).",
                expected=128 + int(signal.SIGKILL),
            )
        )
    if hasattr(signal, "SIGHUP"):
        cases.append(
            param("terminal_closed", exit_status=-int(signal.SIGHUP), reported=None, expected=128 + int(signal.SIGHUP))
        )
    return cases


class _SupervisorTestCase(unittest.TestCase):
    """Base of the test cases for `supervise_editor`: mocks the editor's process, its console, and the file system."""

    def setUp(self):
        # Everything that is loaded lazily is resolved before `open` is patched at the end: `lazy_loader` reads the
        # stub of a package with `open` when the package is first imported, and a package whose stub was read
        # through the mock would be left without its exports.
        self.supervise_editor = ballroom_scribe.gui.supervise_editor
        self.mock_parse_editor_args = self.enterContext(
            patch.object(ballroom_scribe.gui, "parse_editor_args", autospec=True)
        )
        self.mock_current_session_logs = self.enterContext(
            patch.object(ballroom_scribe.gui.utils, "current_session_logs", autospec=True, return_value=_SESSION_LOGS)
        )

        self.mock_popen_cls = self.enterContext(patch.object(subprocess, "Popen", autospec=True))
        self.mock_process = self.mock_popen_cls.return_value
        self.mock_process.stdout = MagicMock()
        self.mock_process.wait.return_value = 0
        self.mock_thread_cls = self.enterContext(patch.object(threading, "Thread", autospec=True))
        self.mock_thread = self.mock_thread_cls.return_value
        self.mock_signal = self.enterContext(patch.object(signal, "signal", autospec=True, return_value=signal.SIG_IGN))
        self.enterContext(patch.dict(os.environ, {"PATH": "/usr/bin"}, clear=True))
        self.mock_stderr = self.enterContext(patch.object(sys, "stderr", new_callable=io.StringIO))
        self.mock_mkdir = self.enterContext(patch.object(pathlib.Path, "mkdir", autospec=True))
        # Whether the editor said that its window was closed in good order.
        self.mock_exists = self.enterContext(patch.object(pathlib.Path, "exists", autospec=True, return_value=False))
        self.reported: list[tuple[ballroom_scribe.gui.models.SessionLogs, str]] = []

        self.mock_open = mock_open()
        self.enterContext(patch("builtins.open", self.mock_open))
        self.mock_console_log = self.mock_open.return_value

    def _supervise(self, argv: list[str] | None = None, **kwargs) -> int:
        """Supervises a stand-in editor, recording what is reported instead of showing a dialog."""
        kwargs.setdefault("session_logs", _SESSION_LOGS)
        kwargs.setdefault("editor_command", _EDITOR_COMMAND)
        kwargs.setdefault("on_crash", lambda session_logs, summary: self.reported.append((session_logs, summary)))
        return self.supervise_editor([] if argv is None else argv, **kwargs)


class TestSuperviseEditor(_SupervisorTestCase):
    """Unit tests for `supervise_editor` with an editor that ends in various ways."""

    def test_runs_the_editor_as_a_child_process_with_the_arguments(self):
        exit_status = self._supervise(["competition.csv", "--verbose"])

        self.assertEqual(exit_status, 0)
        self.mock_parse_editor_args.assert_called_once_with(["competition.csv", "--verbose"])
        # Standard output and standard error go down one pipe, so that what the editor writes stays in order.
        self.mock_popen_cls.assert_called_once_with(
            ["python", "-m", "stand_in_editor", "competition.csv", "--verbose"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=ANY,
        )

    def test_runs_the_editor_module_unless_told_otherwise(self):
        self.supervise_editor([], session_logs=_SESSION_LOGS, on_crash=MagicMock())

        (command,), _ = self.mock_popen_cls.call_args
        # `-P` keeps the folder that the user is in off the editor's import path, and `-u` keeps what it prints in
        # order.
        self.assertEqual(command, [sys.executable, "-P", "-u", "-m", "ballroom_scribe.gui.run_editor"])

    def test_takes_the_arguments_of_the_process_by_default(self):
        self.enterContext(patch.object(sys, "argv", ["edit_timestamps.py", "competition.csv"]))

        self.supervise_editor(session_logs=_SESSION_LOGS, editor_command=_EDITOR_COMMAND, on_crash=MagicMock())

        self.mock_parse_editor_args.assert_called_once_with(["competition.csv"])
        (command,), _ = self.mock_popen_cls.call_args
        self.assertEqual(command, ["python", "-m", "stand_in_editor", "competition.csv"])

    @parameterized.expand(
        [
            param("help", exit_code=0),
            param("mistake", exit_code=2),
        ]
    )
    def test_arguments_are_checked_before_anything_is_started(self, _, exit_code):
        self.mock_parse_editor_args.side_effect = SystemExit(exit_code)

        with self.assertRaises(SystemExit) as context:
            self.supervise_editor(["--help"], editor_command=_EDITOR_COMMAND, on_crash=MagicMock())

        self.assertEqual(context.exception.code, exit_code)
        self.mock_current_session_logs.assert_not_called()
        self.mock_open.assert_not_called()
        self.mock_popen_cls.assert_not_called()

    def test_tells_the_editor_where_the_session_keeps_its_logs(self):
        self._supervise()

        environment = self.mock_popen_cls.call_args.kwargs["env"]
        self.assertEqual(environment["BALLROOM_SCRIBE_SESSION_DIR"], str(_SESSION_LOGS.directory))
        self.mock_current_session_logs.assert_not_called()
        self.mock_mkdir.assert_called_once_with(_SESSION_LOGS.directory, parents=True, exist_ok=True)

    def test_makes_a_folder_for_the_session_unless_given_one(self):
        self.supervise_editor([], editor_command=_EDITOR_COMMAND, on_crash=MagicMock())

        self.mock_current_session_logs.assert_called_once_with()
        environment = self.mock_popen_cls.call_args.kwargs["env"]
        self.assertEqual(environment["BALLROOM_SCRIBE_SESSION_DIR"], str(_SESSION_LOGS.directory))

    def test_editor_finds_the_package_and_writes_utf8(self):
        self._supervise()

        environment = self.mock_popen_cls.call_args.kwargs["env"]
        # The editor stays in the folder that the user is in, so it is told where the package is.
        self.assertEqual(environment["PYTHONPATH"], str(_REPO_ROOT))
        self.assertEqual(environment["PYTHONIOENCODING"], "utf-8:backslashreplace")
        # The rest of the environment is passed on, and that of this process stays as it was.
        self.assertEqual(environment["PATH"], "/usr/bin")
        self.assertEqual(dict(os.environ), {"PATH": "/usr/bin"})

    def test_keeps_the_import_path_that_the_user_set(self):
        os.environ["PYTHONPATH"] = "/somewhere/else"

        self._supervise()

        environment = self.mock_popen_cls.call_args.kwargs["env"]
        self.assertEqual(environment["PYTHONPATH"], f"{_REPO_ROOT}{os.pathsep}/somewhere/else")

    def test_copies_the_console_of_the_editor_on_a_thread_of_its_own(self):
        self._supervise()

        self.mock_open.assert_called_once_with(_SESSION_LOGS.console_log, "ab")
        self.mock_thread_cls.assert_called_once_with(
            target=ANY,
            args=(self.mock_process.stdout, self.mock_console_log),
            name=ANY,
            daemon=True,
        )
        self.assertEqual(self.mock_thread.mock_calls, [call.start(), call.join(ANY)])

    def test_stops_waiting_for_the_rest_of_the_console_after_a_moment(self):
        self._supervise()

        # A program that the editor started may outlive it and keep the pipe open.
        (drain_seconds,), _ = self.mock_thread.join.call_args
        self.assertGreater(drain_seconds, 0.0)
        self.assertLessEqual(drain_seconds, 5.0)

    def test_editor_that_exits_normally_is_not_reported(self):
        exit_status = self._supervise()

        self.assertEqual(exit_status, 0)
        self.assertEqual(self.reported, [])
        # Nothing but what the editor wrote goes into the console log, and nothing is printed.
        self.mock_console_log.write.assert_not_called()
        self.assertEqual(self.mock_stderr.getvalue(), "")
        self.mock_exists.assert_not_called()

    @parameterized.expand(
        [
            param(
                "python_exception",
                exit_status=1,
                reported="The editor exited with status 1.",
                expected=1,
            ),
            param("exit_status", exit_status=3, reported="The editor exited with status 3.", expected=3),
            param(
                "windows_access_violation",
                exit_status=_WINDOWS_ACCESS_VIOLATION,
                reported="The editor crashed with status 0xC0000005 (access violation).",
                # Larger than an exit status can be everywhere.
                expected=1,
            ),
            # The user asked for this, so it is not a crash.
            param("windows_ctrl_c", exit_status=_WINDOWS_CTRL_C_EXIT, reported=None, expected=1),
            *_signal_cases(),
        ]
    )
    def test_reports_an_editor_that_ends_badly(self, _, exit_status, reported, expected):
        self.mock_process.wait.return_value = exit_status

        self.assertEqual(self._supervise(), expected)

        self.assertEqual(self.reported, [] if reported is None else [(_SESSION_LOGS, reported)])

    def test_says_in_the_console_log_and_on_the_console_how_the_editor_ended(self):
        self.mock_process.wait.return_value = 3

        self._supervise()

        self.mock_console_log.write.assert_called_once_with(b"\nThe editor exited with status 3.\n")
        self.assertEqual(self.mock_stderr.getvalue(), "The editor exited with status 3.\n")

    def test_console_log_is_complete_before_the_crash_is_reported(self):
        self.mock_process.wait.return_value = 3
        log_was_closed_when_reported: list[bool] = []

        self._supervise(
            on_crash=lambda session_logs, summary: log_was_closed_when_reported.append(
                self.mock_console_log.__exit__.called
            )
        )

        # A report takes the console log in, so the log has to be written and closed first.
        self.assertEqual(log_was_closed_when_reported, [True])

    def test_editor_that_dies_after_its_window_was_closed_is_not_reported(self):
        self.mock_process.wait.return_value = -int(signal.SIGSEGV)
        self.mock_exists.return_value = True

        exit_status = self._supervise()

        # The user lost nothing, so there is no dialog; the console log says what happened.
        self.assertEqual(self.reported, [])
        self.mock_exists.assert_called_once_with(_SESSION_LOGS.closed_normally_marker)
        (note,), _ = self.mock_console_log.write.call_args
        self.assertEqual(
            note.decode(),
            f"\nThe editor was ended by signal {int(signal.SIGSEGV)} (SIGSEGV) after its window had been closed.\n",
        )
        self.assertEqual(exit_status, 128 + int(signal.SIGSEGV))

    def test_works_without_a_console(self):
        # `pythonw` has no standard error.
        self.enterContext(patch.object(sys, "stderr", None))
        self.mock_process.wait.return_value = 3

        self.assertEqual(self._supervise(), 3)

        self.assertEqual(self.reported, [(_SESSION_LOGS, "The editor exited with status 3.")])

    def test_reports_a_crash_although_the_console_is_gone(self):
        # The terminal that the editor was started from may have been closed since.
        mock_console = self.enterContext(patch.object(sys, "stderr", MagicMock()))
        mock_console.write.side_effect = OSError("Input/output error")
        self.mock_process.wait.return_value = 3

        self.assertEqual(self._supervise(), 3)

        self.assertEqual(self.reported, [(_SESSION_LOGS, "The editor exited with status 3.")])

    def test_keeps_waiting_when_interrupted(self):
        # Ctrl+C in the console reaches this process as well as the editor, which decides what it means.
        self.mock_process.wait.side_effect = [KeyboardInterrupt, KeyboardInterrupt, 0]

        self.assertEqual(self._supervise(), 0)

        self.assertEqual(self.mock_process.wait.call_count, 3)
        self.assertEqual(self.reported, [])

    def test_passes_a_request_to_terminate_on_to_the_editor(self):
        def wait() -> int:
            # This process is asked to terminate while it waits for the editor.
            (_, handler), _ = self.mock_signal.call_args_list[0]
            handler(signal.SIGTERM, None)
            return -int(signal.SIGTERM)

        self.mock_process.wait.side_effect = wait

        exit_status = self._supervise()

        # The editor never runs without its supervisor: it would be left with a console that fails every write.
        self.assertEqual(self.mock_signal.call_args_list[0], call(signal.SIGTERM, ANY))
        self.mock_process.send_signal.assert_called_once_with(signal.SIGTERM)
        self.assertEqual(self.reported, [])
        self.assertEqual(exit_status, 128 + int(signal.SIGTERM))

    def test_request_to_terminate_an_editor_that_is_gone_already_is_ignored(self):
        self.mock_process.send_signal.side_effect = ProcessLookupError("The editor has exited")

        def wait() -> int:
            (_, handler), _ = self.mock_signal.call_args_list[0]
            handler(signal.SIGTERM, None)
            return 0

        self.mock_process.wait.side_effect = wait

        self.assertEqual(self._supervise(), 0)

    @parameterized.expand(
        [
            param("python_handler", previous=signal.SIG_IGN, restored=signal.SIG_IGN),
            # `signal.signal` reports a handler that was not installed from Python as `None`.
            param("handler_of_a_library", previous=None, restored=signal.SIG_DFL),
        ]
    )
    def test_puts_the_signal_handlers_back_once_the_editor_has_exited(self, _, previous, restored):
        self.mock_signal.return_value = previous

        self._supervise()

        installed, restorations = (
            self.mock_signal.call_args_list[: len(self.mock_signal.call_args_list) // 2],
            self.mock_signal.call_args_list[len(self.mock_signal.call_args_list) // 2 :],
        )
        self.assertGreaterEqual(len(installed), 1)
        self.assertEqual(restorations, [call(signal_number, restored) for (signal_number, _), _ in installed])


class TestCopyConsole(_SupervisorTestCase):
    """Unit tests for what the supervisor's thread does with what the editor writes to its console."""

    def setUp(self):
        super().setUp()
        self._supervise()
        self.copy_output = self.mock_thread_cls.call_args.kwargs["target"]
        self.output = MagicMock()
        self.console_log = MagicMock()

    def test_copies_into_the_console_log_and_on_to_the_console(self):
        self.output.read1.side_effect = [b"Loading the video\n", b"Traceback (most recent call last):\n", b""]

        self.copy_output(self.output, self.console_log)

        # Every chunk is written and flushed at once, so that the log is current if this process dies.
        self.assertEqual(
            self.console_log.mock_calls,
            [
                call.write(b"Loading the video\n"),
                call.flush(),
                call.write(b"Traceback (most recent call last):\n"),
                call.flush(),
            ],
        )
        self.assertEqual(self.mock_stderr.getvalue(), "Loading the video\nTraceback (most recent call last):\n")

    def test_character_that_arrives_in_two_chunks_is_printed_whole(self):
        self.output.read1.side_effect = [b"Caf\xc3", b"\xa9 Dvo\xc5\x99\xc3\xa1k\n", b""]

        self.copy_output(self.output, self.console_log)

        self.assertEqual(self.mock_stderr.getvalue(), "Café Dvořák\n")
        # The log gets the bytes as they are.
        self.assertEqual(
            [recorded.args[0] for recorded in self.console_log.write.call_args_list],
            [b"Caf\xc3", b"\xa9 Dvo\xc5\x99\xc3\xa1k\n"],
        )

    def test_bytes_that_are_not_utf8_do_not_stop_the_copy(self):
        self.output.read1.side_effect = [b"Vid\xe9o\n", b"more\n", b""]

        self.copy_output(self.output, self.console_log)

        self.assertEqual(self.mock_stderr.getvalue(), "Vid\ufffdo\nmore\n")
        self.assertEqual(self.console_log.write.call_count, 2)

    def test_copies_into_the_console_log_without_a_console(self):
        self.enterContext(patch.object(sys, "stderr", None))
        self.output.read1.side_effect = [b"Loading the video\n", b""]

        self.copy_output(self.output, self.console_log)

        self.console_log.write.assert_called_once_with(b"Loading the video\n")

    def test_goes_on_copying_into_the_console_log_when_the_console_is_gone(self):
        mock_console = self.enterContext(patch.object(sys, "stderr", MagicMock()))
        mock_console.write.side_effect = OSError("The console was closed")
        self.output.read1.side_effect = [b"one\n", b"two\n", b""]

        self.copy_output(self.output, self.console_log)

        self.assertEqual(
            [recorded.args[0] for recorded in self.console_log.write.call_args_list],
            [b"one\n", b"two\n"],
        )
        # The console is not tried again.
        mock_console.write.assert_called_once()

    def test_stops_once_the_console_log_has_been_closed(self):
        # The supervisor closed the log while a program that the editor had started still held the pipe open.
        self.console_log.write.side_effect = ValueError("I/O operation on closed file")
        self.output.read1.side_effect = [b"late output of ffmpeg\n", b"more\n", b""]

        self.copy_output(self.output, self.console_log)

        self.output.read1.assert_called_once()
        self.assertEqual(self.mock_stderr.getvalue(), "")


class TestCrashReport(_SupervisorTestCase):
    """Unit tests for what `supervise_editor` does by default when the editor has crashed."""

    def setUp(self):
        if importlib.util.find_spec("PySide6") is None:
            self.skipTest("PySide6 is not installed.")
        # Patched before the base class patches `open` (see there): patching resolves what is loaded lazily.
        self.mock_create_application = self.enterContext(
            patch.object(ballroom_scribe.gui.utils, "create_application", autospec=True)
        )
        self.mock_dialog_cls = self.enterContext(
            patch.object(ballroom_scribe.gui, "ProblemReportDialog", autospec=True)
        )
        self.mock_dialog = self.mock_dialog_cls.return_value
        self.mock_write_problem_report = self.enterContext(
            patch.object(
                ballroom_scribe.gui.utils,
                "write_problem_report",
                autospec=True,
                return_value=_SESSION_LOGS.directory / "problem report.txt",
            )
        )
        self.mock_windll = self.enterContext(patch.object(ctypes, "windll", MagicMock(), create=True))
        self.enterContext(patch.object(sys, "platform", "linux"))
        self.enterContext(patch.object(sys, "argv", ["edit_timestamps.py", "competition.csv"]))
        super().setUp()
        self.mock_process.wait.return_value = 3

    def _supervise_with_the_default_report(self) -> int:
        return self.supervise_editor([], session_logs=_SESSION_LOGS, editor_command=_EDITOR_COMMAND)

    def test_offers_to_report_the_crash_in_a_dialog(self):
        manager = MagicMock()
        manager.attach_mock(self.mock_create_application, "create_application")
        manager.attach_mock(self.mock_dialog_cls, "dialog")

        exit_status = self._supervise_with_the_default_report()

        self.assertEqual(exit_status, 3)
        self.assertEqual(
            manager.mock_calls,
            [
                # Qt is given the name of the program only: the arguments were the editor's.
                call.create_application(["edit_timestamps.py"]),
                call.dialog(_SESSION_LOGS, headline=_CRASH_HEADLINE, summary="The editor exited with status 3."),
                # The editor's window has just vanished, so the dialog is brought to where the user is looking.
                call.dialog().show(),
                call.dialog().raise_(),
                call.dialog().activateWindow(),
                call.dialog().exec(),
            ],
        )
        self.mock_windll.user32.MessageBoxW.assert_not_called()

    def test_does_not_load_qt_for_an_editor_that_exits_normally(self):
        self.mock_process.wait.return_value = 0

        self.assertEqual(self._supervise_with_the_default_report(), 0)

        self.mock_create_application.assert_not_called()
        self.mock_dialog_cls.assert_not_called()

    def test_says_on_the_console_where_the_report_is_when_qt_cannot_be_loaded(self):
        # This may be the very reason why the editor did not run.
        self.mock_create_application.side_effect = ImportError("No module named 'PySide6'")

        exit_status = self._supervise_with_the_default_report()

        self.assertEqual(exit_status, 3)
        self.mock_dialog_cls.assert_not_called()
        self.mock_write_problem_report.assert_called_once_with(_SESSION_LOGS, "The editor exited with status 3.")
        printed = self.mock_stderr.getvalue()
        self.assertIn("ImportError: No module named 'PySide6'", printed)
        self.assertIn(_CRASH_HEADLINE, printed)
        self.assertIn("The editor exited with status 3.", printed)
        self.assertIn(str(_SESSION_LOGS.directory / "problem report.txt"), printed)
        self.assertIn("https://github.com/BadgerBallroom/ballroom_scribe/issues/new", printed)
        # The user is told what the file contains before sharing it.
        self.assertIn("may contain speech", printed)
        self.mock_windll.user32.MessageBoxW.assert_not_called()

    def test_shows_a_message_box_on_windows_when_qt_cannot_be_loaded(self):
        self.enterContext(patch.object(sys, "platform", "win32"))
        # `pythonw` has no console, so the message box is all that tells the user that anything happened.
        self.enterContext(patch.object(sys, "stderr", None))
        self.mock_create_application.side_effect = ImportError("No module named 'PySide6'")

        self._supervise_with_the_default_report()

        self.mock_windll.user32.MessageBoxW.assert_called_once_with(None, ANY, _CRASH_HEADLINE, 0x10)
        message = self.mock_windll.user32.MessageBoxW.call_args.args[1]
        self.assertIn("The editor exited with status 3.", message)
        self.assertIn(str(_SESSION_LOGS.directory / "problem report.txt"), message)

    def test_says_where_the_logs_are_when_the_dialog_cannot_be_created(self):
        self.mock_dialog_cls.side_effect = RuntimeError("The dialog could not be created")
        self.mock_write_problem_report.side_effect = PermissionError("The folder of the session is read-only")

        self._supervise_with_the_default_report()

        printed = self.mock_stderr.getvalue()
        self.assertIn("RuntimeError: The dialog could not be created", printed)
        self.assertIn("The editor exited with status 3.", printed)
        self.assertIn(f"The logs of the editor are in:\n{_SESSION_LOGS.directory}", printed)


class TestSupervisorImportFootprint(unittest.TestCase):
    """Verifies that supervising an editor never imports Qt, up to the moment at which a crash is reported.

    The supervisor has to stay small and has to keep running when Qt is what fails.
    """

    def test_supervising_an_editor_that_crashes_does_not_load_qt(self):
        clean_env = {k: v for k, v in os.environ.items() if k != "EAGER_IMPORT"}
        self.enterContext(patch.dict(os.environ, clean_env, clear=True))
        self.enterContext(patch.dict(sys.modules, dict(sys.modules), clear=True))
        # Remove `ballroom_scribe` so that the supervisor's import chain runs from scratch, and make importing
        # anything of Qt fail.
        for name in list(sys.modules):
            if name == "ballroom_scribe" or name.startswith(("ballroom_scribe.", "PySide6.")):
                del sys.modules[name]
        sys.modules["PySide6"] = None
        gui_package = importlib.import_module("ballroom_scribe.gui")
        gui_models_package = importlib.import_module("ballroom_scribe.gui.models")
        session_logs = gui_models_package.SessionLogs(pathlib.Path("/logs/20261121-091645-4242"))
        # Everything that is loaded lazily is resolved before `open` is patched (see `_SupervisorTestCase.setUp`).
        supervise_editor = gui_package.supervise_editor
        gui_package.parse_editor_args([])

        mock_popen_cls = self.enterContext(patch.object(subprocess, "Popen", autospec=True))
        mock_popen_cls.return_value.stdout = MagicMock()
        mock_popen_cls.return_value.wait.return_value = 3
        self.enterContext(patch.object(threading, "Thread", autospec=True))
        self.enterContext(patch.object(signal, "signal", autospec=True))
        self.enterContext(patch.object(sys, "stderr", new_callable=io.StringIO))
        self.enterContext(patch.object(pathlib.Path, "mkdir", autospec=True))
        self.enterContext(patch.object(pathlib.Path, "exists", autospec=True, return_value=False))
        self.enterContext(patch("builtins.open", mock_open()))
        reported: list[str] = []

        exit_status = supervise_editor(
            ["competition.csv", "--verbose"],
            session_logs=session_logs,
            on_crash=lambda _session_logs, summary: reported.append(summary),
        )

        self.assertEqual(exit_status, 3)
        self.assertEqual(reported, ["The editor exited with status 3."])
        loaded_qt_modules = [name for name in sys.modules if name.startswith("PySide6.")]
        self.assertEqual(loaded_qt_modules, [], "The supervisor of the editor imported Qt.")


if __name__ == "__main__":
    unittest.main()
