import faulthandler
import pathlib
import unittest
from unittest.mock import MagicMock, call, patch

try:
    from PySide6 import QtCore
except ImportError as err:
    raise unittest.SkipTest("PySide6 is not installed.") from err

import ballroom_scribe.cli.arguments
import ballroom_scribe.gui
import ballroom_scribe.gui.models
import ballroom_scribe.gui.utils
import ballroom_scribe.utils

_SESSION_LOGS = ballroom_scribe.gui.models.SessionLogs(pathlib.Path("/logs/20261121-091645-4242"))
_VIDEO_INPUT = ballroom_scribe.cli.arguments.InputCacheArgs(
    input_video=pathlib.Path("part1.mp4"),
    input_channel=1,
    parent_cache_dir=pathlib.Path("."),
)


class TestRunEditor(unittest.TestCase):
    """Unit tests for `run_editor` with the window, the application, and the session's log mocked."""

    def setUp(self):
        self.mock_parse_editor_args = self.enterContext(
            patch.object(ballroom_scribe.gui, "parse_editor_args", autospec=True)
        )
        self._set_arguments()
        self.mock_current_session_logs = self.enterContext(
            patch.object(ballroom_scribe.gui.utils, "current_session_logs", autospec=True, return_value=_SESSION_LOGS)
        )
        self.mock_open_session_log = self.enterContext(
            patch.object(ballroom_scribe.gui.utils, "open_session_log", autospec=True)
        )
        self.mock_session_log = self.mock_open_session_log.return_value
        self.mock_configure_logging = self.enterContext(
            patch.object(ballroom_scribe.gui.utils, "configure_logging", autospec=True)
        )
        self.mock_faulthandler_enable = self.enterContext(patch.object(faulthandler, "enable", autospec=True))
        self.enterContext(
            patch.object(ballroom_scribe.utils, "describe_environment", autospec=True, return_value="Ballroom Scribe")
        )
        self.mock_create_application = self.enterContext(
            patch.object(ballroom_scribe.gui.utils, "create_application", autospec=True)
        )
        self.mock_app = self.mock_create_application.return_value
        self.mock_app.exec.return_value = 0
        self.mock_window_cls = self.enterContext(patch.object(ballroom_scribe.gui, "MainWindow", autospec=True))
        self.mock_window = self.mock_window_cls.return_value
        self.mock_exception_reporter_cls = self.enterContext(
            patch.object(ballroom_scribe.gui, "UnhandledExceptionReporter", autospec=True)
        )
        self.mock_exception_reporter = self.mock_exception_reporter_cls.return_value
        self.mock_exception_reporter.exception_reported = MagicMock()
        self.mock_watchdog_cls = self.enterContext(patch.object(ballroom_scribe.gui, "FreezeWatchdog", autospec=True))
        self.mock_watchdog = self.mock_watchdog_cls.return_value

    def _set_arguments(
        self,
        *,
        inputs: list[ballroom_scribe.cli.arguments.InputCacheArgs] | None = None,
        csv: pathlib.Path | None = None,
        verbose: bool = False,
    ) -> None:
        self.mock_parse_editor_args.return_value = (
            ballroom_scribe.cli.arguments.InputCsvCacheArgs(inputs=inputs or [], csv=csv),
            ballroom_scribe.cli.arguments.LoggingArgs(verbose=verbose),
        )

    def test_logs_to_the_file_of_the_session(self):
        self._set_arguments(verbose=True)

        ballroom_scribe.gui.run_editor(["--verbose"])

        self.mock_parse_editor_args.assert_called_once_with(["--verbose"])
        self.mock_open_session_log.assert_called_once_with(_SESSION_LOGS)
        self.mock_configure_logging.assert_called_once_with(True, self.mock_session_log)

    def test_fatal_errors_go_to_the_file_of_the_session(self):
        ballroom_scribe.gui.run_editor([])

        # The file is named explicitly, because a process without a console has no standard error to write to.
        self.mock_faulthandler_enable.assert_called_once_with(file=self.mock_session_log)

    def test_logging_is_set_up_before_anything_of_qt_is_created(self):
        manager = MagicMock()
        manager.attach_mock(self.mock_configure_logging, "configure_logging")
        manager.attach_mock(self.mock_faulthandler_enable, "faulthandler_enable")
        manager.attach_mock(self.mock_create_application, "create_application")
        manager.attach_mock(self.mock_window_cls, "main_window")

        ballroom_scribe.gui.run_editor([])

        self.assertEqual(
            [name for name, _args, _kwargs in manager.mock_calls if "." not in name][:4],
            ["configure_logging", "faulthandler_enable", "create_application", "main_window"],
        )

    def test_watches_the_gui_thread_while_the_event_loop_runs(self):
        manager = MagicMock()
        manager.attach_mock(self.mock_watchdog, "watchdog")
        manager.attach_mock(self.mock_app, "app")

        ballroom_scribe.gui.run_editor([])

        self.mock_watchdog_cls.assert_called_once_with(self.mock_session_log, self.mock_app)
        self.assertEqual(manager.mock_calls, [call.watchdog.start(), call.app.exec(), call.watchdog.stop()])

    def test_window_can_report_problems_with_the_logs_of_the_session(self):
        ballroom_scribe.gui.run_editor([])

        self.mock_window_cls.assert_called_once_with(session_logs=_SESSION_LOGS)

    def test_window_hears_of_unhandled_exceptions_from_its_event_loop(self):
        manager = MagicMock()
        manager.attach_mock(self.mock_exception_reporter, "exception_reporter")
        manager.attach_mock(self.mock_window_cls, "main_window")
        manager.attach_mock(self.mock_app, "app")

        ballroom_scribe.gui.run_editor([])

        self.mock_exception_reporter_cls.assert_called_once_with(self.mock_app)
        self.assertEqual(
            [recorded for recorded in manager.mock_calls if not recorded[0].startswith("main_window().")],
            [
                # Installed before the window is created: an exception that ends the editor while it starts is
                # logged like any other.
                call.exception_reporter.install(),
                call.main_window(session_logs=_SESSION_LOGS),
                # Queued: the window tells the user after the code that raised the exception has finished failing.
                call.exception_reporter.exception_reported.connect(
                    self.mock_window.report_unhandled_exception,
                    QtCore.Qt.ConnectionType.QueuedConnection,
                ),
                call.app.exec(),
            ],
        )

    def test_opens_what_the_command_line_names(self):
        self._set_arguments(inputs=[_VIDEO_INPUT], csv=pathlib.Path("competition.csv"))

        ballroom_scribe.gui.run_editor(["competition.csv", "-i", "part1.mp4", "1"])

        self.assertEqual(
            self.mock_window.mock_calls,
            [call.show(), call.add_videos([_VIDEO_INPUT]), call.open_csv(pathlib.Path("competition.csv"))],
        )

    def test_opens_nothing_without_arguments(self):
        ballroom_scribe.gui.run_editor([])

        self.assertEqual(self.mock_window.mock_calls, [call.show()])

    def test_returns_the_exit_status_of_the_event_loop(self):
        self.mock_app.exec.return_value = 3

        self.assertEqual(ballroom_scribe.gui.run_editor([]), 3)


if __name__ == "__main__":
    unittest.main()
