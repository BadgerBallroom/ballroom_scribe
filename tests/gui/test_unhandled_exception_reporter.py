import logging
import sys
import unittest
from types import TracebackType
from unittest.mock import patch

try:
    import PySide6  # noqa: F401
except ImportError as err:
    raise unittest.SkipTest("PySide6 is not installed.") from err

import ballroom_scribe.gui
import ballroom_scribe.gui.unhandled_exception_reporter

type _ExceptionInfo = tuple[type[BaseException], BaseException, TracebackType | None]


def _raised_in_a_paint_handler() -> _ExceptionInfo:
    """Returns what `sys.excepthook` gets when this line raises; every call raises from the same line."""
    try:
        raise RuntimeError("The envelope has no bins")
    except RuntimeError as err:
        return type(err), err, err.__traceback__


def _raised_in_a_slot() -> _ExceptionInfo:
    """Returns what `sys.excepthook` gets when another line raises the same type of exception."""
    try:
        raise RuntimeError("The table has no rows")
    except RuntimeError as err:
        return type(err), err, err.__traceback__


class TestUnhandledExceptionReporter(unittest.TestCase):
    """Unit tests for `UnhandledExceptionReporter`."""

    def setUp(self):
        self.reporter = ballroom_scribe.gui.UnhandledExceptionReporter()
        self.announced: list[str] = []
        self.reporter.exception_reported.connect(self.announced.append)
        self.logger_name = ballroom_scribe.gui.unhandled_exception_reporter.__name__

    def test_logs_the_traceback_and_announces_the_exception(self):
        with self.assertLogs(self.logger_name, level=logging.ERROR) as logs:
            self.reporter.handle_exception(*_raised_in_a_paint_handler())

        self.assertEqual(self.announced, ["RuntimeError: The envelope has no bins"])
        self.assertEqual(len(logs.records), 1)
        self.assertIn("Traceback (most recent call last):", logs.output[0])
        self.assertIn("_raised_in_a_paint_handler", logs.output[0])
        self.assertIn("RuntimeError: The envelope has no bins", logs.output[0])

    def test_error_that_is_raised_again_is_announced_once(self):
        with self.assertLogs(self.logger_name, level=logging.ERROR) as logs:
            for _ in range(9):
                self.reporter.handle_exception(*_raised_in_a_paint_handler())

        # A paint handler that fails does so on every repaint: one dialog and one traceback are enough.
        self.assertEqual(self.announced, ["RuntimeError: The envelope has no bins"])
        self.assertEqual(len(logs.records), 1)

    def test_logs_how_often_an_error_was_raised_at_every_power_of_ten(self):
        with self.assertLogs(self.logger_name, level=logging.ERROR) as logs:
            for _ in range(1000):
                self.reporter.handle_exception(*_raised_in_a_paint_handler())

        self.assertEqual(len(self.announced), 1)
        self.assertEqual(
            [record.getMessage() for record in logs.records[1:]],
            [
                "Unhandled RuntimeError again: 10 times so far",
                "Unhandled RuntimeError again: 100 times so far",
                "Unhandled RuntimeError again: 1000 times so far",
            ],
        )
        # Only the first record carries the traceback.
        self.assertEqual([record.exc_info is not None for record in logs.records], [True, False, False, False])

    def test_error_from_another_line_is_another_error(self):
        with self.assertLogs(self.logger_name, level=logging.ERROR) as logs:
            self.reporter.handle_exception(*_raised_in_a_paint_handler())
            self.reporter.handle_exception(*_raised_in_a_slot())
            self.reporter.handle_exception(*_raised_in_a_paint_handler())

        self.assertEqual(
            self.announced,
            ["RuntimeError: The envelope has no bins", "RuntimeError: The table has no rows"],
        )
        self.assertEqual(len(logs.records), 2)

    def test_exception_without_a_traceback(self):
        with self.assertLogs(self.logger_name, level=logging.ERROR):
            self.reporter.handle_exception(ValueError, ValueError("The inpoint is negative"), None)

        self.assertEqual(self.announced, ["ValueError: The inpoint is negative"])

    def test_keyboard_interrupt_is_left_to_python(self):
        mock_default_hook = self.enterContext(patch.object(sys, "__excepthook__", autospec=True))
        interrupt = KeyboardInterrupt()

        with self.assertNoLogs(self.logger_name, level=logging.ERROR):
            self.reporter.handle_exception(KeyboardInterrupt, interrupt, None)

        # Ctrl+C in the console is not an error of the editor.
        mock_default_hook.assert_called_once_with(KeyboardInterrupt, interrupt, None)
        self.assertEqual(self.announced, [])

    def test_install_makes_it_the_handler_of_unhandled_exceptions(self):
        # The patch puts the previous hook back after the test.
        self.enterContext(patch.object(sys, "excepthook", sys.excepthook))

        self.reporter.install()

        self.assertEqual(sys.excepthook, self.reporter.handle_exception)


if __name__ == "__main__":
    unittest.main()
