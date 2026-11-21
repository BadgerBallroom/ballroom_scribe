import logging
import os
import unittest
from unittest.mock import MagicMock, patch

try:
    from PySide6 import QtCore
except ImportError as err:
    raise unittest.SkipTest("PySide6 is not installed.") from err

import ballroom_scribe.gui.utils


class TestConfigureLogging(unittest.TestCase):
    """Unit tests for `configure_logging`."""

    def setUp(self):
        self.mock_basic_config = self.enterContext(patch.object(logging, "basicConfig", autospec=True))
        self.mock_set_message_pattern = self.enterContext(patch.object(QtCore, "qSetMessagePattern", autospec=True))
        self.mock_set_filter_rules = self.enterContext(
            patch.object(QtCore.QLoggingCategory, "setFilterRules", autospec=True)
        )
        self.enterContext(patch.dict(os.environ, clear=False))
        os.environ.pop("QT_FORCE_STDERR_LOGGING", None)

    def test_default_logs_at_info_and_keeps_qt_category_defaults(self):
        ballroom_scribe.gui.utils.configure_logging(verbose=False)

        self.mock_basic_config.assert_called_once()
        self.assertEqual(self.mock_basic_config.call_args.kwargs["level"], logging.INFO)
        self.mock_set_message_pattern.assert_called_once()
        self.mock_set_filter_rules.assert_not_called()

    def test_verbose_logs_at_debug_and_enables_qt_multimedia_categories(self):
        ballroom_scribe.gui.utils.configure_logging(verbose=True)

        self.mock_basic_config.assert_called_once()
        self.assertEqual(self.mock_basic_config.call_args.kwargs["level"], logging.DEBUG)
        self.mock_set_message_pattern.assert_called_once()
        self.mock_set_filter_rules.assert_called_once()
        rules = self.mock_set_filter_rules.call_args.args[0].splitlines()
        self.assertIn("qt.multimedia.*=true", rules)
        self.assertIn("*.trace=false", rules)

    def test_without_a_session_log_only_stderr_is_logged_to(self):
        ballroom_scribe.gui.utils.configure_logging(verbose=False)

        (console_handler,) = self.mock_basic_config.call_args.kwargs["handlers"]
        self.assertIsInstance(console_handler, logging.StreamHandler)
        self.assertEqual(console_handler.level, logging.INFO)

    def test_session_log_gets_every_level_while_stderr_keeps_its_own(self):
        session_log = MagicMock()

        ballroom_scribe.gui.utils.configure_logging(verbose=False, session_log=session_log)

        console_handler, session_handler = self.mock_basic_config.call_args.kwargs["handlers"]
        # What the log file holds after a failure does not depend on `--verbose`.
        self.assertIs(session_handler.stream, session_log)
        self.assertEqual(session_handler.level, logging.DEBUG)
        self.assertEqual(self.mock_basic_config.call_args.kwargs["level"], logging.DEBUG)
        # What the console shows does.
        self.assertIsNot(console_handler.stream, session_log)
        self.assertEqual(console_handler.level, logging.INFO)
        # Qt's own debug categories stay off, as they do without `--verbose`.
        self.mock_set_filter_rules.assert_not_called()

    def test_verbose_shows_every_level_on_stderr_as_well(self):
        ballroom_scribe.gui.utils.configure_logging(verbose=True, session_log=MagicMock())

        console_handler, session_handler = self.mock_basic_config.call_args.kwargs["handlers"]
        self.assertEqual(console_handler.level, logging.DEBUG)
        self.assertEqual(session_handler.level, logging.DEBUG)

    def test_forces_qt_logging_to_stderr_unless_overridden(self):
        ballroom_scribe.gui.utils.configure_logging(verbose=False)
        self.assertEqual(os.environ["QT_FORCE_STDERR_LOGGING"], "1")

        os.environ["QT_FORCE_STDERR_LOGGING"] = "0"
        ballroom_scribe.gui.utils.configure_logging(verbose=False)
        self.assertEqual(os.environ["QT_FORCE_STDERR_LOGGING"], "0")

    def test_qt_message_pattern_mirrors_python_log_format(self):
        ballroom_scribe.gui.utils.configure_logging(verbose=False)

        pattern = self.mock_set_message_pattern.call_args.args[0]
        for placeholder in ("%{time", "%{if-debug}DEBUG", "%{if-warning}WARNING", "%{category}", "%{message}"):
            self.assertIn(placeholder, pattern)


if __name__ == "__main__":
    unittest.main()
