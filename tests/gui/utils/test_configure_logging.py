import logging
import os
import unittest
from unittest.mock import patch

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
