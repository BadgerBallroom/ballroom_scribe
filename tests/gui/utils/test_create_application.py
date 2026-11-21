import ctypes
import sys
import unittest
from unittest.mock import MagicMock, call, patch

try:
    from PySide6 import QtCore, QtWidgets
except ImportError as err:
    raise unittest.SkipTest("PySide6 is not installed.") from err

import ballroom_scribe.gui.utils

_ARGV = ["edit_timestamps.py", "--verbose"]
_APP_NAME = "Ballroom Scribe Timestamp Editor"


class TestCreateApplication(unittest.TestCase):
    """Unit tests for `create_application`.

    The calls into macOS and Windows themselves are not exercised: only which platform gets which of them.
    """

    def setUp(self):
        self.enterContext(patch.object(sys, "platform", "linux"))
        self.mock_application_cls = self.enterContext(patch.object(QtWidgets, "QApplication", autospec=True))
        self.mock_set_application_name = self.enterContext(
            patch.object(QtCore.QCoreApplication, "setApplicationName", autospec=True)
        )
        self.mock_create_app_icon = self.enterContext(
            patch.object(ballroom_scribe.gui.utils, "create_app_icon", autospec=True)
        )
        # On macOS, loading the system frameworks for real would rename the process that runs the tests.
        self.mock_load_library = self.enterContext(
            patch.object(ctypes.cdll, "LoadLibrary", autospec=True, side_effect=OSError("The framework is missing"))
        )
        self.mock_windll = self.enterContext(patch.object(ctypes, "windll", MagicMock(), create=True))

    def test_creates_the_application_with_the_name_and_the_icon_of_the_editor(self):
        manager = MagicMock()
        manager.attach_mock(self.mock_set_application_name, "set_application_name")
        manager.attach_mock(self.mock_application_cls, "application")

        app = ballroom_scribe.gui.utils.create_application(_ARGV)

        self.assertIs(app, self.mock_application_cls.return_value)
        # Qt takes the name of the application from what was set before the application was created.
        self.assertEqual(
            manager.mock_calls,
            [
                call.set_application_name(_APP_NAME),
                call.application(_ARGV),
                call.application().setWindowIcon(self.mock_create_app_icon.return_value),
            ],
        )

    def test_asks_nothing_of_other_platforms(self):
        ballroom_scribe.gui.utils.create_application(_ARGV)

        self.mock_load_library.assert_not_called()
        self.mock_windll.shell32.SetCurrentProcessExplicitAppUserModelID.assert_not_called()

    def test_gives_the_process_its_own_task_bar_entry_on_windows(self):
        self.enterContext(patch.object(sys, "platform", "win32"))

        ballroom_scribe.gui.utils.create_application(_ARGV)

        self.mock_windll.shell32.SetCurrentProcessExplicitAppUserModelID.assert_called_once_with(
            "ballroom_scribe.timestamp_editor"
        )
        self.mock_load_library.assert_not_called()

    def test_names_the_process_on_macos_and_survives_when_that_fails(self):
        self.enterContext(patch.object(sys, "platform", "darwin"))

        app = ballroom_scribe.gui.utils.create_application(_ARGV)

        self.mock_load_library.assert_called_once_with(
            "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
        )
        # The name in the Dock is a nicety: the application is created without it.
        self.assertIs(app, self.mock_application_cls.return_value)
        self.mock_windll.shell32.SetCurrentProcessExplicitAppUserModelID.assert_not_called()


if __name__ == "__main__":
    unittest.main()
