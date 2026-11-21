import pathlib
import subprocess
import sys
import unittest
from unittest.mock import patch

from parameterized import param, parameterized

try:
    from PySide6 import QtGui
except ImportError as err:
    raise unittest.SkipTest("PySide6 is not installed.") from err

import ballroom_scribe.gui.utils

_REPORT = pathlib.Path("/logs/20261121-091645-4242/problem report.txt")
_WINDOWS_REPORT = pathlib.Path(r"C:\Users\dancer\AppData\Local\Ballroom Scribe\Logs\problem report.txt")


class TestRevealFile(unittest.TestCase):
    """Unit tests for the `reveal_file` function."""

    def setUp(self):
        self.mock_subprocess_run = self.enterContext(patch.object(subprocess, "run", autospec=True))
        self.mock_open_url = self.enterContext(patch.object(QtGui.QDesktopServices, "openUrl", autospec=True))

    def _opened_folder(self) -> pathlib.Path:
        """Returns the folder that the desktop was asked to open."""
        self.mock_open_url.assert_called_once()
        (url,), _ = self.mock_open_url.call_args
        return pathlib.Path(url.toLocalFile())

    def test_explorer_selects_the_file_on_windows(self):
        self.enterContext(patch.object(sys, "platform", "win32"))

        ballroom_scribe.gui.utils.reveal_file(_WINDOWS_REPORT)

        # One command line, because Explorer wants the path quoted after the comma. Explorer exits with status 1 when
        # it succeeds, so its exit status is not checked.
        self.mock_subprocess_run.assert_called_once_with(
            r'explorer /select,"C:\Users\dancer\AppData\Local\Ballroom Scribe\Logs\problem report.txt"',
            check=False,
        )
        self.mock_open_url.assert_not_called()

    def test_finder_selects_the_file_on_macos(self):
        self.enterContext(patch.object(sys, "platform", "darwin"))

        ballroom_scribe.gui.utils.reveal_file(_REPORT)

        self.mock_subprocess_run.assert_called_once_with(["/usr/bin/open", "-R", str(_REPORT)], check=False)
        self.mock_open_url.assert_not_called()

    def test_folder_of_the_file_is_opened_elsewhere(self):
        self.enterContext(patch.object(sys, "platform", "linux"))

        ballroom_scribe.gui.utils.reveal_file(_REPORT)

        self.assertEqual(self._opened_folder(), _REPORT.parent)
        self.mock_subprocess_run.assert_not_called()

    @parameterized.expand(
        [
            param("explorer", platform="win32"),
            param("finder", platform="darwin"),
        ]
    )
    def test_folder_of_the_file_is_opened_when_the_file_manager_cannot_be_started(self, _, platform):
        self.enterContext(patch.object(sys, "platform", platform))
        self.mock_subprocess_run.side_effect = FileNotFoundError("The file manager is missing")

        ballroom_scribe.gui.utils.reveal_file(_REPORT)

        self.mock_subprocess_run.assert_called_once()
        self.assertEqual(self._opened_folder(), _REPORT.parent)


if __name__ == "__main__":
    unittest.main()
