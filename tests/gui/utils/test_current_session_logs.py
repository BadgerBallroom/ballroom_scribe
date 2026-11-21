import os
import pathlib
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from parameterized import param, parameterized

import ballroom_scribe.gui.models
import ballroom_scribe.gui.utils

_HOME = pathlib.Path("/home/dancer")
_SESSION_NAME = "20261121-091645-4242"
_LINUX_LOG_ROOT = _HOME / ".local" / "state" / "ballroom-scribe" / "logs"


class TestCurrentSessionLogs(unittest.TestCase):
    """Unit tests for the `current_session_logs` function."""

    def setUp(self):
        self.enterContext(patch.dict(os.environ, {}, clear=True))
        self.enterContext(patch.object(sys, "platform", "linux"))
        self.enterContext(patch.object(pathlib.Path, "home", autospec=True, return_value=_HOME))
        self.enterContext(patch.object(time, "strftime", autospec=True, return_value="20261121-091645"))
        self.enterContext(patch.object(os, "getpid", autospec=True, return_value=4242))
        self.mock_mkdir = self.enterContext(patch.object(pathlib.Path, "mkdir", autospec=True))
        self.mock_iterdir = self.enterContext(patch.object(pathlib.Path, "iterdir", autospec=True))
        self.mock_iterdir.return_value = iter(())
        self.mock_is_dir = self.enterContext(patch.object(pathlib.Path, "is_dir", autospec=True, return_value=True))
        self.mock_rmtree = self.enterContext(patch.object(shutil, "rmtree", autospec=True))
        self.mock_mkdtemp = self.enterContext(patch.object(tempfile, "mkdtemp", autospec=True))

    def test_uses_the_folder_that_a_parent_process_made(self):
        os.environ[ballroom_scribe.gui.models.SessionLogs.ENVIRONMENT_VARIABLE] = "/logs/20261121-091640-17"

        session_logs = ballroom_scribe.gui.utils.current_session_logs()

        self.assertEqual(
            session_logs,
            ballroom_scribe.gui.models.SessionLogs(pathlib.Path("/logs/20261121-091640-17")),
        )
        # The parent process made the folder and tidied the old ones.
        self.mock_mkdir.assert_not_called()
        self.mock_iterdir.assert_not_called()
        self.mock_rmtree.assert_not_called()
        self.mock_mkdtemp.assert_not_called()

    @parameterized.expand(
        [
            param(
                "windows",
                platform="win32",
                environment={"LOCALAPPDATA": "/Users/dancer/AppData/Local"},
                expected_root=pathlib.Path("/Users/dancer/AppData/Local/Ballroom Scribe/Logs"),
            ),
            param(
                "windows_without_local_app_data",
                platform="win32",
                environment={},
                expected_root=_HOME / "AppData" / "Local" / "Ballroom Scribe" / "Logs",
            ),
            param(
                "macos",
                platform="darwin",
                environment={},
                expected_root=_HOME / "Library" / "Logs" / "Ballroom Scribe",
            ),
            param("linux", platform="linux", environment={}, expected_root=_LINUX_LOG_ROOT),
            param(
                "linux_with_xdg_state_home",
                platform="linux",
                environment={"XDG_STATE_HOME": "/var/state"},
                expected_root=pathlib.Path("/var/state/ballroom-scribe/logs"),
            ),
        ]
    )
    def test_makes_a_folder_where_the_platform_expects_logs(self, _, platform, environment, expected_root):
        self.enterContext(patch.object(sys, "platform", platform))
        os.environ.update(environment)

        session_logs = ballroom_scribe.gui.utils.current_session_logs()

        # The folder is named after the time and the process that made it.
        self.assertEqual(session_logs.directory, expected_root / _SESSION_NAME)
        self.mock_mkdir.assert_called_once_with(expected_root / _SESSION_NAME, parents=True, exist_ok=True)
        self.mock_iterdir.assert_called_once_with(expected_root)
        self.mock_mkdtemp.assert_not_called()

    def test_deletes_the_folders_of_all_but_the_newest_sessions(self):
        sessions = [_LINUX_LOG_ROOT / f"202611{day:02d}-120000-{1000 + day}" for day in range(1, 23)]
        # The listing is in no particular order.
        self.mock_iterdir.return_value = iter(reversed(sessions))

        ballroom_scribe.gui.utils.current_session_logs()

        # Twenty sessions are kept, so the two oldest of these twenty-two go.
        self.assertEqual([call.args for call in self.mock_rmtree.call_args_list], [(sessions[0],), (sessions[1],)])
        for call in self.mock_rmtree.call_args_list:
            self.assertEqual(call.kwargs, {"ignore_errors": True})

    def test_keeps_all_sessions_while_there_are_few(self):
        self.mock_iterdir.return_value = iter(
            [_LINUX_LOG_ROOT / f"202611{day:02d}-120000-{1000 + day}" for day in range(1, 21)]
        )

        ballroom_scribe.gui.utils.current_session_logs()

        self.mock_rmtree.assert_not_called()

    def test_deletes_nothing_but_the_folders_of_sessions(self):
        sessions = [_LINUX_LOG_ROOT / f"202611{day:02d}-120000-{1000 + day}" for day in range(1, 22)]
        file_named_like_a_session = _LINUX_LOG_ROOT / "20261001-120000-999"
        others = [
            _LINUX_LOG_ROOT / "notes.txt",
            _LINUX_LOG_ROOT / "Old Logs",
            _LINUX_LOG_ROOT / "20261001-120000-998.bak",
            _LINUX_LOG_ROOT / "20261001-120000",
            file_named_like_a_session,
        ]
        self.mock_iterdir.return_value = iter([*others, *sessions])
        self.mock_is_dir.side_effect = lambda path: path != file_named_like_a_session

        ballroom_scribe.gui.utils.current_session_logs()

        # Only the oldest of the twenty-one sessions goes, although every other entry sorts before it.
        self.assertEqual([call.args for call in self.mock_rmtree.call_args_list], [(sessions[0],)])

    def test_makes_a_temporary_folder_where_the_folder_for_logs_cannot_be_written(self):
        self.mock_mkdir.side_effect = PermissionError("The folder for logs is read-only")
        self.mock_mkdtemp.return_value = "/tmp/ballroom-scribe-20261121-091645-4242-k3j2"

        session_logs = ballroom_scribe.gui.utils.current_session_logs()

        self.assertEqual(session_logs.directory, pathlib.Path("/tmp/ballroom-scribe-20261121-091645-4242-k3j2"))
        self.mock_mkdtemp.assert_called_once_with(prefix=f"ballroom-scribe-{_SESSION_NAME}-")
        self.mock_iterdir.assert_not_called()

    def test_keeps_the_new_folder_when_the_old_sessions_cannot_be_listed(self):
        self.mock_iterdir.side_effect = PermissionError("The folder for logs cannot be listed")

        session_logs = ballroom_scribe.gui.utils.current_session_logs()

        self.assertEqual(session_logs.directory, _LINUX_LOG_ROOT / _SESSION_NAME)
        self.mock_rmtree.assert_not_called()
        self.mock_mkdtemp.assert_not_called()


if __name__ == "__main__":
    unittest.main()
