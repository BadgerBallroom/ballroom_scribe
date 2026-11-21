import pathlib
import unittest
from unittest.mock import call, mock_open, patch

import ballroom_scribe.gui.models
import ballroom_scribe.gui.utils

_SESSION_LOGS = ballroom_scribe.gui.models.SessionLogs(pathlib.Path("/logs/20261121-091645-4242"))


class TestOpenSessionLog(unittest.TestCase):
    """Unit tests for the `open_session_log` function."""

    def setUp(self):
        # Resolved before `open` is patched: `lazy_loader` reads the stub of a package with `open` when the package
        # is first imported, and reading it through the mock would leave the package without its exports.
        self.open_session_log = ballroom_scribe.gui.utils.open_session_log
        self.mock_mkdir = self.enterContext(patch.object(pathlib.Path, "mkdir", autospec=True))
        self.mock_open = mock_open()
        self.enterContext(patch("builtins.open", self.mock_open))

    def test_opens_the_log_for_appending_one_line_at_a_time(self):
        session_log = self.open_session_log(_SESSION_LOGS)

        self.assertIs(session_log, self.mock_open.return_value)
        # Appending keeps what an earlier process of the same session wrote. Line buffering leaves nothing unwritten
        # when the process dies.
        self.mock_open.assert_called_once_with(
            pathlib.Path("/logs/20261121-091645-4242/editor.log"),
            "a",
            encoding="utf-8",
            buffering=1,
        )

    def test_makes_the_folder_of_the_session_first(self):
        folder_existed_when_the_log_was_opened = []
        self.mock_open.side_effect = lambda *_args, **_kwargs: (
            folder_existed_when_the_log_was_opened.append(self.mock_mkdir.called) or self.mock_open.return_value
        )

        self.open_session_log(_SESSION_LOGS)

        self.mock_mkdir.assert_called_once_with(_SESSION_LOGS.directory, parents=True, exist_ok=True)
        self.assertEqual(folder_existed_when_the_log_was_opened, [True])

    def test_never_closes_the_log(self):
        session_log = self.open_session_log(_SESSION_LOGS)

        # `faulthandler` writes to the log when the process dies, so it stays open until then.
        self.assertNotIn(call.close(), session_log.mock_calls)
        self.assertNotIn(call.__exit__(None, None, None), session_log.mock_calls)


if __name__ == "__main__":
    unittest.main()
