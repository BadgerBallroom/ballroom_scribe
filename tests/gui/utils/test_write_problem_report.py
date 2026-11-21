import pathlib
import unittest
from unittest.mock import call, patch

import ballroom_scribe.gui.models
import ballroom_scribe.gui.utils
import ballroom_scribe.utils

_SESSION_LOGS = ballroom_scribe.gui.models.SessionLogs(pathlib.Path("/logs/20261121-091645-4242"))
_SUMMARY = "Auto-Find Events failed: the pipeline exited with status 1."
_ENVIRONMENT = "Ballroom Scribe 0911b61b64dc"
_EDITOR_LOG = "INFO Starting the editor\n"
_CONSOLE_LOG = "WARNING qt.multimedia.ffmpeg: The decoder failed\n"
_REPORT = "Ballroom Scribe problem report\n"


class TestWriteProblemReport(unittest.TestCase):
    """Unit tests for the `write_problem_report` function."""

    def setUp(self):
        self.logs = {_SESSION_LOGS.editor_log: _EDITOR_LOG, _SESSION_LOGS.console_log: _CONSOLE_LOG}
        self.mock_read_text = self.enterContext(
            patch.object(pathlib.Path, "read_text", autospec=True, side_effect=self._read_log)
        )
        self.mock_write_text = self.enterContext(patch.object(pathlib.Path, "write_text", autospec=True))
        self.mock_describe_environment = self.enterContext(
            patch.object(ballroom_scribe.utils, "describe_environment", autospec=True, return_value=_ENVIRONMENT)
        )
        self.mock_build_problem_report = self.enterContext(
            patch.object(ballroom_scribe.gui.utils, "build_problem_report", autospec=True, return_value=_REPORT)
        )

    def _read_log(self, path: pathlib.Path, **_kwargs: str) -> str:
        """Stand-in for `Path.read_text` in a folder that holds only the logs in `self.logs`."""
        if path not in self.logs:
            raise FileNotFoundError(path)
        return self.logs[path]

    def test_report_holds_the_summary_the_environment_and_the_logs(self):
        ballroom_scribe.gui.utils.write_problem_report(_SESSION_LOGS, _SUMMARY)

        self.mock_build_problem_report.assert_called_once()
        summary, environment, logs, created = self.mock_build_problem_report.call_args.args
        self.assertEqual(summary, _SUMMARY)
        self.assertEqual(environment, _ENVIRONMENT)
        # The editor's own log comes first, then what its supervisor kept of its console.
        self.assertEqual(list(logs.items()), [("editor.log", _EDITOR_LOG), ("console.log", _CONSOLE_LOG)])
        # The report says in which time zone it was made.
        self.assertIsNotNone(created.tzinfo)

    def test_writes_the_report_into_the_folder_of_the_session(self):
        report_file = ballroom_scribe.gui.utils.write_problem_report(_SESSION_LOGS, _SUMMARY)

        self.mock_write_text.assert_called_once_with(report_file, _REPORT, encoding="utf-8")
        self.assertEqual(report_file.parent, _SESSION_LOGS.directory)

    def test_name_of_the_file_says_when_the_report_was_made(self):
        report_file = ballroom_scribe.gui.utils.write_problem_report(_SESSION_LOGS, _SUMMARY)

        created = self.mock_build_problem_report.call_args.args[3]
        self.assertEqual(
            report_file.name,
            created.strftime("Ballroom Scribe problem report %Y-%m-%d %H-%M-%S.txt"),
        )
        self.assertRegex(
            report_file.name,
            r"^Ballroom Scribe problem report \d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2}\.txt$",
        )

    def test_reads_the_logs_whatever_bytes_they_contain(self):
        ballroom_scribe.gui.utils.write_problem_report(_SESSION_LOGS, _SUMMARY)

        # A process that crashed, or a library that it used, may have written bytes that are not UTF-8.
        self.assertEqual(
            self.mock_read_text.call_args_list,
            [
                call(_SESSION_LOGS.editor_log, encoding="utf-8", errors="replace"),
                call(_SESSION_LOGS.console_log, encoding="utf-8", errors="replace"),
            ],
        )

    def test_leaves_out_a_log_that_the_session_does_not_have(self):
        # An editor that runs without its supervisor has no console log.
        del self.logs[_SESSION_LOGS.console_log]

        ballroom_scribe.gui.utils.write_problem_report(_SESSION_LOGS, _SUMMARY)

        self.assertEqual(self.mock_build_problem_report.call_args.args[2], {"editor.log": _EDITOR_LOG})

    def test_writes_a_report_even_without_any_log(self):
        self.logs.clear()

        ballroom_scribe.gui.utils.write_problem_report(_SESSION_LOGS, _SUMMARY)

        self.assertEqual(self.mock_build_problem_report.call_args.args[2], {})
        self.mock_write_text.assert_called_once()

    def test_failure_to_write_the_report_propagates(self):
        self.mock_write_text.side_effect = PermissionError("The folder of the session is read-only")

        with self.assertRaises(OSError):
            ballroom_scribe.gui.utils.write_problem_report(_SESSION_LOGS, _SUMMARY)


if __name__ == "__main__":
    unittest.main()
