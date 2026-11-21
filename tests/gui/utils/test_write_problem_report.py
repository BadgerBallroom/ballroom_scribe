import pathlib
import unittest
from unittest.mock import patch

import ballroom_scribe.gui.models
import ballroom_scribe.gui.utils
import ballroom_scribe.utils

_SESSION_LOGS = ballroom_scribe.gui.models.SessionLogs(pathlib.Path("/logs/20261121-091645-4242"))
_SUMMARY = "Auto-Find Events failed: the pipeline exited with status 1."
_ENVIRONMENT = "Ballroom Scribe 0911b61b64dc"
_EDITOR_LOG = "INFO Starting the editor\n"
_REPORT = "Ballroom Scribe problem report\n"


class TestWriteProblemReport(unittest.TestCase):
    """Unit tests for the `write_problem_report` function."""

    def setUp(self):
        self.mock_read_text = self.enterContext(
            patch.object(pathlib.Path, "read_text", autospec=True, return_value=_EDITOR_LOG)
        )
        self.mock_write_text = self.enterContext(patch.object(pathlib.Path, "write_text", autospec=True))
        self.mock_describe_environment = self.enterContext(
            patch.object(ballroom_scribe.utils, "describe_environment", autospec=True, return_value=_ENVIRONMENT)
        )
        self.mock_build_problem_report = self.enterContext(
            patch.object(ballroom_scribe.gui.utils, "build_problem_report", autospec=True, return_value=_REPORT)
        )

    def test_report_holds_the_summary_the_environment_and_the_logs(self):
        ballroom_scribe.gui.utils.write_problem_report(_SESSION_LOGS, _SUMMARY)

        self.mock_build_problem_report.assert_called_once()
        summary, environment, logs, created = self.mock_build_problem_report.call_args.args
        self.assertEqual(summary, _SUMMARY)
        self.assertEqual(environment, _ENVIRONMENT)
        self.assertEqual(logs, {"editor.log": _EDITOR_LOG})
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

    def test_reads_a_log_whatever_bytes_it_contains(self):
        ballroom_scribe.gui.utils.write_problem_report(_SESSION_LOGS, _SUMMARY)

        # A process that crashed may have written bytes that are not UTF-8.
        self.mock_read_text.assert_called_once_with(_SESSION_LOGS.editor_log, encoding="utf-8", errors="replace")

    def test_leaves_out_a_log_that_cannot_be_read(self):
        self.mock_read_text.side_effect = FileNotFoundError("The session has not logged anything yet")

        ballroom_scribe.gui.utils.write_problem_report(_SESSION_LOGS, _SUMMARY)

        self.assertEqual(self.mock_build_problem_report.call_args.args[2], {})
        self.mock_write_text.assert_called_once()

    def test_failure_to_write_the_report_propagates(self):
        self.mock_write_text.side_effect = PermissionError("The folder of the session is read-only")

        with self.assertRaises(OSError):
            ballroom_scribe.gui.utils.write_problem_report(_SESSION_LOGS, _SUMMARY)


if __name__ == "__main__":
    unittest.main()
