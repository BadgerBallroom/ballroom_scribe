import datetime
import unittest

import ballroom_scribe.gui.utils

_CREATED = datetime.datetime(2026, 11, 21, 8, 16, 55, tzinfo=datetime.timezone(datetime.timedelta(hours=-6)))
_SUMMARY = "The editor was ended by signal 11 (SIGSEGV)."
_ENVIRONMENT = "Ballroom Scribe 0911b61b64dc\nPython 3.14.3 (python) on Windows-11\nPackages: torch 2.14.0"
_FIRST_SECTION = "== What happened =="
_MAX_LOG_CHARACTERS = 1_000_000


class TestBuildProblemReport(unittest.TestCase):
    """Unit tests for the `build_problem_report` function."""

    def test_says_what_happened_the_environment_and_each_log_in_order(self):
        report = ballroom_scribe.gui.utils.build_problem_report(
            _SUMMARY,
            _ENVIRONMENT,
            {
                "editor.log": "INFO Starting the editor\nFatal Python error: Segmentation fault\n",
                "console.log": "WARNING qt.multimedia.ffmpeg: The decoder failed\n",
            },
            _CREATED,
        )

        self.assertEqual(
            report[report.index(_FIRST_SECTION) :],
            "== What happened ==\n"
            "The editor was ended by signal 11 (SIGSEGV).\n"
            "\n"
            "== Environment ==\n"
            "Ballroom Scribe 0911b61b64dc\n"
            "Python 3.14.3 (python) on Windows-11\n"
            "Packages: torch 2.14.0\n"
            "\n"
            "== editor.log ==\n"
            "INFO Starting the editor\n"
            "Fatal Python error: Segmentation fault\n"
            "\n"
            "== console.log ==\n"
            "WARNING qt.multimedia.ffmpeg: The decoder failed\n",
        )

    def test_starts_with_when_it_was_made_and_a_notice_of_what_it_contains(self):
        report = ballroom_scribe.gui.utils.build_problem_report(_SUMMARY, _ENVIRONMENT, {}, _CREATED)

        preamble = report[: report.index(_FIRST_SECTION)]
        title, created, blank, *notice = preamble.splitlines()
        self.assertEqual(title, "Ballroom Scribe problem report")
        self.assertEqual(created, "Created 2026-11-21 08:16:55 -0600")
        self.assertEqual(blank, "")
        # Whoever opens the file later should know what is in it before passing it on.
        notice_text = " ".join(line for line in notice if line)
        self.assertIn("file names and folder paths", notice_text)
        self.assertIn("may contain speech that was transcribed from the videos", notice_text)
        self.assertIn("before you share it publicly", notice_text)
        # The notice is wrapped, so that it can be read in an editor that does not wrap lines.
        self.assertGreater(len([line for line in notice if line]), 1)
        self.assertLessEqual(max(len(line) for line in notice), 100)

    def test_time_without_a_time_zone(self):
        report = ballroom_scribe.gui.utils.build_problem_report(
            _SUMMARY,
            _ENVIRONMENT,
            {},
            datetime.datetime(2026, 11, 21, 8, 16, 55),
        )

        self.assertEqual(report.splitlines()[1], "Created 2026-11-21 08:16:55")

    def test_report_without_logs_ends_with_the_environment(self):
        report = ballroom_scribe.gui.utils.build_problem_report(_SUMMARY, _ENVIRONMENT, {}, _CREATED)

        self.assertTrue(report.endswith("== Environment ==\n" + _ENVIRONMENT + "\n"))

    def test_marks_what_is_empty(self):
        report = ballroom_scribe.gui.utils.build_problem_report("", _ENVIRONMENT, {"editor.log": "\n"}, _CREATED)

        self.assertIn("== What happened ==\n(empty)\n", report)
        self.assertTrue(report.endswith("== editor.log ==\n(empty)\n"))

    def test_keeps_the_indentation_of_the_first_line_of_a_log(self):
        report = ballroom_scribe.gui.utils.build_problem_report(
            _SUMMARY,
            _ENVIRONMENT,
            {"console.log": '  File "main_window.py", line 276 in set_source\n'},
            _CREATED,
        )

        self.assertTrue(report.endswith('== console.log ==\n  File "main_window.py", line 276 in set_source\n'))

    def test_keeps_only_the_end_of_a_very_long_log(self):
        line = "x" * 99 + "\n"
        log = "the first line of the log\n" + line * (_MAX_LOG_CHARACTERS // len(line)) + "the last line of the log\n"

        report = ballroom_scribe.gui.utils.build_problem_report(_SUMMARY, _ENVIRONMENT, {"editor.log": log}, _CREATED)

        section = report[report.index("== editor.log ==") :]
        heading, left_out, first_kept, *_, last_kept = section.splitlines()
        self.assertEqual(heading, "== editor.log ==")
        self.assertRegex(left_out, r"^\(The first [\d,]+ characters were left out\.\)$")
        # What explains a failure is at the end, and what is kept starts with a whole line.
        self.assertEqual(first_kept, "x" * 99)
        self.assertEqual(last_kept, "the last line of the log")
        self.assertNotIn("the first line of the log", report)
        self.assertLessEqual(len(section), _MAX_LOG_CHARACTERS + 200)

    def test_keeps_all_of_a_log_that_is_just_short_enough(self):
        log = "x" * (_MAX_LOG_CHARACTERS - 1) + "\n"

        report = ballroom_scribe.gui.utils.build_problem_report(_SUMMARY, _ENVIRONMENT, {"editor.log": log}, _CREATED)

        self.assertTrue(report.endswith("== editor.log ==\n" + log))
        self.assertNotIn("were left out", report)


if __name__ == "__main__":
    unittest.main()
