import unittest
import urllib.parse

from parameterized import param, parameterized

import ballroom_scribe.gui.utils

_NEW_ISSUE_URL = "https://github.com/BadgerBallroom/ballroom_scribe/issues/new"
_MAX_URL_LENGTH = 2000
_ENVIRONMENT = (
    "Ballroom Scribe 0911b61b64dc\n"
    "Python 3.14.3 (C:\\Python314\\python.exe) on Windows-11-10.0.26100-SP0\n"
    "Packages: torch 2.14.0+cu132, faster-whisper 1.2.1, ctranslate2 4.8.2"
)
_ELLIPSIS = "…"


def _fields(url: str) -> dict[str, str]:
    """Returns the fields that the address fills in, decoded, after checking where the address leads."""
    address, _, query = url.partition("?")
    if address != _NEW_ISSUE_URL:
        raise AssertionError(f"The address leads to {address}")
    return dict(urllib.parse.parse_qsl(query, keep_blank_values=True, strict_parsing=True))


class TestBuildIssueUrl(unittest.TestCase):
    """Unit tests for the `build_issue_url` function."""

    def test_fills_in_the_form_for_reporting_a_problem(self):
        summary = "The editor was ended by signal 11 (SIGSEGV)."

        url = ballroom_scribe.gui.utils.build_issue_url(summary, _ENVIRONMENT)

        self.assertEqual(
            _fields(url),
            {
                "template": "problem_report.yml",
                "title": summary,
                "what-happened": summary,
                "environment": _ENVIRONMENT,
            },
        )

    def test_address_is_plain_ascii_without_spaces(self):
        url = ballroom_scribe.gui.utils.build_issue_url("Dvořák’s waltz: the editor crashed", _ENVIRONMENT)

        self.assertTrue(url.isascii())
        self.assertNotRegex(url, r"\s")
        self.assertEqual(_fields(url)["what-happened"], "Dvořák’s waltz: the editor crashed")

    def test_title_is_the_first_line_of_the_summary(self):
        summary = "RuntimeError: The envelope has no bins\nTraceback (most recent call last):\n  File ..."

        fields = _fields(ballroom_scribe.gui.utils.build_issue_url(summary, _ENVIRONMENT))

        self.assertEqual(fields["title"], "RuntimeError: The envelope has no bins")
        self.assertEqual(fields["what-happened"], summary)

    def test_long_title_is_cut(self):
        fields = _fields(ballroom_scribe.gui.utils.build_issue_url("ValueError: " + "x" * 300, _ENVIRONMENT))

        self.assertEqual(len(fields["title"]), 100)
        self.assertEqual(fields["title"], "ValueError: " + "x" * 87 + _ELLIPSIS)

    def test_without_a_summary_the_user_says_what_happened(self):
        fields = _fields(ballroom_scribe.gui.utils.build_issue_url("  \n", _ENVIRONMENT))

        self.assertEqual(fields, {"template": "problem_report.yml", "environment": _ENVIRONMENT})

    def test_long_summary_is_cut_to_keep_the_address_short(self):
        summary = "\n".join(f"File frame_{index}.py, line {index}" for index in range(400))

        url = ballroom_scribe.gui.utils.build_issue_url(summary, _ENVIRONMENT)

        self.assertLessEqual(len(url), _MAX_URL_LENGTH)
        fields = _fields(url)
        # The beginning of the summary is kept, as much of it as the address has room for.
        kept = fields["what-happened"]
        self.assertTrue(kept.endswith(_ELLIPSIS))
        self.assertTrue(summary.startswith(kept.removesuffix(_ELLIPSIS)))
        self.assertGreater(len(url), _MAX_URL_LENGTH - 10)
        # The environment is short and tells more per character, so it stays whole.
        self.assertEqual(fields["environment"], _ENVIRONMENT)

    def test_summary_of_characters_that_take_many_bytes_is_cut_too(self):
        summary = "音" * 3000

        url = ballroom_scribe.gui.utils.build_issue_url(summary, _ENVIRONMENT)

        self.assertLessEqual(len(url), _MAX_URL_LENGTH)
        kept = _fields(url)["what-happened"]
        self.assertRegex(kept, f"^音+{_ELLIPSIS}$")

    def test_overlong_environment_is_cut_after_the_summary_is_gone(self):
        environment = "Packages: " + ", ".join(f"package-{index} 1.0.{index}" for index in range(500))

        url = ballroom_scribe.gui.utils.build_issue_url("The editor exited with status 1.", environment)

        self.assertLessEqual(len(url), _MAX_URL_LENGTH)
        fields = _fields(url)
        self.assertNotIn("what-happened", fields)
        self.assertTrue(fields["environment"].endswith(_ELLIPSIS))
        self.assertTrue(environment.startswith(fields["environment"].removesuffix(_ELLIPSIS)))
        # The title still says what happened.
        self.assertEqual(fields["title"], "The editor exited with status 1.")

    @parameterized.expand(
        [
            param("empty", summary_length=0),
            param("one_character", summary_length=1),
            param("short", summary_length=200),
            param("near_the_limit", summary_length=1100),
            param("just_over_the_limit", summary_length=1300),
            param("long", summary_length=20_000),
        ]
    )
    def test_address_is_never_longer_than_the_shell_can_open(self, _, summary_length):
        summary = ("crash " * summary_length)[:summary_length]

        url = ballroom_scribe.gui.utils.build_issue_url(summary, _ENVIRONMENT)

        self.assertLessEqual(len(url), _MAX_URL_LENGTH)
        self.assertEqual(_fields(url)["environment"], _ENVIRONMENT)


if __name__ == "__main__":
    unittest.main()
