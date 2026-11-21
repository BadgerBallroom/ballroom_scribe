import pathlib
import subprocess
import unittest
from unittest.mock import ANY, patch

from parameterized import param, parameterized

import ballroom_scribe.utils

_ARCHIVE_COMMIT = "1f928413b12a95bc98accaa4138a6b7d2b768bc5"
_ARCHIVE_COMMIT_DATE = "2026-11-21T09:03:30-06:00"


class TestDescribeVersion(unittest.TestCase):
    """Unit tests for the `describe_version` function."""

    def setUp(self):
        self.mock_path_exists = self.enterContext(patch.object(pathlib.Path, "exists", autospec=True))
        self.mock_path_exists.return_value = True
        self.mock_subprocess_run = self.enterContext(patch.object(subprocess, "run", autospec=True))
        self.mock_subprocess_run.return_value.stdout = "0911b61b64dc\n"

    def test_describes_the_commit_that_an_archive_was_made_from(self):
        version = ballroom_scribe.utils.describe_version(
            archive_commit=_ARCHIVE_COMMIT,
            archive_commit_date=_ARCHIVE_COMMIT_DATE,
        )

        self.assertEqual(version, "1f928413b12a (2026-11-21)")
        # An archive has no `.git` directory, so `git` is not asked.
        self.mock_path_exists.assert_not_called()
        self.mock_subprocess_run.assert_not_called()

    @parameterized.expand(
        [
            param("clean", stdout="0911b61b64dc\n", expected="0911b61b64dc"),
            param("with_local_changes", stdout="0911b61b64dc-dirty\n", expected="0911b61b64dc-dirty"),
            param("after_a_tag", stdout="v1.0-3-g0911b61b64dc\n", expected="v1.0-3-g0911b61b64dc"),
        ]
    )
    def test_asks_git_about_a_checkout(self, _, stdout, expected):
        self.mock_subprocess_run.return_value.stdout = stdout

        # In a checkout, nothing has replaced the placeholders, which are the default arguments.
        self.assertEqual(ballroom_scribe.utils.describe_version(), expected)

        self.mock_path_exists.assert_called_once()
        (git_dir,), _ = self.mock_path_exists.call_args
        self.assertEqual(git_dir.name, ".git")
        self.mock_subprocess_run.assert_called_once_with(
            ["git", "-C", str(git_dir.parent), "describe", "--always", "--dirty", "--abbrev=12"],
            capture_output=True,
            text=True,
            check=True,
            timeout=ANY,
            env=ANY,
            creationflags=ANY,
        )

    def test_git_takes_no_optional_locks(self):
        ballroom_scribe.utils.describe_version()

        # The user may be running `git` in the checkout at the same time.
        self.assertEqual(self.mock_subprocess_run.call_args.kwargs["env"]["GIT_OPTIONAL_LOCKS"], "0")
        self.assertGreater(self.mock_subprocess_run.call_args.kwargs["timeout"], 0.0)

    def test_version_is_unknown_without_archive_or_checkout(self):
        self.mock_path_exists.return_value = False

        self.assertRegex(ballroom_scribe.utils.describe_version(), "^unknown version")

        self.mock_subprocess_run.assert_not_called()

    @parameterized.expand(
        [
            param("git_is_not_installed", error=FileNotFoundError("git")),
            param("git_fails", error=subprocess.CalledProcessError(returncode=128, cmd=["git"])),
            param("git_does_not_answer", error=subprocess.TimeoutExpired(cmd=["git"], timeout=5.0)),
        ]
    )
    def test_version_is_unknown_when_git_cannot_say(self, _, error):
        self.mock_subprocess_run.side_effect = error

        self.assertRegex(ballroom_scribe.utils.describe_version(), "^unknown version")

    def test_version_is_unknown_when_git_says_nothing(self):
        self.mock_subprocess_run.return_value.stdout = "\n"

        self.assertRegex(ballroom_scribe.utils.describe_version(), "^unknown version")


if __name__ == "__main__":
    unittest.main()
