"""Utility that says which version of Ballroom Scribe is running."""

import os
import pathlib
import subprocess

# `git archive` replaces these two placeholders with the commit that an archive was made from, because
# `.gitattributes` marks this file `export-subst`. That covers the ZIP files that GitHub offers for download, which
# have no `.git` directory to ask.
_ARCHIVE_COMMIT = "$Format:%H$"
_ARCHIVE_COMMIT_DATE = "$Format:%cI$"
# What the placeholders start with for as long as nothing has replaced them.
_UNREPLACED_PREFIX = "$Format"
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_ABBREVIATED_COMMIT_LENGTH = 12
_DATE_LENGTH = len("YYYY-MM-DD")
_GIT_TIMEOUT_SECONDS = 5.0
_UNKNOWN_VERSION = "unknown version (neither a GitHub download nor a git checkout)"


def _describe_checkout() -> str | None:
    """Returns what `git describe` says about the checkout that this file is in, or `None` if it cannot say.

    That is the abbreviated commit, followed by `-dirty` if tracked files have been changed since.
    """
    if not (_REPO_ROOT / ".git").exists():
        return None
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(_REPO_ROOT),
                "describe",
                "--always",
                "--dirty",
                f"--abbrev={_ABBREVIATED_COMMIT_LENGTH}",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            # Without optional locks, this never holds up a `git` command that the user is running in the checkout.
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
            # Keeps Windows from opening a console window for `git` when this process has none (`pythonw`).
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError, subprocess.SubprocessError:
        return None
    return result.stdout.strip() or None


def describe_version(
    archive_commit: str = _ARCHIVE_COMMIT,
    archive_commit_date: str = _ARCHIVE_COMMIT_DATE,
) -> str:
    """Returns which commit of Ballroom Scribe this copy is, for logs and problem reports.

    Args:
        archive_commit: The commit that `git archive` wrote into this file, or the placeholder that it replaces.
        archive_commit_date: The date of that commit in ISO 8601 format, or the placeholder that `git archive`
            replaces.

    Returns:
        `3f2a1bc04d5e (2026-11-20)` for a copy that was downloaded as an archive, what `git describe` says for a git
        checkout (such as `3f2a1bc04d5e-dirty`), and otherwise a phrase saying that the version is unknown.
    """
    if not archive_commit.startswith(_UNREPLACED_PREFIX):
        return f"{archive_commit[:_ABBREVIATED_COMMIT_LENGTH]} ({archive_commit_date[:_DATE_LENGTH]})"
    return _describe_checkout() or _UNKNOWN_VERSION
