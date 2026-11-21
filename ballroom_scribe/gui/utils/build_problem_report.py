"""Utility that puts the text of a problem report together."""

import datetime
import textwrap
from collections.abc import Mapping

_TITLE = "Ballroom Scribe problem report"
# The dialog that offers the report and the form of the GitHub issue say the same in their own words.
_PRIVACY_NOTICE = (
    "This file contains file names and folder paths from the computer on which it was made, and it may contain speech"
    " that was transcribed from the videos that were open. Read it before you share it publicly."
)
_NOTICE_WIDTH = 100
_CREATED_FORMAT = "%Y-%m-%d %H:%M:%S %z"
# How much of the end of each log a report carries. What explains a failure is at the end of a log.
_MAX_LOG_CHARACTERS = 1_000_000
_EMPTY = "(empty)"


def _end_of_log(text: str) -> str:
    """Returns `text`, or only its last lines if it is longer than what a report carries of a log."""
    if len(text) <= _MAX_LOG_CHARACTERS:
        return text
    kept = text[-_MAX_LOG_CHARACTERS:]
    # What was kept begins in the middle of a line; start at the first whole line.
    kept = kept[kept.find("\n") + 1 :]
    return f"(The first {len(text) - len(kept):,} characters were left out.)\n{kept}"


def _section(heading: str, body: str) -> str:
    return f"== {heading} ==\n{body.strip('\n') or _EMPTY}\n"


def build_problem_report(
    summary: str,
    environment: str,
    logs: Mapping[str, str],
    created: datetime.datetime,
) -> str:
    """Returns the text of a problem report: what happened, the environment, and the logs of the session.

    The report begins with a notice of what it contains, because whoever opens the file later should know that before
    passing it on.

    Args:
        summary: What happened, in a sentence or a few lines.
        environment: The version of Ballroom Scribe and the software that it runs on, as `describe_environment`
            returns it.
        logs: The text of each log of the session by the name of its file, in the order in which the report shows
            them. Only the end of a very long log is kept.
        created: When the report is made.

    Returns:
        The report as plain text that ends with a line break.
    """
    return "\n".join(
        (
            f"{_TITLE}\nCreated {created.strftime(_CREATED_FORMAT).strip()}\n",
            f"{textwrap.fill(_PRIVACY_NOTICE, width=_NOTICE_WIDTH)}\n",
            _section("What happened", summary),
            _section("Environment", environment),
            *(_section(name, _end_of_log(text)) for name, text in logs.items()),
        )
    )
