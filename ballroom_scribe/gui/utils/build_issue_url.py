"""Utility that builds the address of the GitHub form for reporting a problem, with its fields filled in."""

import bisect
import urllib.parse
from collections.abc import Callable

_NEW_ISSUE_URL = "https://github.com/BadgerBallroom/ballroom_scribe/issues/new"
# The form in `.github/ISSUE_TEMPLATE`. GitHub fills each field of a form from the query parameter that is named like
# the `id` of the field.
_ISSUE_TEMPLATE = "problem_report.yml"
_WHAT_HAPPENED_FIELD = "what-happened"
_ENVIRONMENT_FIELD = "environment"
# Windows is reported to refuse to open addresses of much more than 2,000 characters, and GitHub rejects very long
# ones. The report file carries the details, so the address only has to carry the gist.
_MAX_URL_LENGTH = 2000
_MAX_TITLE_LENGTH = 100
_ELLIPSIS = "…"


def _url(title: str, what_happened: str, environment: str) -> str:
    fields = {"template": _ISSUE_TEMPLATE}
    if title:
        fields["title"] = title
    if what_happened:
        fields[_WHAT_HAPPENED_FIELD] = what_happened
    if environment:
        fields[_ENVIRONMENT_FIELD] = environment
    return f"{_NEW_ISSUE_URL}?{urllib.parse.urlencode(fields, quote_via=urllib.parse.quote)}"


def _longest_that_fits(text: str, url_with: Callable[[str], str]) -> str:
    """Returns `text`, or as much of its beginning as keeps the address that `url_with` builds short enough.

    Args:
        text: The value of a field of the form.
        url_with: Builds the address for a given value of that field.

    Returns:
        `text` if the address is short enough with all of it. Otherwise the longest beginning of `text` with which it
        is, followed by an ellipsis, or `""` if the address is too long even then.
    """
    if len(url_with(text)) <= _MAX_URL_LENGTH:
        return text
    # The address grows with every character that is kept, so the number to keep can be found by bisection.
    fitting_counts = bisect.bisect_right(
        range(len(text)),
        _MAX_URL_LENGTH,
        key=lambda kept: len(url_with(text[:kept] + _ELLIPSIS)),
    )
    return text[: fitting_counts - 1] + _ELLIPSIS if fitting_counts else ""


def build_issue_url(summary: str, environment: str) -> str:
    """Returns the address of GitHub's page for a new issue, with the form for reporting a problem filled in.

    The first line of `summary` becomes the title of the issue. If the address would get too long, `summary` is cut
    short, and after that `environment`.

    Args:
        summary: What happened, in a sentence or a few lines, or `""` to leave that to the user.
        environment: The version of Ballroom Scribe and the software that it runs on, as `describe_environment`
            returns it.

    Returns:
        The address, which is at most 2,000 characters long.
    """
    summary = summary.strip()
    first_line = summary.partition("\n")[0].strip()
    if len(first_line) > _MAX_TITLE_LENGTH:
        first_line = first_line[: _MAX_TITLE_LENGTH - len(_ELLIPSIS)] + _ELLIPSIS
    what_happened = _longest_that_fits(summary, lambda text: _url(first_line, text, environment))
    if not what_happened:
        environment = _longest_that_fits(environment, lambda text: _url(first_line, "", text))
    return _url(first_line, what_happened, environment)
