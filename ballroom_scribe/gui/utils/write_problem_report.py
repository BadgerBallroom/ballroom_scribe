"""Utility that writes a problem report into the folder of a session of the editor."""

import datetime
import pathlib

import ballroom_scribe.gui.models
import ballroom_scribe.gui.utils
import ballroom_scribe.utils

_REPORT_NAME_FORMAT = "Ballroom Scribe problem report %Y-%m-%d %H-%M-%S.txt"


def _read_logs(session_logs: ballroom_scribe.gui.models.SessionLogs) -> dict[str, str]:
    """Returns the text of each log that the session has so far, by the name of its file.

    The console log exists only when the editor runs under its supervisor.
    """
    logs: dict[str, str] = {}
    for log_file in (session_logs.editor_log, session_logs.console_log):
        try:
            # A process that crashed may have written bytes that are not UTF-8; they must not cost the report.
            logs[log_file.name] = log_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    return logs


def write_problem_report(session_logs: ballroom_scribe.gui.models.SessionLogs, summary: str) -> pathlib.Path:
    """Writes a report of a problem into the folder of the session and returns the file.

    The report is one plain-text file that holds `summary`, the environment, and what the logs of the session
    contain at this moment (see `build_problem_report`), so that there is a single file to attach or to send.

    Args:
        session_logs: Where the session keeps its logs.
        summary: What happened, in a sentence or a few lines.

    Returns:
        The path of the report, whose name says when it was written.

    Raises:
        OSError: If the report cannot be written.
    """
    created = datetime.datetime.now().astimezone()
    report = ballroom_scribe.gui.utils.build_problem_report(
        summary,
        ballroom_scribe.utils.describe_environment(),
        _read_logs(session_logs),
        created,
    )
    report_file = session_logs.directory / created.strftime(_REPORT_NAME_FORMAT)
    report_file.write_text(report, encoding="utf-8")
    return report_file
