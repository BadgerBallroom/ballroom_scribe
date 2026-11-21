"""Utility that opens the log file of one session of the editor."""

from typing import TextIO

import ballroom_scribe.gui.models

# Every log that was opened, so that none is closed for as long as the process runs: `faulthandler` writes to the
# log when the process dies, even while Python shuts down, and the freeze watchdog when the GUI thread hangs.
_OPEN_SESSION_LOGS: list[TextIO] = []


def open_session_log(session_logs: ballroom_scribe.gui.models.SessionLogs) -> TextIO:
    """Opens the editor's log of a session for appending and keeps it open until the process ends.

    The file is written one line at a time. That keeps what `logging` writes through the returned file and what
    `faulthandler` writes to its file descriptor in order, and it leaves nothing unwritten when the process dies.

    Args:
        session_logs: Where the session keeps its logs. The folder is made if it does not exist yet.

    Returns:
        The open file. It must not be closed.
    """
    session_logs.directory.mkdir(parents=True, exist_ok=True)
    session_log = open(session_logs.editor_log, "a", encoding="utf-8", buffering=1)  # noqa: SIM115
    _OPEN_SESSION_LOGS.append(session_log)
    return session_log
