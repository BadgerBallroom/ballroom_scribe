"""Utility that finds or makes the folder in which this run of the editor keeps its logs."""

import contextlib
import os
import pathlib
import re
import shutil
import sys
import tempfile
import time

import ballroom_scribe.gui.models

_KEPT_SESSIONS = 20
_SESSION_TIME_FORMAT = "%Y%m%d-%H%M%S"
# The name of a session's folder: when the session started, then the ID of the process that made the folder. Names
# therefore sort from the oldest session to the newest.
_SESSION_NAME_RE = re.compile(r"\d{8}-\d{6}-\d+")


def _log_root() -> pathlib.Path:
    """Returns this user's folder for the logs of Ballroom Scribe, where the platform expects logs to be."""
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA")
        base = pathlib.Path(local_app_data) if local_app_data else pathlib.Path.home() / "AppData" / "Local"
        return base / "Ballroom Scribe" / "Logs"
    if sys.platform == "darwin":
        return pathlib.Path.home() / "Library" / "Logs" / "Ballroom Scribe"
    state_home = os.environ.get("XDG_STATE_HOME")
    base = pathlib.Path(state_home) if state_home else pathlib.Path.home() / ".local" / "state"
    return base / "ballroom-scribe" / "logs"


def _remove_old_sessions(log_root: pathlib.Path) -> None:
    """Deletes the folders of all but the newest sessions, and nothing else that `log_root` may contain."""
    sessions = sorted(
        entry for entry in log_root.iterdir() if _SESSION_NAME_RE.fullmatch(entry.name) and entry.is_dir()
    )
    for old_session in sessions[:-_KEPT_SESSIONS]:
        # A folder that cannot be deleted is left for next time. On Windows, that includes the folder of a session
        # that is still running, because its log is open.
        shutil.rmtree(old_session, ignore_errors=True)


def _make_session_dir() -> pathlib.Path:
    """Makes the folder of a new session and returns it."""
    session_name = f"{time.strftime(_SESSION_TIME_FORMAT)}-{os.getpid()}"
    log_root = _log_root()
    session_dir = log_root / session_name
    try:
        session_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        # The editor has to start even where the user's folder for logs cannot be written.
        return pathlib.Path(tempfile.mkdtemp(prefix=f"ballroom-scribe-{session_name}-"))
    with contextlib.suppress(OSError):
        _remove_old_sessions(log_root)
    return session_dir


def current_session_logs() -> ballroom_scribe.gui.models.SessionLogs:
    """Returns where this run of the editor keeps its logs, making a folder for them unless a parent process did.

    The supervisor of the editor makes the folder and names it in the environment of the editor that it starts (see
    `SessionLogs.ENVIRONMENT_VARIABLE`), and the editor finds it here. Without that variable, as when the editor
    runs without its supervisor, every call makes a new folder, so call this once per process.

    A new folder is named after the current time and this process, in the user's folder for logs:

    *   Windows: `%LOCALAPPDATA%\\Ballroom Scribe\\Logs`
    *   macOS: `~/Library/Logs/Ballroom Scribe`
    *   Elsewhere: `$XDG_STATE_HOME/ballroom-scribe/logs`, by default `~/.local/state/ballroom-scribe/logs`

    The folders of all but the newest sessions are deleted then. If the folder cannot be made there, a temporary
    folder is made instead, so that the editor starts regardless.
    """
    inherited_dir = os.environ.get(ballroom_scribe.gui.models.SessionLogs.ENVIRONMENT_VARIABLE)
    if inherited_dir:
        return ballroom_scribe.gui.models.SessionLogs(pathlib.Path(inherited_dir))
    return ballroom_scribe.gui.models.SessionLogs(_make_session_dir())
