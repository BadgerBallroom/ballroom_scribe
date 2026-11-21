"""Utility that configures Python and Qt logging for the editor entry points."""

import logging
import os
from typing import TextIO

from PySide6 import QtCore

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
# Mirrors `_LOG_FORMAT` for the messages Qt prints itself (`%{category}` is `default` for uncategorized messages).
_QT_MESSAGE_PATTERN = (
    "%{time yyyy-MM-dd hh:mm:ss.zzz} %{if-debug}DEBUG%{endif}%{if-info}INFO%{endif}%{if-warning}WARNING%{endif}"
    "%{if-critical}ERROR%{endif}%{if-fatal}CRITICAL%{endif} %{category}: %{message}"
)
# Qt's `*.trace` categories log per audio buffer and the stream decoder logs per stale packet after each seek;
# everything else under `qt.multimedia` is low-volume and useful for diagnosing decoding or rendering problems.
_VERBOSE_QT_LOGGING_RULES = (
    "qt.multimedia=true\nqt.multimedia.*=true\n*.trace=false\nqt.multimedia.ffmpeg.streamdecoder.debug=false"
)


def configure_logging(verbose: bool, session_log: TextIO | None = None) -> None:
    """Sends application and Qt log messages to stderr, including Qt multimedia internals when `verbose`.

    With a `session_log`, the application's messages go there as well, at every level whatever `verbose` says, so
    that what a log file holds after a failure never depends on how the editor was started.

    Qt's messages are written by Qt's own default handler rather than forwarded into Python `logging`: Qt Multimedia
    logs from its decoder and renderer threads, and a Python message handler installed with `qInstallMessageHandler`
    would need the GIL on those threads while the GUI thread may hold it inside a blocking Qt call, which deadlocks
    (observed on Windows with the verbose categories enabled). `QT_FORCE_STDERR_LOGGING` stops Qt from diverting
    messages to the debugger output when it cannot detect a console. Qt's messages therefore reach stderr only, not
    `session_log`.

    Args:
        verbose: If true, logs to stderr at `DEBUG` and enables Qt's `qt.multimedia` debug categories; otherwise logs
            to stderr at `INFO` and leaves Qt's default category filters in place.
        session_log: The open log file of this session, which receives the application's messages at `DEBUG`, or
            `None` to log to stderr only.
    """
    console_level = logging.DEBUG if verbose else logging.INFO
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    handlers: list[logging.Handler] = [console_handler]
    if session_log is not None:
        session_handler = logging.StreamHandler(session_log)
        session_handler.setLevel(logging.DEBUG)
        handlers.append(session_handler)
    logging.basicConfig(
        level=logging.DEBUG if session_log is not None else console_level,
        format=_LOG_FORMAT,
        handlers=handlers,
    )
    os.environ.setdefault("QT_FORCE_STDERR_LOGGING", "1")
    QtCore.qSetMessagePattern(_QT_MESSAGE_PATTERN)
    if verbose:
        QtCore.QLoggingCategory.setFilterRules(_VERBOSE_QT_LOGGING_RULES)
