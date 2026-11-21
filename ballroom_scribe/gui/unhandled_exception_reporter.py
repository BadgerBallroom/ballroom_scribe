"""Hook for unhandled Python exceptions that logs them and tells the GUI thread about each new one."""

import collections
import logging
import sys
import threading
import traceback
from types import TracebackType

from PySide6 import QtCore

_LOGGER = logging.getLogger(__name__)
_REPEATS_LOGGED_EVERY_POWER_OF = 10

type _Origin = tuple[str, str, int]


def _origin(exception_type: type[BaseException], exception_traceback: TracebackType | None) -> _Origin:
    """Returns what tells one error from another: the type of the exception and the line that raised it."""
    frames = traceback.extract_tb(exception_traceback)
    if not frames:
        return exception_type.__name__, "", 0
    return exception_type.__name__, frames[-1].filename, frames[-1].lineno or 0


def _is_power_of_ten(count: int) -> bool:
    while count > 1 and count % _REPEATS_LOGGED_EVERY_POWER_OF == 0:
        count //= _REPEATS_LOGGED_EVERY_POWER_OF
    return count == 1


class UnhandledExceptionReporter(QtCore.QObject):
    """Logs the Python exceptions that nothing handles and announces each new one.

    PySide passes an exception that escapes a slot, an event handler, or a `QRunnable` to `sys.excepthook` and then
    carries on. Without this hook, the traceback goes to standard error, and all the user notices is an editor that
    did not do what was asked.

    The hook runs on whichever thread raised the exception, which may be a thread of Qt's thread pool. It therefore
    only logs and emits `exception_reported`. Connect that signal to a bound method of a `QObject` that lives in the
    GUI thread, so that Qt queues the delivery.

    An error that is raised over and over, such as one in a paint handler, is logged with its traceback and
    announced the first time. After that, only how often it has happened is logged, at 10, 100, 1,000, … times.
    """

    exception_reported = QtCore.Signal(str)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._occurrences: collections.Counter[_Origin] = collections.Counter()
        self._lock = threading.Lock()

    def install(self) -> None:
        """Makes this the handler of unhandled exceptions for the rest of the process."""
        sys.excepthook = self.handle_exception

    def handle_exception(
        self,
        exception_type: type[BaseException],
        exception: BaseException,
        exception_traceback: TracebackType | None,
    ) -> None:
        """Logs an unhandled exception and announces it if it is the first from where it was raised.

        This has the signature of `sys.excepthook`.

        Args:
            exception_type: The type of the exception.
            exception: The exception.
            exception_traceback: The traceback of the exception, if it has one.
        """
        if issubclass(exception_type, KeyboardInterrupt):
            # Ctrl+C in the console keeps doing what it does without this hook.
            sys.__excepthook__(exception_type, exception, exception_traceback)
            return
        origin = _origin(exception_type, exception_traceback)
        with self._lock:
            self._occurrences[origin] += 1
            occurrences = self._occurrences[origin]
        if occurrences > 1:
            if _is_power_of_ten(occurrences):
                _LOGGER.error("Unhandled %s again: %d times so far", exception_type.__name__, occurrences)
            return
        _LOGGER.error("Unhandled exception", exc_info=(exception_type, exception, exception_traceback))
        self.exception_reported.emit(f"{exception_type.__name__}: {exception}")
