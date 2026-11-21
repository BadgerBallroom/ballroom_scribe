"""Watchdog that records where every thread is when the GUI thread stops running its event loop."""

import faulthandler
from typing import TextIO

from PySide6 import QtCore

# Longer than anything that the editor waits for on purpose: closing the window waits up to five seconds for the
# transcription worker and five more for the waveform loader.
_FREEZE_SECONDS = 15.0
_HEARTBEATS_PER_FREEZE = 5
_MILLISECONDS_PER_SECOND = 1000


class FreezeWatchdog(QtCore.QObject):
    """Writes the stack of every thread to a log when the GUI thread has not run its event loop for a while.

    A timer on the GUI thread keeps postponing a `faulthandler` dump, so while the event loop turns, nothing is
    written. Once the GUI thread has been stuck for `freeze_seconds`, `faulthandler` writes the dump from a thread of
    its own. That thread needs neither the event loop nor the GIL, so it also records a GUI thread that is deadlocked
    on the GIL. One freeze is recorded once. The editor is not interrupted, and if it recovers, the watchdog goes on
    watching.
    """

    def __init__(
        self,
        log: TextIO,
        parent: QtCore.QObject | None = None,
        *,
        freeze_seconds: float = _FREEZE_SECONDS,
    ) -> None:
        """Initializes the instance.

        Args:
            log: The open file that receives the stacks. It must stay open until `stop` has been called.
            parent: The Qt parent object.
            freeze_seconds: How long the GUI thread has to be stuck before the stacks are written.

        Raises:
            ValueError: If `freeze_seconds` is not positive.
        """
        super().__init__(parent)
        if freeze_seconds <= 0:
            raise ValueError(f"freeze_seconds must be positive, got {freeze_seconds}")
        self._log = log
        self._freeze_seconds = freeze_seconds
        self._heartbeat_timer = QtCore.QTimer(self)
        self._heartbeat_timer.setInterval(round(freeze_seconds * _MILLISECONDS_PER_SECOND / _HEARTBEATS_PER_FREEZE))
        self._heartbeat_timer.timeout.connect(self._postpone_dump)

    def start(self) -> None:
        """Starts watching the thread that calls this, which must be the GUI thread."""
        self._postpone_dump()
        self._heartbeat_timer.start()

    def stop(self) -> None:
        """Stops watching, so that nothing is written however long the GUI thread takes from here on."""
        self._heartbeat_timer.stop()
        faulthandler.cancel_dump_traceback_later()

    def _postpone_dump(self) -> None:
        # Replaces the dump that was pending with one that is `freeze_seconds` away again.
        faulthandler.dump_traceback_later(self._freeze_seconds, file=self._log)
