"""The process of the timestamp editor: sets up its logs and runs the main window until it is closed.

`edit_timestamps.py` is what users run. It starts this module as a child process and supervises it (see
`supervise_editor`). To run the editor without its supervisor, run `python -m ballroom_scribe.gui.run_editor` in the
folder that contains `edit_timestamps.py`.
"""

import contextlib
import faulthandler
import logging
import sys
from collections.abc import Sequence

from PySide6 import QtCore

import ballroom_scribe.gui
import ballroom_scribe.gui.utils
import ballroom_scribe.utils

_LOGGER = logging.getLogger(__name__)


def run_editor(argv: Sequence[str] | None = None) -> int:
    """Parses CLI arguments and runs the Ballroom Scribe timestamp editor until its window is closed.

    Everything that the editor logs goes to the log file of this session, whatever `--verbose` says, and so does the
    stack of every thread if the process dies of a fatal error or its GUI thread freezes. A Python exception that
    nothing handles is logged there too, and the window offers to report it.

    Args:
        argv: The command-line arguments, or `None` for those of the process.

    Returns:
        The exit status of the process.
    """
    cli_args, logging_args = ballroom_scribe.gui.parse_editor_args(argv)
    session_logs = ballroom_scribe.gui.utils.current_session_logs()
    session_log = ballroom_scribe.gui.utils.open_session_log(session_logs)
    ballroom_scribe.gui.utils.configure_logging(logging_args["verbose"], session_log)
    # Given the file explicitly: without a console (`pythonw`), there is no standard error to write to.
    faulthandler.enable(file=session_log)
    _LOGGER.info("Starting the editor; its logs are in %s", session_logs.directory)
    _LOGGER.info("%s", ballroom_scribe.utils.describe_environment())

    app = ballroom_scribe.gui.utils.create_application(sys.argv)
    # Installed before the window is created, so that an exception that ends the editor while it starts is logged.
    exception_reporter = ballroom_scribe.gui.UnhandledExceptionReporter(app)
    exception_reporter.install()
    window = ballroom_scribe.gui.MainWindow(session_logs=session_logs)
    # Queued, so that the window tells the user from its event loop rather than from inside the hook, where the code
    # that raised the exception has not finished failing yet.
    exception_reporter.exception_reported.connect(
        window.report_unhandled_exception,
        QtCore.Qt.ConnectionType.QueuedConnection,
    )
    watchdog = ballroom_scribe.gui.FreezeWatchdog(session_log, app)
    watchdog.start()

    window.show()
    if cli_args["inputs"]:
        window.add_videos(cli_args["inputs"])
    if cli_args["csv"] is not None:
        window.open_csv(cli_args["csv"])
    exit_status = app.exec()
    watchdog.stop()
    if exit_status == 0:
        # Tells the supervisor that the window was closed in good order: if the process dies after this, while Qt
        # and Python shut down, the user has lost nothing and is not bothered with a report.
        with contextlib.suppress(OSError):
            session_logs.closed_normally_marker.touch()
    return exit_status


if __name__ == "__main__":
    raise SystemExit(run_editor())
