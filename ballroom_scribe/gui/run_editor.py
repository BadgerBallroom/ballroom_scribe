"""The process of the timestamp editor: sets up its logs and runs the main window until it is closed.

`edit_timestamps.py` is what users run. To run the editor from this module instead, run
`python -m ballroom_scribe.gui.run_editor` in the folder that contains `edit_timestamps.py`.
"""

import faulthandler
import logging
import sys
from collections.abc import Sequence

import ballroom_scribe.gui
import ballroom_scribe.gui.utils
import ballroom_scribe.utils

_LOGGER = logging.getLogger(__name__)


def run_editor(argv: Sequence[str] | None = None) -> int:
    """Parses CLI arguments and runs the Ballroom Scribe timestamp editor until its window is closed.

    Everything that the editor logs goes to the log file of this session, whatever `--verbose` says, and so does the
    stack of every thread if the process dies of a fatal error or its GUI thread freezes.

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
    window = ballroom_scribe.gui.MainWindow()
    watchdog = ballroom_scribe.gui.FreezeWatchdog(session_log, app)
    watchdog.start()

    window.show()
    if cli_args["inputs"]:
        window.add_videos(cli_args["inputs"])
    if cli_args["csv"] is not None:
        window.open_csv(cli_args["csv"])
    exit_status = app.exec()
    watchdog.stop()
    return exit_status


if __name__ == "__main__":
    raise SystemExit(run_editor())
