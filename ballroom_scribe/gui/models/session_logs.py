"""GUI model naming the files in which one run of the editor keeps its logs."""

import pathlib
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class SessionLogs:
    """Represents the folder in which one run of the timestamp editor keeps its logs.

    Attributes:
        ENVIRONMENT_VARIABLE: The environment variable in which a process names `directory` for the processes that
            it starts, so that they keep their logs in the same folder.
        directory: The folder of this run, inside the user's folder for logs.
    """

    ENVIRONMENT_VARIABLE: ClassVar[str] = "BALLROOM_SCRIBE_SESSION_DIR"

    directory: pathlib.Path

    @property
    def editor_log(self) -> pathlib.Path:
        """Returns the file that holds the editor's log records and, if it crashed or froze, its threads' stacks."""
        return self.directory / "editor.log"

    @property
    def console_log(self) -> pathlib.Path:
        """Returns the file in which the editor's supervisor keeps all that the editor writes to its console."""
        return self.directory / "console.log"

    @property
    def closed_normally_marker(self) -> pathlib.Path:
        """Returns the file that the editor creates once its window has been closed in good order."""
        return self.directory / "closed-normally"
