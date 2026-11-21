"""Function for parsing the command-line arguments of the timestamp editor."""

from collections.abc import Sequence

import ballroom_scribe.cli.arguments

# What the user ran, whichever process parses the arguments.
_PROGRAM_NAME = "edit_timestamps.py"


def parse_editor_args(
    argv: Sequence[str] | None = None,
) -> tuple[ballroom_scribe.cli.arguments.InputCsvCacheArgs, ballroom_scribe.cli.arguments.LoggingArgs]:
    """Parses the command-line arguments of `edit_timestamps.py`.

    Prints a message and exits the process if the arguments ask for help or are mistaken, as `argparse` does.

    Args:
        argv: The command-line arguments, or `None` for those of the process.

    Returns:
        The videos and the events CSV to open, and the logging options.
    """
    return ballroom_scribe.cli.arguments.parse_args(
        (ballroom_scribe.cli.arguments.InputCsvCacheGroup(), ballroom_scribe.cli.arguments.LoggingGroup()),
        argv,
        prog=_PROGRAM_NAME,
        description="Review, generate, and edit ballroom competition event CSV files.",
    )
