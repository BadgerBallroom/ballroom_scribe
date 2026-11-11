"""How the editor treats an opened video whose file name matches an offline video."""

import enum


class RelinkPolicy(enum.Enum):
    """What `MainWindow.add_videos` does with a file whose name matches an offline video.

    Attributes:
        ASK: Prompt for each such file; an "… All" answer settles the rest of the batch (the GUI default).
        RELINK: Replace the offline entry in place and rewrite its rows' `File` paths to the opened file.
        ADD_NEW: Add the file as a separate video and leave the offline entry alone.
        SKIP: Do not open the file.
    """

    ASK = "ask"
    RELINK = "relink"
    ADD_NEW = "add_new"
    SKIP = "skip"
