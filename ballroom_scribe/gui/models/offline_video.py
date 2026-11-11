"""GUI model representing a video referenced by the events CSV that could not be opened."""

import pathlib
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OfflineVideo:
    """A video the events CSV refers to whose media could not be opened (missing, moved, unreadable, …).

    It keeps its place in the Video drop-down so its rows can still be browsed and edited until the media is located.

    Attributes:
        csv_file: Video path as written in the CSV's `File` column; the session key, like `LoadedVideo.csv_file`.
        input_video: Where the file was expected on disk (`csv_file`, resolved against the CSV's folder if relative).
        reason: Why it could not be opened, e.g. `"File not found"` or an `ffprobe` error message.
    """

    csv_file: pathlib.Path
    input_video: pathlib.Path
    reason: str
