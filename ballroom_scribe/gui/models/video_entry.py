"""GUI model describing one session video as listed in the Manage Videos dialog."""

import pathlib
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VideoEntry:
    """One video of the session as the Manage Videos dialog presents it.

    Attributes:
        csv_file: The session key of the video (its CSV `File` value).
        label: The name shown in the Video drop-down, including any `(offline)` or folder suffix.
        row_count: How many table rows belong to the video, so the effect of removing or regrouping it is clear.
    """

    csv_file: pathlib.Path
    label: str
    row_count: int
