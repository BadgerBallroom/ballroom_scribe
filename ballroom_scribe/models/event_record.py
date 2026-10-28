"""Data model for a single row in an exported events CSV file."""

import pathlib
from dataclasses import dataclass
from typing import Self

import ballroom_scribe.utils

from .event import Event


@dataclass(frozen=True, slots=True)
class EventRecord:
    """Represents a single persisted event row in an events CSV file.

    Attributes:
        file: Path to the video file containing the event, as stored in the CSV's `File` column.
        inpoint_seconds: Event start position in seconds from the beginning of `file`.
        level: The ballroom proficiency level, such as bronze, silver, gold, or novice.
        style: The category of the dance, such as smooth, standard, rhythm, or Latin.
        dance: The dance itself, such as waltz, quickstep, rumba, or jive.
        round: The competition round, such as quarterfinal, semifinal, or final.
    """

    file: pathlib.Path
    inpoint_seconds: float
    level: str
    style: str
    dance: str
    round: str

    @property
    def inpoint_formatted(self) -> str:
        """Returns `inpoint_seconds` formatted as `HH:MM:SS.cc`."""
        return ballroom_scribe.utils.format_duration(self.inpoint_seconds)

    @classmethod
    def from_event(cls, input_video: pathlib.Path, event: Event) -> Self:
        """Constructs an `EventRecord` from an input video path and a detected `Event`.

        Args:
            input_video: Path to the video file in which `event` was detected.
            event: The detected event.

        Returns:
            A new `EventRecord` populated from `input_video` and `event`.
        """
        return cls(
            file=input_video,
            inpoint_seconds=event.announcement_start_seconds,
            level=event.level,
            style=event.style,
            dance=event.dance,
            round=event.round,
        )
