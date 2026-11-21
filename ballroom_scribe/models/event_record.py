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
        event_number: The event number in the competition program, if any.
        flight: The flight number or ordinal within the event, if any.
        heat: The heat number or ordinal within the round, if any.
        age_division: The age division, such as youth, adult, senior, or all ages.
        category: The competitor category, such as amateur, professional, pro-am, or rookie/vet.
        level: The ballroom proficiency level, such as bronze, silver, gold, or novice.
        style: The category of the dance, such as smooth, standard, rhythm, or Latin.
        dance: The dance itself, such as waltz, quickstep, rumba, or jive.
        dance_ordinal: Which dance in a multi-dance sequence is being danced, such as first or second.
        specialness: Non-competitive event designation, such as exhibition, critique, fun, or showcase.
        round: The competition round, such as quarterfinal, semifinal, or final.
        comment: Free-text annotation for human readers, possibly spanning several lines; never used by the pipeline.
    """

    file: pathlib.Path
    inpoint_seconds: float
    event_number: str = ""
    flight: str = ""
    heat: str = ""
    age_division: str = ""
    category: str = ""
    level: str = ""
    style: str = ""
    dance: str = ""
    dance_ordinal: str = ""
    specialness: str = ""
    round: str = ""
    comment: str = ""

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
            event_number=event.event_number,
            flight=event.flight,
            heat=event.heat,
            age_division=event.age_division,
            category=event.category,
            level=event.level,
            style=event.style,
            dance=event.dance,
            dance_ordinal=event.dance_ordinal,
            specialness=event.specialness,
            round=event.round,
        )
