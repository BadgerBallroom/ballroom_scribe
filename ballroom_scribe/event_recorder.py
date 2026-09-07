"""Persistence and export of extracted events."""

import csv
import pathlib
from collections.abc import Iterable

import ballroom_scribe.models
import ballroom_scribe.utils


class EventRecorder:
    """Exports ballroom competition events to a CSV file."""

    COLUMNS = (
        "File",
        "Inpoint",
        "Inpoint Formatted",
        "Level",
        "Style",
        "Dance",
        "Round",
    )

    def record(
        self,
        input_video: pathlib.Path,
        events: Iterable[ballroom_scribe.models.Event],
        target: pathlib.Path,
    ) -> None:
        """Writes events to a CSV file."""
        with open(target, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.COLUMNS)
            for event in events:
                writer.writerow(
                    (
                        input_video,
                        event.announcement_start_seconds,
                        ballroom_scribe.utils.format_duration(event.announcement_start_seconds),
                        event.level,
                        event.style,
                        event.dance,
                        event.round,
                    )
                )
