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
        events: Iterable[tuple[pathlib.Path, ballroom_scribe.models.Event]],
        target: pathlib.Path,
    ) -> None:
        """Writes events to a CSV file.

        Args:
            events: Pairs of the input video in which each event was detected and the event itself, in output order.
            target: The path to the output CSV file, which is overwritten.
        """
        with open(target, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.COLUMNS)
            for input_video, event in events:
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
