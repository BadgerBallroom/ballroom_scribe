"""Persistence, import, and export of ballroom competition events."""

import csv
import math
import pathlib
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence

import ballroom_scribe.models

_REQUIRED_COLUMNS = (
    "File",
    "Inpoint",
    "Level",
    "Style",
    "Dance",
    "Round",
)


def _validate_fieldnames(fieldnames: Sequence[str] | None) -> tuple[str, ...]:
    """Validates CSV header columns and returns any unknown extra columns."""
    if not fieldnames:
        raise ValueError("CSV file has no header row.")
    present = frozenset(fieldnames)
    missing = [column for column in _REQUIRED_COLUMNS if column not in present]
    if missing:
        missing_list = ", ".join(repr(column) for column in missing)
        raise ValueError(f"CSV file is missing required column(s): {missing_list}.")
    known = frozenset(EventRecorder.COLUMNS)
    return tuple(column for column in fieldnames if column not in known)


def _parse_row(row: Mapping[str, str | None], row_number: int) -> ballroom_scribe.models.EventRecord:
    """Converts a single `csv.DictReader` row into an `EventRecord`."""
    for column in _REQUIRED_COLUMNS:
        if row.get(column) is None:
            raise ValueError(f"Row {row_number}: missing value for column {column!r}.")

    raw_file = row["File"]
    assert raw_file is not None
    if not raw_file.strip():
        raise ValueError(f"Row {row_number}: 'File' cannot be empty.")

    raw_inpoint = row["Inpoint"]
    assert raw_inpoint is not None
    try:
        inpoint_seconds = float(raw_inpoint)
    except ValueError:
        raise ValueError(f"Row {row_number}: invalid 'Inpoint' value {raw_inpoint!r}.") from None
    if not math.isfinite(inpoint_seconds) or inpoint_seconds < 0.0:
        raise ValueError(f"Row {row_number}: 'Inpoint' must be a non-negative finite number, got {raw_inpoint!r}.")

    return ballroom_scribe.models.EventRecord(
        file=pathlib.Path(raw_file),
        inpoint_seconds=inpoint_seconds,
        level=row["Level"] or "",
        style=row["Style"] or "",
        dance=row["Dance"] or "",
        round=row["Round"] or "",
        comment=row.get("Comment") or "",  # Optional: files written before comments existed load without it.
    )


class EventRecorder:
    """Reads and writes ballroom competition events in CSV format."""

    COLUMNS = (
        "File",
        "Inpoint",
        "Inpoint Formatted",
        "Level",
        "Style",
        "Dance",
        "Round",
        "Comment",
    )

    def write(
        self,
        records: Iterable[ballroom_scribe.models.EventRecord],
        target: pathlib.Path,
    ) -> None:
        """Writes event records to a CSV file.

        Comments containing commas, quotes, or line breaks are quoted by the `csv` module.

        Args:
            records: The event records to write, in output order.
            target: The path to the output CSV file, which is overwritten.
        """
        with open(target, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.COLUMNS)
            for record in records:
                writer.writerow(
                    (
                        record.file,
                        record.inpoint_seconds,
                        record.inpoint_formatted,
                        record.level,
                        record.style,
                        record.dance,
                        record.round,
                        record.comment,
                    )
                )

    def record(
        self,
        events: Iterable[tuple[pathlib.Path, ballroom_scribe.models.Event]],
        target: pathlib.Path,
    ) -> None:
        """Writes pipeline `(input_video, event)` pairs to a CSV file.

        Args:
            events: Pairs of the input video in which each event was detected and the event itself, in output order.
            target: The path to the output CSV file, which is overwritten.
        """
        self.write(
            records=(
                ballroom_scribe.models.EventRecord.from_event(input_video, event) for input_video, event in events
            ),
            target=target,
        )

    def read(
        self,
        source: pathlib.Path,
        *,
        on_extra_columns: Callable[[tuple[str, ...]], None] | None = None,
    ) -> Generator[ballroom_scribe.models.EventRecord]:
        """Streams `EventRecord`s from an existing CSV file.

        `Inpoint Formatted` and any extra unknown columns are ignored (`Inpoint Formatted` is recomputed on save), and
        the `Comment` column is optional: files written before it existed load with empty comments.

        Args:
            source: Path to the CSV file to read.
            on_extra_columns: Optional callback invoked once after reading the header if unknown columns are present.

        Yields:
            Parsed `EventRecord` instances in file order.

        Raises:
            ValueError: If the CSV lacks a header or required columns, or if any row is malformed.
        """
        with open(source, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            extra_columns = _validate_fieldnames(reader.fieldnames)
            if extra_columns and on_extra_columns is not None:
                on_extra_columns(extra_columns)
            for row_number, row in enumerate(reader, start=1):
                yield _parse_row(row, row_number)
