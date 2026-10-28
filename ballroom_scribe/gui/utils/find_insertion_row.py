"""Chooses where an event row goes so each video's events stay together and in `Inpoint` order."""

import pathlib
from collections.abc import Callable, Sequence

import ballroom_scribe.models

type _VideoRank = Callable[[pathlib.Path], int | None]


def _ranks(records: Sequence[ballroom_scribe.models.EventRecord], video_rank: _VideoRank) -> list[int | None]:
    """Ranks every row's video, consulting `video_rank` once per distinct `File` path."""
    by_file: dict[pathlib.Path, int | None] = {}
    ranks: list[int | None] = []
    for record in records:
        if record.file not in by_file:
            by_file[record.file] = video_rank(record.file)
        ranks.append(by_file[record.file])
    return ranks


def _is_in_order(
    records: Sequence[ballroom_scribe.models.EventRecord],
    ranks: Sequence[int | None],
    row: int,
    record: ballroom_scribe.models.EventRecord,
) -> bool:
    """Returns whether `record`, sitting at `row`, is in `Inpoint` order among the other rows of its video."""
    for other_row, (other, other_rank) in enumerate(zip(records, ranks, strict=True)):
        if other_row == row or other_rank != ranks[row]:
            continue
        if other_row < row and other.inpoint_seconds > record.inpoint_seconds:
            return False
        if other_row > row and other.inpoint_seconds < record.inpoint_seconds:
            return False
    return True


def _placement(
    records: Sequence[ballroom_scribe.models.EventRecord],
    ranks: Sequence[int | None],
    new_record: ballroom_scribe.models.EventRecord,
    new_rank: int,
) -> int:
    same_video = [row for row, rank in enumerate(ranks) if rank == new_rank]
    if same_video:
        later = [row for row in same_video if records[row].inpoint_seconds > new_record.inpoint_seconds]
        return later[0] if later else same_video[-1] + 1
    earlier_videos = [row for row, rank in enumerate(ranks) if rank is not None and rank < new_rank]
    if earlier_videos:
        return earlier_videos[-1] + 1
    later_videos = [row for row, rank in enumerate(ranks) if rank is not None and rank > new_rank]
    return later_videos[0] if later_videos else len(records)


def find_insertion_row(
    records: Sequence[ballroom_scribe.models.EventRecord],
    new_record: ballroom_scribe.models.EventRecord,
    video_rank: _VideoRank,
    *,
    replacing_row: int | None = None,
) -> int:
    """Chooses the row at which `new_record` keeps each video's events together and in `Inpoint` order.

    Args:
        records: The table's rows, in their current order.
        new_record: The event being added, or the new contents of the row `replacing_row`.
        video_rank: Maps a `File` path to the position of its video in the session (the Video drop-down order),
            or `None` when the path matches no loaded video.
        replacing_row: When `new_record` replaces an existing row rather than being added, that row's index. The
            row keeps its place while it still belongs to the same video and is in order among that video's other
            rows (equal inpoints included); otherwise the result is its index after moving among the other rows.

    Returns:
        The index to insert at (or, when replacing, the row's index afterwards): within the event's video, before
        its first row with a later inpoint (after every row with the same or an earlier inpoint), so a block that
        is in order stays in order; for a video without rows, after the rows of lower-ranked videos, else before
        the rows of higher-ranked videos, else the end of the table. Rows of unknown videos never influence the
        choice, and an event of an unknown video is appended (or, when replacing, left where it is).
    """
    new_rank = video_rank(new_record.file)
    if new_rank is None:
        return len(records) if replacing_row is None else replacing_row
    ranks = _ranks(records, video_rank)
    if replacing_row is not None:
        if ranks[replacing_row] == new_rank and _is_in_order(records, ranks, replacing_row, new_record):
            return replacing_row
        records = [record for row, record in enumerate(records) if row != replacing_row]
        ranks = [rank for row, rank in enumerate(ranks) if row != replacing_row]
    return _placement(records, ranks, new_record, new_rank)
