"""Sorts each video's event rows by `Inpoint` without moving rows between videos."""

import pathlib
from collections.abc import Callable, Sequence

import ballroom_scribe.models


def sort_rows_by_inpoint(
    records: Sequence[ballroom_scribe.models.EventRecord],
    video_rank: Callable[[pathlib.Path], int | None],
) -> tuple[int, ...]:
    """Returns the row indices with each video's rows sorted by `Inpoint` among the positions that video occupies.

    Rows are never moved between videos' positions, so interleaved videos stay interleaved: `A@30, B@5, A@10`
    becomes `A@10, B@5, A@30`. Rows of one video are those sharing a non-`None` `video_rank`; rows whose `File`
    matches no video form one group per distinct `File`. Equal inpoints keep their current order, and `video_rank`
    is consulted once per distinct `File`.

    Args:
        records: The table's rows, in their current order.
        video_rank: Maps a `File` path to the position of its video in the session, or `None` when the path
            matches no video.

    Returns:
        A permutation of the row indices: new row `i` holds the record now at index `i` of the result.
    """
    ranks: dict[pathlib.Path, int | None] = {}
    groups: dict[int | pathlib.Path, list[int]] = {}
    for row, record in enumerate(records):
        if record.file not in ranks:
            ranks[record.file] = video_rank(record.file)
        rank = ranks[record.file]
        groups.setdefault(rank if rank is not None else record.file, []).append(row)

    order = list(range(len(records)))
    for positions in groups.values():
        by_inpoint = sorted(positions, key=lambda row: (records[row].inpoint_seconds, row))
        for position, row in zip(positions, by_inpoint, strict=True):
            order[position] = row
    return tuple(order)
