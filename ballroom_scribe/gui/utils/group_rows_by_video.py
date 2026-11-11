"""Rearranges event rows so each video's rows sit together, in the session's video order."""

import pathlib
from collections.abc import Callable, Sequence

import ballroom_scribe.models


def group_rows_by_video(
    records: Sequence[ballroom_scribe.models.EventRecord],
    video_rank: Callable[[pathlib.Path], int | None],
) -> tuple[int, ...]:
    """Returns the row indices rearranged so each video's rows are together, videos in `video_rank` order.

    Ranked videos come first, by rank; rows whose `File` matches no video (`video_rank` returns `None`) follow,
    grouped by `File` in order of first appearance. Within a group the rows keep their current relative order (a
    stable sort by group). `video_rank` is consulted once per distinct `File`.

    Args:
        records: The table's rows, in their current order.
        video_rank: Maps a `File` path to the position of its video in the session (the Video drop-down order),
            or `None` when the path matches no video.

    Returns:
        A permutation of the row indices: new row `i` holds the record now at index `i` of the result.
    """
    ranks: dict[pathlib.Path, int | None] = {}
    first_seen: dict[pathlib.Path, int] = {}

    def sort_key(row: int) -> tuple[int, int, int]:
        file = records[row].file
        if file not in ranks:
            ranks[file] = video_rank(file)
            first_seen[file] = len(first_seen)
        rank = ranks[file]
        return (0, rank, row) if rank is not None else (1, first_seen[file], row)

    return tuple(sorted(range(len(records)), key=sort_key))
