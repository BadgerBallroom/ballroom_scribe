import pathlib
import unittest
from unittest.mock import Mock

from parameterized import param, parameterized

import ballroom_scribe.gui.utils
import ballroom_scribe.models

_VIDEO_A = pathlib.Path("/videos/a.mp4")
_VIDEO_B = pathlib.Path("/videos/b.mp4")
_UNKNOWN = pathlib.Path("/videos/unknown.mp4")
_RANKS = {_VIDEO_A: 0, _VIDEO_B: 1}


def _event(file: pathlib.Path, inpoint_seconds: float) -> ballroom_scribe.models.EventRecord:
    return ballroom_scribe.models.EventRecord(
        file=file,
        inpoint_seconds=inpoint_seconds,
        level="",
        style="",
        dance="",
        round="",
    )


def _rank(file: pathlib.Path) -> int | None:
    return _RANKS.get(file)


_SORTED_TABLE = (_event(_VIDEO_A, 10.0), _event(_VIDEO_A, 20.0), _event(_VIDEO_A, 30.0), _event(_VIDEO_B, 5.0))
_INTERLEAVED_TABLE = (_event(_VIDEO_A, 10.0), _event(_VIDEO_B, 5.0), _event(_VIDEO_A, 30.0))


class TestFindInsertionRow(unittest.TestCase):
    """Unit tests for `find_insertion_row` when adding a row."""

    @parameterized.expand(
        [
            param("between_rows_of_its_video", records=_SORTED_TABLE, new=_event(_VIDEO_A, 25.0), expected=2),
            param("after_its_video_last_row", records=_SORTED_TABLE, new=_event(_VIDEO_A, 31.0), expected=3),
            param("before_its_video_first_row", records=_SORTED_TABLE, new=_event(_VIDEO_A, 1.0), expected=0),
            param("after_a_row_with_the_same_inpoint", records=_SORTED_TABLE, new=_event(_VIDEO_A, 20.0), expected=2),
            param("later_video_block", records=_SORTED_TABLE, new=_event(_VIDEO_B, 1.0), expected=3),
            param("end_of_later_video_block", records=_SORTED_TABLE, new=_event(_VIDEO_B, 9.0), expected=4),
            param(
                "unsorted_block_goes_before_the_first_later_row",
                records=(_event(_VIDEO_A, 30.0), _event(_VIDEO_A, 10.0), _event(_VIDEO_A, 20.0)),
                new=_event(_VIDEO_A, 15.0),
                expected=0,
            ),
            param(
                "interleaved_videos_stay_with_their_video",
                records=_INTERLEAVED_TABLE,
                new=_event(_VIDEO_A, 20.0),
                expected=2,
            ),
            param(
                "video_without_rows_goes_after_lower_ranked_videos",
                records=(_event(_VIDEO_A, 10.0), _event(_VIDEO_A, 20.0)),
                new=_event(_VIDEO_B, 1.0),
                expected=2,
            ),
            param(
                "video_without_rows_goes_before_higher_ranked_videos",
                records=(_event(_VIDEO_B, 10.0), _event(_VIDEO_B, 20.0)),
                new=_event(_VIDEO_A, 99.0),
                expected=0,
            ),
            param(
                "unknown_rows_are_skipped_over",
                records=(_event(_UNKNOWN, 1.0), _event(_VIDEO_A, 10.0), _event(_UNKNOWN, 2.0), _event(_VIDEO_B, 3.0)),
                new=_event(_VIDEO_A, 50.0),
                expected=2,
            ),
            param(
                "only_unknown_rows_appends",
                records=(_event(_UNKNOWN, 1.0), _event(_UNKNOWN, 2.0)),
                new=_event(_VIDEO_A, 5.0),
                expected=2,
            ),
            param("unknown_new_video_appends", records=_SORTED_TABLE, new=_event(_UNKNOWN, 0.0), expected=4),
            param("empty_table", records=(), new=_event(_VIDEO_B, 5.0), expected=0),
        ]
    )
    def test_places_the_event(self, _, records, new, expected: int):
        self.assertEqual(ballroom_scribe.gui.utils.find_insertion_row(records, new, _rank), expected)

    def test_consults_video_rank_once_per_distinct_file(self):
        rank = Mock(side_effect=_rank)

        ballroom_scribe.gui.utils.find_insertion_row(_SORTED_TABLE, _event(_VIDEO_A, 25.0), rank)

        self.assertEqual(sorted(call.args[0] for call in rank.call_args_list), sorted([_VIDEO_A, _VIDEO_A, _VIDEO_B]))


class TestFindInsertionRowReplacing(unittest.TestCase):
    """Unit tests for `find_insertion_row` when a row's contents change (`replacing_row`)."""

    @parameterized.expand(
        [
            param("still_in_order_stays", records=_SORTED_TABLE, row=1, new=_event(_VIDEO_A, 25.0), expected=1),
            param(
                "equal_to_the_next_inpoint_stays", records=_SORTED_TABLE, row=1, new=_event(_VIDEO_A, 30.0), expected=1
            ),
            param(
                "equal_to_the_previous_inpoint_stays",
                records=_SORTED_TABLE,
                row=1,
                new=_event(_VIDEO_A, 10.0),
                expected=1,
            ),
            param(
                "later_than_the_rows_below_moves_down",
                records=_SORTED_TABLE,
                row=0,
                new=_event(_VIDEO_A, 25.0),
                expected=1,
            ),
            param(
                "latest_of_its_video_moves_to_the_block_end",
                records=_SORTED_TABLE,
                row=0,
                new=_event(_VIDEO_A, 99.0),
                expected=2,
            ),
            param(
                "earlier_than_the_rows_above_moves_up",
                records=_SORTED_TABLE,
                row=2,
                new=_event(_VIDEO_A, 1.0),
                expected=0,
            ),
            param(
                "other_video_with_rows_joins_that_block",
                records=_SORTED_TABLE,
                row=1,
                new=_event(_VIDEO_B, 9.0),
                expected=3,
            ),
            param(
                "other_video_without_rows_goes_after_lower_ranked_rows",
                records=(_event(_VIDEO_A, 10.0), _event(_VIDEO_A, 20.0), _event(_VIDEO_A, 30.0)),
                row=1,
                new=_event(_VIDEO_B, 1.0),
                expected=2,
            ),
            param(
                "interleaved_but_in_order_stays",
                records=_INTERLEAVED_TABLE,
                row=0,
                new=_event(_VIDEO_A, 20.0),
                expected=0,
            ),
            param(
                "interleaved_and_out_of_order_moves_next_to_its_video",
                records=_INTERLEAVED_TABLE,
                row=0,
                new=_event(_VIDEO_A, 40.0),
                expected=2,
            ),
            param("unknown_video_stays", records=_SORTED_TABLE, row=1, new=_event(_UNKNOWN, 0.0), expected=1),
        ]
    )
    def test_keeps_or_moves_the_row(self, _, records, row: int, new, expected: int):
        self.assertEqual(ballroom_scribe.gui.utils.find_insertion_row(records, new, _rank, replacing_row=row), expected)


if __name__ == "__main__":
    unittest.main()
