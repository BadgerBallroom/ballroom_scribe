import pathlib
import unittest
from unittest.mock import Mock

from parameterized import param, parameterized

import ballroom_scribe.gui.utils
import ballroom_scribe.models

_VIDEO_A = pathlib.Path("/videos/a.mp4")
_RELATIVE_A = pathlib.Path("a.mp4")
_VIDEO_B = pathlib.Path("/videos/b.mp4")
_UNKNOWN_X = pathlib.Path("/videos/x.mp4")
_UNKNOWN_Y = pathlib.Path("/videos/y.mp4")
_RANKS = {_VIDEO_A: 0, _RELATIVE_A: 0, _VIDEO_B: 1}


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


class TestSortRowsByInpoint(unittest.TestCase):
    """Unit tests for `sort_rows_by_inpoint`."""

    @parameterized.expand(
        [
            param(
                "sorts_a_video_among_its_own_positions_keeping_the_interleaving",
                records=(_event(_VIDEO_A, 30.0), _event(_VIDEO_B, 5.0), _event(_VIDEO_A, 10.0)),
                expected=(2, 1, 0),
            ),
            param(
                "sorts_each_video_independently",
                records=(_event(_VIDEO_A, 30.0), _event(_VIDEO_A, 10.0), _event(_VIDEO_B, 9.0), _event(_VIDEO_B, 5.0)),
                expected=(1, 0, 3, 2),
            ),
            param(
                "relative_and_absolute_spellings_sort_together",
                records=(_event(_VIDEO_A, 30.0), _event(_RELATIVE_A, 10.0)),
                expected=(1, 0),
            ),
            param(
                "equal_inpoints_keep_their_order",
                records=(_event(_VIDEO_A, 10.0), _event(_VIDEO_A, 10.0), _event(_VIDEO_A, 5.0)),
                expected=(2, 0, 1),
            ),
            param(
                "unranked_files_sort_per_distinct_file",
                records=(
                    _event(_UNKNOWN_X, 9.0),
                    _event(_UNKNOWN_Y, 1.0),
                    _event(_UNKNOWN_X, 2.0),
                    _event(_UNKNOWN_Y, 0.0),
                ),
                expected=(2, 3, 0, 1),
            ),
            param(
                "already_sorted_rows_give_the_identity",
                records=(_event(_VIDEO_A, 10.0), _event(_VIDEO_A, 20.0), _event(_VIDEO_B, 5.0)),
                expected=(0, 1, 2),
            ),
            param("empty_table", records=(), expected=()),
        ]
    )
    def test_sorts_each_video_by_inpoint(self, _, records, expected: tuple[int, ...]):
        self.assertEqual(ballroom_scribe.gui.utils.sort_rows_by_inpoint(records, _rank), expected)

    def test_consults_video_rank_once_per_distinct_file(self):
        rank = Mock(side_effect=_rank)
        records = (_event(_VIDEO_A, 30.0), _event(_VIDEO_B, 5.0), _event(_VIDEO_A, 10.0), _event(_RELATIVE_A, 1.0))

        ballroom_scribe.gui.utils.sort_rows_by_inpoint(records, rank)

        self.assertEqual([call.args[0] for call in rank.call_args_list], [_VIDEO_A, _VIDEO_B, _RELATIVE_A])


if __name__ == "__main__":
    unittest.main()
