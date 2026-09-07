import unittest

from parameterized import param, parameterized

import ballroom_scribe.utils


class TestFindClosestMatching(unittest.TestCase):
    """Unit tests for the `find_closest_matching` function."""

    def test_empty_haystack(self):
        with self.assertRaises(ValueError):
            ballroom_scribe.utils.find_closest_matching(needle=[0], haystack=[])

    @parameterized.expand(
        [
            param(
                "empty_needle",
                needle=[],
                haystack=[0],
                expected_index=0,
            ),
            param(
                "needle_equals_haystack",
                needle=[1, 2, 3, 4],
                haystack=[1, 2, 3, 4],
                expected_index=0,
            ),
            param(
                "same_lengths",
                needle=[1, 2, 3],
                haystack=[3, 4, 5],
                expected_index=0,
            ),
            param(
                "needle_longer_than_haystack",
                needle=[1, 2, 3],
                haystack=[4, 5],
                expected_index=0,
            ),
            param(
                "needle_length_1",
                needle=[2],
                haystack=[0, 1, 2, 3, 4],
                expected_index=2,
            ),
            param(
                "needle_exactly_matches_beginning",
                needle=[1, 2, 3],
                haystack=[1, 2, 3, 4],
                expected_index=0,
            ),
            param(
                "needle_partially_matches_beginning",
                needle=[1, 2, 3],
                haystack=[1, 2, 4, 5],
                expected_index=0,
            ),
            param(
                "needle_starts_before_haystack",
                needle=[1, 2, 3],
                haystack=[2, 3, 4, 5],
                expected_index=0,
            ),
            param(
                "needle_exactly_matches_middle",
                needle="gor",
                haystack="algorithm",
                expected_index=2,
            ),
            param(
                "needle_partially_matches_middle",
                needle="stay",
                haystack="haystack",
                expected_index=3,
            ),
            param(
                "needle_exactly_matches_end",
                needle="string",
                haystack="substring",
                expected_index=3,
            ),
            param(
                "needle_partially_matches_end",
                needle="sellers",
                haystack="wagnerfischer",
                expected_index=8,
            ),
            param(
                "needle_ends_after_haystack",
                needle="badgers",
                haystack="honey badger",
                expected_index=6,
            ),
        ]
    )
    def test_find_closest_matching(self, _, needle, haystack, expected_index):
        index_in_needle = ballroom_scribe.utils.find_closest_matching(needle=needle, haystack=haystack)
        self.assertEqual(index_in_needle, expected_index)


if __name__ == "__main__":
    unittest.main()
