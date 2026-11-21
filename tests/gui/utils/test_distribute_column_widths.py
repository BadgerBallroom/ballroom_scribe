import unittest

from parameterized import param, parameterized

import ballroom_scribe.gui.utils


class TestDistributeColumnWidths(unittest.TestCase):
    """Unit tests for `distribute_column_widths`."""

    def test_empty_columns_returns_empty_list(self):
        self.assertEqual(ballroom_scribe.gui.utils.distribute_column_widths(500, (), ()), [])

    def test_returns_min_widths_when_total_is_at_or_below_sum_of_minimums(self):
        mins = (52, 52, 90, 90)
        self.assertEqual(
            ballroom_scribe.gui.utils.distribute_column_widths(284, (1.0, 1.0, 1.0, 1.0), mins), list(mins)
        )
        self.assertEqual(
            ballroom_scribe.gui.utils.distribute_column_widths(100, (1.0, 1.0, 1.0, 1.0), mins), list(mins)
        )

    def test_equal_weights_keep_columns_within_one_pixel_and_match_total(self):
        mins = [90] * 7
        widths = ballroom_scribe.gui.utils.distribute_column_widths(703, [1.0] * 7, mins)

        self.assertEqual(sum(widths), 703)
        self.assertLessEqual(max(widths) - min(widths), 1)
        self.assertTrue(all(width >= 90 for width in widths))

    def test_proportional_weights_scale_narrow_and_wide_columns_cleanly(self):
        mins = [52, 52, 52, 90, 90, 90, 52, 90]
        weights = [float(width) for width in mins]
        widths = ballroom_scribe.gui.utils.distribute_column_widths(1000, weights, mins)

        self.assertEqual(sum(widths), 1000)
        narrow = [widths[i] for i, m in enumerate(mins) if m == 52]
        wide = [widths[i] for i, m in enumerate(mins) if m == 90]
        self.assertLessEqual(max(narrow) - min(narrow), 1)
        self.assertLessEqual(max(wide) - min(wide), 1)
        self.assertGreater(min(wide), max(narrow))

    def test_pins_columns_below_their_minimum_without_exceeding_total(self):
        # First two columns have tiny weights that fall below their 52 px minimum; remaining width goes to the third.
        widths = ballroom_scribe.gui.utils.distribute_column_widths(300, (1.0, 1.0, 100.0), (52, 52, 90))

        self.assertEqual(widths, [52, 52, 196])
        self.assertEqual(sum(widths), 300)

    def test_all_zero_weights_falls_back_to_equal_shares_respecting_minimums(self):
        widths = ballroom_scribe.gui.utils.distribute_column_widths(300, (0.0, 0.0, 0.0), (52, 90, 90))

        self.assertEqual(widths, [100, 100, 100])

    @parameterized.expand(
        [
            param("mismatched_lengths", weights=(1.0, 1.0), mins=(52,), regex="same length"),
            param("negative_weight", weights=(1.0, -0.5), mins=(52, 90), regex="non-negative"),
        ]
    )
    def test_invalid_arguments_raise_value_error(self, _, weights, mins, regex: str):
        with self.assertRaisesRegex(ValueError, regex):
            ballroom_scribe.gui.utils.distribute_column_widths(300, weights, mins)


if __name__ == "__main__":
    unittest.main()
