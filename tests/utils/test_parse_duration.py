import unittest

from parameterized import param, parameterized

import ballroom_scribe.utils


class TestParseDuration(unittest.TestCase):
    """Unit tests for the `parse_duration` function."""

    @parameterized.expand(
        [
            param("zero_int", text="0", expected=0.0),
            param("plain_seconds", text="42.375", expected=42.375),
            param("plain_seconds_over_minute", text="125.5", expected=125.5),
            param("surrounding_whitespace", text="  1:05.25  ", expected=65.25),
            param("minutes_and_seconds", text="02:03", expected=123.0),
            param("hours_minutes_seconds", text="1:01:01.05", expected=3661.05),
            param("zero_hours_minutes_seconds", text="0:00:12.35", expected=12.35),
            param("triple_digit_hours", text="100:00:00.00", expected=360000.0),
        ]
    )
    def test_valid_durations(self, _, text: str, expected: float):
        self.assertAlmostEqual(ballroom_scribe.utils.parse_duration(text), expected)

    @parameterized.expand(
        [
            param("empty", text=""),
            param("whitespace_only", text="   "),
            param("negative_seconds", text="-5"),
            param("plus_prefix", text="+5"),
            param("non_numeric", text="abc"),
            param("too_many_colons", text="1:02:03:04"),
            param("empty_component", text="1::02"),
            param("trailing_dot", text="12."),
            param("leading_dot", text=".5"),
            param("seconds_sixty_in_mm_ss", text="01:60"),
            param("minutes_sixty_in_mm_ss", text="60:00"),
            param("minutes_sixty_in_hh_mm_ss", text="1:60:00"),
            param("seconds_sixty_in_hh_mm_ss", text="1:00:60.00"),
            param("scientific_notation", text="1e3"),
            param("nan", text="nan"),
        ]
    )
    def test_invalid_durations(self, _, text: str):
        with self.assertRaises(ValueError):
            ballroom_scribe.utils.parse_duration(text)

    @parameterized.expand(
        [
            param("zero", seconds=0.0),
            param("sub_minute", seconds=12.34),
            param("multi_minute", seconds=754.89),
            param("multi_hour", seconds=3661.05),
            param("hundred_hours", seconds=360000.0),
        ]
    )
    def test_round_trip_with_format_duration(self, _, seconds: float):
        formatted = ballroom_scribe.utils.format_duration(seconds)
        self.assertAlmostEqual(ballroom_scribe.utils.parse_duration(formatted), seconds)


if __name__ == "__main__":
    unittest.main()
