import unittest

from parameterized import param, parameterized

import ballroom_scribe.utils


class TestFormatDuration(unittest.TestCase):
    """Unit tests for the `format_duration` function."""

    @parameterized.expand(
        [
            param(
                "zero",
                seconds=0.0,
                expected="0:00:00.00",
            ),
            param(
                "rounds_centiseconds",
                seconds=1.236,
                expected="0:00:01.24",
            ),
            param(
                "rolls_over_to_next_minute",
                seconds=59.999,
                expected="0:01:00.00",
            ),
            param(
                "single_digit_hours_minutes_seconds_centiseconds",
                seconds=3661.05,
                expected="1:01:01.05",
            ),
            param(
                "two_digit_hours",
                seconds=36000.0,
                expected="10:00:00.00",
            ),
            param(
                "hours_exceeding_two_digits_not_wrapped_to_days",
                seconds=360000.0,
                expected="100:00:00.00",
            ),
        ]
    )
    def test_format_duration(self, _, seconds: float, expected: str):
        self.assertEqual(ballroom_scribe.utils.format_duration(seconds), expected)


if __name__ == "__main__":
    unittest.main()
