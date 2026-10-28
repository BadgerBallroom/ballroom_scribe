import unittest

from parameterized import param, parameterized

try:
    from PySide6 import QtCore  # noqa: F401
except ImportError as err:
    raise unittest.SkipTest("PySide6 is not installed.") from err

import ballroom_scribe.gui.waveform_widget


class TestFormatSpan(unittest.TestCase):
    """Unit tests for `_format_span`, the zoom readout in the waveform ruler."""

    @parameterized.expand(
        [
            param("minimum_zoom", seconds=2.0, expected="2 s"),
            param("fractional_seconds_rounded_to_tenths", seconds=26.832, expected="26.8 s"),
            param("thirty_seconds", seconds=30.0, expected="30 s"),
            param("just_under_a_minute", seconds=59.96, expected="60 s"),
            param("minutes_and_seconds", seconds=90.0, expected="1:30"),
            param("rounds_to_whole_seconds", seconds=119.6, expected="2:00"),
            param("default_zoom", seconds=720.0, expected="12:00"),
            param("hours", seconds=4320.0, expected="1:12:00"),
            param("hours_with_padding", seconds=3 * 3600 + 7 * 60 + 5, expected="3:07:05"),
        ]
    )
    def test_formats_span(self, _, seconds, expected):
        self.assertEqual(ballroom_scribe.gui.waveform_widget._format_span(seconds), expected)


if __name__ == "__main__":
    unittest.main()
