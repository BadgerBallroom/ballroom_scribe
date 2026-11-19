import unittest
from unittest.mock import MagicMock

from parameterized import param, parameterized

try:
    from PySide6 import QtCore
except ImportError as err:
    raise unittest.SkipTest("PySide6 is not installed.") from err

import ballroom_scribe.gui.waveform_widget
import ballroom_scribe.models

_PIXELS_PER_CHARACTER = 6.0
_LINE_HEIGHT = 12
_ELIDE_RIGHT = QtCore.Qt.TextElideMode.ElideRight
# More pixels than any label in these tests is wide: the whole label can be seen.
_ALL_VISIBLE = 100_000.0


def _words(*texts: str) -> tuple[ballroom_scribe.models.Word, ...]:
    """Helper to create words whose timing does not matter."""
    return tuple(ballroom_scribe.models.Word(word=text, start=0.0, end=0.0) for text in texts)


def _metrics(
    pixels_per_character: float = _PIXELS_PER_CHARACTER,
    average_character_width: float | None = None,
) -> MagicMock:
    """Helper to create font metrics in which every character of a text is `pixels_per_character` wide.

    The font's average character is as wide as that unless `average_character_width` says otherwise, and a line of
    text is 12 pixels high. `elidedText` keeps the characters that fit together with an ellipsis that is one
    character wide.
    """

    def elide(text: str, mode: QtCore.Qt.TextElideMode, width: int) -> str:
        del mode
        return text[: max(0, int(width / pixels_per_character) - 1)] + "…"

    metrics = MagicMock()
    metrics.height.return_value = _LINE_HEIGHT
    metrics.averageCharWidth.return_value = (
        pixels_per_character if average_character_width is None else average_character_width
    )
    metrics.horizontalAdvance.side_effect = lambda text: len(text) * pixels_per_character
    metrics.elidedText.side_effect = elide
    return metrics


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


class TestStartOfLabel(unittest.TestCase):
    """Unit tests for `_start_of_label`."""

    @parameterized.expand(
        [
            param("no_words", texts=(), min_characters=10, expected=("", True)),
            param("short_label_is_whole", texts=("Thank", "you"), min_characters=100, expected=("Thank you", True)),
            param(
                "stops_with_the_word_that_makes_it_longer",
                texts=("aa", "bb", "cc", "dd"),
                min_characters=4,
                expected=("aa bb", False),
            ),
            param(
                "goes_on_at_exactly_the_minimum",
                texts=("aa", "bb", "cc", "dd"),
                min_characters=5,
                expected=("aa bb cc", False),
            ),
            param(
                "whole_when_the_last_word_makes_it_longer",
                texts=("aa", "bb"),
                min_characters=4,
                expected=("aa bb", True),
            ),
            param(
                "first_word_alone_can_be_longer",
                texts=("Wonderful,", "thanks"),
                min_characters=0,
                expected=("Wonderful,", False),
            ),
        ]
    )
    def test_joins_words_until_longer_than_minimum(self, _, texts, min_characters, expected):
        self.assertEqual(
            ballroom_scribe.gui.waveform_widget._start_of_label(_words(*texts), min_characters),
            expected,
        )


class TestFitLabel(unittest.TestCase):
    """Unit tests for `_fit_label`, with font metrics in which every character is 6 pixels wide by default."""

    def test_returns_label_that_fits(self):
        metrics = _metrics()

        # "Thank you" is 54 pixels wide.
        label = ballroom_scribe.gui.waveform_widget._fit_label(_words("Thank", "you"), 60.0, _ALL_VISIBLE, metrics)

        self.assertEqual(label, "Thank you")
        metrics.elidedText.assert_not_called()

    def test_returns_label_that_is_two_pixels_wider_than_room(self):
        metrics = _metrics()

        label = ballroom_scribe.gui.waveform_widget._fit_label(_words("Thank", "you"), 52.0, _ALL_VISIBLE, metrics)

        self.assertEqual(label, "Thank you")
        metrics.elidedText.assert_not_called()

    def test_cuts_off_label_that_is_too_wide(self):
        metrics = _metrics()

        label = ballroom_scribe.gui.waveform_widget._fit_label(_words("Thank", "you"), 51.9, _ALL_VISIBLE, metrics)

        self.assertEqual(label, "Thank y…")
        metrics.elidedText.assert_called_once_with("Thank you", _ELIDE_RIGHT, 51)

    def test_measures_only_start_of_long_label_that_is_too_wide(self):
        metrics = _metrics()
        # The label has 4,999 characters. Its first four words are 114 pixels wide, which is more than the room.
        words = _words(*["word"] * 1000)

        label = ballroom_scribe.gui.waveform_widget._fit_label(words, 100.0, _ALL_VISIBLE, metrics)

        self.assertEqual(label, "word word word …")
        metrics.horizontalAdvance.assert_called_once_with("word word word word")
        metrics.elidedText.assert_called_once_with("word word word word", _ELIDE_RIGHT, 100)

    def test_returns_start_of_fitting_label_that_runs_past_what_is_visible(self):
        metrics = _metrics()
        # The label fits in its room, as the last label of a waveform always does, but only its first 100 pixels can
        # be seen. Six words are the first to be wider than those and four lines' height more, 148 pixels.
        words = _words(*["word"] * 1000)

        label = ballroom_scribe.gui.waveform_widget._fit_label(words, 50_000.0, 100.0, metrics)

        self.assertEqual(label, "word word word word word word")
        metrics.horizontalAdvance.assert_called_once_with("word word word word word word")
        metrics.elidedText.assert_not_called()

    def test_returns_start_of_wide_label_whose_ellipsis_would_be_out_of_sight(self):
        metrics = _metrics()
        # The label is too wide for its room of 200 pixels, but it would be cut off beyond the 148 pixels that matter
        # to what can be seen.
        words = _words(*["word"] * 1000)

        label = ballroom_scribe.gui.waveform_widget._fit_label(words, 200.0, 100.0, metrics)

        self.assertEqual(label, "word word word word word word")
        metrics.elidedText.assert_not_called()

    def test_cuts_off_label_whose_room_ends_just_past_what_is_visible(self):
        metrics = _metrics()
        # The room of 145 pixels ends within four lines' height of the 100 pixels that can be seen, so the ellipsis
        # may show, although five words would cover what can be seen. Six words are the first to be wider than the
        # room.
        words = _words(*["word"] * 1000)

        label = ballroom_scribe.gui.waveform_widget._fit_label(words, 145.0, 100.0, metrics)

        self.assertEqual(label, "word word word word wor…")
        metrics.elidedText.assert_called_once_with("word word word word word word", _ELIDE_RIGHT, 145)

    @parameterized.expand(
        [
            param("too_little_room_for_text", texts=("Thank", "you"), room=13.9),
            param("negative_room", texts=("Thank",), room=-5.0),
        ]
    )
    def test_returns_nothing_without_room_for_wide_label(self, _, texts, room):
        metrics = _metrics()

        label = ballroom_scribe.gui.waveform_widget._fit_label(_words(*texts), room, _ALL_VISIBLE, metrics)

        self.assertEqual(label, "")
        metrics.elidedText.assert_not_called()

    def test_returns_short_label_that_fits_in_little_room(self):
        # "a" is 6 pixels wide, which is within 2 pixels of the room.
        label = ballroom_scribe.gui.waveform_widget._fit_label(_words("a"), 5.0, _ALL_VISIBLE, _metrics())

        self.assertEqual(label, "a")

    def test_measures_whole_label_when_text_is_narrower_than_the_average(self):
        # Every character is 2 pixels wide in a font whose average character is 6 pixels wide, so the first guesses
        # at how much of the label to measure fall short. The label is 118 pixels wide and fits.
        metrics = _metrics(pixels_per_character=2.0, average_character_width=6.0)
        whole_label = " ".join(["abcd"] * 12)

        label = ballroom_scribe.gui.waveform_widget._fit_label(_words(*["abcd"] * 12), 150.0, _ALL_VISIBLE, metrics)

        self.assertEqual(label, whole_label)
        self.assertEqual(metrics.horizontalAdvance.call_count, 3)
        metrics.elidedText.assert_not_called()

    def test_measures_longer_start_when_text_is_narrower_than_the_average(self):
        # The starts of the label that are measured have 8, 14, and 28 words. The last is 278 pixels wide, which is
        # the first that is wider than the room.
        metrics = _metrics(pixels_per_character=2.0, average_character_width=6.0)
        start_of_label = " ".join(["abcd"] * 28)

        label = ballroom_scribe.gui.waveform_widget._fit_label(_words(*["abcd"] * 100), 200.0, _ALL_VISIBLE, metrics)

        self.assertEqual(label, start_of_label[:99] + "…")
        self.assertEqual(metrics.horizontalAdvance.call_count, 3)
        metrics.elidedText.assert_called_once_with(start_of_label, _ELIDE_RIGHT, 200)


if __name__ == "__main__":
    unittest.main()
