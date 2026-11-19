import unittest

from parameterized import param, parameterized

import ballroom_scribe.gui.utils
import ballroom_scribe.models


def _caption(text: str, start_seconds: float, end_seconds: float) -> ballroom_scribe.models.TranscriptChunk:
    """Helper to create a caption with one word, at 1,000 samples per second."""
    return ballroom_scribe.models.TranscriptChunk(
        start_samples=round(start_seconds * 1000),
        sample_rate=1000,
        duration_samples=round((end_seconds - start_seconds) * 1000),
        channel=0,
        words=(ballroom_scribe.models.Word(word=text, start=start_seconds, end=end_seconds),),
    )


# "second" starts when "first" ends, and nothing is said between "second" and "third".
_CAPTIONS = (_caption("first", 1.0, 2.0), _caption("second", 2.0, 2.5), _caption("third", 4.0, 5.0))


class TestFindCaption(unittest.TestCase):
    """Unit tests for `find_caption`."""

    @parameterized.expand(
        [
            param("before_the_first_caption", seconds=0.5, expected=None),
            param("at_the_start_of_a_caption", seconds=1.0, expected="first"),
            param("inside_a_caption", seconds=1.5, expected="first"),
            param("just_before_the_end_of_a_caption", seconds=1.999, expected="first"),
            param("at_the_end_of_a_caption_where_the_next_one_starts", seconds=2.0, expected="second"),
            param("at_the_end_of_a_caption_before_a_gap", seconds=2.5, expected=None),
            param("between_captions", seconds=3.0, expected=None),
            param("inside_the_last_caption", seconds=4.5, expected="third"),
            param("at_the_end_of_the_last_caption", seconds=5.0, expected=None),
            param("after_the_last_caption", seconds=60.0, expected=None),
        ]
    )
    def test_finds_the_caption_at_a_position(self, _, seconds, expected):
        caption = ballroom_scribe.gui.utils.find_caption(_CAPTIONS, seconds)

        self.assertEqual(None if caption is None else caption.get_text(), expected)

    def test_no_captions(self):
        self.assertIsNone(ballroom_scribe.gui.utils.find_caption((), 1.0))


if __name__ == "__main__":
    unittest.main()
