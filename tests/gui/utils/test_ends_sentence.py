import unittest

from parameterized import param, parameterized

import ballroom_scribe.gui.utils


class TestEndsSentence(unittest.TestCase):
    """Unit tests for `ends_sentence`."""

    @parameterized.expand(
        [
            param("period", text="floor.", expected=True),
            param("exclamation_mark", text="Congratulations!", expected=True),
            param("question_mark", text="ready?", expected=True),
            param("several_periods", text="and...", expected=True),
            param("abbreviation", text="Mr.", expected=True),
            param("closing_double_quote", text='floor."', expected=True),
            param("closing_single_quote", text="floor.'", expected=True),
            param("closing_curly_quote", text="floor.\u201d", expected=True),
            param("closing_bracket", text="again!)", expected=True),
            param("several_closing_marks", text='ready?")', expected=True),
            param("comma", text="ladies,", expected=False),
            param("colon", text="following:", expected=False),
            param("bare_word", text="waltz", expected=False),
            param("period_inside_a_number", text="3.5", expected=False),
            param("closing_quote_without_punctuation", text='waltz"', expected=False),
            param("opening_quote_after_punctuation", text='."Next', expected=False),
            param("empty", text="", expected=False),
        ]
    )
    def test_ends_sentence(self, _, text, expected):
        self.assertEqual(ballroom_scribe.gui.utils.ends_sentence(text), expected)


if __name__ == "__main__":
    unittest.main()
