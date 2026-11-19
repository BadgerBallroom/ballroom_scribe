import unittest

from parameterized import param, parameterized

import ballroom_scribe.gui.utils
import ballroom_scribe.models

_SAMPLE_RATE = 1000
_CHANNEL = 1


def _word(text: str, start: float, end: float | None = None) -> ballroom_scribe.models.Word:
    """Helper to create a word that lasts 0.2 seconds unless `end` is given."""
    return ballroom_scribe.models.Word(word=text, start=start, end=start + 0.2 if end is None else end)


def _words(texts: list[str], start: float = 0.0) -> list[ballroom_scribe.models.Word]:
    """Helper to create words that are spoken 0.3 seconds apart, beginning at `start`."""
    return [_word(text, start + 0.3 * index) for index, text in enumerate(texts)]


def _chunk(
    start_samples: int,
    duration_samples: int,
    *words: ballroom_scribe.models.Word,
    sample_rate: int = _SAMPLE_RATE,
) -> ballroom_scribe.models.TranscriptChunk:
    return ballroom_scribe.models.TranscriptChunk(
        start_samples=start_samples,
        sample_rate=sample_rate,
        duration_samples=duration_samples,
        channel=_CHANNEL,
        words=words,
    )


def _caption_texts(words: list[ballroom_scribe.models.Word]) -> list[str]:
    captions = ballroom_scribe.gui.utils.group_words_into_captions([_chunk(0, 60_000, *words)])
    return [caption.get_text() for caption in captions]


class TestGroupWordsIntoCaptions(unittest.TestCase):
    """Unit tests for `group_words_into_captions`."""

    @parameterized.expand(
        [
            param("pause_of_one_second", pause=1.0, expected=["Thank you", "Next up"]),
            param("longer_pause", pause=5.0, expected=["Thank you", "Next up"]),
            param("shorter_pause", pause=0.9, expected=["Thank you Next up"]),
        ]
    )
    def test_caption_ends_at_a_pause(self, _, pause, expected):
        thank_you = [_word("Thank", 0.0), _word("you", 0.3, end=0.5)]
        next_up = [_word("Next", 0.5 + pause), _word("up", 0.8 + pause)]

        self.assertEqual(_caption_texts(thank_you + next_up), expected)

    @parameterized.expand(
        [
            # "Welcome back, ladies and gentlemen." has 35 characters.
            param(
                "sentence_of_at_least_28_characters",
                texts=["Welcome", "back,", "ladies", "and", "gentlemen.", "Our", "first", "event"],
                expected=["Welcome back, ladies and gentlemen.", "Our first event"],
            ),
            # "Thank you, dancers." has 19 characters, and "Thank you, dancers. Please be seated!" has 37.
            param(
                "short_sentence_shares_a_caption_with_the_next_one",
                texts=["Thank", "you,", "dancers.", "Please", "be", "seated!", "Next"],
                expected=["Thank you, dancers. Please be seated!", "Next"],
            ),
            param(
                "closing_quote_after_the_sentence_end",
                texts=["The", "judges", "said", '"clear', "the", 'floor."', "Next"],
                expected=['The judges said "clear the floor."', "Next"],
            ),
        ]
    )
    def test_caption_ends_after_a_sentence_once_it_is_long_enough(self, _, texts, expected):
        self.assertEqual(_caption_texts(_words(texts)), expected)

    @parameterized.expand(
        [
            # Nine six-character words and eight spaces make 62 characters.
            param("comma_after_at_least_56_characters", last_text="abcde,", expected_first_word_count=9),
            param("semicolon_after_at_least_56_characters", last_text="abcde;", expected_first_word_count=9),
            param("colon_after_at_least_56_characters", last_text="abcde:", expected_first_word_count=9),
            # Without a comma the caption runs on until it is full: twelve words make 83 characters.
            param("no_punctuation", last_text="abcdef", expected_first_word_count=12),
        ]
    )
    def test_caption_ends_after_a_clause_once_it_is_long_enough(self, _, last_text, expected_first_word_count):
        texts = ["abcdef"] * 8 + [last_text] + ["abcdef"] * 5

        captions = ballroom_scribe.gui.utils.group_words_into_captions([_chunk(0, 60_000, *_words(texts))])

        self.assertEqual(len(captions[0].words), expected_first_word_count)

    def test_comma_does_not_end_a_short_caption(self):
        self.assertEqual(_caption_texts(_words(["Ladies,", "gentlemen,", "welcome"])), ["Ladies, gentlemen, welcome"])

    def test_caption_never_exceeds_84_characters(self):
        # Twelve six-character words and eleven spaces make 83 characters, and a thirteenth word would make 90.
        captions = ballroom_scribe.gui.utils.group_words_into_captions([_chunk(0, 60_000, *_words(["abcdef"] * 27))])

        self.assertEqual([len(caption.words) for caption in captions], [12, 12, 3])
        self.assertEqual([len(caption.get_text()) for caption in captions], [83, 83, 20])

    def test_caption_lasts_until_the_next_one_or_a_second_after_its_last_word(self):
        # The sentence ends when "gentlemen." has been spoken, at 1.4 s, and "Our" starts at 1.5 s.
        words = _words(["Welcome", "back,", "ladies", "and", "gentlemen.", "Our", "first", "event"])

        captions = ballroom_scribe.gui.utils.group_words_into_captions([_chunk(0, 60_000, *words)])

        self.assertEqual(
            captions,
            (
                # A second after its last word would be 2.4 s, but the next caption starts first.
                _chunk(0, 1500, *words[:5]),
                # The last word, "event", has been spoken at 2.3 s.
                _chunk(1500, 1800, *words[5:]),
            ),
        )

    def test_caption_is_showing_at_the_millisecond_to_which_its_first_word_rounds(self):
        # The editor seeks to whole milliseconds, so an event that starts at 12.3454 s is shown at 12.345 s.
        word = _word("Welcome", 12.3454)

        (caption,) = ballroom_scribe.gui.utils.group_words_into_captions(
            [_chunk(0, 1_920_000, word, sample_rate=32_000)]
        )

        self.assertEqual(caption.start_samples, 12_345 * 32)
        self.assertLessEqual(caption.start_seconds, round(word.start * 1000) / 1000)
        self.assertIs(ballroom_scribe.gui.utils.find_caption((caption,), round(word.start * 1000) / 1000), caption)

    def test_captions_span_transcript_chunks_and_keep_their_sample_rate_and_channel(self):
        thank, you = _word("Thank", 0.0), _word("you", 0.3)

        captions = ballroom_scribe.gui.utils.group_words_into_captions([_chunk(0, 250, thank), _chunk(250, 250, you)])

        self.assertEqual(captions, (_chunk(0, 1500, thank, you),))

    @parameterized.expand(
        [
            param("no_chunks", transcript=[]),
            param("chunks_without_words", transcript=[_chunk(0, 1000), _chunk(2000, 1000)]),
        ]
    )
    def test_no_words_give_no_captions(self, _, transcript):
        self.assertEqual(ballroom_scribe.gui.utils.group_words_into_captions(transcript), ())


if __name__ == "__main__":
    unittest.main()
