import unittest

from parameterized import param, parameterized

import ballroom_scribe.gui.utils
import ballroom_scribe.models

_SAMPLE_RATE = 1000
_CHANNEL = 1
_SPACE_WIDTH = 5.0
_LABEL_PADDING = 6.0


def _word(text: str, start: float, end: float | None = None) -> ballroom_scribe.models.Word:
    """Helper to create a word that lasts 0.1 seconds unless `end` is given."""
    return ballroom_scribe.models.Word(word=text, start=start, end=start + 0.1 if end is None else end)


def _chunk(
    start_samples: int,
    duration_samples: int,
    *words: ballroom_scribe.models.Word,
) -> ballroom_scribe.models.TranscriptChunk:
    return ballroom_scribe.models.TranscriptChunk(
        start_samples=start_samples,
        sample_rate=_SAMPLE_RATE,
        duration_samples=duration_samples,
        channel=_CHANNEL,
        words=words,
    )


def _merge(
    transcript: list[ballroom_scribe.models.TranscriptChunk],
    word_widths: list[float],
    pixels_per_second: float,
) -> tuple[ballroom_scribe.models.TranscriptChunk, ...]:
    return ballroom_scribe.gui.utils.merge_speech_markers(
        transcript,
        word_widths,
        pixels_per_second=pixels_per_second,
        space_width=_SPACE_WIDTH,
        label_padding=_LABEL_PADDING,
    )


class TestMergeSpeechMarkers(unittest.TestCase):
    """Unit tests for `merge_speech_markers`."""

    def test_words_whose_labels_fit_stay_separate(self):
        one, two, three = _word("one", 0.0), _word("two", 1.0), _word("three", 2.0)

        # At 100 pixels per second, a 30-pixel label ends long before the next word starts 100 pixels later.
        markers = _merge([_chunk(0, 2500, one, two, three)], [30.0, 30.0, 30.0], pixels_per_second=100.0)

        self.assertEqual(
            markers,
            (
                # Each marker lasts until the next one starts.
                _chunk(0, 1000, one),
                _chunk(1000, 1000, two),
                # The last marker lasts as long as its label and the padding need: 36 pixels.
                _chunk(2000, 360, three),
            ),
        )

    def test_overlapping_words_chain_into_one_marker_at_the_first_words_start(self):
        one, two, three, four = _word("one", 0.0), _word("two", 0.1), _word("three", 0.2), _word("four", 5.0)

        # "two" starts 10 pixels after "one", inside its 30-pixel label, and "three" starts inside the label
        # "one two" (65 pixels). "four" starts beyond the label "one two three" (100 pixels).
        markers = _merge([_chunk(0, 5500, one, two, three, four)], [30.0, 30.0, 30.0, 30.0], pixels_per_second=100.0)

        self.assertEqual(markers, (_chunk(0, 5000, one, two, three), _chunk(5000, 360, four)))
        self.assertEqual(markers[0].get_text(), "one two three")

    def test_sentence_start_begins_a_marker_and_cuts_off_the_label_before_it(self):
        welcome, following = _word("Welcome.", 0.0), _word("Next", 0.6)

        # "Next" starts 60 pixels after "Welcome.", inside its 80-pixel label, but it starts a sentence.
        markers = _merge([_chunk(0, 1000, welcome, following)], [80.0, 30.0], pixels_per_second=100.0)

        self.assertEqual(markers, (_chunk(0, 600, welcome), _chunk(600, 360, following)))
        # The first label has less room than it needs, so the widget cuts it off.
        self.assertLess(markers[0].duration_seconds * 100.0 - _LABEL_PADDING, 80.0)

    @parameterized.expand(
        [
            # The label before the sentence would keep 60 - 6 = 54 pixels.
            param("leaving_at_least_48_pixels", next_start=0.6, expected_marker_count=2),
            # The label before the sentence would keep 53 - 6 = 47 pixels.
            param("leaving_less_than_48_pixels", next_start=0.53, expected_marker_count=1),
        ]
    )
    def test_sentence_start_needs_room_for_the_label_before_it(self, _, next_start, expected_marker_count):
        transcript = [_chunk(0, 1000, _word("Welcome.", 0.0), _word("Next", next_start))]

        markers = _merge(transcript, [80.0, 30.0], pixels_per_second=100.0)

        self.assertEqual(len(markers), expected_marker_count)

    @parameterized.expand(
        [
            param("pause_of_two_seconds", first_end=0.5, next_start=2.5, expected_marker_count=2),
            param("longer_pause", first_end=0.3, next_start=6.0, expected_marker_count=2),
            param("shorter_pause", first_end=0.5, next_start=2.4, expected_marker_count=1),
            # The pause is measured from the end of the word before it, not from its start.
            param("long_word_before_a_short_pause", first_end=2.0, next_start=2.5, expected_marker_count=1),
        ]
    )
    def test_long_pause_starts_a_sentence(self, _, first_end, next_start, expected_marker_count):
        # Neither word ends a sentence. At 40 pixels per second, the second word starts inside the first label.
        transcript = [_chunk(0, 7000, _word("one", 0.0, end=first_end), _word("two", next_start))]

        markers = _merge(transcript, [300.0, 30.0], pixels_per_second=40.0)

        self.assertEqual(len(markers), expected_marker_count)

    def test_markers_merge_across_transcript_chunks(self):
        one, two = _word("one", 0.0), _word("two", 0.1)

        markers = _merge([_chunk(0, 100, one), _chunk(100, 100, two)], [30.0, 30.0], pixels_per_second=100.0)

        # One marker holds the words of both chunks, with the transcript's sample rate and channel.
        self.assertEqual(markers, (_chunk(0, 710, one, two),))

    def test_zooming_out_merges_more_words(self):
        # Two pairs of words: the words of a pair are half a second apart, and the pairs are three seconds apart.
        words = [_word(text, start) for text, start in (("one", 0.0), ("two", 0.5), ("three", 3.0), ("four", 3.5))]
        transcript = [_chunk(0, 4000, *words)]
        widths = [30.0, 30.0, 30.0, 30.0]

        marker_counts = [len(_merge(transcript, widths, pixels_per_second=zoom)) for zoom in (100.0, 40.0, 10.0)]

        # Every word alone, then one marker per pair, then one marker for everything.
        self.assertEqual(marker_counts, [4, 2, 1])

    @parameterized.expand(
        [
            param("no_chunks", transcript=[], word_widths=[], pixels_per_second=100.0),
            param("chunk_without_words", transcript=[_chunk(0, 1000)], word_widths=[], pixels_per_second=100.0),
            param(
                "no_pixels_per_second",
                transcript=[_chunk(0, 1000, _word("one", 0.0))],
                word_widths=[30.0],
                pixels_per_second=0.0,
            ),
        ]
    )
    def test_nothing_to_show(self, _, transcript, word_widths, pixels_per_second):
        self.assertEqual(_merge(transcript, word_widths, pixels_per_second=pixels_per_second), ())

    def test_rejects_a_width_count_that_does_not_match_the_words(self):
        transcript = [_chunk(0, 1000, _word("one", 0.0), _word("two", 0.5))]

        with self.assertRaisesRegex(ValueError, "one width per word"):
            _merge(transcript, [30.0], pixels_per_second=100.0)


if __name__ == "__main__":
    unittest.main()
