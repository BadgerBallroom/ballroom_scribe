import unittest

from parameterized import param, parameterized

import ballroom_scribe
import ballroom_scribe.models
import tests.helpers.transcript_chunk

_SPLIT_ON_SILENCE_THRESHOLD = 10.0  # seconds


class TestTranscriptChunker(unittest.TestCase):
    """Unit tests for the `TranscriptChunker` class."""

    @parameterized.expand(
        [
            param(
                "empty_words",
                words=[],
                expected_chunks=[],
            ),
            param(
                "single_window",
                words=[
                    ballroom_scribe.models.Word(word="Next", start=5.0, end=5.5),
                    ballroom_scribe.models.Word(word="up", start=5.5, end=6.0),
                    ballroom_scribe.models.Word(word="waltz", start=6.0, end=6.5),
                ],
                expected_chunks=[
                    tests.helpers.transcript_chunk.Maker(start_time=5.0).make(["Next", "up", "waltz"]),
                ],
            ),
            param(
                "multiple_windows",
                words=[
                    ballroom_scribe.models.Word(word="Next", start=5.0, end=5.5),
                    ballroom_scribe.models.Word(word="up", start=5.5, end=6.0),
                    ballroom_scribe.models.Word(word="waltz", start=6.0, end=6.5),
                    ballroom_scribe.models.Word(word="Thank", start=20.0, end=20.5),
                    ballroom_scribe.models.Word(word="you,", start=20.5, end=21.0),
                    ballroom_scribe.models.Word(word="dancers!", start=21.0, end=21.5),
                ],
                expected_chunks=[
                    tests.helpers.transcript_chunk.Maker(start_time=5.0).make(["Next", "up", "waltz"]),
                    tests.helpers.transcript_chunk.Maker(start_time=20.0).make(["Thank", "you,", "dancers!"]),
                ],
            ),
        ]
    )
    def test_chunk(self, _, words, expected_chunks):
        chunker = ballroom_scribe.TranscriptChunker(split_on_silence_threshold=_SPLIT_ON_SILENCE_THRESHOLD)
        chunks = list(chunker.chunk(words))

        self.assertEqual(chunks, expected_chunks)


if __name__ == "__main__":
    unittest.main()
