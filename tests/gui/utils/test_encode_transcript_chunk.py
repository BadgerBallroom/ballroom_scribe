import json
import unittest

import ballroom_scribe.gui.utils
import ballroom_scribe.models


class TestEncodeTranscriptChunk(unittest.TestCase):
    """Unit tests for `encode_transcript_chunk`."""

    def setUp(self):
        self.transcript_chunk = ballroom_scribe.models.TranscriptChunk(
            start_samples=320_000,
            sample_rate=32_000,
            duration_samples=160_000,
            channel=1,
            words=(
                ballroom_scribe.models.Word(word="Jos\u00e9\u2019s,", start=10.0, end=10.25, probability=0.875),
                ballroom_scribe.models.Word(word="two\nlines", start=10.5, end=11.0),
            ),
        )

    def test_encodes_every_field_as_json(self):
        line = ballroom_scribe.gui.utils.encode_transcript_chunk(self.transcript_chunk)

        self.assertEqual(
            json.loads(line),
            {
                "start_samples": 320_000,
                "sample_rate": 32_000,
                "duration_samples": 160_000,
                "channel": 1,
                "words": [
                    {"word": "Jos\u00e9\u2019s,", "start": 10.0, "end": 10.25, "probability": 0.875},
                    {"word": "two\nlines", "start": 10.5, "end": 11.0, "probability": 1.0},
                ],
            },
        )

    def test_encoding_is_one_line_of_ascii(self):
        line = ballroom_scribe.gui.utils.encode_transcript_chunk(self.transcript_chunk)

        self.assertTrue(line.isascii())
        self.assertEqual(line.splitlines(), [line])


if __name__ == "__main__":
    unittest.main()
