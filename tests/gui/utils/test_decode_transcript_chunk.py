import json
import unittest

from parameterized import param, parameterized

import ballroom_scribe.gui.utils
import ballroom_scribe.models

_WORD = {"word": "waltz", "start": 10.0, "end": 10.5, "probability": 0.5}
_CHUNK = {"start_samples": 320_000, "sample_rate": 32_000, "duration_samples": 160_000, "channel": 1, "words": [_WORD]}


def _without(fields: dict, key: str) -> dict:
    return {name: value for name, value in fields.items() if name != key}


class TestDecodeTranscriptChunk(unittest.TestCase):
    """Unit tests for `decode_transcript_chunk`."""

    @parameterized.expand(
        [
            param(
                "words_with_non_ascii_characters",
                words=(
                    ballroom_scribe.models.Word(word="Jos\u00e9\u2019s,", start=10.0, end=10.25, probability=0.875),
                    ballroom_scribe.models.Word(word="\u00a1Ol\u00e9!", start=10.5, end=11.0),
                ),
            ),
            param("no_words", words=()),
        ]
    )
    def test_decodes_what_encode_transcript_chunk_encoded(self, _, words):
        transcript_chunk = ballroom_scribe.models.TranscriptChunk(
            start_samples=320_000,
            sample_rate=32_000,
            duration_samples=160_000,
            channel=1,
            words=words,
        )
        line = ballroom_scribe.gui.utils.encode_transcript_chunk(transcript_chunk)

        self.assertEqual(ballroom_scribe.gui.utils.decode_transcript_chunk(line), transcript_chunk)
        # The worker ends every line with a newline.
        self.assertEqual(ballroom_scribe.gui.utils.decode_transcript_chunk(line + "\n"), transcript_chunk)

    def test_decodes_whole_numbers_as_word_times(self):
        line = json.dumps({**_CHUNK, "words": [{"word": "waltz", "start": 10, "end": 11, "probability": 1}]})

        transcript_chunk = ballroom_scribe.gui.utils.decode_transcript_chunk(line)

        self.assertEqual(
            transcript_chunk.words,
            (ballroom_scribe.models.Word(word="waltz", start=10.0, end=11.0, probability=1.0),),
        )
        self.assertIsInstance(transcript_chunk.words[0].start, float)

    @parameterized.expand(
        [
            param("empty_line", line=""),
            param("not_json", line="Loaded Faster Whisper model turbo on cpu using compute type int8."),
            param("cut_off_json", line=json.dumps(_CHUNK)[:40]),
            param("json_list", line="[1, 2]"),
            param("json_number", line="42"),
            param("json_null", line="null"),
            param("other_json_object", line='{"status": "ok"}'),
            param("missing_channel", line=json.dumps(_without(_CHUNK, "channel"))),
            param("missing_words", line=json.dumps(_without(_CHUNK, "words"))),
            param("word_without_end", line=json.dumps({**_CHUNK, "words": [_without(_WORD, "end")]})),
            param("start_samples_as_text", line=json.dumps({**_CHUNK, "start_samples": "320000"})),
            param("fractional_start_samples", line=json.dumps({**_CHUNK, "start_samples": 0.5})),
            param("channel_as_boolean", line=json.dumps({**_CHUNK, "channel": True})),
            param("words_as_object", line=json.dumps({**_CHUNK, "words": {}})),
            param("word_as_text", line=json.dumps({**_CHUNK, "words": ["waltz"]})),
            param("word_text_as_number", line=json.dumps({**_CHUNK, "words": [{**_WORD, "word": 5}]})),
            param("word_start_as_text", line=json.dumps({**_CHUNK, "words": [{**_WORD, "start": "10.0"}]})),
        ]
    )
    def test_rejects_anything_else(self, _, line):
        with self.assertRaisesRegex(ValueError, "Not an encoded transcript chunk"):
            ballroom_scribe.gui.utils.decode_transcript_chunk(line)


if __name__ == "__main__":
    unittest.main()
