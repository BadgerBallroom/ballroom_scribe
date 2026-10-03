import unittest
from unittest.mock import MagicMock

import numpy as np
from parameterized import param, parameterized

import ballroom_scribe.models


def _make_audio_chunk(num_samples: int) -> ballroom_scribe.models.AudioChunk:
    return ballroom_scribe.models.AudioChunk(
        data=np.zeros(num_samples),
        sample_rate=10,
        classification=MagicMock(spec=ballroom_scribe.models.AudioChunk.Classification),
        start_samples=0,
    )


class TestMusicValley(unittest.TestCase):
    @parameterized.expand(
        [
            param("empty_tuple", speech_chunks=(), expected=False),
            param("only_empty_audio_chunks", speech_chunks=(_make_audio_chunk(0),), expected=False),
            param(
                "at_least_one_non_empty_audio_chunk",
                speech_chunks=(_make_audio_chunk(0), _make_audio_chunk(5)),
                expected=True,
            ),
        ]
    )
    def test_bool(self, _, speech_chunks, expected):
        valley = ballroom_scribe.models.MusicValley(
            start_samples=100,
            sample_rate=10,
            duration_samples=10,
            speech_chunks=speech_chunks,
        )
        self.assertEqual(bool(valley), expected)


if __name__ == "__main__":
    unittest.main()
