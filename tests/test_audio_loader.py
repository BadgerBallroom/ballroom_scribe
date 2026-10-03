import pathlib
import unittest
from collections.abc import Iterable
from unittest.mock import ANY, MagicMock, call, patch

import numpy as np
import soundfile
from parameterized import param, parameterized

import ballroom_scribe
import ballroom_scribe.models

_SAMPLE_RATE = 10
_DTYPE = "int32"
_MINUTES_PER_FIVE_SAMPLES = 1.0 / 120.0
_SECONDS_PER_SAMPLE = 0.1


class FakeReadSideEffect:
    """A fake `soundfile.SoundFile.read` function. Performs in-place mutations like the real function."""

    def __init__(self, chunks: Iterable[soundfile.AudioData]):
        """Initializes the instance.

        Args:
            chunks: An iterable of NumPy arrays to return in order.
        """
        self._iter_chunks = iter(chunks)

    def __call__(self, frames=None, dtype=None, out=None):
        del frames, dtype
        chunk = next(self._iter_chunks)
        if out is not None:
            # Simulate in-place mutation by assigning directly into the buffer.
            out[: chunk.shape[0]] = chunk
            # Return the view, just like the real `soundfile.SoundFile.read`.
            return out[: chunk.shape[0]]
        return chunk


class TestAudioLoader(unittest.TestCase):
    """Tests for the `AudioLoader` class."""

    def setUp(self):
        self.input_audio = MagicMock(spec=pathlib.Path)

        self.mock_soundfile_cls = self.enterContext(patch.object(soundfile, "SoundFile", autospec=True))
        self.mock_soundfile = self.mock_soundfile_cls.return_value
        self.mock_soundfile.__enter__.return_value = self.mock_soundfile

        self.enterContext(patch.object(ballroom_scribe.audio_loader.AudioLoader, "SAMPLE_RATE", new=_SAMPLE_RATE))
        self.mock_soundfile.samplerate = _SAMPLE_RATE
        self.mock_soundfile.channels = 1

        self.enterContext(patch.object(ballroom_scribe.audio_loader.AudioLoader, "DTYPE", new=_DTYPE))

    def test_init_non_positive_chunk_length(self):
        with self.assertRaisesRegex(ValueError, "chunk_minutes must be positive, but got 0.0."):
            ballroom_scribe.audio_loader.AudioLoader(chunk_minutes=0.0, chunk_overlap_seconds=1.0)

    def test_init_negative_overlap(self):
        with self.assertRaisesRegex(ValueError, "chunk_overlap_seconds must be non-negative, but got -1.0."):
            ballroom_scribe.audio_loader.AudioLoader(chunk_minutes=1.0, chunk_overlap_seconds=-1.0)

    @parameterized.expand(
        [
            param(
                "incorrect_sample_rate",
                frames=100,
                samplerate=20,
                channels=1,
                expected_regex="Audio file .* has sample rate 20, which is not 10.",
            ),
            param(
                "zero_duration",
                frames=0,
                samplerate=_SAMPLE_RATE,
                channels=1,
                expected_regex="Audio file .* has no audio content.",
            ),
            param(
                "too_long",
                frames=2**53,
                samplerate=_SAMPLE_RATE,
                channels=1,
                expected_regex="Audio file cannot be longer than .*",
            ),
            param(
                "too_many_channels",
                frames=100,
                samplerate=_SAMPLE_RATE,
                channels=2,
                expected_regex="Audio file .* has 2 channels, but only mono is supported.",
            ),
        ]
    )
    def test_load_file_errors(self, _, frames, samplerate, channels, expected_regex):
        self.mock_soundfile.frames = frames
        self.mock_soundfile.samplerate = samplerate
        self.mock_soundfile.channels = channels
        audio_loader = ballroom_scribe.audio_loader.AudioLoader(chunk_minutes=1.0, chunk_overlap_seconds=1.0)

        with self.assertRaisesRegex(ValueError, expected_regex):
            list(audio_loader.load(self.input_audio))

        self.mock_soundfile.read.assert_not_called()

    @parameterized.expand(
        [
            param(
                "no_overlap",
                frames=18,
                overlap_seconds=0,
                fake_reads=[
                    np.arange(0, 5, dtype=_DTYPE),
                    np.arange(5, 10, dtype=_DTYPE),
                    np.arange(10, 15, dtype=_DTYPE),
                    np.arange(15, 18, dtype=_DTYPE),
                ],
                expected_read_calls=[call(frames=5, dtype=_DTYPE)] + [call(frames=5, dtype=_DTYPE, out=ANY)] * 3,
                expected_chunks=[
                    ballroom_scribe.models.AudioChunk(
                        start_samples=0,
                        sample_rate=_SAMPLE_RATE,
                        data=np.arange(0, 5, dtype=_DTYPE),
                        classification=ballroom_scribe.models.AudioChunk.Classification.UNKNOWN,
                    ),
                    ballroom_scribe.models.AudioChunk(
                        start_samples=5,
                        sample_rate=_SAMPLE_RATE,
                        data=np.arange(5, 10, dtype=_DTYPE),
                        classification=ballroom_scribe.models.AudioChunk.Classification.UNKNOWN,
                    ),
                    ballroom_scribe.models.AudioChunk(
                        start_samples=10,
                        sample_rate=_SAMPLE_RATE,
                        data=np.arange(10, 15, dtype=_DTYPE),
                        classification=ballroom_scribe.models.AudioChunk.Classification.UNKNOWN,
                    ),
                    ballroom_scribe.models.AudioChunk(
                        start_samples=15,
                        sample_rate=_SAMPLE_RATE,
                        data=np.arange(15, 18, dtype=_DTYPE),
                        classification=ballroom_scribe.models.AudioChunk.Classification.UNKNOWN,
                    ),
                ],
            ),
            param(
                "with_overlap",
                frames=18,
                overlap_seconds=_SECONDS_PER_SAMPLE,
                fake_reads=[
                    np.arange(0, 6, dtype=_DTYPE),
                    np.arange(6, 11, dtype=_DTYPE),
                    np.arange(11, 16, dtype=_DTYPE),
                    np.arange(16, 18, dtype=_DTYPE),
                ],
                expected_read_calls=[call(frames=6, dtype=_DTYPE)] + [call(frames=5, dtype=_DTYPE, out=ANY)] * 3,
                expected_chunks=[
                    ballroom_scribe.models.AudioChunk(
                        start_samples=0,
                        sample_rate=_SAMPLE_RATE,
                        data=np.arange(0, 6, dtype=_DTYPE),
                        classification=ballroom_scribe.models.AudioChunk.Classification.UNKNOWN,
                    ),
                    ballroom_scribe.models.AudioChunk(
                        start_samples=5,
                        sample_rate=_SAMPLE_RATE,
                        data=np.arange(5, 11, dtype=_DTYPE),
                        classification=ballroom_scribe.models.AudioChunk.Classification.UNKNOWN,
                    ),
                    ballroom_scribe.models.AudioChunk(
                        start_samples=10,
                        sample_rate=_SAMPLE_RATE,
                        data=np.arange(10, 16, dtype=_DTYPE),
                        classification=ballroom_scribe.models.AudioChunk.Classification.UNKNOWN,
                    ),
                    ballroom_scribe.models.AudioChunk(
                        start_samples=15,
                        sample_rate=_SAMPLE_RATE,
                        data=np.arange(15, 18, dtype=_DTYPE),
                        classification=ballroom_scribe.models.AudioChunk.Classification.UNKNOWN,
                    ),
                ],
            ),
            param(
                "shorter_than_first_chunk",
                frames=4,
                overlap_seconds=_SECONDS_PER_SAMPLE,
                fake_reads=[np.arange(0, 4, dtype=_DTYPE)],
                expected_read_calls=[call(frames=6, dtype=_DTYPE)],
                expected_chunks=[
                    ballroom_scribe.models.AudioChunk(
                        start_samples=0,
                        sample_rate=_SAMPLE_RATE,
                        data=np.arange(0, 4, dtype=_DTYPE),
                        classification=ballroom_scribe.models.AudioChunk.Classification.UNKNOWN,
                    ),
                ],
            ),
            param(
                "ends_exactly_on_boundary",
                frames=11,
                overlap_seconds=_SECONDS_PER_SAMPLE,
                fake_reads=[
                    np.arange(0, 6, dtype=_DTYPE),
                    np.arange(6, 11, dtype=_DTYPE),
                ],
                expected_read_calls=[call(frames=6, dtype=_DTYPE), call(frames=5, dtype=_DTYPE, out=ANY)],
                expected_chunks=[
                    ballroom_scribe.models.AudioChunk(
                        start_samples=0,
                        sample_rate=_SAMPLE_RATE,
                        data=np.arange(0, 6, dtype=_DTYPE),
                        classification=ballroom_scribe.models.AudioChunk.Classification.UNKNOWN,
                    ),
                    ballroom_scribe.models.AudioChunk(
                        start_samples=5,
                        sample_rate=_SAMPLE_RATE,
                        data=np.arange(5, 11, dtype=_DTYPE),
                        classification=ballroom_scribe.models.AudioChunk.Classification.UNKNOWN,
                    ),
                ],
            ),
        ]
    )
    def test_load_chunks(self, _, frames, overlap_seconds, fake_reads, expected_read_calls, expected_chunks):
        self.mock_soundfile.frames = frames
        self.mock_soundfile.read.side_effect = FakeReadSideEffect(fake_reads)

        audio_loader = ballroom_scribe.audio_loader.AudioLoader(
            chunk_minutes=_MINUTES_PER_FIVE_SAMPLES,
            chunk_overlap_seconds=overlap_seconds,
        )

        chunks = list(audio_loader.load(self.input_audio))

        self.mock_soundfile.read.assert_has_calls(expected_read_calls)
        self.assertEqual(chunks, expected_chunks)


if __name__ == "__main__":
    unittest.main()
