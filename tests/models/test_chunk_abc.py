import unittest
from typing import override

import ballroom_scribe.models


class _ConcreteChunk(ballroom_scribe.models.Chunk):
    @override
    def __bool__(self) -> bool:
        return True


class TestChunk(unittest.TestCase):
    def setUp(self):
        # 10 samples at a sample rate of 10 Hz are a duration of 1 second.
        self.chunk = _ConcreteChunk(
            start_samples=100,
            sample_rate=10,
            duration_samples=10,
        )

    def test_start_seconds(self):
        self.assertAlmostEqual(self.chunk.start_seconds, 10.0)

    def test_duration_seconds(self):
        self.assertAlmostEqual(self.chunk.duration_seconds, 1.0)

    def test_end_samples(self):
        self.assertEqual(self.chunk.end_samples, 110)

    def test_end_seconds(self):
        self.assertAlmostEqual(self.chunk.end_seconds, 11.0)

    def test_midpoint_seconds(self):
        self.assertAlmostEqual(self.chunk.midpoint_seconds, 10.5)

    def test_get_filename_stem(self):
        self.assertEqual(self.chunk.get_filename_stem(), "0000000100-0000000110")

    def test_format_time_range(self):
        self.assertEqual(self.chunk.format_time_range(), "0:00:10.00 - 0:00:11.00\t(0:00:01.00)")


if __name__ == "__main__":
    unittest.main()
