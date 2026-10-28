import unittest

import numpy as np
from parameterized import param, parameterized

import ballroom_scribe.gui.models


class TestInitAndProperties(unittest.TestCase):
    """Tests for `WaveformEnvelope` initialization, immutability, equality, and properties."""

    def test_marks_arrays_read_only_and_computes_duration(self):
        mins = np.array([-0.5, -0.2, -0.8, -0.1], dtype=np.float32)
        maxs = np.array([0.5, 0.3, 0.9, 0.2], dtype=np.float32)
        envelope = ballroom_scribe.gui.models.WaveformEnvelope(mins=mins, maxs=maxs, bins_per_second=2.0)

        self.assertFalse(envelope.mins.flags.writeable)
        self.assertFalse(envelope.maxs.flags.writeable)
        self.assertEqual(envelope.num_bins, 4)
        self.assertAlmostEqual(envelope.duration_seconds, 2.0)

    def test_equality_compares_arrays_and_rate(self):
        a = ballroom_scribe.gui.models.WaveformEnvelope(
            mins=np.array([-1.0, -0.5], dtype=np.float32),
            maxs=np.array([1.0, 0.5], dtype=np.float32),
            bins_per_second=10.0,
        )
        b = ballroom_scribe.gui.models.WaveformEnvelope(
            mins=np.array([-1.0, -0.5], dtype=np.float32),
            maxs=np.array([1.0, 0.5], dtype=np.float32),
            bins_per_second=10.0,
        )
        c = ballroom_scribe.gui.models.WaveformEnvelope(
            mins=np.array([-1.0, -0.2], dtype=np.float32),
            maxs=np.array([1.0, 0.5], dtype=np.float32),
            bins_per_second=10.0,
        )

        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, "not an envelope")

    @parameterized.expand(
        [
            param(
                "2d_mins",
                mins=np.zeros((2, 2), dtype=np.float32),
                maxs=np.zeros(4, dtype=np.float32),
                bins_per_second=10.0,
            ),
            param(
                "mismatched_lengths",
                mins=np.zeros(3, dtype=np.float32),
                maxs=np.zeros(4, dtype=np.float32),
                bins_per_second=10.0,
            ),
            param(
                "zero_rate",
                mins=np.zeros(4, dtype=np.float32),
                maxs=np.zeros(4, dtype=np.float32),
                bins_per_second=0.0,
            ),
            param(
                "negative_rate",
                mins=np.zeros(4, dtype=np.float32),
                maxs=np.zeros(4, dtype=np.float32),
                bins_per_second=-5.0,
            ),
            param(
                "nan_rate",
                mins=np.zeros(4, dtype=np.float32),
                maxs=np.zeros(4, dtype=np.float32),
                bins_per_second=float("nan"),
            ),
        ]
    )
    def test_invalid_init_raises_value_error(self, _, mins, maxs, bins_per_second):
        with self.assertRaises(ValueError):
            ballroom_scribe.gui.models.WaveformEnvelope(
                mins=mins,
                maxs=maxs,
                bins_per_second=bins_per_second,
            )


class TestResample(unittest.TestCase):
    """Tests for `WaveformEnvelope.resample`."""

    def setUp(self):
        # 4 seconds at 1 bin/s: bins [0, 1, 2, 3]
        self.envelope = ballroom_scribe.gui.models.WaveformEnvelope(
            mins=np.array([-0.1, -0.4, -0.2, -0.8], dtype=np.float32),
            maxs=np.array([0.2, 0.5, 0.3, 0.9], dtype=np.float32),
            bins_per_second=1.0,
        )

    def test_downsamples_full_window(self):
        mins, maxs = self.envelope.resample(start_seconds=0.0, end_seconds=4.0, count=2)
        np.testing.assert_allclose(mins, np.array([-0.4, -0.8], dtype=np.float32))
        np.testing.assert_allclose(maxs, np.array([0.5, 0.9], dtype=np.float32))

    def test_upsamples_when_count_exceeds_bins(self):
        mins, maxs = self.envelope.resample(start_seconds=0.0, end_seconds=2.0, count=4)
        np.testing.assert_allclose(mins, np.array([-0.1, -0.1, -0.4, -0.4], dtype=np.float32))
        np.testing.assert_allclose(maxs, np.array([0.2, 0.2, 0.5, 0.5], dtype=np.float32))

    def test_marks_columns_outside_audio_bounds_as_nan(self):
        mins, maxs = self.envelope.resample(start_seconds=-2.0, end_seconds=6.0, count=4)
        self.assertTrue(np.isnan(mins[0]))
        self.assertTrue(np.isnan(maxs[0]))
        np.testing.assert_allclose(mins[1:3], np.array([-0.4, -0.8], dtype=np.float32))
        np.testing.assert_allclose(maxs[1:3], np.array([0.5, 0.9], dtype=np.float32))
        self.assertTrue(np.isnan(mins[3]))
        self.assertTrue(np.isnan(maxs[3]))

    def test_completely_outside_window_returns_all_nan(self):
        mins, maxs = self.envelope.resample(start_seconds=10.0, end_seconds=12.0, count=3)
        self.assertTrue(np.all(np.isnan(mins)))
        self.assertTrue(np.all(np.isnan(maxs)))

    def test_zero_count_returns_empty_arrays(self):
        mins, maxs = self.envelope.resample(start_seconds=0.0, end_seconds=4.0, count=0)
        self.assertEqual(mins.shape, (0,))
        self.assertEqual(maxs.shape, (0,))

    @parameterized.expand(
        [
            param("negative_count", start=0.0, end=1.0, count=-1),
            param("equal_bounds", start=1.0, end=1.0, count=4),
            param("reversed_bounds", start=2.0, end=1.0, count=4),
        ]
    )
    def test_invalid_arguments_raise_value_error(self, _, start: float, end: float, count: int):
        with self.assertRaises(ValueError):
            self.envelope.resample(start_seconds=start, end_seconds=end, count=count)


if __name__ == "__main__":
    unittest.main()
