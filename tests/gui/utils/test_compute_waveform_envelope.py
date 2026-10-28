import pathlib
import unittest
from unittest.mock import patch

import numpy as np
import soundfile
from parameterized import param, parameterized

import ballroom_scribe.gui.models
import ballroom_scribe.gui.utils

_FAKE_AUDIO = pathlib.Path("fake_cache/Audio-Channel-0.wav")


class TestComputeWaveformEnvelope(unittest.TestCase):
    """Unit tests for `compute_waveform_envelope`."""

    def setUp(self):
        self.mock_soundfile_cls = self.enterContext(patch.object(soundfile, "SoundFile", autospec=True))
        self.mock_soundfile = self.mock_soundfile_cls.return_value
        self.mock_soundfile.__enter__.return_value = self.mock_soundfile
        self.mock_soundfile.channels = 1
        self.mock_soundfile.samplerate = 4
        self.mock_soundfile.frames = 9

    def test_computes_min_max_across_block_boundaries_and_partial_last_bin(self):
        # 4 Hz at 2 bins/s -> 2 samples per bin.
        # Samples: [0.1, -0.3] | [0.5, -0.2] | [-0.7, 0.4] | [0.0, 0.8] | [-0.1]
        self.mock_soundfile.blocks.return_value = iter(
            [
                np.array([0.1, -0.3, 0.5], dtype=np.float32),
                np.array([-0.2, -0.7, 0.4, 0.0], dtype=np.float32),
                np.array([0.8, -0.1], dtype=np.float32),
            ]
        )

        envelope = ballroom_scribe.gui.utils.compute_waveform_envelope(_FAKE_AUDIO, bins_per_second=2)

        self.assertEqual(
            envelope,
            ballroom_scribe.gui.models.WaveformEnvelope(
                mins=np.array([-0.3, -0.2, -0.7, 0.0, -0.1], dtype=np.float32),
                maxs=np.array([0.1, 0.5, 0.4, 0.8, -0.1], dtype=np.float32),
                bins_per_second=2.0,
            ),
        )

    @parameterized.expand(
        [
            param("non_positive_bins", channels=1, frames=10, samplerate=100, bins_per_second=0),
            param("stereo_audio", channels=2, frames=10, samplerate=100, bins_per_second=10),
            param("empty_audio", channels=1, frames=0, samplerate=100, bins_per_second=10),
            param("samplerate_below_bins", channels=1, frames=10, samplerate=5, bins_per_second=20),
        ]
    )
    def test_invalid_inputs_raise_value_error(
        self,
        _,
        channels: int,
        frames: int,
        samplerate: int,
        bins_per_second: int,
    ):
        self.mock_soundfile.channels = channels
        self.mock_soundfile.frames = frames
        self.mock_soundfile.samplerate = samplerate

        with self.assertRaises(ValueError):
            ballroom_scribe.gui.utils.compute_waveform_envelope(_FAKE_AUDIO, bins_per_second=bins_per_second)


if __name__ == "__main__":
    unittest.main()
