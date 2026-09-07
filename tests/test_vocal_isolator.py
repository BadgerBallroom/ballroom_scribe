import pathlib
import unittest
from unittest.mock import MagicMock, patch

import demucs.api
import demucs.audio

import ballroom_scribe
import ballroom_scribe.utils

_INPUT_AUDIO = pathlib.Path("cache/audio.wav")
_SAMPLE_RATE = 44100


class TestVocalIsolator(unittest.TestCase):
    """Unit tests for the `VocalIsolator` class."""

    def setUp(self):
        self.output_vocals = MagicMock(spec=pathlib.Path)

        self.mock_demucs_list_models = self.enterContext(patch.object(demucs.api, "list_models", autospec=True))
        self.mock_demucs_list_models.return_value = {"bag": ["htdemucs"], "single": []}

        self.mock_demucs_separator_cls = self.enterContext(patch.object(demucs.api, "Separator"))
        self.mock_demucs_separator = self.mock_demucs_separator_cls.return_value

        self.mock_validate_torch_device = self.enterContext(
            patch.object(ballroom_scribe.utils, "validate_torch_device", autospec=True)
        )

        self.mock_save_audio = self.enterContext(patch.object(demucs.audio, "save_audio"))

        self.mock_atomic_path = self.enterContext(patch.object(ballroom_scribe.utils, "atomic_path", autospec=True))
        self.partial_vocals = MagicMock(spec=pathlib.Path)
        self.mock_atomic_path.return_value.__enter__.return_value = self.partial_vocals

        self.mock_vocals = MagicMock()
        self.mock_demucs_separator.separate_audio_file.return_value = (
            MagicMock(),
            {"vocals": self.mock_vocals},
        )
        self.mock_demucs_separator.samplerate = _SAMPLE_RATE

    def test_init_validates_demucs_name(self):
        with self.assertRaisesRegex(ValueError, "Unknown Demucs model:"):
            ballroom_scribe.VocalIsolator(demucs_name="unknown", demucs_device="cuda")

    def test_init_validates_torch_device(self):
        ballroom_scribe.VocalIsolator(demucs_name="htdemucs", demucs_device="fake_device")
        self.mock_validate_torch_device.assert_called_once_with("fake_device")

    def test_separate_lazy_loads_demucs(self):
        self.output_vocals.exists.return_value = False
        separator = ballroom_scribe.VocalIsolator(demucs_name="htdemucs", demucs_device="cuda")

        self.mock_demucs_separator_cls.assert_not_called()
        separator.separate(_INPUT_AUDIO, self.output_vocals)
        separator.separate(_INPUT_AUDIO, self.output_vocals)
        self.mock_demucs_separator_cls.assert_called_once_with(model="htdemucs", device="cuda")

    def test_separate_skips_demucs_when_vocal_track_already_exists(self):
        self.output_vocals.exists.return_value = True
        separator = ballroom_scribe.VocalIsolator(demucs_name="htdemucs", demucs_device="cuda")

        separator.separate(_INPUT_AUDIO, self.output_vocals)

        self.output_vocals.exists.assert_called_once()
        self.mock_demucs_separator_cls.assert_not_called()
        self.mock_save_audio.assert_not_called()
        self.mock_atomic_path.assert_not_called()

    def test_separate_invokes_demucs_when_vocal_track_missing(self):
        self.output_vocals.exists.return_value = False

        separator = ballroom_scribe.VocalIsolator(demucs_name="htdemucs", demucs_device="cpu")
        separator.separate(_INPUT_AUDIO, self.output_vocals)

        self.output_vocals.exists.assert_called_once()
        self.mock_demucs_separator_cls.assert_called_once_with(model="htdemucs", device="cpu")
        self.mock_demucs_separator.separate_audio_file.assert_called_once_with(_INPUT_AUDIO)
        self.mock_atomic_path.assert_called_once_with(self.output_vocals)
        self.mock_save_audio.assert_called_once_with(self.mock_vocals, self.partial_vocals, samplerate=_SAMPLE_RATE)

    def test_separate_raises_value_error_when_no_vocals(self):
        self.output_vocals.exists.return_value = False

        self.mock_demucs_separator.separate_audio_file.return_value = (
            MagicMock(),
            {"drums": MagicMock()},
        )

        separator = ballroom_scribe.VocalIsolator(demucs_name="htdemucs", demucs_device="cuda")
        with self.assertRaisesRegex(ValueError, "The audio did not contain vocals."):
            separator.separate(_INPUT_AUDIO, self.output_vocals)

        self.mock_save_audio.assert_not_called()
        self.mock_atomic_path.assert_not_called()


if __name__ == "__main__":
    unittest.main()
