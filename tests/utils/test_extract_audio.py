import pathlib
import subprocess
import unittest
from unittest.mock import ANY, MagicMock, patch

import panns_inference

import ballroom_scribe.utils

_INPUT_VIDEO = pathlib.Path("raw_data/clip.mp4")
_CACHE_DIR = pathlib.Path("fake_cache")
_EXTRACTED_AUDIO = _CACHE_DIR / "Audio-Channel-1.wav"


class TestExtractAudio(unittest.TestCase):
    """Tests for the `extract_audio` function."""

    def setUp(self):
        self.mock_subprocess_run = self.enterContext(patch.object(subprocess, "run", autospec=True))
        self.mock_validate_input_channel = self.enterContext(
            patch.object(ballroom_scribe.utils, "validate_input_channel", autospec=True)
        )
        self.mock_path_exists = self.enterContext(patch.object(pathlib.Path, "exists", autospec=True))
        self.mock_path_exists.return_value = False

        self.mock_atomic_path = self.enterContext(patch.object(ballroom_scribe.utils, "atomic_path", autospec=True))
        self.mock_partial_audio = MagicMock(spec=pathlib.Path)
        self.mock_atomic_path.return_value.__enter__.return_value = self.mock_partial_audio

    def test_skips_extraction_when_output_file_exists(self):
        """Should skip FFmpeg execution if the destination audio already exists."""
        self.mock_path_exists.return_value = True

        result = ballroom_scribe.utils.extract_audio(_INPUT_VIDEO, 1, _CACHE_DIR)

        self.assertEqual(result, _EXTRACTED_AUDIO)
        self.mock_path_exists.assert_called_once_with(_EXTRACTED_AUDIO)
        self.mock_validate_input_channel.assert_not_called()
        self.mock_atomic_path.assert_not_called()
        self.mock_subprocess_run.assert_not_called()

    def test_runs_ffmpeg_when_output_file_does_not_exist(self):
        """Should invoke FFmpeg with exact arguments to write the audio atomically when output file is missing."""
        result = ballroom_scribe.utils.extract_audio(_INPUT_VIDEO, 1, _CACHE_DIR)

        self.assertEqual(result, _EXTRACTED_AUDIO)
        self.mock_path_exists.assert_called_once_with(_EXTRACTED_AUDIO)
        self.mock_validate_input_channel.assert_called_once_with(input_video=_INPUT_VIDEO, input_channel=1)
        self.mock_atomic_path.assert_called_once_with(_EXTRACTED_AUDIO)
        self.mock_subprocess_run.assert_called_once_with(
            [
                "ffmpeg",
                "-y",
                "-i",
                _INPUT_VIDEO,
                "-c:a",
                "pcm_s16le",
                "-map",
                "a",
                "-af",
                "pan=mono|c0=c1,highpass=f=100",
                "-ar",
                str(panns_inference.config.sample_rate),
                str(self.mock_partial_audio),
            ],
            check=True,
        )

    def test_propagates_ffmpeg_failure(self):
        """Should bubble up CalledProcessError if FFmpeg fails, letting the atomic path clean up."""
        self.mock_subprocess_run.side_effect = subprocess.CalledProcessError(returncode=1, cmd=["ffmpeg"])

        with self.assertRaises(subprocess.CalledProcessError):
            ballroom_scribe.utils.extract_audio(_INPUT_VIDEO, 1, _CACHE_DIR)

        self.mock_atomic_path.return_value.__exit__.assert_called_once_with(subprocess.CalledProcessError, ANY, ANY)

    def test_propagates_invalid_channel(self):
        """Should not run FFmpeg if `validate_input_channel` rejects the channel."""
        self.mock_validate_input_channel.side_effect = ValueError("out of range")

        with self.assertRaises(ValueError):
            ballroom_scribe.utils.extract_audio(_INPUT_VIDEO, 9, _CACHE_DIR)

        self.mock_atomic_path.assert_not_called()
        self.mock_subprocess_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
