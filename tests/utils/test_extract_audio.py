import pathlib
import subprocess
import unittest
from unittest.mock import ANY, MagicMock, patch

import ballroom_scribe.utils

_INPUT_VIDEO = pathlib.Path("raw_data/clip.mp4")


class TestExtractAudio(unittest.TestCase):
    """Tests for the `extract_audio` function."""

    def setUp(self):
        self.mock_subprocess_run = self.enterContext(patch.object(subprocess, "run", autospec=True))

        self.mock_atomic_path = self.enterContext(patch.object(ballroom_scribe.utils, "atomic_path", autospec=True))
        self.mock_partial_audio = MagicMock(spec=pathlib.Path)
        self.mock_atomic_path.return_value.__enter__.return_value = self.mock_partial_audio

    def test_skips_extraction_when_output_file_exists(self):
        """Should skip FFmpeg execution if the destination audio already exists."""
        mock_output = MagicMock(spec=pathlib.Path)
        mock_output.exists.return_value = True

        ballroom_scribe.utils.extract_audio(_INPUT_VIDEO, mock_output)

        mock_output.exists.assert_called_once()
        self.mock_atomic_path.assert_not_called()
        self.mock_subprocess_run.assert_not_called()

    def test_runs_ffmpeg_when_output_file_does_not_exist(self):
        """Should invoke FFmpeg with exact arguments to write the audio atomically when output file is missing."""
        mock_output = MagicMock(spec=pathlib.Path)
        mock_output.exists.return_value = False

        ballroom_scribe.utils.extract_audio(_INPUT_VIDEO, mock_output)

        mock_output.exists.assert_called_once()
        self.mock_atomic_path.assert_called_once_with(mock_output)
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
                "-ac",
                "1",
                str(self.mock_partial_audio),
            ],
            check=True,
        )

    def test_propagates_ffmpeg_failure(self):
        """Should bubble up CalledProcessError if FFmpeg fails, letting the atomic path clean up."""
        mock_output = MagicMock(spec=pathlib.Path)
        mock_output.exists.return_value = False
        self.mock_subprocess_run.side_effect = subprocess.CalledProcessError(returncode=1, cmd=["ffmpeg"])

        with self.assertRaises(subprocess.CalledProcessError):
            ballroom_scribe.utils.extract_audio(_INPUT_VIDEO, mock_output)

        self.mock_atomic_path.return_value.__exit__.assert_called_once_with(subprocess.CalledProcessError, ANY, ANY)


if __name__ == "__main__":
    unittest.main()
