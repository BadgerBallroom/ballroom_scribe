import pathlib
import subprocess
import unittest
from unittest.mock import patch

from parameterized import param, parameterized

import ballroom_scribe.utils

_INPUT_VIDEO = pathlib.Path("raw_data/clip.mp4")


class TestCountAudioChannels(unittest.TestCase):
    """Unit tests for the `count_audio_channels` function."""

    def setUp(self):
        self.mock_subprocess_run = self.enterContext(patch.object(subprocess, "run", autospec=True))
        self.mock_subprocess_run.return_value.stdout = "2\n"

    def test_runs_ffprobe_with_exact_arguments(self):
        """Should probe only the first audio stream and print the bare channel count."""
        ballroom_scribe.utils.count_audio_channels(_INPUT_VIDEO)

        self.mock_subprocess_run.assert_called_once_with(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=channels",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                _INPUT_VIDEO,
            ],
            capture_output=True,
            text=True,
            check=True,
        )

    @parameterized.expand(
        [
            param("mono", stdout="1\n", expected=1),
            param("stereo", stdout="2\n", expected=2),
            param("surround", stdout="6\n", expected=6),
            param("no_audio_stream", stdout="", expected=0),
        ]
    )
    def test_parses_channel_count(self, _, stdout, expected):
        """Should parse ffprobe's output, treating an empty output as zero channels."""
        self.mock_subprocess_run.return_value.stdout = stdout

        self.assertEqual(ballroom_scribe.utils.count_audio_channels(_INPUT_VIDEO), expected)

    def test_propagates_ffprobe_failure(self):
        """Should bubble up CalledProcessError if ffprobe fails (e.g., missing file)."""
        self.mock_subprocess_run.side_effect = subprocess.CalledProcessError(returncode=1, cmd=["ffprobe"])

        with self.assertRaises(subprocess.CalledProcessError):
            ballroom_scribe.utils.count_audio_channels(_INPUT_VIDEO)


if __name__ == "__main__":
    unittest.main()
