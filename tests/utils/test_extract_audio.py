import concurrent.futures
import pathlib
import subprocess
import threading
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
        self.mock_popen_cls = self.enterContext(patch.object(subprocess, "Popen", autospec=True))
        self.mock_popen_cls.return_value.__exit__.return_value = False  # never swallow exceptions
        self.mock_process = self.mock_popen_cls.return_value.__enter__.return_value
        self.mock_process.returncode = 0
        self.mock_validate_input_channel = self.enterContext(
            patch.object(ballroom_scribe.utils, "validate_input_channel", autospec=True)
        )
        self.mock_path_exists = self.enterContext(patch.object(pathlib.Path, "exists", autospec=True))
        self.mock_path_exists.return_value = False

        self.mock_atomic_path = self.enterContext(patch.object(ballroom_scribe.utils, "atomic_path", autospec=True))
        self.mock_partial_audio = MagicMock(spec=pathlib.Path)
        self.mock_atomic_path.return_value.__enter__.return_value = self.mock_partial_audio

    def _expected_command(self):
        return [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-i",
            str(_INPUT_VIDEO),
            "-c:a",
            "pcm_s16le",
            "-map",
            "a",
            "-af",
            "pan=mono|c0=c1,highpass=f=100",
            "-ar",
            str(panns_inference.config.sample_rate),
            str(self.mock_partial_audio),
        ]

    def test_skips_extraction_when_output_file_exists(self):
        """Should skip FFmpeg execution if the destination audio already exists."""
        self.mock_path_exists.return_value = True

        result = ballroom_scribe.utils.extract_audio(_INPUT_VIDEO, 1, _CACHE_DIR)

        self.assertEqual(result, _EXTRACTED_AUDIO)
        self.mock_path_exists.assert_called_once_with(_EXTRACTED_AUDIO)
        self.mock_validate_input_channel.assert_not_called()
        self.mock_atomic_path.assert_not_called()
        self.mock_popen_cls.assert_not_called()

    def test_runs_ffmpeg_when_output_file_does_not_exist(self):
        """Should invoke FFmpeg with exact arguments to write the audio atomically when output file is missing."""
        result = ballroom_scribe.utils.extract_audio(_INPUT_VIDEO, 1, _CACHE_DIR)

        self.assertEqual(result, _EXTRACTED_AUDIO)
        self.mock_path_exists.assert_called_once_with(_EXTRACTED_AUDIO)
        self.mock_validate_input_channel.assert_called_once_with(input_video=_INPUT_VIDEO, input_channel=1)
        self.mock_atomic_path.assert_called_once_with(_EXTRACTED_AUDIO)
        self.mock_popen_cls.assert_called_once_with(self._expected_command())
        self.mock_process.wait.assert_called_once_with(timeout=None)
        self.mock_process.kill.assert_not_called()

    def test_polls_for_cancellation_when_event_is_given(self):
        """Should wait in short intervals so a cancellation request is noticed promptly."""
        cancel = threading.Event()
        self.mock_process.wait.side_effect = [subprocess.TimeoutExpired(cmd="ffmpeg", timeout=0.1), 0]

        ballroom_scribe.utils.extract_audio(_INPUT_VIDEO, 1, _CACHE_DIR, cancel=cancel)

        self.assertEqual(self.mock_process.wait.call_count, 2)
        poll_timeout = self.mock_process.wait.call_args.kwargs["timeout"]
        self.assertGreater(poll_timeout, 0.0)
        self.assertLessEqual(poll_timeout, 1.0)
        self.mock_process.kill.assert_not_called()

    def test_kills_ffmpeg_and_raises_when_cancelled(self):
        """Should kill FFmpeg, let `atomic_path` discard the partial file, and raise `CancelledError`."""
        cancel = threading.Event()
        cancel.set()
        self.mock_process.wait.side_effect = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=0.1)

        with self.assertRaises(concurrent.futures.CancelledError):
            ballroom_scribe.utils.extract_audio(_INPUT_VIDEO, 1, _CACHE_DIR, cancel=cancel)

        self.mock_process.kill.assert_called_once_with()
        self.mock_atomic_path.return_value.__exit__.assert_called_once_with(concurrent.futures.CancelledError, ANY, ANY)

    def test_kills_ffmpeg_when_interrupted(self):
        """Should kill FFmpeg if waiting is interrupted (e.g., by Ctrl+C) and re-raise."""
        self.mock_process.wait.side_effect = KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            ballroom_scribe.utils.extract_audio(_INPUT_VIDEO, 1, _CACHE_DIR)

        self.mock_process.kill.assert_called_once_with()

    def test_propagates_ffmpeg_failure(self):
        """Should raise CalledProcessError if FFmpeg exits with a non-zero status, letting the atomic path clean up."""
        self.mock_process.returncode = 1

        with self.assertRaises(subprocess.CalledProcessError) as context:
            ballroom_scribe.utils.extract_audio(_INPUT_VIDEO, 1, _CACHE_DIR)

        self.assertEqual(context.exception.returncode, 1)
        self.assertEqual(context.exception.cmd, self._expected_command())
        self.mock_atomic_path.return_value.__exit__.assert_called_once_with(subprocess.CalledProcessError, ANY, ANY)

    def test_propagates_invalid_channel(self):
        """Should not run FFmpeg if `validate_input_channel` rejects the channel."""
        self.mock_validate_input_channel.side_effect = ValueError("out of range")

        with self.assertRaises(ValueError):
            ballroom_scribe.utils.extract_audio(_INPUT_VIDEO, 9, _CACHE_DIR)

        self.mock_atomic_path.assert_not_called()
        self.mock_popen_cls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
