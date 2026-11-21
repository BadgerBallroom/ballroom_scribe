import importlib.metadata
import platform
import sys
import unittest
from unittest.mock import call, patch

import ballroom_scribe.utils

_INSTALLED_VERSIONS = {
    "torch": "2.14.0+cu132",
    "faster-whisper": "1.2.1",
    "ctranslate2": "4.8.2",
    "panns-inference": "0.1.1",
    "librosa": "1.0.0",
    "numpy": "2.5.4",
    "scipy": "1.18.1",
    "soundfile": "0.14.0",
    "ollama": "0.6.2",
    "nvidia-cublas-cu12": "12.9.2.10",
}


def _installed_version(distribution_name: str) -> str:
    """Stand-in for `importlib.metadata.version` on a machine without PySide6 and without cuDNN."""
    if distribution_name not in _INSTALLED_VERSIONS:
        raise importlib.metadata.PackageNotFoundError(distribution_name)
    return _INSTALLED_VERSIONS[distribution_name]


class TestDescribeEnvironment(unittest.TestCase):
    """Unit tests for the `describe_environment` function."""

    def setUp(self):
        self.mock_describe_version = self.enterContext(
            patch.object(ballroom_scribe.utils, "describe_version", autospec=True, return_value="0911b61b64dc-dirty")
        )
        self.mock_version = self.enterContext(
            patch.object(importlib.metadata, "version", autospec=True, side_effect=_installed_version)
        )
        self.enterContext(patch.object(platform, "python_version", autospec=True, return_value="3.14.3"))
        self.enterContext(patch.object(platform, "platform", autospec=True, return_value="Windows-11-10.0.26100-SP0"))
        self.enterContext(patch.object(sys, "executable", r"C:\Python314\pythonw.exe"))

    def test_names_the_version_python_the_platform_and_the_packages(self):
        self.assertEqual(
            ballroom_scribe.utils.describe_environment().splitlines(),
            [
                "Ballroom Scribe 0911b61b64dc-dirty",
                r"Python 3.14.3 (C:\Python314\pythonw.exe) on Windows-11-10.0.26100-SP0",
                (
                    "Packages: torch 2.14.0+cu132, faster-whisper 1.2.1, ctranslate2 4.8.2, panns-inference 0.1.1,"
                    " librosa 1.0.0, numpy 2.5.4, scipy 1.18.1, soundfile 0.14.0, ollama 0.6.2,"
                    " PySide6 not installed, nvidia-cublas-cu12 12.9.2.10, nvidia-cudnn-cu12 not installed"
                ),
            ],
        )
        self.mock_describe_version.assert_called_once_with()

    def test_has_no_line_break_at_the_end(self):
        self.assertFalse(ballroom_scribe.utils.describe_environment().endswith("\n"))

    def test_reads_the_version_of_each_distribution_from_its_metadata(self):
        ballroom_scribe.utils.describe_environment()

        # `importlib.metadata.version` reads what is installed without importing it, which keeps the pipeline's
        # models out of the editor's process.
        self.assertEqual(
            self.mock_version.call_args_list,
            [
                call("torch"),
                call("faster-whisper"),
                call("ctranslate2"),
                call("panns-inference"),
                call("librosa"),
                call("numpy"),
                call("scipy"),
                call("soundfile"),
                call("ollama"),
                call("PySide6"),
                call("nvidia-cublas-cu12"),
                call("nvidia-cudnn-cu12"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
