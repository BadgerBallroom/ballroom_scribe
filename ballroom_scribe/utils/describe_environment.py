"""Utility that describes the software that a process runs on, for logs and problem reports."""

import importlib.metadata
import platform
import sys

import ballroom_scribe.utils

# The distributions whose versions matter when diagnosing a problem. Their versions are read from the metadata of
# what is installed, which never imports them: the editor must not load the pipeline's models (see
# `tests/gui/test_import_footprint.py`). The NVIDIA distributions are installed only by `requirements-cuda.txt`.
_DISTRIBUTIONS = (
    "torch",
    "faster-whisper",
    "ctranslate2",
    "panns-inference",
    "librosa",
    "numpy",
    "scipy",
    "soundfile",
    "ollama",
    "PySide6",
    "nvidia-cublas-cu12",
    "nvidia-cudnn-cu12",
)
_NOT_INSTALLED = "not installed"


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return _NOT_INSTALLED


def describe_environment() -> str:
    """Returns a few lines that name this copy of Ballroom Scribe, Python, the operating system, and key packages.

    None of the packages is imported, so the editor can call this without loading the pipeline's models.

    Returns:
        Three lines without a line break at the end, such as:

            Ballroom Scribe 3f2a1bc04d5e (2026-11-20)
            Python 3.14.3 (/usr/bin/python3) on macOS-26.7.3-arm64-arm-64bit-Mach-O
            Packages: torch 2.14.0, faster-whisper 1.2.1, ..., nvidia-cudnn-cu12 not installed
    """
    packages = ", ".join(f"{name} {_distribution_version(name)}" for name in _DISTRIBUTIONS)
    return "\n".join(
        (
            f"Ballroom Scribe {ballroom_scribe.utils.describe_version()}",
            f"Python {platform.python_version()} ({sys.executable}) on {platform.platform()}",
            f"Packages: {packages}",
        )
    )
