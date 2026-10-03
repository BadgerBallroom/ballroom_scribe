"""Utilities for probing the audio streams of media files."""

import pathlib
import subprocess


def count_audio_channels(input_video: pathlib.Path) -> int:
    """Counts the channels in the first audio stream of a media file using `ffprobe`.

    Args:
        input_video: Path to the input media file.

    Returns:
        The number of channels in the first audio stream, or `0` if the file has no audio stream.

    Raises:
        subprocess.CalledProcessError: If `ffprobe` fails (e.g., the file does not exist or cannot be read).
    """
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",  # The first audio stream, whose channels the `pan` filter in `extract_audio` indexes.
            "-show_entries",
            "stream=channels",
            "-of",
            "default=noprint_wrappers=1:nokey=1",  # Print only the bare value, e.g. `2`.
            input_video,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    # `ffprobe` prints nothing (and still exits successfully) when the file has no audio stream.
    channel_count = result.stdout.strip()
    return int(channel_count) if channel_count else 0
