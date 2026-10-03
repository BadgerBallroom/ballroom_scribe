"""Validation of an audio channel index against a media file's first audio stream."""

import pathlib

import ballroom_scribe.utils


def validate_input_channel(input_video: pathlib.Path, input_channel: int) -> None:
    """Raises a `ValueError` if `input_channel` is not a valid channel index for `input_video`.

    This must be checked before extracting audio because `ffmpeg`'s `pan` filter silently maps a nonexistent input
    channel to silence (with a zero exit code) rather than failing.

    Args:
        input_video: Path to the input media file.
        input_channel: Zero-based index of the audio channel to validate.

    Raises:
        ValueError: If `input_channel` is not a valid channel index for the first audio stream of `input_video`.
        subprocess.CalledProcessError: If `ffprobe` fails (e.g., the file does not exist or cannot be read).
    """
    channel_count = ballroom_scribe.utils.count_audio_channels(input_video)
    if not 0 <= input_channel < channel_count:
        raise ValueError(
            f"Audio channel index {input_channel} is out of range (found {channel_count} channel(s) in {input_video})."
        )
