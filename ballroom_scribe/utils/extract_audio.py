"""Utilities for preprocessing the video files prior to ingestion."""

import pathlib
import subprocess

import ballroom_scribe.utils


def extract_audio(input_video: pathlib.Path, output_audio: pathlib.Path) -> None:
    """Extracts audio to the target path if it doesn't already exist.

    Outputs audio as a 16-bit mono WAV file.

    Args:
        input_video: Path to the input video file.
        output_audio: Path to the output WAV file.

    Raises:
        subprocess.CalledProcessError: If `ffprobe` or `ffmpeg` fails.
    """
    if output_audio.exists():
        print("Audio already extracted. Not extracting again.")
    else:
        with ballroom_scribe.utils.atomic_path(output_audio) as partial_audio:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    input_video,
                    "-c:a",
                    "pcm_s16le",  # 16-bit PCM audio
                    "-map",
                    "a",  # Extract only the audio stream
                    "-ac",
                    "1",  # Convert to mono
                    str(partial_audio),
                ],
                check=True,
            )
