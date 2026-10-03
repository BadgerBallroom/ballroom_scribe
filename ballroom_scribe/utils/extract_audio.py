"""Utilities for preprocessing the video files prior to ingestion."""

import pathlib
import subprocess

import panns_inference

import ballroom_scribe.utils


def extract_audio(input_video: pathlib.Path, input_channel: int, cache_dir: pathlib.Path) -> pathlib.Path:
    """Extracts an audio channel into the video's cache directory if it hasn't already been extracted.

    Outputs audio as a 16-bit, 32,000 Hz, mono WAV file named `Audio-Channel-{input_channel}.wav` inside `cache_dir`.
    This is exactly the format that `AudioLoader` requires.

    Args:
        input_video: Path to the input video file.
        input_channel: Zero-based index of the audio channel to extract.
        cache_dir: The video's cache directory. It must already exist.

    Returns:
        The path to the extracted WAV file inside `cache_dir`.

    Raises:
        ValueError: If `input_channel` is not a valid channel index for the first audio stream of `input_video`.
        subprocess.CalledProcessError: If `ffprobe` or `ffmpeg` fails.
    """
    output_audio = cache_dir / f"Audio-Channel-{input_channel}.wav"
    if output_audio.exists():
        print("Audio already extracted. Not extracting again.")
    else:
        ballroom_scribe.utils.validate_input_channel(input_video=input_video, input_channel=input_channel)
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
                    "-af",
                    # Extract only the specified channel without downmixing, and apply a 100 Hz high-pass filter.
                    f"pan=mono|c0=c{input_channel},highpass=f=100",
                    "-ar",
                    str(panns_inference.config.sample_rate),  # Downsample to 32kHz
                    str(partial_audio),
                ],
                check=True,
            )
    return output_audio
