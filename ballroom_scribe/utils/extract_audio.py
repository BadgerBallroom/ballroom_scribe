"""Utilities for preprocessing the video files prior to ingestion."""

import concurrent.futures
import pathlib
import subprocess
import threading

import lazy_loader

import ballroom_scribe.utils

# Loaded lazily so lightweight callers (such as the GUI on a cache hit) do not import PyTorch or librosa.
panns_inference = lazy_loader.load("panns_inference")

_CANCEL_POLL_SECONDS = 0.1


def _ffmpeg_command(input_video: pathlib.Path, input_channel: int, output_audio: pathlib.Path) -> list[str]:
    """Builds the `ffmpeg` command extracting `input_channel` of `input_video` to a 32,000 Hz mono WAV."""
    assert panns_inference is not None
    return [
        "ffmpeg",
        "-y",
        "-nostdin",  # Never read the console's keyboard, even when run from a terminal.
        "-i",
        str(input_video),
        "-c:a",
        "pcm_s16le",  # 16-bit PCM audio
        "-map",
        "a",  # Extract only the audio stream
        "-af",
        # Extract only the specified channel without downmixing, and apply a 100 Hz high-pass filter.
        f"pan=mono|c0=c{input_channel},highpass=f=100",
        "-ar",
        str(panns_inference.config.sample_rate),  # Downsample to 32kHz
        str(output_audio),
    ]


def _run_until_done(command: list[str], cancel: threading.Event | None) -> None:
    """Runs `command` to completion like `subprocess.run(check=True)`, killing it if `cancel` is set meanwhile.

    Raises:
        concurrent.futures.CancelledError: If `cancel` was set before the command finished.
        subprocess.CalledProcessError: If the command exited with a non-zero status.
    """
    with subprocess.Popen(command) as process:
        try:
            while True:
                try:
                    process.wait(timeout=None if cancel is None else _CANCEL_POLL_SECONDS)
                    break
                except subprocess.TimeoutExpired:
                    if cancel is not None and cancel.is_set():
                        raise concurrent.futures.CancelledError("Audio extraction was canceled.") from None
        except BaseException:
            # Also covers `KeyboardInterrupt`, so Ctrl+C in the CLI never leaves `ffmpeg` running.
            process.kill()
            raise
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)


def extract_audio(
    input_video: pathlib.Path,
    input_channel: int,
    cache_dir: pathlib.Path,
    *,
    cancel: threading.Event | None = None,
) -> pathlib.Path:
    """Extracts an audio channel into the video's cache directory if it hasn't already been extracted.

    Outputs audio as a 16-bit, 32,000 Hz, mono WAV file named `Audio-Channel-{input_channel}.wav` inside `cache_dir`.
    This is exactly the format that `AudioLoader` requires. The file is written through `atomic_path`, so it only
    exists under its final name once it is complete.

    Args:
        input_video: Path to the input video file.
        input_channel: Zero-based index of the audio channel to extract.
        cache_dir: The video's cache directory. It must already exist.
        cancel: Optional event; when set while `ffmpeg` runs, the process is killed, the partial output is removed, and
            `concurrent.futures.CancelledError` is raised.

    Returns:
        The path to the extracted WAV file inside `cache_dir`.

    Raises:
        ValueError: If `input_channel` is not a valid channel index for the first audio stream of `input_video`.
        subprocess.CalledProcessError: If `ffprobe` or `ffmpeg` fails.
        concurrent.futures.CancelledError: If `cancel` was set before extraction finished.
    """
    output_audio = cache_dir / f"Audio-Channel-{input_channel}.wav"
    if output_audio.exists():
        print("Audio already extracted. Not extracting again.")
        return output_audio

    ballroom_scribe.utils.validate_input_channel(input_video=input_video, input_channel=input_channel)
    with ballroom_scribe.utils.atomic_path(output_audio) as partial_audio:
        _run_until_done(_ffmpeg_command(input_video, input_channel, partial_audio), cancel)
    return output_audio
