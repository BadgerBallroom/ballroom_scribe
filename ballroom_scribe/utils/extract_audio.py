"""Utilities for preprocessing the video files prior to ingestion."""

import concurrent.futures
import pathlib
import subprocess
import sys
import threading

import ballroom_scribe
import ballroom_scribe.utils

_CANCEL_POLL_SECONDS = 0.1


def _ffmpeg_command(input_video: pathlib.Path, input_channel: int, output_audio: pathlib.Path) -> list[str]:
    """Builds the `ffmpeg` command extracting `input_channel` of `input_video` to a 32,000 Hz mono WAV."""
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
        str(ballroom_scribe.PannInferencer.SAMPLE_RATE),  # Downsample to 32kHz
        str(output_audio),
    ]


def _wait_for_process(process: subprocess.Popen[bytes], cancel: threading.Event | None) -> None:
    """Waits for `process` to exit, polling `cancel` when given and raising `CancelledError` once it is set."""
    while True:
        try:
            process.wait(timeout=None if cancel is None else _CANCEL_POLL_SECONDS)
            return
        except subprocess.TimeoutExpired:
            if cancel is not None and cancel.is_set():
                raise concurrent.futures.CancelledError("Audio extraction was canceled.") from None


def _run_ffmpeg(command: list[str], cancel: threading.Event | None) -> None:
    """Runs the `ffmpeg` `command` to completion, killing it on interruption and reporting how it exited.

    Raises:
        concurrent.futures.CancelledError: If `cancel` was set before `ffmpeg` finished.
        subprocess.CalledProcessError: If `ffmpeg` exited with a non-zero status.
    """
    interrupted = False
    try:
        with subprocess.Popen(command) as process:
            try:
                _wait_for_process(process, cancel)
            except BaseException:
                # Also covers `KeyboardInterrupt`, so Ctrl+C in the CLI never leaves `ffmpeg` running.
                interrupted = True
                process.kill()
                raise
    finally:
        if interrupted:
            # Leading newline keeps any `\r`-terminated FFmpeg progress line intact rather than overwriting it.
            print("\nFFmpeg interrupted.", file=sys.stderr, flush=True)

    if process.returncode != 0:
        print(f"FFmpeg exited with error (status {process.returncode}).", file=sys.stderr, flush=True)
        raise subprocess.CalledProcessError(process.returncode, command)
    print(f"FFmpeg exited normally (status {process.returncode}).", file=sys.stderr, flush=True)


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
        _run_ffmpeg(_ffmpeg_command(input_video, input_channel, partial_audio), cancel)
    return output_audio
