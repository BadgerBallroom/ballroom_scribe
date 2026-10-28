"""Utility for streaming a mono WAV file into a `WaveformEnvelope`."""

import pathlib

import numpy as np
import soundfile

import ballroom_scribe.gui.models

_TARGET_BLOCK_SECONDS = 60


def compute_waveform_envelope(
    input_audio: pathlib.Path,
    bins_per_second: int = 100,
) -> ballroom_scribe.gui.models.WaveformEnvelope:
    """Streams a mono audio file and computes per-bin min/max amplitude peaks.

    Reads the file in fixed-size blocks via `soundfile.SoundFile.blocks` so multi-hour recordings never need to fit in
    RAM at once.

    Args:
        input_audio: Path to a mono audio file.
        bins_per_second: Target number of min/max bins per second of audio.

    Returns:
        A `WaveformEnvelope` covering the entire audio stream.

    Raises:
        ValueError: If `bins_per_second < 1`, the audio is not mono, the file is empty, or its sample rate is smaller
            than `bins_per_second`.
    """
    if bins_per_second < 1:
        raise ValueError(f"bins_per_second must be at least 1, got {bins_per_second}.")

    with soundfile.SoundFile(input_audio, mode="r") as f:
        if f.channels != 1:
            raise ValueError(f"Audio file {input_audio} has {f.channels} channels, but only mono is supported.")
        if f.frames == 0:
            raise ValueError(f"Audio file {input_audio} has no audio content.")

        samples_per_bin = round(f.samplerate / bins_per_second)
        if samples_per_bin < 1:
            raise ValueError(
                f"Audio sample rate ({f.samplerate}) must be at least bins_per_second ({bins_per_second})."
            )
        actual_bins_per_second = f.samplerate / samples_per_bin
        block_frames = samples_per_bin * max(1, round(_TARGET_BLOCK_SECONDS * actual_bins_per_second))

        min_chunks: list[np.typing.NDArray[np.float32]] = []
        max_chunks: list[np.typing.NDArray[np.float32]] = []
        carry = np.empty(0, dtype=np.float32)

        for raw_block in f.blocks(blocksize=block_frames, dtype="float32"):
            block = np.asarray(raw_block, dtype=np.float32)
            samples = np.concatenate((carry, block)) if carry.size else block
            full_bins = samples.shape[0] // samples_per_bin
            usable = full_bins * samples_per_bin
            if full_bins > 0:
                reshaped = samples[:usable].reshape(full_bins, samples_per_bin)
                min_chunks.append(reshaped.min(axis=1))
                max_chunks.append(reshaped.max(axis=1))
            carry = samples[usable:].copy()

        if carry.size:
            min_chunks.append(np.array([carry.min()], dtype=np.float32))
            max_chunks.append(np.array([carry.max()], dtype=np.float32))

    if not min_chunks:
        raise ValueError(f"Audio file {input_audio} has no audio content.")

    return ballroom_scribe.gui.models.WaveformEnvelope(
        mins=np.concatenate(min_chunks),
        maxs=np.concatenate(max_chunks),
        bins_per_second=actual_bins_per_second,
    )
