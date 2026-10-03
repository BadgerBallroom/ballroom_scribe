"""Utilities for loading audio from file."""

import math
import pathlib
from collections.abc import Generator

import numpy as np
import panns_inference
import soundfile

import ballroom_scribe.models


class AudioLoader:
    """Loads audio from a file."""

    SAMPLE_RATE = panns_inference.config.sample_rate
    DTYPE = "float32"

    def __init__(self, chunk_minutes: float, chunk_overlap_seconds: float):
        """Initializes the instance.

        Args:
            chunk_minutes: How many minutes of audio to load at a time, not counting any overlap.
            chunk_overlap_seconds: How many seconds to overlap between chunks for better classification accuracy.
        """
        if chunk_minutes <= 0.0:
            raise ValueError(f"chunk_minutes must be positive, but got {chunk_minutes}.")

        if chunk_overlap_seconds < 0.0:
            raise ValueError(f"chunk_overlap_seconds must be non-negative, but got {chunk_overlap_seconds}.")

        self._overlap_samples = math.floor(chunk_overlap_seconds * self.SAMPLE_RATE)

        # The core is the part of the chunk that doesn't overlap with the chunk before or after.
        # It also happens to be the number of samples that are read from the file at a time; that's because with each
        # read, the first `_overlap_samples` samples of the core have already been read but the first `_overlap_samples`
        # samples of the next core need to be read.
        self._core_samples = math.floor(chunk_minutes * 60.0 * self.SAMPLE_RATE)

        # This is the actual number of samples in each chunk except the first and last.
        self._chunk_actual_shape = (self._overlap_samples * 2 + self._core_samples,)

    def load(self, input_audio: pathlib.Path) -> Generator[ballroom_scribe.models.AudioChunk]:
        """Loads audio in overlapping chunks from the input audio file.

        The first block is extended by the overlap duration at the end only.
        Every block in the middle is extended by the overlap duration at both the start and the end.
        The last block is extended by the overlap duration at the start only.

        Args:
            input_audio: The path to a 32,000 Hz mono WAV file.
        """
        with soundfile.SoundFile(input_audio, mode="r") as f:
            if f.samplerate != self.SAMPLE_RATE:
                raise ValueError(
                    f"Audio file {input_audio} has sample rate {f.samplerate}, which is not {self.SAMPLE_RATE}."
                )

            if f.frames == 0:
                raise ValueError(f"Audio file {input_audio} has no audio content.")

            if f.frames >= 2**53:
                # Elsewhere, we have to be able to represent the number of samples as a `np.float64`, which has a 52-bit
                # mantissa, giving us 52 bits of precision plus 1 hidden bit.
                raise ValueError("Audio file cannot be longer than approximately 8,919 years.")

            if f.channels != 1:
                raise ValueError(f"Audio file {input_audio} has {f.channels} channels, but only mono is supported.")

            # `SoundFile.blocks` yields blocks of the same length until the last block. This is not what we want; the
            # first block should be slightly shorter because it is only extended by the overlap duration at the end.
            new_block = f.read(frames=self._core_samples + self._overlap_samples, dtype=self.DTYPE)
            yield self._make_audio_chunk(start_samples=0, data=new_block)
            start_samples = new_block.shape[0]

            # A frame contains simultaneous samples from all channels.
            # Because we are working with mono audio, every frame has exactly one sample.
            while start_samples < f.frames:
                old_block = new_block
                new_block = np.empty(self._chunk_actual_shape, dtype=old_block.dtype)

                if self._overlap_samples:
                    # Copy the overlap from the previous block to the current.
                    new_block[: self._overlap_samples] = old_block[-self._overlap_samples :]
                    start_samples -= self._overlap_samples

                read_data = f.read(frames=self._core_samples, dtype=self.DTYPE, out=new_block[self._overlap_samples :])

                # If it's the last chunk, fewer than `self._read_samples` samples may have been read. Slice the block.
                new_block = new_block[: self._overlap_samples + read_data.shape[0]]

                yield self._make_audio_chunk(start_samples=start_samples, data=new_block)
                start_samples += new_block.shape[0]

    def _make_audio_chunk(self, *, start_samples: int, data: soundfile.AudioData) -> ballroom_scribe.models.AudioChunk:
        return ballroom_scribe.models.AudioChunk(
            start_samples=start_samples,
            data=data,
            sample_rate=self.SAMPLE_RATE,
            # The classification is always unknown here. Classifying is `AudioClassifier`'s job.
            classification=ballroom_scribe.models.AudioChunk.Classification.UNKNOWN,
        )
