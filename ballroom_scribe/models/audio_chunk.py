import enum
from dataclasses import dataclass, field, replace
from typing import Self, override

import librosa
import numpy as np
import soundfile

from .chunk_abc import Chunk


@dataclass(frozen=True, slots=True)
class AudioChunk(Chunk):
    """Represents a segment of audio with a label and timestamps.

    Attributes:
        data: The audio data as a NumPy array.
        classification: The classification of the audio chunk (e.g., speech, music, other).

    Inherited Attributes:
        From `Chunk`: `start_samples`, `sample_rate`, `duration_samples`. See :class:`Chunk` for details.
    """

    type ClassificationNumPy = np.int8

    class Classification(enum.IntFlag):
        NONE = 0
        UNKNOWN = enum.auto()
        SPEECH = enum.auto()
        MUSIC = enum.auto()

        @classmethod
        @override
        def _missing_(cls, value: object) -> Self:
            if isinstance(value, np.integer):
                return super()._missing_(int(value))
            return super()._missing_(value)

        def numpy(self) -> AudioChunk.ClassificationNumPy:
            """Returns this enum value as the smallest NumPy integer type that can represent any possible value of this enum.

            To convert back to the enum type, pass the integer to the constructor, e.g. `Classification(np.int8(3))`."""
            return np.int8(self)

    data: soundfile.AudioData
    classification: Classification
    duration_samples: int = field(init=False)

    def __post_init__(self):
        # Make the NumPy array immutable.
        # Technically, you can set the flag back to `True`, but this is enough to catch accidental attempts at mutation.
        self.data.flags.writeable = False

        object.__setattr__(self, "duration_samples", self.data.shape[0])

    def classification_is(self, classification: Classification) -> bool:
        """Returns whether the audio chunk has the given classification."""
        return bool(self.classification & classification)

    def split(self, *, new_start_samples: int) -> tuple[Self, Self]:
        """Splits the audio chunk into two.

        In each resulting chunk, the `data` attribute is a view of the original. No copying is performed. Even if one
        chunk is deleted, no memory will be freed. To free up memory, call the :meth:`copy` method of one resulting
        chunk, store that copy, and then let the original chunk and resulting chunks be garbage collected.

        Args:
            new_start_samples:
                The number of samples from the start of the original audio chunk to start the second audio chunk.
                Everything before that goes into the first chunk.

        Returns:
            A tuple containing two `AudioChunk` instances, before and after the split.
        """
        if new_start_samples < self.start_samples or self.end_samples < new_start_samples:
            raise ValueError(f"new_start_samples must be in the range [{self.start_samples}, {self.end_samples}].")

        split_index = new_start_samples - self.start_samples
        return (
            replace(
                self,
                data=self.data[:split_index],
            ),
            replace(
                self,
                data=self.data[split_index:],
                start_samples=new_start_samples,
            ),
        )

    def copy(self) -> Self:
        """Returns a copy of this `AudioChunk` that shares no memory with the original."""
        return replace(self, data=self.data.copy())

    def copy_if_sharing_memory(self) -> Self:
        """Returns a copy of this `AudioChunk` if it shares memory with another, else returns this `AudioChunk`."""
        if self.data.base is not None:
            return self.copy()
        return self

    def resample(self, new_sample_rate: int) -> Self:
        """Returns a copy of this `AudioChunk` with the specified sample rate if it is different.

        If it is the same, returns this `AudioChunk`.

        Args:
            new_sample_rate: The new sample rate.

        Returns:
            The `AudioChunk` with the specified sample rate.
        """
        if new_sample_rate == self.sample_rate:
            return self

        return replace(
            self,
            start_samples=round(self.start_seconds * new_sample_rate),
            data=librosa.resample(self.data, orig_sr=self.sample_rate, target_sr=new_sample_rate),
            sample_rate=new_sample_rate,
        )

    def __add__(self, other):
        """Appends another AudioChunk to this one, creating a new AudioChunk that spans both.

        The sample rate and classification must match. The start time of the second chunk will be ignored.
        """
        if not isinstance(other, AudioChunk):
            raise TypeError("Can only add AudioChunk instances.")

        if self.sample_rate != other.sample_rate:
            raise ValueError("Sample rates must match.")

        if self.classification != other.classification:
            raise ValueError("Classifications must match.")

        return replace(self, data=np.concatenate([self.data, other.data]))

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, type(self))
            and self.start_samples == value.start_samples
            and np.array_equal(self.data, value.data)
            and self.sample_rate == value.sample_rate
            and self.classification == value.classification
        )

    @override
    def __bool__(self) -> bool:
        return bool(self.duration_samples)
