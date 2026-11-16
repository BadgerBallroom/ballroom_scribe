from abc import ABC, abstractmethod
from dataclasses import dataclass

import ballroom_scribe.utils


@dataclass(frozen=True, slots=True)
class Chunk(ABC):
    """Base class for a chunk with a start sample offset, sample rate, duration in samples, and source audio channel.

    Attributes:
        start_samples:
            The index of the sample in the original audio file at which this chunk starts. In other words, the number of
            samples in the original audio file before the start of this chunk.
        sample_rate:
            The number of audio samples per second. If this chunk contains something other than audio, then this is the
            sample rate of the original audio on which this was based.
        duration_samples:
            The duration of the chunk in samples. If this chunk contains something other than audio, then this is the
            duration of the original audio on which this was based.
        channel:
            The zero-based index of the audio channel of the input video from which the original audio file was
            extracted. Every cache file derived from a chunk includes it in its name, so results for different channels
            of the same video never collide.
    """

    start_samples: int
    sample_rate: int
    duration_samples: int
    channel: int

    @property
    def start_seconds(self) -> float:
        """The start time of the chunk, expressed as seconds since the beginning of the original audio file."""
        return self.start_samples / self.sample_rate

    @property
    def duration_seconds(self) -> float:
        """The duration of the chunk in seconds."""
        return self.duration_samples / self.sample_rate

    @property
    def end_samples(self) -> int:
        """The index of the sample in the original audio file at which this chunk ends.

        In other words, the number of samples in the original audio file before the start of this chunk, plus the number
        of samples in this chunk.
        """
        return self.start_samples + self.duration_samples

    @property
    def end_seconds(self) -> float:
        """The end time of the chunk, expressed as seconds since the beginning of the original audio file."""
        return self.start_seconds + self.duration_seconds

    @property
    def midpoint_seconds(self) -> float:
        """The midpoint of the chunk, expressed as seconds since the beginning of the original audio file."""
        return (self.start_seconds + self.end_seconds) / 2

    def get_filename_stem(self) -> str:
        """Returns a filename stem formatted from the start and end sample indices and the audio channel."""
        return f"{self.start_samples:010d}-{self.end_samples:010d}-{self.channel}"

    def format_time_range(self) -> str:
        """Returns a formatted string of the chunk's start time, end time, and duration."""
        start = ballroom_scribe.utils.format_duration(self.start_seconds)
        end = ballroom_scribe.utils.format_duration(self.end_seconds)
        duration = ballroom_scribe.utils.format_duration(self.duration_seconds)
        return f"{start} - {end}\t({duration})"

    @abstractmethod
    def __bool__(self) -> bool:
        """Returns whether the chunk contains any content."""
