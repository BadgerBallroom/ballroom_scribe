from dataclasses import dataclass

from .word import Word


@dataclass(frozen=True, slots=True)
class TranscriptChunk:
    """Represents a time-windowed slice of transcript text and word timestamps.

    Attributes:
        words: A tuple of words that were spoken in this part of the transcript.
        start_time:
            The number of seconds since the beginning of the original audio file that the first word in the chunk
            started to be spoken.
        end_time:
            The number of seconds since the beginning of the original audio file that the last word in the chunk stopped
            being spoken.
    """

    words: tuple[Word, ...]
    start_time: float
    end_time: float

    def get_text(self) -> str:
        """A flattened representation of the transcript text."""
        return " ".join(word.word for word in self.words)
