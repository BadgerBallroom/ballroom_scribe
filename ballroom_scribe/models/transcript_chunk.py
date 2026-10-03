from dataclasses import dataclass
from typing import override

from .chunk_abc import Chunk
from .word import Word


@dataclass(frozen=True, slots=True)
class TranscriptChunk(Chunk):
    """Represents a time-windowed slice of transcript text and word timestamps.

    Attributes:
        words: A tuple of words that were spoken in this part of the transcript.

    Inherited Attributes:
        From `Chunk`: `start_samples`, `sample_rate`, `duration_samples`. See :class:`Chunk` for details.
    """

    words: tuple[Word, ...]

    def get_text(self) -> str:
        """A flattened representation of the transcript text."""
        return " ".join(word.word for word in self.words)

    @override
    def __bool__(self) -> bool:
        return bool(self.words)
