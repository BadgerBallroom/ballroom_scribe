from dataclasses import dataclass
from typing import override

from .audio_chunk import AudioChunk
from .chunk_abc import Chunk


@dataclass(frozen=True, slots=True)
class MusicValley(Chunk):
    """Represents the transition between two dances and holds only the detached speech `AudioChunks`.

    Attributes:
        speech_chunks: Chunks of speech within this music valley.

    Inherited Attributes:
        From `Chunk`: `start_samples`, `sample_rate`, `duration_samples`. See :class:`Chunk` for details.
    """

    speech_chunks: tuple[AudioChunk, ...]

    @override
    def __bool__(self) -> bool:
        return any(self.speech_chunks)
