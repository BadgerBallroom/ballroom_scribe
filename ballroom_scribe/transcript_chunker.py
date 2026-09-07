"""Sliding window chunking algorithm for transcript words."""

from collections.abc import Generator, Iterable

import ballroom_scribe.models


def _words_to_chunk(words: list[ballroom_scribe.models.Word]) -> ballroom_scribe.models.TranscriptChunk:
    """Converts a list of Word objects into a TranscriptChunk."""
    if not words:
        return ballroom_scribe.models.TranscriptChunk(
            words=(),
            start_time=0.0,
            end_time=0.0,
        )

    return ballroom_scribe.models.TranscriptChunk(
        words=tuple(words),
        start_time=words[0].start,
        end_time=words[-1].end,
    )


class TranscriptChunker:
    """Chunks word-level transcripts into overlapping time windows."""

    def __init__(self, split_on_silence_threshold: float):
        self._split_on_silence_threshold = split_on_silence_threshold

    def chunk(self, words: Iterable[ballroom_scribe.models.Word]) -> Generator[ballroom_scribe.models.TranscriptChunk]:
        """Splits the words into chunks at gaps of silence."""
        current_chunk: list[ballroom_scribe.models.Word] = []
        for word in words:
            # If the chunk has words, check the gap between the last word and the current one.
            if current_chunk:
                last_word = current_chunk[-1]
                if word.start - last_word.end > self._split_on_silence_threshold:
                    # The gap is large enough; yield the current chunk and start a new one.
                    yield _words_to_chunk(current_chunk)
                    current_chunk = []

            current_chunk.append(word)

        # Yield the final chunk if it contains anything.
        if current_chunk:
            yield _words_to_chunk(current_chunk)
