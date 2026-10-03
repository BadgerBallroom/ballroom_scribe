"""Utilities for generating `TranscriptChunk`s for testing purposes."""

import ballroom_scribe.models
import tests.helpers.word_chunk


class Maker(tests.helpers.word_chunk.Maker[ballroom_scribe.models.Word, ballroom_scribe.models.TranscriptChunk]):
    """Generates `TranscriptChunk`s for testing purposes."""

    SAMPLE_RATE = 10

    def __init__(self, start_samples: int = 0):
        super().__init__(
            word_maker=lambda word, start, end: ballroom_scribe.models.Word(
                word=word,
                start=start,
                end=end,
            ),
            chunk_maker=lambda start, end, words: ballroom_scribe.models.TranscriptChunk(
                start_samples=round(start * self.SAMPLE_RATE),
                sample_rate=self.SAMPLE_RATE,
                duration_samples=round((end - start) * self.SAMPLE_RATE),
                words=tuple(words) if words else (),
            ),
            start_time=start_samples / self.SAMPLE_RATE,
        )
