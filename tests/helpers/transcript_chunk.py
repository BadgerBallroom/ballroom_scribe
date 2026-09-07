"""Utilities for generating `TranscriptChunk`s for testing purposes."""

import ballroom_scribe.models
import tests.helpers.word_chunk


class Maker(tests.helpers.word_chunk.Maker[ballroom_scribe.models.Word, ballroom_scribe.models.TranscriptChunk]):
    """Generates `TranscriptChunk`s for testing purposes."""

    def __init__(self, start_time=0.0):
        super().__init__(
            word_maker=lambda word, start, end: ballroom_scribe.models.Word(
                word=word,
                start=start,
                end=end,
            ),
            chunk_maker=lambda start, end, words: ballroom_scribe.models.TranscriptChunk(
                words=tuple(words) if words else (),
                start_time=start,
                end_time=end,
            ),
            start_time=start_time,
        )
