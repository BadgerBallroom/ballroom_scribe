"""Utilities for generating Faster Whisper `Segment`s for testing purposes."""

import itertools

import faster_whisper.transcribe

import tests.helpers.word_chunk


class Maker(tests.helpers.word_chunk.Maker[faster_whisper.transcribe.Word, faster_whisper.transcribe.Segment]):
    """Generates Faster Whisper `Segment`s for testing purposes."""

    def __init__(self, start_time=0.0):
        segment_id_counter = itertools.count()
        super().__init__(
            word_maker=lambda word, start, end: faster_whisper.transcribe.Word(
                start=start,
                end=end,
                word=word,
                probability=start,
            ),
            chunk_maker=lambda start, end, words: faster_whisper.transcribe.Segment(
                id=next(segment_id_counter),
                seek=0,
                start=start,
                end=end,
                text="",
                tokens=[],
                avg_logprob=0.0,
                compression_ratio=0.0,
                no_speech_prob=0.0,
                words=words,
                temperature=0.0,
            ),
            start_time=start_time,
        )
