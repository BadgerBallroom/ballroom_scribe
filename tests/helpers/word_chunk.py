"""Utilities for adding fake timing to chunks of words."""

import itertools
from collections.abc import Iterable
from typing import Protocol


class _Word(Protocol):
    @property
    def start(self) -> float: ...

    @property
    def end(self) -> float: ...


class WordMaker[Word: _Word](Protocol):
    def __call__(self, word: str, start: float, end: float) -> Word: ...


class ChunkMaker[Word, Chunk](Protocol):
    def __call__(self, start: float, end: float, words: list[Word] | None) -> Chunk: ...


class Maker[Word: _Word, Chunk]:
    """Adds fake timing to sequences of words."""

    def __init__(self, word_maker: WordMaker[Word], chunk_maker: ChunkMaker[Word, Chunk], start_time=0.0):
        self._word_maker = word_maker
        self._chunk_maker = chunk_maker
        self._time_counter = itertools.pairwise(itertools.count(start_time, 0.5))

    def make(self, words_str: Iterable[str] | None) -> Chunk:
        """Generates a chunk of words with fake timing.

        Each word starts 0.5 seconds after the previous. This continues for the lifetime of this `Maker`.

        Args:
            words_str:
                An `Iterable` of strings, each of which will become a word. If this is `None` or empty, an empty
                chunk that lasts 0.5 seconds will be returned.

        Returns:
            The chunk
        """
        if words_str is None:
            words = None
        else:
            words = [
                self._word_maker(start=start, end=end, word=word)
                for word, (start, end) in zip(words_str, self._time_counter)
            ]

        if words:
            start = words[0].start
            end = words[-1].end
        else:
            start, end = next(self._time_counter)

        return self._chunk_maker(
            start=start,
            end=end,
            words=words,
        )
