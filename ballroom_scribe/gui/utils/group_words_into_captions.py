"""Splits transcribed words into the closed captions shown with the video."""

import math
from collections.abc import Iterable

import ballroom_scribe.gui.utils
import ballroom_scribe.models

# Two lines of 42 characters, the customary size of a closed caption.
_MAX_CAPTION_CHARS = 84
_MIN_SENTENCE_CHARS = 28
_MIN_CLAUSE_CHARS = 56
_CLAUSE_ENDINGS = (",", ";", ":")
_MAX_PAUSE_SECONDS = 1.0
_LINGER_SECONDS = 1.0
_MILLISECONDS_PER_SECOND = 1000


def _spoken_until(word: ballroom_scribe.models.Word) -> float:
    """The time at which `word` has been spoken, tolerating a word whose end precedes its start."""
    return max(word.start, word.end)


def _starts_caption(previous: ballroom_scribe.models.Word, word: ballroom_scribe.models.Word, length: int) -> bool:
    """Returns whether `word` starts a new caption; `length` is the number of characters in the caption so far."""
    return (
        word.start - _spoken_until(previous) >= _MAX_PAUSE_SECONDS
        or length + 1 + len(word.word) > _MAX_CAPTION_CHARS
        or (ballroom_scribe.gui.utils.ends_sentence(previous.word) and length >= _MIN_SENTENCE_CHARS)
        or (previous.word.endswith(_CLAUSE_ENDINGS) and length >= _MIN_CLAUSE_CHARS)
    )


def _group_words(
    transcript: Iterable[ballroom_scribe.models.TranscriptChunk],
) -> list[list[ballroom_scribe.models.Word]]:
    """Returns the words of each caption."""
    groups: list[list[ballroom_scribe.models.Word]] = []
    length = 0
    for chunk in transcript:
        for word in chunk.words:
            if groups and not _starts_caption(groups[-1][-1], word, length):
                groups[-1].append(word)
                length += 1 + len(word.word)
            else:
                groups.append([word])
                length = len(word.word)
    return groups


def group_words_into_captions(
    transcript: Iterable[ballroom_scribe.models.TranscriptChunk],
) -> tuple[ballroom_scribe.models.TranscriptChunk, ...]:
    """Splits the words of a transcript into captions of at most two lines.

    A caption ends at a pause of 1 second or more, after a sentence end once it has 28 characters, after a comma,
    semicolon, or colon once it has 56 characters, and before it would exceed 84 characters.

    Args:
        transcript: The transcript chunks of one audio channel of one video, in chronological order. They share a
            sample rate and a channel.

    Returns:
        One transcript chunk per caption, in chronological order, with the sample rate and channel of `transcript`.
        It holds the caption's words, so `get_text()` is the caption. It starts at its first word, rounded down to
        the millisecond, which is the resolution of the media player, so that the caption is showing at the inpoint
        of an event that begins with that word. It lasts until 1 second after its last word, or until the next
        caption starts if that is sooner.
    """
    chunks = list(transcript)
    groups = _group_words(chunks)
    if not groups:
        return ()

    sample_rate = chunks[0].sample_rate
    channel = chunks[0].channel
    starts = [
        math.floor(group[0].start * _MILLISECONDS_PER_SECOND) * sample_rate // _MILLISECONDS_PER_SECOND
        for group in groups
    ]

    captions: list[ballroom_scribe.models.TranscriptChunk] = []
    for index, group in enumerate(groups):
        end = round((max(_spoken_until(word) for word in group) + _LINGER_SECONDS) * sample_rate)
        if index + 1 < len(groups):
            end = min(end, starts[index + 1])
        captions.append(
            ballroom_scribe.models.TranscriptChunk(
                start_samples=starts[index],
                sample_rate=sample_rate,
                duration_samples=max(0, end - starts[index]),
                channel=channel,
                words=tuple(group),
            )
        )
    return tuple(captions)
