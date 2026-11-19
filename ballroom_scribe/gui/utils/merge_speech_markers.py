"""Merges transcribed words into the speech markers shown along the bottom of the waveform."""

import math
from collections.abc import Iterable, Sequence

import ballroom_scribe.gui.utils
import ballroom_scribe.models

# A label is cut off to give the next sentence its own marker only if at least this much of the label stays visible.
_MIN_LABEL_PIXELS = 48.0
# Whisper does not always end an announcement with punctuation, so a pause this long starts a sentence as well.
_SENTENCE_PAUSE_SECONDS = 2.0


def _starts_sentence(previous: ballroom_scribe.models.Word, word: ballroom_scribe.models.Word) -> bool:
    """Returns whether `word` starts a sentence, given the word spoken before it."""
    return (
        ballroom_scribe.gui.utils.ends_sentence(previous.word)
        or word.start - max(previous.start, previous.end) >= _SENTENCE_PAUSE_SECONDS
    )


def _group_words(
    words: Sequence[ballroom_scribe.models.Word],
    word_widths: Sequence[float],
    starts: Sequence[int],
    *,
    pixels_per_sample: float,
    space_width: float,
    label_padding: float,
) -> list[int]:
    """Returns the index of the first word of every marker; `starts` are the words' start samples."""
    firsts: list[int] = []
    index = 0
    while index < len(words):
        firsts.append(index)
        left = starts[index] * pixels_per_sample
        width = word_widths[index]
        index += 1
        while index < len(words):
            room = starts[index] * pixels_per_sample - left - label_padding
            if width <= room:
                break  # The label ends before the next word starts: that word begins its own marker.
            if _starts_sentence(words[index - 1], words[index]) and room >= _MIN_LABEL_PIXELS:
                break  # A sentence gets its own marker; this label is cut off with an ellipsis.
            width += space_width + word_widths[index]
            index += 1
    return firsts


def merge_speech_markers(
    transcript: Iterable[ballroom_scribe.models.TranscriptChunk],
    word_widths: Sequence[float],
    *,
    pixels_per_second: float,
    space_width: float,
    label_padding: float,
) -> tuple[ballroom_scribe.models.TranscriptChunk, ...]:
    """Groups the words of a transcript into markers whose labels do not overlap at the given zoom.

    A marker is drawn at the start time of its first word and labelled with its words. A word merges into the marker
    before it while that marker's label would otherwise overlap it, and so on down the line, so the fewer pixels a
    second gets, the more words share a marker. A word that starts a sentence (it follows a word that
    `ends_sentence`, or a pause of 2 seconds or more) begins a marker of its own even then, and the label before it is
    cut off, provided that at least 48 pixels of that label stay visible. Markers may span several chunks of
    `transcript`.

    The grouping depends on the zoom but not on the scroll position, so markers do not regroup while the waveform
    scrolls.

    Args:
        transcript: The transcript chunks of one audio channel of one video, in chronological order. They share a
            sample rate and a channel.
        word_widths: The width in pixels of the text of every word of `transcript`, in order.
        pixels_per_second: The zoom: how many pixels one second of audio takes up.
        space_width: The width in pixels of the space that joins two words in a label.
        label_padding: The pixels to keep clear between the end of a label and the start of the next marker.

    Returns:
        One transcript chunk per marker, with the sample rate and channel of `transcript`. It starts at its first
        word and holds the marker's words, so `get_text()` is the label. It lasts until the next marker starts, which
        is the room the label has; a label wider than that room minus `label_padding` is meant to be cut off. The last
        marker lasts as long as its label needs. Empty if there are no words or `pixels_per_second` is not positive.

    Raises:
        ValueError: If `word_widths` does not have one width per word.
    """
    chunks = list(transcript)
    words = [word for chunk in chunks for word in chunk.words]
    if len(word_widths) != len(words):
        raise ValueError(f"Expected one width per word ({len(words)}), got {len(word_widths)}.")
    if not words or pixels_per_second <= 0:
        return ()

    sample_rate = chunks[0].sample_rate
    channel = chunks[0].channel
    starts = [round(word.start * sample_rate) for word in words]
    firsts = _group_words(
        words,
        word_widths,
        starts,
        pixels_per_sample=pixels_per_second / sample_rate,
        space_width=space_width,
        label_padding=label_padding,
    )

    markers: list[ballroom_scribe.models.TranscriptChunk] = []
    for first, end in zip(firsts, [*firsts[1:], len(words)], strict=True):
        if end < len(words):
            duration_samples = starts[end] - starts[first]
        else:
            label_width = sum(word_widths[first:end]) + space_width * (end - first - 1)
            duration_samples = math.ceil((label_width + label_padding) * sample_rate / pixels_per_second)
        markers.append(
            ballroom_scribe.models.TranscriptChunk(
                start_samples=starts[first],
                sample_rate=sample_rate,
                duration_samples=duration_samples,
                channel=channel,
                words=tuple(words[first:end]),
            )
        )
    return tuple(markers)
