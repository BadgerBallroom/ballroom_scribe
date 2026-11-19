"""Finds the closed caption to show at a playback position."""

import bisect
from collections.abc import Sequence

import ballroom_scribe.models


def _start_seconds(caption: ballroom_scribe.models.TranscriptChunk) -> float:
    return caption.start_seconds


def find_caption(
    captions: Sequence[ballroom_scribe.models.TranscriptChunk],
    seconds: float,
) -> ballroom_scribe.models.TranscriptChunk | None:
    """Returns the caption whose time span contains `seconds`, or `None` between captions.

    Args:
        captions: The captions in chronological order, as returned by `group_words_into_captions`.
        seconds: The playback position, in seconds since the beginning of the video.

    Returns:
        The caption that starts at or before `seconds` and ends after it. A caption's start is inclusive and its end
        is exclusive.
    """
    index = bisect.bisect_right(captions, seconds, key=_start_seconds) - 1
    if index < 0:
        return None
    caption = captions[index]
    return caption if seconds < caption.end_seconds else None
