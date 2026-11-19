"""Tells an out-of-date position report of a media player from one that playback can have reached."""

# A report is checked only this long after playback started or the playhead was moved; later reports are believed.
_SETTLING_SECONDS = 1.0
# Players report a position in steps, e.g. one video frame at a time, so a report may lead real time by this much.
_LEAD_SECONDS = 0.5


def is_stale_position(start_seconds: float, seconds: float, elapsed_seconds: float) -> bool:
    """Returns whether a position that a media player reports while playing is out of date and to be ignored.

    Qt Multimedia's FFmpeg backend on Windows has been seen to report, for 0.2 s after playback starts, the target of
    a seek that a later seek had superseded moments before, while it played from the right place. Playback moves the
    playhead forward only, and no faster than real time. A report from shortly after playback started is therefore
    out of date when it lies behind the position where playback started, or further ahead of it than the time that
    has passed allows.

    Args:
        start_seconds: Where playback started, or where the playhead was last moved to during playback.
        seconds: The reported position.
        elapsed_seconds: How long ago playback started or the playhead was moved.

    Returns:
        Whether to ignore the report. A report that comes more than a second after the start is never ignored, so
        that a player that keeps reporting an unexpected position is believed in the end.
    """
    if elapsed_seconds > _SETTLING_SECONDS:
        return False
    return not start_seconds <= seconds <= start_seconds + elapsed_seconds + _LEAD_SECONDS
