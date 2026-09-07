def format_duration(seconds: float) -> str:
    """Formats a duration in seconds as `HH:MM:SS.cc`.

    Args:
        seconds: The duration in seconds.

    Returns:
        The formatted duration string.
    """
    centiseconds = round(seconds * 100)
    seconds_int, centiseconds = divmod(centiseconds, 100)
    minutes, seconds_int = divmod(seconds_int, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{seconds_int:02d}.{centiseconds:02d}"
