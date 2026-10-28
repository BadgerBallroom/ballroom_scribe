"""Utility for parsing formatted or numeric duration strings into seconds."""

import re

_UINT_RE = re.compile(r"^\d+$")
_UFLOAT_RE = re.compile(r"^\d+(?:\.\d+)?$")


def _parse_non_negative_int(part: str, text: str) -> int:
    if not _UINT_RE.match(part):
        raise ValueError(f"Invalid duration {text!r}.")
    return int(part)


def _parse_non_negative_float(part: str, text: str) -> float:
    if not _UFLOAT_RE.match(part):
        raise ValueError(f"Invalid duration {text!r}.")
    return float(part)


def parse_duration(text: str) -> float:
    """Parses a non-negative duration string into seconds.

    Accepts plain seconds (`SS` or `SS.cc`), `MM:SS[.cc]`, or `H:MM:SS[.cc]` (with arbitrary-digit hours, matching
    `format_duration`). Whitespace around `text` is ignored.

    Args:
        text: The duration text to parse.

    Returns:
        The parsed duration in seconds.

    Raises:
        ValueError: If `text` is empty, negative, or does not match a supported duration format.
    """
    stripped = text.strip()
    if not stripped:
        raise ValueError("Duration cannot be empty.")

    parts = stripped.split(":")
    if len(parts) == 1:
        return _parse_non_negative_float(parts[0], text)

    if len(parts) == 2:
        minutes = _parse_non_negative_int(parts[0], text)
        seconds = _parse_non_negative_float(parts[1], text)
        if minutes >= 60 or seconds >= 60.0:
            raise ValueError(f"Minutes and seconds must be less than 60 in {text!r}.")
        return minutes * 60 + seconds

    if len(parts) == 3:
        hours = _parse_non_negative_int(parts[0], text)
        minutes = _parse_non_negative_int(parts[1], text)
        seconds = _parse_non_negative_float(parts[2], text)
        if minutes >= 60 or seconds >= 60.0:
            raise ValueError(f"Minutes and seconds must be less than 60 in {text!r}.")
        total_whole_seconds = hours * 3600 + minutes * 60
        return total_whole_seconds + seconds if total_whole_seconds else seconds

    raise ValueError(f"Invalid duration {text!r}.")
