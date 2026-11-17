"""Utility for turning arbitrary text, such as a model name, into a safe file name component."""

import re

_UNSAFE_CHARACTERS = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename_component(text: str) -> str:
    """Replaces every run of characters other than letters, digits, `.`, `_`, and `-` with a single `_`.

    Model identifiers contain characters that Windows forbids in file names (`qwen2.5:7b-instruct-q6_K`,
    `Systran/faster-whisper-large-v3`, `C:\\models\\whisper`), so cache files that depend on a model embed this token
    instead. Distinct inputs can collide (`a:b` and `a_b`), which is acceptable for a cache key.

    Args:
        text: The text to make safe, such as a model name or path.

    Returns:
        `text` with every run of unsafe characters replaced by a single `_`.

    Raises:
        ValueError: If `text` contains nothing but unsafe characters.
    """
    sanitized = _UNSAFE_CHARACTERS.sub("_", text)
    if not sanitized.strip("_"):
        raise ValueError(f"{text!r} contains no characters that are safe in a file name.")
    return sanitized
