"""Utility that splits streamed console text into the lines that it completes."""

import re

# `\r\n` is one line ending. A `\r` alone ends a line too: programs write it to rewrite a progress line.
_LINE_ENDING_RE = re.compile(r"\r\n|\r|\n")


def split_console_lines(pending: str, text: str) -> tuple[list[str], str]:
    r"""Splits what a child process has written into the lines that are complete and an unfinished rest.

    Console output arrives in chunks that end anywhere, so the last line of a chunk is usually unfinished. Passing
    the rest of one call to the next call puts the lines back together, whatever the chunk boundaries are. Blank
    lines are left out, which also makes a `\r\n` that is split between two chunks count once.

    Args:
        pending: The unfinished rest that the previous call returned, or `""` for the first chunk.
        text: The newly arrived console output, which may contain `\r`, `\n`, or `\r\n` line endings.

    Returns:
        `(lines, pending)`, where `lines` are the lines that are complete now, without their line endings, and
        `pending` is the unfinished rest to pass to the next call. Whoever reads the last chunk treats what is still
        pending as the last line.
    """
    *complete_lines, rest = _LINE_ENDING_RE.split(pending + text)
    return [line for line in complete_lines if line.strip()], rest
