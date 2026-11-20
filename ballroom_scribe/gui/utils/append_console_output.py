"""Appends streamed console text to a `QTextDocument`, honoring `\\r` line rewrites."""

import re

from PySide6 import QtGui

_LINE_BREAK_RE = re.compile(r"(\r|\n)")


def _resolve_console_chunk(text: str, overwrite_line: bool) -> tuple[str, bool, bool]:
    r"""Resolves `\r` and `\n` within `text` into a single edit at the end of a document.

    A standalone `\r` defers overwriting the current line until printable characters arrive before a `\n`, so a
    progress line ending in `\r` remains visible between updates and a `\r\n` pair (even split across chunks) acts as
    a plain line break.

    Args:
        text: The newly arrived console chunk.
        overwrite_line: Whether the previous chunk left a `\r` pending on the document's last block.

    Returns:
        `(rendered_text, replace_current_line, next_overwrite_line)`, where `rendered_text` uses `\n` line breaks,
        `replace_current_line` indicates whether `rendered_text` replaces (rather than appends to) the document's
        current last block, and `next_overwrite_line` is the updated pending-`\r` state after `text`.
    """
    lines = [""]
    replace_current_line = False
    pending_cr = overwrite_line

    for part in _LINE_BREAK_RE.split(text):
        if not part:
            continue
        if part == "\r":
            pending_cr = True
        elif part == "\n":
            pending_cr = False
            lines.append("")
        elif pending_cr:
            if len(lines) == 1:
                replace_current_line = True
            lines[-1] = part
            pending_cr = False
        else:
            lines[-1] += part

    return "\n".join(lines), replace_current_line, pending_cr


def append_console_output(
    document: QtGui.QTextDocument,
    text: str,
    *,
    overwrite_line: bool = False,
) -> bool:
    r"""Appends streamed console `text` to `document`, rewriting the last block after standalone `\r`.

    Args:
        document: The target text document holding the console log.
        text: The newly arrived console output chunk, which may contain `\r`, `\n`, or `\r\n` sequences split at
            arbitrary boundaries.
        overwrite_line: Whether the previous call left a standalone `\r` pending on `document`'s last block.

    Returns:
        Whether a standalone `\r` is pending on `document`'s last block after `text` has been applied; pass this back
        as `overwrite_line` on the next call.
    """
    rendered, replace_current_line, next_overwrite_line = _resolve_console_chunk(text, overwrite_line)
    if not replace_current_line and not rendered:
        return next_overwrite_line

    cursor = QtGui.QTextCursor(document)
    cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
    if replace_current_line:
        cursor.movePosition(
            QtGui.QTextCursor.MoveOperation.StartOfBlock,
            QtGui.QTextCursor.MoveMode.KeepAnchor,
        )
    cursor.insertText(rendered)
    return next_overwrite_line
