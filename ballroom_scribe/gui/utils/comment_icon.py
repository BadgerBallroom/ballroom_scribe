"""Vector speech-bubble icons marking whether an event has a comment."""

import functools
from typing import override

from PySide6 import QtCore, QtGui

_PRESENT_NAME = "comment-present"
_EMPTY_NAME = "comment-empty"
_EMPTY_ALPHA = 0.45
_DISABLED_ALPHA = 0.3
# `QIcon` takes ownership of its engine; the Python wrappers stay referenced here so Qt can keep calling `paint`.
_ENGINES: list[_CommentIconEngine] = []


def _bubble_path(rect: QtCore.QRectF) -> QtGui.QPainterPath:
    """Returns a rounded speech bubble with a small tail at the bottom left, filling `rect`."""
    inset = rect.width() / 8.0
    body = rect.adjusted(inset, inset, -inset, -inset * 2.4)
    radius = body.height() / 3.0
    path = QtGui.QPainterPath()
    path.addRoundedRect(body, radius, radius)
    tail_x = body.left() + body.width() * 0.2
    tail = QtGui.QPolygonF(
        [
            QtCore.QPointF(tail_x, body.bottom() - radius / 2.0),
            QtCore.QPointF(tail_x, rect.bottom() - inset / 2.0),
            QtCore.QPointF(body.left() + body.width() * 0.5, body.bottom() - radius / 2.0),
        ]
    )
    path.addPolygon(tail)
    path.closeSubpath()
    return path.simplified()


class _CommentIconEngine(QtGui.QIconEngine):
    """Paints a speech bubble: a faint outline for rows without a comment, filled for rows with one.

    Colours follow the application palette, switching to the highlighted-text colour on selected rows.
    """

    def __init__(self, has_comment: bool) -> None:
        super().__init__()
        self._has_comment = has_comment
        _ENGINES.append(self)

    @override
    def paint(
        self,
        painter: QtGui.QPainter,
        rect: QtCore.QRect,
        mode: QtGui.QIcon.Mode,
        state: QtGui.QIcon.State,
    ) -> None:
        del state
        palette = QtGui.QGuiApplication.palette()
        role = (
            QtGui.QPalette.ColorRole.HighlightedText
            if mode == QtGui.QIcon.Mode.Selected
            else QtGui.QPalette.ColorRole.Text
        )
        color = QtGui.QColor(palette.color(role))
        if mode == QtGui.QIcon.Mode.Disabled:
            color.setAlphaF(_DISABLED_ALPHA)
        elif not self._has_comment:
            color.setAlphaF(_EMPTY_ALPHA)
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        pen = QtGui.QPen(color, max(1.0, rect.width() / 12.0))
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(color if self._has_comment else QtCore.Qt.BrushStyle.NoBrush)
        painter.drawPath(_bubble_path(QtCore.QRectF(rect)))
        painter.restore()

    @override
    def clone(self) -> QtGui.QIconEngine:
        return _CommentIconEngine(self._has_comment)

    @override
    def iconName(self) -> str:
        return _PRESENT_NAME if self._has_comment else _EMPTY_NAME


@functools.cache
def comment_icon(has_comment: bool) -> QtGui.QIcon:
    """Returns the cached speech-bubble icon for rows with (`True`) or without (`False`) a comment.

    `QIcon.name()` is `"comment-present"` or `"comment-empty"`, so callers and tests can tell the two apart.
    """
    return QtGui.QIcon(_CommentIconEngine(has_comment))
