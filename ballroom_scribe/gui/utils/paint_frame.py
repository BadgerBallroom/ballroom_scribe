"""Paints a crisp frame just inside a rectangle."""

from PySide6 import QtCore, QtGui


def paint_frame(
    painter: QtGui.QPainter,
    rect: QtCore.QRect | QtCore.QRectF,
    color: QtGui.QColor,
    width: int = 2,
) -> None:
    """Draws a `width` px frame in `color` along the inside of `rect`, with sharp mitred corners.

    The frame covers exactly the outermost `width` pixels of `rect` on every side (no anti-aliasing), so it lines up
    with cell and widget edges. The painter's pen, brush, and render hints are restored afterwards.

    Args:
        painter: Active painter to draw with.
        rect: Rectangle whose inside edge the frame hugs.
        color: Frame colour.
        width: Frame thickness in logical pixels.
    """
    pen = QtGui.QPen(color, width)
    pen.setJoinStyle(QtCore.Qt.PenJoinStyle.MiterJoin)
    inset = width / 2
    painter.save()
    try:
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawRect(QtCore.QRectF(rect).adjusted(inset, inset, -inset, -inset))
    finally:
        painter.restore()
