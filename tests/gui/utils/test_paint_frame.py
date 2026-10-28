import unittest

from parameterized import param, parameterized

try:
    from PySide6 import QtCore, QtGui
except ImportError as err:
    raise unittest.SkipTest("PySide6 is not installed.") from err

import ballroom_scribe.gui.utils

_BACKGROUND = QtGui.QColor("white")
_FRAME = QtGui.QColor("red")


def _paint(rect: QtCore.QRect, width: int) -> QtGui.QImage:
    image = QtGui.QImage(24, 16, QtGui.QImage.Format.Format_ARGB32)
    image.fill(_BACKGROUND)
    painter = QtGui.QPainter(image)
    try:
        ballroom_scribe.gui.utils.paint_frame(painter, rect, _FRAME, width)
    finally:
        painter.end()
    return image


def _framed_pixels(image: QtGui.QImage) -> set[tuple[int, int]]:
    return {(x, y) for y in range(image.height()) for x in range(image.width()) if image.pixelColor(x, y) == _FRAME}


def _expected_frame(rect: QtCore.QRect, width: int) -> set[tuple[int, int]]:
    outer = {(x, y) for y in range(rect.top(), rect.bottom() + 1) for x in range(rect.left(), rect.right() + 1)}
    inner = rect.adjusted(width, width, -width, -width)
    hole = {(x, y) for y in range(inner.top(), inner.bottom() + 1) for x in range(inner.left(), inner.right() + 1)}
    return outer - hole


class TestPaintFrame(unittest.TestCase):
    """Unit tests for `paint_frame`."""

    @parameterized.expand(
        [
            param("two_px_at_origin", rect=QtCore.QRect(0, 0, 24, 16), width=2),
            param("one_px_offset", rect=QtCore.QRect(3, 2, 10, 7), width=1),
            param("three_px_offset", rect=QtCore.QRect(5, 1, 12, 12), width=3),
        ]
    )
    def test_covers_exactly_the_outer_pixels_of_the_rect(self, _, rect: QtCore.QRect, width: int):
        image = _paint(rect, width)

        self.assertEqual(_framed_pixels(image), _expected_frame(rect, width))

    def test_restores_painter_state(self):
        image = QtGui.QImage(8, 8, QtGui.QImage.Format.Format_ARGB32)
        painter = QtGui.QPainter(image)
        try:
            painter.setPen(QtGui.QPen(QtGui.QColor("blue"), 5))
            painter.setBrush(QtGui.QColor("green"))
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

            ballroom_scribe.gui.utils.paint_frame(painter, QtCore.QRect(0, 0, 8, 8), _FRAME)

            self.assertEqual(painter.pen().color(), QtGui.QColor("blue"))
            self.assertEqual(painter.pen().width(), 5)
            self.assertEqual(painter.brush().color(), QtGui.QColor("green"))
            self.assertTrue(painter.renderHints() & QtGui.QPainter.RenderHint.Antialiasing)
        finally:
            painter.end()


if __name__ == "__main__":
    unittest.main()
