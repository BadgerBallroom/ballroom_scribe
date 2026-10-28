import unittest
from unittest.mock import patch

from parameterized import param, parameterized

try:
    from PySide6 import QtGui
except ImportError as err:
    raise unittest.SkipTest("PySide6 is not installed.") from err

import ballroom_scribe.gui.utils

_WHITE = QtGui.QColor(255, 255, 255)
_BAR_BLACK = QtGui.QColor(10, 10, 10)
_CRIMSON = QtGui.QColor(197, 5, 12)


def _premultiplied_channels(color: QtGui.QColor) -> tuple[int, int, int, int]:
    """Returns 8-bit `(r, g, b, a)` channels in premultiplied alpha space."""
    red, green, blue, alpha = color.getRgb()
    return (round(red * alpha / 255), round(green * alpha / 255), round(blue * alpha / 255), alpha)


def _dark_bar_height(image: QtGui.QImage, x: int) -> int:
    """Counts vertically contiguous dark pixels in column `x`."""
    return sum(1 for y in range(image.height()) if image.pixelColor(x, y) == _BAR_BLACK)


class TestCreateAppIcon(unittest.TestCase):
    """Unit tests for `create_app_icon`."""

    def setUp(self):
        self.mock_icon_cls = self.enterContext(patch.object(QtGui, "QIcon", autospec=True))
        self.mock_icon = self.mock_icon_cls.return_value
        self.mock_from_image = self.enterContext(patch.object(QtGui.QPixmap, "fromImage", autospec=True))

    def _render_single_size(self, size: int) -> QtGui.QImage:
        icon = ballroom_scribe.gui.utils.create_app_icon(sizes=(size,))
        self.assertIs(icon, self.mock_icon)
        self.mock_from_image.assert_called_once()
        return self.mock_from_image.call_args.args[0]

    def test_populates_default_multi_resolution_sizes(self):
        icon = ballroom_scribe.gui.utils.create_app_icon()

        self.assertIs(icon, self.mock_icon)
        rendered_sizes = [(call.args[0].width(), call.args[0].height()) for call in self.mock_from_image.call_args_list]
        expected_sizes = [(s, s) for s in (16, 20, 24, 32, 48, 64, 128, 256, 512)]
        self.assertEqual(rendered_sizes, expected_sizes)
        self.assertEqual(self.mock_icon.addPixmap.call_count, len(expected_sizes))

    @parameterized.expand(
        [
            param("title_bar_16", size=16),
            param("taskbar_32", size=32),
            param("dock_128", size=128),
        ]
    )
    def test_renders_transparent_corners_and_horizontal_symmetry(self, _, size: int):
        image = self._render_single_size(size)

        self.assertEqual(image.width(), size)
        self.assertEqual(image.height(), size)
        for corner_x in (0, size - 1):
            for corner_y in (0, size - 1):
                self.assertEqual(image.pixelColor(corner_x, corner_y).alpha(), 0)

        for y in range(size):
            for x in range(size // 2):
                left = _premultiplied_channels(image.pixelColor(x, y))
                right = _premultiplied_channels(image.pixelColor(size - 1 - x, y))
                for left_ch, right_ch in zip(left, right, strict=True):
                    self.assertAlmostEqual(left_ch, right_ch, delta=20)

    def test_paints_white_badge_gray_border_bowtie_bars_and_crimson_playhead(self):
        image = self._render_single_size(128)

        # White interior and subtle neutral gray border along the top edge.
        self.assertEqual(image.pixelColor(54, 70), _WHITE)
        top_border = image.pixelColor(64, 4)
        self.assertEqual(top_border.alpha(), 255)
        self.assertEqual(top_border.red(), top_border.green())
        self.assertEqual(top_border.green(), top_border.blue())
        self.assertLess(top_border.red(), 220)
        self.assertGreater(top_border.red(), 170)

        # Outer bars are taller than inner bars (~52% bowtie taper).
        outer_left = _dark_bar_height(image, 25)
        inner_left = _dark_bar_height(image, 44)
        inner_right = _dark_bar_height(image, 84)
        outer_right = _dark_bar_height(image, 103)
        self.assertEqual(outer_left, outer_right)
        self.assertEqual(inner_left, inner_right)
        self.assertGreater(outer_left, inner_left)
        self.assertAlmostEqual(inner_left / outer_left, 0.52, delta=0.05)

        # Crimson playhead stem and enlarged top triangle wings.
        self.assertEqual(image.pixelColor(64, 70), _CRIMSON)
        self.assertEqual(image.pixelColor(64, 104), _CRIMSON)
        self.assertEqual(image.pixelColor(46, 22), _CRIMSON)
        self.assertEqual(image.pixelColor(82, 22), _CRIMSON)

    def test_enlarged_triangle_remains_visibly_wider_than_stem_at_16x16(self):
        image = self._render_single_size(16)

        def is_crimson_tinted(x: int, y: int) -> bool:
            pixel = image.pixelColor(x, y)
            return pixel.red() > 150 and pixel.red() - pixel.green() > 60

        # Row y=2 sits across the upper triangle wings, wider than the center stem at y=10.
        top_crimson_cols = [x for x in range(16) if is_crimson_tinted(x, 2)]
        stem_crimson_cols = [x for x in range(16) if is_crimson_tinted(x, 10)]
        self.assertGreater(len(top_crimson_cols), len(stem_crimson_cols))
        self.assertGreaterEqual(len(top_crimson_cols), 4)

    @parameterized.expand(
        [
            param("empty_sizes", sizes=()),
            param("zero_size", sizes=(0,)),
            param("negative_size", sizes=(16, -32)),
        ]
    )
    def test_invalid_sizes_raise_value_error(self, _, sizes: tuple[int, ...]):
        with self.assertRaises(ValueError):
            ballroom_scribe.gui.utils.create_app_icon(sizes=sizes)


if __name__ == "__main__":
    unittest.main()
