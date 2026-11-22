import importlib
import io
import pathlib
import struct
import sys
import unittest
from unittest.mock import patch

from parameterized import param, parameterized

try:
    from PySide6 import QtCore, QtGui
except ImportError as err:
    raise unittest.SkipTest("PySide6 is not installed.") from err

import ballroom_scribe.gui.utils

_CREATE_APP_ICON_MODULE = importlib.import_module("ballroom_scribe.gui.utils.create_app_icon")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
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


class TestEncodeIco(unittest.TestCase):
    """Unit tests for `_encode_ico`."""

    def test_encodes_default_windows_multi_resolution_sizes(self):
        ico_bytes = _CREATE_APP_ICON_MODULE._encode_ico()

        expected_sizes = (16, 20, 24, 32, 48, 64, 128, 256)
        reserved, ico_type, count = struct.unpack_from("<HHH", ico_bytes, 0)
        self.assertEqual((reserved, ico_type, count), (0, 1, len(expected_sizes)))

        expected_offset = 6 + 16 * len(expected_sizes)
        for index, size in enumerate(expected_sizes):
            width_b, height_b, colors, entry_reserved, planes, bpp, byte_len, offset = struct.unpack_from(
                "<BBBBHHII",
                ico_bytes,
                6 + 16 * index,
            )
            expected_dim = 0 if size == 256 else size
            self.assertEqual(
                (width_b, height_b, colors, entry_reserved, planes, bpp), (expected_dim, expected_dim, 0, 0, 1, 32)
            )
            self.assertEqual(offset, expected_offset)
            self.assertGreater(byte_len, len(_PNG_SIGNATURE))
            self.assertEqual(ico_bytes[offset : offset + len(_PNG_SIGNATURE)], _PNG_SIGNATURE)
            expected_offset += byte_len
        self.assertEqual(expected_offset, len(ico_bytes))

        byte_array = QtCore.QByteArray(ico_bytes)
        buffer = QtCore.QBuffer(byte_array)
        buffer.open(QtCore.QIODevice.OpenModeFlag.ReadOnly)
        reader = QtGui.QImageReader(buffer, b"ico")
        self.assertEqual(reader.imageCount(), len(expected_sizes))
        decoded_sizes: list[tuple[int, int]] = []
        for index, size in enumerate(expected_sizes):
            reader.jumpToImage(index)
            frame = reader.read()
            self.assertFalse(frame.isNull())
            expected_frame = _CREATE_APP_ICON_MODULE._render_icon_image(size).convertToFormat(
                QtGui.QImage.Format.Format_ARGB32
            )
            self.assertEqual(frame.convertToFormat(QtGui.QImage.Format.Format_ARGB32), expected_frame)
            decoded_sizes.append((frame.width(), frame.height()))
        self.assertEqual(decoded_sizes, [(size, size) for size in expected_sizes])

    @parameterized.expand(
        [
            param("empty_sizes", sizes=()),
            param("zero_size", sizes=(0,)),
            param("negative_size", sizes=(16, -32)),
            param("exceeds_ico_max", sizes=(256, 512)),
        ]
    )
    def test_invalid_sizes_raise_value_error(self, _, sizes: tuple[int, ...]):
        with self.assertRaises(ValueError):
            _CREATE_APP_ICON_MODULE._encode_ico(sizes=sizes)


class TestWriteIco(unittest.TestCase):
    """Unit tests for `_write_ico`."""

    def setUp(self):
        self.mock_write_bytes = self.enterContext(patch.object(pathlib.Path, "write_bytes", autospec=True))
        self.mock_encode_ico = self.enterContext(
            patch.object(_CREATE_APP_ICON_MODULE, "_encode_ico", autospec=True, return_value=b"fake-ico")
        )

    def test_writes_default_sizes_to_output_path(self):
        output_path = pathlib.Path("edit_timestamps.ico")

        _CREATE_APP_ICON_MODULE._write_ico(output_path)

        self.mock_encode_ico.assert_called_once_with((16, 20, 24, 32, 48, 64, 128, 256))
        self.mock_write_bytes.assert_called_once_with(output_path, b"fake-ico")

    def test_writes_custom_sizes_to_output_path(self):
        output_path = pathlib.Path("icons/app.ico")

        _CREATE_APP_ICON_MODULE._write_ico(output_path, sizes=(16, 32, 256))

        self.mock_encode_ico.assert_called_once_with((16, 32, 256))
        self.mock_write_bytes.assert_called_once_with(output_path, b"fake-ico")


class TestMain(unittest.TestCase):
    """Unit tests for `_main`."""

    def setUp(self):
        self.mock_stdout = self.enterContext(patch.object(sys, "stdout", new_callable=io.StringIO))
        self.mock_stderr = self.enterContext(patch.object(sys, "stderr", new_callable=io.StringIO))
        self.mock_write_ico = self.enterContext(patch.object(_CREATE_APP_ICON_MODULE, "_write_ico", autospec=True))

    def test_writes_to_output_path_when_given(self):
        output_path = pathlib.Path("build/shortcut.ico")

        exit_status = _CREATE_APP_ICON_MODULE._main([str(output_path)])

        self.assertEqual(exit_status, 0)
        self.mock_write_ico.assert_called_once_with(output_path)
        self.assertIn(str(output_path), self.mock_stdout.getvalue())

    def test_exits_with_error_when_output_path_is_missing(self):
        with self.assertRaises(SystemExit) as context:
            _CREATE_APP_ICON_MODULE._main([])

        self.assertNotEqual(context.exception.code, 0)
        self.mock_write_ico.assert_not_called()


if __name__ == "__main__":
    unittest.main()
