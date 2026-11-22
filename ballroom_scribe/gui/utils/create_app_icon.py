"""Renders the multi-resolution application icon for the Ballroom Scribe timestamp editor.

To export the icon as a multi-resolution `.ico` file (for example, to assign to a Windows `.lnk` shortcut), run from
the repository root:

    python -m ballroom_scribe.gui.utils.create_app_icon C:\\path\\to\\edit_timestamps.ico
"""

import argparse
import pathlib
import struct
from collections.abc import Sequence

from PySide6 import QtCore, QtGui

_CANVAS_SIZE = 128.0
_ICON_SIZES = (16, 20, 24, 32, 48, 64, 128, 256, 512)
_ICO_MAX_SIZE = 256
_ICO_SIZES = (16, 20, 24, 32, 48, 64, 128, 256)
_ICO_TYPE_ICON = 1
_ICO_COLOR_PLANES = 1
_ICO_BITS_PER_PIXEL = 32
_ICO_HEADER_STRUCT = struct.Struct("<HHH")
_ICO_ENTRY_STRUCT = struct.Struct("<BBBBHHII")

_BADGE_INSET = 4.0
_BADGE_RADIUS = 28.0
_BADGE_BORDER_WIDTH = 3.0
_BADGE_FILL = QtGui.QColor(255, 255, 255)
_BADGE_BORDER = QtGui.QColor(0, 0, 0, 61)

_BAR_COLOR = QtGui.QColor(10, 10, 10)
_BAR_WIDTH = 12.0
_BAR_RADIUS = 6.0
_BAR_MID_Y = 70.0
_OUTER_BAR_HEIGHT = 70.0
_BOWTIE_TAPER = 0.52
_OUTER_BAR_XS = (19.0, 97.0)
_INNER_BAR_XS = (38.0, 78.0)

_PLAYHEAD_COLOR = QtGui.QColor(197, 5, 12)
_TOP_PAD = 17.0
_TRIANGLE_WIDTH = 52.0
_TRIANGLE_HEIGHT_RATIO = 0.54
_TRIANGLE_STROKE_WIDTH = 4.0
_STEM_X = 58.0


def _paint_badge(painter: QtGui.QPainter) -> None:
    """Paints the white rounded-square badge with a subtle neutral border."""
    badge_size = _CANVAS_SIZE - 2.0 * _BADGE_INSET
    badge_rect = QtCore.QRectF(_BADGE_INSET, _BADGE_INSET, badge_size, badge_size)
    painter.setBrush(_BADGE_FILL)
    painter.setPen(QtGui.QPen(_BADGE_BORDER, _BADGE_BORDER_WIDTH))
    painter.drawRoundedRect(badge_rect, _BADGE_RADIUS, _BADGE_RADIUS)


def _paint_bowtie_bars(painter: QtGui.QPainter) -> None:
    """Paints symmetric outer-tall and inner-short waveform bars forming a bowtie silhouette."""
    inner_height = round(_OUTER_BAR_HEIGHT * _BOWTIE_TAPER)
    outer_y = _BAR_MID_Y - _OUTER_BAR_HEIGHT / 2.0
    inner_y = _BAR_MID_Y - inner_height / 2.0

    painter.setBrush(_BAR_COLOR)
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    for x in _OUTER_BAR_XS:
        painter.drawRoundedRect(QtCore.QRectF(x, outer_y, _BAR_WIDTH, _OUTER_BAR_HEIGHT), _BAR_RADIUS, _BAR_RADIUS)
    for x in _INNER_BAR_XS:
        painter.drawRoundedRect(QtCore.QRectF(x, inner_y, _BAR_WIDTH, inner_height), _BAR_RADIUS, _BAR_RADIUS)


def _paint_playhead(painter: QtGui.QPainter) -> None:
    """Paints the crimson vertical playhead stem and enlarged top triangle."""
    center_x = _CANVAS_SIZE / 2.0
    half_width = (_TRIANGLE_WIDTH - _TRIANGLE_STROKE_WIDTH) / 2.0
    triangle_height = round(_TRIANGLE_WIDTH * _TRIANGLE_HEIGHT_RATIO)
    triangle_top_y = _TOP_PAD + 2.0
    triangle_bottom_y = triangle_top_y + triangle_height - 3.0

    triangle = QtGui.QPolygonF(
        [
            QtCore.QPointF(center_x - half_width, triangle_top_y),
            QtCore.QPointF(center_x + half_width, triangle_top_y),
            QtCore.QPointF(center_x, triangle_bottom_y),
        ]
    )
    triangle_pen = QtGui.QPen(_PLAYHEAD_COLOR, _TRIANGLE_STROKE_WIDTH)
    triangle_pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
    painter.setBrush(_PLAYHEAD_COLOR)
    painter.setPen(triangle_pen)
    painter.drawPolygon(triangle)

    stem_top_y = triangle_top_y + 8.0
    stem_bottom_y = _CANVAS_SIZE - _TOP_PAD
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.drawRoundedRect(
        QtCore.QRectF(_STEM_X, stem_top_y, _BAR_WIDTH, stem_bottom_y - stem_top_y),
        _BAR_RADIUS,
        _BAR_RADIUS,
    )


def _render_icon_image(size: int) -> QtGui.QImage:
    """Renders the application icon onto a square `size` × `size` `QImage`."""
    if size <= 0:
        raise ValueError(f"Icon size must be positive, got {size}.")

    image = QtGui.QImage(size, size, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QtCore.Qt.GlobalColor.transparent)

    scale = size / _CANVAS_SIZE
    painter = QtGui.QPainter(image)
    try:
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.scale(scale, scale)
        _paint_badge(painter)
        _paint_bowtie_bars(painter)
        _paint_playhead(painter)
    finally:
        painter.end()
    return image


def create_app_icon(sizes: Sequence[int] = _ICON_SIZES) -> QtGui.QIcon:
    """Builds a multi-resolution `QIcon` depicting a bowtie-shaped music valley and crimson playhead.

    Args:
        sizes: Pixel edge lengths to pre-render into the returned `QIcon` so both small Windows title-bar icons
            (`16×16`–`32×32`) and large macOS Dock icons (`128×128`–`512×512`) rasterize crisply without resampling.

    Returns:
        A `QIcon` populated with anti-aliased pixmaps at each requested size.

    Raises:
        ValueError: If `sizes` is empty or contains a non-positive dimension.
    """
    if not sizes:
        raise ValueError("At least one icon size must be provided.")

    icon = QtGui.QIcon()
    for size in sizes:
        icon.addPixmap(QtGui.QPixmap.fromImage(_render_icon_image(size)))
    return icon


def _encode_png(image: QtGui.QImage) -> bytes:
    """Encodes `image` as lossless PNG bytes in memory."""
    byte_array = QtCore.QByteArray()
    buffer = QtCore.QBuffer(byte_array)
    buffer.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
    try:
        if not image.save(buffer, "PNG"):
            raise RuntimeError(f"Failed to encode {image.width()}x{image.height()} icon frame as PNG.")
    finally:
        buffer.close()
    return bytes(byte_array)


def _encode_ico(sizes: Sequence[int] = _ICO_SIZES) -> bytes:
    """Renders and packs a multi-resolution Windows `.ico` binary containing PNG-compressed frames.

    Args:
        sizes: Square pixel edge lengths (`1`–`256`) to include in the `.ico` directory.

    Returns:
        Complete `.ico` file contents as `bytes`.

    Raises:
        ValueError: If `sizes` is empty or contains a dimension outside `1`–`256`.
    """
    if not sizes:
        raise ValueError("At least one icon size must be provided.")
    for size in sizes:
        if size <= 0 or size > _ICO_MAX_SIZE:
            raise ValueError(f"ICO frame size must be between 1 and {_ICO_MAX_SIZE}, got {size}.")

    frames = [(size, _encode_png(_render_icon_image(size))) for size in sizes]
    header = _ICO_HEADER_STRUCT.pack(0, _ICO_TYPE_ICON, len(frames))
    offset = _ICO_HEADER_STRUCT.size + _ICO_ENTRY_STRUCT.size * len(frames)

    entries: list[bytes] = []
    for size, png_bytes in frames:
        # In the Windows ICO directory entry format, a 1-byte dimension of `0` represents `256` pixels.
        dim_byte = 0 if size == _ICO_MAX_SIZE else size
        entries.append(
            _ICO_ENTRY_STRUCT.pack(
                dim_byte,
                dim_byte,
                0,
                0,
                _ICO_COLOR_PLANES,
                _ICO_BITS_PER_PIXEL,
                len(png_bytes),
                offset,
            )
        )
        offset += len(png_bytes)

    return b"".join((header, *entries, *(png_bytes for _, png_bytes in frames)))


# Run from the repository root to generate a multi-resolution `.ico` file (e.g. for a Windows shortcut):
#     python -m ballroom_scribe.gui.utils.create_app_icon C:\path\to\edit_timestamps.ico
def _write_ico(output_path: pathlib.Path, sizes: Sequence[int] = _ICO_SIZES) -> None:
    """Writes a multi-resolution Windows `.ico` file of the application icon to `output_path`."""
    output_path.write_bytes(_encode_ico(sizes))


def _main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for exporting the multi-resolution `.ico` file."""
    parser = argparse.ArgumentParser(
        prog="python -m ballroom_scribe.gui.utils.create_app_icon",
        description="Exports the Ballroom Scribe editor icon as a multi-resolution Windows .ico file.",
    )
    parser.add_argument(
        "output",
        type=pathlib.Path,
        help="Destination .ico file path.",
    )
    args = parser.parse_args(argv)
    _write_ico(args.output)
    print(f"Wrote multi-resolution icon ({', '.join(f'{s}x{s}' for s in _ICO_SIZES)}) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
