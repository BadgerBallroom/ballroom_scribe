"""Decides whether closed captions fit below the video picture."""

from PySide6 import QtCore


def layout_captions(
    area: QtCore.QSize,
    video_size: QtCore.QSize,
    strip_height: int,
) -> tuple[QtCore.QRect, QtCore.QRect | None]:
    """Returns the rectangle of the video widget, and that of the caption strip if it fits below the picture.

    There is room below the picture when the picture, at the size it would have without captions, and the strip both
    fit in the height of `area`. The picture never shrinks to make room. It stays vertically centered when the bar
    below it is tall enough for the strip, and otherwise moves up just far enough.

    Args:
        area: The size of the pane that holds the video and its captions.
        video_size: The size of the video's frames, which gives the picture's aspect ratio.
        strip_height: The height of the caption strip, or `0` if no strip is wanted.

    Returns:
        The video widget's rectangle and the strip's rectangle, both within `area`. When the strip does not fit, is
        not wanted, or `area` or `video_size` is empty, the video widget gets all of `area` and the strip is `None`.
    """
    whole_area = QtCore.QRect(QtCore.QPoint(0, 0), area)
    if strip_height <= 0 or area.isEmpty() or video_size.isEmpty():
        return whole_area, None
    picture_height = round(area.width() * video_size.height() / video_size.width())
    if picture_height + strip_height > area.height():
        return whole_area, None
    top = min((area.height() - picture_height) // 2, area.height() - picture_height - strip_height)
    return (
        QtCore.QRect(0, top, area.width(), picture_height),
        QtCore.QRect(0, top + picture_height, area.width(), strip_height),
    )
