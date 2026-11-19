"""Video widget that shows closed captions below the picture when there is room, and on the picture otherwise."""

import enum
from typing import override

from PySide6 import QtCore, QtGui, QtMultimedia, QtMultimediaWidgets, QtWidgets

import ballroom_scribe.gui.utils

# Qt draws subtitles on the picture at this fraction of the picture's height; the strip below the picture matches it.
_FONT_HEIGHT_FRACTION = 0.045
_MIN_FONT_PIXELS = 11
_MAX_FONT_PIXELS = 30
_STRIP_LINES = 2
_STRIP_MARGIN = 6
_MAX_TEXT_HEIGHT = 100_000


class CaptionedVideoWidget(QtWidgets.QWidget):
    """A black pane that holds a `QVideoWidget` and the closed caption of the frame it shows.

    `QVideoWidget` embeds a native window, which hides any widget laid over it, so a caption cannot simply be drawn on
    top of the picture. Instead, the caption goes into a strip directly below the picture when the pane has room for
    it there (see `layout_captions`), and otherwise it is handed to `QVideoSink.setSubtitleText`, which makes Qt draw
    it on the picture. While captions are visible, the strip keeps its place between captions, so the picture does
    not move as captions come and go.

    Qt draws the on-picture text with the frames that it renders. A caller that changes the caption while the player
    is paused must therefore have the current frame rendered again; `overlay_changed` says when.
    """

    class Placement(enum.Enum):
        """Where the caption is shown."""

        HIDDEN = enum.auto()
        BELOW = enum.auto()
        OVERLAY = enum.auto()

    overlay_changed = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        palette = self.palette()
        palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("black"))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        self._video_widget = QtMultimediaWidgets.QVideoWidget(self)

        self._strip = QtWidgets.QLabel(self)
        self._strip.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self._strip.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._strip.setWordWrap(True)
        self._strip.setContentsMargins(_STRIP_MARGIN, _STRIP_MARGIN, _STRIP_MARGIN, _STRIP_MARGIN)
        strip_palette = self._strip.palette()
        strip_palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("black"))
        strip_palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("white"))
        self._strip.setPalette(strip_palette)
        self._strip.setAutoFillBackground(True)
        self._strip.hide()

        self._captions_visible = False
        self._caption = ""
        self._overlay_text = ""
        self._placement = CaptionedVideoWidget.Placement.HIDDEN
        self._strip_font_pixels = _MIN_FONT_PIXELS

        # The sink emits this signal on the decoder's thread; a bound slot of this widget makes Qt queue the call.
        self.video_sink().videoSizeChanged.connect(self._relayout)

    @property
    def placement(self) -> CaptionedVideoWidget.Placement:
        """Where the caption is shown: below the picture, on the picture, or not at all."""
        return self._placement

    def video_sink(self) -> QtMultimedia.QVideoSink:
        """Returns the `QVideoSink` to which a media player delivers the frames to show."""
        return self._video_widget.videoSink()

    def set_captions_visible(self, visible: bool) -> None:
        """Shows captions, reserving the strip below the picture if it fits, or hides them."""
        if visible != self._captions_visible:
            self._captions_visible = visible
            self._relayout()

    def set_caption(self, text: str) -> None:
        """Sets the caption of the current frame; an empty string means that nothing is being said."""
        if text != self._caption:
            self._caption = text
            self._show_caption()

    @override
    def sizeHint(self) -> QtCore.QSize:
        return self._video_widget.sizeHint()

    @override
    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._relayout()

    @QtCore.Slot()
    def _relayout(self) -> None:
        """Places the video widget and the strip, and moves the caption to where it now belongs."""
        strip_height = 0
        if self._captions_visible:
            self._strip_font_pixels = self._font_pixels_for_picture()
            strip_height = _STRIP_LINES * QtGui.QFontMetrics(self._strip_font()).lineSpacing() + 2 * _STRIP_MARGIN
        video_rect, strip_rect = ballroom_scribe.gui.utils.layout_captions(
            self.size(),
            self.video_sink().videoSize(),
            strip_height,
        )
        self._video_widget.setGeometry(video_rect)
        if strip_rect is None:
            self._strip.hide()
        else:
            self._strip.setGeometry(strip_rect)
            self._strip.show()

        if not self._captions_visible:
            self._placement = CaptionedVideoWidget.Placement.HIDDEN
        elif strip_rect is None:
            self._placement = CaptionedVideoWidget.Placement.OVERLAY
        else:
            self._placement = CaptionedVideoWidget.Placement.BELOW
        self._show_caption()

    def _font_pixels_for_picture(self) -> int:
        """Returns the caption font's height for the picture as it is shown when it takes the full width."""
        video_size = self.video_sink().videoSize()
        if video_size.isEmpty():
            return _MIN_FONT_PIXELS
        picture_height = self.width() * video_size.height() / video_size.width()
        return min(max(round(_FONT_HEIGHT_FRACTION * picture_height), _MIN_FONT_PIXELS), _MAX_FONT_PIXELS)

    def _strip_font(self, pixels: int | None = None) -> QtGui.QFont:
        font = QtGui.QFont(self.font())
        font.setPixelSize(self._strip_font_pixels if pixels is None else pixels)
        return font

    def _show_caption(self) -> None:
        """Puts the caption in the strip or on the picture, according to the placement."""
        if self._placement == CaptionedVideoWidget.Placement.BELOW:
            self._strip.setFont(self._fitting_strip_font())
            self._strip.setText(self._caption)
        else:
            self._strip.clear()

        overlay_text = self._caption if self._placement == CaptionedVideoWidget.Placement.OVERLAY else ""
        if overlay_text != self._overlay_text:
            self._overlay_text = overlay_text
            self.video_sink().setSubtitleText(overlay_text)
            self.overlay_changed.emit()

    def _fitting_strip_font(self) -> QtGui.QFont:
        """Returns the strip's font, made smaller if the caption would otherwise not fit in the strip."""
        available = self._strip.contentsRect()
        flags = QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.TextFlag.TextWordWrap
        pixels = self._strip_font_pixels
        while pixels > _MIN_FONT_PIXELS:
            bounds = QtGui.QFontMetrics(self._strip_font(pixels)).boundingRect(
                QtCore.QRect(0, 0, available.width(), _MAX_TEXT_HEIGHT),
                flags,
                self._caption,
            )
            if bounds.height() <= available.height():
                break
            pixels -= 1
        return self._strip_font(pixels)
