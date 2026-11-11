"""Custom `QPainter` audio waveform widget with click-to-seek, recentering playhead drift, zoom, and event markers."""

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import override

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

import ballroom_scribe.gui.models
import ballroom_scribe.utils

_MIN_WINDOW_SECONDS = 2.0
_DEFAULT_WINDOW_SECONDS = 720.0
_MAX_WINDOW_FACTOR = 1.1  # The whole clip plus a margin so both ends stay visible.
_ZOOM_STEP = 0.8  # Window factor per zoom-in step; zooming out uses the reciprocal.
_WHEEL_NOTCH_ANGLE_DELTA = 120  # `QWheelEvent.angleDelta()` units per notch of a mouse wheel.
_RECENTER_DURATION_SECONDS = 5.0
_MAX_PLAYBACK_TICK_SECONDS = 0.5
_ASYNC_SNAP_TOLERANCE_SECONDS = 0.05
_CENTER_EPSILON = 1e-6
_RULER_HEIGHT = 22
_TICK_STEPS = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0)


@dataclass(frozen=True, slots=True)
class _Marker:
    row: int
    seconds: float
    label: str


def _choose_tick_step(window_seconds: float, width_px: int) -> float:
    """Picks a human-friendly tick interval aiming for ~90 pixels between ruler labels."""
    target_ticks = max(2.0, width_px / 95.0)
    ideal_step = window_seconds / target_ticks
    for step in _TICK_STEPS:
        if step >= ideal_step:
            return step
    return _TICK_STEPS[-1]


def _format_span(seconds: float) -> str:
    """Formats a visible span compactly: `2.5 s`, `30 s`, `1:30`, or `1:12:00`."""
    if seconds < 60.0:
        return f"{round(seconds, 1):g} s"
    minutes, whole_seconds = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{whole_seconds:02d}"
    return f"{minutes}:{whole_seconds:02d}"


def _paint_focus_frame(painter: QtGui.QPainter, rect: QtCore.QRect) -> None:
    """Outlines `rect` with a 1 px focus frame, explicitly disabling any fill left behind by earlier painting."""
    painter.save()
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    painter.setPen(QtGui.QPen(QtGui.QColor(90, 155, 245), 1))
    painter.drawRect(rect.adjusted(0, 0, -1, -1))
    painter.restore()


class WaveformWidget(QtWidgets.QWidget):
    """Renders a `WaveformEnvelope` with zoom, click-to-seek, 5 s recentering drift, drag-to-pan, and markers."""

    seek_requested = QtCore.Signal(float)
    step_requested = QtCore.Signal(float)
    window_changed = QtCore.Signal(float)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setMinimumHeight(125)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        self._envelope: ballroom_scribe.gui.models.WaveformEnvelope | None = None
        self._peak_amplitude: float = 1.0
        self._position_seconds: float = 0.0
        self._window_seconds: float = _DEFAULT_WINDOW_SECONDS
        self._status_message: str = "Open a video or CSV to view its audio waveform."
        self._is_error: bool = False
        self._is_offline: bool = False
        self._markers: tuple[_Marker, ...] = ()
        self._selected_rows: frozenset[int] = frozenset()

        self._pressing: bool = False
        self._dragging: bool = False
        self._drag_anchor_x: float = 0.0
        self._drag_anchor_seconds: float = 0.0
        self._playhead_offset_fraction: float = 0.0
        self._recenter_remaining_seconds: float = 0.0
        self._is_playing: bool = False
        self._in_explicit_seek: bool = False

    @property
    def envelope(self) -> ballroom_scribe.gui.models.WaveformEnvelope | None:
        """Returns the displayed `WaveformEnvelope`, or `None` while loading or after an error."""
        return self._envelope

    @property
    def position_seconds(self) -> float:
        """Returns the current playhead position in seconds."""
        return self._position_seconds

    @property
    def playhead_offset_fraction(self) -> float:
        """Returns the playhead's horizontal offset from center in `[-0.5, 0.5]`."""
        return self._playhead_offset_fraction

    @property
    def visible_range(self) -> tuple[float, float]:
        """Returns `(start_seconds, end_seconds)` currently shown across the widget width."""
        start_sec = self._position_seconds - (0.5 + self._playhead_offset_fraction) * self._window_seconds
        return start_sec, start_sec + self._window_seconds

    @property
    def status_message(self) -> str:
        """Returns the message shown while no envelope is displayed."""
        return self._status_message

    @property
    def is_error(self) -> bool:
        """Returns whether the status message describes a loading failure."""
        return self._is_error

    @property
    def is_offline(self) -> bool:
        """Returns whether the widget shows that the active video's media is offline (see `set_offline`)."""
        return self._is_offline

    @property
    def window_seconds(self) -> float:
        """Returns the visible span of the waveform in seconds."""
        return self._window_seconds

    def set_window_seconds(self, seconds: float) -> None:
        """Sets the visible span around the playhead, clamped to `[2 s, 1.1 × duration]`.

        Emits `window_changed` when the span actually changes.
        """
        max_window = _MIN_WINDOW_SECONDS
        if self._envelope is not None:
            max_window = max(_MIN_WINDOW_SECONDS, self._envelope.duration_seconds * _MAX_WINDOW_FACTOR)
        clamped = min(max(seconds, _MIN_WINDOW_SECONDS), max_window)
        if clamped != self._window_seconds:
            self._window_seconds = clamped
            self.window_changed.emit(clamped)
            self.update()

    def zoom_in(self) -> None:
        """Narrows the visible span by one zoom step."""
        self.set_window_seconds(self._window_seconds * _ZOOM_STEP)

    def zoom_out(self) -> None:
        """Widens the visible span by one zoom step."""
        self.set_window_seconds(self._window_seconds / _ZOOM_STEP)

    def reset_zoom(self) -> None:
        """Restores the default visible span."""
        self.set_window_seconds(_DEFAULT_WINDOW_SECONDS)

    def clear(self) -> None:
        """Resets the widget to its initial empty state."""
        self._envelope = None
        self._peak_amplitude = 1.0
        self._position_seconds = 0.0
        self._markers = ()
        self._selected_rows = frozenset()
        self._pressing = False
        self._dragging = False
        self._playhead_offset_fraction = 0.0
        self._recenter_remaining_seconds = 0.0
        self._status_message = "Open a video or CSV to view its audio waveform."
        self._is_error = False
        self._is_offline = False
        self.update()

    def set_loading(self, video_name: str) -> None:
        """Displays a loading state while `WaveformLoader` extracts or computes peaks."""
        self.set_status(f"Loading audio waveform for {video_name}…")

    def set_status(self, message: str) -> None:
        """Displays a neutral (non-error) status message instead of an envelope."""
        self._envelope = None
        self._pressing = False
        self._dragging = False
        self._playhead_offset_fraction = 0.0
        self._recenter_remaining_seconds = 0.0
        self._status_message = message
        self._is_error = False
        self._is_offline = False
        self.update()

    def set_offline(self, video_name: str, reason: str) -> None:
        """Shows that the active video's media is offline (missing or unreadable) instead of an envelope."""
        self.set_status(f"{video_name} could not be opened ({reason}). Use File → Locate Video… to find it.")
        self._is_offline = True
        self.update()

    def set_error(self, message: str) -> None:
        """Displays an error message when audio extraction or waveform loading fails."""
        self._envelope = None
        self._pressing = False
        self._dragging = False
        self._playhead_offset_fraction = 0.0
        self._recenter_remaining_seconds = 0.0
        self._status_message = f"Waveform unavailable: {message}"
        self._is_error = True
        self._is_offline = False
        self.update()

    def set_envelope(self, envelope: ballroom_scribe.gui.models.WaveformEnvelope) -> None:
        """Sets the active `WaveformEnvelope` and triggers a repaint."""
        self._envelope = envelope
        if envelope.num_bins > 0:
            max_abs = max(
                float(np.max(np.abs(envelope.mins))),
                float(np.max(np.abs(envelope.maxs))),
            )
            self._peak_amplitude = max(0.05, max_abs)
        else:
            self._peak_amplitude = 1.0
        self._is_error = False
        self._is_offline = False
        self.set_window_seconds(self._window_seconds)  # Re-clamps the span to the new clip's length.
        self.update()

    def set_playing(self, playing: bool) -> None:
        """Updates whether the media player is actively playing (advancing the 5 s recentering drift)."""
        self._is_playing = playing

    def recenter_playhead(self) -> None:
        """Places the playhead back at the horizontal center of the widget."""
        if self._playhead_offset_fraction != 0.0 or self._recenter_remaining_seconds != 0.0:
            self._playhead_offset_fraction = 0.0
            self._recenter_remaining_seconds = 0.0
            self.update()

    def apply_step(self, step_fn: Callable[[float], None], delta_seconds: float) -> None:
        """Runs `step_fn(delta_seconds)`, keeping the visible waveform stationary while paused."""
        old_start_sec, _ = self.visible_range
        self._in_explicit_seek = True
        try:
            step_fn(delta_seconds)
        finally:
            self._in_explicit_seek = False
        if not self._is_playing:
            self._set_offset_keeping_start(old_start_sec)
            self.update()

    def set_position(self, seconds: float) -> None:
        """Updates the playhead position in seconds, advancing recentering drift during live playback."""
        clamped = max(0.0, seconds)
        if clamped == self._position_seconds:
            return
        old_start_sec, _ = self.visible_range
        dt = clamped - self._position_seconds
        self._position_seconds = clamped
        if not self._dragging and not self._in_explicit_seek:
            if self._is_playing:
                if self._recenter_remaining_seconds > 0.0 and 0.0 < dt <= _MAX_PLAYBACK_TICK_SECONDS:
                    self._advance_recenter_drift(dt)
            elif self._playhead_offset_fraction != 0.0 and abs(dt) <= _ASYNC_SNAP_TOLERANCE_SECONDS:
                # Keep the visible window pinned if the media backend snaps a paused seek to a nearby packet.
                self._set_offset_keeping_start(old_start_sec)
        self.update()

    def set_markers(
        self,
        markers: Sequence[tuple[int, float, str]],
        selected_rows: Iterable[int] = (),
    ) -> None:
        """Updates the event markers drawn for the active video."""
        self._markers = tuple(_Marker(row=row, seconds=sec, label=label) for row, sec, label in markers)
        self._selected_rows = frozenset(selected_rows)
        self.update()

    def set_selected_rows(self, selected_rows: Iterable[int]) -> None:
        """Updates which event marker rows are highlighted."""
        updated = frozenset(selected_rows)
        if updated != self._selected_rows:
            self._selected_rows = updated
            self.update()

    @override
    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(800, 140)

    @override
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)

        rect = self.rect()
        width = rect.width()
        height = rect.height()
        painter.fillRect(rect, QtGui.QColor(24, 26, 31))

        if self._envelope is None or width <= 2 or height <= _RULER_HEIGHT + 4:
            self._paint_status_text(painter, rect)
            return

        start_sec, end_sec = self.visible_range
        body_top = _RULER_HEIGHT
        body_height = height - body_top

        self._paint_out_of_bounds(painter, width, body_top, body_height, start_sec, end_sec)
        self._paint_ruler(painter, width, start_sec, end_sec)
        self._paint_waveform(painter, width, body_top, body_height, start_sec, end_sec)
        self._paint_markers(painter, width, body_top, body_height, start_sec, end_sec)
        self._paint_playhead(painter, width, height)

        if self.hasFocus():
            _paint_focus_frame(painter, rect)

    def _paint_status_text(self, painter: QtGui.QPainter, rect: QtCore.QRect) -> None:
        area = rect.adjusted(12, 6, -12, -6)
        if self._is_offline:
            # Like an NLE's "media offline" slate: a bold headline over the detail text.
            headline_font = QtGui.QFont(painter.font())
            headline_font.setBold(True)
            headline_font.setPointSize(headline_font.pointSize() + 4)
            painter.setFont(headline_font)
            painter.setPen(QtGui.QColor(255, 176, 60))
            upper = QtCore.QRect(area.left(), area.top(), area.width(), area.height() // 2 - 2)
            painter.drawText(
                upper, QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignBottom, "MEDIA OFFLINE"
            )
            painter.setFont(self.font())
            painter.setPen(QtGui.QColor(185, 192, 206))
            lower = QtCore.QRect(area.left(), area.top() + area.height() // 2 + 2, area.width(), area.height() // 2 - 2)
            painter.drawText(
                lower,
                QtCore.Qt.AlignmentFlag.AlignHCenter
                | QtCore.Qt.AlignmentFlag.AlignTop
                | QtCore.Qt.TextFlag.TextWordWrap,
                self._status_message,
            )
            return
        color = QtGui.QColor(235, 110, 110) if self._is_error else QtGui.QColor(165, 172, 185)
        painter.setPen(color)
        painter.drawText(
            area,
            QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.TextFlag.TextWordWrap,
            self._status_message,
        )

    def _paint_out_of_bounds(
        self,
        painter: QtGui.QPainter,
        width: int,
        body_top: int,
        body_height: int,
        start_sec: float,
        end_sec: float,
    ) -> None:
        assert self._envelope is not None
        out_color = QtGui.QColor(15, 16, 20)
        if start_sec < 0.0:
            zero_x = round((0.0 - start_sec) / (end_sec - start_sec) * width)
            painter.fillRect(0, body_top, max(0, min(width, zero_x)), body_height, out_color)
        duration = self._envelope.duration_seconds
        if end_sec > duration:
            end_x = round((duration - start_sec) / (end_sec - start_sec) * width)
            clamped_x = max(0, min(width, end_x))
            painter.fillRect(clamped_x, body_top, width - clamped_x, body_height, out_color)

    def _paint_ruler(
        self,
        painter: QtGui.QPainter,
        width: int,
        start_sec: float,
        end_sec: float,
    ) -> None:
        assert self._envelope is not None
        painter.fillRect(0, 0, width, _RULER_HEIGHT, QtGui.QColor(33, 36, 43))
        painter.setPen(QtGui.QColor(55, 60, 72))
        painter.drawLine(0, _RULER_HEIGHT - 1, width, _RULER_HEIGHT - 1)

        step = _choose_tick_step(self._window_seconds, width)
        first_tick = math.ceil(max(0.0, start_sec) / step) * step
        max_time = min(end_sec, self._envelope.duration_seconds)

        tick_pen = QtGui.QPen(QtGui.QColor(135, 142, 156))
        painter.setPen(tick_pen)
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)

        tick = first_tick
        while tick <= max_time + 1e-9:
            x = round((tick - start_sec) / self._window_seconds * width)
            if 0 <= x <= width:
                painter.drawLine(x, _RULER_HEIGHT - 5, x, _RULER_HEIGHT - 1)
                label = ballroom_scribe.utils.format_duration(max(0.0, tick))
                painter.drawText(x + 3, _RULER_HEIGHT - 6, label)
            tick += step

        # Visible-span readout (on its own background so it stays legible over tick labels).
        span_text = _format_span(self._window_seconds)
        span_width = painter.fontMetrics().horizontalAdvance(span_text) + 12
        span_rect = QtCore.QRect(width - span_width, 0, span_width, _RULER_HEIGHT - 1)
        painter.fillRect(span_rect, QtGui.QColor(33, 36, 43))
        painter.setPen(QtGui.QColor(185, 192, 206))
        painter.drawText(
            span_rect.adjusted(0, 0, -6, 0),
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter,
            span_text,
        )

    def _paint_waveform(
        self,
        painter: QtGui.QPainter,
        width: int,
        body_top: int,
        body_height: int,
        start_sec: float,
        end_sec: float,
    ) -> None:
        assert self._envelope is not None
        mid_y = body_top + body_height / 2.0
        half_height = (body_height - 8) / 2.0

        painter.setPen(QtGui.QColor(45, 50, 60))
        painter.drawLine(0, round(mid_y), width, round(mid_y))

        col_mins, col_maxs = self._envelope.resample(start_sec, end_sec, width)
        valid_indices = np.flatnonzero(~np.isnan(col_mins))
        if valid_indices.size == 0:
            return

        norm_mins = np.clip(col_mins[valid_indices] / self._peak_amplitude, -1.0, 1.0)
        norm_maxs = np.clip(col_maxs[valid_indices] / self._peak_amplitude, -1.0, 1.0)
        y_tops = mid_y - norm_maxs * half_height
        y_bottoms = np.maximum(y_tops + 1.0, mid_y - norm_mins * half_height)

        lines = [
            QtCore.QLineF(float(x), float(y_top), float(x), float(y_bot))
            for x, y_top, y_bot in zip(valid_indices, y_tops, y_bottoms, strict=True)
        ]
        painter.setPen(QtGui.QPen(QtGui.QColor(78, 168, 222), 1))
        painter.drawLines(lines)

    def _paint_markers(
        self,
        painter: QtGui.QPainter,
        width: int,
        body_top: int,
        body_height: int,
        start_sec: float,
        end_sec: float,
    ) -> None:
        if not self._markers:
            return
        metrics = painter.fontMetrics()
        # Draw unselected markers first so selected markers always render on top.
        ordered = sorted(self._markers, key=lambda m: m.row in self._selected_rows)
        for marker in ordered:
            if not start_sec <= marker.seconds <= end_sec:
                continue
            x = round((marker.seconds - start_sec) / self._window_seconds * width)
            is_selected = marker.row in self._selected_rows
            color = QtGui.QColor(255, 196, 61) if is_selected else QtGui.QColor(215, 150, 55, 190)
            pen_width = 2 if is_selected else 1
            painter.setPen(QtGui.QPen(color, pen_width))
            painter.drawLine(x, body_top, x, body_top + body_height)

            if marker.label:
                text = metrics.elidedText(marker.label, QtCore.Qt.TextElideMode.ElideRight, 140)
                text_width = metrics.horizontalAdvance(text) + 8
                text_height = metrics.height() + 4
                pill_rect = QtCore.QRect(x + 2, body_top + 3, text_width, text_height)
                bg = QtGui.QColor(255, 196, 61, 230) if is_selected else QtGui.QColor(45, 48, 58, 215)
                fg = QtGui.QColor(20, 20, 24) if is_selected else QtGui.QColor(235, 205, 135)
                painter.fillRect(pill_rect, bg)
                painter.setPen(fg)
                painter.drawText(pill_rect, QtCore.Qt.AlignmentFlag.AlignCenter, text)

    def _paint_playhead(self, painter: QtGui.QPainter, width: int, height: int) -> None:
        playhead_x = min(max(0, round((0.5 + self._playhead_offset_fraction) * width)), width - 1)
        playhead_color = QtGui.QColor(255, 89, 94)
        painter.save()
        painter.setPen(QtGui.QPen(playhead_color, 2))
        painter.drawLine(playhead_x, 0, playhead_x, height)

        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(playhead_color)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        triangle = QtGui.QPolygonF(
            [
                QtCore.QPointF(playhead_x - 5.0, 0.0),
                QtCore.QPointF(playhead_x + 5.0, 0.0),
                QtCore.QPointF(float(playhead_x), 7.0),
            ]
        )
        painter.drawPolygon(triangle)
        painter.restore()

    def _emit_seek(self, seconds: float) -> None:
        self._in_explicit_seek = True
        try:
            self.seek_requested.emit(seconds)
        finally:
            self._in_explicit_seek = False

    def _set_offset_keeping_start(self, start_sec: float) -> None:
        raw_offset = (self._position_seconds - start_sec) / self._window_seconds - 0.5
        clamped_offset = min(max(-0.5, raw_offset), 0.5)
        if abs(clamped_offset) < _CENTER_EPSILON:
            self._playhead_offset_fraction = 0.0
            self._recenter_remaining_seconds = 0.0
        else:
            self._playhead_offset_fraction = clamped_offset
            self._recenter_remaining_seconds = _RECENTER_DURATION_SECONDS

    def _advance_recenter_drift(self, dt: float) -> None:
        if dt >= self._recenter_remaining_seconds:
            self._playhead_offset_fraction = 0.0
            self._recenter_remaining_seconds = 0.0
            return
        remaining = self._recenter_remaining_seconds - dt
        self._playhead_offset_fraction *= remaining / self._recenter_remaining_seconds
        if abs(self._playhead_offset_fraction) < _CENTER_EPSILON:
            self._playhead_offset_fraction = 0.0
            self._recenter_remaining_seconds = 0.0
        else:
            self._recenter_remaining_seconds = remaining

    @override
    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if self._envelope is None:
            event.ignore()
            return
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        # Proportional to the scrolled amount: a mouse-wheel notch zooms one step and a touchpad zooms smoothly.
        self.set_window_seconds(self._window_seconds * _ZOOM_STEP ** (delta / _WHEEL_NOTCH_ANGLE_DELTA))
        event.accept()

    @override
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() != QtCore.Qt.MouseButton.LeftButton or self._envelope is None:
            super().mousePressEvent(event)
            return
        self.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
        self._pressing = True
        self._dragging = False
        self._drag_anchor_x = event.position().x()
        self._drag_anchor_seconds = self._position_seconds
        event.accept()

    @override
    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self._pressing or self._envelope is None:
            super().mouseMoveEvent(event)
            return
        delta_x = event.position().x() - self._drag_anchor_x
        if not self._dragging and abs(delta_x) < QtWidgets.QApplication.startDragDistance():
            event.accept()
            return
        self._dragging = True
        width = max(1, self.width())
        # Keep `playhead_offset_fraction` fixed while dragging so the audio under the cursor follows the mouse 1:1
        # without jumping: dragging to the right moves the playhead to an *earlier* time.
        dragged_sec = (delta_x / width) * self._window_seconds
        clamped = min(max(0.0, self._drag_anchor_seconds - dragged_sec), self._envelope.duration_seconds)
        self._emit_seek(clamped)
        event.accept()

    @override
    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._pressing:
            was_dragging = self._dragging
            self._pressing = False
            self._dragging = False
            if not was_dragging and self._envelope is not None:
                width = max(1, self.width())
                old_start_sec, _ = self.visible_range
                clicked_sec = old_start_sec + (self._drag_anchor_x / width) * self._window_seconds
                clamped = min(max(0.0, clicked_sec), self._envelope.duration_seconds)
                self._emit_seek(clamped)
                self._set_offset_keeping_start(old_start_sec)
                self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    @override
    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        fine = bool(event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier)
        if event.key() == QtCore.Qt.Key.Key_Left:
            self.step_requested.emit(-0.1 if fine else -1.0)
            event.accept()
            return
        if event.key() == QtCore.Qt.Key.Key_Right:
            self.step_requested.emit(0.1 if fine else 1.0)
            event.accept()
            return
        super().keyPressEvent(event)
