"""Video player widget wrapping `QMediaPlayer`, `QAudioOutput`, and `QVideoWidget`."""

import logging
import pathlib
from typing import override

from PySide6 import QtCore, QtGui, QtMultimedia, QtMultimediaWidgets, QtWidgets

import ballroom_scribe.utils

_LOGGER = logging.getLogger(__name__)
_POLL_INTERVAL_MS = 16
_FIRST_FRAME_TIMEOUT_MS = 3000
_READY_STATUSES = frozenset(
    {
        QtMultimedia.QMediaPlayer.MediaStatus.LoadedMedia,
        QtMultimedia.QMediaPlayer.MediaStatus.BufferedMedia,
    }
)


def _describe_video_track(track: QtMultimedia.QMediaMetaData) -> str:
    key = QtMultimedia.QMediaMetaData.Key
    fields = (
        ("codec", key.VideoCodec),
        ("resolution", key.Resolution),
        ("fps", key.VideoFrameRate),
        ("bitrate", key.VideoBitRate),
    )
    described = [f"{label}={track.stringValue(k)}" for label, k in fields if track.stringValue(k)]
    return " ".join(described) if described else "no metadata"


def _describe_video_frame(frame: QtMultimedia.QVideoFrame) -> str:
    size = frame.size()
    return (
        f"{size.width()}x{size.height()} {frame.pixelFormat().name} ({frame.handleType().name})"
        f" at {frame.startTime() / 1_000_000:.3f} s"
    )


class VideoPlayerWidget(QtWidgets.QWidget):
    """Displays video playback with transport controls and ~60 Hz position updates.

    Qt renders no video frames while a `QMediaPlayer` is in `StoppedState`, so this widget pauses the player as soon as
    media finishes loading (unless playback should resume). That keeps the frame at the playhead visible when a video is
    opened or an event row is selected, and seeking while paused refreshes the displayed frame.

    `position_changed` reports every playhead move, whereas `playback_advanced` reports only the stretches of media
    time covered by live playback (never seeks, steps, or slider scrubs), so listeners can react to the playhead
    crossing a timestamp.

    Loading, media metadata, the first rendered frame, playback errors, and status transitions are logged via
    `logging` to help diagnose decoding or rendering problems.
    """

    position_changed = QtCore.Signal(float)
    playback_advanced = QtCore.Signal(float, float)
    duration_changed = QtCore.Signal(float)
    playing_changed = QtCore.Signal(bool)
    slider_scrubbed = QtCore.Signal()
    error_occurred = QtCore.Signal(str)
    step_requested = QtCore.Signal(float)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

        self._player = QtMultimedia.QMediaPlayer(self)
        self._audio_output = QtMultimedia.QAudioOutput(self)
        self._player.setAudioOutput(self._audio_output)

        self._video_widget = QtMultimediaWidgets.QVideoWidget(self)
        self._video_widget.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
        self._video_widget.setMinimumHeight(220)
        self._video_widget.installEventFilter(self)
        self._player.setVideoOutput(self._video_widget)

        self._play_button = QtWidgets.QPushButton(self)
        self._play_button.setEnabled(False)
        self._play_button.setToolTip("Play / Pause (Ctrl+K, Space, or K)")
        self._play_button.clicked.connect(self.toggle_play_pause)
        self._update_play_icon()

        self._slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self)
        self._slider.setRange(0, 0)
        self._slider.setEnabled(False)
        self._slider_scrubbing = False
        self._slider.sliderPressed.connect(self._on_slider_pressed)
        self._slider.sliderMoved.connect(self._on_slider_moved)
        self._slider.sliderReleased.connect(self._on_slider_released)

        self._time_label = QtWidgets.QLabel("0:00:00.00 / 0:00:00.00", self)
        mono_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        self._time_label.setFont(mono_font)

        controls_layout = QtWidgets.QHBoxLayout()
        controls_layout.setContentsMargins(6, 4, 6, 4)
        controls_layout.addWidget(self._play_button)
        controls_layout.addWidget(self._slider, stretch=1)
        controls_layout.addWidget(self._time_label)

        root_layout = QtWidgets.QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(self._video_widget, stretch=1)
        root_layout.addLayout(controls_layout)

        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.setInterval(_POLL_INTERVAL_MS)
        self._poll_timer.timeout.connect(self._emit_live_position)

        self._first_frame_timer = QtCore.QTimer(self)
        self._first_frame_timer.setSingleShot(True)
        self._first_frame_timer.setInterval(_FIRST_FRAME_TIMEOUT_MS)
        self._first_frame_timer.timeout.connect(self._on_first_frame_timeout)

        self._has_source = False
        self._source_path: pathlib.Path | None = None
        self._is_playing = False
        self._awaiting_first_frame = False
        self._pending_seek_ms: int | None = None
        self._resume_playback_after_load = False
        self._cached_position_ms: int = 0
        self._duration_ms: int = 0

        self._player.playbackStateChanged.connect(self._on_playback_state_changed)
        self._player.positionChanged.connect(self._on_player_position_changed)
        self._player.durationChanged.connect(self._on_player_duration_changed)
        self._player.mediaStatusChanged.connect(self._on_media_status_changed)
        self._player.errorOccurred.connect(self._on_error_occurred)

    @property
    def playback_state(self) -> QtMultimedia.QMediaPlayer.PlaybackState:
        """Returns the underlying `QMediaPlayer` playback state."""
        return self._player.playbackState()

    @property
    def is_playing(self) -> bool:
        """Returns whether the player is currently in `PlayingState`."""
        return self._is_playing

    @property
    def source_path(self) -> pathlib.Path | None:
        """Returns the path of the loaded video, if any."""
        return self._source_path

    def video_sink(self) -> QtMultimedia.QVideoSink:
        """Returns the `QVideoSink` that receives decoded frames, e.g. to observe rendering from a smoke test."""
        return self._video_widget.videoSink()

    def set_source(
        self,
        video_path: pathlib.Path | None,
        *,
        initial_seconds: float | None = None,
        preserve_playing: bool = False,
    ) -> None:
        """Loads `video_path` into the media player, optionally seeking once media is ready."""
        was_playing = (
            preserve_playing and self._player.playbackState() == QtMultimedia.QMediaPlayer.PlaybackState.PlayingState
        )
        self._poll_timer.stop()

        if video_path is None:
            _LOGGER.debug("Clearing video source")
            self._stop_awaiting_first_frame()
            self._has_source = False
            self._source_path = None
            self._pending_seek_ms = None
            self._resume_playback_after_load = False
            self._cached_position_ms = 0
            self._duration_ms = 0
            self._player.stop()
            self._set_playing(False)
            self._player.setSource(QtCore.QUrl())
            self._play_button.setEnabled(False)
            self._slider.setEnabled(False)
            self._slider.setRange(0, 0)
            self._update_time_label()
            self.position_changed.emit(0.0)
            return

        self._stop_awaiting_first_frame()
        self._source_path = video_path
        target_ms = max(0, round((initial_seconds or 0.0) * 1000))
        self._pending_seek_ms = target_ms
        self._cached_position_ms = target_ms
        self._resume_playback_after_load = was_playing
        self._duration_ms = 0

        _LOGGER.info("Loading %s (initial position %.3f s)", video_path, target_ms / 1000.0)
        self._play_button.setEnabled(True)
        self._slider.setEnabled(True)
        # `setSource()` stops the previous media first, synchronously emitting its `LoadedMedia` status; keep
        # `_has_source` cleared until it returns so that stale status is not mistaken for the new video.
        self._has_source = False
        self._player.setSource(QtCore.QUrl.fromLocalFile(str(video_path)))
        self._has_source = True
        self._start_awaiting_first_frame()
        self._update_time_label()
        self.position_changed.emit(target_ms / 1000.0)
        if self._player.mediaStatus() in _READY_STATUSES:
            self._handle_media_ready()

    def seek(self, seconds: float) -> None:
        """Seeks to `seconds` (clamped to valid media bounds)."""
        if not self._has_source:
            return
        target_ms = max(0, round(seconds * 1000))
        if self._duration_ms > 0:
            target_ms = min(target_ms, self._duration_ms)
        self._cached_position_ms = target_ms

        status = self._player.mediaStatus()
        if status in (
            QtMultimedia.QMediaPlayer.MediaStatus.LoadingMedia,
            QtMultimedia.QMediaPlayer.MediaStatus.NoMedia,
        ):
            self._pending_seek_ms = target_ms
        else:
            self._pending_seek_ms = None
            self._player.setPosition(target_ms)

        if not self._slider_scrubbing:
            self._slider.setValue(target_ms)
        self._update_time_label()
        self.position_changed.emit(target_ms / 1000.0)

    def step(self, delta_seconds: float) -> None:
        """Steps the playhead by `delta_seconds` relative to the current position."""
        self.seek(max(0.0, self.position_seconds() + delta_seconds))

    def toggle_play_pause(self) -> None:
        """Toggles between playing and paused states."""
        if not self._has_source:
            return
        if self._player.playbackState() == QtMultimedia.QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def pause(self) -> None:
        """Pauses playback if currently playing."""
        if self._player.playbackState() == QtMultimedia.QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()

    def position_seconds(self) -> float:
        """Returns the current playhead position in seconds."""
        if self._pending_seek_ms is not None:
            return self._pending_seek_ms / 1000.0
        return self._cached_position_ms / 1000.0

    @override
    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if watched is self._video_widget and event.type() == QtCore.QEvent.Type.KeyPress:
            assert isinstance(event, QtGui.QKeyEvent)
            if self._handle_arrow_step(event):
                return True
        return super().eventFilter(watched, event)

    @override
    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if self._handle_arrow_step(event):
            return
        super().keyPressEvent(event)

    def _handle_arrow_step(self, event: QtGui.QKeyEvent) -> bool:
        fine = bool(event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier)
        if event.key() == QtCore.Qt.Key.Key_Left:
            self.step_requested.emit(-0.1 if fine else -1.0)
            event.accept()
            return True
        if event.key() == QtCore.Qt.Key.Key_Right:
            self.step_requested.emit(0.1 if fine else 1.0)
            event.accept()
            return True
        return False

    def _set_playing(self, playing: bool) -> None:
        if playing != self._is_playing:
            self._is_playing = playing
            self.playing_changed.emit(playing)

    def _on_playback_state_changed(self, state: QtMultimedia.QMediaPlayer.PlaybackState) -> None:
        _LOGGER.debug("Playback state: %s", state.name)
        self._update_play_icon()
        playing = state == QtMultimedia.QMediaPlayer.PlaybackState.PlayingState
        if playing:
            self._poll_timer.start()
        else:
            self._poll_timer.stop()
        self._set_playing(playing)

    def _on_player_position_changed(self, position_ms: int) -> None:
        if self._pending_seek_ms is None:
            self._apply_position_update(position_ms)

    def _on_player_duration_changed(self, duration_ms: int) -> None:
        self._duration_ms = max(0, duration_ms)
        self._slider.setRange(0, self._duration_ms)
        self._update_time_label()
        self.duration_changed.emit(self._duration_ms / 1000.0)

    def _on_media_status_changed(self, status: QtMultimedia.QMediaPlayer.MediaStatus) -> None:
        _LOGGER.debug("Media status: %s", status.name)
        if self._has_source and status in _READY_STATUSES:
            self._handle_media_ready()

    def _handle_media_ready(self) -> None:
        """Applies the pending seek and resumes or pauses playback once the current media can render frames."""
        if self._awaiting_first_frame and not self._first_frame_timer.isActive():
            self._log_loaded_media()
            self._first_frame_timer.start()
        if self._pending_seek_ms is not None:
            target_ms = self._pending_seek_ms
            self._pending_seek_ms = None
            self._player.setPosition(target_ms)
        if self._resume_playback_after_load:
            self._resume_playback_after_load = False
            self._player.play()
        elif self._player.playbackState() == QtMultimedia.QMediaPlayer.PlaybackState.StoppedState:
            # Qt renders no frames while stopped; pausing displays the frame at the playhead.
            self._player.pause()

    def _on_error_occurred(
        self,
        error: QtMultimedia.QMediaPlayer.Error,
        error_string: str,
    ) -> None:
        if error != QtMultimedia.QMediaPlayer.Error.NoError:
            detail = error_string or error.name
            _LOGGER.error("Playback error %s for %s: %s", error.name, self._source_path, detail)
            self.error_occurred.emit(f"Playback error: {detail}")

    def _log_loaded_media(self) -> None:
        tracks = self._player.videoTracks()
        _LOGGER.info(
            "Loaded %s: duration %.3f s, audio available: %s, video available: %s, video tracks: %s",
            self._source_path,
            self._player.duration() / 1000.0,
            self._player.hasAudio(),
            self._player.hasVideo(),
            "; ".join(_describe_video_track(track) for track in tracks) if tracks else "none",
        )

    def _start_awaiting_first_frame(self) -> None:
        self._first_frame_timer.stop()
        if not self._awaiting_first_frame:
            self._awaiting_first_frame = True
            self.video_sink().videoFrameChanged.connect(self._on_video_frame_changed)

    def _stop_awaiting_first_frame(self) -> None:
        self._first_frame_timer.stop()
        if self._awaiting_first_frame:
            self._awaiting_first_frame = False
            self.video_sink().videoFrameChanged.disconnect(self._on_video_frame_changed)

    @QtCore.Slot(QtMultimedia.QVideoFrame)
    def _on_video_frame_changed(self, frame: QtMultimedia.QVideoFrame) -> None:
        if not frame.isValid():
            return
        _LOGGER.info("First video frame of %s: %s", self._source_path, _describe_video_frame(frame))
        self._stop_awaiting_first_frame()

    def _on_first_frame_timeout(self) -> None:
        _LOGGER.warning(
            "No video frame was rendered within %.1f s of loading %s (video available: %s, media status: %s, playback"
            " state: %s). If the video area stays black, see the troubleshooting section of README.md.",
            _FIRST_FRAME_TIMEOUT_MS / 1000.0,
            self._source_path,
            self._player.hasVideo(),
            self._player.mediaStatus().name,
            self._player.playbackState().name,
        )
        self._stop_awaiting_first_frame()

    def _emit_live_position(self) -> None:
        if self._pending_seek_ms is None:
            self._apply_position_update(self._player.position())

    def _apply_position_update(self, position_ms: int) -> None:
        """Publishes a position reported by the backend, flagging live playback progress via `playback_advanced`.

        Seeks store their target in `_cached_position_ms` before the backend reports it, so while playing, the span
        from the previously cached position to `position_ms` covers only media time that actually played.
        """
        previous_ms = self._cached_position_ms
        if position_ms == previous_ms:
            return
        self._cached_position_ms = position_ms
        if not self._slider_scrubbing:
            self._slider.setValue(position_ms)
        self._update_time_label()
        self.position_changed.emit(position_ms / 1000.0)
        if self._is_playing and position_ms > previous_ms:
            self.playback_advanced.emit(previous_ms / 1000.0, position_ms / 1000.0)

    def _on_slider_pressed(self) -> None:
        self._slider_scrubbing = True

    def _on_slider_moved(self, value_ms: int) -> None:
        self._cached_position_ms = value_ms
        self._pending_seek_ms = None
        self._player.setPosition(value_ms)
        self._update_time_label()
        self.slider_scrubbed.emit()
        self.position_changed.emit(value_ms / 1000.0)

    def _on_slider_released(self) -> None:
        self._slider_scrubbing = False
        self._on_slider_moved(self._slider.value())

    def _update_play_icon(self) -> None:
        playing = self._player.playbackState() == QtMultimedia.QMediaPlayer.PlaybackState.PlayingState
        pixmap = (
            QtWidgets.QStyle.StandardPixmap.SP_MediaPause if playing else QtWidgets.QStyle.StandardPixmap.SP_MediaPlay
        )
        self._play_button.setIcon(self.style().standardIcon(pixmap))

    def _update_time_label(self) -> None:
        current_str = ballroom_scribe.utils.format_duration(self.position_seconds())
        duration_str = ballroom_scribe.utils.format_duration(self._duration_ms / 1000.0)
        self._time_label.setText(f"{current_str} / {duration_str}")
