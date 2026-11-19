"""Video player widget wrapping `QMediaPlayer`, `QAudioOutput`, and `CaptionedVideoWidget`."""

import logging
import pathlib
import time
from collections.abc import Sequence
from typing import override

from PySide6 import QtCore, QtGui, QtMultimedia, QtWidgets

import ballroom_scribe.gui
import ballroom_scribe.gui.utils
import ballroom_scribe.models
import ballroom_scribe.utils

_LOGGER = logging.getLogger(__name__)
_POLL_INTERVAL_MS = 16
_FIRST_FRAME_TIMEOUT_MS = 3000
# While playing, a frame that carries a changed on-picture caption is rendered well within this time.
_OVERLAY_FRAME_SECONDS = 0.25
_CAPTIONS_TOOLTIP = "Show or hide closed captions (C)"
_NO_CAPTIONS_TOOLTIP = "Closed captions appear once the speech of this video has been transcribed"
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
    crossing a timestamp. For a moment after playback starts, the backend can report a position that playback
    cannot have reached, such as the target of an earlier seek; neither signal reports it (see `is_stale_position`).

    The closed caption at the playhead (see `set_captions`) is shown below the picture when the pane has room there
    and on the picture otherwise, and the `CC` button hides it. Qt draws an on-picture caption with the frames that
    it renders, so when that caption changes while the player is paused, the widget seeks to the position it is
    already at, which renders the frame again.

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

        self._captions: tuple[ballroom_scribe.models.TranscriptChunk, ...] = ()
        self._caption_text = ""
        # Set while a seek is about to be issued, whose frame will carry any change of the on-picture caption.
        self._frame_is_on_its_way = False
        self._overlay_changed_at = -float("inf")

        self._video_view = ballroom_scribe.gui.CaptionedVideoWidget(self)
        self._video_view.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
        self._video_view.setMinimumHeight(220)
        self._video_view.installEventFilter(self)
        self._video_view.overlay_changed.connect(self._on_overlay_changed)
        self._player.setVideoSink(self._video_view.video_sink())

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

        # Disabled until the active video has captions; `MainWindow` also offers the action in its View menu.
        self._captions_action = QtGui.QAction("Closed Captions", self)
        self._captions_action.setIconText("CC")
        self._captions_action.setCheckable(True)
        self._captions_action.setChecked(True)
        self._captions_action.setEnabled(False)
        self._captions_action.setToolTip(_NO_CAPTIONS_TOOLTIP)
        self._captions_action.toggled.connect(self._on_captions_toggled)
        self._captions_button = QtWidgets.QToolButton(self)
        self._captions_button.setObjectName("captionsButton")
        self._captions_button.setDefaultAction(self._captions_action)

        controls_layout = QtWidgets.QHBoxLayout()
        controls_layout.setContentsMargins(6, 4, 6, 4)
        controls_layout.addWidget(self._play_button)
        controls_layout.addWidget(self._slider, stretch=1)
        controls_layout.addWidget(self._time_label)
        controls_layout.addWidget(self._captions_button)

        root_layout = QtWidgets.QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(self._video_view, stretch=1)
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
        # Where and when playback started or the playhead was last moved, for telling an out-of-date position report
        # from playback progress, and the last report that was ignored as out of date.
        self._progress_start_ms: int = 0
        self._progress_started_at = -float("inf")
        self._ignored_position_ms: int | None = None

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

    @property
    def captions_action(self) -> QtGui.QAction:
        """Returns the checkable action that shows or hides closed captions, which the `CC` button triggers."""
        return self._captions_action

    @property
    def captions(self) -> tuple[ballroom_scribe.models.TranscriptChunk, ...]:
        """Returns the closed captions of the loaded video (see `set_captions`)."""
        return self._captions

    @property
    def caption_text(self) -> str:
        """Returns the text of the caption at the playhead, or an empty string if nothing is being said there."""
        return self._caption_text

    @property
    def caption_placement(self) -> ballroom_scribe.gui.CaptionedVideoWidget.Placement:
        """Returns where captions are shown: below the picture, on the picture, or not at all."""
        return self._video_view.placement

    def video_sink(self) -> QtMultimedia.QVideoSink:
        """Returns the `QVideoSink` that receives decoded frames, e.g. to observe rendering from a smoke test."""
        return self._video_view.video_sink()

    def set_captions(self, captions: Sequence[ballroom_scribe.models.TranscriptChunk]) -> None:
        """Sets the closed captions of the loaded video and shows the one at the playhead.

        Call it again whenever the transcript grows: the picture changes only if the caption at the playhead did.
        Loading another video with `set_source` removes the captions.

        Args:
            captions: The captions in chronological order, as returned by `group_words_into_captions`. Each one is
                shown while the playhead is inside its time span. Without captions, the `CC` button is disabled.
        """
        self._captions = tuple(captions)
        has_captions = bool(self._captions)
        self._captions_action.setEnabled(has_captions)
        self._captions_action.setToolTip(_CAPTIONS_TOOLTIP if has_captions else _NO_CAPTIONS_TOOLTIP)
        self._video_view.set_captions_visible(has_captions and self._captions_action.isChecked())
        self._show_caption_at(self.position_seconds())

    def toggle_captions(self) -> None:
        """Shows or hides closed captions, as clicking the `CC` button does; does nothing without captions."""
        if self._captions_action.isEnabled():
            self._captions_action.toggle()

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
        # A new video must never show the previous video's captions.
        self.set_captions(())

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
        self._mark_progress_start()

        status = self._player.mediaStatus()
        if status in (
            QtMultimedia.QMediaPlayer.MediaStatus.LoadingMedia,
            QtMultimedia.QMediaPlayer.MediaStatus.NoMedia,
        ):
            self._pending_seek_ms = target_ms
            self._show_caption_at(target_ms / 1000.0)
        else:
            self._pending_seek_ms = None
            # The caption is set before the seek so that the frame decoded for the seek already carries it.
            self._show_caption_at(target_ms / 1000.0, frame_is_on_its_way=True)
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
        if watched is self._video_view and event.type() == QtCore.QEvent.Type.KeyPress:
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

    def _mark_progress_start(self) -> None:
        """Notes where and when playback starts or the playhead is moved, which `_apply_position_update` relies on."""
        self._progress_start_ms = self._cached_position_ms
        self._progress_started_at = time.monotonic()

    def _on_playback_state_changed(self, state: QtMultimedia.QMediaPlayer.PlaybackState) -> None:
        _LOGGER.debug("Playback state: %s", state.name)
        self._update_play_icon()
        playing = state == QtMultimedia.QMediaPlayer.PlaybackState.PlayingState
        if playing:
            self._mark_progress_start()
            self._poll_timer.start()
        else:
            self._poll_timer.stop()
        self._set_playing(playing)
        if (
            state == QtMultimedia.QMediaPlayer.PlaybackState.PausedState
            and time.monotonic() - self._overlay_changed_at < _OVERLAY_FRAME_SECONDS
        ):
            # The on-picture caption changed just before playback paused, perhaps after the last frame was rendered.
            self._render_frame_again()

    def _on_captions_toggled(self, checked: bool) -> None:
        self._video_view.set_captions_visible(checked and bool(self._captions))

    def _show_caption_at(self, seconds: float, *, frame_is_on_its_way: bool = False) -> None:
        """Shows the caption whose time span contains `seconds`.

        Args:
            seconds: The playhead position.
            frame_is_on_its_way: Whether the caller is about to seek. The frame decoded for that seek carries a
                changed on-picture caption, so the current frame does not need to be rendered again.
        """
        caption = ballroom_scribe.gui.utils.find_caption(self._captions, seconds)
        self._caption_text = caption.get_text() if caption is not None else ""
        self._frame_is_on_its_way = frame_is_on_its_way
        try:
            self._video_view.set_caption(self._caption_text)
        finally:
            self._frame_is_on_its_way = False

    def _on_overlay_changed(self) -> None:
        """Gets a changed on-picture caption onto the screen, which takes a newly rendered frame."""
        self._overlay_changed_at = time.monotonic()
        if not self._frame_is_on_its_way:
            self._render_frame_again()

    def _render_frame_again(self) -> None:
        """Seeks to the current position while paused, which makes Qt render the frame there again.

        While playing, the next frame is moments away. While media is loading or a seek is pending, the frame that
        follows carries the caption. While stopped, Qt renders no frames at all.
        """
        if (
            self._has_source
            and self._pending_seek_ms is None
            and self._player.playbackState() == QtMultimedia.QMediaPlayer.PlaybackState.PausedState
        ):
            self._player.setPosition(self._cached_position_ms)

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

        A report that playback cannot have reached is ignored for a moment after playback starts or the playhead is
        moved (see `is_stale_position`). Otherwise the playhead would jump there and back, and listeners to
        `playback_advanced` would take the jump for playback across everything in between.
        """
        previous_ms = self._cached_position_ms
        if position_ms == previous_ms:
            return
        if self._is_playing and ballroom_scribe.gui.utils.is_stale_position(
            self._progress_start_ms / 1000.0,
            position_ms / 1000.0,
            time.monotonic() - self._progress_started_at,
        ):
            if position_ms != self._ignored_position_ms:
                self._ignored_position_ms = position_ms
                _LOGGER.debug(
                    "Ignoring the out-of-date position %.3f s, reported while playing from %.3f s",
                    position_ms / 1000.0,
                    self._progress_start_ms / 1000.0,
                )
            return
        self._ignored_position_ms = None
        self._cached_position_ms = position_ms
        if not self._slider_scrubbing:
            self._slider.setValue(position_ms)
        self._update_time_label()
        self._show_caption_at(position_ms / 1000.0)
        self.position_changed.emit(position_ms / 1000.0)
        if self._is_playing and position_ms > previous_ms:
            self.playback_advanced.emit(previous_ms / 1000.0, position_ms / 1000.0)

    def _on_slider_pressed(self) -> None:
        self._slider_scrubbing = True

    def _on_slider_moved(self, value_ms: int) -> None:
        self._cached_position_ms = value_ms
        self._mark_progress_start()
        self._pending_seek_ms = None
        self._show_caption_at(value_ms / 1000.0, frame_is_on_its_way=True)
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
