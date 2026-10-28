"""Headless end-to-end smoke test for the timestamp editor GUI.

Generates two short synthetic clips with `ffmpeg`, drives `ballroom_scribe.gui.MainWindow` through the main editing
workflows under Qt's `offscreen` platform (or in a real window with `--windowed`), and saves a screenshot. Unlike the
unit tests, this runs real `ffmpeg`/`ffprobe` subprocesses and Qt Multimedia decoding, so it is the quickest way to
confirm the editor works end to end on a new machine.
"""

import argparse
import contextlib
import dataclasses
import math
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator

import numpy as np
from PySide6 import QtCore, QtGui, QtMultimedia, QtTest, QtWidgets

import ballroom_scribe
import ballroom_scribe.cli.arguments
import ballroom_scribe.gui
import ballroom_scribe.gui.utils

_ML_MODULES = ("torch", "torchaudio", "faster_whisper", "ollama", "librosa")
_CLIP_SECONDS = 10.0
# The two clips differ in resolution so frames delivered to the video sink can be attributed to a clip.
_CLIP_SIZE = QtCore.QSize(640, 360)
_SECOND_CLIP_SIZE = QtCore.QSize(480, 270)
# A long, tiny clip whose audio extraction takes a while, so the window can be closed while it is in flight. PCM
# audio in a `.mov` container keeps generation nearly free (AAC encoding 30 minutes takes tens of seconds).
_LONG_CLIP_SIZE = QtCore.QSize(64, 36)
_LONG_CLIP_SECONDS = 1800.0
_LONG_CLIP_FPS = 2
_LONG_CLIP_AUDIO_EXPRESSIONS = "0.3*sin(440*2*PI*t)"
_LONG_CLIP_SAMPLE_RATE = 8000
_LONG_CLIP_AUDIO_ARGS = ("-c:a", "pcm_s16le")
_DEFAULT_AUDIO_ARGS = ("-c:a", "aac", "-b:a", "128k")
_CLOSE_TIMEOUT_SECONDS = 5.0
_CLIP_FPS = 24
_AUDIO_SAMPLE_RATE = 44100
# Channel 0 carries a 440 Hz tone from 1-9 s and channel 1 an 880 Hz tone from 2.5-7.5 s, so the extracted channel
# can be recognized from the waveform envelope alone.
_AUDIO_EXPRESSIONS = "0.5*sin(440*2*PI*t)*between(t,1,9)|0.5*sin(880*2*PI*t)*between(t,2.5,7.5)"
_CHANNEL_1_SILENT_SECONDS = 1.5
_CHANNEL_1_TONE_SECONDS = 5.0
_VIDEO_ENCODERS = (("libx264", ("-pix_fmt", "yuv420p", "-preset", "veryfast")), ("mpeg4", ("-q:v", "4")))
_FIRST_ROW_SECONDS = 2.5
_SECOND_ROW_SECONDS = 6.75
_SECOND_CLIP_ROW_SECONDS = 3.25
_INITIAL_ROW_COUNT = 3
_FRAME_TOLERANCE_SECONDS = 0.1
_WAVEFORM_TIMEOUT_SECONDS = 90.0
_FRAME_TIMEOUT_SECONDS = 20.0
_SCREENSHOT_SETTLE_SECONDS = 0.5
_PLAYBACK_TIMEOUT_SECONDS = 5.0
# Pixels sampled this far inside a cell's right edge are clear of both the current-cell frame and left-aligned text.
_CELL_INTERIOR_INSET = 6
_COLOR_TOLERANCE = 8
_ALT = QtCore.Qt.KeyboardModifier.AltModifier
_CTRL = QtCore.Qt.KeyboardModifier.ControlModifier
_NO_MODIFIER = QtCore.Qt.KeyboardModifier.NoModifier


class _SmokeTestError(RuntimeError):
    """Raised when an end-to-end expectation is not met."""


@dataclasses.dataclass(frozen=True)
class _Fixture:
    app: QtWidgets.QApplication
    window: ballroom_scribe.gui.MainWindow
    work_dir: pathlib.Path
    clip: pathlib.Path
    second_clip: pathlib.Path
    long_clip: pathlib.Path
    windowed: bool


@dataclasses.dataclass(frozen=True)
class _DeliveredFrame:
    seconds: float
    size: QtCore.QSize


@dataclasses.dataclass(frozen=True)
class _CellSample:
    """Colours sampled from one table cell in a viewport grab."""

    edge: QtGui.QColor
    interior: QtGui.QColor


class _FrameRecorder(QtCore.QObject):
    """Records frames delivered to a `QVideoSink`.

    Lives in the GUI thread so Qt queues `videoFrameChanged` deliveries instead of running Python on the renderer
    thread, which could deadlock against the GIL.
    """

    def __init__(self) -> None:
        super().__init__()
        self.frames: list[_DeliveredFrame] = []

    @QtCore.Slot(QtMultimedia.QVideoFrame)
    def record(self, frame: QtMultimedia.QVideoFrame) -> None:
        if frame.isValid():
            self.frames.append(_DeliveredFrame(seconds=frame.startTime() / 1_000_000, size=frame.size()))


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise _SmokeTestError(message)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an end-to-end smoke test of the timestamp editor GUI.")
    parser.add_argument(
        "--work-dir",
        type=pathlib.Path,
        default=None,
        help="Directory for the generated clips, CSVs, caches, and screenshot (default: a new temporary directory)",
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="Show a real window instead of using Qt's offscreen platform",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging, including Qt multimedia diagnostics",
    )
    return parser.parse_args(argv)


def _generate_clip(
    ffmpeg: str,
    output: pathlib.Path,
    size: QtCore.QSize,
    *,
    seconds: float = _CLIP_SECONDS,
    fps: int = _CLIP_FPS,
    audio_expressions: str = _AUDIO_EXPRESSIONS,
    channel_layout: str = "stereo",
    sample_rate: int = _AUDIO_SAMPLE_RATE,
    audio_args: tuple[str, ...] = _DEFAULT_AUDIO_ARGS,
) -> None:
    """Renders a test-pattern video with synthetic audio, falling back to MPEG-4 Part 2 without libx264."""
    failures: list[str] = []
    video_source = f"testsrc2=size={size.width()}x{size.height()}:rate={fps}:duration={seconds}"
    audio_source = (
        f"aevalsrc=exprs='{audio_expressions}':sample_rate={sample_rate}"
        f":channel_layout={channel_layout}:duration={seconds}"
    )
    for encoder, encoder_args in _VIDEO_ENCODERS:
        command = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            video_source,
            "-f",
            "lavfi",
            "-i",
            audio_source,
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            encoder,
            *encoder_args,
            *audio_args,
            str(output),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode == 0 and output.exists():
            print(f"Generated {output} with the {encoder} encoder")
            return
        failures.append(f"{encoder}: {result.stderr.strip() or f'exit code {result.returncode}'}")
    raise _SmokeTestError("ffmpeg could not generate the synthetic clip:\n" + "\n".join(failures))


def _wait_until(
    app: QtWidgets.QApplication,
    predicate: Callable[[], bool],
    timeout_seconds: float,
    description: str,
) -> None:
    """Pumps the Qt event loop until `predicate()` holds or `timeout_seconds` elapse."""
    deadline = time.monotonic() + timeout_seconds
    while not predicate():
        if time.monotonic() >= deadline:
            raise _SmokeTestError(f"Timed out after {timeout_seconds:.0f} s waiting for {description}")
        app.processEvents()
        time.sleep(0.02)


def _settle(app: QtWidgets.QApplication, seconds: float) -> None:
    """Pumps the Qt event loop for `seconds` so pending repaints reach the screen."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.02)


@contextlib.contextmanager
def _delivered_frames(window: ballroom_scribe.gui.MainWindow) -> Iterator[list[_DeliveredFrame]]:
    """Records every valid frame handed to the video sink while the context is active."""
    recorder = _FrameRecorder()
    sink = window.player.video_sink()
    sink.videoFrameChanged.connect(recorder.record)
    try:
        yield recorder.frames
    finally:
        sink.videoFrameChanged.disconnect(recorder.record)


def _wait_for_frame(
    fixture: _Fixture,
    frames: list[_DeliveredFrame],
    seconds: float,
    size: QtCore.QSize,
    description: str,
) -> None:
    """Waits until a frame of the given `size` within `_FRAME_TOLERANCE_SECONDS` of `seconds` has been delivered."""

    def frame_arrived() -> bool:
        return any(abs(frame.seconds - seconds) <= _FRAME_TOLERANCE_SECONDS and frame.size == size for frame in frames)

    _wait_until(fixture.app, frame_arrived, _FRAME_TIMEOUT_SECONDS, description)


def _expect_paused(fixture: _Fixture, context: str) -> None:
    state = fixture.window.player.playback_state
    _expect(
        state == QtMultimedia.QMediaPlayer.PlaybackState.PausedState,
        f"Player should be paused {context}, but is {state.name}",
    )


def _expect_source(fixture: _Fixture, clip: pathlib.Path, context: str) -> None:
    source = fixture.window.player.source_path
    _expect(source is not None and source.name == clip.name, f"Player source is {source} {context}")


def _focus_table(fixture: _Fixture) -> bool:
    """Activates the window and focuses the table; returns false only when a real window manager refuses."""
    window = fixture.window
    window.activateWindow()
    window.table_view.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
    fixture.app.processEvents()
    if window.table_view.hasFocus():
        return True
    _expect(fixture.windowed, "Could not focus the event table")
    print("  (skipped: the window manager did not grant focus)")
    return False


def _press(fixture: _Fixture, key: QtCore.Qt.Key, modifier: QtCore.Qt.KeyboardModifier = _NO_MODIFIER) -> None:
    """Sends a key click to the event table (window-level shortcuts see it first) and pumps events."""
    window = fixture.window
    if fixture.app.focusWidget() is None:
        # Closing a completer popup under the offscreen platform deactivates the window; re-activate it first.
        window.activateWindow()
        window.table_view.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
        fixture.app.processEvents()
    QtTest.QTest.keyClick(window.table_view, key, modifier)
    fixture.app.processEvents()


def _release_alt(fixture: _Fixture) -> None:
    """Clears the sticky `AltModifier` that synthesized `QTest` `Alt` events leave on `QApplication`."""
    QtTest.QTest.keyRelease(fixture.window.table_view, QtCore.Qt.Key.Key_Alt)
    fixture.app.processEvents()
    _expect(
        QtWidgets.QApplication.keyboardModifiers() == _NO_MODIFIER,
        f"Modifiers stuck after releasing Alt: {QtWidgets.QApplication.keyboardModifiers()}",
    )


def _hover(fixture: _Fixture, widget: QtWidgets.QWidget, position: QtCore.QPoint) -> None:
    """Delivers a button-less mouse move to `widget`.

    `QTest.mouseMove` without a pressed button only warps the real pointer, which a window system does not
    necessarily turn into a synchronous move event; sending the event directly works in both modes.
    """
    global_position = widget.mapToGlobal(QtCore.QPointF(position))
    event = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseMove,
        QtCore.QPointF(position),
        global_position,
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.MouseButton.NoButton,
        _NO_MODIFIER,
    )
    fixture.app.sendEvent(widget, event)
    fixture.app.processEvents()


def _open_cell_editor(fixture: _Fixture) -> QtWidgets.QWidget | None:
    """Returns the editor open on the table's current cell, if any.

    Looked up through the view rather than `focusWidget()`, which is `None` whenever the offscreen platform has
    deactivated the window (it does so when a completer popup closes).
    """
    view = fixture.window.table_view
    if view.state() != QtWidgets.QAbstractItemView.State.EditingState:
        return None
    return view.indexWidget(view.currentIndex())


def _cancel_cell_editor(fixture: _Fixture) -> None:
    editor = _open_cell_editor(fixture)
    if editor is not None:
        QtTest.QTest.keyClick(editor, QtCore.Qt.Key.Key_Escape)
        fixture.app.processEvents()


def _check_import_footprint() -> None:
    loaded = [name for name in _ML_MODULES if name in sys.modules]
    _expect(not loaded, f"Importing the GUI loaded heavy ML modules: {loaded}")


def _check_video_and_waveform(fixture: _Fixture) -> None:
    window = fixture.window
    window.add_videos(
        [
            ballroom_scribe.cli.arguments.InputCacheArgs(
                input_video=fixture.clip,
                input_channel=1,
                cache_dir=fixture.work_dir,
            )
        ]
    )
    _expect(len(window.session) == 1, "Video was not added to the session")
    video = window.session.videos[0]
    _expect(video.channel_count == 2, f"Expected 2 audio channels, got {video.channel_count}")

    _wait_until(
        fixture.app,
        lambda: window.waveform.envelope is not None or window.waveform.is_error,
        _WAVEFORM_TIMEOUT_SECONDS,
        "audio extraction and waveform peaks",
    )
    _expect(not window.waveform.is_error, f"Waveform error: {window.waveform.status_message}")
    envelope = window.waveform.envelope
    assert envelope is not None
    _expect(
        abs(envelope.duration_seconds - _CLIP_SECONDS) < 0.25,
        f"Waveform duration {envelope.duration_seconds:.2f} s, expected ~{_CLIP_SECONDS:.0f} s",
    )
    silent_peak = float(envelope.maxs[int(_CHANNEL_1_SILENT_SECONDS * envelope.bins_per_second)])
    tone_peak = float(envelope.maxs[int(_CHANNEL_1_TONE_SECONDS * envelope.bins_per_second)])
    _expect(
        silent_peak < 0.05 < tone_peak,
        f"Waveform does not match audio channel 1 (peak {silent_peak:.3f} at {_CHANNEL_1_SILENT_SECONDS} s,"
        f" {tone_peak:.3f} at {_CHANNEL_1_TONE_SECONDS} s)",
    )
    for cache_file in ("Audio-Channel-1.wav", "Waveform-Channel-1.npz"):
        _expect((video.cache_dir / cache_file).exists(), f"Missing cache file {video.cache_dir / cache_file}")


def _check_csv_row_jump_renders_frames(fixture: _Fixture) -> None:
    window = fixture.window
    clip = fixture.clip.resolve()
    second_clip = fixture.second_clip.resolve()
    initial_csv = fixture.work_dir / "initial_events.csv"
    initial_csv.write_text(
        "File,Inpoint,Inpoint Formatted,Level,Style,Dance,Round\n"
        f"{clip},{_FIRST_ROW_SECONDS},0:00:02.50,Bronze,Standard,Waltz,Quarterfinal\n"
        f"{clip},{_SECOND_ROW_SECONDS},0:00:06.75,Silver,Smooth,Tango,Final\n"
        f"{second_clip},{_SECOND_CLIP_ROW_SECONDS},0:00:03.25,Gold,Latin,Cha Cha,Final\n",
        encoding="utf-8",
    )
    _expect(window.open_csv(initial_csv, interactive_missing_videos=False), "open_csv failed")
    _expect(window.table_model.rowCount() == _INITIAL_ROW_COUNT, "CSV rows were not loaded into the table")
    _expect(len(window.session) == 2, "Opening the CSV did not load the second video it references")

    # Level, Style, Dance, and Round share the remaining width equally and fill the table.
    header = window.table_view.horizontalHeader()
    col = ballroom_scribe.gui.EventTableModel.Column
    text_widths = [header.sectionSize(int(column)) for column in (col.LEVEL, col.STYLE, col.DANCE, col.ROUND)]
    _expect(
        max(text_widths) - min(text_widths) <= 2,
        f"Level/Style/Dance/Round widths are not equal: {text_widths}",
    )
    # They fill the table (the columns may exceed it when the table is narrower than their minimum widths).
    total_width = sum(header.sectionSize(section) for section in range(header.count()))
    _expect(
        total_width >= window.table_view.viewport().width() - 2,
        f"Columns span only {total_width} px of the {window.table_view.viewport().width()} px wide table",
    )

    with _delivered_frames(window) as frames:
        window.table_view.setCurrentIndex(window.table_model.index(0, 0))
        fixture.app.processEvents()
        _expect(
            abs(window.player.position_seconds() - _FIRST_ROW_SECONDS) < 1e-3,
            f"Playhead at {window.player.position_seconds():.3f} s after clicking row 1",
        )
        _expect(
            abs(window.waveform.position_seconds - _FIRST_ROW_SECONDS) < 1e-3,
            "Waveform playhead did not follow the row click",
        )
        _wait_for_frame(fixture, frames, _FIRST_ROW_SECONDS, _CLIP_SIZE, "a video frame at the clicked row's inpoint")
        _expect_paused(fixture, "after loading")

        frames.clear()
        window.player.seek(_SECOND_ROW_SECONDS)
        _wait_for_frame(fixture, frames, _SECOND_ROW_SECONDS, _CLIP_SIZE, "a video frame after seeking while paused")

    # Clicking (or double-clicking to edit) inside the current row must not seek back to its inpoint.
    col = ballroom_scribe.gui.EventTableModel.Column
    level_cell = window.table_view.visualRect(window.table_model.index(0, int(col.LEVEL))).center()
    QtTest.QTest.mouseClick(window.table_view.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=level_cell)
    fixture.app.processEvents()
    _expect(
        abs(window.player.position_seconds() - _SECOND_ROW_SECONDS) < 1e-3,
        f"Clicking a cell of the current row moved the playhead to {window.player.position_seconds():.3f} s",
    )
    QtTest.QTest.mouseDClick(window.table_view.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=level_cell)
    fixture.app.processEvents()
    _expect(
        abs(window.player.position_seconds() - _SECOND_ROW_SECONDS) < 1e-3,
        f"Double-clicking a cell to edit it moved the playhead to {window.player.position_seconds():.3f} s",
    )
    _cancel_cell_editor(fixture)
    # Clicking a different row still jumps to that row's inpoint (move the playhead away first).
    window.player.seek(4.0)
    other_cell = window.table_view.visualRect(window.table_model.index(1, int(col.LEVEL))).center()
    QtTest.QTest.mouseClick(window.table_view.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=other_cell)
    fixture.app.processEvents()
    _expect(
        abs(window.player.position_seconds() - _SECOND_ROW_SECONDS) < 1e-3
        and window.table_view.currentIndex().row() == 1,
        "Clicking another row did not select it and jump to its inpoint",
    )
    window.table_view.setCurrentIndex(window.table_model.index(0, 0))
    fixture.app.processEvents()


def _viewport_pixel(image: QtGui.QImage, point: QtCore.QPoint) -> QtGui.QColor:
    """Returns the colour of a widget grab at the logical widget position `point`."""
    ratio = image.devicePixelRatio()
    return image.pixelColor(round(point.x() * ratio), round(point.y() * ratio))


def _colors_match(first: QtGui.QColor, second: QtGui.QColor) -> bool:
    return all(abs(a - b) <= _COLOR_TOLERANCE for a, b in zip(first.getRgb()[:3], second.getRgb()[:3], strict=True))


def _sample_cell(view: QtWidgets.QTableView, image: QtGui.QImage, index: QtCore.QModelIndex) -> _CellSample:
    """Samples a cell's top edge (where the current-cell frame is drawn) and its interior, clear of its text."""
    rect = view.visualRect(index)
    return _CellSample(
        edge=_viewport_pixel(image, QtCore.QPoint(rect.center().x(), rect.top() + 1)),
        interior=_viewport_pixel(image, QtCore.QPoint(rect.right() - _CELL_INTERIOR_INSET, rect.center().y())),
    )


def _check_current_cell_frame(fixture: _Fixture) -> None:
    """The focused table frames its current cell and leaves it unfilled, so the keyboard target stands out."""
    window = fixture.window
    view = window.table_view
    model = window.table_model
    col = ballroom_scribe.gui.EventTableModel.Column
    if not _focus_table(fixture):
        return
    level_index = model.index(0, int(col.LEVEL))
    style_index = model.index(0, int(col.STYLE))
    highlight = view.palette().color(QtGui.QPalette.ColorGroup.Active, QtGui.QPalette.ColorRole.Highlight)

    view.setCurrentIndex(level_index)
    fixture.app.processEvents()
    _expect(view.selectionModel().isRowSelected(0), "Setting the current cell did not select its row")
    image = view.viewport().grab().toImage()
    level = _sample_cell(view, image, level_index)
    style = _sample_cell(view, image, style_index)
    _expect(
        _colors_match(level.edge, highlight),
        f"Current cell is not framed in the highlight colour: edge {level.edge.name()}, highlight {highlight.name()}",
    )
    _expect(not _colors_match(level.interior, level.edge), "Current cell has no visible frame around its interior")
    _expect(
        not _colors_match(level.interior, style.interior),
        f"Current cell ({level.interior.name()}) is filled like the rest of the selected row ({style.interior.name()})",
    )

    view.setCurrentIndex(style_index)
    fixture.app.processEvents()
    image = view.viewport().grab().toImage()
    moved_level = _sample_cell(view, image, level_index)
    moved_style = _sample_cell(view, image, style_index)
    _expect(
        _colors_match(moved_style.edge, highlight)
        and _colors_match(moved_style.interior, level.interior)
        and _colors_match(moved_level.interior, style.interior),
        "The frame did not follow the current cell to the next column",
    )

    view.clearFocus()
    fixture.app.processEvents()
    image = view.viewport().grab().toImage()
    _expect(
        _colors_match(_sample_cell(view, image, style_index).interior, _sample_cell(view, image, level_index).interior),
        "The current cell is still singled out after the table lost focus",
    )
    _focus_table(fixture)

    # Other rows with the same non-empty Level, Style, and Dance as the selected row are faintly highlighted.
    style_row_0 = model.index(0, int(col.STYLE))
    style_row_2 = model.index(2, int(col.STYLE))
    view.setCurrentIndex(level_index)
    fixture.app.processEvents()
    plain_row_2 = _sample_cell(view, view.viewport().grab().toImage(), style_row_2).interior
    original_records = model.records
    model.set_records(
        [
            original_records[0],
            original_records[1],
            dataclasses.replace(original_records[2], level="bronze", style="Standard", dance="Waltz"),
        ]
    )
    view.setCurrentIndex(model.index(0, int(col.LEVEL)))
    fixture.app.processEvents()
    image = view.viewport().grab().toImage()
    tinted_row_2 = _sample_cell(view, image, style_row_2).interior
    selected_row_0 = _sample_cell(view, image, style_row_0).interior
    _expect(
        not _colors_match(tinted_row_2, plain_row_2),
        f"Row with the same Level/Style/Dance was not faintly highlighted ({tinted_row_2.name()} vs {plain_row_2.name()})",
    )
    _expect(
        not _colors_match(tinted_row_2, selected_row_0),
        f"Faintly highlighted row ({tinted_row_2.name()}) looks identical to the selected row ({selected_row_0.name()})",
    )
    view.setCurrentIndex(model.index(1, int(col.LEVEL)))
    fixture.app.processEvents()
    _expect(
        _colors_match(_sample_cell(view, view.viewport().grab().toImage(), style_row_2).interior, plain_row_2),
        "Row stayed faintly highlighted after selecting a row with a different event",
    )
    model.set_records(original_records)
    view.setCurrentIndex(model.index(0, int(col.LEVEL)))
    fixture.app.processEvents()


def _check_fill_handle(fixture: _Fixture) -> None:
    """Dragging the current cell's fill handle copies its value over the rows passed, as one undo step."""
    window = fixture.window
    view = window.table_view
    viewport = view.viewport()
    model = window.table_model
    col = ballroom_scribe.gui.EventTableModel.Column
    left = QtCore.Qt.MouseButton.LeftButton
    if not _focus_table(fixture):
        return

    def start_drag(origin: QtCore.QModelIndex, target_row: int) -> QtCore.QPoint:
        """Presses the handle of `origin` and drags onto `target_row`; returns where the button is held."""
        view.setCurrentIndex(origin)
        fixture.app.processEvents()
        handle = view.fill_handle_rect(origin)
        _expect(
            not handle.isNull(), f"No fill handle on the {col(origin.column()).name} cell of row {origin.row() + 1}"
        )
        _hover(fixture, viewport, handle.center())
        _expect(
            viewport.cursor().shape() == QtCore.Qt.CursorShape.CrossCursor,
            f"Pointer over the fill handle is {viewport.cursor().shape().name}, expected a crosshair",
        )
        QtTest.QTest.mousePress(viewport, left, pos=handle.center())
        target = view.visualRect(model.index(target_row, origin.column())).center()
        QtTest.QTest.mouseMove(viewport, target)
        fixture.app.processEvents()
        _expect(view.is_filling, "Pressing the fill handle did not start a fill")
        return target

    def release(target: QtCore.QPoint) -> None:
        QtTest.QTest.mouseRelease(viewport, left, pos=target)
        fixture.app.processEvents()

    def column_values(column: ballroom_scribe.gui.EventTableModel.Column, rows: int) -> list[str]:
        return [str(model.data(model.index(row, int(column)))) for row in range(rows)]

    def undo_state() -> tuple[int, int]:
        # `count()` keeps undone commands until a new one is pushed, so the index is tracked as well.
        return model.undo_stack.count(), model.undo_stack.index()

    undo_before_fill = undo_state()
    playhead = window.player.position_seconds()
    source = window.player.source_path

    # Fill down: row 1's Level onto rows 2 and 3.
    levels = column_values(col.LEVEL, 3)
    release(start_drag(model.index(0, int(col.LEVEL)), 2))
    _expect(column_values(col.LEVEL, 3) == [levels[0]] * 3, f"Filling down produced {column_values(col.LEVEL, 3)}")
    _expect(
        undo_state() == (undo_before_fill[0] + 1, undo_before_fill[1] + 1)
        and model.undo_stack.undoText() == "Fill Level",
        f"Fill was not recorded as a single 'Fill Level' undo step ({model.undo_stack.undoText()!r})",
    )
    _expect(view.currentIndex() == model.index(0, int(col.LEVEL)), "Filling moved the current cell")
    _expect(
        window.player.source_path == source and abs(window.player.position_seconds() - playhead) < 1e-6,
        "Filling moved the playhead",
    )
    _expect(
        not view.is_filling and view.state() != QtWidgets.QAbstractItemView.State.EditingState,
        "The table is still filling or editing after the release",
    )
    model.undo_stack.undo()
    _expect(column_values(col.LEVEL, 3) == levels, "Undo did not restore the filled cells")

    # Fill up: row 2's Style onto row 1 (the same video, so making row 2 current only seeks).
    styles = column_values(col.STYLE, 3)
    release(start_drag(model.index(1, int(col.STYLE)), 0))
    _expect(
        column_values(col.STYLE, 3) == [styles[1], styles[1], styles[2]],
        f"Filling up produced {column_values(col.STYLE, 3)}",
    )
    model.undo_stack.undo()
    _expect(column_values(col.STYLE, 3) == styles, "Undo did not restore the cell filled upwards")

    # Esc cancels a drag in progress.
    dances = column_values(col.DANCE, 3)
    undo_before_cancel = undo_state()
    target = start_drag(model.index(0, int(col.DANCE)), 2)
    _press(fixture, QtCore.Qt.Key.Key_Escape)
    _expect(not view.is_filling, "Esc did not cancel the fill")
    release(target)
    _expect(
        column_values(col.DANCE, 3) == dances and undo_state() == undo_before_cancel,
        "A cancelled fill still changed the table",
    )

    # Columns without a handle: the same corner of an Inpoint cell is an ordinary click.
    inpoint_index = model.index(0, int(col.INPOINT))
    view.setCurrentIndex(inpoint_index)
    fixture.app.processEvents()
    _expect(view.fill_handle_rect(inpoint_index).isNull(), "Inpoint cells must not offer a fill handle")
    QtTest.QTest.mouseClick(viewport, left, pos=view.visualRect(inpoint_index).bottomRight())
    fixture.app.processEvents()
    _expect(
        not view.is_filling and undo_state() == undo_before_cancel,
        "Clicking the corner of an Inpoint cell started a fill",
    )
    view.setCurrentIndex(model.index(0, 0))
    fixture.app.processEvents()


def _check_switching_video_renders_frames(fixture: _Fixture) -> None:
    window = fixture.window
    model = window.table_model
    with _delivered_frames(window) as frames:
        window.table_view.setCurrentIndex(model.index(2, 0))
        fixture.app.processEvents()
        _expect_source(fixture, fixture.second_clip, "after clicking a row of the second video")
        _expect(
            abs(window.player.position_seconds() - _SECOND_CLIP_ROW_SECONDS) < 1e-3,
            f"Playhead at {window.player.position_seconds():.3f} s after switching to the second video",
        )
        _wait_for_frame(
            fixture,
            frames,
            _SECOND_CLIP_ROW_SECONDS,
            _SECOND_CLIP_SIZE,
            "a frame of the second video at its row's inpoint",
        )
        _expect_paused(fixture, "after switching to the second video")

        frames.clear()
        window.table_view.setCurrentIndex(model.index(0, 0))
        fixture.app.processEvents()
        _expect_source(fixture, fixture.clip, "after clicking a row of the first video again")
        _wait_for_frame(
            fixture, frames, _FIRST_ROW_SECONDS, _CLIP_SIZE, "a frame of the first video after switching back"
        )
        _expect_paused(fixture, "after switching back to the first video")


def _send_wheel(fixture: _Fixture, angle_delta_y: int) -> None:
    """Delivers a wheel event over the middle of the waveform."""
    waveform = fixture.window.waveform
    local = QtCore.QPointF(waveform.width() / 2, waveform.height() / 2)
    event = QtGui.QWheelEvent(
        local,
        waveform.mapToGlobal(local),
        QtCore.QPoint(0, 0),
        QtCore.QPoint(0, angle_delta_y),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    fixture.app.sendEvent(waveform, event)
    fixture.app.processEvents()


def _expect_span(fixture: _Fixture, expected_seconds: float, context: str) -> None:
    actual = fixture.window.waveform.window_seconds
    _expect(
        abs(actual - expected_seconds) < 1e-6,
        f"Waveform span is {actual:.3f} s {context}, expected {expected_seconds:.3f} s",
    )


def _check_waveform_zoom(fixture: _Fixture) -> None:
    window = fixture.window
    waveform = window.waveform
    _expect(waveform.envelope is not None, "Waveform is not loaded")
    assert waveform.envelope is not None
    if not _focus_table(fixture):
        return
    # The 10 s clip is shorter than the 12 min default span, so the default is clamped to 1.1 x the clip length.
    widest = waveform.envelope.duration_seconds * 1.1
    spans: list[float] = []
    waveform.window_changed.connect(spans.append)
    try:
        waveform.reset_zoom()
        _expect_span(fixture, widest, "after resetting the zoom (clamped to the clip)")
        _press(fixture, QtCore.Qt.Key.Key_Equal, _CTRL)
        _press(fixture, QtCore.Qt.Key.Key_Equal, _CTRL)
        _expect_span(fixture, widest * 0.8 * 0.8, "after Ctrl+= twice")
        _press(fixture, QtCore.Qt.Key.Key_Minus, _CTRL)
        _expect_span(fixture, widest * 0.8, "after Ctrl+-")
        _press(fixture, QtCore.Qt.Key.Key_0, _CTRL)
        _expect_span(fixture, widest, "after Ctrl+0")

        _send_wheel(fixture, 120)  # One notch away from the user: zoom in one step.
        _expect_span(fixture, widest * 0.8, "after one wheel notch")
        _send_wheel(fixture, -60)  # Half a notch back: half a step out.
        _expect_span(fixture, widest * 0.8 / (0.8**0.5), "after half a wheel notch back")

        for _ in range(10):
            _press(fixture, QtCore.Qt.Key.Key_Minus, _CTRL)
        _expect_span(fixture, widest, "after zooming far out (should clamp to the clip)")
        for _ in range(20):
            _press(fixture, QtCore.Qt.Key.Key_Equal, _CTRL)
        _expect_span(fixture, 2.0, "after zooming far in (should clamp to 2 s)")
        _expect(bool(spans) and abs(spans[-1] - 2.0) < 1e-6, f"window_changed reported {spans[-3:]}")
        waveform.reset_zoom()
    finally:
        waveform.window_changed.disconnect(spans.append)


def _waveform_point(waveform: ballroom_scribe.gui.WaveformWidget, target_seconds: float) -> QtCore.QPoint:
    """Returns a point in the waveform body under `target_seconds`, clamped just inside the widget's width."""
    start_sec, _ = waveform.visible_range
    fraction = (target_seconds - start_sec) / waveform.window_seconds
    x = min(max(1, round(fraction * waveform.width())), waveform.width() - 2)
    return QtCore.QPoint(x, waveform.height() - 10)


def _check_waveform_click_and_drag(fixture: _Fixture) -> None:
    window = fixture.window
    waveform = window.waveform
    player = window.player
    view = window.table_view
    model = window.table_model
    col = ballroom_scribe.gui.EventTableModel.Column
    left = QtCore.Qt.MouseButton.LeftButton
    _expect(waveform.envelope is not None, "Waveform is not loaded")

    def point_at(target_seconds: float) -> QtCore.QPoint:
        return _waveform_point(waveform, target_seconds)

    before_click = waveform.position_seconds
    range_before_click = waveform.visible_range

    # 1. Clicking (without dragging) right of center seeks forward on release, draws the playhead where clicked while
    # keeping the visible waveform stationary, highlights row 1, and stays frozen while paused.
    click_pos = point_at(7.5)
    QtTest.QTest.mouseClick(waveform, left, pos=click_pos)
    fixture.app.processEvents()
    after_click = waveform.position_seconds
    offset_after_click = waveform.playhead_offset_fraction
    _expect(after_click > before_click, f"Clicking right of the playhead moved it from {before_click} to {after_click}")
    _expect(
        offset_after_click > 0.2,
        f"Clicking right of center left playhead_offset_fraction={offset_after_click:.4f}, expected > 0.2",
    )
    _expect(
        abs(waveform.visible_range[0] - range_before_click[0]) < 1e-6,
        f"Clicking waveform moved visible_range from {range_before_click} to {waveform.visible_range}",
    )
    grab = waveform.grab().toImage()
    dpr = grab.devicePixelRatio()
    pixels = _image_pixels(grab)
    sample_row = pixels[round(click_pos.y() * dpr)]
    # `QImage.Format_ARGB32` on little-endian stores BGRA bytes: red playhead rgb(255, 89, 94) -> B=94, G=89, R=255.
    red_xs = np.flatnonzero(
        (np.abs(sample_row[:, 2].astype(int) - 255) <= _COLOR_TOLERANCE)
        & (np.abs(sample_row[:, 1].astype(int) - 89) <= _COLOR_TOLERANCE)
        & (np.abs(sample_row[:, 0].astype(int) - 94) <= _COLOR_TOLERANCE)
    )
    expected_playhead_px = click_pos.x() * dpr
    _expect(
        red_xs.size > 0 and abs(float(np.mean(red_xs)) - expected_playhead_px) <= max(3.0, 2.0 * dpr),
        f"Playhead rendered at device x={red_xs.tolist()}, expected near {expected_playhead_px:.1f}",
    )
    _expect(
        view.selected_rows() == (1,) and view.currentIndex().row() == 1,
        f"Clicking waveform at {after_click:.3f} s selected {view.selected_rows()}, expected (1,)",
    )
    _settle(fixture.app, 0.2)
    _expect(
        abs(waveform.position_seconds - after_click) < 1e-6
        and abs(waveform.playhead_offset_fraction - offset_after_click) < 1e-6
        and abs(waveform.visible_range[0] - range_before_click[0]) < 1e-6,
        "Playhead or waveform moved while the video was paused",
    )

    # Fine-stepping while paused moves the playhead across the stationary waveform.
    QtTest.QTest.keyClick(waveform, QtCore.Qt.Key.Key_Left, QtCore.Qt.KeyboardModifier.ShiftModifier)
    fixture.app.processEvents()
    _expect(
        abs(waveform.position_seconds - (after_click - 0.1)) < 1e-2
        and abs(waveform.visible_range[0] - range_before_click[0]) < 1e-6,
        f"Paused Shift+Left step changed visible_range from {range_before_click} to {waveform.visible_range}",
    )
    after_step = waveform.position_seconds

    # 2. Pressing down to start a drag must NOT jump the playhead; dragging right pans the playhead earlier from
    # `after_step` across 6.75 s into [2.5, 6.75), highlighting row 0 (and ignoring second_clip's 3.25 s row 2);
    # dragging left past `anchor` highlights row 1 again, and releasing after a drag does not jump either.
    anchor = point_at(4.0)
    QtTest.QTest.mousePress(waveform, left, pos=anchor)
    fixture.app.processEvents()
    _expect(
        abs(waveform.position_seconds - after_step) < 1e-6 and view.selected_rows() == (1,),
        f"Pressing down on waveform jumped playhead from {after_step:.3f} s to {waveform.position_seconds:.3f} s",
    )
    pan_earlier_px = max(
        QtWidgets.QApplication.startDragDistance() + 4,
        round(2.5 / waveform.window_seconds * waveform.width()),
    )
    pan_later_px = max(
        QtWidgets.QApplication.startDragDistance() + 4,
        round(1.0 / waveform.window_seconds * waveform.width()),
    )
    QtTest.QTest.mouseMove(waveform, anchor + QtCore.QPoint(pan_earlier_px, 0))
    fixture.app.processEvents()
    after_drag_right = waveform.position_seconds
    _expect(
        after_drag_right < after_step,
        f"Dragging right moved the playhead from {after_step} to {after_drag_right}; it should move earlier",
    )
    _expect(
        view.selected_rows() == (0,),
        f"Dragging waveform to {after_drag_right:.3f} s selected {view.selected_rows()}, expected (0,)",
    )
    QtTest.QTest.mouseMove(waveform, anchor - QtCore.QPoint(pan_later_px, 0))
    fixture.app.processEvents()
    after_drag_left = waveform.position_seconds
    _expect(
        after_drag_left > after_step,
        f"Dragging left moved the playhead from {after_step} to {after_drag_left}; it should move later",
    )
    _expect(
        view.selected_rows() == (1,),
        f"Dragging waveform to {after_drag_left:.3f} s selected {view.selected_rows()}, expected (1,)",
    )
    QtTest.QTest.mouseRelease(waveform, left, pos=anchor - QtCore.QPoint(pan_later_px, 0))
    fixture.app.processEvents()
    _expect(
        abs(waveform.position_seconds - after_drag_left) < 1e-6
        and abs(player.position_seconds() - after_drag_left) < 1e-6,
        "Releasing after dragging jumped the playhead or desynchronized the player",
    )

    # Clicking inside the waveform-highlighted row (row 1) to edit it must not jump back to 6.75 s.
    QtTest.QTest.mouseClick(view.viewport(), left, pos=view.visualRect(model.index(1, int(col.LEVEL))).center())
    fixture.app.processEvents()
    _expect(
        abs(player.position_seconds() - after_drag_left) < 1e-6,
        f"Clicking inside waveform-highlighted row 1 moved the playhead to {player.position_seconds():.3f} s",
    )

    # 3. Clicking between 2.5 s and 6.75 s highlights row 0; clicking before 2.5 s clears the selection, and clicking
    # row 0 in the table afterward jumps back to 2.5 s and recenters the playhead.
    QtTest.QTest.mouseClick(waveform, left, pos=point_at(4.5))
    fixture.app.processEvents()
    _expect(view.selected_rows() == (0,), f"Clicking waveform at ~4.5 s selected {view.selected_rows()}, expected (0,)")
    QtTest.QTest.mouseClick(waveform, left, pos=point_at(1.0))
    fixture.app.processEvents()
    _expect(
        waveform.position_seconds < _FIRST_ROW_SECONDS and view.selected_rows() == (),
        f"Clicking waveform before the first event ({waveform.position_seconds:.3f} s) left selected_rows="
        f"{view.selected_rows()}",
    )
    QtTest.QTest.mouseClick(view.viewport(), left, pos=view.visualRect(model.index(0, int(col.LEVEL))).center())
    fixture.app.processEvents()
    _expect(
        view.selected_rows() == (0,)
        and abs(player.position_seconds() - _FIRST_ROW_SECONDS) < 1e-3
        and waveform.playhead_offset_fraction == 0.0,
        f"Clicking row 0 after clearing selection left playhead at {player.position_seconds():.3f} s,"
        f" offset={waveform.playhead_offset_fraction:.4f}",
    )

    # 4. Clicking left of center and playing drifts the playhead back to center over 5 s (freezing when paused
    # mid-drift, and scrolling the waveform more slowly than 1x while catching up).
    waveform.set_window_seconds(4.0)
    QtTest.QTest.mouseClick(waveform, left, pos=point_at(1.2))
    fixture.app.processEvents()
    drift_click_sec = waveform.position_seconds
    drift_start_offset = waveform.playhead_offset_fraction
    drift_initial_window_start = waveform.visible_range[0]
    _expect(
        drift_start_offset < -0.2,
        f"Clicking left of center gave offset={drift_start_offset:.4f}, expected < -0.2",
    )
    player.toggle_play_pause()
    _wait_until(fixture.app, lambda: player.is_playing, _PLAYBACK_TIMEOUT_SECONDS, "playback to start for drift check")
    _wait_until(
        fixture.app,
        lambda: player.position_seconds() >= drift_click_sec + 1.5,
        _PLAYBACK_TIMEOUT_SECONDS,
        "~1.5 s of playback during recentering drift",
    )
    player.pause()
    _wait_until(fixture.app, lambda: not player.is_playing, _PLAYBACK_TIMEOUT_SECONDS, "playback to pause mid-drift")
    mid_offset = waveform.playhead_offset_fraction
    mid_range = waveform.visible_range
    elapsed_media = waveform.position_seconds - drift_click_sec
    scrolled_window = mid_range[0] - drift_initial_window_start
    _expect(
        drift_start_offset < mid_offset < -0.02,
        f"Mid-drift offset={mid_offset:.4f}, expected strictly between {drift_start_offset:.4f} and -0.02",
    )
    _expect(
        scrolled_window < elapsed_media,
        f"Left-of-center drift scrolled waveform by {scrolled_window:.3f} s over {elapsed_media:.3f} s of playback;"
        " expected slower than 1x",
    )
    _settle(fixture.app, 0.2)
    _expect(
        abs(waveform.playhead_offset_fraction - mid_offset) < 1e-6 and waveform.visible_range == mid_range,
        "Mid-drift pause did not freeze playhead offset and waveform",
    )
    player.toggle_play_pause()
    _wait_until(
        fixture.app,
        lambda: player.position_seconds() >= drift_click_sec + 5.2,
        10.0,
        "remaining playback to complete the 5 s recentering drift",
    )
    player.pause()
    _wait_until(fixture.app, lambda: not player.is_playing, _PLAYBACK_TIMEOUT_SECONDS, "playback to pause after drift")
    _expect(
        waveform.playhead_offset_fraction == 0.0,
        f"Playhead offset after >5 s of playback is {waveform.playhead_offset_fraction:.6f}, expected 0.0",
    )
    waveform.reset_zoom()

    # 5. Holding Alt while clicking the waveform moves the playhead without changing the selected row.
    QtTest.QTest.mouseClick(view.viewport(), left, pos=view.visualRect(model.index(0, int(col.LEVEL))).center())
    fixture.app.processEvents()
    QtTest.QTest.mouseClick(waveform, left, _ALT, pos=point_at(7.5))
    fixture.app.processEvents()
    _release_alt(fixture)
    _expect(
        waveform.position_seconds > _SECOND_ROW_SECONDS and view.selected_rows() == (0,),
        f"Alt+clicking waveform at {waveform.position_seconds:.3f} s changed selected_rows to {view.selected_rows()}",
    )


def _image_pixels(image: QtGui.QImage) -> np.ndarray:
    """Returns the pixels of `image` as an `(height, width, 4)` `uint8` array."""
    argb = image.convertToFormat(QtGui.QImage.Format.Format_ARGB32)
    rows = np.frombuffer(argb.constBits(), dtype=np.uint8).reshape(argb.height(), argb.bytesPerLine())
    return rows[:, : argb.width() * 4].reshape(argb.height(), argb.width(), 4).copy()


def _check_playback_follows_rows(fixture: _Fixture) -> None:
    """Playback highlights the row whose `Inpoint` it crosses, unless editing, multi-selecting, or holding `Alt`."""
    window = fixture.window
    waveform = window.waveform
    view = window.table_view
    viewport = view.viewport()
    model = window.table_model
    player = window.player
    col = ballroom_scribe.gui.EventTableModel.Column
    left = QtCore.Qt.MouseButton.LeftButton
    editing = QtWidgets.QAbstractItemView.State.EditingState
    _expect(waveform.envelope is not None, "Waveform is not loaded")

    def play_until(target_seconds: float, context: str) -> None:
        """Plays from the current position until the playhead passes `target_seconds`, then pauses."""
        player.toggle_play_pause()
        _wait_until(fixture.app, lambda: player.is_playing, _PLAYBACK_TIMEOUT_SECONDS, f"playback to start {context}")
        _wait_until(
            fixture.app,
            lambda: player.position_seconds() >= target_seconds,
            _PLAYBACK_TIMEOUT_SECONDS,
            f"the playhead to pass {target_seconds:.2f} s {context}",
        )
        player.pause()
        _wait_until(
            fixture.app, lambda: not player.is_playing, _PLAYBACK_TIMEOUT_SECONDS, f"playback to pause {context}"
        )
        _settle(fixture.app, 0.1)  # Lets any trailing position report from the backend land before asserting.

    # 1. With no row selected (after clicking the waveform before the first event), playing across row 0's 2.5 s
    # Inpoint highlights row 0 without moving the playhead.
    view.setCurrentIndex(model.index(0, 0))
    fixture.app.processEvents()
    window.go_to_selected_event()
    _expect_source(fixture, fixture.clip, "after going to row 0")
    QtTest.QTest.mouseClick(waveform, left, pos=_waveform_point(waveform, 1.0))
    fixture.app.processEvents()
    _expect(
        view.selected_rows() == () and player.position_seconds() < _FIRST_ROW_SECONDS,
        f"Clicking the waveform at 1 s left selected_rows={view.selected_rows()} at {player.position_seconds():.3f} s",
    )
    play_until(_FIRST_ROW_SECONDS + 0.5, "across row 0's Inpoint")
    _expect(
        view.selected_rows() == (0,) and view.currentIndex().row() == 0,
        f"Playing across {_FIRST_ROW_SECONDS} s selected {view.selected_rows()}, expected (0,)",
    )
    _expect(
        player.position_seconds() >= _FIRST_ROW_SECONDS,
        f"Following row 0 moved the playhead back to {player.position_seconds():.3f} s",
    )

    # 2. A paused seek leaves the selection alone; playing across row 1's 6.75 s Inpoint highlights row 1, which
    # becomes the row the user is on (clicking inside it does not seek).
    player.seek(6.0)
    fixture.app.processEvents()
    _expect(view.selected_rows() == (0,), f"Seeking while paused changed selected_rows to {view.selected_rows()}")
    play_until(_SECOND_ROW_SECONDS + 0.25, "across row 1's Inpoint")
    _expect(
        view.selected_rows() == (1,) and view.currentIndex().row() == 1,
        f"Playing across {_SECOND_ROW_SECONDS} s selected {view.selected_rows()}, expected (1,)",
    )
    paused_at = player.position_seconds()
    _expect(paused_at >= _SECOND_ROW_SECONDS, f"Following row 1 moved the playhead back to {paused_at:.3f} s")
    QtTest.QTest.mouseClick(viewport, left, pos=view.visualRect(model.index(1, int(col.LEVEL))).center())
    fixture.app.processEvents()
    _expect(
        abs(player.position_seconds() - paused_at) < 0.1,
        f"Clicking inside the followed row moved the playhead to {player.position_seconds():.3f} s",
    )

    # 3. An open cell editor suppresses following, and the missed crossing is not applied once the editor closes.
    view.setCurrentIndex(model.index(0, int(col.LEVEL)))
    fixture.app.processEvents()
    player.seek(6.0)
    view.edit(model.index(0, int(col.LEVEL)))
    fixture.app.processEvents()
    _expect(
        view.state() == editing and view.selected_rows() == (0,),
        f"Could not open a cell editor on row 0 (state {view.state().name}, selected {view.selected_rows()})",
    )
    play_until(_SECOND_ROW_SECONDS + 0.25, "with a cell editor open")
    _expect(
        view.selected_rows() == (0,) and view.state() == editing,
        f"Playing with an open editor selected {view.selected_rows()} (state {view.state().name})",
    )
    _cancel_cell_editor(fixture)
    _expect(view.state() != editing, "Esc did not close the editor")
    play_until(player.position_seconds() + 0.2, "after closing the editor")
    _expect(view.selected_rows() == (0,), f"Closing the editor belatedly selected {view.selected_rows()}")

    # 4. Two or more selected rows suppress following.
    player.seek(6.0)
    view.selectionModel().select(
        model.index(1, 0),
        QtCore.QItemSelectionModel.SelectionFlag.Select | QtCore.QItemSelectionModel.SelectionFlag.Rows,
    )
    fixture.app.processEvents()
    _expect(view.selected_rows() == (0, 1), f"Could not select rows 0 and 1: {view.selected_rows()}")
    play_until(_SECOND_ROW_SECONDS + 0.25, "with two rows selected")
    _expect(view.selected_rows() == (0, 1), f"Playing with two rows selected changed them to {view.selected_rows()}")

    # 5. Holding Alt suppresses following (an Alt+click on the waveform seeks and leaves Alt held), and the missed
    # crossing is not applied once Alt is released.
    view.setCurrentIndex(model.index(0, 0))
    fixture.app.processEvents()
    _expect(view.selected_rows() == (0,), f"Could not restore a single selected row: {view.selected_rows()}")
    player.seek(6.0)
    QtTest.QTest.mouseClick(waveform, left, _ALT, pos=_waveform_point(waveform, 6.0))
    fixture.app.processEvents()
    _expect(
        abs(player.position_seconds() - 6.0) < 0.1 and view.selected_rows() == (0,),
        f"Alt+clicking the waveform at 6 s left the playhead at {player.position_seconds():.3f} s with"
        f" selected_rows={view.selected_rows()}",
    )
    play_until(_SECOND_ROW_SECONDS + 0.25, "with Alt held")
    _release_alt(fixture)
    _expect(view.selected_rows() == (0,), f"Playing with Alt held selected {view.selected_rows()}")
    play_until(player.position_seconds() + 0.2, "after releasing Alt")
    _expect(view.selected_rows() == (0,), f"Releasing Alt belatedly selected {view.selected_rows()}")


def _check_focused_waveform_paints_interior(fixture: _Fixture) -> None:
    window = fixture.window
    _focus_table(fixture)  # The previous step's click left the waveform focused; start from the unfocused look.
    unfocused = window.waveform.grab().toImage()
    window.activateWindow()
    window.waveform.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
    fixture.app.processEvents()
    if not window.waveform.hasFocus():
        _expect(fixture.windowed, "Could not focus the waveform widget")
        print("  (skipped: the window manager did not grant focus)")
        return
    focused = window.waveform.grab().toImage()
    window.table_view.setFocus()
    fixture.app.processEvents()

    def grabs_summary() -> str:
        unfocused_path = fixture.work_dir / "waveform_unfocused.png"
        focused_path = fixture.work_dir / "waveform_focused.png"
        unfocused.save(str(unfocused_path))
        focused.save(str(focused_path))
        return (
            f"{focused.width()}x{focused.height()} grab at device pixel ratio {focused.devicePixelRatio():g};"
            f" compare {unfocused_path} with {focused_path}"
        )

    if focused.size() != unfocused.size():
        raise _SmokeTestError(f"Waveform grab size changed while focusing: {unfocused.size()} -> {grabs_summary()}")
    changed = np.any(_image_pixels(unfocused) != _image_pixels(focused), axis=2)
    if not changed.any():
        raise _SmokeTestError(f"Focused waveform did not draw a focus frame: {grabs_summary()}")
    # The 1 px focus frame spans `devicePixelRatio` device pixels per edge; nothing inside it may change.
    margin = math.ceil(2 * focused.devicePixelRatio())
    interior = changed[margin:-margin, margin:-margin]
    if interior.any():
        ys, xs = np.nonzero(interior)
        raise _SmokeTestError(
            f"Focused waveform repainted {int(interior.sum())} interior pixel(s) within"
            f" x={xs.min() + margin}..{xs.max() + margin}, y={ys.min() + margin}..{ys.max() + margin}"
            f" (focus frame leaking a fill?): {grabs_summary()}"
        )


def _popup_items(completer: QtWidgets.QCompleter) -> list[str]:
    popup_model = completer.completionModel()
    return [str(popup_model.data(popup_model.index(row, 0))) for row in range(popup_model.rowCount())]


def _check_autocomplete_from_other_rows(fixture: _Fixture) -> None:
    window = fixture.window
    model = window.table_model
    col = ballroom_scribe.gui.EventTableModel.Column
    if not _focus_table(fixture):
        return
    original_level = model.record_at(0).level
    original_round = model.record_at(0).round
    level_index = model.index(0, int(col.LEVEL))
    window.table_view.setCurrentIndex(level_index)
    window.table_view.edit(level_index)
    _settle(fixture.app, 0.1)
    editor = fixture.app.focusWidget()
    _expect(isinstance(editor, QtWidgets.QLineEdit), f"Cell editor is {type(editor).__name__}, expected a line edit")
    assert isinstance(editor, QtWidgets.QLineEdit)
    completer = editor.completer()
    _expect(completer is not None, "Cell editor has no completer")
    assert completer is not None

    # The full list of the other rows' values drops down without typing.
    _expect(completer.popup().isVisible(), "Suggestion list did not open when the cell editor appeared")
    _expect(
        _popup_items(completer) == ["Gold", "Silver"],
        f"Expected ['Gold', 'Silver'] from the other rows, got {_popup_items(completer)}",
    )

    # Typing filters it case-insensitively; Down + Enter picks the highlighted value and commits in one go.
    QtTest.QTest.keyClicks(editor, "sil")
    _settle(fixture.app, 0.1)
    _expect(_popup_items(completer) == ["Silver"], f"Typing 'sil' left suggestions {_popup_items(completer)}")
    QtTest.QTest.keyClick(completer.popup(), QtCore.Qt.Key.Key_Down)
    QtTest.QTest.keyClick(completer.popup(), QtCore.Qt.Key.Key_Return)
    _settle(fixture.app, 0.1)
    _expect(model.record_at(0).level == "Silver", f"Picked suggestion was not committed: {model.record_at(0).level!r}")
    _expect(
        window.table_view.state() != QtWidgets.QAbstractItemView.State.EditingState,
        "Cell editor stayed open after committing a suggestion",
    )

    # Tab with the list open still moves editing to the next cell.
    if not _focus_table(fixture):
        return
    window.table_view.edit(level_index)
    _settle(fixture.app, 0.1)
    editor = fixture.app.focusWidget()
    assert isinstance(editor, QtWidgets.QLineEdit) and editor.completer() is not None
    QtTest.QTest.keyClick(editor.completer().popup(), QtCore.Qt.Key.Key_Tab)
    _settle(fixture.app, 0.1)
    _expect(
        window.table_view.state() == QtWidgets.QAbstractItemView.State.EditingState
        and window.table_view.currentIndex().column() == int(col.STYLE),
        "Tab from the Level editor did not move editing to Style",
    )
    _cancel_cell_editor(fixture)

    # A mouse pick commits and moves editing to the next column; in the last column it just commits.
    if not _focus_table(fixture):
        return
    window.table_view.setCurrentIndex(level_index)
    window.table_view.edit(level_index)
    _settle(fixture.app, 0.1)
    editor = fixture.app.focusWidget()
    _expect(
        isinstance(editor, QtWidgets.QLineEdit) and editor.completer() is not None,
        f"Level editor for the mouse-pick check is {type(editor).__name__}",
    )
    assert isinstance(editor, QtWidgets.QLineEdit) and editor.completer() is not None
    popup = editor.completer().popup()
    _expect(popup.isVisible(), "Suggestion list did not open for the mouse-pick check")
    first_item = popup.visualRect(popup.model().index(0, 0)).center()
    QtTest.QTest.mouseClick(popup.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=first_item)
    _settle(fixture.app, 0.1)
    _expect(model.record_at(0).level == "Gold", f"Mouse pick was not committed: {model.record_at(0).level!r}")
    _expect(
        window.table_view.state() == QtWidgets.QAbstractItemView.State.EditingState
        and window.table_view.currentIndex().column() == int(col.STYLE),
        "Picking a Level with the mouse did not move editing to Style",
    )
    _cancel_cell_editor(fixture)
    if not _focus_table(fixture):
        return
    round_index = model.index(0, int(col.ROUND))
    window.table_view.setCurrentIndex(round_index)
    window.table_view.edit(round_index)
    _settle(fixture.app, 0.1)
    editor = fixture.app.focusWidget()
    _expect(
        isinstance(editor, QtWidgets.QLineEdit) and editor.completer() is not None,
        f"Round editor for the mouse-pick check is {type(editor).__name__}",
    )
    assert isinstance(editor, QtWidgets.QLineEdit) and editor.completer() is not None
    popup = editor.completer().popup()
    first_item = popup.visualRect(popup.model().index(0, 0)).center()
    QtTest.QTest.mouseClick(popup.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=first_item)
    _settle(fixture.app, 0.1)
    _expect(
        window.table_view.state() != QtWidgets.QAbstractItemView.State.EditingState,
        "Picking a Round (last column) with the mouse should commit without opening another editor",
    )
    model.setData(round_index, original_round)

    model.setData(level_index, original_level)


def _check_editing_commands(fixture: _Fixture) -> None:
    window = fixture.window
    model = window.table_model
    view = window.table_view
    col = ballroom_scribe.gui.EventTableModel.Column
    clip = fixture.clip.name
    second_clip = fixture.second_clip.name
    editing = QtWidgets.QAbstractItemView.State.EditingState

    def rows() -> list[tuple[str, float]]:
        return [(record.file.name, record.inpoint_seconds) for record in model.records]

    view.setCurrentIndex(model.index(0, 0))
    fixture.app.processEvents()
    window.player.seek(4.125)
    window.set_selected_inpoint_to_playhead()
    _expect(abs(model.record_at(0).inpoint_seconds - 4.125) < 1e-6, "Set Inpoint did not update the inpoint")
    _expect(model.record_at(0).inpoint_formatted == "0:00:04.12", "Set Inpoint did not update Inpoint Formatted")
    _expect(not model.undo_stack.isClean(), "Set Inpoint did not mark the document modified")

    _expect(model.setData(model.index(1, int(col.INPOINT)), "0:00:07.50"), "Typing an inpoint was rejected")
    _expect(abs(model.record_at(1).inpoint_seconds - 7.5) < 1e-6, "Typed inpoint was not parsed")
    _expect(rows() == [(clip, 4.125), (clip, 7.5), (second_clip, 3.25)], f"Unexpected rows before adding: {rows()}")

    # Add Event places the row among the active video's rows in inpoint order, not below the selected row 1.
    window.player.seek(8.25)
    window.add_event_at_playhead()
    _expect(model.rowCount() == _INITIAL_ROW_COUNT + 1, "Add Event did not insert a row")
    _expect(
        rows() == [(clip, 4.125), (clip, 7.5), (clip, 8.25), (second_clip, 3.25)],
        f"Add Event at 8.25 s did not place the row after the video's events: {rows()}",
    )
    _expect(view.currentIndex().row() == 2 and view.state() == editing, "The new event is not open for editing")
    _cancel_cell_editor(fixture)
    window.player.seek(5.0)
    window.add_event_at_playhead()
    _expect(
        rows()[:3] == [(clip, 4.125), (clip, 5.0), (clip, 7.5)], f"Add Event at 5 s went to the wrong row: {rows()}"
    )
    _cancel_cell_editor(fixture)
    model.undo_stack.undo()
    _expect(model.rowCount() == _INITIAL_ROW_COUNT + 1, "Undo did not remove the added event")

    # An event for the other video goes into that video's block (selecting its row makes it the active video).
    view.setCurrentIndex(model.index(3, 0))
    fixture.app.processEvents()
    _expect_source(fixture, fixture.second_clip, "after selecting the second video's row")
    window.player.seek(1.0)
    window.add_event_at_playhead()
    _expect(
        rows() == [(clip, 4.125), (clip, 7.5), (clip, 8.25), (second_clip, 1.0), (second_clip, 3.25)],
        f"Add Event for the second video did not join its block: {rows()}",
    )
    _cancel_cell_editor(fixture)
    model.undo_stack.undo()

    # Set Inpoint moves the row to keep the video's events in order; the selection follows it.
    view.setCurrentIndex(model.index(0, 0))
    fixture.app.processEvents()
    _expect_source(fixture, fixture.clip, "after selecting the first video's row again")
    window.player.seek(9.0)
    window.set_selected_inpoint_to_playhead()
    _expect(
        rows() == [(clip, 7.5), (clip, 8.25), (clip, 9.0), (second_clip, 3.25)],
        f"Set Inpoint did not move the row into inpoint order: {rows()}",
    )
    _expect(view.currentIndex().row() == 2, "The selection did not follow the row moved by Set Inpoint")
    _expect(model.undo_stack.undoText() == "Set Inpoint", f"Set Inpoint undo text is {model.undo_stack.undoText()!r}")
    model.undo_stack.undo()
    _expect(
        rows()[0] == (clip, 4.125) and view.currentIndex().row() == 0,
        f"Undo did not move the row back to the top: {rows()}, current row {view.currentIndex().row()}",
    )

    # A typed inpoint moves the row when it is committed, not while it is being typed.
    _focus_table(fixture)
    inpoint_index = model.index(1, int(col.INPOINT))
    view.setCurrentIndex(inpoint_index)
    view.edit(inpoint_index)
    fixture.app.processEvents()
    editor = _open_cell_editor(fixture)
    _expect(isinstance(editor, QtWidgets.QLineEdit), f"Inpoint editor is {type(editor).__name__}, expected a line edit")
    assert isinstance(editor, QtWidgets.QLineEdit)
    editor.selectAll()
    QtTest.QTest.keyClicks(editor, "1.5")
    fixture.app.processEvents()
    _expect(rows()[1] == (clip, 7.5), f"The row moved while its inpoint was still being typed: {rows()}")
    QtTest.QTest.keyClick(editor, QtCore.Qt.Key.Key_Return)
    fixture.app.processEvents()
    _expect(
        rows() == [(clip, 1.5), (clip, 4.125), (clip, 8.25), (second_clip, 3.25)],
        f"Committing the typed inpoint did not move the row: {rows()}",
    )
    _expect(view.currentIndex().row() == 0 and view.state() != editing, "The current row did not follow the commit")
    model.undo_stack.undo()
    _expect(rows()[1] == (clip, 7.5), f"Undo did not move the row back: {rows()}")

    # Deleting the selected event moves the selection on, but leaves the playhead alone.
    view.setCurrentIndex(model.index(2, 0))
    fixture.app.processEvents()
    playhead_before_delete = window.player.position_seconds()
    source_before_delete = window.player.source_path
    window.delete_selected_events()
    fixture.app.processEvents()
    _expect(model.rowCount() == _INITIAL_ROW_COUNT, "Delete Event did not remove the row")
    # The selection moves on to the next row (which belongs to the other video), but the playhead stays put.
    _expect(
        view.currentIndex().row() == 2 and view.selected_rows() == (2,),
        f"Delete Event did not select the next row: current {view.currentIndex().row()}, selected {view.selected_rows()}",
    )
    _expect(
        window.player.source_path == source_before_delete
        and abs(window.player.position_seconds() - playhead_before_delete) < 1e-6,
        "Deleting an event moved the playhead to the next event",
    )
    # The user is now on that row, so clicking inside it does not seek either, even after stamping the playhead onto
    # it with Set Inpoint and moving the playhead on (previously the first click jumped back to its inpoint).
    window.set_selected_inpoint_to_playhead()
    _expect(
        rows()[2] == (clip, playhead_before_delete),
        f"Set Inpoint after Delete Event did not stamp the playhead onto the selected row: {rows()}",
    )
    window.player.seek(6.0)
    QtTest.QTest.mouseClick(
        view.viewport(),
        QtCore.Qt.MouseButton.LeftButton,
        pos=view.visualRect(model.index(2, int(col.LEVEL))).center(),
    )
    fixture.app.processEvents()
    _expect(
        window.player.source_path == source_before_delete and abs(window.player.position_seconds() - 6.0) < 1e-6,
        f"Clicking inside the row selected after Delete Event moved the playhead to"
        f" {window.player.position_seconds():.3f} s",
    )
    model.undo_stack.undo()  # Set Inpoint
    model.undo_stack.undo()  # Delete Event
    _expect(model.rowCount() == _INITIAL_ROW_COUNT + 1, "Undo did not restore the deleted row")
    model.undo_stack.redo()
    _expect(model.rowCount() == _INITIAL_ROW_COUNT, "Redo did not delete the row again")


def _check_keyboard_shortcuts(fixture: _Fixture) -> None:
    window = fixture.window
    model = window.table_model
    player = window.player
    col = ballroom_scribe.gui.EventTableModel.Column
    key = QtCore.Qt.Key
    if not _focus_table(fixture):
        return

    _press(fixture, key.Key_K, _CTRL)
    _wait_until(
        fixture.app,
        lambda: player.playback_state == QtMultimedia.QMediaPlayer.PlaybackState.PlayingState,
        _PLAYBACK_TIMEOUT_SECONDS,
        "Ctrl+K to start playback",
    )
    _press(fixture, key.Key_K, _CTRL)
    _wait_until(
        fixture.app,
        lambda: player.playback_state == QtMultimedia.QMediaPlayer.PlaybackState.PausedState,
        _PLAYBACK_TIMEOUT_SECONDS,
        "Ctrl+K to pause playback",
    )

    player.seek(1.0)
    _press(fixture, key.Key_L, _CTRL)
    _expect(
        abs(player.position_seconds() - _CLIP_SECONDS) < 1e-3,
        f"Ctrl+L moved the playhead to {player.position_seconds():.3f} s, expected {_CLIP_SECONDS} (clamped)",
    )
    _press(fixture, key.Key_J, _CTRL)
    _expect(abs(player.position_seconds()) < 1e-3, f"Ctrl+J moved the playhead to {player.position_seconds():.3f} s")

    window.table_view.setCurrentIndex(model.index(0, 0))
    fixture.app.processEvents()
    row_count = model.rowCount()
    player.seek(5.0)
    _press(fixture, key.Key_E, _CTRL)
    _expect(model.rowCount() == row_count + 1, "Ctrl+E did not add an event")
    _expect(abs(model.record_at(1).inpoint_seconds - 5.0) < 1e-6, "Ctrl+E did not add the event at the playhead")
    _cancel_cell_editor(fixture)
    window.table_view.setCurrentIndex(model.index(1, 0))
    player.seek(5.5)
    _press(fixture, key.Key_I, _CTRL)
    _expect(abs(model.record_at(1).inpoint_seconds - 5.5) < 1e-6, "Ctrl+I did not stamp the playhead onto the row")
    _press(fixture, key.Key_D, _CTRL)
    _expect(model.rowCount() == row_count, "Ctrl+D did not delete the event")

    for edit_key in (key.Key_Return, key.Key_F2):
        window.table_view.setCurrentIndex(model.index(0, int(col.LEVEL)))
        window.table_view.setFocus()
        fixture.app.processEvents()
        _press(fixture, edit_key)
        _expect(
            window.table_view.state() == QtWidgets.QAbstractItemView.State.EditingState
            and isinstance(fixture.app.focusWidget(), QtWidgets.QLineEdit),
            f"{QtGui.QKeySequence(edit_key).toString()} did not open the cell editor",
        )
        _cancel_cell_editor(fixture)
        _expect(
            window.table_view.state() != QtWidgets.QAbstractItemView.State.EditingState, "Esc did not close the editor"
        )


def _check_spreadsheet_typing_and_clear(fixture: _Fixture) -> None:
    """Typing a letter or digit edits the current cell; `Space` still plays; `Delete` and `Backspace` clear cells."""
    window = fixture.window
    view = window.table_view
    model = window.table_model
    player = window.player
    col = ballroom_scribe.gui.EventTableModel.Column
    key = QtCore.Qt.Key
    editing = QtWidgets.QAbstractItemView.State.EditingState
    if not _focus_table(fixture):
        return

    def undo_state() -> tuple[int, int]:
        return model.undo_stack.count(), model.undo_stack.index()

    # 1. Typing a letter on Style edits the current cell (no row jump) and keeps the popup filtered after settling.
    style_index = model.index(0, int(col.STYLE))
    view.setCurrentIndex(style_index)
    fixture.app.processEvents()
    player.seek(3.0)
    _press(fixture, key.Key_S)
    _expect(view.currentIndex() == style_index, "Typing 's' jumped away from the current cell")
    _expect(abs(player.position_seconds() - 3.0) < 1e-3, "Typing 's' moved the playhead")
    editor = _open_cell_editor(fixture)
    _expect(
        isinstance(editor, QtWidgets.QLineEdit) and editor.text() == "s",
        f"Typing 's' on Style opened {type(editor).__name__} with text {getattr(editor, 'text', lambda: None)()!r}",
    )
    assert isinstance(editor, QtWidgets.QLineEdit)
    _settle(fixture.app, 0.1)
    completer = editor.completer()
    _expect(
        completer is not None and completer.popup().isVisible() and _popup_items(completer) == ["Smooth"],
        f"Typing 's' did not keep the suggestion list filtered to ['Smooth']: "
        f"{_popup_items(completer) if completer is not None else None}",
    )
    _cancel_cell_editor(fixture)
    _expect(model.record_at(0).style == "Standard", "Esc after typing 's' changed Style")

    # 2. Bare K, J, L, Shift+J, Shift+L, and I type into the focused table instead of firing shortcuts.
    level_index = model.index(0, int(col.LEVEL))
    view.setCurrentIndex(level_index)
    fixture.app.processEvents()
    player.seek(3.0)
    inpoint_before = model.record_at(0).inpoint_seconds
    for expected_text in ("k", "j", "l", "J", "L", "i"):
        if not _focus_table(fixture):
            return
        QtTest.QTest.keyClicks(view, expected_text)
        fixture.app.processEvents()
        editor = _open_cell_editor(fixture)
        _expect(
            isinstance(editor, QtWidgets.QLineEdit) and editor.text() == expected_text,
            f"Typing {expected_text!r} on Level did not open the editor with {expected_text!r}",
        )
        _expect_paused(fixture, f"after typing {expected_text!r} in the table")
        _expect(
            abs(player.position_seconds() - 3.0) < 1e-3,
            f"Typing {expected_text!r} in the table moved the playhead to {player.position_seconds():.3f} s",
        )
        _expect(
            abs(model.record_at(0).inpoint_seconds - inpoint_before) < 1e-6,
            f"Typing {expected_text!r} in the table changed the row's Inpoint",
        )
        _cancel_cell_editor(fixture)

    # 3. Typing a digit on Inpoint opens its editor with that digit.
    if not _focus_table(fixture):
        return
    inpoint_index = model.index(0, int(col.INPOINT))
    view.setCurrentIndex(inpoint_index)
    fixture.app.processEvents()
    _press(fixture, key.Key_3)
    editor = _open_cell_editor(fixture)
    _expect(
        isinstance(editor, QtWidgets.QLineEdit) and editor.text() == "3",
        "Typing '3' on Inpoint did not open its editor with '3'",
    )
    _cancel_cell_editor(fixture)

    # 4. Typing a letter on a read-only File cell opens no editor, jumps to no row, and triggers no shortcut.
    if not _focus_table(fixture):
        return
    file_index = model.index(0, int(col.FILE))
    view.setCurrentIndex(file_index)
    fixture.app.processEvents()
    player.seek(3.0)
    for readonly_key in (key.Key_S, key.Key_K):
        _press(fixture, readonly_key)
        _expect(
            view.state() != editing and view.currentIndex() == file_index,
            f"Typing {QtGui.QKeySequence(readonly_key).toString()!r} on File opened an editor or moved the selection",
        )
        _expect_paused(fixture, "after typing on a read-only File cell")
        _expect(abs(player.position_seconds() - 3.0) < 1e-3, "Typing on File moved the playhead")

    # 5. Space still plays and pauses while the table has focus, without opening an editor.
    view.setCurrentIndex(level_index)
    fixture.app.processEvents()
    _press(fixture, key.Key_Space)
    _wait_until(
        fixture.app,
        lambda: player.playback_state == QtMultimedia.QMediaPlayer.PlaybackState.PlayingState,
        _PLAYBACK_TIMEOUT_SECONDS,
        "Space to start playback with the table focused",
    )
    _expect(view.state() != editing, "Space opened the cell editor")
    _press(fixture, key.Key_Space)
    _wait_until(
        fixture.app,
        lambda: player.playback_state == QtMultimedia.QMediaPlayer.PlaybackState.PausedState,
        _PLAYBACK_TIMEOUT_SECONDS,
        "Space to pause playback with the table focused",
    )

    # 6. Delete on Round and Backspace on Dance clear the cell (not the event) as one undo step.
    row_count = model.rowCount()
    original_round = model.record_at(0).round
    round_index = model.index(0, int(col.ROUND))
    view.setCurrentIndex(round_index)
    fixture.app.processEvents()
    _press(fixture, key.Key_Delete)
    _expect(model.rowCount() == row_count, "Delete removed the event row instead of clearing the cell")
    _expect(
        model.record_at(0).round == "" and view.state() != editing and model.undo_stack.undoText() == "Clear Round",
        f"Delete on Round left {model.record_at(0).round!r} (undo {model.undo_stack.undoText()!r})",
    )
    model.undo_stack.undo()
    _expect(model.record_at(0).round == original_round, "Undo did not restore the cleared Round cell")

    original_dance = model.record_at(0).dance
    dance_index = model.index(0, int(col.DANCE))
    view.setCurrentIndex(dance_index)
    fixture.app.processEvents()
    _press(fixture, key.Key_Backspace)
    _expect(model.rowCount() == row_count, "Backspace removed the event row instead of clearing the cell")
    _expect(
        model.record_at(0).dance == "" and view.state() != editing and model.undo_stack.undoText() == "Clear Dance",
        f"Backspace on Dance left {model.record_at(0).dance!r} (undo {model.undo_stack.undoText()!r})",
    )
    model.undo_stack.undo()
    _expect(model.record_at(0).dance == original_dance, "Undo did not restore the cleared Dance cell")

    # 7. Delete on Inpoint changes nothing and leaves the undo stack untouched.
    view.setCurrentIndex(inpoint_index)
    fixture.app.processEvents()
    undo_before = undo_state()
    _press(fixture, key.Key_Delete)
    _expect(
        abs(model.record_at(0).inpoint_seconds - inpoint_before) < 1e-6 and undo_state() == undo_before,
        "Delete on Inpoint changed the inpoint or pushed onto the undo stack",
    )

    # 8. Bare L still steps +1 s when the waveform has focus instead of the table.
    player.seek(2.0)
    window.activateWindow()
    window.waveform.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
    fixture.app.processEvents()
    if window.waveform.hasFocus():
        QtTest.QTest.keyClick(window.waveform, key.Key_L)
        fixture.app.processEvents()
        _expect(
            abs(player.position_seconds() - 3.0) < 1e-3,
            f"Bare L on the waveform moved the playhead to {player.position_seconds():.3f} s, expected 3.0 s",
        )
    else:
        _expect(fixture.windowed, "Could not focus the waveform widget")

    _focus_table(fixture)
    view.setCurrentIndex(model.index(0, 0))
    fixture.app.processEvents()


def _action_named(window: ballroom_scribe.gui.MainWindow, text: str) -> QtGui.QAction:
    for action in window.findChildren(QtGui.QAction):
        if action.text() == text:
            return action
    raise _SmokeTestError(f"Could not find QAction {text!r}")


def _check_multi_row_selection(fixture: _Fixture) -> None:
    """Multi-row selection hides single-cell affordances, disables Set Inpoint, and deletes all selected rows at once."""
    window = fixture.window
    view = window.table_view
    viewport = view.viewport()
    model = window.table_model
    player = window.player
    col = ballroom_scribe.gui.EventTableModel.Column
    key = QtCore.Qt.Key
    left = QtCore.Qt.MouseButton.LeftButton
    shift = QtCore.Qt.KeyboardModifier.ShiftModifier
    editing = QtWidgets.QAbstractItemView.State.EditingState
    if not _focus_table(fixture):
        return

    saved_records = list(model.records)
    clip = fixture.clip.resolve()
    second_clip = fixture.second_clip.resolve()
    five_rows = [
        ballroom_scribe.models.EventRecord(
            file=clip,
            inpoint_seconds=1.5,
            level="Bronze",
            style="Standard",
            dance="Waltz",
            round="Quarterfinal",
        ),
        ballroom_scribe.models.EventRecord(
            file=clip,
            inpoint_seconds=4.0,
            level="Bronze",
            style="Standard",
            dance="Waltz",
            round="Semifinal",
        ),
        ballroom_scribe.models.EventRecord(
            file=clip,
            inpoint_seconds=7.5,
            level="Silver",
            style="Smooth",
            dance="Tango",
            round="Final",
        ),
        ballroom_scribe.models.EventRecord(
            file=second_clip,
            inpoint_seconds=2.0,
            level="Gold",
            style="Latin",
            dance="Cha Cha",
            round="Semifinal",
        ),
        ballroom_scribe.models.EventRecord(
            file=second_clip,
            inpoint_seconds=6.0,
            level="Gold",
            style="Latin",
            dance="Rumba",
            round="Final",
        ),
    ]
    model.set_records(five_rows)
    set_inpoint_action = _action_named(window, "Set Inpoint")
    delete_event_action = _action_named(window, "Delete Event")
    highlight = view.palette().color(QtGui.QPalette.ColorGroup.Active, QtGui.QPalette.ColorRole.Highlight)

    def cell_center(row: int, column: ballroom_scribe.gui.EventTableModel.Column = col.LEVEL) -> QtCore.QPoint:
        return view.visualRect(model.index(row, int(column))).center()

    # Sample row 0's plain unselected Style background (row 2 has a different event, so row 0 is untinted).
    view.setCurrentIndex(model.index(2, int(col.LEVEL)))
    fixture.app.processEvents()
    plain_row_0 = _sample_cell(view, viewport.grab().toImage(), model.index(0, int(col.STYLE))).interior

    # 1. Click row 1's Level cell, then Shift+click row 2's Level cell.
    QtTest.QTest.mouseClick(viewport, left, pos=cell_center(1))
    fixture.app.processEvents()
    _expect(view.selected_rows() == (1,), f"Plain click on row 1 selected {view.selected_rows()}")
    _expect(set_inpoint_action.isEnabled() and delete_event_action.isEnabled(), "Actions disabled with 1 row selected")

    QtTest.QTest.mouseClick(viewport, left, shift, pos=cell_center(2))
    fixture.app.processEvents()
    _expect(
        view.selected_rows() == (1, 2) and view.has_multiple_selected_rows and view.selected_row() is None,
        f"Shift+click on row 2 left selected_rows={view.selected_rows()}, selected_row={view.selected_row()}",
    )
    _expect(
        abs(player.position_seconds() - 7.5) < 1e-3,
        f"Shift+click on row 2 left playhead at {player.position_seconds():.3f} s, expected 7.5 s",
    )
    _expect(not set_inpoint_action.isEnabled(), "Set Inpoint stayed enabled with 2 rows selected")
    _expect(delete_event_action.isEnabled(), "Delete Event was disabled with 2 rows selected")

    # 2. No single cell is singled out while several rows are selected.
    _expect(view.fill_handle_rect(view.currentIndex()).isNull(), "Fill handle is still shown with 2 rows selected")
    image = viewport.grab().toImage()
    current_level = _sample_cell(view, image, model.index(2, int(col.LEVEL)))
    mate_style = _sample_cell(view, image, model.index(2, int(col.STYLE)))
    _expect(
        _colors_match(current_level.interior, mate_style.interior),
        f"Current cell ({current_level.interior.name()}) is not painted like its row-mate ({mate_style.interior.name()})"
        " when 2 rows are selected",
    )
    _expect(
        _colors_match(_sample_cell(view, image, model.index(0, int(col.STYLE))).interior, plain_row_0),
        "Unselected row 0 stayed faintly highlighted when multiple rows were selected",
    )
    _press(fixture, key.Key_Delete)
    _press(fixture, key.Key_Backspace)
    _expect(
        model.record_at(2).level == "Silver" and model.undo_stack.isClean(),
        "Delete/Backspace cleared a cell while multiple rows were selected",
    )
    QtTest.QTest.keyClicks(view, "s")
    fixture.app.processEvents()
    _expect(view.state() != editing, "Typing 's' opened the cell editor while multiple rows were selected")
    _press(fixture, key.Key_Return)
    _press(fixture, key.Key_F2)
    _expect(view.state() != editing, "Enter/F2 opened the cell editor while multiple rows were selected")
    QtTest.QTest.keyClicks(view, "k")
    fixture.app.processEvents()
    _expect(view.state() != editing, "Typing 'k' opened the cell editor while multiple rows were selected")
    _expect_paused(fixture, "after typing 'k' with multiple rows selected")

    # A plain click on row 2 collapses the selection back to one row and restores the frame and fill handle.
    QtTest.QTest.mouseClick(viewport, left, pos=cell_center(2))
    fixture.app.processEvents()
    _expect(
        view.selected_rows() == (2,) and not view.has_multiple_selected_rows,
        f"Plain click did not collapse selection to (2,): {view.selected_rows()}",
    )
    _expect(not view.fill_handle_rect(view.currentIndex()).isNull(), "Fill handle did not return for a single row")
    single_image = viewport.grab().toImage()
    _expect(
        not _colors_match(
            _sample_cell(view, single_image, model.index(2, int(col.LEVEL))).interior,
            _sample_cell(view, single_image, model.index(2, int(col.STYLE))).interior,
        ),
        "Current-cell frame did not return after collapsing selection to one row",
    )

    # 3. Click row 1, Shift+click row 2, then Shift+Down extends to row 3 (switching to the second video).
    QtTest.QTest.mouseClick(viewport, left, pos=cell_center(1))
    QtTest.QTest.mouseClick(viewport, left, shift, pos=cell_center(2))
    fixture.app.processEvents()
    _press(fixture, key.Key_Down, shift)
    _expect(view.selected_rows() == (1, 2, 3), f"Shift+Down produced selected_rows={view.selected_rows()}")
    _expect_source(fixture, fixture.second_clip, "after Shift+Down into the second video's row")
    _expect(
        abs(player.position_seconds() - 2.0) < 1e-3,
        f"Shift+Down into row 3 left playhead at {player.position_seconds():.3f} s, expected 2.0 s",
    )

    # 4. Set Inpoint and Ctrl+I do nothing when several rows are selected.
    player.seek(5.25)
    window.set_selected_inpoint_to_playhead()
    _press(fixture, key.Key_I, _CTRL)
    _expect(
        list(model.records) == five_rows and model.undo_stack.isClean(),
        "Set Inpoint or Ctrl+I modified records with 3 rows selected",
    )

    # 5. Ctrl+D deletes all three selected rows as one undo step and selects the row that followed them without seeking.
    playhead_before = player.position_seconds()
    source_before = player.source_path
    _press(fixture, key.Key_D, _CTRL)
    _expect(
        list(model.records) == [five_rows[0], five_rows[4]],
        f"Ctrl+D with rows (1, 2, 3) selected left {[r.inpoint_seconds for r in model.records]}",
    )
    _expect(
        model.undo_stack.undoText() == "Delete 3 Events",
        f"Expected undo text 'Delete 3 Events', got {model.undo_stack.undoText()!r}",
    )
    _expect(
        view.currentIndex().row() == 1 and view.selected_rows() == (1,),
        f"After deleting (1, 2, 3), current={view.currentIndex().row()}, selected={view.selected_rows()}",
    )
    _expect(
        player.source_path == source_before and abs(player.position_seconds() - playhead_before) < 1e-6,
        "Deleting multiple rows moved the playhead or switched videos",
    )
    _expect(set_inpoint_action.isEnabled(), "Set Inpoint was not re-enabled after deleting down to 1 selected row")
    QtTest.QTest.mouseClick(viewport, left, pos=cell_center(1))
    fixture.app.processEvents()
    _expect(
        player.source_path == source_before and abs(player.position_seconds() - playhead_before) < 1e-6,
        "Clicking inside the row selected after deleting several rows moved the playhead",
    )
    model.undo_stack.undo()
    fixture.app.processEvents()
    _expect(
        list(model.records) == five_rows and model.undo_stack.isClean(),
        "Undo did not restore all 3 deleted rows in order",
    )

    # 6. Ctrl+A selects all five rows; Ctrl+click row 0 drops it from the selection.
    _focus_table(fixture)
    _press(fixture, key.Key_A, _CTRL)
    _expect(
        view.selected_rows() == (0, 1, 2, 3, 4) and not set_inpoint_action.isEnabled(),
        f"Ctrl+A left selected_rows={view.selected_rows()}",
    )
    QtTest.QTest.mouseClick(viewport, left, _CTRL, pos=cell_center(0))
    fixture.app.processEvents()
    _expect(view.selected_rows() == (1, 2, 3, 4), f"Ctrl+click on row 0 left selected_rows={view.selected_rows()}")

    # 7. Deleting the last two rows selects the new last row (row 2).
    QtTest.QTest.mouseClick(viewport, left, pos=cell_center(3))
    QtTest.QTest.mouseClick(viewport, left, shift, pos=cell_center(4))
    fixture.app.processEvents()
    _expect(view.selected_rows() == (3, 4), f"Expected (3, 4) selected, got {view.selected_rows()}")
    _press(fixture, key.Key_D, _CTRL)
    _expect(
        model.rowCount() == 3 and view.currentIndex().row() == 2 and view.selected_rows() == (2,),
        f"Deleting the last 2 rows left rowCount={model.rowCount()}, current={view.currentIndex().row()}, "
        f"selected={view.selected_rows()}",
    )
    model.undo_stack.undo()
    fixture.app.processEvents()
    _expect(list(model.records) == five_rows, "Undo did not restore the last 2 deleted rows")

    # 8. Clicking below the last row clears the selection, disables both actions, and keeps the current cell framed.
    last_rect = view.visualRect(model.index(4, int(col.LEVEL)))
    below_last = QtCore.QPoint(last_rect.center().x(), last_rect.bottom() + 20)
    QtTest.QTest.mouseClick(viewport, left, pos=below_last)
    fixture.app.processEvents()
    _expect(view.selected_rows() == (), f"Clicking below the last row left selected_rows={view.selected_rows()}")
    _expect(
        not set_inpoint_action.isEnabled() and not delete_event_action.isEnabled(),
        "Set Inpoint or Delete Event stayed enabled with 0 rows selected",
    )
    window.delete_selected_events()
    _expect(
        model.rowCount() == 5 and model.undo_stack.isClean(), "delete_selected_events modified rows with 0 selected"
    )
    _expect(
        _colors_match(_sample_cell(view, viewport.grab().toImage(), view.currentIndex()).edge, highlight),
        "Current cell lost its frame when 0 rows were selected",
    )

    # 9. Restore the previous records and selection so the save step sees the expected clean table.
    model.set_records(saved_records)
    view.setCurrentIndex(model.index(0, 0))
    fixture.app.processEvents()


def _check_go_to_event_and_alt_navigation(fixture: _Fixture) -> None:
    """Go to Event (Ctrl+G) seeks to one selected row; holding Alt moves between rows without seeking."""
    window = fixture.window
    view = window.table_view
    viewport = view.viewport()
    model = window.table_model
    player = window.player
    col = ballroom_scribe.gui.EventTableModel.Column
    key = QtCore.Qt.Key
    left = QtCore.Qt.MouseButton.LeftButton
    shift = QtCore.Qt.KeyboardModifier.ShiftModifier
    if not _focus_table(fixture):
        return

    go_to_event_action = _action_named(window, "Go to Event")
    _expect(model.rowCount() == _INITIAL_ROW_COUNT, f"Expected {_INITIAL_ROW_COUNT} rows, got {model.rowCount()}")
    row_0_inpoint = model.record_at(0).inpoint_seconds
    row_1_inpoint = model.record_at(1).inpoint_seconds
    row_2_inpoint = model.record_at(2).inpoint_seconds

    def cell_center(row: int, column: ballroom_scribe.gui.EventTableModel.Column = col.LEVEL) -> QtCore.QPoint:
        return view.visualRect(model.index(row, int(column))).center()

    # 1. Click row 0, seek elsewhere, and press Ctrl+G to jump back to row 0's Inpoint.
    QtTest.QTest.mouseClick(viewport, left, pos=cell_center(0))
    fixture.app.processEvents()
    _expect(go_to_event_action.isEnabled(), "Go to Event was disabled with 1 row selected")
    _expect_source(fixture, fixture.clip, "after clicking row 0")
    player.seek(1.0)
    _press(fixture, key.Key_G, _CTRL)
    _expect(
        abs(player.position_seconds() - row_0_inpoint) < 1e-3,
        f"Ctrl+G moved the playhead to {player.position_seconds():.3f} s, expected {row_0_inpoint:.3f} s",
    )

    # 2. Alt+click row 1 selects it without seeking; a plain click inside row 1 still does not seek;
    # go_to_selected_event jumps to row 1's Inpoint.
    player.seek(1.0)
    QtTest.QTest.mouseClick(viewport, left, _ALT, pos=cell_center(1))
    fixture.app.processEvents()
    _release_alt(fixture)
    _expect(
        view.selected_rows() == (1,) and view.currentIndex().row() == 1,
        f"Alt+click on row 1 left selected_rows={view.selected_rows()}, current={view.currentIndex().row()}",
    )
    _expect(
        abs(player.position_seconds() - 1.0) < 1e-3,
        f"Alt+click on row 1 moved the playhead to {player.position_seconds():.3f} s",
    )
    QtTest.QTest.mouseClick(viewport, left, pos=cell_center(1, col.STYLE))
    fixture.app.processEvents()
    _expect(
        abs(player.position_seconds() - 1.0) < 1e-3,
        f"Plain click inside Alt-reached row 1 moved the playhead to {player.position_seconds():.3f} s",
    )
    window.go_to_selected_event()
    _expect(
        abs(player.position_seconds() - row_1_inpoint) < 1e-3,
        f"go_to_selected_event moved the playhead to {player.position_seconds():.3f} s, expected {row_1_inpoint:.3f} s",
    )

    # 3. Alt+Down onto row 2 (second video) leaves video and playhead untouched; Ctrl+G switches clip and seeks.
    player.seek(2.0)
    _press(fixture, key.Key_Down, _ALT)
    _release_alt(fixture)
    _expect(
        view.currentIndex().row() == 2 and view.selected_rows() == (2,),
        f"Alt+Down left current={view.currentIndex().row()}, selected_rows={view.selected_rows()}",
    )
    _expect_source(fixture, fixture.clip, "after Alt+Down onto the second video's row")
    _expect(
        abs(player.position_seconds() - 2.0) < 1e-3,
        f"Alt+Down moved the playhead to {player.position_seconds():.3f} s",
    )
    _press(fixture, key.Key_G, _CTRL)
    _expect_source(fixture, fixture.second_clip, "after Ctrl+G on the second video's row")
    _expect(
        abs(player.position_seconds() - row_2_inpoint) < 1e-3,
        f"Ctrl+G on row 2 moved the playhead to {player.position_seconds():.3f} s, expected {row_2_inpoint:.3f} s",
    )

    # 4. Alt+Up back to row 1 keeps the second clip active; a plain Up to row 0 switches back and seeks as usual.
    _press(fixture, key.Key_Up, _ALT)
    _release_alt(fixture)
    _expect(
        view.currentIndex().row() == 1 and view.selected_rows() == (1,),
        f"Alt+Up left current={view.currentIndex().row()}, selected_rows={view.selected_rows()}",
    )
    _expect_source(fixture, fixture.second_clip, "after Alt+Up onto the first video's row")
    _expect(
        abs(player.position_seconds() - row_2_inpoint) < 1e-3,
        f"Alt+Up moved the playhead to {player.position_seconds():.3f} s",
    )
    _press(fixture, key.Key_Up)
    _expect(view.currentIndex().row() == 0, f"Plain Up left current={view.currentIndex().row()}")
    _expect_source(fixture, fixture.clip, "after plain Up to row 0")
    _expect(
        abs(player.position_seconds() - row_0_inpoint) < 1e-3,
        f"Plain Up to row 0 moved the playhead to {player.position_seconds():.3f} s, expected {row_0_inpoint:.3f} s",
    )

    # 5. Alt+Shift+Down extends the selection without seeking and disables Go to Event (as does 0 selected rows).
    player.seek(1.0)
    _press(fixture, key.Key_Down, _ALT | shift)
    _release_alt(fixture)
    _expect(view.selected_rows() == (0, 1), f"Alt+Shift+Down left selected_rows={view.selected_rows()}")
    _expect(
        abs(player.position_seconds() - 1.0) < 1e-3,
        f"Alt+Shift+Down moved the playhead to {player.position_seconds():.3f} s",
    )
    _expect(not go_to_event_action.isEnabled(), "Go to Event stayed enabled with 2 rows selected")
    _press(fixture, key.Key_G, _CTRL)
    window.go_to_selected_event()
    _expect(
        abs(player.position_seconds() - 1.0) < 1e-3,
        "Ctrl+G or go_to_selected_event moved the playhead with 2 rows selected",
    )
    last_rect = view.visualRect(model.index(2, int(col.LEVEL)))
    below_last = QtCore.QPoint(last_rect.center().x(), last_rect.bottom() + 20)
    QtTest.QTest.mouseClick(viewport, left, pos=below_last)
    fixture.app.processEvents()
    _expect(
        view.selected_rows() == () and not go_to_event_action.isEnabled(),
        "Go to Event stayed enabled with 0 rows selected",
    )

    # 6. Restore selection to row 0.
    QtTest.QTest.mouseClick(viewport, left, pos=cell_center(0))
    fixture.app.processEvents()
    _expect(view.selected_rows() == (0,) and model.undo_stack.isClean(), "Restore after Alt navigation check failed")


def _check_find_and_replace(fixture: _Fixture) -> None:
    """Inline Find and Replace highlights matches, jumps (unless Alt is held), scopes to rows/columns, and undoes."""
    window = fixture.window
    view = window.table_view
    viewport = view.viewport()
    model = window.table_model
    player = window.player
    bar = window.find_replace_bar
    col = ballroom_scribe.gui.EventTableModel.Column
    key = QtCore.Qt.Key
    left = QtCore.Qt.MouseButton.LeftButton
    shift = QtCore.Qt.KeyboardModifier.ShiftModifier
    if not _focus_table(fixture):
        return

    original_records = list(model.records)
    test_rows = [
        ballroom_scribe.models.EventRecord(
            file=fixture.clip,
            inpoint_seconds=1.5,
            level="Pre-Champ",
            style="Standard",
            dance="Waltz",
            round="Semi-Final",
        ),
        ballroom_scribe.models.EventRecord(
            file=fixture.clip,
            inpoint_seconds=4.0,
            level="Champ",
            style="Latin",
            dance="cha-cha",
            round="Final",
        ),
        ballroom_scribe.models.EventRecord(
            file=fixture.second_clip,
            inpoint_seconds=2.0,
            level="champ",
            style="Smooth",
            dance="Waltz",
            round="Semi-Final",
        ),
        ballroom_scribe.models.EventRecord(
            file=fixture.second_clip,
            inpoint_seconds=5.5,
            level="Open",
            style="Open Standard",
            dance="Foxtrot",
            round="Final",
        ),
    ]
    model.set_records(test_rows)
    fixture.app.processEvents()

    def cell_center(row: int, column: ballroom_scribe.gui.EventTableModel.Column = col.LEVEL) -> QtCore.QPoint:
        return view.visualRect(model.index(row, int(column))).center()

    QtTest.QTest.mouseClick(viewport, left, pos=cell_center(0))
    fixture.app.processEvents()
    player.seek(7.0)
    _expect(not bar.isVisible(), "FindReplaceBar should start hidden")

    # 1. Ctrl+F opens Find; live typing highlights matches and updates count without moving the playhead.
    _press(fixture, key.Key_F, _CTRL)
    _expect(bar.isVisible() and not bar.is_replace_visible, "Ctrl+F did not open Find-only bar")
    QtTest.QTest.keyClicks(bar.find_edit, "champ")
    fixture.app.processEvents()
    expected_champ_matches = frozenset({(0, int(col.LEVEL)), (1, int(col.LEVEL)), (2, int(col.LEVEL))})
    _expect(
        bar.match_count == 3 and bar.status_text == "3 matches" and view.search_matches == expected_champ_matches,
        f"Live Find for 'champ' produced count={bar.match_count}, status={bar.status_text!r}",
    )
    _expect(
        abs(player.position_seconds() - 7.0) < 1e-3,
        f"Typing in Find moved the playhead to {player.position_seconds():.3f} s",
    )

    # 2. Enter / Shift+Enter step through matches, frame the active match while Find keeps focus, and jump to Inpoint.
    QtTest.QTest.keyClick(bar.find_edit, key.Key_Return)
    fixture.app.processEvents()
    _expect(
        bar.active_match == (0, int(col.LEVEL)) and bar.status_text == "1 of 3",
        f"First Find Next produced active={bar.active_match}, status={bar.status_text!r}",
    )
    _expect(
        abs(player.position_seconds() - 1.5) < 1e-3,
        f"First Find Next moved playhead to {player.position_seconds():.3f} s, expected 1.5 s",
    )
    color_group = QtGui.QPalette.ColorGroup.Active if window.isActiveWindow() else QtGui.QPalette.ColorGroup.Inactive
    highlight = view.palette().color(color_group, QtGui.QPalette.ColorRole.Highlight)
    active_sample = _sample_cell(view, view.viewport().grab().toImage(), model.index(0, int(col.LEVEL)))
    _expect(
        _colors_match(active_sample.edge, highlight),
        f"Active search match edge {active_sample.edge.name()} was not framed with {highlight.name()}",
    )

    QtTest.QTest.keyClick(bar.find_edit, key.Key_Return)
    fixture.app.processEvents()
    _expect(
        bar.active_match == (1, int(col.LEVEL)) and abs(player.position_seconds() - 4.0) < 1e-3,
        "Second Find Next did not jump to row 1 (4.0 s)",
    )

    QtTest.QTest.keyClick(bar.find_edit, key.Key_Return)
    fixture.app.processEvents()
    _expect(
        bar.active_match == (2, int(col.LEVEL)) and bar.status_text == "3 of 3", "Third Find Next did not reach 3/3"
    )
    _expect_source(fixture, fixture.second_clip, "after Find Next onto the second video's row")
    _expect(abs(player.position_seconds() - 2.0) < 1e-3, "Third Find Next did not seek to 2.0 s")

    QtTest.QTest.keyClick(bar.find_edit, key.Key_Return, shift)
    fixture.app.processEvents()
    _expect(bar.active_match == (1, int(col.LEVEL)) and bar.status_text == "2 of 3", "Shift+Enter did not step back")
    _expect_source(fixture, fixture.clip, "after Find Previous back to the first video's row")
    _expect(abs(player.position_seconds() - 4.0) < 1e-3, "Find Previous did not seek back to 4.0 s")

    # 3. Holding Alt (Alt+Enter and Alt+F3) steps to matches without moving the playhead or switching videos.
    player.seek(8.0)
    QtTest.QTest.keyClick(bar.find_edit, key.Key_Return, _ALT)
    fixture.app.processEvents()
    QtTest.QTest.keyRelease(bar.find_edit, key.Key_Alt)
    _release_alt(fixture)
    _expect(
        bar.active_match == (2, int(col.LEVEL)) and view.selected_rows() == (2,),
        f"Alt+Enter in Find left active={bar.active_match}, selected_rows={view.selected_rows()}",
    )
    _expect_source(fixture, fixture.clip, "after Alt+Enter onto the second video's row")
    _expect(
        abs(player.position_seconds() - 8.0) < 1e-3,
        f"Alt+Enter in Find moved playhead to {player.position_seconds():.3f} s",
    )

    _press(fixture, key.Key_F3, _ALT)
    _release_alt(fixture)
    _expect(
        bar.active_match == (0, int(col.LEVEL)) and abs(player.position_seconds() - 8.0) < 1e-3,
        f"Alt+F3 left active={bar.active_match}, playhead={player.position_seconds():.3f} s",
    )

    # 4. Match case, Match entire cell, and Column filter narrow matches accurately.
    bar.find_edit.setText("Champ")
    bar.match_case_checkbox.setChecked(True)
    fixture.app.processEvents()
    _expect(bar.match_count == 2, f"Case-sensitive 'Champ' matched {bar.match_count} cells, expected 2")
    bar.whole_cell_checkbox.setChecked(True)
    fixture.app.processEvents()
    _expect(bar.match_count == 1, f"Whole-cell case-sensitive 'Champ' matched {bar.match_count} cells, expected 1")

    bar.match_case_checkbox.setChecked(False)
    bar.whole_cell_checkbox.setChecked(False)
    bar.find_edit.setText("open")
    fixture.app.processEvents()
    _expect(bar.match_count == 2, f"'open' across all text columns matched {bar.match_count} cells, expected 2")
    style_combo_index = bar.column_combo.findData(col.STYLE)
    bar.column_combo.setCurrentIndex(style_combo_index)
    fixture.app.processEvents()
    _expect(bar.match_count == 1, f"'open' filtered to Style matched {bar.match_count} cells, expected 1")
    bar.column_combo.setCurrentIndex(0)
    fixture.app.processEvents()

    # 5. Esc in the Find input closes the bar, clears search highlights, and returns focus to the table.
    QtTest.QTest.keyClick(bar.find_edit, key.Key_Escape)
    fixture.app.processEvents()
    _expect(
        not bar.isVisible() and view.search_matches == frozenset() and view.active_search_match is None,
        "Esc did not close FindReplaceBar and clear search highlights",
    )

    # 6. Multi-row selection + Ctrl+H auto-enables Selected rows (2); Replace All is one undo step and keeps playhead.
    QtTest.QTest.mouseClick(viewport, left, pos=cell_center(1))
    QtTest.QTest.mouseClick(viewport, left, shift, pos=cell_center(2))
    fixture.app.processEvents()
    _expect(view.selected_rows() == (1, 2), f"Expected rows (1, 2) selected, got {view.selected_rows()}")
    player.seek(6.0)

    _press(fixture, key.Key_H, _CTRL)
    _expect(bar.isVisible() and bar.is_replace_visible, "Ctrl+H did not open Replace mode")
    _expect(
        bar.selected_rows_only and bar.scoped_rows == (1, 2),
        f"Ctrl+H with rows (1, 2) selected left selected_rows_only={bar.selected_rows_only}, scoped={bar.scoped_rows}",
    )
    bar.find_edit.setText("champ")
    bar.replace_edit.setText("Championship")
    fixture.app.processEvents()
    _expect(bar.match_count == 2, f"Scoped 'champ' matched {bar.match_count} cells, expected 2 (excluding row 0)")

    QtTest.QTest.mouseClick(bar.replace_all_button, left)
    fixture.app.processEvents()
    _expect(
        model.record_at(0).level == "Pre-Champ"
        and model.record_at(1).level == "Championship"
        and model.record_at(2).level == "Championship",
        f"Scoped Replace All produced levels {[r.level for r in model.records]}",
    )
    _expect(
        model.undo_stack.undoText() == "Replace All" and abs(player.position_seconds() - 6.0) < 1e-3,
        f"Replace All undoText={model.undo_stack.undoText()!r}, playhead={player.position_seconds():.3f} s",
    )
    model.undo_stack.undo()
    fixture.app.processEvents()
    _expect(list(model.records) == test_rows and model.undo_stack.isClean(), "Undo did not revert Replace All")
    model.undo_stack.redo()
    fixture.app.processEvents()
    _expect(model.record_at(1).level == "Championship" and model.record_at(2).level == "Championship", "Redo failed")
    model.undo_stack.undo()
    fixture.app.processEvents()

    # 7. Single Replace steps onto the nearest match first, then replaces it and advances to the next match.
    bar.selected_rows_checkbox.setChecked(False)
    bar.find_edit.setText("Semi-Final")
    bar.replace_edit.setText("Semifinal")
    fixture.app.processEvents()
    _expect(bar.match_count == 2 and bar.active_match is None, "Expected 2 unactivated matches for 'Semi-Final'")
    QtTest.QTest.keyClick(bar.replace_edit, key.Key_Return)
    fixture.app.processEvents()
    _expect(
        bar.active_match == (2, int(col.ROUND)) and model.undo_stack.isClean(),
        f"First Replace click should activate nearest match without mutating, got active={bar.active_match}",
    )
    QtTest.QTest.keyClick(bar.replace_edit, key.Key_Return)
    fixture.app.processEvents()
    _expect(
        model.record_at(2).round == "Semifinal"
        and model.undo_stack.undoText() == "Replace Round"
        and bar.active_match == (0, int(col.ROUND)),
        f"Second Replace left round={model.record_at(2).round!r}, active={bar.active_match}",
    )
    model.undo_stack.undo()
    fixture.app.processEvents()

    # 8. Esc while the table itself has focus also closes an open FindReplaceBar.
    _focus_table(fixture)
    _press(fixture, key.Key_Escape)
    _expect(not bar.isVisible(), "Esc on focused table did not close FindReplaceBar")

    model.set_records(original_records)
    view.setCurrentIndex(model.index(0, int(col.LEVEL)))
    fixture.app.processEvents()
    _expect(model.undo_stack.isClean(), "Undo stack not clean after restoring original records")


def _check_close_session_shortcut(fixture: _Fixture) -> None:
    window = fixture.window
    if not _focus_table(fixture):
        window.close_session()
        return
    # A dirty document would open a modal Save/Discard prompt; the save step must have run first.
    _expect(window.table_model.undo_stack.isClean(), "Document must be clean before Ctrl+W")
    _press(fixture, QtCore.Qt.Key.Key_W, _CTRL)
    _expect(
        len(window.session) == 0 and window.table_model.rowCount() == 0 and window.current_csv is None,
        "Ctrl+W did not close the session",
    )


def _check_close_during_extraction(fixture: _Fixture) -> None:
    window = fixture.window
    window.add_videos(
        [
            ballroom_scribe.cli.arguments.InputCacheArgs(
                input_video=fixture.long_clip,
                input_channel=0,
                cache_dir=fixture.work_dir,
            )
        ]
    )
    fixture.app.processEvents()
    _expect(len(window.session) == 1, "Long clip was not added to the session")
    cache_dir = window.session.videos[0].cache_dir
    _expect(
        window.waveform.envelope is None and not window.waveform.is_error,
        "The long clip's audio was extracted before the window could be closed; raise _LONG_CLIP_SECONDS",
    )

    started = time.monotonic()
    _expect(window.close(), "Window refused to close")
    fixture.app.processEvents()
    elapsed = time.monotonic() - started
    _expect(
        elapsed < _CLOSE_TIMEOUT_SECONDS,
        f"Closing took {elapsed:.1f} s while audio extraction was running (ffmpeg was not cancelled?)",
    )
    leftovers = sorted(path.name for path in cache_dir.glob("*.partial.*"))
    _expect(not leftovers, f"Partial cache files left behind: {leftovers}")
    _expect(
        not (cache_dir / "Audio-Channel-0.wav").exists(),
        "The cancelled extraction still produced Audio-Channel-0.wav",
    )


def _check_save_round_trip(fixture: _Fixture) -> None:
    window = fixture.window
    saved_csv = fixture.work_dir / "saved_events.csv"
    _expect(window.save_csv(saved_csv), "save_csv failed")
    _expect(window.table_model.undo_stack.isClean(), "Saving did not mark the document clean")
    reloaded = list(ballroom_scribe.EventRecorder().read(saved_csv))
    _expect(reloaded == list(window.table_model.records), "Saved CSV does not round-trip through EventRecorder")


def _screenshot_path(work_dir: pathlib.Path) -> pathlib.Path:
    return work_dir / "editor_screenshot.png"


def _save_screenshot(fixture: _Fixture) -> None:
    _settle(fixture.app, _SCREENSHOT_SETTLE_SECONDS)
    pixmap = QtGui.QPixmap()
    if fixture.windowed:
        # Grabbing from the screen includes the native video surface, which `QWidget.grab()` cannot capture.
        screen = fixture.window.screen() or fixture.app.primaryScreen()
        pixmap = screen.grabWindow(fixture.window.winId())
    if pixmap.isNull():
        pixmap = fixture.window.grab()
    screenshot = _screenshot_path(fixture.work_dir)
    _expect(pixmap.save(str(screenshot)), f"Could not save {screenshot}")


def _run_checks(fixture: _Fixture) -> None:
    steps: tuple[tuple[str, Callable[[_Fixture], None]], ...] = (
        ("Load video, extract audio, and compute the waveform", _check_video_and_waveform),
        ("Open CSV, jump to a row, and render paused frames", _check_csv_row_jump_renders_frames),
        ("Frame the current cell while the table has focus", _check_current_cell_frame),
        ("Drag the fill handle to copy a value", _check_fill_handle),
        ("Switch between videos from the table", _check_switching_video_renders_frames),
        ("Zoom the waveform", _check_waveform_zoom),
        ("Click and drag the waveform", _check_waveform_click_and_drag),
        ("Follow playback in the table", _check_playback_follows_rows),
        ("Paint the focused waveform", _check_focused_waveform_paints_interior),
        ("Autocomplete from other rows", _check_autocomplete_from_other_rows),
        ("Set Inpoint, typed inpoints, Add/Delete Event, Undo/Redo", _check_editing_commands),
        ("Keyboard shortcuts", _check_keyboard_shortcuts),
        ("Type to edit and Delete to clear", _check_spreadsheet_typing_and_clear),
        ("Select several rows, delete them, and keep Set Inpoint to one row", _check_multi_row_selection),
        ("Go to Event and move between rows with Alt", _check_go_to_event_and_alt_navigation),
        ("Find and replace across rows and columns", _check_find_and_replace),
        ("Save CSV and reload it", _check_save_round_trip),
        ("Capture a screenshot", _save_screenshot),
        ("Close the session with Ctrl+W", _check_close_session_shortcut),
        ("Close the window while audio is being extracted", _check_close_during_extraction),
    )
    for number, (description, step) in enumerate(steps, start=1):
        print(f"[{number}/{len(steps)}] {description}")
        step(fixture)


def main(argv: list[str] | None = None) -> int:
    """Runs the smoke test and returns a process exit code."""
    args = _parse_args(argv)
    if sys.stdout is not None:
        # Keep progress lines interleaved correctly with log output when stdout is piped to a file.
        sys.stdout.reconfigure(line_buffering=True)
    ballroom_scribe.gui.utils.configure_logging(args.verbose)
    _check_import_footprint()

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("ffmpeg was not found on PATH; install it as described in README.md", file=sys.stderr)
        return 1
    work_dir = args.work_dir or pathlib.Path(tempfile.mkdtemp(prefix="ballroom_scribe_smoke_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"Work directory: {work_dir}")
    if not args.windowed:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    try:
        clip = work_dir / "smoke_clip.mp4"
        second_clip = work_dir / "smoke_clip_b.mp4"
        long_clip = work_dir / "smoke_clip_long.mov"
        _generate_clip(ffmpeg, clip, _CLIP_SIZE)
        _generate_clip(ffmpeg, second_clip, _SECOND_CLIP_SIZE)
        _generate_clip(
            ffmpeg,
            long_clip,
            _LONG_CLIP_SIZE,
            seconds=_LONG_CLIP_SECONDS,
            fps=_LONG_CLIP_FPS,
            audio_expressions=_LONG_CLIP_AUDIO_EXPRESSIONS,
            channel_layout="mono",
            sample_rate=_LONG_CLIP_SAMPLE_RATE,
            audio_args=_LONG_CLIP_AUDIO_ARGS,
        )

        app = QtWidgets.QApplication(sys.argv)
        window = ballroom_scribe.gui.MainWindow()
        _expect(not app.windowIcon().isNull(), "Application window icon was not set")
        _expect(not window.windowIcon().isNull(), "MainWindow icon was not set")
        window.show()
        app.processEvents()
        fixture = _Fixture(
            app=app,
            window=window,
            work_dir=work_dir,
            clip=clip,
            second_clip=second_clip,
            long_clip=long_clip,
            windowed=args.windowed,
        )
        _run_checks(fixture)
        if window.isVisible():
            window.close()
        app.processEvents()
    except _SmokeTestError as err:
        print(f"SMOKE TEST FAILED: {err}", file=sys.stderr)
        return 1

    print(f"SMOKE TEST PASSED. Screenshot saved to {_screenshot_path(work_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
