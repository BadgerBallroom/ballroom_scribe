"""Main window of the PySide6 Ballroom Scribe timestamp editor."""

import pathlib
import subprocess
from collections.abc import Sequence
from typing import override

from PySide6 import QtCore, QtGui, QtWidgets

import ballroom_scribe
import ballroom_scribe.cli.arguments
import ballroom_scribe.gui
import ballroom_scribe.gui.models
import ballroom_scribe.gui.utils
import ballroom_scribe.models

_VIDEO_FILE_FILTER = "Video Files (*.mp4 *.mov *.mkv *.avi *.m4v *.webm *.mts *.m2ts);;All Files (*)"
_CSV_FILE_FILTER = "CSV Files (*.csv);;All Files (*)"
_MIN_FILE_COLUMN_WIDTH = 160
_MAX_FILE_COLUMN_WIDTH = 180
_MIN_TEXT_COLUMN_WIDTH = 90
# The `Comment` column shows its text only when it can start this wide while `Level` … `Round` keep their minimum;
# otherwise it shrinks to a presence icon.
_MIN_COMMENT_TEXT_WIDTH = 160
_COMMENT_ICON_COLUMN_WIDTH = 28
_DEFAULT_WINDOW_WIDTH = 1400
_DEFAULT_WINDOW_HEIGHT = 820
# Initial video : table width ratio of the horizontal splitter.
_CONTENT_SPLIT_RATIO = (1, 1)
_SKIP_SECONDS = 10.0


def _quit_key_sequences() -> list[QtGui.QKeySequence]:
    """Returns the platform's standard Quit bindings, or `Alt+F4` where it has none (Windows)."""
    standard = QtGui.QKeySequence.keyBindings(QtGui.QKeySequence.StandardKey.Quit)
    return list(standard) if standard else [QtGui.QKeySequence("Alt+F4")]


def _with_shortcut_hint(action: QtGui.QAction, description: str) -> None:
    """Sets `action`'s tooltip to `description` followed by its shortcuts in native notation."""
    keys = ", ".join(seq.toString(QtGui.QKeySequence.SequenceFormat.NativeText) for seq in action.shortcuts())
    action.setToolTip(f"{description} ({keys})" if keys else description)


def _text_input_has_focus() -> bool:
    focused = QtWidgets.QApplication.focusWidget()
    return isinstance(
        focused,
        (
            QtWidgets.QLineEdit,
            QtWidgets.QAbstractSpinBox,
            QtWidgets.QPlainTextEdit,
            QtWidgets.QTextEdit,
        ),
    )


def _alt_is_held() -> bool:
    """Returns whether `Alt` (`Option` on macOS) is held, which moves between rows without moving the playhead."""
    return bool(QtWidgets.QApplication.keyboardModifiers() & QtCore.Qt.KeyboardModifier.AltModifier)


def _marker_label(row: int, record: ballroom_scribe.models.EventRecord) -> str:
    parts = [part for part in (record.level, record.style, record.dance, record.round) if part]
    return " ".join(parts) if parts else f"Event #{row + 1}"


def _distribute_width(total: int, weights: Sequence[float]) -> list[int]:
    """Splits `total` pixels across columns in proportion to `weights`, each at least `_MIN_TEXT_COLUMN_WIDTH`.

    Rounding leftovers go to the last column so the columns fill `total` exactly.
    """
    total_weight = sum(weights)
    widths = [max(_MIN_TEXT_COLUMN_WIDTH, round(total * weight / total_weight)) for weight in weights]
    widths[-1] += max(0, total - sum(widths))
    return widths


class MainWindow(QtWidgets.QMainWindow):
    """Top-level window coordinating video playback, waveform display, event table editing, and pipeline runs."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        app = QtGui.QGuiApplication.instance()
        if isinstance(app, QtGui.QGuiApplication) and app.windowIcon().isNull():
            app.setWindowIcon(ballroom_scribe.gui.utils.create_app_icon())
        self.resize(_DEFAULT_WINDOW_WIDTH, _DEFAULT_WINDOW_HEIGHT)

        self._session = ballroom_scribe.gui.VideoSession()
        self._recorder = ballroom_scribe.EventRecorder()
        self._table_model = ballroom_scribe.gui.EventTableModel(self, video_rank=self._video_rank)
        self._waveform_loader = ballroom_scribe.gui.WaveformLoader(self)
        self._active_video: ballroom_scribe.gui.models.LoadedVideo | None = None
        self._current_csv: pathlib.Path | None = None
        self._updating_selector = False
        self._selecting_without_jump = False
        # The row the user is on: the playhead was last moved to its inpoint, it was selected without jumping (after
        # deleting rows, clicking the waveform, or as playback crossed its inpoint), or it was reached with `Alt`
        # held. Qt keeps it pointing at the same event across insertions, removals, and moves. Clicks within this row
        # (e.g. double-clicking a cell to edit it) must not seek.
        self._visited_index = QtCore.QPersistentModelIndex()

        self._player = ballroom_scribe.gui.VideoPlayerWidget(self)
        self._waveform = ballroom_scribe.gui.WaveformWidget(self)
        self._table_view = ballroom_scribe.gui.EventTableView(self)
        self._find_replace_bar = ballroom_scribe.gui.FindReplaceBar(self._table_model, self._table_view, self)
        self._configure_table_view()

        table_pane = QtWidgets.QWidget(self)
        table_pane_layout = QtWidgets.QVBoxLayout(table_pane)
        table_pane_layout.setContentsMargins(0, 0, 0, 0)
        table_pane_layout.setSpacing(0)
        table_pane_layout.addWidget(self._find_replace_bar)
        table_pane_layout.addWidget(self._table_view, stretch=1)

        # Video beside the event table, both above a full-width waveform; every pane is user-resizable.
        content_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self)
        content_splitter.setChildrenCollapsible(False)
        content_splitter.addWidget(self._player)
        content_splitter.addWidget(table_pane)
        video_share, table_share = _CONTENT_SPLIT_RATIO
        content_splitter.setStretchFactor(0, video_share)
        content_splitter.setStretchFactor(1, table_share)
        content_splitter.setSizes(
            [
                _DEFAULT_WINDOW_WIDTH * video_share // sum(_CONTENT_SPLIT_RATIO),
                _DEFAULT_WINDOW_WIDTH * table_share // sum(_CONTENT_SPLIT_RATIO),
            ]
        )

        root_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, self)
        root_splitter.setChildrenCollapsible(False)
        root_splitter.addWidget(content_splitter)
        root_splitter.addWidget(self._waveform)
        root_splitter.setStretchFactor(0, 1)
        root_splitter.setStretchFactor(1, 0)
        self.setCentralWidget(root_splitter)

        self._video_combo = QtWidgets.QComboBox(self)
        self._video_combo.setMinimumWidth(220)
        self._video_combo.setEnabled(False)
        self._video_combo.currentIndexChanged.connect(self._on_video_combo_changed)

        self._channel_spin = QtWidgets.QSpinBox(self)
        self._channel_spin.setRange(0, 0)
        self._channel_spin.setEnabled(False)
        self._channel_spin.setToolTip("Zero-based audio channel index used for the waveform and pipeline")
        self._channel_spin.valueChanged.connect(self._on_channel_spin_changed)

        self._build_actions_menus_and_toolbar()
        self._register_shortcuts()
        self._connect_signals()
        self.statusBar().showMessage("Ready")
        self._update_window_title()
        self._update_action_states()

    @property
    def session(self) -> ballroom_scribe.gui.VideoSession:
        """Returns the current `VideoSession`."""
        return self._session

    @property
    def table_model(self) -> ballroom_scribe.gui.EventTableModel:
        """Returns the underlying `EventTableModel`."""
        return self._table_model

    @property
    def current_csv(self) -> pathlib.Path | None:
        """Returns the path of the currently open CSV file, if any."""
        return self._current_csv

    @property
    def player(self) -> ballroom_scribe.gui.VideoPlayerWidget:
        """Returns the video player widget."""
        return self._player

    @property
    def waveform(self) -> ballroom_scribe.gui.WaveformWidget:
        """Returns the audio waveform widget."""
        return self._waveform

    @property
    def table_view(self) -> ballroom_scribe.gui.EventTableView:
        """Returns the event table view."""
        return self._table_view

    @property
    def find_replace_bar(self) -> ballroom_scribe.gui.FindReplaceBar:
        """Returns the inline Find and Replace bar."""
        return self._find_replace_bar

    def add_videos(self, inputs: Sequence[ballroom_scribe.cli.arguments.InputCacheArgs]) -> None:
        """Adds one or more video inputs to the current session."""
        if not inputs:
            return
        added_any = False
        first_added_csv_file: pathlib.Path | None = None

        for video_args in inputs:
            try:
                loaded = self._session.add(
                    input_video=video_args["input_video"],
                    input_channel=video_args["input_channel"],
                    parent_cache_dir=video_args["cache_dir"],
                )
            except (ValueError, subprocess.CalledProcessError, OSError) as err:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Could Not Load Video",
                    f"Failed to load {video_args['input_video']}:\n{err}",
                )
                continue

            added_any = True
            if first_added_csv_file is None:
                first_added_csv_file = loaded.csv_file

        if not added_any:
            return

        self._rebuild_video_combo(
            select_csv_file=self._active_video.csv_file if self._active_video else first_added_csv_file
        )
        self._refresh_waveform_markers()
        self._update_action_states()

        # Never load events implicitly; only point out a previous pipeline result sitting next to the video.
        if len(self._session) == 1 and self._current_csv is None:
            sibling_csv = self._session.videos[0].input_video.with_suffix(".csv")
            if sibling_csv.exists():
                self.statusBar().showMessage(
                    f"Found existing {sibling_csv.name} next to the video; use File > Open CSV to load it.",
                    10000,
                )

    def open_csv(
        self,
        csv_path: pathlib.Path,
        *,
        interactive_missing_videos: bool = True,
    ) -> bool:
        """Loads `csv_path` into the table and adds any referenced videos to the session."""
        extra_columns: tuple[str, ...] = ()

        def capture_extra(columns: tuple[str, ...]) -> None:
            nonlocal extra_columns
            extra_columns = columns

        try:
            records = list(self._recorder.read(csv_path, on_extra_columns=capture_extra))
        except (OSError, ValueError) as err:
            QtWidgets.QMessageBox.critical(
                self,
                "Failed to Open CSV",
                f"Could not read {csv_path}:\n{err}",
            )
            return False

        if extra_columns:
            joined = ", ".join(extra_columns)
            QtWidgets.QMessageBox.warning(
                self,
                "Extra CSV Columns Ignored",
                f"The CSV contains unknown column(s) ({joined}) that are ignored and will not be preserved on save.",
            )

        resolved_csv = csv_path.resolve()
        self._current_csv = resolved_csv
        self._table_model.set_records(records)
        self._load_videos_referenced_by_records(
            records,
            csv_dir=resolved_csv.parent,
            interactive=interactive_missing_videos,
        )
        self._resize_table_columns()
        self._refresh_waveform_markers()
        self._update_window_title()
        self._update_action_states()
        self.statusBar().showMessage(f"Loaded {len(records)} event(s) from {resolved_csv.name}", 5000)
        return True

    def save_csv(self, target_path: pathlib.Path | None = None) -> bool:
        """Writes the table's event records to `target_path` or `current_csv`."""
        destination = target_path or self._current_csv
        if destination is None:
            return self._save_csv_as()

        try:
            self._recorder.write(self._table_model.records, destination)
        except OSError as err:
            QtWidgets.QMessageBox.critical(
                self,
                "Failed to Save CSV",
                f"Could not write {destination}:\n{err}",
            )
            return False

        self._current_csv = destination.resolve()
        self._table_model.undo_stack.setClean()
        self._update_window_title()
        self.statusBar().showMessage(f"Saved {self._current_csv}", 5000)
        return True

    def close_session(self) -> bool:
        """Clears all loaded videos and table rows after prompting for unsaved edits."""
        if not self._confirm_discard_or_save():
            return False
        self._waveform_loader.cancel()
        self._player.set_source(None)
        self._waveform.clear()
        self._session.clear()
        self._active_video = None
        self._current_csv = None
        self._table_model.set_records(())
        self._rebuild_video_combo(select_csv_file=None)
        self._update_window_title()
        self._update_action_states()
        self.statusBar().showMessage("Session closed", 3000)
        return True

    def set_selected_inpoint_to_playhead(self) -> None:
        """Overwrites the selected row's `File` and `Inpoint` with the active video and playhead.

        Requires exactly one selected row; the bare `I` shortcut bypasses the action's enabled state, so the
        method checks for itself and explains in the status bar otherwise.
        """
        rows = self._table_view.selected_rows()
        if self._active_video is None or not rows:
            return
        if len(rows) > 1:
            self.statusBar().showMessage(f"Set Inpoint needs a single selected row; {len(rows)} are selected.", 5000)
            return
        inpoint_seconds = round(self._player.position_seconds(), 3)
        self._table_model.set_inpoint(
            row=rows[0],
            file=self._active_video.csv_file,
            inpoint_seconds=inpoint_seconds,
        )

    def add_event_at_playhead(self) -> None:
        """Adds an event for the active video at the playhead and starts editing its `Level`.

        The row is placed among the active video's rows in `Inpoint` order (see `EventTableModel.add_record`).
        """
        if self._active_video is None:
            return
        record = ballroom_scribe.models.EventRecord(
            file=self._active_video.csv_file,
            inpoint_seconds=round(self._player.position_seconds(), 3),
            level="",
            style="",
            dance="",
            round="",
        )
        insert_row = self._table_model.add_record(record)
        level_index = self._table_model.index(
            insert_row,
            int(ballroom_scribe.gui.EventTableModel.Column.LEVEL),
        )
        self._table_view.setCurrentIndex(level_index)
        self._table_view.edit(level_index)

    def go_to_selected_event(self) -> None:
        """Moves the playhead to the selected row's `Inpoint`, switching to its video if needed.

        Requires exactly one selected row; the action is disabled otherwise.
        """
        rows = self._table_view.selected_rows()
        if len(rows) != 1:
            return
        column = max(self._table_view.currentIndex().column(), 0)
        self._table_view.scrollTo(self._table_model.index(rows[0], column))
        self._jump_to_row(rows[0])

    def delete_selected_events(self) -> None:
        """Deletes every selected row as one undo step and selects the row that followed them.

        The playhead stays where it is, as when a single row is deleted.
        """
        rows = self._table_view.selected_rows()
        if not rows:
            return
        self._table_model.remove_records(rows)
        self._select_row_keeping_playhead(min(rows[0], self._table_model.rowCount() - 1))
        self._update_action_states()

    def open_find(self) -> None:
        """Shows the inline Find bar above the event table and focuses its query input."""
        self._find_replace_bar.open_find()

    def open_replace(self) -> None:
        """Shows the inline Find and Replace bar above the event table."""
        self._find_replace_bar.open_replace()

    @override
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if not self._confirm_discard_or_save():
            event.ignore()
            return
        self._player.pause()
        # Kills any in-flight `ffmpeg` and waits for its worker; otherwise Qt would block at exit until it finished.
        self._waveform_loader.shutdown()
        super().closeEvent(event)

    def _configure_table_view(self) -> None:
        self._table_view.setModel(self._table_model)
        self._table_view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        # `ExtendedSelection` supports `Shift`+click / `Shift+↑↓` ranges, `Ctrl`+click (`⌘` on macOS) row toggling,
        # press-and-drag ranges, and `Ctrl+A`; when ≥2 rows are selected, `CurrentCellDelegate` and `EventTableView`
        # hide single-cell affordances.
        self._table_view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table_view.setAlternatingRowColors(True)
        self._table_view.setWordWrap(False)
        self._table_view.setTextElideMode(QtCore.Qt.TextElideMode.ElideMiddle)
        # `EventTableView` only lets letters and digits use `AnyKeyPressed` (see `EventTableView.keyPressEvent`).
        self._table_view.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
            | QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed
            | QtWidgets.QAbstractItemView.EditTrigger.AnyKeyPressed
        )
        header = self._table_view.horizontalHeader()
        header.setStretchLastSection(False)  # `_fit_text_columns` fills the width; the comment column may stay narrow.
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)

        # Rows are selected as a whole, so the delegates frame the current cell to show where typing or `Enter` edits.
        self._table_view.setItemDelegate(ballroom_scribe.gui.CurrentCellDelegate(self._table_view))
        delegate = ballroom_scribe.gui.AutocompleteDelegate(self._table_view)
        for text_column in ballroom_scribe.gui.EventTableModel.TEXT_COLUMNS:
            self._table_view.setItemDelegateForColumn(int(text_column), delegate)
        self._table_view.setItemDelegateForColumn(
            int(ballroom_scribe.gui.EventTableModel.Column.COMMENT),
            ballroom_scribe.gui.CommentDelegate(self._table_view),
        )
        # Keeps the text columns filling the table as it is resized (see `eventFilter`).
        self._table_view.viewport().installEventFilter(self)
        self._resize_table_columns()

    @override
    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if watched is self._table_view.viewport() and event.type() == QtCore.QEvent.Type.Resize:
            assert isinstance(event, QtGui.QResizeEvent)
            if event.size().width() != event.oldSize().width():
                self._fit_text_columns(equalize=False)
        return super().eventFilter(watched, event)

    def _resize_table_columns(self) -> None:
        """Sizes the fixed-content columns to their contents and splits the remaining width equally among the rest."""
        self._table_view.resizeColumnsToContents()
        header = self._table_view.horizontalHeader()
        file_section = int(ballroom_scribe.gui.EventTableModel.Column.FILE)
        file_width = header.sectionSize(file_section)
        header.resizeSection(file_section, min(max(file_width, _MIN_FILE_COLUMN_WIDTH), _MAX_FILE_COLUMN_WIDTH))
        self._fit_text_columns(equalize=True)

    def _fit_text_columns(self, *, equalize: bool) -> None:
        """Makes the text columns share the width left over by the other columns, fitting `Comment` in as it allows.

        `Level`, `Style`, `Dance`, and `Round` always share that width. `Comment` joins them (starting at least
        `_MIN_COMMENT_TEXT_WIDTH` wide) when the four can still keep `_MIN_TEXT_COLUMN_WIDTH` each; otherwise it
        becomes a fixed `_COMMENT_ICON_COLUMN_WIDTH` column showing only whether a comment exists.

        Args:
            equalize: Give the shared columns equal widths; otherwise keep their current proportions (as adjusted by
                the user) while filling the available width. A switch between the comment presentations always
                re-equalizes, since the previous widths no longer mean anything.
        """
        header = self._table_view.horizontalHeader()
        model_cls = ballroom_scribe.gui.EventTableModel
        shared = [int(column) for column in model_cls.TEXT_COLUMNS]
        comment = int(model_cls.Column.COMMENT)
        fixed_width = sum(
            header.sectionSize(section)
            for section in range(header.count())
            if section not in shared and section != comment
        )
        available = self._table_view.viewport().width() - fixed_width
        compact = available < len(shared) * _MIN_TEXT_COLUMN_WIDTH + _MIN_COMMENT_TEXT_WIDTH
        display = model_cls.CommentDisplay.ICON if compact else model_cls.CommentDisplay.TEXT
        equalize = equalize or display != self._table_model.comment_display
        self._table_model.set_comment_display(display)

        if compact:
            header.setSectionResizeMode(comment, QtWidgets.QHeaderView.ResizeMode.Fixed)
            header.resizeSection(comment, _COMMENT_ICON_COLUMN_WIDTH)
            available -= _COMMENT_ICON_COLUMN_WIDTH
            comment_width = None
        else:
            header.setSectionResizeMode(comment, QtWidgets.QHeaderView.ResizeMode.Interactive)
            shared.append(comment)
            comment_width = max(_MIN_COMMENT_TEXT_WIDTH, round(available / len(shared))) if equalize else None
        available = max(available, len(shared) * _MIN_TEXT_COLUMN_WIDTH)

        if comment_width is not None:
            widths = _distribute_width(available - comment_width, [1.0] * (len(shared) - 1)) + [comment_width]
        else:
            weights = [1.0] * len(shared)
            if not equalize:
                current = [float(header.sectionSize(section)) for section in shared]
                if sum(current) > 0:
                    weights = current
            widths = _distribute_width(available, weights)
        for section, width in zip(shared, widths, strict=True):
            header.resizeSection(section, width)

    def _build_actions_menus_and_toolbar(self) -> None:
        self._open_videos_action = QtGui.QAction("Open Videos…", self)
        self._open_videos_action.setShortcut(QtGui.QKeySequence("Ctrl+O"))
        self._open_videos_action.triggered.connect(self._prompt_open_videos)

        self._open_csv_action = QtGui.QAction("Open CSV…", self)
        self._open_csv_action.setShortcut(QtGui.QKeySequence("Ctrl+Shift+O"))
        self._open_csv_action.triggered.connect(self._prompt_open_csv)

        self._set_cache_dir_action = QtGui.QAction("Set Cache Directory…", self)
        self._set_cache_dir_action.triggered.connect(self._prompt_set_cache_dir)

        self._save_action = QtGui.QAction("Save", self)
        self._save_action.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        self._save_action.triggered.connect(lambda: self.save_csv())

        self._save_as_action = QtGui.QAction("Save As…", self)
        self._save_as_action.setShortcut(QtGui.QKeySequence("Ctrl+Shift+S"))
        self._save_as_action.triggered.connect(self._save_csv_as)

        self._close_session_action = QtGui.QAction("Close Session", self)
        self._close_session_action.setShortcut(QtGui.QKeySequence("Ctrl+W"))
        self._close_session_action.triggered.connect(self.close_session)

        self._exit_action = QtGui.QAction("Exit", self)
        self._exit_action.setShortcuts(_quit_key_sequences())
        self._exit_action.triggered.connect(self.close)

        self._undo_action = self._table_model.undo_stack.createUndoAction(self, "Undo")
        self._undo_action.setShortcut(QtGui.QKeySequence.StandardKey.Undo)

        self._redo_action = self._table_model.undo_stack.createRedoAction(self, "Redo")
        self._redo_action.setShortcuts([QtGui.QKeySequence("Ctrl+Y"), QtGui.QKeySequence.StandardKey.Redo])

        self._set_inpoint_action = QtGui.QAction("Set Inpoint", self)
        self._set_inpoint_action.setShortcut(QtGui.QKeySequence("Ctrl+I"))
        _with_shortcut_hint(
            self._set_inpoint_action,
            "Overwrite selected row's File, Inpoint, and Inpoint Formatted with current video and playhead"
            " (needs exactly one selected row)",
        )
        self._set_inpoint_action.triggered.connect(self.set_selected_inpoint_to_playhead)

        self._add_event_action = QtGui.QAction("Add Event", self)
        self._add_event_action.setShortcut(QtGui.QKeySequence("Ctrl+E"))
        _with_shortcut_hint(self._add_event_action, "Insert a new event row at the current playhead")
        self._add_event_action.triggered.connect(self.add_event_at_playhead)

        self._delete_event_action = QtGui.QAction("Delete Event", self)
        self._delete_event_action.setShortcut(QtGui.QKeySequence("Ctrl+D"))
        _with_shortcut_hint(self._delete_event_action, "Delete the selected event row(s)")
        self._delete_event_action.triggered.connect(self.delete_selected_events)

        self._find_action = QtGui.QAction("Find…", self)
        self._find_action.setShortcut(QtGui.QKeySequence.StandardKey.Find)
        _with_shortcut_hint(self._find_action, "Find text in the event table")
        self._find_action.triggered.connect(self.open_find)

        self._replace_action = QtGui.QAction("Replace…", self)
        self._replace_action.setShortcut(QtGui.QKeySequence("Ctrl+H"))
        _with_shortcut_hint(self._replace_action, "Find and replace text in the event table")
        self._replace_action.triggered.connect(self.open_replace)

        self._find_next_action = QtGui.QAction("Find Next", self)
        self._find_next_action.setShortcut(QtGui.QKeySequence("F3"))
        _with_shortcut_hint(self._find_next_action, "Step to the next search match (hold Alt to keep playhead)")
        self._find_next_action.triggered.connect(
            lambda: self._step_find_match(forward=True, keep_playhead=_alt_is_held())
        )

        self._find_previous_action = QtGui.QAction("Find Previous", self)
        self._find_previous_action.setShortcut(QtGui.QKeySequence("Shift+F3"))
        _with_shortcut_hint(
            self._find_previous_action,
            "Step to the previous search match (hold Alt to keep playhead)",
        )
        self._find_previous_action.triggered.connect(
            lambda: self._step_find_match(forward=False, keep_playhead=_alt_is_held())
        )

        self._play_pause_action = QtGui.QAction("Play / Pause", self)
        self._play_pause_action.setShortcut(QtGui.QKeySequence("Ctrl+K"))
        self._play_pause_action.triggered.connect(self._player.toggle_play_pause)

        self._skip_back_action = QtGui.QAction("Skip Back 10 s", self)
        self._skip_back_action.setShortcut(QtGui.QKeySequence("Ctrl+J"))
        self._skip_back_action.triggered.connect(lambda: self._step_playhead(-_SKIP_SECONDS))

        self._skip_forward_action = QtGui.QAction("Skip Forward 10 s", self)
        self._skip_forward_action.setShortcut(QtGui.QKeySequence("Ctrl+L"))
        self._skip_forward_action.triggered.connect(lambda: self._step_playhead(_SKIP_SECONDS))

        self._go_to_event_action = QtGui.QAction("Go to Event", self)
        self._go_to_event_action.setShortcut(QtGui.QKeySequence("Ctrl+G"))
        _with_shortcut_hint(
            self._go_to_event_action,
            "Move the playhead to the selected row's Inpoint, switching video if needed"
            " (needs exactly one selected row)",
        )
        self._go_to_event_action.triggered.connect(self.go_to_selected_event)

        self._zoom_in_action = QtGui.QAction("Zoom In Waveform", self)
        self._zoom_in_action.setShortcuts(
            [QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.ZoomIn), QtGui.QKeySequence("Ctrl+=")]
        )
        self._zoom_in_action.triggered.connect(self._waveform.zoom_in)

        self._zoom_out_action = QtGui.QAction("Zoom Out Waveform", self)
        self._zoom_out_action.setShortcut(QtGui.QKeySequence.StandardKey.ZoomOut)
        self._zoom_out_action.triggered.connect(self._waveform.zoom_out)

        self._reset_zoom_action = QtGui.QAction("Reset Waveform Zoom", self)
        self._reset_zoom_action.setShortcut(QtGui.QKeySequence("Ctrl+0"))
        self._reset_zoom_action.triggered.connect(self._waveform.reset_zoom)

        self._run_pipeline_action = QtGui.QAction("Auto-Find Events…", self)
        self._run_pipeline_action.setShortcut(QtGui.QKeySequence("Ctrl+R"))
        _with_shortcut_hint(self._run_pipeline_action, "Run find_timestamps.py on the loaded video(s)")
        self._run_pipeline_action.triggered.connect(self._open_pipeline_dialog)

        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self._open_videos_action)
        file_menu.addAction(self._open_csv_action)
        file_menu.addAction(self._set_cache_dir_action)
        file_menu.addSeparator()
        file_menu.addAction(self._save_action)
        file_menu.addAction(self._save_as_action)
        file_menu.addSeparator()
        file_menu.addAction(self._close_session_action)
        file_menu.addSeparator()
        file_menu.addAction(self._exit_action)

        edit_menu = self.menuBar().addMenu("&Edit")
        edit_menu.addAction(self._undo_action)
        edit_menu.addAction(self._redo_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self._set_inpoint_action)
        edit_menu.addAction(self._add_event_action)
        edit_menu.addAction(self._delete_event_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self._find_action)
        edit_menu.addAction(self._replace_action)
        edit_menu.addAction(self._find_next_action)
        edit_menu.addAction(self._find_previous_action)

        playback_menu = self.menuBar().addMenu("&Playback")
        playback_menu.addAction(self._play_pause_action)
        playback_menu.addAction(self._skip_back_action)
        playback_menu.addAction(self._skip_forward_action)
        playback_menu.addSeparator()
        playback_menu.addAction(self._go_to_event_action)

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self._zoom_in_action)
        view_menu.addAction(self._zoom_out_action)
        view_menu.addAction(self._reset_zoom_action)

        tools_menu = self.menuBar().addMenu("&Tools")
        tools_menu.addAction(self._run_pipeline_action)

        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        toolbar.addAction(self._open_videos_action)
        toolbar.addAction(self._open_csv_action)
        toolbar.addAction(self._save_action)
        toolbar.addSeparator()
        toolbar.addWidget(QtWidgets.QLabel(" Video: ", self))
        toolbar.addWidget(self._video_combo)
        toolbar.addWidget(QtWidgets.QLabel(" Ch: ", self))
        toolbar.addWidget(self._channel_spin)
        toolbar.addSeparator()
        toolbar.addAction(self._go_to_event_action)
        toolbar.addAction(self._set_inpoint_action)
        toolbar.addAction(self._add_event_action)
        toolbar.addAction(self._delete_event_action)
        toolbar.addSeparator()
        toolbar.addAction(self._run_pipeline_action)

    def _register_shortcuts(self) -> None:
        self._bind_global_key("Space", self._player.toggle_play_pause)
        # Bare-letter bindings yield to `EventTableView` while it has focus, because typing a letter or digit there
        # edits the current cell instead of triggering shortcuts (see `EventTableView.event`).
        self._bind_global_key("K", self._player.toggle_play_pause)
        self._bind_global_key("J", lambda: self._step_playhead(-1.0))
        self._bind_global_key("L", lambda: self._step_playhead(1.0))
        self._bind_global_key("Shift+J", lambda: self._step_playhead(-10.0))
        self._bind_global_key("Shift+L", lambda: self._step_playhead(10.0))
        self._bind_global_key(",", lambda: self._step_playhead(-0.1))
        self._bind_global_key(".", lambda: self._step_playhead(0.1))
        self._bind_global_key("I", self.set_selected_inpoint_to_playhead)

        insert_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Insert"), self._table_view)
        insert_shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        insert_shortcut.activated.connect(self._on_table_insert_shortcut)

        # Qt starts editing on `F2` (Windows, Linux) or `Enter` (macOS); offer both everywhere. The shortcuts
        # match only while the table itself has focus, so `Enter` inside an open editor still commits.
        for key in ("Return", "Enter", "F2"):
            edit_shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self._table_view)
            edit_shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetShortcut)
            edit_shortcut.activated.connect(self._edit_current_cell)

        alt_find_next = QtGui.QShortcut(QtGui.QKeySequence("Alt+F3"), self)
        alt_find_next.setContext(QtCore.Qt.ShortcutContext.WindowShortcut)
        alt_find_next.activated.connect(lambda: self._step_find_match(forward=True, keep_playhead=True))

        alt_find_previous = QtGui.QShortcut(QtGui.QKeySequence("Alt+Shift+F3"), self)
        alt_find_previous.setContext(QtCore.Qt.ShortcutContext.WindowShortcut)
        alt_find_previous.activated.connect(lambda: self._step_find_match(forward=False, keep_playhead=True))

    def _bind_global_key(self, sequence: str, callback: object) -> None:
        shortcut = QtGui.QShortcut(QtGui.QKeySequence(sequence), self)
        shortcut.setContext(QtCore.Qt.ShortcutContext.WindowShortcut)
        shortcut.activated.connect(lambda cb=callback: cb() if not _text_input_has_focus() else None)

    def _connect_signals(self) -> None:
        self._player.position_changed.connect(self._waveform.set_position)
        self._player.playback_advanced.connect(self._on_playback_advanced)
        self._player.playing_changed.connect(self._waveform.set_playing)
        self._player.slider_scrubbed.connect(self._waveform.recenter_playhead)
        self._player.step_requested.connect(self._step_playhead)
        self._player.error_occurred.connect(lambda msg: self.statusBar().showMessage(msg, 8000))

        self._waveform.seek_requested.connect(self._on_waveform_seek_requested)
        self._waveform.step_requested.connect(self._step_playhead)

        self._waveform_loader.loaded.connect(self._on_waveform_loaded)
        self._waveform_loader.failed.connect(self._on_waveform_failed)

        selection_model = self._table_view.selectionModel()
        selection_model.currentRowChanged.connect(self._on_current_row_changed)
        selection_model.selectionChanged.connect(self._on_selection_changed)
        self._table_view.clicked.connect(self._on_table_clicked)
        self._table_view.escape_pressed.connect(self._find_replace_bar.close_bar)

        self._find_replace_bar.match_activated.connect(self._on_find_match_activated)
        self._find_replace_bar.highlights_changed.connect(self._table_view.set_search_highlights)
        self._find_replace_bar.closed.connect(
            lambda: self._table_view.setFocus(QtCore.Qt.FocusReason.ShortcutFocusReason)
        )
        self._find_replace_bar.status_message.connect(lambda msg: self.statusBar().showMessage(msg, 5000))

        self._table_model.undo_stack.cleanChanged.connect(self._on_clean_changed)
        self._table_model.dataChanged.connect(self._on_model_contents_changed)
        self._table_model.rowsInserted.connect(self._on_model_rows_changed)
        self._table_model.rowsRemoved.connect(self._on_model_rows_changed)
        self._table_model.rowsMoved.connect(self._on_model_rows_changed)
        self._table_model.modelReset.connect(self._on_model_reset)

    def _prompt_open_videos(self) -> None:
        start_dir = str(self._active_video.input_video.parent) if self._active_video else ""
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Open Video File(s)",
            start_dir,
            _VIDEO_FILE_FILTER,
        )
        if not paths:
            return
        inputs = [
            ballroom_scribe.cli.arguments.InputCacheArgs(
                input_video=pathlib.Path(raw_path),
                input_channel=ballroom_scribe.gui.VideoSession.infer_channel(
                    pathlib.Path(raw_path),
                    pathlib.Path(raw_path).parent,
                ),
                cache_dir=pathlib.Path(raw_path).parent,
            )
            for raw_path in paths
        ]
        self.add_videos(inputs)

    def _prompt_open_csv(self) -> None:
        if not self._confirm_discard_or_save():
            return
        start_dir = str(self._current_csv.parent) if self._current_csv else ""
        selected, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Events CSV",
            start_dir,
            _CSV_FILE_FILTER,
        )
        if selected:
            self.open_csv(pathlib.Path(selected))

    def _prompt_set_cache_dir(self) -> None:
        if self._active_video is None:
            return
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            f"Select Parent Cache Directory for {self._active_video.input_video.name}",
            str(self._active_video.parent_cache_dir),
        )
        if not chosen:
            return
        try:
            updated = self._session.set_parent_cache_dir(
                self._active_video.csv_file,
                pathlib.Path(chosen),
            )
        except ValueError as err:
            QtWidgets.QMessageBox.warning(self, "Invalid Cache Directory", str(err))
            return
        self._active_video = updated
        self._start_waveform_load(updated)
        self.statusBar().showMessage(f"Cache directory set to {updated.cache_dir}", 5000)

    def _save_csv_as(self) -> bool:
        if self._current_csv is not None:
            default_path = str(self._current_csv)
        elif self._active_video is not None:
            default_path = str(self._active_video.input_video.with_suffix(".csv"))
        else:
            default_path = "events.csv"

        selected, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Events CSV",
            default_path,
            _CSV_FILE_FILTER,
        )
        if not selected:
            return False
        return self.save_csv(pathlib.Path(selected))

    def _load_videos_referenced_by_records(
        self,
        records: Sequence[ballroom_scribe.models.EventRecord],
        *,
        csv_dir: pathlib.Path,
        interactive: bool,
    ) -> None:
        seen_files: set[pathlib.Path] = set()
        first_matched: pathlib.Path | None = None

        for record in records:
            if record.file in seen_files:
                continue
            seen_files.add(record.file)

            existing = self._session.find_for_record(record.file, csv_dir=csv_dir)
            if existing is not None:
                if first_matched is None:
                    first_matched = existing.csv_file
                continue

            disk_path = self._resolve_record_video_on_disk(record.file, csv_dir=csv_dir, interactive=interactive)
            if disk_path is None:
                continue

            parent_cache_dir = disk_path.parent
            channel = ballroom_scribe.gui.VideoSession.infer_channel(disk_path, parent_cache_dir)
            try:
                loaded = self._session.add(
                    input_video=disk_path,
                    input_channel=channel,
                    parent_cache_dir=parent_cache_dir,
                    csv_file=record.file,
                )
            except (ValueError, subprocess.CalledProcessError, OSError) as err:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Could Not Load Video",
                    f"Failed to load {disk_path}:\n{err}",
                )
                continue

            if first_matched is None:
                first_matched = loaded.csv_file

        target_selection = self._active_video.csv_file if self._active_video else first_matched
        self._rebuild_video_combo(select_csv_file=target_selection)

    def _resolve_record_video_on_disk(
        self,
        record_file: pathlib.Path,
        *,
        csv_dir: pathlib.Path,
        interactive: bool,
    ) -> pathlib.Path | None:
        if record_file.exists():
            return record_file
        relative_candidate = csv_dir / record_file
        if not record_file.is_absolute() and relative_candidate.exists():
            return relative_candidate
        if not interactive:
            return None

        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Question)
        box.setWindowTitle("Missing Video File")
        box.setText(f"Could not find video referenced by CSV:\n{record_file}")
        locate_button = box.addButton("Locate Video…", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Skip", QtWidgets.QMessageBox.ButtonRole.RejectRole)
        box.exec()

        if box.clickedButton() is locate_button:
            located, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                f"Locate {record_file.name}",
                str(csv_dir),
                _VIDEO_FILE_FILTER,
            )
            if located:
                return pathlib.Path(located)
        return None

    def _rebuild_video_combo(self, *, select_csv_file: pathlib.Path | None) -> None:
        self._updating_selector = True
        try:
            self._video_combo.clear()
            videos = self._session.videos
            name_counts: dict[str, int] = {}
            for video in videos:
                name_counts[video.input_video.name] = name_counts.get(video.input_video.name, 0) + 1

            selected_index = -1
            for index, video in enumerate(videos):
                label = video.input_video.name
                if name_counts[label] > 1:
                    label = f"{label} ({video.input_video.parent.name})"
                self._video_combo.addItem(label, video.csv_file)
                self._video_combo.setItemData(index, str(video.input_video), QtCore.Qt.ItemDataRole.ToolTipRole)
                if select_csv_file is not None and video.csv_file == select_csv_file:
                    selected_index = index

            self._video_combo.setEnabled(bool(videos))
            if videos:
                selected_index = max(selected_index, 0)
                self._video_combo.setCurrentIndex(selected_index)
        finally:
            self._updating_selector = False

        if self._session:
            chosen = self._video_combo.currentData()
            if isinstance(chosen, pathlib.Path):
                target = self._session.get(chosen)
                if target is not None and target != self._active_video:
                    self._activate_video(target)
        else:
            self._active_video = None
            self._channel_spin.setEnabled(False)

    def _activate_video(
        self,
        video: ballroom_scribe.gui.models.LoadedVideo,
        *,
        initial_seconds: float | None = None,
        preserve_playing: bool = False,
    ) -> None:
        self._active_video = video

        self._channel_spin.blockSignals(True)
        self._channel_spin.setRange(0, max(0, video.channel_count - 1))
        self._channel_spin.setValue(video.input_channel)
        self._channel_spin.setEnabled(video.channel_count > 1)
        self._channel_spin.blockSignals(False)

        self._waveform.recenter_playhead()
        self._player.set_source(
            video.input_video,
            initial_seconds=initial_seconds,
            preserve_playing=preserve_playing,
        )
        self._start_waveform_load(video)
        self._refresh_waveform_markers()
        self._update_action_states()

    def _start_waveform_load(self, video: ballroom_scribe.gui.models.LoadedVideo) -> None:
        self._waveform.set_loading(video.input_video.name)
        self._waveform_loader.load(video)

    def _on_video_combo_changed(self, index: int) -> None:
        if self._updating_selector or index < 0:
            return
        csv_file = self._video_combo.itemData(index)
        if isinstance(csv_file, pathlib.Path):
            video = self._session.get(csv_file)
            if video is not None and video != self._active_video:
                self._activate_video(video)

    def _on_channel_spin_changed(self, channel: int) -> None:
        if self._active_video is None or channel == self._active_video.input_channel:
            return
        updated = self._session.set_channel(self._active_video.csv_file, channel)
        self._active_video = updated
        self._start_waveform_load(updated)

    def _on_waveform_loaded(
        self,
        video: ballroom_scribe.gui.models.LoadedVideo,
        envelope: ballroom_scribe.gui.models.WaveformEnvelope,
    ) -> None:
        if self._active_video is not None and video == self._active_video:
            self._waveform.set_envelope(envelope)
            self.statusBar().showMessage(
                f"Waveform ready ({video.input_video.name}, channel {video.input_channel})",
                4000,
            )

    def _on_waveform_failed(
        self,
        video: ballroom_scribe.gui.models.LoadedVideo,
        message: str,
    ) -> None:
        if self._active_video is not None and video == self._active_video:
            self._waveform.set_error(message)
            self.statusBar().showMessage(f"Waveform error: {message}", 8000)

    def _video_rank(self, file: pathlib.Path) -> int | None:
        """Returns the position of `file`'s video in the session (the Video drop-down order), if it is loaded."""
        csv_dir = self._current_csv.parent if self._current_csv else None
        video = self._session.find_for_record(file, csv_dir=csv_dir)
        return self._session.videos.index(video) if video is not None else None

    def _mark_visited(self, row: int) -> None:
        """Records `row` as the row the user is on, so clicks within it do not seek (see `_on_table_clicked`)."""
        self._visited_index = QtCore.QPersistentModelIndex(self._table_model.index(row, 0))

    def _step_playhead(self, delta_seconds: float) -> None:
        self._waveform.apply_step(self._player.step, delta_seconds)

    def _on_waveform_seek_requested(self, seconds: float) -> None:
        self._player.seek(seconds)
        if not _alt_is_held():
            self._highlight_event_at_playhead()

    def _on_playback_advanced(self, previous_seconds: float, seconds: float) -> None:
        """Highlights the active video's event whose `Inpoint` playback just crossed, so the table follows along.

        Nothing happens while a cell editor is open, while rows are being drag-selected or filled, while two or more
        rows are selected, or while `Alt` (`Option` on macOS) is held; a crossing missed for one of those reasons is
        not applied later, so the selection stays put until the next `Inpoint` is crossed.
        """
        if self._active_video is None or _alt_is_held():
            return
        if self._table_view.state() != QtWidgets.QAbstractItemView.State.NoState or self._table_view.is_filling:
            return
        if self._table_view.has_multiple_selected_rows:
            return
        row = self._table_model.row_entered(self._active_video.csv_file, previous_seconds, seconds)
        if row is not None:
            self._select_row_keeping_playhead(row)

    def _highlight_event_at_playhead(self) -> None:
        """Selects the active video's last event at or before the playhead, or clears selection if none precedes it."""
        if self._active_video is None:
            return
        row = self._table_model.row_at_or_before(
            self._active_video.csv_file,
            self._player.position_seconds(),
        )
        if row is None:
            self._clear_table_selection_keeping_playhead()
        else:
            self._select_row_keeping_playhead(row)

    def _clear_table_selection_keeping_playhead(self) -> None:
        """Clears the table's row selection without moving the playhead."""
        if self._table_view.selected_rows():
            self._table_view.selectionModel().clearSelection()
        self._visited_index = QtCore.QPersistentModelIndex()

    def _select_row_keeping_playhead(self, row: int) -> None:
        """Makes `row` the current, selected row without seeking the video to its inpoint.

        The row is scrolled into view and becomes the one the user is on, so later clicks within it do not seek
        either.
        """
        if not 0 <= row < self._table_model.rowCount():
            return
        column = max(self._table_view.currentIndex().column(), 0)
        target_index = self._table_model.index(row, column)
        if self._table_view.selected_rows() != (row,) or self._table_view.currentIndex() != target_index:
            self._selecting_without_jump = True
            try:
                self._table_view.setCurrentIndex(target_index)
            finally:
                self._selecting_without_jump = False
        self._table_view.scrollTo(target_index)
        self._mark_visited(row)

    def _on_current_row_changed(
        self,
        current: QtCore.QModelIndex,
        previous: QtCore.QModelIndex,
    ) -> None:
        del previous
        self._update_action_states()
        if not current.isValid() or self._table_model.is_removing_rows or self._selecting_without_jump:
            return  # Deleting rows or selecting from the waveform moves `currentIndex` without jumping the playhead.
        if _alt_is_held():
            self._mark_visited(current.row())  # `Alt`+click / `Alt`+arrow: move to the row, leave the playhead alone.
            return
        self._jump_to_row(current.row())

    def _on_selection_changed(
        self,
        selected: QtCore.QItemSelection,
        deselected: QtCore.QItemSelection,
    ) -> None:
        del selected, deselected
        self._update_action_states()
        self._waveform.set_selected_rows(self._table_view.selected_rows())
        self._find_replace_bar.on_table_selection_changed(is_stepping=self._selecting_without_jump)

    def _step_find_match(self, *, forward: bool, keep_playhead: bool) -> None:
        if not self._find_replace_bar.isVisible():
            self._find_replace_bar.open_find()
        if forward:
            self._find_replace_bar.find_next(keep_playhead=keep_playhead)
        else:
            self._find_replace_bar.find_previous(keep_playhead=keep_playhead)

    def _on_find_match_activated(self, row: int, column: int, keep_playhead: bool) -> None:
        if not 0 <= row < self._table_model.rowCount():
            return
        target_index = self._table_model.index(row, column)
        if self._table_view.selected_rows() != (row,) or self._table_view.currentIndex() != target_index:
            self._selecting_without_jump = True
            try:
                self._table_view.setCurrentIndex(target_index)
            finally:
                self._selecting_without_jump = False
        self._table_view.scrollTo(target_index)
        if keep_playhead:
            self._mark_visited(row)
        else:
            self._jump_to_row(row)

    def _on_table_clicked(self, index: QtCore.QModelIndex) -> None:
        if not index.isValid():
            return
        if _alt_is_held():
            self._mark_visited(index.row())
            return
        if self._visited_index.isValid() and self._visited_index.row() == index.row():
            return  # Still on the same event: keep playing rather than restarting from its inpoint.
        self._jump_to_row(index.row())

    def _jump_to_row(self, row: int) -> None:
        if not 0 <= row < self._table_model.rowCount():
            return
        record = self._table_model.record_at(row)
        csv_dir = self._current_csv.parent if self._current_csv else None
        matched_video = self._session.find_for_record(record.file, csv_dir=csv_dir)

        if matched_video is None:
            self.statusBar().showMessage(f"Video not loaded for row {row + 1}: {record.file}", 5000)
            return

        if self._active_video != matched_video:
            self._updating_selector = True
            try:
                combo_idx = self._video_combo.findData(matched_video.csv_file)
                if combo_idx >= 0:
                    self._video_combo.setCurrentIndex(combo_idx)
            finally:
                self._updating_selector = False
            self._activate_video(
                matched_video,
                initial_seconds=record.inpoint_seconds,
                preserve_playing=True,
            )
        else:
            self._waveform.recenter_playhead()
            self._player.seek(record.inpoint_seconds)

        self._mark_visited(row)

    def _edit_current_cell(self) -> None:
        index = self._table_view.currentIndex()
        if (
            index.isValid()
            and not self._table_view.has_multiple_selected_rows
            and self._table_view.state() != QtWidgets.QAbstractItemView.State.EditingState
        ):
            self._table_view.edit(index)

    def _on_table_insert_shortcut(self) -> None:
        if self._table_view.state() != QtWidgets.QAbstractItemView.State.EditingState and not _text_input_has_focus():
            self.add_event_at_playhead()

    def _open_pipeline_dialog(self) -> None:
        if not self._session:
            return
        dialog = ballroom_scribe.gui.PipelineDialog(
            inputs=self._session.inputs(),
            default_output_csv=self._current_csv,
            has_unsaved_edits=not self._table_model.undo_stack.isClean(),
            parent=self,
        )
        dialog.pipeline_starting.connect(self._yield_extraction_to_pipeline)
        dialog.pipeline_succeeded.connect(self._on_pipeline_succeeded)
        dialog.exec()
        # Covers a failed or cancelled run (or none at all): the GUI extracts the audio itself again.
        self._reload_waveform_if_missing()

    def _yield_extraction_to_pipeline(self) -> None:
        """Stops the GUI's extraction so the pipeline is the only writer of the cache files (see `atomic_path`)."""
        self._waveform_loader.shutdown()
        if self._active_video is not None and self._waveform.envelope is None:
            name = self._active_video.input_video.name
            self._waveform.set_status(
                f"The pipeline is extracting audio for {name}; the waveform will load when it finishes."
            )

    def _on_pipeline_succeeded(self, csv_path: pathlib.Path) -> None:
        self.open_csv(csv_path)
        # Recomputed from the pipeline's complete WAV while the dialog is still open.
        self._reload_waveform_if_missing()

    def _reload_waveform_if_missing(self) -> None:
        if self._active_video is not None and self._waveform.envelope is None:
            self._start_waveform_load(self._active_video)

    def _refresh_waveform_markers(self) -> None:
        if self._active_video is None:
            self._waveform.set_markers(())
            return
        csv_dir = self._current_csv.parent if self._current_csv else None
        markers: list[tuple[int, float, str]] = []
        for row, record in enumerate(self._table_model.records):
            if record.file == self._active_video.csv_file or (
                self._session.find_for_record(record.file, csv_dir=csv_dir) == self._active_video
            ):
                markers.append((row, record.inpoint_seconds, _marker_label(row, record)))
        self._waveform.set_markers(markers, selected_rows=self._table_view.selected_rows())

    def _on_model_contents_changed(
        self,
        top_left: QtCore.QModelIndex,
        bottom_right: QtCore.QModelIndex,
    ) -> None:
        del top_left, bottom_right
        self._refresh_waveform_markers()
        self._find_replace_bar.refresh_matches()

    def _on_model_rows_changed(self, *_: object) -> None:
        """Handles `rowsInserted`, `rowsRemoved`, and `rowsMoved`, whose arguments are not needed."""
        self._refresh_waveform_markers()
        self._find_replace_bar.refresh_matches()
        self._update_action_states()

    def _on_model_reset(self) -> None:
        self._refresh_waveform_markers()
        self._find_replace_bar.on_model_reset()
        self._update_action_states()

    def _on_clean_changed(self, is_clean: bool) -> None:
        self.setWindowModified(not is_clean)
        self._update_window_title()

    def _update_window_title(self) -> None:
        doc_name = self._current_csv.name if self._current_csv is not None else "Untitled"
        self.setWindowTitle(f"{doc_name}[*] — Ballroom Scribe Timestamp Editor")

    def _update_action_states(self) -> None:
        has_video = self._active_video is not None
        selected_rows = self._table_view.selected_rows()
        self._set_cache_dir_action.setEnabled(has_video)
        self._go_to_event_action.setEnabled(len(selected_rows) == 1)
        self._set_inpoint_action.setEnabled(has_video and len(selected_rows) == 1)
        self._add_event_action.setEnabled(has_video)
        self._delete_event_action.setEnabled(bool(selected_rows))
        self._play_pause_action.setEnabled(has_video)
        self._skip_back_action.setEnabled(has_video)
        self._skip_forward_action.setEnabled(has_video)
        self._zoom_in_action.setEnabled(has_video)
        self._zoom_out_action.setEnabled(has_video)
        self._reset_zoom_action.setEnabled(has_video)
        self._run_pipeline_action.setEnabled(bool(self._session))

    def _confirm_discard_or_save(self) -> bool:
        if self._table_model.undo_stack.isClean():
            return True
        choice = QtWidgets.QMessageBox.warning(
            self,
            "Unsaved Changes",
            "The current events table has unsaved changes. Save before continuing?",
            QtWidgets.QMessageBox.StandardButton.Save
            | QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Save,
        )
        if choice == QtWidgets.QMessageBox.StandardButton.Save:
            return self.save_csv()
        return choice == QtWidgets.QMessageBox.StandardButton.Discard
