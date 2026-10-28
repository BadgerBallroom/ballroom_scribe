"""Inline Find and Replace bar docked above the event table."""

from collections.abc import Sequence
from typing import override

from PySide6 import QtCore, QtGui, QtWidgets

import ballroom_scribe
import ballroom_scribe.gui

_ENTER_KEYS = frozenset({QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter})
_TOGGLE_BUTTON_WIDTH = 24
_STATUS_LABEL_MIN_WIDTH = 84


def _alt_is_held(modifiers: QtCore.Qt.KeyboardModifier = QtCore.Qt.KeyboardModifier.NoModifier) -> bool:
    """Returns whether `Alt` (`Option` on macOS) is held in `modifiers` or on `QApplication`."""
    combined = modifiers | QtWidgets.QApplication.keyboardModifiers()
    return bool(combined & QtCore.Qt.KeyboardModifier.AltModifier)


class FindReplaceBar(QtWidgets.QWidget):
    """Compact inline Find and Replace bar scoped to the editable text columns of `EventTableModel`.

    Typing in the Find box live-updates cell highlights and the match count without moving the playhead. Stepping
    with `Enter` / `F3` or `Shift+Enter` / `Shift+F3` (or the `↑` / `↓` buttons) activates the matched cell and
    emits `match_activated` with `keep_playhead=True` when `Alt` (`Option` on macOS) is held.
    """

    match_activated = QtCore.Signal(int, int, bool)
    highlights_changed = QtCore.Signal(object, object)
    closed = QtCore.Signal()
    status_message = QtCore.Signal(str)

    def __init__(
        self,
        model: ballroom_scribe.gui.EventTableModel,
        table_view: ballroom_scribe.gui.EventTableView,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._model = model
        self._table_view = table_view
        self._matches: tuple[tuple[int, int], ...] = ()
        self._active_match: tuple[int, int] | None = None
        self._pinned_rows: tuple[QtCore.QPersistentModelIndex, ...] = ()

        self._toggle_replace_button = QtWidgets.QToolButton(self)
        self._toggle_replace_button.setFixedWidth(_TOGGLE_BUTTON_WIDTH)
        self._toggle_replace_button.setToolTip("Toggle Replace (Ctrl+H)")
        self._toggle_replace_button.clicked.connect(self._on_toggle_replace_clicked)

        self._find_edit = QtWidgets.QLineEdit(self)
        self._find_edit.setPlaceholderText("Find")
        self._find_edit.setClearButtonEnabled(True)
        self._find_edit.installEventFilter(self)
        self._find_edit.textChanged.connect(self._on_search_criteria_changed)

        self._status_label = QtWidgets.QLabel(self)
        self._status_label.setMinimumWidth(_STATUS_LABEL_MIN_WIDTH)
        self._status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)

        self._prev_button = QtWidgets.QToolButton(self)
        self._prev_button.setText("↑")
        self._prev_button.setToolTip("Previous match (Shift+Enter, Shift+F3; hold Alt to keep playhead)")
        self._prev_button.clicked.connect(lambda: self.find_previous(keep_playhead=_alt_is_held()))

        self._next_button = QtWidgets.QToolButton(self)
        self._next_button.setText("↓")
        self._next_button.setToolTip("Next match (Enter, F3; hold Alt to keep playhead)")
        self._next_button.clicked.connect(lambda: self.find_next(keep_playhead=_alt_is_held()))

        self._close_button = QtWidgets.QToolButton(self)
        self._close_button.setText("✕")
        self._close_button.setToolTip("Close (Esc)")
        self._close_button.clicked.connect(self.close_bar)

        self._replace_row = QtWidgets.QWidget(self)
        self._replace_edit = QtWidgets.QLineEdit(self._replace_row)
        self._replace_edit.setPlaceholderText("Replace")
        self._replace_edit.setClearButtonEnabled(True)
        self._replace_edit.installEventFilter(self)

        self._replace_button = QtWidgets.QPushButton("Replace", self._replace_row)
        self._replace_button.setToolTip("Replace current match and advance (Enter; hold Alt to keep playhead)")
        self._replace_button.clicked.connect(lambda: self.replace_current(keep_playhead=_alt_is_held()))

        self._replace_all_button = QtWidgets.QPushButton("Replace All", self._replace_row)
        self._replace_all_button.setToolTip("Replace every match in scope as a single undo step")
        self._replace_all_button.clicked.connect(self.replace_all)

        self._column_combo = QtWidgets.QComboBox(self)
        self._column_combo.addItem("All Text Columns", None)
        for column in ballroom_scribe.gui.EventTableModel.TEXT_COLUMNS:
            self._column_combo.addItem(ballroom_scribe.EventRecorder.COLUMNS[int(column)], column)
        self._column_combo.currentIndexChanged.connect(self._on_search_criteria_changed)

        self._match_case_check = QtWidgets.QCheckBox("Match case", self)
        self._match_case_check.toggled.connect(self._on_search_criteria_changed)

        self._whole_cell_check = QtWidgets.QCheckBox("Match entire cell", self)
        self._whole_cell_check.toggled.connect(self._on_search_criteria_changed)

        self._selected_rows_check = QtWidgets.QCheckBox("Selected rows only", self)
        self._selected_rows_check.toggled.connect(self._on_selected_rows_toggled)

        self._build_layout()
        self._register_shortcuts()
        self._set_replace_visible(False)
        self._sync_selected_rows_ui()
        self._emit_state()
        self.setVisible(False)

    @property
    def is_replace_visible(self) -> bool:
        """Returns whether the Replace row is expanded."""
        return not self._replace_row.isHidden()

    @property
    def query(self) -> str:
        """Returns the current Find query string."""
        return self._find_edit.text()

    @property
    def replacement(self) -> str:
        """Returns the current Replace string."""
        return self._replace_edit.text()

    @property
    def status_text(self) -> str:
        """Returns the human-readable match counter text."""
        return self._status_label.text()

    @property
    def match_count(self) -> int:
        """Returns the number of cells currently matching the Find query."""
        return len(self._matches)

    @property
    def active_match(self) -> tuple[int, int] | None:
        """Returns the `(row, column)` cell currently stepped onto, if any."""
        return self._active_match

    @property
    def selected_column(self) -> ballroom_scribe.gui.EventTableModel.Column | None:
        """Returns the single text column selected in the Column filter, or `None` for all text columns."""
        data = self._column_combo.currentData()
        return data if isinstance(data, ballroom_scribe.gui.EventTableModel.Column) else None

    @property
    def case_sensitive(self) -> bool:
        """Returns whether **Match case** is checked."""
        return self._match_case_check.isChecked()

    @property
    def whole_cell(self) -> bool:
        """Returns whether **Match entire cell** is checked."""
        return self._whole_cell_check.isChecked()

    @property
    def selected_rows_only(self) -> bool:
        """Returns whether **Selected rows only** is checked."""
        return self._selected_rows_check.isChecked()

    @property
    def scoped_rows(self) -> tuple[int, ...]:
        """Returns the pinned row indices when **Selected rows only** is checked, else `()`."""
        return self._valid_pinned_rows() if self.selected_rows_only else ()

    @property
    def find_edit(self) -> QtWidgets.QLineEdit:
        """Returns the Find query input widget."""
        return self._find_edit

    @property
    def replace_edit(self) -> QtWidgets.QLineEdit:
        """Returns the Replace input widget."""
        return self._replace_edit

    @property
    def previous_button(self) -> QtWidgets.QToolButton:
        """Returns the Find Previous button."""
        return self._prev_button

    @property
    def next_button(self) -> QtWidgets.QToolButton:
        """Returns the Find Next button."""
        return self._next_button

    @property
    def replace_button(self) -> QtWidgets.QPushButton:
        """Returns the single-match Replace button."""
        return self._replace_button

    @property
    def replace_all_button(self) -> QtWidgets.QPushButton:
        """Returns the Replace All button."""
        return self._replace_all_button

    @property
    def column_combo(self) -> QtWidgets.QComboBox:
        """Returns the column-scope combo box."""
        return self._column_combo

    @property
    def match_case_checkbox(self) -> QtWidgets.QCheckBox:
        """Returns the **Match case** checkbox."""
        return self._match_case_check

    @property
    def whole_cell_checkbox(self) -> QtWidgets.QCheckBox:
        """Returns the **Match entire cell** checkbox."""
        return self._whole_cell_check

    @property
    def selected_rows_checkbox(self) -> QtWidgets.QCheckBox:
        """Returns the **Selected rows only** checkbox."""
        return self._selected_rows_check

    def open_find(self) -> None:
        """Shows the bar (collapsing Replace if newly opened) and focuses the Find input."""
        was_visible = self.isVisible()
        if not was_visible:
            self._set_replace_visible(False)
        self._apply_selection_scope_on_open(was_visible=was_visible)
        self.setVisible(True)
        self.refresh_matches()
        self._find_edit.setFocus(QtCore.Qt.FocusReason.ShortcutFocusReason)
        self._find_edit.selectAll()

    def open_replace(self) -> None:
        """Shows the bar with the Replace row expanded and focuses Find (or Replace if Find already has text)."""
        was_visible = self.isVisible()
        was_replace_visible = self.is_replace_visible
        self._apply_selection_scope_on_open(was_visible=was_visible)
        self._set_replace_visible(True)
        self.setVisible(True)
        self.refresh_matches()
        target_edit = (
            self._replace_edit if was_visible and not was_replace_visible and bool(self.query) else self._find_edit
        )
        target_edit.setFocus(QtCore.Qt.FocusReason.ShortcutFocusReason)
        target_edit.selectAll()

    def close_bar(self) -> None:
        """Hides the bar, clears search highlights, and notifies `MainWindow` to return focus to the table."""
        if not self.isVisible():
            return
        self.setVisible(False)
        self._matches = ()
        self._active_match = None
        self.highlights_changed.emit(frozenset(), None)
        self.closed.emit()

    def find_next(self, *, keep_playhead: bool = False) -> bool:
        """Steps forward to the next match (wrapping around) and activates it."""
        if not self.isVisible():
            return False
        self.refresh_matches()
        if not self._matches:
            return False
        next_match = self._step_match(forward=True)
        self._activate_match(next_match, keep_playhead=keep_playhead)
        return True

    def find_previous(self, *, keep_playhead: bool = False) -> bool:
        """Steps backward to the previous match (wrapping around) and activates it."""
        if not self.isVisible():
            return False
        self.refresh_matches()
        if not self._matches:
            return False
        prev_match = self._step_match(forward=False)
        self._activate_match(prev_match, keep_playhead=keep_playhead)
        return True

    def replace_current(self, *, keep_playhead: bool = False) -> bool:
        """Replaces the active match (or steps onto the nearest match first) and advances to the next match."""
        if not self.isVisible():
            return False
        self.refresh_matches()
        if not self._matches:
            self.status_message.emit("No matches to replace")
            return False

        target = self._resolvable_replace_target()
        if target is None:
            self.find_next(keep_playhead=keep_playhead)
            return False

        index = self._model.index(target[0], target[1])
        changed = self._model.replace_in_cell(
            index,
            self.query,
            self.replacement,
            case_sensitive=self.case_sensitive,
            whole_cell=self.whole_cell,
        )
        self.refresh_matches()
        candidates = [match for match in self._matches if match != target]
        if candidates:
            next_match = next((match for match in candidates if match > target), candidates[0])
            self._activate_match(next_match, keep_playhead=keep_playhead)
        else:
            self._active_match = None
            self._emit_state()
        if changed:
            self.status_message.emit("Replaced 1 cell")
        return changed

    def replace_all(self) -> int:
        """Replaces every match in scope as a single `Replace All` undo step without moving the playhead."""
        if not self.isVisible():
            return 0
        count = self._model.replace_all(
            self.query,
            self.replacement,
            columns=self._target_columns(),
            rows=self._target_rows(),
            case_sensitive=self.case_sensitive,
            whole_cell=self.whole_cell,
        )
        self._active_match = None
        self.refresh_matches()
        noun = "cell" if count == 1 else "cells"
        self.status_message.emit(f"Replaced {count} {noun}" if count else "No matches to replace")
        return count

    def refresh_matches(self) -> None:
        """Recomputes matches and updates the counter, button states, and table highlights."""
        self._sync_selected_rows_ui()
        if not self.isVisible():
            self._matches = ()
            self._active_match = None
            return
        raw_matches = self._model.find_matches(
            self.query,
            columns=self._target_columns(),
            rows=self._target_rows(),
            case_sensitive=self.case_sensitive,
            whole_cell=self.whole_cell,
        )
        self._matches = tuple((row, int(col)) for row, col in raw_matches)
        if self._active_match not in self._matches:
            self._active_match = None
        self._emit_state()

    def on_table_selection_changed(self, *, is_stepping: bool) -> None:
        """Updates **Selected rows only** availability and refreshes pinned multi-row selections."""
        selected = self._table_view.selected_rows()
        if not is_stepping and self.selected_rows_only and len(selected) > 1:
            self._pin_rows(selected)
            self.refresh_matches()
            return
        self._sync_selected_rows_ui()

    def on_model_reset(self) -> None:
        """Clears pinned row indices after a CSV load or session close and refreshes matches."""
        self._pinned_rows = ()
        if self._selected_rows_check.isChecked():
            self._set_selected_rows_checked_silently(False)
        self._active_match = None
        self.refresh_matches()

    @override
    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.KeyPress:
            assert isinstance(event, QtGui.QKeyEvent)
            if event.key() in _ENTER_KEYS:
                alt_held = _alt_is_held(event.modifiers())
                shift_held = bool(event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier)
                if watched is self._find_edit:
                    if shift_held:
                        self.find_previous(keep_playhead=alt_held)
                    else:
                        self.find_next(keep_playhead=alt_held)
                    return True
                if watched is self._replace_edit:
                    self.replace_current(keep_playhead=alt_held)
                    return True
        return super().eventFilter(watched, event)

    def _build_layout(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        content = QtWidgets.QWidget(self)
        rows_layout = QtWidgets.QVBoxLayout(content)
        rows_layout.setContentsMargins(6, 4, 6, 4)
        rows_layout.setSpacing(4)

        find_row = QtWidgets.QHBoxLayout()
        find_row.setContentsMargins(0, 0, 0, 0)
        find_row.setSpacing(4)
        find_row.addWidget(self._toggle_replace_button)
        find_row.addWidget(self._find_edit, stretch=1)
        find_row.addWidget(self._status_label)
        find_row.addWidget(self._prev_button)
        find_row.addWidget(self._next_button)
        find_row.addWidget(self._close_button)
        rows_layout.addLayout(find_row)

        replace_layout = QtWidgets.QHBoxLayout(self._replace_row)
        replace_layout.setContentsMargins(_TOGGLE_BUTTON_WIDTH + 4, 0, 0, 0)
        replace_layout.setSpacing(4)
        replace_layout.addWidget(self._replace_edit, stretch=1)
        replace_layout.addWidget(self._replace_button)
        replace_layout.addWidget(self._replace_all_button)
        rows_layout.addWidget(self._replace_row)

        options_row = QtWidgets.QHBoxLayout()
        options_row.setContentsMargins(_TOGGLE_BUTTON_WIDTH + 4, 0, 0, 0)
        options_row.setSpacing(10)
        options_row.addWidget(QtWidgets.QLabel("Column:", self))
        options_row.addWidget(self._column_combo)
        options_row.addWidget(self._match_case_check)
        options_row.addWidget(self._whole_cell_check)
        options_row.addWidget(self._selected_rows_check)
        options_row.addStretch(1)
        rows_layout.addLayout(options_row)

        separator = QtWidgets.QFrame(self)
        separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)

        outer.addWidget(content)
        outer.addWidget(separator)

    def _register_shortcuts(self) -> None:
        esc_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Escape), self)
        esc_shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        esc_shortcut.activated.connect(self.close_bar)

    def _set_replace_visible(self, visible: bool) -> None:
        self._replace_row.setVisible(visible)
        self._toggle_replace_button.setText("▾" if visible else "▸")

    def _on_toggle_replace_clicked(self) -> None:
        self._set_replace_visible(not self.is_replace_visible)
        if self.is_replace_visible and self.query:
            self._replace_edit.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
            self._replace_edit.selectAll()
        else:
            self._find_edit.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)

    def _on_search_criteria_changed(self, *_: object) -> None:
        self._active_match = None
        self.refresh_matches()

    def _on_selected_rows_toggled(self, checked: bool) -> None:
        if checked:
            self._pin_rows(self._table_view.selected_rows())
        else:
            self._pinned_rows = ()
        self._active_match = None
        self.refresh_matches()

    def _apply_selection_scope_on_open(self, *, was_visible: bool) -> None:
        selected = self._table_view.selected_rows()
        if len(selected) > 1:
            self._pin_rows(selected)
            self._set_selected_rows_checked_silently(True)
        elif not was_visible:
            self._pinned_rows = ()
            self._set_selected_rows_checked_silently(False)

    def _set_selected_rows_checked_silently(self, checked: bool) -> None:
        blocked = self._selected_rows_check.blockSignals(True)
        try:
            self._selected_rows_check.setChecked(checked)
        finally:
            self._selected_rows_check.blockSignals(blocked)

    def _pin_rows(self, rows: Sequence[int]) -> None:
        self._pinned_rows = tuple(
            QtCore.QPersistentModelIndex(self._model.index(row, 0))
            for row in sorted(set(rows))
            if 0 <= row < self._model.rowCount()
        )

    def _valid_pinned_rows(self) -> tuple[int, ...]:
        return tuple(sorted({idx.row() for idx in self._pinned_rows if idx.isValid()}))

    def _sync_selected_rows_ui(self) -> None:
        if self._selected_rows_check.isChecked():
            count = len(self._valid_pinned_rows())
            self._selected_rows_check.setText(f"Selected rows ({count})")
            self._selected_rows_check.setEnabled(True)
        else:
            self._selected_rows_check.setText("Selected rows only")
            self._selected_rows_check.setEnabled(bool(self._table_view.selected_rows()))

    def _target_columns(self) -> tuple[ballroom_scribe.gui.EventTableModel.Column, ...]:
        selected = self.selected_column
        if selected is not None:
            return (selected,)
        return ballroom_scribe.gui.EventTableModel.TEXT_COLUMNS

    def _target_rows(self) -> tuple[int, ...] | None:
        if self.selected_rows_only:
            return self._valid_pinned_rows()
        return None

    def _step_match(self, *, forward: bool) -> tuple[int, int]:
        if self._active_match in self._matches:
            index = self._matches.index(self._active_match)
            offset = 1 if forward else -1
            return self._matches[(index + offset) % len(self._matches)]

        current = self._table_view.currentIndex()
        if not current.isValid():
            return self._matches[0] if forward else self._matches[-1]
        anchor = (current.row(), current.column())
        if forward:
            return next((match for match in self._matches if match >= anchor), self._matches[0])
        return next((match for match in reversed(self._matches) if match <= anchor), self._matches[-1])

    def _resolvable_replace_target(self) -> tuple[int, int] | None:
        if self._active_match in self._matches:
            return self._active_match
        current = self._table_view.currentIndex()
        if current.isValid():
            candidate = (current.row(), current.column())
            if candidate in self._matches:
                return candidate
        return None

    def _activate_match(self, match: tuple[int, int], *, keep_playhead: bool) -> None:
        self._active_match = match
        self.match_activated.emit(match[0], match[1], keep_playhead)
        self._emit_state()

    def _emit_state(self) -> None:
        has_matches = bool(self._matches)
        self._prev_button.setEnabled(has_matches)
        self._next_button.setEnabled(has_matches)
        self._replace_button.setEnabled(has_matches)
        self._replace_all_button.setEnabled(has_matches)

        if not self.query:
            self._status_label.setText("")
        elif not self._matches:
            self._status_label.setText("No matches")
        elif self._active_match in self._matches:
            position = self._matches.index(self._active_match) + 1
            self._status_label.setText(f"{position} of {len(self._matches)}")
        else:
            noun = "match" if len(self._matches) == 1 else "matches"
            self._status_label.setText(f"{len(self._matches)} {noun}")

        if self.isVisible():
            self.highlights_changed.emit(frozenset(self._matches), self._active_match)
        else:
            self.highlights_changed.emit(frozenset(), None)
