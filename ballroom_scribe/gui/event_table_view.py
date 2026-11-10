"""Event table view with spreadsheet-style typing, cell clearing, and a fill handle on the current cell."""

from collections.abc import Sequence
from typing import override

from PySide6 import QtCore, QtGui, QtWidgets

import ballroom_scribe.gui
import ballroom_scribe.gui.utils

# The painted handle: a square tucked inside the cell's bottom-right corner, over the current-cell frame.
_HANDLE_SIZE = 6
# The press target is larger than the handle and centred on the cell's corner, so it is forgiving in every direction.
_HANDLE_HIT_SIZE = 12
_OUTLINE_WIDTH = 2
_AUTOSCROLL_INTERVAL_MS = 50
_TOOLTIP_OFFSET = QtCore.QPoint(16, 16)
_EDITING = QtWidgets.QAbstractItemView.State.EditingState
_TEXT_MODIFIERS = QtCore.Qt.KeyboardModifier.ShiftModifier | QtCore.Qt.KeyboardModifier.KeypadModifier
_CLEAR_KEYS = frozenset({QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace})


def _typed_character(text: str, modifiers: QtCore.Qt.KeyboardModifier) -> str:
    """Returns the printable character a key press types, or `""` for control keys and shortcuts."""
    if len(text) != 1 or not text.isprintable() or modifiers & ~_TEXT_MODIFIERS:
        return ""
    return text


def _paint_handle(painter: QtGui.QPainter, rect: QtCore.QRect, palette: QtGui.QPalette, *, alternate: bool) -> None:
    """Paints the handle as a highlight-coloured square set off from the cell frame by a one pixel gap."""
    gap = rect.adjusted(-1, -1, 0, 0)
    painter.fillRect(gap, palette.base())
    if alternate:
        painter.fillRect(gap, palette.alternateBase())
    painter.fillRect(rect, palette.highlight())


def _fill_tooltip(count: int, value: str) -> str:
    cells = "cell" if count == 1 else "cells"
    if count == 0:
        return f"Drag over the rows to fill with “{value}”" if value else "Drag over the rows to clear"
    return f"Fill {count} {cells} with “{value}”" if value else f"Clear {count} {cells}"


class EventTableView(QtWidgets.QTableView):
    """Spreadsheet-style event table with direct typing, cell clearing, and a fill handle on the current cell.

    While the table has keyboard focus:

    *   Typing a letter or digit claims the key from single-key window shortcuts (such as `K`, `J`, `L`, `I`, while
        leaving `Space`, `,`, `.`, and `Ctrl+...` shortcuts intact) and opens the current cell's editor with the
        typed character replacing its value (requires `EditTrigger.AnyKeyPressed`, set by `MainWindow`). Type-to-search
        is disabled so typing never jumps to another row.
    *   `Delete` and `Backspace` clear the current `Level`, `Style`, `Dance`, `Round`, or `Comment` cell as one undo
        step via `EventTableModel.clear_cell` instead of deleting the event.
    *   The current cell of a fillable column (see `EventTableModel.is_fillable`) shows a small square in its
        bottom-right corner. Pressing it and dragging up or down outlines the rows passed over and, on release, copies
        the cell's value into them through `EventTableModel.fill` as one undo step. The press never reaches
        `QTableView`, so it does not change the selection or current row. `Esc` cancels a drag.
    *   While the `Comment` column shows only presence icons (`EventTableModel.CommentDisplay.ICON`), a plain click on
        one of them opens the comment editor once the click has selected its row as usual.
    *   While two or more rows are selected (`has_multiple_selected_rows`), no single cell is singled out: the
        current-cell frame and fill handle are hidden, `selected_row()` returns `None` (disabling the faint
        same-event tint), and single-cell keys (`Delete`, `Backspace`, letters, digits) are accepted as no-ops.
    """

    escape_pressed = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        # Delivers mouse moves with no button pressed, so the pointer can change over the handle.
        self.viewport().setMouseTracking(True)
        self._fill_origin = QtCore.QPersistentModelIndex()
        self._fill_target_row = -1
        self._drag_position = QtCore.QPoint()
        self._pressed_comment_icon = QtCore.QPersistentModelIndex()
        self._search_matches: frozenset[tuple[int, int]] = frozenset()
        self._active_search_match: tuple[int, int] | None = None
        self._autoscroll_timer = QtCore.QTimer(self)
        self._autoscroll_timer.setInterval(_AUTOSCROLL_INTERVAL_MS)
        self._autoscroll_timer.timeout.connect(self._autoscroll)

    @property
    def is_filling(self) -> bool:
        """Returns whether a fill-handle drag is in progress."""
        return self._fill_origin.isValid()

    @property
    def search_matches(self) -> frozenset[tuple[int, int]]:
        """Returns the `(row, column)` cells currently highlighted by Find and Replace."""
        return self._search_matches

    @property
    def active_search_match(self) -> tuple[int, int] | None:
        """Returns the `(row, column)` cell currently stepped onto by Find and Replace, if any."""
        return self._active_search_match

    def set_search_highlights(
        self,
        matches: frozenset[tuple[int, int]],
        active_match: tuple[int, int] | None,
    ) -> None:
        """Updates the Find and Replace cell highlights and schedules a repaint when they change."""
        if matches != self._search_matches or active_match != self._active_search_match:
            self._search_matches = matches
            self._active_search_match = active_match
            self.viewport().update()

    def selected_rows(self) -> tuple[int, ...]:
        """Returns the selected rows in ascending order, or `()` when no row is selected."""
        selection_model = self.selectionModel()
        if selection_model is None:
            return ()
        return tuple(sorted({index.row() for index in selection_model.selectedRows()}))

    @property
    def has_multiple_selected_rows(self) -> bool:
        """Returns whether two or more rows are selected, in which case no single cell is singled out."""
        return len(self.selected_rows()) > 1

    def selected_row(self) -> int | None:
        """Returns the selected row, or `None` when no row or several rows are selected."""
        rows = self.selected_rows()
        return rows[0] if len(rows) == 1 else None

    def fill_handle_rect(self, index: QtCore.QModelIndex) -> QtCore.QRect:
        """Returns the fill handle's rectangle in viewport coordinates, or a null rectangle if `index` shows none."""
        handle_index = self._handle_index()
        if handle_index is None or handle_index != index:
            return QtCore.QRect()
        return self._handle_rect(handle_index)

    def setCurrentIndex(self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex) -> None:
        """Makes `index` the current cell and selects its row as the sole selected row."""
        selection_model = self.selectionModel()
        if selection_model is None:
            super().setCurrentIndex(index)
            return
        flag = (
            QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect | QtCore.QItemSelectionModel.SelectionFlag.Rows
            if index.isValid()
            else QtCore.QItemSelectionModel.SelectionFlag.Clear
        )
        selection_model.setCurrentIndex(index, flag)

    @override
    def selectionChanged(
        self,
        selected: QtCore.QItemSelection,
        deselected: QtCore.QItemSelection,
    ) -> None:
        super().selectionChanged(selected, deselected)
        # Repaints other visible rows whose same-event highlight changed with the selection.
        self.viewport().update()

    @override
    def currentChanged(self, current: QtCore.QModelIndex, previous: QtCore.QModelIndex) -> None:
        super().currentChanged(current, previous)
        if current.row() != previous.row():
            self.viewport().update()

    @override
    def dataChanged(
        self,
        topLeft: QtCore.QModelIndex,
        bottomRight: QtCore.QModelIndex,
        roles: Sequence[int] = (),
    ) -> None:
        super().dataChanged(topLeft, bottomRight, list(roles))
        self.viewport().update()

    @override
    def event(self, event: QtCore.QEvent) -> bool:
        if (
            event.type() == QtCore.QEvent.Type.ShortcutOverride
            and isinstance(event, QtGui.QKeyEvent)
            and self.currentIndex().isValid()
            and _typed_character(event.text(), event.modifiers()).isalnum()
        ):
            event.accept()  # Typed into the current cell (see `keyPressEvent`), not a single-key shortcut.
            return True
        return super().event(event)

    @override
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        handle_index = self._handle_index()
        if handle_index is None:
            return
        painter = QtGui.QPainter(self.viewport())
        try:
            if self.is_filling:
                origin = QtCore.QModelIndex(self._fill_origin)
                target = origin.siblingAtRow(self._fill_target_row)
                outline = self.visualRect(origin) | self.visualRect(target)
                ballroom_scribe.gui.utils.paint_frame(
                    painter, outline, self.palette().highlight().color(), _OUTLINE_WIDTH
                )
            alternate = self.alternatingRowColors() and handle_index.row() % 2 == 1
            _paint_handle(painter, self._handle_rect(handle_index), self.palette(), alternate=alternate)
        finally:
            painter.end()

    @override
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        position = event.position().toPoint()
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._handle_at(position) is not None:
            self._begin_fill(position)
            event.accept()
            return
        self._pressed_comment_icon = self._comment_icon_at(position, event)
        super().mousePressEvent(event)

    @override
    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        position = event.position().toPoint()
        if self.is_filling:
            self._drag_position = position
            self._update_fill_target()
            event.accept()
            return
        self._update_hover_cursor(position)
        super().mouseMoveEvent(event)

    @override
    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.is_filling:
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                self._finish_fill()
            event.accept()
            return
        pressed = QtCore.QModelIndex(self._pressed_comment_icon)
        self._pressed_comment_icon = QtCore.QPersistentModelIndex()
        super().mouseReleaseEvent(event)  # Selects the row (and jumps to its inpoint) as for any other cell.
        if (
            pressed.isValid()
            and event.button() == QtCore.Qt.MouseButton.LeftButton
            and self.indexAt(event.position().toPoint()) == pressed
            and self.state() != _EDITING
            and not self.has_multiple_selected_rows
        ):
            self.edit(pressed)

    @override
    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._handle_at(event.position().toPoint()) is not None:
            event.accept()  # Double-clicking the handle must not open the cell editor.
            return
        super().mouseDoubleClickEvent(event)

    @override
    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if self.is_filling:
            if event.key() == QtCore.Qt.Key.Key_Escape:
                self._cancel_fill()
            event.accept()  # Navigating during a drag would move the origin out from under it.
            return
        if self.state() != _EDITING:
            if event.key() == QtCore.Qt.Key.Key_Escape and not event.modifiers():
                self.escape_pressed.emit()
                event.accept()
                return
            single_cell = not self.has_multiple_selected_rows  # With several rows selected, no cell is singled out.
            if event.key() in _CLEAR_KEYS and not event.modifiers() & ~QtCore.Qt.KeyboardModifier.KeypadModifier:
                if single_cell:
                    self._clear_current_cell()
                event.accept()
                return
            typed = _typed_character(event.text(), event.modifiers())
            if typed:
                if typed.isalnum() and single_cell and self.currentIndex().isValid():
                    # Opens the editor and hands it this key press, which replaces the pre-selected value.
                    self.edit(self.currentIndex(), QtWidgets.QAbstractItemView.EditTrigger.AnyKeyPressed, event)
                event.accept()  # Never type-to-search; other printable keys do nothing here.
                return
        super().keyPressEvent(event)

    @override
    def keyboardSearch(self, search: str) -> None:
        del search  # Typing edits the current cell instead of jumping to a row whose text matches.

    @override
    def focusOutEvent(self, event: QtGui.QFocusEvent) -> None:
        if self.is_filling:
            self._cancel_fill()
        super().focusOutEvent(event)

    @override
    def viewportEvent(self, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.Leave and not self.is_filling:
            self.viewport().unsetCursor()
        return super().viewportEvent(event)

    def _clear_current_cell(self) -> None:
        model = self.model()
        if isinstance(model, ballroom_scribe.gui.EventTableModel):
            model.clear_cell(self.currentIndex())

    def _comment_icon_at(self, position: QtCore.QPoint, event: QtGui.QMouseEvent) -> QtCore.QPersistentModelIndex:
        """Returns the comment icon under a plain left press (no `Shift` / `Ctrl`), or an invalid index."""
        model = self.model()
        index = self.indexAt(position)
        if (
            event.button() != QtCore.Qt.MouseButton.LeftButton
            or event.modifiers()
            & (QtCore.Qt.KeyboardModifier.ShiftModifier | QtCore.Qt.KeyboardModifier.ControlModifier)
            or not index.isValid()
            or not isinstance(model, ballroom_scribe.gui.EventTableModel)
            or index.column() != int(ballroom_scribe.gui.EventTableModel.Column.COMMENT)
            or model.comment_display != ballroom_scribe.gui.EventTableModel.CommentDisplay.ICON
        ):
            return QtCore.QPersistentModelIndex()
        return QtCore.QPersistentModelIndex(index)

    def _handle_index(self) -> QtCore.QModelIndex | None:
        """Returns the cell showing a handle: the current cell of a fillable column while focused and not editing."""
        current = self.currentIndex()
        model = self.model()
        if (
            not current.isValid()
            or not isinstance(model, ballroom_scribe.gui.EventTableModel)
            or not model.is_fillable(current.column())
            or not self.hasFocus()
            or self.state() == _EDITING
            or self.has_multiple_selected_rows
        ):
            return None
        return current

    def _handle_at(self, position: QtCore.QPoint) -> QtCore.QModelIndex | None:
        """Returns the cell whose handle's press target contains `position`, if any."""
        handle_index = self._handle_index()
        if handle_index is not None and self._handle_hit_rect(handle_index).contains(position):
            return handle_index
        return None

    def _handle_rect(self, index: QtCore.QModelIndex) -> QtCore.QRect:
        corner = self.visualRect(index).bottomRight()
        return QtCore.QRect(corner.x() - _HANDLE_SIZE + 1, corner.y() - _HANDLE_SIZE + 1, _HANDLE_SIZE, _HANDLE_SIZE)

    def _handle_hit_rect(self, index: QtCore.QModelIndex) -> QtCore.QRect:
        corner = self.visualRect(index).bottomRight()
        half = _HANDLE_HIT_SIZE // 2
        return QtCore.QRect(corner.x() - half, corner.y() - half, _HANDLE_HIT_SIZE, _HANDLE_HIT_SIZE)

    def _update_hover_cursor(self, position: QtCore.QPoint) -> None:
        if self._handle_at(position) is not None:
            self.viewport().setCursor(QtCore.Qt.CursorShape.CrossCursor)
        else:
            self.viewport().unsetCursor()

    def _begin_fill(self, position: QtCore.QPoint) -> None:
        self._fill_origin = QtCore.QPersistentModelIndex(self.currentIndex())
        self._drag_position = position
        self.viewport().setCursor(QtCore.Qt.CursorShape.CrossCursor)
        self._update_fill_target()

    def _update_fill_target(self) -> None:
        """Re-derives the fill range from the pointer, autoscrolling while it is past the top or bottom edge."""
        model = self.model()
        if model is None:
            return
        y = self._drag_position.y()
        row = self.rowAt(y)
        if row < 0:
            row = 0 if y < 0 else model.rowCount() - 1
        self._fill_target_row = row
        if 0 <= y < self.viewport().height():
            self._autoscroll_timer.stop()
        elif not self._autoscroll_timer.isActive():
            self._autoscroll_timer.start()
        self._show_fill_tooltip()
        self.viewport().update()

    def _autoscroll(self) -> None:
        """Scrolls one row towards the pointer so a drag can continue past the visible rows."""
        model = self.model()
        if not self.is_filling or model is None:
            self._autoscroll_timer.stop()
            return
        if self._drag_position.y() < 0:
            row = self.rowAt(0) - 1
        else:
            edge_row = self.rowAt(self.viewport().height() - 1)
            row = edge_row + 1 if edge_row >= 0 else -1  # No row at the edge: everything below is already visible.
        if not 0 <= row < model.rowCount():
            self._autoscroll_timer.stop()
            return
        self.scrollTo(
            model.index(row, self._fill_origin.column()), QtWidgets.QAbstractItemView.ScrollHint.EnsureVisible
        )
        self._update_fill_target()

    def _show_fill_tooltip(self) -> None:
        count = abs(self._fill_target_row - self._fill_origin.row())
        value = str(self._fill_origin.data(QtCore.Qt.ItemDataRole.DisplayRole) or "")
        QtWidgets.QToolTip.showText(
            self.viewport().mapToGlobal(self._drag_position) + _TOOLTIP_OFFSET,
            _fill_tooltip(count, value),
            self.viewport(),
        )

    def _finish_fill(self) -> None:
        origin = QtCore.QModelIndex(self._fill_origin)
        target_row = self._fill_target_row
        self._end_fill_drag()
        model = self.model()
        if origin.isValid() and isinstance(model, ballroom_scribe.gui.EventTableModel) and target_row != origin.row():
            model.fill(origin, target_row)

    def _cancel_fill(self) -> None:
        self._end_fill_drag()

    def _end_fill_drag(self) -> None:
        self._fill_origin = QtCore.QPersistentModelIndex()
        self._fill_target_row = -1
        self._autoscroll_timer.stop()
        QtWidgets.QToolTip.hideText()
        self.viewport().unsetCursor()
        self.viewport().update()
