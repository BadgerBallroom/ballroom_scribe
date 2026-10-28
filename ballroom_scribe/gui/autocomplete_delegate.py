"""Item delegate providing a drop-down of the column's other values with case-insensitive filtering as you type."""

from collections.abc import Sequence
from typing import override

from PySide6 import QtCore, QtGui, QtWidgets

import ballroom_scribe.gui

# Imported directly rather than as `ballroom_scribe.gui.CurrentCellDelegate`: a base class is needed while this module
# is being imported, and under `EAGER_IMPORT=1` that happens before the package's lazy exports exist.
from .current_cell_delegate import CurrentCellDelegate

_NO_MODIFIER = QtCore.Qt.KeyboardModifier.NoModifier
_COMMIT_KEYS = frozenset(
    {
        QtCore.Qt.Key.Key_Return,
        QtCore.Qt.Key.Key_Enter,
        QtCore.Qt.Key.Key_Tab,
        QtCore.Qt.Key.Key_Backtab,
    }
)
_MAX_VISIBLE_SUGGESTIONS = 12


class _CompletingLineEdit(QtWidgets.QLineEdit):
    """Line edit whose completer lists suggestions as soon as editing starts and filters as the user types.

    `QCompleter` alone only pops up after the first typed character and swallows `Enter`/`Tab` while its popup is
    open, which forces a second key press to commit the cell and breaks tabbing along the row. This editor drops the
    list down immediately — unfiltered when opened on the existing cell value, or filtered by the character typed to
    start editing (and unfiltered again on `Down` or when the text is cleared) — applies the highlighted suggestion on
    `Enter`/`Tab`, and re-sends those keys to itself so the item view's delegate filter commits the cell or moves
    editing to the adjacent cell.
    """

    def __init__(
        self,
        suggestions: Sequence[str],
        *,
        advance_on_pick: bool,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initializes the editor.

        Args:
            suggestions: Values offered in the drop-down.
            advance_on_pick: Whether picking a suggestion with the mouse should also move editing to the next cell
                (as `Tab` does) instead of just committing, so a row can be filled in with successive clicks.
            parent: Parent widget.
        """
        super().__init__(parent)
        self._advance_on_pick = advance_on_pick
        self._completer = QtWidgets.QCompleter(list(suggestions), self)
        self._completer.setCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
        self._completer.setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
        self._completer.setCompletionMode(QtWidgets.QCompleter.CompletionMode.PopupCompletion)
        self._completer.setMaxVisibleItems(_MAX_VISIBLE_SUGGESTIONS)
        self.setCompleter(self._completer)
        # Installed after `QCompleter`'s own filter, so this one sees the popup's key presses first. Mouse picks are
        # handled on the popup's viewport rather than via `QCompleter.activated`: by the time that signal fires the
        # popup has been hidden, which on some platforms drops the editor's focus and closes it prematurely.
        self._completer.popup().installEventFilter(self)
        self._completer.popup().viewport().installEventFilter(self)

        self._popup_timer = QtCore.QTimer(self)
        self._popup_timer.setSingleShot(True)
        self._popup_timer.timeout.connect(self._refresh_popup)
        self.textEdited.connect(self._on_text_edited)

    @override
    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        # Defer until the view has positioned the editor so the popup opens underneath it.
        self._popup_timer.start(0)

    @override
    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Down and not self._completer.popup().isVisible():
            self._show_all_suggestions()
            event.accept()
            return
        super().keyPressEvent(event)

    @override
    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        popup = self._completer.popup()
        if watched is popup and event.type() == QtCore.QEvent.Type.KeyPress:
            assert isinstance(event, QtGui.QKeyEvent)
            if event.key() in _COMMIT_KEYS:
                # The popup is not hidden here: hiding it can drop focus, which makes the delegate commit and
                # close the editor before the re-sent key is handled. Closing the editor tears the popup down.
                self._apply_highlighted_suggestion()
                self._resend_key(event.key(), event.modifiers(), event.text())
                return True
        elif watched is popup.viewport() and event.type() == QtCore.QEvent.Type.MouseButtonRelease:
            assert isinstance(event, QtGui.QMouseEvent)
            picked = popup.indexAt(event.position().toPoint())
            if event.button() == QtCore.Qt.MouseButton.LeftButton and picked.isValid():
                self.setText(str(picked.data()))
                if self._advance_on_pick:
                    self._resend_key(QtCore.Qt.Key.Key_Tab, _NO_MODIFIER, "\t")
                else:
                    self._resend_key(QtCore.Qt.Key.Key_Return, _NO_MODIFIER, "")
                return True
        return super().eventFilter(watched, event)

    def _refresh_popup(self) -> None:
        """Drops the list down filtered by what was typed, or with every suggestion when nothing has been typed."""
        if not self.isVisible():
            return
        # Opened with `Enter`, `F2`, or a double-click, the editor still shows the cell's value (`isModified()` is
        # false) and lists everything; opened by typing into the table, the typed text has replaced it.
        self._completer.setCompletionPrefix(self.text() if self.isModified() else "")
        self._completer.complete()

    def _show_all_suggestions(self) -> None:
        if self.isVisible():
            self._completer.setCompletionPrefix("")
            self._completer.complete()

    def _on_text_edited(self, text: str) -> None:
        # `QLineEdit` hides the popup once the text is empty; bring the full list back instead.
        if not text:
            self._popup_timer.start(0)

    def _apply_highlighted_suggestion(self) -> None:
        current = self._completer.popup().currentIndex()
        if current.isValid():
            self.setText(str(current.data()))

    def _resend_key(self, key: int, modifiers: QtCore.Qt.KeyboardModifier, text: str) -> None:
        """Delivers a key press to this editor so the view's delegate filter handles committing."""
        if not self.isVisible():
            return  # The view already closed this editor (e.g. after a focus change).
        QtCore.QCoreApplication.sendEvent(self, QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, key, modifiers, text))


class AutocompleteDelegate(CurrentCellDelegate):
    """Creates a line edit that drops down the other rows' values in the same column and filters them as you type.

    Picking a value with the mouse moves editing to the next cell so a row can be filled in with successive clicks;
    in the last column it just commits. Cells are painted by `CurrentCellDelegate`, so the current cell stays
    recognizable while the table has keyboard focus.
    """

    @override
    def createEditor(
        self,
        parent: QtWidgets.QWidget,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ) -> QtWidgets.QWidget:
        del option
        model = index.model()
        if isinstance(model, ballroom_scribe.gui.EventTableModel):
            suggestions = model.unique_values(index.column(), exclude_row=index.row())
            if suggestions:
                is_last_column = index.column() == model.columnCount() - 1
                return _CompletingLineEdit(suggestions, advance_on_pick=not is_last_column, parent=parent)
        return QtWidgets.QLineEdit(parent)
