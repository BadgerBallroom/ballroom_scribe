"""Item delegate for the Comment column: text or a presence icon in the cell, edited in a multi-line overlay."""

from typing import override

from PySide6 import QtCore, QtGui, QtWidgets

# Imported directly rather than as `ballroom_scribe.gui.CurrentCellDelegate`: a base class is needed while this module
# is being imported, and under `EAGER_IMPORT=1` that happens before the package's lazy exports exist.
from .current_cell_delegate import CurrentCellDelegate

_EDITOR_MIN_WIDTH = 280
_EDITOR_LINES = 4
_COMMIT_KEYS = frozenset({QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter})


def _overlay_rect(editor: QtWidgets.QPlainTextEdit, cell: QtCore.QRect) -> QtCore.QRect:
    """Returns the editor's geometry: anchored to `cell`, wide and tall enough to read, inside the viewport.

    The editor's right edge lines up with the cell's, so from the rightmost column it extends left over the
    neighbouring cells; its top lines up with the cell's unless that would run past the bottom of the viewport, in
    which case it grows upward instead.
    """
    parent = editor.parentWidget()
    area = parent.rect() if parent is not None and parent.rect().isValid() else cell
    margin = int(editor.document().documentMargin())
    height = _EDITOR_LINES * editor.fontMetrics().lineSpacing() + 2 * (editor.frameWidth() + margin)
    height = min(height, area.height())
    width = min(max(_EDITOR_MIN_WIDTH, cell.width()), area.width())
    left = max(0, min(cell.right() + 1 - width, area.width() - width))
    top = cell.top()
    if top + height > area.height():
        top = max(0, cell.bottom() + 1 - height)
    return QtCore.QRect(left, top, width, height)


class _CommentEditor(QtWidgets.QPlainTextEdit):
    """Word-wrapping overlay editor: `Enter` commits, `Shift+Enter` starts a new line, `Esc` reverts.

    `QStyledItemDelegate`'s editor filter deliberately leaves `Enter` to text edits, so this editor asks the delegate
    to commit through `commit_requested` instead; `Tab`, `Esc`, and focus changes are handled by that filter.
    """

    commit_requested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setTabChangesFocus(True)
        self.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.setPlaceholderText("Comment (Shift+Enter for a new line)")

    @override
    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() in _COMMIT_KEYS:
            if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
                self.insertPlainText("\n")
            else:
                self.commit_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class CommentDelegate(CurrentCellDelegate):
    """Paints the `Comment` cell and edits it in an overlay that fits several lines of text.

    The model decides what the cell shows (the comment's first line, or in icon mode a centred presence icon, see
    `EventTableModel.set_comment_display`). The editor is an ordinary item editor — a child of the viewport sized by
    `_overlay_rect` — so the view's usual commit, revert, `Tab`, and "an editor is open" handling all apply.
    """

    @override
    def initStyleOption(
        self,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ) -> None:
        super().initStyleOption(option, index)
        option.decorationAlignment = QtCore.Qt.AlignmentFlag.AlignCenter

    @override
    def createEditor(
        self,
        parent: QtWidgets.QWidget,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ) -> QtWidgets.QWidget:
        del option, index
        editor = _CommentEditor(parent)
        editor.commit_requested.connect(lambda: self._commit(editor))
        return editor

    @override
    def updateEditorGeometry(
        self,
        editor: QtWidgets.QWidget,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ) -> None:
        if isinstance(editor, QtWidgets.QPlainTextEdit):
            editor.setGeometry(_overlay_rect(editor, option.rect))
            return
        super().updateEditorGeometry(editor, option, index)

    @override
    def setEditorData(
        self, editor: QtWidgets.QWidget, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex
    ) -> None:
        if isinstance(editor, QtWidgets.QPlainTextEdit):
            editor.setPlainText(str(index.data(QtCore.Qt.ItemDataRole.EditRole) or ""))
            editor.selectAll()  # As in a line edit, typing replaces the value; `End` or a click keeps it.
            return
        super().setEditorData(editor, index)

    @override
    def setModelData(
        self,
        editor: QtWidgets.QWidget,
        model: QtCore.QAbstractItemModel,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ) -> None:
        if isinstance(editor, QtWidgets.QPlainTextEdit):
            model.setData(index, editor.toPlainText(), QtCore.Qt.ItemDataRole.EditRole)
            return
        super().setModelData(editor, model, index)

    def _commit(self, editor: QtWidgets.QWidget) -> None:
        self.commitData.emit(editor)
        self.closeEditor.emit(editor, QtWidgets.QAbstractItemDelegate.EndEditHint.NoHint)
