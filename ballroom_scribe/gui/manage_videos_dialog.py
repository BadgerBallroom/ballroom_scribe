"""Dialog listing the session's videos so the user can reorder them or remove some before applying."""

import pathlib
from collections.abc import Sequence
from typing import override

from PySide6 import QtCore, QtGui, QtWidgets

import ballroom_scribe.gui.models

_CSV_FILE_ROLE = int(QtCore.Qt.ItemDataRole.UserRole)
_ROW_COUNT_ROLE = int(QtCore.Qt.ItemDataRole.UserRole) + 1
_LABEL_ROLE = int(QtCore.Qt.ItemDataRole.UserRole) + 2
_LIST_MIN_SIZE = QtCore.QSize(420, 220)


def _row_count_text(row_count: int) -> str:
    return "1 row" if row_count == 1 else f"{row_count} rows"


class ManageVideosDialog(QtWidgets.QDialog):
    """Lists the session's videos in Video drop-down order and lets the user reorder or remove them before applying.

    Each entry shows the drop-down label and how many table rows belong to the video; ▲ / ▼ or dragging rearranges.
    The Remove button asks for confirmation, warning that the video's rows will be deleted, then takes the entry off
    the list. Nothing is applied until OK: `order()` is the remaining videos in list order and `removed()` the ones
    taken off; Cancel discards everything.
    """

    def __init__(
        self,
        entries: Sequence[ballroom_scribe.gui.models.VideoEntry],
        current: pathlib.Path | None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Manage Videos")
        self._removed: list[pathlib.Path] = []

        self._list = QtWidgets.QListWidget(self)
        self._list.setMinimumSize(_LIST_MIN_SIZE)
        self._list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self._list.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self._list.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
        for entry in entries:
            item = QtWidgets.QListWidgetItem(f"{entry.label}  —  {_row_count_text(entry.row_count)}")
            item.setData(_CSV_FILE_ROLE, entry.csv_file)
            item.setData(_ROW_COUNT_ROLE, entry.row_count)
            item.setData(_LABEL_ROLE, entry.label)
            item.setToolTip(str(entry.csv_file))
            item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsDropEnabled)  # Reorder only; no nesting.
            self._list.addItem(item)
            if current is not None and entry.csv_file == current:
                self._list.setCurrentItem(item)
        if self._list.currentRow() < 0 and self._list.count():
            self._list.setCurrentRow(0)
        self._list.currentRowChanged.connect(lambda _row: self._update_buttons())
        self._list.model().rowsInserted.connect(self._on_rows_inserted)

        style = self.style()
        self._up_button = QtWidgets.QToolButton(self)
        self._up_button.setIcon(style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ArrowUp))
        self._up_button.setToolTip("Move the selected video up")
        self._up_button.clicked.connect(lambda: self.move_selected(-1))

        self._down_button = QtWidgets.QToolButton(self)
        self._down_button.setIcon(style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ArrowDown))
        self._down_button.setToolTip("Move the selected video down")
        self._down_button.clicked.connect(lambda: self.move_selected(1))

        self._remove_button = QtWidgets.QToolButton(self)
        self._remove_button.setText("Remove")
        self._remove_button.setToolTip("Remove the selected video from the session together with its rows")
        self._remove_button.clicked.connect(self._confirm_and_remove_selected)

        buttons_layout = QtWidgets.QVBoxLayout()
        buttons_layout.addWidget(self._up_button)
        buttons_layout.addWidget(self._down_button)
        buttons_layout.addSpacing(12)
        buttons_layout.addWidget(self._remove_button)
        buttons_layout.addStretch(1)

        list_layout = QtWidgets.QHBoxLayout()
        list_layout.addWidget(self._list, stretch=1)
        list_layout.addLayout(buttons_layout)

        self._button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)

        hint = QtWidgets.QLabel(
            "Drag or use ▲ / ▼ to change the Video drop-down order; rows regroup by video in that order. Removing a"
            " video also deletes its rows. Changes are applied when you press OK.",
            self,
        )
        hint.setWordWrap(True)

        root_layout = QtWidgets.QVBoxLayout(self)
        root_layout.addWidget(hint)
        root_layout.addLayout(list_layout, stretch=1)
        root_layout.addWidget(self._button_box)
        self._update_buttons()

    def order(self) -> tuple[pathlib.Path, ...]:
        """Returns the remaining videos' keys in list order."""
        return tuple(self._list.item(row).data(_CSV_FILE_ROLE) for row in range(self._list.count()))

    def removed(self) -> tuple[pathlib.Path, ...]:
        """Returns the keys of the videos taken off the list, in the order they were removed."""
        return tuple(self._removed)

    def move_selected(self, offset: int) -> None:
        """Moves the selected video `offset` places (negative = up) and keeps it selected."""
        row = self._list.currentRow()
        target = row + offset
        if row < 0 or not 0 <= target < self._list.count():
            return
        item = self._list.takeItem(row)
        self._list.insertItem(target, item)
        self._list.setCurrentItem(item)
        self._update_buttons()

    def remove_selected(self) -> None:
        """Takes the selected video off the list without asking (the Remove button confirms first)."""
        row = self._list.currentRow()
        if row < 0:
            return
        item = self._list.takeItem(row)
        self._removed.append(item.data(_CSV_FILE_ROLE))
        if self._list.count():
            self._list.setCurrentRow(min(row, self._list.count() - 1))
        self._update_buttons()

    @override
    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        # `Enter` in the list must not accept the dialog behind the user's back.
        if event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter) and self._list.hasFocus():
            event.accept()
            return
        super().keyPressEvent(event)

    def _confirm_and_remove_selected(self) -> None:
        item = self._list.currentItem()
        if item is None:
            return
        row_count = int(item.data(_ROW_COUNT_ROLE))
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setWindowTitle("Remove Video?")
        box.setText(f"Remove {item.data(_LABEL_ROLE)} from the session?")
        if row_count:
            box.setInformativeText(f"Its {_row_count_text(row_count)} will be deleted from the table.")
        box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
        box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.No)
        if box.exec() == QtWidgets.QMessageBox.StandardButton.Yes:
            self.remove_selected()

    def _on_rows_inserted(self, _parent: QtCore.QModelIndex, first: int, _last: int) -> None:
        # A drag-and-drop move re-inserts the dragged item at its new place; keep it selected there, as ▲ / ▼ do.
        self._list.setCurrentRow(first)
        self._update_buttons()

    def _update_buttons(self) -> None:
        row = self._list.currentRow()
        count = self._list.count()
        self._up_button.setEnabled(row > 0)
        self._down_button.setEnabled(0 <= row < count - 1)
        self._remove_button.setEnabled(row >= 0)
