"""Qt table model and undo/redo commands for editing `EventRecord` rows."""

import dataclasses
import enum
import pathlib
import re
from collections.abc import Callable, Iterable, Sequence
from typing import Any, override

from PySide6 import QtCore, QtGui

import ballroom_scribe
import ballroom_scribe.gui.utils
import ballroom_scribe.models
import ballroom_scribe.utils

_INVALID_INDEX = QtCore.QModelIndex()


class _ReplaceRecordCommand(QtGui.QUndoCommand):
    """Undoable replacement of a single row's `EventRecord`."""

    def __init__(
        self,
        model: EventTableModel,
        row: int,
        old_record: ballroom_scribe.models.EventRecord,
        new_record: ballroom_scribe.models.EventRecord,
        description: str,
    ) -> None:
        super().__init__(description)
        self._model = model
        self._row = row
        self._old_record = old_record
        self._new_record = new_record

    @override
    def redo(self) -> None:
        self._model._apply_replace(self._row, self._new_record)

    @override
    def undo(self) -> None:
        self._model._apply_replace(self._row, self._old_record)


class _MoveRecordCommand(QtGui.QUndoCommand):
    """Undoable move of a row to another position, pushed alongside a replacement that changes its ordering."""

    def __init__(self, model: EventTableModel, from_row: int, to_row: int) -> None:
        super().__init__("Move Event")
        self._model = model
        self._from_row = from_row
        self._to_row = to_row

    @override
    def redo(self) -> None:
        self._model._apply_move(self._from_row, self._to_row)

    @override
    def undo(self) -> None:
        self._model._apply_move(self._to_row, self._from_row)


class _InsertRecordCommand(QtGui.QUndoCommand):
    """Undoable insertion of a new `EventRecord` row."""

    def __init__(
        self,
        model: EventTableModel,
        row: int,
        record: ballroom_scribe.models.EventRecord,
    ) -> None:
        super().__init__("Add Event")
        self._model = model
        self._row = row
        self._record = record

    @override
    def redo(self) -> None:
        self._model._apply_insert(self._row, self._record)

    @override
    def undo(self) -> None:
        self._model._apply_remove(self._row)


class _RemoveRecordCommand(QtGui.QUndoCommand):
    """Undoable removal of an existing `EventRecord` row."""

    def __init__(
        self,
        model: EventTableModel,
        row: int,
        record: ballroom_scribe.models.EventRecord,
    ) -> None:
        super().__init__("Delete Event")
        self._model = model
        self._row = row
        self._record = record

    @override
    def redo(self) -> None:
        self._model._apply_remove(self._row)

    @override
    def undo(self) -> None:
        self._model._apply_insert(self._row, self._record)


def _column_text_value(record: ballroom_scribe.models.EventRecord, column: EventTableModel.Column) -> str:
    """Returns the string field of `record` corresponding to a text `Column`."""
    match column:
        case EventTableModel.Column.FILE:
            return str(record.file)
        case EventTableModel.Column.INPOINT:
            return f"{record.inpoint_seconds:.3f}"
        case EventTableModel.Column.INPOINT_FORMATTED:
            return record.inpoint_formatted
        case EventTableModel.Column.LEVEL:
            return record.level
        case EventTableModel.Column.STYLE:
            return record.style
        case EventTableModel.Column.DANCE:
            return record.dance
        case EventTableModel.Column.ROUND:
            return record.round
        case EventTableModel.Column.COMMENT:
            return record.comment


def _with_text_value(
    record: ballroom_scribe.models.EventRecord,
    column: EventTableModel.Column,
    text: str,
) -> ballroom_scribe.models.EventRecord:
    """Returns `record` with the free-text field of `column` set to `text`.

    Raises:
        ValueError: If `column` is not one of the free-text columns.
    """
    match column:
        case EventTableModel.Column.LEVEL:
            return dataclasses.replace(record, level=text)
        case EventTableModel.Column.STYLE:
            return dataclasses.replace(record, style=text)
        case EventTableModel.Column.DANCE:
            return dataclasses.replace(record, dance=text)
        case EventTableModel.Column.ROUND:
            return dataclasses.replace(record, round=text)
        case EventTableModel.Column.COMMENT:
            return dataclasses.replace(record, comment=text)
        case _:
            raise ValueError(f"Column {column.name} has no free-text value.")


def _comment_summary(comment: str) -> str:
    """Returns the first line of `comment`, marking further lines with an ellipsis, for a single-line cell."""
    lines = comment.splitlines()
    if not lines:
        return ""
    return f"{lines[0]} …" if len(lines) > 1 else lines[0]


def _rows_between(first: int, last: int) -> range:
    """Returns the rows after `first` up to and including `last`, counting up or down as needed."""
    step = 1 if last >= first else -1
    return range(first + step, last + step, step)


def _event_key(record: ballroom_scribe.models.EventRecord) -> tuple[str, str, str] | None:
    """Returns the case-folded `(level, style, dance)` key of `record`, or `None` when any of the three is empty."""
    level = record.level.strip()
    style = record.style.strip()
    dance = record.dance.strip()
    if not (level and style and dance):
        return None
    return (level.casefold(), style.casefold(), dance.casefold())


def _compile_search_pattern(query: str, *, case_sensitive: bool) -> re.Pattern[str]:
    """Compiles a literal regular expression for `query` with optional case sensitivity."""
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(re.escape(query), flags)


def _cell_matches(text: str, pattern: re.Pattern[str], *, whole_cell: bool) -> bool:
    """Returns whether `text` matches `pattern` across the whole cell or as a substring."""
    match = pattern.fullmatch(text) if whole_cell else pattern.search(text)
    return match is not None


def _replace_cell_text(text: str, pattern: re.Pattern[str], replacement: str, *, whole_cell: bool) -> str:
    """Replaces matches of `pattern` in `text` with literal `replacement` and strips outer whitespace."""
    replaced = replacement if whole_cell else pattern.sub(lambda _: replacement, text)
    return replaced.strip()


class EventTableModel(QtCore.QAbstractTableModel):
    """Editable table model backed by a mutable `list[EventRecord]` and a `QUndoStack`.

    Rows are kept grouped by video and in `Inpoint` order: `add_record` places a new event accordingly, and an
    edit that changes a row's `File` or `Inpoint` (`set_inpoint`, or committing a typed inpoint through
    `setData`) moves the row to where it now belongs as part of the same undo step. Which rows belong to the
    same video, and the order of the videos, comes from `video_rank`.

    The `Comment` column is free text that may span several lines. It is presented either as text (its first
    line, with an ellipsis when there is more) or, when the view is too narrow (`set_comment_display`), as an icon
    telling whether a comment exists; the full comment is always available through `EditRole` and `ToolTipRole`.
    """

    class Column(enum.IntEnum):
        FILE = 0
        INPOINT = 1
        INPOINT_FORMATTED = 2
        LEVEL = 3
        STYLE = 4
        DANCE = 5
        ROUND = 6
        COMMENT = 7

    class CommentDisplay(enum.Enum):
        """How `Column.COMMENT` presents itself: the comment text, or an icon telling whether one exists."""

        TEXT = "text"
        ICON = "icon"

    # The columns describing the event: autocompleted, fillable, and sharing the table's spare width.
    TEXT_COLUMNS = (
        Column.LEVEL,
        Column.STYLE,
        Column.DANCE,
        Column.ROUND,
    )
    # Every free-text column, i.e. the scope of Find and Replace.
    SEARCHABLE_COLUMNS = (*TEXT_COLUMNS, Column.COMMENT)
    _EDITABLE_COLUMNS = frozenset(
        {
            Column.INPOINT,
            *SEARCHABLE_COLUMNS,
        }
    )
    # Columns whose value can be copied into neighbouring rows with the fill handle; an inpoint never should be, and
    # a comment cell may be only an icon wide.
    _FILLABLE_COLUMNS = frozenset(TEXT_COLUMNS)
    # Columns that may be left empty; `Inpoint` is required and `File` / `Inpoint Formatted` are read-only.
    _CLEARABLE_COLUMNS = frozenset(SEARCHABLE_COLUMNS)
    _RIGHT_ALIGNED_COLUMNS = frozenset(
        {
            Column.INPOINT,
            Column.INPOINT_FORMATTED,
        }
    )

    def __init__(
        self,
        parent: QtCore.QObject | None = None,
        video_rank: Callable[[pathlib.Path], int | None] | None = None,
    ) -> None:
        """Initializes an empty model.

        Args:
            parent: Owning `QObject`.
            video_rank: Maps a `File` path to the position of its video in the session (`None` when the path
                matches no loaded video), used to keep each video's rows together and in `Inpoint` order.
                Defaults to grouping rows by their exact `File` path in order of first appearance.
        """
        super().__init__(parent)
        self._records: list[ballroom_scribe.models.EventRecord] = []
        self._undo_stack = QtGui.QUndoStack(self)
        self._removing_rows = False
        self._video_rank = video_rank
        self._comment_display = self.CommentDisplay.TEXT

    @property
    def is_removing_rows(self) -> bool:
        """Returns whether a row removal is in progress (between `rowsAboutToBeRemoved` and `rowsRemoved`).

        Views move their current row to a neighbour while a row is being removed; the editor checks this flag to
        keep the playhead where it is instead of treating that move as the user selecting the neighbour.
        """
        return self._removing_rows

    @property
    def comment_display(self) -> CommentDisplay:
        """Returns how the `Comment` column is presented (see `set_comment_display`)."""
        return self._comment_display

    def set_comment_display(self, display: CommentDisplay) -> None:
        """Presents the `Comment` column as text or as a presence icon, notifying views when that changes.

        In `ICON` mode the column's cells and header show an icon through `DecorationRole` and no `DisplayRole`
        text, which suits a column only an icon wide; the comment itself stays available via `EditRole` and
        `ToolTipRole`.
        """
        if display == self._comment_display:
            return
        self._comment_display = display
        column = int(self.Column.COMMENT)
        self.headerDataChanged.emit(QtCore.Qt.Orientation.Horizontal, column, column)
        if self._records:
            self.dataChanged.emit(
                self.index(0, column),
                self.index(len(self._records) - 1, column),
                [QtCore.Qt.ItemDataRole.DisplayRole, QtCore.Qt.ItemDataRole.DecorationRole],
            )

    @classmethod
    def is_fillable(cls, column: Column | int) -> bool:
        """Returns whether `column` offers the fill handle (the free-text columns)."""
        return cls.Column(column) in cls._FILLABLE_COLUMNS

    @property
    def undo_stack(self) -> QtGui.QUndoStack:
        """Returns the model's `QUndoStack`."""
        return self._undo_stack

    @property
    def video_rank(self) -> Callable[[pathlib.Path], int | None]:
        """Returns the function ranking a `File` path's video (see `__init__`)."""
        return self._video_rank if self._video_rank is not None else self._rank_by_first_appearance

    @property
    def records(self) -> Sequence[ballroom_scribe.models.EventRecord]:
        """Returns a snapshot copy of the current `EventRecord` rows."""
        return list(self._records)

    def record_at(self, row: int) -> ballroom_scribe.models.EventRecord:
        """Returns the `EventRecord` at `row`."""
        return self._records[row]

    def set_records(self, records: Iterable[ballroom_scribe.models.EventRecord]) -> None:
        """Replaces all rows in the table and resets the undo history to a clean state."""
        materialized = list(records)
        self.beginResetModel()
        self._records = materialized
        self.endResetModel()
        self._undo_stack.clear()
        self._undo_stack.setClean()

    @override
    def rowCount(
        self,
        parent: QtCore.QModelIndex | QtCore.QPersistentModelIndex = _INVALID_INDEX,
    ) -> int:
        if parent.isValid():
            return 0
        return len(self._records)

    @override
    def columnCount(
        self,
        parent: QtCore.QModelIndex | QtCore.QPersistentModelIndex = _INVALID_INDEX,
    ) -> int:
        if parent.isValid():
            return 0
        return len(ballroom_scribe.EventRecorder.COLUMNS)

    @override
    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if orientation == QtCore.Qt.Orientation.Vertical:
            return str(section + 1) if role == QtCore.Qt.ItemDataRole.DisplayRole else None
        if not 0 <= section < len(ballroom_scribe.EventRecorder.COLUMNS):
            return None
        name = ballroom_scribe.EventRecorder.COLUMNS[section]
        if section == int(self.Column.COMMENT):
            compact = self._comment_display == self.CommentDisplay.ICON
            if role == QtCore.Qt.ItemDataRole.DisplayRole:
                return "" if compact else name
            if role == QtCore.Qt.ItemDataRole.DecorationRole and compact:
                return ballroom_scribe.gui.utils.comment_icon(False)
            if role == QtCore.Qt.ItemDataRole.ToolTipRole:
                return name
            return None
        return name if role == QtCore.Qt.ItemDataRole.DisplayRole else None

    @override
    def flags(self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex) -> QtCore.Qt.ItemFlag:
        if not index.isValid():
            return QtCore.Qt.ItemFlag.NoItemFlags
        base_flags = super().flags(index)
        if self.Column(index.column()) in self._EDITABLE_COLUMNS:
            return base_flags | QtCore.Qt.ItemFlag.ItemIsEditable
        return base_flags

    @override
    def data(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if not index.isValid() or not 0 <= index.row() < len(self._records):
            return None
        record = self._records[index.row()]
        column = self.Column(index.column())

        if column == self.Column.COMMENT:
            return self._comment_data(record, role)
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            return _column_text_value(record, column)
        if role == QtCore.Qt.ItemDataRole.EditRole:
            if column == self.Column.INPOINT:
                return str(record.inpoint_seconds)
            return _column_text_value(record, column)
        if role == QtCore.Qt.ItemDataRole.ToolTipRole and column == self.Column.FILE:
            return str(record.file)
        if role == QtCore.Qt.ItemDataRole.TextAlignmentRole and column in self._RIGHT_ALIGNED_COLUMNS:
            return int(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        return None

    def _comment_data(self, record: ballroom_scribe.models.EventRecord, role: int) -> Any:
        """Returns the `Comment` cell's data for `role`, honouring `comment_display`."""
        compact = self._comment_display == self.CommentDisplay.ICON
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            return "" if compact else _comment_summary(record.comment)
        if role == QtCore.Qt.ItemDataRole.EditRole:
            return record.comment
        if role == QtCore.Qt.ItemDataRole.DecorationRole and compact:
            return ballroom_scribe.gui.utils.comment_icon(bool(record.comment))
        if role == QtCore.Qt.ItemDataRole.ToolTipRole:
            if record.comment:
                return record.comment
            return "Click to add a comment" if compact else None
        return None

    @override
    def setData(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        value: Any,
        role: int = QtCore.Qt.ItemDataRole.EditRole,
    ) -> bool:
        if (
            not index.isValid()
            or role != QtCore.Qt.ItemDataRole.EditRole
            or value is None
            or not 0 <= index.row() < len(self._records)
        ):
            return False

        row = index.row()
        column = self.Column(index.column())
        if column not in self._EDITABLE_COLUMNS:
            return False

        old_record = self._records[row]
        text = str(value).strip()

        if column == self.Column.INPOINT:
            try:
                seconds = ballroom_scribe.utils.parse_duration(text)
            except ValueError:
                return False
            new_record = dataclasses.replace(old_record, inpoint_seconds=seconds)
        else:
            if column == self.Column.COMMENT:
                text = text.replace("\r\n", "\n").replace("\r", "\n")  # Comments may span lines; store them uniformly.
            new_record = _with_text_value(old_record, column, text)

        if new_record == old_record:
            return False

        header_name = ballroom_scribe.EventRecorder.COLUMNS[int(column)]
        self._push_replacement(
            row,
            old_record,
            new_record,
            f"Edit {header_name}",
            reorder=column == self.Column.INPOINT,
        )
        return True

    def clear_cell(self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex) -> bool:
        """Empties a `Level`, `Style`, `Dance`, `Round`, or `Comment` cell as an undoable command.

        Returns:
            Whether the cell changed. Other columns, invalid indexes, and empty cells are left alone, and nothing is
            pushed onto the undo stack.
        """
        if not index.isValid() or not 0 <= index.row() < len(self._records):
            return False
        column = self.Column(index.column())
        if column not in self._CLEARABLE_COLUMNS:
            return False
        old_record = self._records[index.row()]
        new_record = _with_text_value(old_record, column, "")
        if new_record == old_record:
            return False
        header_name = ballroom_scribe.EventRecorder.COLUMNS[int(column)]
        self._undo_stack.push(_ReplaceRecordCommand(self, index.row(), old_record, new_record, f"Clear {header_name}"))
        return True

    def fill(self, source: QtCore.QModelIndex | QtCore.QPersistentModelIndex, last_row: int) -> int:
        """Copies `source`'s value into its column from `source.row()` through `last_row`, as one undo step.

        Args:
            source: The cell whose value is copied; must be in a fillable column.
            last_row: The far end of the range (inclusive); may be above or below `source`.

        Returns:
            The number of cells whose value changed. Nothing is pushed onto the undo stack when that is zero.

        Raises:
            ValueError: If `source` is not a cell of a fillable column.
            IndexError: If `last_row` is out of range.
        """
        if not source.isValid() or not self.is_fillable(source.column()):
            raise ValueError(f"Cell {source.row()}, {source.column()} cannot be filled from.")
        if not 0 <= last_row < len(self._records):
            raise IndexError(f"Row index {last_row} out of range.")
        column = self.Column(source.column())
        value = _column_text_value(self._records[source.row()], column)
        replacements: list[tuple[int, ballroom_scribe.models.EventRecord, ballroom_scribe.models.EventRecord]] = []
        for row in _rows_between(source.row(), last_row):
            old_record = self._records[row]
            new_record = _with_text_value(old_record, column, value)
            if new_record != old_record:
                replacements.append((row, old_record, new_record))
        if not replacements:
            return 0

        header_name = ballroom_scribe.EventRecorder.COLUMNS[int(column)]
        self._undo_stack.beginMacro(f"Fill {header_name}")
        try:
            for row, old_record, new_record in replacements:
                self._undo_stack.push(_ReplaceRecordCommand(self, row, old_record, new_record, f"Edit {header_name}"))
        finally:
            self._undo_stack.endMacro()
        return len(replacements)

    def set_inpoint(self, row: int, file: pathlib.Path, inpoint_seconds: float) -> None:
        """Overwrites `File`, `Inpoint`, and `Inpoint Formatted` of `row` as an undoable command."""
        if not 0 <= row < len(self._records):
            raise IndexError(f"Row index {row} out of range.")
        if inpoint_seconds < 0.0:
            raise ValueError(f"inpoint_seconds must be non-negative, got {inpoint_seconds}.")
        old_record = self._records[row]
        new_record = dataclasses.replace(old_record, file=file, inpoint_seconds=inpoint_seconds)
        if new_record != old_record:
            self._push_replacement(row, old_record, new_record, "Set Inpoint", reorder=True)

    def add_record(self, record: ballroom_scribe.models.EventRecord) -> int:
        """Inserts `record` where it keeps its video's rows together and in `Inpoint` order; returns the row."""
        row = ballroom_scribe.gui.utils.find_insertion_row(self._records, record, self.video_rank)
        self.insert_record(row, record)
        return row

    def insert_record(self, row: int, record: ballroom_scribe.models.EventRecord) -> None:
        """Inserts `record` at `row` as an undoable command."""
        if not 0 <= row <= len(self._records):
            raise IndexError(f"Insertion row index {row} out of range.")
        self._undo_stack.push(_InsertRecordCommand(self, row, record))

    def remove_records(self, rows: Iterable[int]) -> None:
        """Removes the records at `rows` as one undoable command (`Delete Event` or `Delete N Events`).

        Args:
            rows: Row indices to remove, in any order; duplicates are ignored and no rows is a no-op.

        Raises:
            IndexError: If any row is out of range; nothing is removed then.
        """
        unique_rows = sorted(set(rows), reverse=True)
        for row in unique_rows:
            if not 0 <= row < len(self._records):
                raise IndexError(f"Row index {row} out of range.")
        if not unique_rows:
            return
        if len(unique_rows) == 1:
            self._undo_stack.push(_RemoveRecordCommand(self, unique_rows[0], self._records[unique_rows[0]]))
            return
        # Bottom-up keeps the remaining indices valid; undo replays the children in reverse, i.e. top-down.
        self._undo_stack.beginMacro(f"Delete {len(unique_rows)} Events")
        try:
            for row in unique_rows:
                self._undo_stack.push(_RemoveRecordCommand(self, row, self._records[row]))
        finally:
            self._undo_stack.endMacro()

    def remove_record(self, row: int) -> None:
        """Removes the record at `row` as an undoable command."""
        self.remove_records((row,))

    def unique_values(
        self,
        column: Column | int,
        *,
        exclude_row: int | None = None,
    ) -> tuple[str, ...]:
        """Returns sorted, case-insensitively deduplicated non-empty values in `column`."""
        col = self.Column(column)
        by_folded: dict[str, str] = {}
        for row_index, record in enumerate(self._records):
            if exclude_row is not None and row_index == exclude_row:
                continue
            value = _column_text_value(record, col).strip()
            if value:
                by_folded.setdefault(value.casefold(), value)
        return tuple(by_folded[key] for key in sorted(by_folded))

    def rows_for_file(self, file: pathlib.Path) -> tuple[int, ...]:
        """Returns the indices of all rows whose `file` equals `file`."""
        return tuple(index for index, record in enumerate(self._records) if record.file == file)

    def row_at_or_before(self, file: pathlib.Path, seconds: float) -> int | None:
        """Returns the row of `file`'s latest event with `inpoint_seconds <= seconds`, or `None` if none precedes it.

        Rows belong to `file`'s video when `video_rank` gives them the same non-`None` rank as `file`. Ties in
        `inpoint_seconds` go to the later row in table order.
        """
        rank = self.video_rank
        target_rank = rank(file)
        if target_rank is None:
            return None
        by_file: dict[pathlib.Path, int | None] = {file: target_rank}
        best: tuple[float, int] | None = None
        for row, record in enumerate(self._records):
            if record.file not in by_file:
                by_file[record.file] = rank(record.file)
            if by_file[record.file] == target_rank and record.inpoint_seconds <= seconds:
                candidate = (record.inpoint_seconds, row)
                if best is None or candidate > best:
                    best = candidate
        return best[1] if best is not None else None

    def row_entered(self, file: pathlib.Path, previous_seconds: float, seconds: float) -> int | None:
        """Returns the row of `file`'s latest event whose `inpoint_seconds` lies in `(previous_seconds, seconds]`.

        This is the event the playhead entered while advancing from `previous_seconds` to `seconds`. Returns `None`
        when no `Inpoint` of `file`'s video was crossed, including whenever `seconds <= previous_seconds`.
        """
        row = self.row_at_or_before(file, seconds)
        if row is None or self._records[row].inpoint_seconds <= previous_seconds:
            return None
        return row

    def has_same_event(self, row_a: int, row_b: int) -> bool:
        """Returns whether `row_a` and `row_b` share the same non-empty `Level`, `Style`, and `Dance`."""
        if not (0 <= row_a < len(self._records) and 0 <= row_b < len(self._records)):
            return False
        key_a = _event_key(self._records[row_a])
        return key_a is not None and key_a == _event_key(self._records[row_b])

    def find_matches(
        self,
        query: str,
        *,
        columns: Sequence[Column | int] = SEARCHABLE_COLUMNS,
        rows: Iterable[int] | None = None,
        case_sensitive: bool = False,
        whole_cell: bool = False,
    ) -> tuple[tuple[int, Column], ...]:
        """Returns matching `(row, column)` cells in reading order across `rows` and `columns`.

        Args:
            query: Literal text to search for; an empty string matches nothing.
            columns: Free-text columns to search (defaults to `SEARCHABLE_COLUMNS`).
            rows: Row indices to search, in any order (`None` searches every row).
            case_sensitive: Match exact letter case instead of ignoring case.
            whole_cell: Require the entire cell value to match `query` rather than any substring.

        Raises:
            ValueError: If `columns` includes a column outside `SEARCHABLE_COLUMNS`.
            IndexError: If `rows` includes an out-of-range row index.
        """
        target_columns = self._validated_text_columns(columns)
        target_rows = self._validated_rows(rows)
        if not query or not target_columns or not target_rows:
            return ()
        pattern = _compile_search_pattern(query, case_sensitive=case_sensitive)
        matches: list[tuple[int, EventTableModel.Column]] = []
        for row in target_rows:
            record = self._records[row]
            for column in target_columns:
                if _cell_matches(_column_text_value(record, column), pattern, whole_cell=whole_cell):
                    matches.append((row, column))
        return tuple(matches)

    def replace_in_cell(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        query: str,
        replacement: str,
        *,
        case_sensitive: bool = False,
        whole_cell: bool = False,
    ) -> bool:
        """Replaces `query` with `replacement` in `index` as an undoable command if `index` matches.

        Returns:
            Whether the cell's value changed. Nothing is pushed onto `undo_stack` when `False`.

        Raises:
            ValueError: If `index` is valid and its column is outside `SEARCHABLE_COLUMNS`.
        """
        if not index.isValid() or not 0 <= index.row() < len(self._records):
            return False
        (column,) = self._validated_text_columns((index.column(),))
        if not query:
            return False
        old_record = self._records[index.row()]
        current_text = _column_text_value(old_record, column)
        pattern = _compile_search_pattern(query, case_sensitive=case_sensitive)
        if not _cell_matches(current_text, pattern, whole_cell=whole_cell):
            return False
        new_text = _replace_cell_text(current_text, pattern, replacement, whole_cell=whole_cell)
        new_record = _with_text_value(old_record, column, new_text)
        if new_record == old_record:
            return False
        header_name = ballroom_scribe.EventRecorder.COLUMNS[int(column)]
        self._undo_stack.push(
            _ReplaceRecordCommand(self, index.row(), old_record, new_record, f"Replace {header_name}")
        )
        return True

    def replace_all(
        self,
        query: str,
        replacement: str,
        *,
        columns: Sequence[Column | int] = SEARCHABLE_COLUMNS,
        rows: Iterable[int] | None = None,
        case_sensitive: bool = False,
        whole_cell: bool = False,
    ) -> int:
        """Replaces `query` with `replacement` across `rows` and `columns` as one `Replace All` undo step.

        Returns:
            The number of cells whose value changed. Nothing is pushed onto `undo_stack` when zero.

        Raises:
            ValueError: If `columns` includes a column outside `SEARCHABLE_COLUMNS`.
            IndexError: If `rows` includes an out-of-range row index.
        """
        target_columns = self._validated_text_columns(columns)
        target_rows = self._validated_rows(rows)
        if not query or not target_columns or not target_rows:
            return 0
        pattern = _compile_search_pattern(query, case_sensitive=case_sensitive)
        row_replacements: list[tuple[int, ballroom_scribe.models.EventRecord, ballroom_scribe.models.EventRecord]] = []
        changed_cells = 0
        for row in target_rows:
            old_record = self._records[row]
            new_record = old_record
            for column in target_columns:
                current_text = _column_text_value(new_record, column)
                if not _cell_matches(current_text, pattern, whole_cell=whole_cell):
                    continue
                new_text = _replace_cell_text(current_text, pattern, replacement, whole_cell=whole_cell)
                if new_text != current_text:
                    new_record = _with_text_value(new_record, column, new_text)
                    changed_cells += 1
            if new_record != old_record:
                row_replacements.append((row, old_record, new_record))
        if not row_replacements:
            return 0

        self._undo_stack.beginMacro("Replace All")
        try:
            for row, old_record, new_record in row_replacements:
                self._undo_stack.push(_ReplaceRecordCommand(self, row, old_record, new_record, "Replace All"))
        finally:
            self._undo_stack.endMacro()
        return changed_cells

    @classmethod
    def _validated_text_columns(cls, columns: Sequence[Column | int]) -> tuple[Column, ...]:
        converted: list[EventTableModel.Column] = []
        for raw_col in columns:
            col = cls.Column(raw_col)
            if col not in cls.SEARCHABLE_COLUMNS:
                raise ValueError(f"Column {col.name} does not support text find/replace.")
            converted.append(col)
        return tuple(sorted(set(converted), key=int))

    def _validated_rows(self, rows: Iterable[int] | None) -> tuple[int, ...]:
        if rows is None:
            return tuple(range(len(self._records)))
        unique_rows = sorted(set(rows))
        for row in unique_rows:
            if not 0 <= row < len(self._records):
                raise IndexError(f"Row index {row} out of range.")
        return tuple(unique_rows)

    def _rank_by_first_appearance(self, file: pathlib.Path) -> int | None:
        """Default `video_rank`: ranks each distinct `File` path by where it first appears in the table."""
        order: dict[pathlib.Path, int] = {}
        for record in self._records:
            order.setdefault(record.file, len(order))
        return order.get(file)

    def _ordered_row(self, row: int, record: ballroom_scribe.models.EventRecord) -> int:
        """Returns `row` if `record` can stay there, else where it belongs among its video's rows."""
        return ballroom_scribe.gui.utils.find_insertion_row(self._records, record, self.video_rank, replacing_row=row)

    def _push_replacement(
        self,
        row: int,
        old_record: ballroom_scribe.models.EventRecord,
        new_record: ballroom_scribe.models.EventRecord,
        description: str,
        *,
        reorder: bool,
    ) -> None:
        """Pushes an undoable replacement of `row`; with `reorder`, the row also moves to where it now belongs.

        The replacement and the move form one undo step.
        """
        replacement = _ReplaceRecordCommand(self, row, old_record, new_record, description)
        target_row = self._ordered_row(row, new_record) if reorder else row
        if target_row == row:
            self._undo_stack.push(replacement)
            return
        self._undo_stack.beginMacro(description)
        try:
            self._undo_stack.push(replacement)
            self._undo_stack.push(_MoveRecordCommand(self, row, target_row))
        finally:
            self._undo_stack.endMacro()

    def _apply_replace(self, row: int, record: ballroom_scribe.models.EventRecord) -> None:
        self._records[row] = record
        left = self.index(row, 0)
        right = self.index(row, self.columnCount() - 1)
        self.dataChanged.emit(left, right)

    def _apply_move(self, from_row: int, to_row: int) -> None:
        """Moves the row at `from_row` so that it ends up at index `to_row`."""
        if from_row == to_row:
            return
        # Qt names the destination in pre-move numbering: the row the moved one will be placed in front of.
        destination = to_row + 1 if to_row > from_row else to_row
        if not self.beginMoveRows(QtCore.QModelIndex(), from_row, from_row, QtCore.QModelIndex(), destination):
            return
        self._records.insert(to_row, self._records.pop(from_row))
        self.endMoveRows()

    def _apply_insert(self, row: int, record: ballroom_scribe.models.EventRecord) -> None:
        self.beginInsertRows(QtCore.QModelIndex(), row, row)
        self._records.insert(row, record)
        self.endInsertRows()

    def _apply_remove(self, row: int) -> None:
        self._removing_rows = True
        try:
            self.beginRemoveRows(QtCore.QModelIndex(), row, row)
            del self._records[row]
            self.endRemoveRows()
        finally:
            self._removing_rows = False
