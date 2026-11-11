import dataclasses
import pathlib
import unittest
from unittest.mock import Mock

from parameterized import param, parameterized

try:
    from PySide6 import QtCore
except ImportError as err:
    raise unittest.SkipTest("PySide6 is not installed.") from err

import ballroom_scribe
import ballroom_scribe.gui
import ballroom_scribe.models

_VIDEO_A = pathlib.Path("/videos/part1.mp4")
_VIDEO_B = pathlib.Path("/videos/part2.mp4")


def _make_sample_records() -> list[ballroom_scribe.models.EventRecord]:
    return [
        ballroom_scribe.models.EventRecord(
            file=_VIDEO_A,
            inpoint_seconds=12.3456,
            level="Bronze",
            style="Standard",
            dance="Waltz",
            round="Final",
        ),
        ballroom_scribe.models.EventRecord(
            file=_VIDEO_B,
            inpoint_seconds=65.0,
            level="bronze",
            style="Latin",
            dance="Cha Cha",
            round="Semifinal",
        ),
        ballroom_scribe.models.EventRecord(
            file=_VIDEO_A,
            inpoint_seconds=120.5,
            level="Silver",
            style="",
            dance="Foxtrot",
            round="Final",
        ),
    ]


class TestEventTableModel(unittest.TestCase):
    """Unit tests for `EventTableModel`."""

    def setUp(self):
        self.model = ballroom_scribe.gui.EventTableModel()
        self.model.set_records(_make_sample_records())

    def test_shape_headers_and_flags(self):
        self.assertEqual(self.model.rowCount(), 3)
        self.assertEqual(self.model.columnCount(), len(ballroom_scribe.EventRecorder.COLUMNS))
        for col_index, expected_header in enumerate(ballroom_scribe.EventRecorder.COLUMNS):
            self.assertEqual(
                self.model.headerData(col_index, QtCore.Qt.Orientation.Horizontal),
                expected_header,
            )

        col = ballroom_scribe.gui.EventTableModel.Column
        self.assertFalse(bool(self.model.flags(self.model.index(0, col.FILE)) & QtCore.Qt.ItemFlag.ItemIsEditable))
        self.assertFalse(
            bool(self.model.flags(self.model.index(0, col.INPOINT_FORMATTED)) & QtCore.Qt.ItemFlag.ItemIsEditable)
        )
        for editable_col in (col.INPOINT, col.LEVEL, col.STYLE, col.DANCE, col.ROUND, col.COMMENT):
            self.assertTrue(
                bool(self.model.flags(self.model.index(0, editable_col)) & QtCore.Qt.ItemFlag.ItemIsEditable)
            )

    def test_display_and_edit_roles_preserve_full_float_precision(self):
        col = ballroom_scribe.gui.EventTableModel.Column
        inpoint_idx = self.model.index(0, col.INPOINT)
        formatted_idx = self.model.index(0, col.INPOINT_FORMATTED)

        self.assertEqual(self.model.data(inpoint_idx, QtCore.Qt.ItemDataRole.DisplayRole), "12.346")
        self.assertEqual(self.model.data(inpoint_idx, QtCore.Qt.ItemDataRole.EditRole), "12.3456")
        self.assertEqual(self.model.data(formatted_idx, QtCore.Qt.ItemDataRole.DisplayRole), "0:00:12.35")
        right_vcenter = int(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.assertEqual(self.model.data(inpoint_idx, QtCore.Qt.ItemDataRole.TextAlignmentRole), right_vcenter)
        self.assertEqual(self.model.data(formatted_idx, QtCore.Qt.ItemDataRole.TextAlignmentRole), right_vcenter)
        self.assertIsNone(self.model.data(self.model.index(0, col.LEVEL), QtCore.Qt.ItemDataRole.TextAlignmentRole))

    @parameterized.expand(
        [
            param("plain_seconds", typed="45.25", expected_seconds=45.25),
            # Later than the video's other row (120.5 s), so the row moves below it on commit.
            param("formatted_duration", typed="1:02:03.50", expected_seconds=3723.5, expected_row=2),
        ]
    )
    def test_set_data_parses_typed_inpoint_and_supports_undo_redo(
        self,
        _,
        typed: str,
        expected_seconds: float,
        expected_row: int = 0,
    ):
        col = ballroom_scribe.gui.EventTableModel.Column
        idx = self.model.index(0, col.INPOINT)
        self.assertTrue(self.model.undo_stack.isClean())

        ok = self.model.setData(idx, typed, QtCore.Qt.ItemDataRole.EditRole)

        self.assertTrue(ok)
        self.assertFalse(self.model.undo_stack.isClean())
        self.assertAlmostEqual(self.model.record_at(expected_row).inpoint_seconds, expected_seconds)
        self.assertEqual(self.model.record_at(expected_row).file, _VIDEO_A)

        self.model.undo_stack.undo()
        self.assertAlmostEqual(self.model.record_at(0).inpoint_seconds, 12.3456)
        self.assertTrue(self.model.undo_stack.isClean())

        self.model.undo_stack.redo()
        self.assertAlmostEqual(self.model.record_at(expected_row).inpoint_seconds, expected_seconds)

    def test_set_data_rejects_invalid_inpoint_and_read_only_columns(self):
        col = ballroom_scribe.gui.EventTableModel.Column
        self.assertFalse(self.model.setData(self.model.index(0, col.INPOINT), "not-a-time"))
        self.assertFalse(self.model.setData(self.model.index(0, col.FILE), "other.mp4"))
        self.assertTrue(self.model.undo_stack.isClean())

    def test_edit_text_set_inpoint_insert_and_remove_undo_redo(self):
        col = ballroom_scribe.gui.EventTableModel.Column
        self.assertTrue(self.model.setData(self.model.index(1, col.DANCE), "Rumba"))
        self.assertEqual(self.model.record_at(1).dance, "Rumba")

        self.model.set_inpoint(row=1, file=_VIDEO_A, inpoint_seconds=99.125)
        self.assertEqual(self.model.record_at(1).file, _VIDEO_A)
        self.assertAlmostEqual(self.model.record_at(1).inpoint_seconds, 99.125)

        inserted = ballroom_scribe.models.EventRecord(
            file=_VIDEO_B,
            inpoint_seconds=3.0,
            level="Gold",
            style="Smooth",
            dance="Tango",
            round="Final",
        )
        self.model.insert_record(1, inserted)
        self.assertEqual(self.model.rowCount(), 4)
        self.assertEqual(self.model.record_at(1), inserted)

        self.model.remove_record(0)
        self.assertEqual(self.model.rowCount(), 3)
        self.assertEqual(self.model.record_at(0), inserted)

        self.model.undo_stack.undo()
        self.assertEqual(self.model.rowCount(), 4)
        self.model.undo_stack.undo()
        self.assertEqual(self.model.rowCount(), 3)

    def test_is_removing_rows_only_while_a_removal_is_in_progress(self):
        observed: list[bool] = []
        self.model.rowsAboutToBeRemoved.connect(lambda *_: observed.append(self.model.is_removing_rows))
        self.model.rowsRemoved.connect(lambda *_: observed.append(self.model.is_removing_rows))

        self.assertFalse(self.model.is_removing_rows)
        self.model.remove_record(0)
        self.assertEqual(observed, [True, True])
        self.assertFalse(self.model.is_removing_rows)

        self.model.undo_stack.undo()  # Re-inserting is not a removal.
        self.model.undo_stack.redo()
        self.assertEqual(observed, [True, True, True, True])
        self.assertFalse(self.model.is_removing_rows)

    def test_unique_values_and_rows_for_file(self):
        col = ballroom_scribe.gui.EventTableModel.Column
        self.assertEqual(self.model.unique_values(col.LEVEL), ("Bronze", "Silver"))
        self.assertEqual(self.model.unique_values(col.LEVEL, exclude_row=2), ("Bronze",))
        self.assertEqual(self.model.unique_values(col.STYLE), ("Latin", "Standard"))
        self.assertEqual(self.model.rows_for_file(_VIDEO_A), (0, 2))
        self.assertEqual(self.model.rows_for_file(_VIDEO_B), (1,))

    @parameterized.expand(
        [
            param("level", column=ballroom_scribe.gui.EventTableModel.Column.LEVEL, field="level"),
            param("style", column=ballroom_scribe.gui.EventTableModel.Column.STYLE, field="style"),
            param("dance", column=ballroom_scribe.gui.EventTableModel.Column.DANCE, field="dance"),
            param("round", column=ballroom_scribe.gui.EventTableModel.Column.ROUND, field="round"),
            param("comment", column=ballroom_scribe.gui.EventTableModel.Column.COMMENT, field="comment"),
        ]
    )
    def test_set_data_edits_each_text_column(self, _, column, field: str):
        self.assertTrue(self.model.setData(self.model.index(2, column), "  New Value "))

        self.assertEqual(getattr(self.model.record_at(2), field), "New Value")
        self.assertEqual(self.model.undo_stack.undoText(), f"Edit {ballroom_scribe.EventRecorder.COLUMNS[column]}")

    def test_set_data_keeps_comment_line_breaks_but_normalizes_their_endings(self):
        col = ballroom_scribe.gui.EventTableModel.Column
        index = self.model.index(0, col.COMMENT)

        self.assertTrue(self.model.setData(index, " Check music\r\nbefore heat 2\r "))

        self.assertEqual(self.model.record_at(0).comment, "Check music\nbefore heat 2")
        self.assertEqual(self.model.data(index, QtCore.Qt.ItemDataRole.EditRole), "Check music\nbefore heat 2")
        self.assertEqual(self.model.data(index), "Check music …")
        self.assertEqual(self.model.data(index, QtCore.Qt.ItemDataRole.ToolTipRole), "Check music\nbefore heat 2")


class TestFill(unittest.TestCase):
    """Unit tests for `EventTableModel.fill` and `EventTableModel.is_fillable`."""

    def setUp(self):
        self.model = ballroom_scribe.gui.EventTableModel()
        self.model.set_records(_make_sample_records())
        self.col = ballroom_scribe.gui.EventTableModel.Column

    @parameterized.expand(
        [
            param("file", column=ballroom_scribe.gui.EventTableModel.Column.FILE, expected=False),
            param("inpoint", column=ballroom_scribe.gui.EventTableModel.Column.INPOINT, expected=False),
            param(
                "inpoint_formatted", column=ballroom_scribe.gui.EventTableModel.Column.INPOINT_FORMATTED, expected=False
            ),
            param("level", column=ballroom_scribe.gui.EventTableModel.Column.LEVEL, expected=True),
            param("style_as_int", column=4, expected=True),
            param("dance", column=ballroom_scribe.gui.EventTableModel.Column.DANCE, expected=True),
            param("round", column=ballroom_scribe.gui.EventTableModel.Column.ROUND, expected=True),
            param("comment", column=ballroom_scribe.gui.EventTableModel.Column.COMMENT, expected=False),
        ]
    )
    def test_is_fillable(self, _, column, expected: bool):
        self.assertEqual(ballroom_scribe.gui.EventTableModel.is_fillable(column), expected)

    def test_fill_down_copies_the_value_as_a_single_undo_step(self):
        changed = self.model.fill(self.model.index(0, self.col.LEVEL), 2)

        self.assertEqual(changed, 2)
        self.assertEqual([record.level for record in self.model.records], ["Bronze", "Bronze", "Bronze"])
        self.assertEqual(self.model.undo_stack.count(), 1)
        self.assertEqual(self.model.undo_stack.undoText(), "Fill Level")

        self.model.undo_stack.undo()
        self.assertEqual([record.level for record in self.model.records], ["Bronze", "bronze", "Silver"])
        self.assertTrue(self.model.undo_stack.isClean())

        self.model.undo_stack.redo()
        self.assertEqual([record.level for record in self.model.records], ["Bronze", "Bronze", "Bronze"])

    def test_fill_up_skips_rows_that_already_hold_the_value(self):
        changed = self.model.fill(self.model.index(2, self.col.ROUND), 0)

        self.assertEqual(changed, 1)
        self.assertEqual([record.round for record in self.model.records], ["Final", "Final", "Final"])
        self.assertEqual(self.model.undo_stack.count(), 1)

        self.model.undo_stack.undo()
        self.assertEqual([record.round for record in self.model.records], ["Final", "Semifinal", "Final"])

    def test_fill_emits_data_changed_for_each_changed_row(self):
        changed_rows: list[int] = []
        self.model.dataChanged.connect(lambda top_left, *_: changed_rows.append(top_left.row()))

        self.model.fill(self.model.index(0, self.col.DANCE), 2)

        self.assertEqual(changed_rows, [1, 2])

    def test_fill_to_its_own_row_changes_nothing(self):
        changed = self.model.fill(self.model.index(1, self.col.STYLE), 1)

        self.assertEqual(changed, 0)
        self.assertEqual(self.model.undo_stack.count(), 0)
        self.assertTrue(self.model.undo_stack.isClean())

    def test_fill_leaves_no_undo_entry_when_every_row_already_matches(self):
        self.assertTrue(self.model.setData(self.model.index(1, self.col.LEVEL), "Bronze"))
        self.model.undo_stack.setClean()

        changed = self.model.fill(self.model.index(0, self.col.LEVEL), 1)

        self.assertEqual(changed, 0)
        self.assertTrue(self.model.undo_stack.isClean())

    def test_fill_from_an_empty_cell_clears_the_targets(self):
        changed = self.model.fill(self.model.index(2, self.col.STYLE), 0)

        self.assertEqual(changed, 2)
        self.assertEqual([record.style for record in self.model.records], ["", "", ""])
        self.assertEqual(self.model.undo_stack.undoText(), "Fill Style")

        self.model.undo_stack.undo()
        self.assertEqual([record.style for record in self.model.records], ["Standard", "Latin", ""])

    @parameterized.expand(
        [
            param("file", column=ballroom_scribe.gui.EventTableModel.Column.FILE),
            param("inpoint", column=ballroom_scribe.gui.EventTableModel.Column.INPOINT),
            param("inpoint_formatted", column=ballroom_scribe.gui.EventTableModel.Column.INPOINT_FORMATTED),
            param("comment", column=ballroom_scribe.gui.EventTableModel.Column.COMMENT),
        ]
    )
    def test_fill_rejects_columns_without_a_handle(self, _, column):
        with self.assertRaises(ValueError):
            self.model.fill(self.model.index(0, column), 2)
        self.assertTrue(self.model.undo_stack.isClean())

    def test_fill_rejects_invalid_source_and_rows_out_of_range(self):
        with self.assertRaises(ValueError):
            self.model.fill(QtCore.QModelIndex(), 1)
        with self.assertRaises(IndexError):
            self.model.fill(self.model.index(0, self.col.LEVEL), 3)
        with self.assertRaises(IndexError):
            self.model.fill(self.model.index(0, self.col.LEVEL), -1)
        self.assertTrue(self.model.undo_stack.isClean())


class TestVideoOrder(unittest.TestCase):
    """Unit tests for how `EventTableModel` keeps each video's rows together and in `Inpoint` order.

    The sample rows are `A 12.3456`, `B 65.0`, `A 120.5`; the default `video_rank` ranks `A` before `B`.
    """

    def setUp(self):
        self.model = ballroom_scribe.gui.EventTableModel()
        self.model.set_records(_make_sample_records())
        self.col = ballroom_scribe.gui.EventTableModel.Column
        self.moves: list[tuple[int, int]] = []
        self.model.rowsMoved.connect(lambda _parent, first, _last, _destination, to: self.moves.append((first, to)))

    def _inpoints(self) -> list[tuple[pathlib.Path, float]]:
        return [(record.file, record.inpoint_seconds) for record in self.model.records]

    def test_add_record_places_the_row_among_its_video_in_inpoint_order(self):
        new_record = dataclasses.replace(_make_sample_records()[0], inpoint_seconds=50.0)

        row = self.model.add_record(new_record)

        self.assertEqual(row, 2)
        self.assertEqual(self._inpoints(), [(_VIDEO_A, 12.3456), (_VIDEO_B, 65.0), (_VIDEO_A, 50.0), (_VIDEO_A, 120.5)])
        self.assertEqual(self.model.undo_stack.undoText(), "Add Event")

    def test_add_record_for_a_new_video_appends(self):
        new_record = dataclasses.replace(_make_sample_records()[0], file=pathlib.Path("/videos/part3.mp4"))

        self.assertEqual(self.model.add_record(new_record), 3)

    def test_set_inpoint_moves_the_row_as_one_undo_step(self):
        self.model.set_inpoint(row=0, file=_VIDEO_A, inpoint_seconds=200.0)

        self.assertEqual(self._inpoints(), [(_VIDEO_B, 65.0), (_VIDEO_A, 120.5), (_VIDEO_A, 200.0)])
        self.assertEqual(self.moves, [(0, 3)])  # Qt reports the destination in pre-move numbering.
        self.assertEqual(self.model.undo_stack.count(), 1)
        self.assertEqual(self.model.undo_stack.undoText(), "Set Inpoint")

        self.model.undo_stack.undo()
        self.assertEqual(self._inpoints(), [(_VIDEO_A, 12.3456), (_VIDEO_B, 65.0), (_VIDEO_A, 120.5)])
        self.assertTrue(self.model.undo_stack.isClean())

        self.model.undo_stack.redo()
        self.assertEqual(self._inpoints(), [(_VIDEO_B, 65.0), (_VIDEO_A, 120.5), (_VIDEO_A, 200.0)])

    def test_set_inpoint_that_keeps_the_order_does_not_move_the_row(self):
        self.model.set_inpoint(row=0, file=_VIDEO_A, inpoint_seconds=50.0)

        self.assertEqual(self._inpoints(), [(_VIDEO_A, 50.0), (_VIDEO_B, 65.0), (_VIDEO_A, 120.5)])
        self.assertEqual(self.moves, [])

    def test_set_inpoint_with_another_video_moves_the_row_into_that_block(self):
        self.model.set_inpoint(row=2, file=_VIDEO_B, inpoint_seconds=60.0)

        self.assertEqual(self._inpoints(), [(_VIDEO_A, 12.3456), (_VIDEO_B, 60.0), (_VIDEO_B, 65.0)])
        self.assertEqual(self.moves, [(2, 1)])

    def test_committed_inpoint_edit_moves_the_row_and_persistent_indexes_follow(self):
        edited = QtCore.QPersistentModelIndex(self.model.index(2, self.col.LEVEL))

        self.assertTrue(self.model.setData(self.model.index(2, self.col.INPOINT), "0:00:01.00"))

        self.assertEqual(self._inpoints(), [(_VIDEO_A, 1.0), (_VIDEO_A, 12.3456), (_VIDEO_B, 65.0)])
        self.assertEqual(edited.row(), 0)
        self.assertEqual(self.model.undo_stack.undoText(), "Edit Inpoint")

        self.model.undo_stack.undo()
        self.assertEqual(self._inpoints(), [(_VIDEO_A, 12.3456), (_VIDEO_B, 65.0), (_VIDEO_A, 120.5)])
        self.assertEqual(edited.row(), 2)

    def test_text_edits_never_move_rows(self):
        self.assertTrue(self.model.setData(self.model.index(2, self.col.LEVEL), "Gold"))
        self.model.fill(self.model.index(2, self.col.LEVEL), 0)

        self.assertEqual(self.moves, [])

    def test_custom_video_rank_orders_the_videos(self):
        part3 = pathlib.Path("/videos/part3.mp4")
        ranks = {part3: 0, _VIDEO_A: 1, _VIDEO_B: 2}
        model = ballroom_scribe.gui.EventTableModel(video_rank=ranks.get)
        model.set_records(_make_sample_records())

        row = model.add_record(dataclasses.replace(_make_sample_records()[0], file=part3))

        self.assertEqual(row, 0)
        self.assertEqual(model.video_rank(_VIDEO_B), 2)


class TestHasSameEvent(unittest.TestCase):
    """Unit tests for `EventTableModel.has_same_event`."""

    def setUp(self):
        self.model = ballroom_scribe.gui.EventTableModel()
        self.model.set_records(
            [
                ballroom_scribe.models.EventRecord(
                    file=_VIDEO_A,
                    inpoint_seconds=10.0,
                    level="Bronze",
                    style="Standard",
                    dance="Waltz",
                    round="Quarterfinal",
                ),
                ballroom_scribe.models.EventRecord(
                    file=_VIDEO_B,
                    inpoint_seconds=80.0,
                    level="bronze",
                    style="STANDARD",
                    dance="waltz",
                    round="Final",
                ),
                ballroom_scribe.models.EventRecord(
                    file=_VIDEO_A,
                    inpoint_seconds=45.0,
                    level="Bronze",
                    style="Standard",
                    dance="Tango",
                    round="Quarterfinal",
                ),
            ]
        )

    def test_matches_same_level_style_and_dance_across_rounds_and_videos_case_insensitively(self):
        self.assertTrue(self.model.has_same_event(0, 1))
        self.assertTrue(self.model.has_same_event(1, 0))
        self.assertTrue(self.model.has_same_event(0, 0))

    def test_rejects_rows_with_different_level_style_or_dance(self):
        self.assertFalse(self.model.has_same_event(0, 2))

    @parameterized.expand(
        [
            param("empty_level", level="", style="Standard", dance="Waltz"),
            param("empty_style", level="Bronze", style="", dance="Waltz"),
            param("empty_dance", level="Bronze", style="Standard", dance=""),
            param("whitespace_only_dance", level="Bronze", style="Standard", dance="   "),
            param("all_empty", level="", style="", dance=""),
        ]
    )
    def test_rejects_rows_when_any_field_is_empty(self, _, level, style, dance):
        incomplete = ballroom_scribe.models.EventRecord(
            file=_VIDEO_A,
            inpoint_seconds=5.0,
            level=level,
            style=style,
            dance=dance,
            round="Final",
        )
        self.model.set_records([incomplete, incomplete])

        self.assertFalse(self.model.has_same_event(0, 1))

    @parameterized.expand(
        [
            param("negative_first", row_a=-1, row_b=0),
            param("negative_second", row_a=0, row_b=-1),
            param("past_end_first", row_a=3, row_b=0),
            param("past_end_second", row_a=0, row_b=3),
        ]
    )
    def test_rejects_out_of_range_rows(self, _, row_a, row_b):
        self.assertFalse(self.model.has_same_event(row_a, row_b))


class TestClearCell(unittest.TestCase):
    """Unit tests for `EventTableModel.clear_cell`."""

    def setUp(self):
        self.model = ballroom_scribe.gui.EventTableModel()
        self.model.set_records(_make_sample_records())
        self.col = ballroom_scribe.gui.EventTableModel.Column

    @parameterized.expand(
        [
            param(
                "level",
                column=ballroom_scribe.gui.EventTableModel.Column.LEVEL,
                original="Bronze",
                undo_text="Clear Level",
            ),
            param(
                "style",
                column=ballroom_scribe.gui.EventTableModel.Column.STYLE,
                original="Standard",
                undo_text="Clear Style",
            ),
            param(
                "dance",
                column=ballroom_scribe.gui.EventTableModel.Column.DANCE,
                original="Waltz",
                undo_text="Clear Dance",
            ),
            param(
                "round",
                column=ballroom_scribe.gui.EventTableModel.Column.ROUND,
                original="Final",
                undo_text="Clear Round",
            ),
        ]
    )
    def test_clears_text_cell_and_supports_undo_and_redo(self, _, column, original, undo_text):
        index = self.model.index(0, column)

        self.assertTrue(self.model.clear_cell(index))
        self.assertEqual(self.model.data(index), "")
        self.assertEqual(self.model.undo_stack.count(), 1)
        self.assertEqual(self.model.undo_stack.undoText(), undo_text)

        self.model.undo_stack.undo()
        self.assertEqual(self.model.data(index), original)
        self.assertTrue(self.model.undo_stack.isClean())

        self.model.undo_stack.redo()
        self.assertEqual(self.model.data(index), "")

    def test_already_empty_cell_leaves_undo_stack_clean(self):
        empty_style = self.model.index(2, self.col.STYLE)

        self.assertFalse(self.model.clear_cell(empty_style))
        self.assertEqual(self.model.undo_stack.count(), 0)
        self.assertTrue(self.model.undo_stack.isClean())

    def test_clears_comment_as_its_own_undo_step(self):
        index = self.model.index(1, self.col.COMMENT)
        self.assertFalse(self.model.clear_cell(index))  # Nothing to clear yet.
        self.assertTrue(self.model.setData(index, "Re-check the dance\nagainst the heat list"))

        self.assertTrue(self.model.clear_cell(index))
        self.assertEqual(self.model.record_at(1).comment, "")
        self.assertEqual(self.model.undo_stack.undoText(), "Clear Comment")

        self.model.undo_stack.undo()
        self.assertEqual(self.model.record_at(1).comment, "Re-check the dance\nagainst the heat list")

    @parameterized.expand(
        [
            param("file", column=ballroom_scribe.gui.EventTableModel.Column.FILE),
            param("inpoint", column=ballroom_scribe.gui.EventTableModel.Column.INPOINT),
            param("inpoint_formatted", column=ballroom_scribe.gui.EventTableModel.Column.INPOINT_FORMATTED),
        ]
    )
    def test_ignores_non_clearable_columns(self, _, column):
        index = self.model.index(0, column)
        before = self.model.data(index)

        self.assertFalse(self.model.clear_cell(index))
        self.assertEqual(self.model.data(index), before)
        self.assertEqual(self.model.undo_stack.count(), 0)
        self.assertTrue(self.model.undo_stack.isClean())

    def test_ignores_invalid_index(self):
        self.assertFalse(self.model.clear_cell(QtCore.QModelIndex()))
        self.assertEqual(self.model.undo_stack.count(), 0)
        self.assertTrue(self.model.undo_stack.isClean())


class TestRemoveRecords(unittest.TestCase):
    """Unit tests for `EventTableModel.remove_records`."""

    def setUp(self):
        self.model = ballroom_scribe.gui.EventTableModel()
        self.initial_records = [
            ballroom_scribe.models.EventRecord(
                file=_VIDEO_A,
                inpoint_seconds=float((index + 1) * 10),
                level="Bronze",
                style="Standard",
                dance=f"Dance {index}",
                round="Final",
            )
            for index in range(5)
        ]
        self.model.set_records(self.initial_records)

    def test_removes_unordered_rows_with_duplicates_as_one_undo_step(self):
        self.model.remove_records((3, 1, 3))

        self.assertEqual(
            self.model.records,
            [self.initial_records[0], self.initial_records[2], self.initial_records[4]],
        )
        self.assertEqual(self.model.undo_stack.count(), 1)
        self.assertEqual(self.model.undo_stack.undoText(), "Delete 2 Events")

        self.model.undo_stack.undo()
        self.assertEqual(self.model.records, self.initial_records)
        self.assertTrue(self.model.undo_stack.isClean())

        self.model.undo_stack.redo()
        self.assertEqual(
            self.model.records,
            [self.initial_records[0], self.initial_records[2], self.initial_records[4]],
        )

    def test_single_row_keeps_delete_event_undo_text(self):
        self.model.remove_records((2, 2))

        self.assertEqual(self.model.rowCount(), 4)
        self.assertEqual(self.model.undo_stack.count(), 1)
        self.assertEqual(self.model.undo_stack.undoText(), "Delete Event")

    def test_removes_every_row_and_undo_restores_all_in_order(self):
        self.model.remove_records(range(5))

        self.assertEqual(self.model.rowCount(), 0)
        self.assertEqual(self.model.undo_stack.count(), 1)
        self.assertEqual(self.model.undo_stack.undoText(), "Delete 5 Events")

        self.model.undo_stack.undo()
        self.assertEqual(self.model.records, self.initial_records)
        self.assertTrue(self.model.undo_stack.isClean())

    def test_empty_rows_is_a_no_op(self):
        self.model.remove_records(())

        self.assertEqual(self.model.records, self.initial_records)
        self.assertEqual(self.model.undo_stack.count(), 0)
        self.assertTrue(self.model.undo_stack.isClean())

    @parameterized.expand(
        [
            param("negative_row", rows=(-1, 2)),
            param("past_end_row", rows=(1, 5)),
        ]
    )
    def test_rejects_out_of_range_row_before_removing_any(self, _, rows: tuple[int, ...]):
        with self.assertRaises(IndexError):
            self.model.remove_records(rows)

        self.assertEqual(self.model.records, self.initial_records)
        self.assertEqual(self.model.undo_stack.count(), 0)
        self.assertTrue(self.model.undo_stack.isClean())

    def test_emits_rows_removed_from_bottom_to_top(self):
        removed_rows: list[int] = []
        self.model.rowsRemoved.connect(lambda _parent, first, _last: removed_rows.append(first))

        self.model.remove_records((1, 3))

        self.assertEqual(removed_rows, [3, 1])


class TestReorderRows(unittest.TestCase):
    """Unit tests for `EventTableModel.reorder_rows`."""

    def setUp(self):
        self.model = ballroom_scribe.gui.EventTableModel()
        self.initial_records = [
            TestRowAtOrBefore._event(_VIDEO_A, 30.0),
            TestRowAtOrBefore._event(_VIDEO_B, 5.0),
            TestRowAtOrBefore._event(_VIDEO_A, 10.0),
        ]
        self.model.set_records(self.initial_records)

    def _records_in_order(self, order: tuple[int, ...]) -> list[ballroom_scribe.models.EventRecord]:
        return [self.initial_records[row] for row in order]

    def test_applies_the_permutation_as_one_undo_step(self):
        moved = self.model.reorder_rows((2, 1, 0), "Sort Rows by Inpoint")

        self.assertTrue(moved)
        self.assertEqual(self.model.records, self._records_in_order((2, 1, 0)))
        self.assertEqual(self.model.undo_stack.count(), 1)
        self.assertEqual(self.model.undo_stack.undoText(), "Sort Rows by Inpoint")

        self.model.undo_stack.undo()
        self.assertEqual(self.model.records, self.initial_records)
        self.assertTrue(self.model.undo_stack.isClean())

        self.model.undo_stack.redo()
        self.assertEqual(self.model.records, self._records_in_order((2, 1, 0)))

    def test_undo_inverts_a_cyclic_permutation(self):
        # Applying `(1, 2, 0)` twice would not restore the original order, so undo must use the inverse.
        self.model.reorder_rows((1, 2, 0), "Group Rows by Video")
        self.assertEqual(self.model.records, self._records_in_order((1, 2, 0)))

        self.model.undo_stack.undo()

        self.assertEqual(self.model.records, self.initial_records)

    def test_identity_order_moves_nothing_and_pushes_nothing(self):
        self.assertFalse(self.model.reorder_rows((0, 1, 2), "Group Rows by Video"))

        self.assertEqual(self.model.records, self.initial_records)
        self.assertEqual(self.model.undo_stack.count(), 0)
        self.assertTrue(self.model.undo_stack.isClean())

    @parameterized.expand(
        [
            param("too_few_rows", order=(1, 0)),
            param("repeated_row", order=(0, 0, 1)),
            param("out_of_range_row", order=(0, 1, 3)),
        ]
    )
    def test_rejects_a_non_permutation_before_changing_anything(self, _, order: tuple[int, ...]):
        with self.assertRaises(ValueError):
            self.model.reorder_rows(order, "Group Rows by Video")

        self.assertEqual(self.model.records, self.initial_records)
        self.assertEqual(self.model.undo_stack.count(), 0)

    def test_moves_rows_as_one_layout_change_with_a_vertical_sort_hint(self):
        signature = "QList<QPersistentModelIndex>,QAbstractItemModel::LayoutChangeHint"
        layout_signals: list[tuple[str, QtCore.QAbstractItemModel.LayoutChangeHint]] = []
        self.model.layoutAboutToBeChanged[signature].connect(
            lambda _parents, hint: layout_signals.append(("about", hint))
        )
        self.model.layoutChanged[signature].connect(lambda _parents, hint: layout_signals.append(("changed", hint)))
        row_signals: list[str] = []
        self.model.rowsInserted.connect(lambda *_: row_signals.append("inserted"))
        self.model.rowsRemoved.connect(lambda *_: row_signals.append("removed"))
        self.model.rowsMoved.connect(lambda *_: row_signals.append("moved"))
        self.model.modelReset.connect(lambda: row_signals.append("reset"))

        self.model.reorder_rows((2, 1, 0), "Sort Rows by Inpoint")

        vertical = QtCore.QAbstractItemModel.LayoutChangeHint.VerticalSortHint
        self.assertEqual(layout_signals, [("about", vertical), ("changed", vertical)])
        self.assertEqual(row_signals, [])

    def test_persistent_indexes_follow_their_records(self):
        moved = QtCore.QPersistentModelIndex(self.model.index(0, 3))
        untouched = QtCore.QPersistentModelIndex(self.model.index(1, 2))

        self.model.reorder_rows((2, 1, 0), "Sort Rows by Inpoint")

        self.assertEqual((moved.row(), moved.column()), (2, 3))
        self.assertEqual((untouched.row(), untouched.column()), (1, 2))
        self.assertEqual(self.model.record_at(moved.row()), self.initial_records[0])

        self.model.undo_stack.undo()

        self.assertEqual(moved.row(), 0)
        self.assertEqual(untouched.row(), 1)


class TestRowAtOrBefore(unittest.TestCase):
    """Unit tests for `EventTableModel.row_at_or_before`."""

    @staticmethod
    def _event(file: pathlib.Path, inpoint_seconds: float) -> ballroom_scribe.models.EventRecord:
        return ballroom_scribe.models.EventRecord(
            file=file,
            inpoint_seconds=inpoint_seconds,
            level="Bronze",
            style="Standard",
            dance="Waltz",
            round="Final",
        )

    @parameterized.expand(
        [
            param("strictly_before_first_event", file=_VIDEO_A, seconds=9.999, expected=None),
            param("exact_first_inpoint", file=_VIDEO_A, seconds=10.0, expected=0),
            param("between_first_and_second_ignoring_other_video", file=_VIDEO_A, seconds=25.0, expected=0),
            param("exact_second_inpoint", file=_VIDEO_A, seconds=30.0, expected=1),
            param("after_last_event", file=_VIDEO_A, seconds=999.0, expected=2),
            param("other_video_before_its_first_event", file=_VIDEO_B, seconds=14.9, expected=None),
            param("other_video_after_its_event", file=_VIDEO_B, seconds=20.0, expected=3),
            param("unknown_video", file=pathlib.Path("/videos/missing.mp4"), seconds=999.0, expected=None),
        ]
    )
    def test_finds_latest_preceding_row_for_video(
        self,
        _,
        file: pathlib.Path,
        seconds: float,
        expected: int | None,
    ):
        model = ballroom_scribe.gui.EventTableModel()
        model.set_records(
            [
                self._event(_VIDEO_A, 10.0),
                self._event(_VIDEO_A, 30.0),
                self._event(_VIDEO_A, 60.0),
                self._event(_VIDEO_B, 15.0),
            ]
        )

        self.assertEqual(model.row_at_or_before(file, seconds), expected)

    def test_ties_pick_later_row_and_unsorted_rows_still_pick_latest_inpoint(self):
        model = ballroom_scribe.gui.EventTableModel()
        model.set_records(
            [
                self._event(_VIDEO_A, 50.0),
                self._event(_VIDEO_A, 20.0),
                self._event(_VIDEO_A, 20.0),
            ]
        )

        self.assertEqual(model.row_at_or_before(_VIDEO_A, 20.0), 2)
        self.assertEqual(model.row_at_or_before(_VIDEO_A, 35.0), 2)
        self.assertEqual(model.row_at_or_before(_VIDEO_A, 55.0), 0)

    def test_uses_custom_video_rank_once_per_distinct_file_and_handles_empty_table(self):
        relative_a = pathlib.Path("part1.mp4")
        unknown = pathlib.Path("unloaded.mp4")
        ranks = {_VIDEO_A: 0, relative_a: 0, _VIDEO_B: 1}
        rank_mock = Mock(side_effect=ranks.get)
        model = ballroom_scribe.gui.EventTableModel(video_rank=rank_mock)

        self.assertIsNone(model.row_at_or_before(_VIDEO_A, 10.0))

        rank_mock.reset_mock()
        model.set_records(
            [
                self._event(unknown, 5.0),
                self._event(relative_a, 12.0),
                self._event(relative_a, 24.0),
                self._event(_VIDEO_B, 18.0),
            ]
        )

        self.assertEqual(model.row_at_or_before(_VIDEO_A, 20.0), 1)
        self.assertIsNone(model.row_at_or_before(unknown, 99.0))
        # First call queried `_VIDEO_A`, `unknown`, `relative_a`, `_VIDEO_B` (skipping the duplicate `relative_a`);
        # second call queried `unknown` and stopped immediately.
        self.assertEqual(
            [call.args[0] for call in rank_mock.call_args_list],
            [_VIDEO_A, unknown, relative_a, _VIDEO_B, unknown],
        )


class TestRowEntered(unittest.TestCase):
    """Unit tests for `EventTableModel.row_entered`."""

    def setUp(self):
        self.model = ballroom_scribe.gui.EventTableModel()
        self.model.set_records(
            [
                TestRowAtOrBefore._event(_VIDEO_A, 10.0),
                TestRowAtOrBefore._event(_VIDEO_A, 30.0),
                TestRowAtOrBefore._event(_VIDEO_A, 30.0),
                TestRowAtOrBefore._event(_VIDEO_B, 15.0),
            ]
        )

    @parameterized.expand(
        [
            param("crosses_first_inpoint", previous=9.9, seconds=10.0, expected=0),
            param("crosses_tied_inpoints_picks_later_row", previous=29.95, seconds=30.01, expected=2),
            param("crosses_several_inpoints_picks_latest", previous=5.0, seconds=45.0, expected=2),
            param("ignores_other_video_inpoint", previous=14.0, seconds=16.0, expected=None),
            param("starts_exactly_on_inpoint", previous=10.0, seconds=10.5, expected=None),
            param("within_an_event", previous=12.0, seconds=12.02, expected=None),
            param("before_first_event", previous=0.0, seconds=9.99, expected=None),
            param("no_advance", previous=10.0, seconds=10.0, expected=None),
            param("backwards_across_inpoint", previous=12.0, seconds=9.0, expected=None),
        ]
    )
    def test_reports_row_whose_inpoint_was_crossed(self, _, previous: float, seconds: float, expected: int | None):
        self.assertEqual(self.model.row_entered(_VIDEO_A, previous, seconds), expected)

    def test_other_video_and_unknown_video(self):
        self.assertEqual(self.model.row_entered(_VIDEO_B, 14.0, 16.0), 3)
        self.assertIsNone(self.model.row_entered(pathlib.Path("/videos/missing.mp4"), 0.0, 999.0))


class TestRowsOfVideo(unittest.TestCase):
    """Unit tests for `EventTableModel.rows_of_video`, `first_row_of_video`, and `relink_rows`."""

    def setUp(self):
        relative_a = pathlib.Path("part1.mp4")
        ranks = {_VIDEO_A: 0, relative_a: 0, _VIDEO_B: 1}
        self.model = ballroom_scribe.gui.EventTableModel(video_rank=ranks.get)
        self.model.set_records(
            [
                TestRowAtOrBefore._event(_VIDEO_B, 5.0),
                TestRowAtOrBefore._event(_VIDEO_A, 30.0),
                TestRowAtOrBefore._event(relative_a, 10.0),
                TestRowAtOrBefore._event(pathlib.Path("unloaded.mp4"), 1.0),
                TestRowAtOrBefore._event(_VIDEO_A, 60.0),
            ]
        )

    def test_rows_of_video_groups_relative_and_absolute_spellings_in_table_order(self):
        self.assertEqual(self.model.rows_of_video(_VIDEO_A), (1, 2, 4))
        self.assertEqual(self.model.rows_of_video(pathlib.Path("part1.mp4")), (1, 2, 4))
        self.assertEqual(self.model.rows_of_video(_VIDEO_B), (0,))
        self.assertEqual(self.model.rows_of_video(pathlib.Path("unloaded.mp4")), ())
        self.assertEqual(self.model.first_row_of_video(_VIDEO_A), 1)
        self.assertEqual(self.model.first_row_of_video(_VIDEO_B), 0)
        self.assertIsNone(self.model.first_row_of_video(pathlib.Path("unloaded.mp4")))

    def test_relink_rows_rewrites_only_the_given_rows_as_one_undo_step(self):
        new_file = pathlib.Path("found/part1.mp4")
        before = self.model.records

        changed = self.model.relink_rows(self.model.rows_of_video(_VIDEO_A), new_file, "Locate Video")

        self.assertEqual(changed, 3)
        self.assertEqual(
            [record.file for record in self.model.records],
            [_VIDEO_B, new_file, new_file, pathlib.Path("unloaded.mp4"), new_file],
        )
        self.assertEqual([record.inpoint_seconds for record in self.model.records], [5.0, 30.0, 10.0, 1.0, 60.0])
        self.assertEqual(self.model.undo_stack.count(), 1)
        self.assertEqual(self.model.undo_stack.undoText(), "Locate Video")

        self.model.undo_stack.undo()
        self.assertEqual(self.model.records, before)
        self.model.undo_stack.redo()
        self.assertEqual(self.model.record_at(4).file, new_file)

    def test_relink_rows_skips_rows_already_pointing_at_the_file_and_validates_rows(self):
        self.assertEqual(self.model.relink_rows((1, 4), _VIDEO_A, "Locate Video"), 0)
        self.assertEqual(self.model.undo_stack.count(), 0)
        self.assertEqual(self.model.relink_rows((), pathlib.Path("x.mp4"), "Locate Video"), 0)

        with self.assertRaises(IndexError):
            self.model.relink_rows((1, 5), pathlib.Path("x.mp4"), "Locate Video")
        self.assertEqual(self.model.record_at(1).file, _VIDEO_A)
        self.assertTrue(self.model.undo_stack.isClean())


class TestCommentDisplay(unittest.TestCase):
    """Unit tests for `EventTableModel.set_comment_display` and the `Comment` column's roles."""

    def setUp(self):
        self.model = ballroom_scribe.gui.EventTableModel()
        self.model.set_records(_make_sample_records())
        self.col = ballroom_scribe.gui.EventTableModel.Column
        self.display = ballroom_scribe.gui.EventTableModel.CommentDisplay
        self.model.setData(self.model.index(0, self.col.COMMENT), "Check music\nbefore heat 2")
        self.model.undo_stack.clear()
        self.annotated = self.model.index(0, self.col.COMMENT)
        self.blank = self.model.index(1, self.col.COMMENT)

    def test_defaults_to_text_showing_the_first_line(self):
        decoration = QtCore.Qt.ItemDataRole.DecorationRole
        self.assertEqual(self.model.comment_display, self.display.TEXT)
        self.assertEqual(self.model.data(self.annotated), "Check music …")
        self.assertEqual(self.model.data(self.blank), "")
        self.assertIsNone(self.model.data(self.annotated, decoration))
        self.assertIsNone(self.model.data(self.blank, QtCore.Qt.ItemDataRole.ToolTipRole))
        self.assertEqual(self.model.headerData(self.col.COMMENT, QtCore.Qt.Orientation.Horizontal), "Comment")
        self.assertIsNone(self.model.headerData(self.col.COMMENT, QtCore.Qt.Orientation.Horizontal, decoration))

    def test_icon_mode_swaps_text_for_presence_icons_and_keeps_the_comment_available(self):
        decoration = QtCore.Qt.ItemDataRole.DecorationRole
        tooltip = QtCore.Qt.ItemDataRole.ToolTipRole
        horizontal = QtCore.Qt.Orientation.Horizontal

        self.model.set_comment_display(self.display.ICON)

        self.assertEqual(self.model.comment_display, self.display.ICON)
        self.assertEqual(self.model.data(self.annotated), "")
        self.assertEqual(self.model.data(self.annotated, decoration).name(), "comment-present")
        self.assertEqual(self.model.data(self.blank, decoration).name(), "comment-empty")
        self.assertEqual(self.model.data(self.annotated, QtCore.Qt.ItemDataRole.EditRole), "Check music\nbefore heat 2")
        self.assertEqual(self.model.data(self.annotated, tooltip), "Check music\nbefore heat 2")
        self.assertEqual(self.model.data(self.blank, tooltip), "Click to add a comment")
        self.assertEqual(self.model.headerData(self.col.COMMENT, horizontal), "")
        self.assertEqual(self.model.headerData(self.col.COMMENT, horizontal, decoration).name(), "comment-empty")
        self.assertEqual(self.model.headerData(self.col.COMMENT, horizontal, tooltip), "Comment")
        # The other columns are untouched.
        self.assertEqual(self.model.headerData(self.col.ROUND, horizontal), "Round")
        self.assertIsNone(self.model.data(self.model.index(0, self.col.ROUND), decoration))

    def test_switching_notifies_the_comment_column_once_and_is_idempotent(self):
        data_changes: list[tuple[int, int, int, int]] = []
        header_changes: list[tuple[int, int]] = []
        self.model.dataChanged.connect(
            lambda top_left, bottom_right, _roles=(): data_changes.append(
                (top_left.row(), top_left.column(), bottom_right.row(), bottom_right.column())
            )
        )
        self.model.headerDataChanged.connect(lambda _orientation, first, last: header_changes.append((first, last)))

        self.model.set_comment_display(self.display.ICON)
        self.model.set_comment_display(self.display.ICON)

        comment = int(self.col.COMMENT)
        self.assertEqual(data_changes, [(0, comment, 2, comment)])
        self.assertEqual(header_changes, [(comment, comment)])

        self.model.set_comment_display(self.display.TEXT)
        self.assertEqual(len(data_changes), 2)
        self.assertEqual(self.model.data(self.annotated), "Check music …")

    def test_switching_an_empty_table_only_updates_the_header(self):
        empty = ballroom_scribe.gui.EventTableModel()
        data_changes: list[object] = []
        empty.dataChanged.connect(lambda *args: data_changes.append(args))

        empty.set_comment_display(self.display.ICON)

        self.assertEqual(data_changes, [])
        self.assertEqual(empty.headerData(self.col.COMMENT, QtCore.Qt.Orientation.Horizontal), "")


def _make_search_records() -> list[ballroom_scribe.models.EventRecord]:
    return [
        ballroom_scribe.models.EventRecord(
            file=_VIDEO_A,
            inpoint_seconds=10.0,
            level="Pre-Champ",
            style="Intl. Standard",
            dance="Waltz",
            round="Semi-Final",
        ),
        ballroom_scribe.models.EventRecord(
            file=_VIDEO_A,
            inpoint_seconds=30.0,
            level="Champ",
            style="Intl. Latin",
            dance="cha-cha",
            round="Final",
        ),
        ballroom_scribe.models.EventRecord(
            file=_VIDEO_B,
            inpoint_seconds=20.0,
            level="champ",
            style="American Smooth",
            dance="Cha.Cha (1)",
            round="semi-final",
        ),
        ballroom_scribe.models.EventRecord(
            file=_VIDEO_B,
            inpoint_seconds=50.0,
            level="Open",
            style="Open Standard",
            dance="",
            round="",
        ),
    ]


class TestFindMatches(unittest.TestCase):
    """Unit tests for `EventTableModel.find_matches`."""

    def setUp(self):
        self.model = ballroom_scribe.gui.EventTableModel()
        self.model.set_records(_make_search_records())

    @parameterized.expand(
        [
            param(
                "substring_case_insensitive_default",
                query="champ",
                kwargs={},
                expected=[
                    (0, ballroom_scribe.gui.EventTableModel.Column.LEVEL),
                    (1, ballroom_scribe.gui.EventTableModel.Column.LEVEL),
                    (2, ballroom_scribe.gui.EventTableModel.Column.LEVEL),
                ],
            ),
            param(
                "substring_case_sensitive",
                query="Champ",
                kwargs={"case_sensitive": True},
                expected=[
                    (0, ballroom_scribe.gui.EventTableModel.Column.LEVEL),
                    (1, ballroom_scribe.gui.EventTableModel.Column.LEVEL),
                ],
            ),
            param(
                "whole_cell_case_insensitive",
                query="champ",
                kwargs={"whole_cell": True},
                expected=[
                    (1, ballroom_scribe.gui.EventTableModel.Column.LEVEL),
                    (2, ballroom_scribe.gui.EventTableModel.Column.LEVEL),
                ],
            ),
            param(
                "whole_cell_case_sensitive",
                query="Champ",
                kwargs={"case_sensitive": True, "whole_cell": True},
                expected=[(1, ballroom_scribe.gui.EventTableModel.Column.LEVEL)],
            ),
            param(
                "multiple_columns_in_reading_order",
                query="open",
                kwargs={
                    "columns": (
                        ballroom_scribe.gui.EventTableModel.Column.STYLE,
                        ballroom_scribe.gui.EventTableModel.Column.LEVEL,
                    )
                },
                expected=[
                    (3, ballroom_scribe.gui.EventTableModel.Column.LEVEL),
                    (3, ballroom_scribe.gui.EventTableModel.Column.STYLE),
                ],
            ),
            param(
                "single_column_filter",
                query="open",
                kwargs={"columns": (ballroom_scribe.gui.EventTableModel.Column.STYLE,)},
                expected=[(3, ballroom_scribe.gui.EventTableModel.Column.STYLE)],
            ),
            param(
                "row_filter_deduplicated_and_sorted",
                query="final",
                kwargs={"rows": (2, 0, 2)},
                expected=[
                    (0, ballroom_scribe.gui.EventTableModel.Column.ROUND),
                    (2, ballroom_scribe.gui.EventTableModel.Column.ROUND),
                ],
            ),
            param("empty_row_filter", query="final", kwargs={"rows": ()}, expected=[]),
            param("empty_query_substring", query="", kwargs={}, expected=[]),
            param("empty_query_whole_cell", query="", kwargs={"whole_cell": True}, expected=[]),
            param(
                "regex_metacharacters_matched_literally",
                query="Cha.Cha (1)",
                kwargs={},
                expected=[(2, ballroom_scribe.gui.EventTableModel.Column.DANCE)],
            ),
        ]
    )
    def test_find_matches_variants(self, _, query: str, kwargs: dict, expected: list):
        self.assertEqual(self.model.find_matches(query, **kwargs), tuple(expected))

    def test_searches_comments_by_default_and_when_asked_for_that_column(self):
        col = ballroom_scribe.gui.EventTableModel.Column
        self.assertTrue(self.model.setData(self.model.index(3, col.COMMENT), "Waltz music\ncut short"))

        self.assertEqual(self.model.find_matches("waltz"), ((0, col.DANCE), (3, col.COMMENT)))
        self.assertEqual(self.model.find_matches("cut short", columns=(col.COMMENT,)), ((3, col.COMMENT),))
        self.assertEqual(
            self.model.find_matches("waltz", columns=ballroom_scribe.gui.EventTableModel.TEXT_COLUMNS),
            ((0, col.DANCE),),
        )
        self.assertEqual(self.model.replace_all("cut short", "faded early", columns=(col.COMMENT,)), 1)
        self.assertEqual(self.model.record_at(3).comment, "Waltz music\nfaded early")

    @parameterized.expand(
        [
            param("file_column", column=ballroom_scribe.gui.EventTableModel.Column.FILE),
            param("inpoint_column", column=ballroom_scribe.gui.EventTableModel.Column.INPOINT),
            param("inpoint_formatted_column", column=ballroom_scribe.gui.EventTableModel.Column.INPOINT_FORMATTED),
        ]
    )
    def test_rejects_non_text_columns(self, _, column: ballroom_scribe.gui.EventTableModel.Column):
        with self.assertRaises(ValueError):
            self.model.find_matches("10", columns=(column,))

    @parameterized.expand(
        [
            param("negative_row", rows=(-1,)),
            param("past_end_row", rows=(4,)),
        ]
    )
    def test_rejects_out_of_range_rows(self, _, rows: tuple[int, ...]):
        with self.assertRaises(IndexError):
            self.model.find_matches("", rows=rows)


class TestReplaceInCell(unittest.TestCase):
    """Unit tests for `EventTableModel.replace_in_cell`."""

    def setUp(self):
        self.model = ballroom_scribe.gui.EventTableModel()
        self.model.set_records(_make_search_records())

    def test_replaces_multiple_case_insensitive_occurrences_and_supports_undo_redo(self):
        col = ballroom_scribe.gui.EventTableModel.Column
        index = self.model.index(1, col.DANCE)

        changed = self.model.replace_in_cell(index, "CHA", "Cha")

        self.assertTrue(changed)
        self.assertEqual(self.model.record_at(1).dance, "Cha-Cha")
        self.assertEqual(self.model.undo_stack.undoText(), "Replace Dance")
        self.assertFalse(self.model.undo_stack.isClean())

        self.model.undo_stack.undo()
        self.assertEqual(self.model.record_at(1).dance, "cha-cha")
        self.assertTrue(self.model.undo_stack.isClean())

        self.model.undo_stack.redo()
        self.assertEqual(self.model.record_at(1).dance, "Cha-Cha")

    def test_treats_backslashes_literally_and_trims_outer_whitespace(self):
        col = ballroom_scribe.gui.EventTableModel.Column
        style_index = self.model.index(3, col.STYLE)

        self.assertTrue(self.model.replace_in_cell(style_index, "Open", ""))
        self.assertEqual(self.model.record_at(3).style, "Standard")

        dance_index = self.model.index(0, col.DANCE)
        self.assertTrue(self.model.replace_in_cell(dance_index, "Waltz", r"Slow\Waltz"))
        self.assertEqual(self.model.record_at(0).dance, r"Slow\Waltz")

    @parameterized.expand(
        [
            param("empty_query", row=0, query="", replacement="X", kwargs={}),
            param("non_matching_query", row=0, query="Samba", replacement="X", kwargs={}),
            param("case_mismatch", row=2, query="Cha", replacement="X", kwargs={"case_sensitive": True}),
            param("whole_cell_mismatch", row=0, query="Champ", replacement="X", kwargs={"whole_cell": True}),
            param("identical_result", row=0, query="Pre-Champ", replacement="Pre-Champ", kwargs={}),
            param("invalid_index", row=-1, query="Champ", replacement="X", kwargs={}),
        ]
    )
    def test_noop_cases_leave_undo_stack_clean(
        self,
        _,
        row: int,
        query: str,
        replacement: str,
        kwargs: dict,
    ):
        col = ballroom_scribe.gui.EventTableModel.Column
        index = QtCore.QModelIndex() if row < 0 else self.model.index(row, col.LEVEL)

        self.assertFalse(self.model.replace_in_cell(index, query, replacement, **kwargs))
        self.assertEqual(self.model.undo_stack.count(), 0)
        self.assertTrue(self.model.undo_stack.isClean())

    def test_rejects_non_text_column(self):
        col = ballroom_scribe.gui.EventTableModel.Column
        with self.assertRaises(ValueError):
            self.model.replace_in_cell(self.model.index(0, col.INPOINT), "10", "20")


class TestReplaceAll(unittest.TestCase):
    """Unit tests for `EventTableModel.replace_all`."""

    def setUp(self):
        self.model = ballroom_scribe.gui.EventTableModel()
        self.model.set_records(_make_search_records())

    def test_replaces_across_rows_and_multiple_columns_in_same_row_as_single_undo_step(self):
        original = list(self.model.records)

        changed_cells = self.model.replace_all("open", "Championship")

        self.assertEqual(changed_cells, 2)
        self.assertEqual(self.model.record_at(3).level, "Championship")
        self.assertEqual(self.model.record_at(3).style, "Championship Standard")
        self.assertEqual(self.model.undo_stack.count(), 1)
        self.assertEqual(self.model.undo_stack.undoText(), "Replace All")

        self.model.undo_stack.undo()
        self.assertEqual(list(self.model.records), original)
        self.assertTrue(self.model.undo_stack.isClean())

        self.model.undo_stack.redo()
        self.assertEqual(self.model.record_at(3).level, "Championship")
        self.assertEqual(self.model.record_at(3).style, "Championship Standard")

    def test_respects_whole_cell_case_column_and_row_filters(self):
        col = ballroom_scribe.gui.EventTableModel.Column
        changed = self.model.replace_all(
            "champ",
            "Championship",
            columns=(col.LEVEL,),
            rows=(0, 1, 2),
            whole_cell=True,
        )

        self.assertEqual(changed, 2)
        self.assertEqual(self.model.record_at(0).level, "Pre-Champ")
        self.assertEqual(self.model.record_at(1).level, "Championship")
        self.assertEqual(self.model.record_at(2).level, "Championship")

    def test_noop_pushes_nothing_onto_undo_stack(self):
        self.assertEqual(self.model.replace_all("Nonexistent", "X"), 0)
        self.assertEqual(self.model.replace_all("", "X"), 0)
        self.assertEqual(self.model.replace_all("Waltz", "Waltz"), 0)
        self.assertEqual(self.model.undo_stack.count(), 0)
        self.assertTrue(self.model.undo_stack.isClean())

    def test_validates_columns_and_rows_before_mutating(self):
        col = ballroom_scribe.gui.EventTableModel.Column
        original = list(self.model.records)

        with self.assertRaises(ValueError):
            self.model.replace_all("10", "20", columns=(col.FILE,))
        with self.assertRaises(IndexError):
            self.model.replace_all("Champ", "X", rows=(0, 99))

        self.assertEqual(list(self.model.records), original)
        self.assertEqual(self.model.undo_stack.count(), 0)


if __name__ == "__main__":
    unittest.main()
