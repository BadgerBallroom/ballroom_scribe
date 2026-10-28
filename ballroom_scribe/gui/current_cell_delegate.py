"""Item delegate that frames the current cell and faintly highlights rows sharing the selected row's event."""

from typing import override

from PySide6 import QtCore, QtGui, QtWidgets

import ballroom_scribe.gui
import ballroom_scribe.gui.utils

_FRAME_WIDTH = 2
_MATCH_HIGHLIGHT_ALPHA = 45
_SEARCH_MATCH_RGB = (255, 195, 0)
_SEARCH_MATCH_ALPHA = 70
_ACTIVE_SEARCH_MATCH_ALPHA = 110


def _color_group(option: QtWidgets.QStyleOptionViewItem) -> QtGui.QPalette.ColorGroup:
    """Returns the palette group styles use for `option`, as `QCommonStyle` does when painting items."""
    if not option.state & QtWidgets.QStyle.StateFlag.State_Enabled:
        return QtGui.QPalette.ColorGroup.Disabled
    if not option.state & QtWidgets.QStyle.StateFlag.State_Active:
        return QtGui.QPalette.ColorGroup.Inactive
    return QtGui.QPalette.ColorGroup.Active


def _paint_plain_background(painter: QtGui.QPainter, option: QtWidgets.QStyleOptionViewItem) -> None:
    """Covers the selection fill some styles paint for the whole row before the delegate runs.

    Composes the background the way the view does for an unselected cell: `Base` from the viewport, with
    `AlternateBase` (translucent on macOS) layered on top for alternate rows.
    """
    group = _color_group(option)
    painter.fillRect(option.rect, option.palette.brush(group, QtGui.QPalette.ColorRole.Base))
    if option.features & QtWidgets.QStyleOptionViewItem.ViewItemFeature.Alternate:
        painter.fillRect(option.rect, option.palette.brush(group, QtGui.QPalette.ColorRole.AlternateBase))


def _paint_match_highlight(painter: QtGui.QPainter, option: QtWidgets.QStyleOptionViewItem) -> None:
    """Tints `option.rect` with a faint wash of the palette's selection colour."""
    color = QtGui.QColor(option.palette.color(_color_group(option), QtGui.QPalette.ColorRole.Highlight))
    color.setAlpha(_MATCH_HIGHLIGHT_ALPHA)
    painter.fillRect(option.rect, color)


def _paint_search_highlight(
    painter: QtGui.QPainter,
    option: QtWidgets.QStyleOptionViewItem,
    *,
    active: bool,
) -> None:
    """Tints `option.rect` with an amber wash marking a Find/Replace match."""
    alpha = _ACTIVE_SEARCH_MATCH_ALPHA if active else _SEARCH_MATCH_ALPHA
    painter.fillRect(option.rect, QtGui.QColor(*_SEARCH_MATCH_RGB, alpha))


def _unselected_plain_option(option: QtWidgets.QStyleOptionViewItem) -> QtWidgets.QStyleOptionViewItem:
    """Returns `option` stripped of selection, focus, and alternate-row flags for painting over a custom fill."""
    plain = QtWidgets.QStyleOptionViewItem(option)
    plain.state = plain.state & ~(QtWidgets.QStyle.StateFlag.State_Selected | QtWidgets.QStyle.StateFlag.State_HasFocus)
    # Clear `Alternate` so the style does not paint an opaque `AlternateBase` over the custom background/tint.
    plain.features = plain.features & ~QtWidgets.QStyleOptionViewItem.ViewItemFeature.Alternate
    return plain


def _paint_frame(painter: QtGui.QPainter, option: QtWidgets.QStyleOptionViewItem) -> None:
    """Draws a `_FRAME_WIDTH` px frame in the selection colour just inside the cell."""
    highlight = option.palette.color(_color_group(option), QtGui.QPalette.ColorRole.Highlight)
    ballroom_scribe.gui.utils.paint_frame(painter, option.rect, highlight, _FRAME_WIDTH)


def _shares_selected_event(
    option: QtWidgets.QStyleOptionViewItem,
    index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
) -> bool:
    """Returns whether `index` is on an unselected row sharing `(Level, Style, Dance)` with the selected row."""
    view = option.widget
    if not isinstance(view, ballroom_scribe.gui.EventTableView):
        return False
    selected_row = view.selected_row()
    if selected_row is None or selected_row == index.row():
        return False
    model = index.model()
    if not isinstance(model, ballroom_scribe.gui.EventTableModel):
        return False
    return model.has_same_event(selected_row, index.row())


def _selects_several_rows(option: QtWidgets.QStyleOptionViewItem) -> bool:
    """Returns whether `option`'s view has two or more selected rows."""
    view = option.widget
    return isinstance(view, ballroom_scribe.gui.EventTableView) and view.has_multiple_selected_rows


def _search_highlight_state(
    option: QtWidgets.QStyleOptionViewItem,
    index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
) -> tuple[bool, bool]:
    """Returns `(is_search_match, is_active_search_match)` for `index` in `option`'s view."""
    view = option.widget
    if not isinstance(view, ballroom_scribe.gui.EventTableView):
        return False, False
    cell = (index.row(), index.column())
    is_active = cell == view.active_search_match
    return is_active or cell in view.search_matches, is_active


class CurrentCellDelegate(QtWidgets.QStyledItemDelegate):
    """Paints the focused view's current cell and faintly highlights rows matching the selected row's event.

    With `SelectRows` selection behaviour every cell of the selected row is highlighted alike, and some styles (such
    as Windows 11) draw no focus indicator on top, so keyboard users cannot tell which cell `Enter`, `F2`, or typing
    will edit. While the view has keyboard focus (exactly when those keys reach it), the current cell keeps the row's
    plain background and gets a thick frame in the selection colour, which stands out on light and dark palettes
    alike and moves with the arrow keys. Other rows with the same non-empty `Level`, `Style`, and `Dance` as the
    selected row receive a faint tint of the selection colour over their plain background. When two or more rows are
    selected, no cell is singled out: the focused cell is painted like the rest of its selected row (without a frame or
    focus rectangle) and the faint same-event highlight is off.
    """

    @override
    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
    ) -> None:
        is_search_match, is_active_search_match = _search_highlight_state(option, index)
        if is_active_search_match:
            plain = _unselected_plain_option(option)
            _paint_plain_background(painter, option)
            _paint_search_highlight(painter, option, active=True)
            super().paint(painter, plain, index)
            _paint_frame(painter, option)
            return
        if option.state & QtWidgets.QStyle.StateFlag.State_HasFocus:
            if _selects_several_rows(option) and not is_search_match:
                # Several rows are selected: no cell is singled out, so paint this one like the rest of its row.
                unframed = QtWidgets.QStyleOptionViewItem(option)
                unframed.state = unframed.state & ~QtWidgets.QStyle.StateFlag.State_HasFocus
                super().paint(painter, unframed, index)
                return
            if not _selects_several_rows(option):
                plain = _unselected_plain_option(option)
                _paint_plain_background(painter, option)
                if is_search_match:
                    _paint_search_highlight(painter, option, active=False)
                super().paint(painter, plain, index)
                _paint_frame(painter, option)
                return
        if is_search_match:
            plain = _unselected_plain_option(option)
            _paint_plain_background(painter, option)
            _paint_search_highlight(painter, option, active=False)
            super().paint(painter, plain, index)
            return
        if not (option.state & QtWidgets.QStyle.StateFlag.State_Selected) and _shares_selected_event(option, index):
            plain = _unselected_plain_option(option)
            _paint_plain_background(painter, option)
            _paint_match_highlight(painter, option)
            super().paint(painter, plain, index)
            return
        super().paint(painter, option, index)
