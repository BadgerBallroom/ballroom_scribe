import unittest

from parameterized import param, parameterized

try:
    from PySide6 import QtCore
except ImportError as err:
    raise unittest.SkipTest("PySide6 is not installed.") from err

import ballroom_scribe.gui.event_table_view


class TestTypedCharacter(unittest.TestCase):
    """Unit tests for `_typed_character`, which classifies key presses on `EventTableView`."""

    @parameterized.expand(
        [
            param("lowercase_letter", text="s", modifiers=QtCore.Qt.KeyboardModifier.NoModifier, expected="s"),
            param("shifted_letter", text="S", modifiers=QtCore.Qt.KeyboardModifier.ShiftModifier, expected="S"),
            param("accented_letter", text="é", modifiers=QtCore.Qt.KeyboardModifier.NoModifier, expected="é"),
            param("digit", text="3", modifiers=QtCore.Qt.KeyboardModifier.NoModifier, expected="3"),
            param("keypad_digit", text="7", modifiers=QtCore.Qt.KeyboardModifier.KeypadModifier, expected="7"),
            param("space", text=" ", modifiers=QtCore.Qt.KeyboardModifier.NoModifier, expected=" "),
            param("punctuation", text=",", modifiers=QtCore.Qt.KeyboardModifier.NoModifier, expected=","),
        ]
    )
    def test_returns_printable_character(self, _, text, modifiers, expected):
        self.assertEqual(ballroom_scribe.gui.event_table_view._typed_character(text, modifiers), expected)

    @parameterized.expand(
        [
            param("ctrl_letter", text="k", modifiers=QtCore.Qt.KeyboardModifier.ControlModifier),
            param("ctrl_control_char", text="\x0b", modifiers=QtCore.Qt.KeyboardModifier.ControlModifier),
            param("alt_letter", text="f", modifiers=QtCore.Qt.KeyboardModifier.AltModifier),
            param("meta_letter", text="k", modifiers=QtCore.Qt.KeyboardModifier.MetaModifier),
            param(
                "ctrl_shift_letter",
                text="O",
                modifiers=QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.KeyboardModifier.ShiftModifier,
            ),
            param("return_key", text="\r", modifiers=QtCore.Qt.KeyboardModifier.NoModifier),
            param("tab_key", text="\t", modifiers=QtCore.Qt.KeyboardModifier.NoModifier),
            param("escape_key", text="\x1b", modifiers=QtCore.Qt.KeyboardModifier.NoModifier),
            param("delete_key", text="\x7f", modifiers=QtCore.Qt.KeyboardModifier.NoModifier),
            param("backspace_key", text="\x08", modifiers=QtCore.Qt.KeyboardModifier.NoModifier),
            param("key_without_text", text="", modifiers=QtCore.Qt.KeyboardModifier.NoModifier),
            param("multi_character_text", text="ab", modifiers=QtCore.Qt.KeyboardModifier.NoModifier),
        ]
    )
    def test_rejects_shortcuts_control_characters_and_non_character_keys(self, _, text, modifiers):
        self.assertEqual(ballroom_scribe.gui.event_table_view._typed_character(text, modifiers), "")


if __name__ == "__main__":
    unittest.main()
