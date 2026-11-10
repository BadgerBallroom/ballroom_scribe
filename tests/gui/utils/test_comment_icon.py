import unittest

try:
    from PySide6 import QtGui
except ImportError as err:
    raise unittest.SkipTest("PySide6 is not installed.") from err

import ballroom_scribe.gui.utils


class TestCommentIcon(unittest.TestCase):
    """Unit tests for `comment_icon`."""

    def test_names_tell_the_two_states_apart(self):
        self.assertEqual(ballroom_scribe.gui.utils.comment_icon(True).name(), "comment-present")
        self.assertEqual(ballroom_scribe.gui.utils.comment_icon(False).name(), "comment-empty")

    def test_icons_are_usable_and_cached(self):
        for has_comment in (True, False):
            icon = ballroom_scribe.gui.utils.comment_icon(has_comment)
            self.assertIsInstance(icon, QtGui.QIcon)
            self.assertFalse(icon.isNull())
            self.assertIs(icon, ballroom_scribe.gui.utils.comment_icon(has_comment))


if __name__ == "__main__":
    unittest.main()
