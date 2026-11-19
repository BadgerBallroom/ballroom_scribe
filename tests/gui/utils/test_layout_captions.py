import unittest

from parameterized import param, parameterized

try:
    from PySide6 import QtCore
except ImportError as err:
    raise unittest.SkipTest("PySide6 is not installed.") from err

import ballroom_scribe.gui.utils

_FULL_HD = QtCore.QSize(1920, 1080)
_STRIP_HEIGHT = 60


class TestLayoutCaptions(unittest.TestCase):
    """Unit tests for `layout_captions`."""

    @parameterized.expand(
        [
            # An 800-pixel-wide picture is 450 pixels high. In a 600-pixel-high pane, the bar below the centered
            # picture is 75 pixels high, which is enough for the strip.
            param(
                "room_with_the_picture_centered",
                area=QtCore.QSize(800, 600),
                expected_video=QtCore.QRect(0, 75, 800, 450),
                expected_strip=QtCore.QRect(0, 525, 800, 60),
            ),
            # In a 530-pixel-high pane, the bar below the centered picture would be only 40 pixels high, so the
            # picture moves up by 20 pixels.
            param(
                "room_after_moving_the_picture_up",
                area=QtCore.QSize(800, 530),
                expected_video=QtCore.QRect(0, 20, 800, 450),
                expected_strip=QtCore.QRect(0, 470, 800, 60),
            ),
            param(
                "exactly_enough_room",
                area=QtCore.QSize(800, 510),
                expected_video=QtCore.QRect(0, 0, 800, 450),
                expected_strip=QtCore.QRect(0, 450, 800, 60),
            ),
        ]
    )
    def test_strip_goes_directly_below_the_picture(self, _, area, expected_video, expected_strip):
        video, strip = ballroom_scribe.gui.utils.layout_captions(area, _FULL_HD, _STRIP_HEIGHT)

        self.assertEqual(video, expected_video)
        self.assertEqual(strip, expected_strip)

    @parameterized.expand(
        [
            param("one_pixel_too_few", area=QtCore.QSize(800, 509), video_size=_FULL_HD, strip_height=_STRIP_HEIGHT),
            # The picture fills the height of the pane and has bars to its left and right.
            param(
                "pane_wider_than_the_picture",
                area=QtCore.QSize(1600, 600),
                video_size=_FULL_HD,
                strip_height=_STRIP_HEIGHT,
            ),
            param("no_strip_wanted", area=QtCore.QSize(800, 600), video_size=_FULL_HD, strip_height=0),
            param(
                "unknown_video_size",
                area=QtCore.QSize(800, 600),
                video_size=QtCore.QSize(),
                strip_height=_STRIP_HEIGHT,
            ),
            param(
                "video_without_height",
                area=QtCore.QSize(800, 600),
                video_size=QtCore.QSize(1920, 0),
                strip_height=_STRIP_HEIGHT,
            ),
        ]
    )
    def test_video_gets_the_whole_area_without_a_strip(self, _, area, video_size, strip_height):
        video, strip = ballroom_scribe.gui.utils.layout_captions(area, video_size, strip_height)

        self.assertEqual(video, QtCore.QRect(0, 0, area.width(), area.height()))
        self.assertIsNone(strip)

    def test_empty_area(self):
        video, strip = ballroom_scribe.gui.utils.layout_captions(QtCore.QSize(0, 0), _FULL_HD, _STRIP_HEIGHT)

        self.assertTrue(video.isEmpty())
        self.assertIsNone(strip)


if __name__ == "__main__":
    unittest.main()
