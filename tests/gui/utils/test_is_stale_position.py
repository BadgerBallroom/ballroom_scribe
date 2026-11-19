import unittest

from parameterized import param, parameterized

import ballroom_scribe.gui.utils


class TestIsStalePosition(unittest.TestCase):
    """Unit tests for `is_stale_position`."""

    @parameterized.expand(
        [
            param("where_playback_started", start=1.0, seconds=1.0, elapsed=0.0),
            param("one_frame_after_the_start", start=0.998, seconds=1.041, elapsed=0.05),
            param("as_far_as_real_time_allows", start=1.0, seconds=1.25, elapsed=0.25),
            param("ahead_of_real_time_by_the_allowed_lead", start=1.0, seconds=1.75, elapsed=0.25),
            param("the_next_frame_of_a_slow_video_at_once", start=10.2, seconds=10.5, elapsed=0.0),
            param("behind_real_time", start=1.0, seconds=1.1, elapsed=0.9),
            param("at_the_end_of_the_checked_time", start=1.0, seconds=2.5, elapsed=1.0),
        ]
    )
    def test_believes_a_position_that_playback_can_have_reached(self, _, start, seconds, elapsed):
        self.assertFalse(ballroom_scribe.gui.utils.is_stale_position(start, seconds, elapsed))

    @parameterized.expand(
        [
            # What Qt's FFmpeg backend reported on Windows: the target of the first of three seeks, 64 ms after
            # playback started at the target of the last one.
            param("the_target_of_an_earlier_seek_ahead", start=0.998, seconds=7.501, elapsed=0.064),
            param("the_target_of_an_earlier_seek_behind", start=7.5, seconds=1.0, elapsed=0.064),
            param("just_behind_the_start", start=1.0, seconds=0.999, elapsed=0.5),
            param("just_beyond_the_allowed_lead", start=1.0, seconds=1.751, elapsed=0.25),
            param("far_ahead_at_the_end_of_the_checked_time", start=1.0, seconds=2.501, elapsed=1.0),
            param("behind_the_start_at_the_end_of_the_checked_time", start=1.0, seconds=0.5, elapsed=1.0),
        ]
    )
    def test_ignores_a_position_that_playback_cannot_have_reached(self, _, start, seconds, elapsed):
        self.assertTrue(ballroom_scribe.gui.utils.is_stale_position(start, seconds, elapsed))

    @parameterized.expand(
        [
            param("far_ahead", start=1.0, seconds=7.5, elapsed=1.001),
            param("behind_the_start", start=7.5, seconds=1.0, elapsed=1.001),
            param("far_ahead_long_after_the_start", start=1.0, seconds=3600.0, elapsed=60.0),
        ]
    )
    def test_believes_any_position_after_the_checked_time(self, _, start, seconds, elapsed):
        self.assertFalse(ballroom_scribe.gui.utils.is_stale_position(start, seconds, elapsed))


if __name__ == "__main__":
    unittest.main()
