import pathlib
import unittest
from unittest.mock import patch

from parameterized import param, parameterized

import ballroom_scribe.utils

_INPUT_VIDEO = pathlib.Path("raw_data/clip.mp4")
_CHANNEL_COUNT = 2


class TestValidateInputChannel(unittest.TestCase):
    """Tests for the `validate_input_channel` function."""

    def setUp(self):
        self.mock_count_audio_channels = self.enterContext(
            patch.object(ballroom_scribe.utils, "count_audio_channels", autospec=True)
        )
        self.mock_count_audio_channels.return_value = _CHANNEL_COUNT

    @parameterized.expand(
        [
            param("first_channel", input_channel=0),
            param("last_channel", input_channel=_CHANNEL_COUNT - 1),
        ]
    )
    def test_accepts_valid_channel(self, _, input_channel):
        """Should return without raising when the channel index is in `[0, channel_count)`."""
        ballroom_scribe.utils.validate_input_channel(_INPUT_VIDEO, input_channel)

        self.mock_count_audio_channels.assert_called_once_with(_INPUT_VIDEO)

    @parameterized.expand(
        [
            param("equal_to_channel_count", input_channel=_CHANNEL_COUNT),
            param("greater_than_channel_count", input_channel=_CHANNEL_COUNT + 3),
            param("negative", input_channel=-1),
        ]
    )
    def test_rejects_out_of_range_channel(self, _, input_channel):
        """Should raise `ValueError` if the channel index is not in `[0, channel_count)`."""
        with self.assertRaisesRegex(
            ValueError,
            rf"Audio channel index {input_channel} is out of range \(found {_CHANNEL_COUNT} channel\(s\)",
        ):
            ballroom_scribe.utils.validate_input_channel(_INPUT_VIDEO, input_channel)

    def test_rejects_video_without_audio_stream(self):
        """Should raise `ValueError` if the input has no audio stream."""
        self.mock_count_audio_channels.return_value = 0

        with self.assertRaisesRegex(ValueError, r"Audio channel index 0 is out of range \(found 0 channel\(s\)"):
            ballroom_scribe.utils.validate_input_channel(_INPUT_VIDEO, 0)


if __name__ == "__main__":
    unittest.main()
