import csv
import pathlib
import unittest
from unittest.mock import call, mock_open, patch

from parameterized import param, parameterized

import ballroom_scribe
import ballroom_scribe.models
import ballroom_scribe.utils

_FAKE_INPUT_VIDEO = pathlib.Path("fake_input.mp4")
_FAKE_OUTPUT_CSV = pathlib.Path("fake_output.csv")


class TestEventRecorder(unittest.TestCase):
    """Unit tests for the `EventRecorder` class."""

    def setUp(self):
        self.mock_open = mock_open()
        self.enterContext(patch("builtins.open", self.mock_open))

        self.mock_csv_writer = self.enterContext(patch.object(csv, "writer", autospec=True))
        self.mock_format_duration = self.enterContext(
            patch.object(ballroom_scribe.utils, "format_duration", autospec=True)
        )

    @parameterized.expand(
        [
            param(
                "with_events",
                events=[
                    ballroom_scribe.models.Event(
                        level="Bronze",
                        style="Standard",
                        dance="Waltz",
                        round="Final",
                        announcement_quote="Next up on the floor bronze waltz",
                        announcement_start_seconds=12.345,
                        speech_detected=True,
                        event_detected=True,
                    ),
                    ballroom_scribe.models.Event(
                        level="Silver",
                        style="Smooth",
                        dance="Tango",
                        round="Final",
                        announcement_quote="Moving on to silver tango",
                        announcement_start_seconds=67.890,
                        speech_detected=True,
                        event_detected=True,
                    ),
                ],
                formatted_inpoints=["0:00:12.35", "0:01:07.89"],
                expected_rows=[
                    ("File", "Inpoint", "Inpoint Formatted", "Level", "Style", "Dance", "Round"),
                    (_FAKE_INPUT_VIDEO, 12.345, "0:00:12.35", "Bronze", "Standard", "Waltz", "Final"),
                    (_FAKE_INPUT_VIDEO, 67.890, "0:01:07.89", "Silver", "Smooth", "Tango", "Final"),
                ],
            ),
            param(
                "empty_events",
                events=[],
                formatted_inpoints=[],
                expected_rows=[
                    ("File", "Inpoint", "Inpoint Formatted", "Level", "Style", "Dance", "Round"),
                ],
            ),
        ]
    )
    def test_record(self, _, events, formatted_inpoints, expected_rows):
        self.mock_format_duration.side_effect = formatted_inpoints

        recorder = ballroom_scribe.EventRecorder()
        recorder.record(input_video=_FAKE_INPUT_VIDEO, events=events, target=_FAKE_OUTPUT_CSV)

        self.mock_open.assert_called_once_with(_FAKE_OUTPUT_CSV, "w", newline="", encoding="utf-8")
        self.mock_csv_writer.assert_called_once_with(self.mock_open())
        self.assertEqual(
            self.mock_format_duration.call_args_list,
            [call(event.announcement_start_seconds) for event in events],
        )
        self.assertEqual(
            self.mock_csv_writer.return_value.writerow.call_args_list,
            [call(row) for row in expected_rows],
        )


if __name__ == "__main__":
    unittest.main()
