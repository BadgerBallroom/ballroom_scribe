import csv
import pathlib
import unittest
from collections.abc import Generator
from unittest.mock import call, mock_open, patch

from parameterized import param, parameterized

import ballroom_scribe
import ballroom_scribe.models
import ballroom_scribe.utils

_FAKE_INPUT_VIDEO_A = pathlib.Path("fake_input_a.mp4")
_FAKE_INPUT_VIDEO_B = pathlib.Path("fake_input_b.mp4")
_FAKE_OUTPUT_CSV = pathlib.Path("fake_output.csv")


class TestRecord(unittest.TestCase):
    """Unit tests for `EventRecorder.record`."""

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
                    (
                        _FAKE_INPUT_VIDEO_A,
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
                    ),
                    (
                        _FAKE_INPUT_VIDEO_A,
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
                    ),
                    (
                        _FAKE_INPUT_VIDEO_B,
                        ballroom_scribe.models.Event(
                            level="Gold",
                            style="Latin",
                            dance="Cha Cha",
                            round="Semifinal",
                            announcement_quote="Gold Latin Cha Cha semifinal",
                            announcement_start_seconds=5.0,
                            speech_detected=True,
                            event_detected=True,
                        ),
                    ),
                ],
                formatted_inpoints=["0:00:12.35", "0:01:07.89", "0:00:05.00"],
                expected_rows=[
                    ("File", "Inpoint", "Inpoint Formatted", "Level", "Style", "Dance", "Round", "Comment"),
                    (_FAKE_INPUT_VIDEO_A, 12.345, "0:00:12.35", "Bronze", "Standard", "Waltz", "Final", ""),
                    (_FAKE_INPUT_VIDEO_A, 67.890, "0:01:07.89", "Silver", "Smooth", "Tango", "Final", ""),
                    (_FAKE_INPUT_VIDEO_B, 5.0, "0:00:05.00", "Gold", "Latin", "Cha Cha", "Semifinal", ""),
                ],
            ),
            param(
                "empty_events",
                events=[],
                formatted_inpoints=[],
                expected_rows=[
                    ("File", "Inpoint", "Inpoint Formatted", "Level", "Style", "Dance", "Round", "Comment"),
                ],
            ),
        ]
    )
    def test_record(self, _, events, formatted_inpoints, expected_rows):
        self.mock_format_duration.side_effect = formatted_inpoints

        recorder = ballroom_scribe.EventRecorder()
        recorder.record(events=events, target=_FAKE_OUTPUT_CSV)

        self.mock_open.assert_called_once_with(_FAKE_OUTPUT_CSV, "w", newline="", encoding="utf-8")
        self.mock_csv_writer.assert_called_once_with(self.mock_open())
        self.assertEqual(
            self.mock_format_duration.call_args_list,
            [call(event.announcement_start_seconds) for _, event in events],
        )
        self.assertEqual(
            self.mock_csv_writer.return_value.writerow.call_args_list,
            [call(row) for row in expected_rows],
        )


class TestWrite(unittest.TestCase):
    """Unit tests for `EventRecorder.write`."""

    def setUp(self):
        self.mock_open = mock_open()
        self.enterContext(patch("builtins.open", self.mock_open))
        self.mock_csv_writer = self.enterContext(patch.object(csv, "writer", autospec=True))

    def test_writes_header_and_records(self):
        records = [
            ballroom_scribe.models.EventRecord(
                file=_FAKE_INPUT_VIDEO_A,
                inpoint_seconds=12.345,
                level="Bronze",
                style="Standard",
                dance="Waltz",
                round="Final",
                comment="Check music,\nthen lights",
            ),
            ballroom_scribe.models.EventRecord(
                file=_FAKE_INPUT_VIDEO_B,
                inpoint_seconds=5.0,
                level="",
                style="",
                dance="",
                round="",
            ),
        ]

        recorder = ballroom_scribe.EventRecorder()
        recorder.write(records=records, target=_FAKE_OUTPUT_CSV)

        self.mock_open.assert_called_once_with(_FAKE_OUTPUT_CSV, "w", newline="", encoding="utf-8")
        self.assertEqual(
            self.mock_csv_writer.return_value.writerow.call_args_list,
            [
                call(("File", "Inpoint", "Inpoint Formatted", "Level", "Style", "Dance", "Round", "Comment")),
                call(
                    (
                        _FAKE_INPUT_VIDEO_A,
                        12.345,
                        "0:00:12.34",
                        "Bronze",
                        "Standard",
                        "Waltz",
                        "Final",
                        "Check music,\nthen lights",
                    )
                ),
                call((_FAKE_INPUT_VIDEO_B, 5.0, "0:00:05.00", "", "", "", "", "")),
            ],
        )


class TestRead(unittest.TestCase):
    """Unit tests for `EventRecorder.read`."""

    def _mock_csv(self, content: str) -> None:
        self.mock_open = mock_open(read_data=content)
        self.enterContext(patch("builtins.open", self.mock_open))

    def test_streams_records_and_ignores_formatted_inpoint(self):
        self._mock_csv(
            "File,Inpoint,Inpoint Formatted,Level,Style,Dance,Round\n"
            "fake_input_a.mp4,12.345,99:99:99.99,Bronze,Standard,Waltz,Final\n"
            "fake_input_b.mp4,5.0,0:00:05.00,,,,\n"
        )
        recorder = ballroom_scribe.EventRecorder()

        stream = recorder.read(_FAKE_OUTPUT_CSV)
        self.assertIsInstance(stream, Generator)
        records = list(stream)

        self.mock_open.assert_called_once_with(_FAKE_OUTPUT_CSV, newline="", encoding="utf-8")
        self.assertEqual(
            records,
            [
                ballroom_scribe.models.EventRecord(
                    file=_FAKE_INPUT_VIDEO_A,
                    inpoint_seconds=12.345,
                    level="Bronze",
                    style="Standard",
                    dance="Waltz",
                    round="Final",
                ),
                ballroom_scribe.models.EventRecord(
                    file=_FAKE_INPUT_VIDEO_B,
                    inpoint_seconds=5.0,
                    level="",
                    style="",
                    dance="",
                    round="",
                ),
            ],
        )

    def test_reports_extra_columns_and_ignores_values(self):
        self._mock_csv(
            "File,Inpoint,Inpoint Formatted,Level,Style,Dance,Round,Notes,Heat\n"
            "fake_input_a.mp4,1.5,0:00:01.50,Silver,Smooth,Foxtrot,Final,check camera,42\n"
        )
        captured: list[tuple[str, ...]] = []
        recorder = ballroom_scribe.EventRecorder()

        records = list(recorder.read(_FAKE_OUTPUT_CSV, on_extra_columns=captured.append))

        self.assertEqual(captured, [("Notes", "Heat")])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].dance, "Foxtrot")
        self.assertEqual(records[0].comment, "")

    def test_reads_optional_comment_column_including_quoted_commas_and_line_breaks(self):
        self._mock_csv(
            "File,Inpoint,Inpoint Formatted,Level,Style,Dance,Round,Comment,Heat\n"
            'fake_input_a.mp4,1.5,0:00:01.50,Silver,Smooth,Foxtrot,Final,"Check music,\nthen lights",42\n'
            "fake_input_b.mp4,5.0,0:00:05.00,,,,,,43\n"
        )
        captured: list[tuple[str, ...]] = []
        recorder = ballroom_scribe.EventRecorder()

        records = list(recorder.read(_FAKE_OUTPUT_CSV, on_extra_columns=captured.append))

        self.assertEqual(captured, [("Heat",)])  # `Comment` is a known column, so only `Heat` is extra.
        self.assertEqual([record.comment for record in records], ["Check music,\nthen lights", ""])

    @parameterized.expand(
        [
            param(
                "empty_file",
                content="",
                expected_regex="CSV file has no header row",
            ),
            param(
                "missing_required_columns",
                content="File,Inpoint,Level\na.mp4,1.0,Gold\n",
                expected_regex=r"missing required column\(s\): 'Style', 'Dance', 'Round'",
            ),
            param(
                "short_row",
                content="File,Inpoint,Level,Style,Dance,Round\na.mp4,1.0,Gold\n",
                expected_regex="Row 1: missing value for column 'Style'",
            ),
            param(
                "empty_file_column",
                content="File,Inpoint,Level,Style,Dance,Round\n ,1.0,Gold,Latin,Rumba,Final\n",
                expected_regex="Row 1: 'File' cannot be empty",
            ),
            param(
                "non_numeric_inpoint",
                content=(
                    "File,Inpoint,Level,Style,Dance,Round\n"
                    "a.mp4,1.0,Gold,Latin,Rumba,Final\n"
                    "a.mp4,bad,Gold,Latin,Jive,Final\n"
                ),
                expected_regex="Row 2: invalid 'Inpoint' value 'bad'",
            ),
            param(
                "negative_inpoint",
                content="File,Inpoint,Level,Style,Dance,Round\na.mp4,-2.5,Gold,Latin,Rumba,Final\n",
                expected_regex="Row 1: 'Inpoint' must be a non-negative finite number",
            ),
            param(
                "nan_inpoint",
                content="File,Inpoint,Level,Style,Dance,Round\na.mp4,nan,Gold,Latin,Rumba,Final\n",
                expected_regex="Row 1: 'Inpoint' must be a non-negative finite number",
            ),
        ]
    )
    def test_invalid_csv_raises_value_error(self, _, content: str, expected_regex: str):
        self._mock_csv(content)
        recorder = ballroom_scribe.EventRecorder()

        with self.assertRaisesRegex(ValueError, expected_regex):
            list(recorder.read(_FAKE_OUTPUT_CSV))


if __name__ == "__main__":
    unittest.main()
