import dataclasses
import pathlib
import unittest
from unittest.mock import patch

import ballroom_scribe.models
import ballroom_scribe.utils


class TestEventRecord(unittest.TestCase):
    """Unit tests for the `EventRecord` model."""

    def test_inpoint_formatted_delegates_to_format_duration(self):
        mock_format_duration = self.enterContext(
            patch.object(ballroom_scribe.utils, "format_duration", autospec=True, return_value="0:01:23.46")
        )
        record = ballroom_scribe.models.EventRecord(
            file=pathlib.Path("session.mp4"),
            inpoint_seconds=83.456,
            level="Silver",
            style="Smooth",
            dance="Foxtrot",
            round="Semifinal",
        )

        self.assertEqual(record.inpoint_formatted, "0:01:23.46")
        mock_format_duration.assert_called_once_with(83.456)

    def test_optional_text_fields_default_to_empty_and_comment_may_span_lines(self):
        record = ballroom_scribe.models.EventRecord(
            file=pathlib.Path("session.mp4"),
            inpoint_seconds=1.0,
        )

        for field_name in (
            "event_number",
            "flight",
            "heat",
            "age_division",
            "category",
            "level",
            "style",
            "dance",
            "dance_ordinal",
            "specialness",
            "round",
            "comment",
        ):
            self.assertEqual(getattr(record, field_name), "")
        annotated = dataclasses.replace(record, comment="Check music\nbefore heat 2")
        self.assertEqual(annotated.comment, "Check music\nbefore heat 2")
        self.assertNotEqual(annotated, record)

    def test_from_event_copies_fields(self):
        video_path = pathlib.Path("/videos/heat1.mp4")
        event = ballroom_scribe.models.Event(
            event_number="34",
            flight="2",
            heat="1",
            age_division="Adult",
            category="Amateur",
            level="Gold",
            style="Standard",
            dance="Quickstep",
            dance_ordinal="Fourth",
            specialness="Critique",
            round="Final",
            announcement_quote="Gold standard quickstep final",
            announcement_start_seconds=42.5,
            speech_detected=True,
            event_detected=True,
        )

        record = ballroom_scribe.models.EventRecord.from_event(video_path, event)

        self.assertEqual(
            record,
            ballroom_scribe.models.EventRecord(
                file=video_path,
                inpoint_seconds=42.5,
                event_number="34",
                flight="2",
                heat="1",
                age_division="Adult",
                category="Amateur",
                level="Gold",
                style="Standard",
                dance="Quickstep",
                dance_ordinal="Fourth",
                specialness="Critique",
                round="Final",
            ),
        )


if __name__ == "__main__":
    unittest.main()
