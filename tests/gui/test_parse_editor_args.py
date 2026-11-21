import io
import pathlib
import sys
import unittest
from unittest.mock import patch

import ballroom_scribe.cli.arguments
import ballroom_scribe.gui


class TestParseEditorArgs(unittest.TestCase):
    """Unit tests for the `parse_editor_args` function."""

    def test_without_arguments_nothing_is_opened(self):
        cli_args, logging_args = ballroom_scribe.gui.parse_editor_args([])

        self.assertEqual(cli_args, ballroom_scribe.cli.arguments.InputCsvCacheArgs(inputs=[], csv=None))
        self.assertEqual(logging_args, ballroom_scribe.cli.arguments.LoggingArgs(verbose=False))

    def test_csv_videos_and_verbose(self):
        cli_args, logging_args = ballroom_scribe.gui.parse_editor_args(
            ["competition.csv", "-i", "part1.mp4", "1", "-c", "cache", "-i", "part2.mp4", "0", "--verbose"]
        )

        self.assertEqual(
            cli_args,
            ballroom_scribe.cli.arguments.InputCsvCacheArgs(
                inputs=[
                    ballroom_scribe.cli.arguments.InputCacheArgs(
                        input_video=pathlib.Path("part1.mp4"),
                        input_channel=1,
                        parent_cache_dir=pathlib.Path("cache"),
                    ),
                    ballroom_scribe.cli.arguments.InputCacheArgs(
                        input_video=pathlib.Path("part2.mp4"),
                        input_channel=0,
                        parent_cache_dir=pathlib.Path("."),
                    ),
                ],
                csv=pathlib.Path("competition.csv"),
            ),
        )
        self.assertEqual(logging_args, ballroom_scribe.cli.arguments.LoggingArgs(verbose=True))

    def test_help_names_the_script_that_the_user_ran(self):
        stdout = self.enterContext(patch.object(sys, "stdout", new_callable=io.StringIO))

        with self.assertRaises(SystemExit) as context:
            ballroom_scribe.gui.parse_editor_args(["--help"])

        self.assertEqual(context.exception.code, 0)
        # The editor's own process is started as a module, but the user ran `edit_timestamps.py`.
        self.assertIn("usage: edit_timestamps.py", stdout.getvalue())

    def test_mistake_ends_the_process_with_a_usage_message(self):
        stderr = self.enterContext(patch.object(sys, "stderr", new_callable=io.StringIO))

        with self.assertRaises(SystemExit) as context:
            ballroom_scribe.gui.parse_editor_args(["-i", "a_video_without_its_channel.mp4"])

        self.assertEqual(context.exception.code, 2)
        self.assertIn("edit_timestamps.py: error:", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
