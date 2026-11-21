import unittest

from parameterized import param, parameterized

import ballroom_scribe.gui.utils


class TestSplitConsoleLines(unittest.TestCase):
    """Unit tests for the `split_console_lines` function."""

    def test_returns_complete_lines_and_keeps_the_unfinished_rest(self):
        lines, pending = ballroom_scribe.gui.utils.split_console_lines("", "Checking input videos...\nExtracting au")

        self.assertEqual(lines, ["Checking input videos..."])
        self.assertEqual(pending, "Extracting au")

    def test_finishes_the_line_that_was_pending(self):
        lines, pending = ballroom_scribe.gui.utils.split_console_lines("Extracting au", "dio from clip.mp4...\n")

        self.assertEqual(lines, ["Extracting audio from clip.mp4..."])
        self.assertEqual(pending, "")

    def test_text_without_a_line_ending_stays_pending(self):
        lines, pending = ballroom_scribe.gui.utils.split_console_lines("The model", " is thinking")

        self.assertEqual(lines, [])
        self.assertEqual(pending, "The model is thinking")

    @parameterized.expand(
        [
            param("line_feed", text="one\ntwo\n", expected=["one", "two"]),
            param("carriage_return_and_line_feed", text="one\r\ntwo\r\n", expected=["one", "two"]),
            # Every update of a progress line that rewrites itself is a line.
            param("carriage_return", text="size=16KiB\rsize=32KiB\r", expected=["size=16KiB", "size=32KiB"]),
            param("mixed", text="size=16KiB\r\nsize=32KiB\rdone\n", expected=["size=16KiB", "size=32KiB", "done"]),
        ]
    )
    def test_every_kind_of_line_ending_ends_a_line(self, _, text, expected):
        lines, pending = ballroom_scribe.gui.utils.split_console_lines("", text)

        self.assertEqual(lines, expected)
        self.assertEqual(pending, "")

    def test_leaves_out_blank_lines(self):
        lines, pending = ballroom_scribe.gui.utils.split_console_lines("", "\nDetecting events...\n\n   \nThinking:\n")

        self.assertEqual(lines, ["Detecting events...", "Thinking:"])
        self.assertEqual(pending, "")

    def test_line_ending_that_is_split_between_two_chunks_counts_once(self):
        first_lines, pending = ballroom_scribe.gui.utils.split_console_lines("", "Checking input videos...\r")
        second_lines, pending = ballroom_scribe.gui.utils.split_console_lines(pending, "\nExtracting audio...\r\n")

        self.assertEqual(first_lines + second_lines, ["Checking input videos...", "Extracting audio..."])
        self.assertEqual(pending, "")

    @parameterized.expand(
        [
            param("single_character_reads", step=1),
            param("two_character_reads", step=2),
            param("three_character_reads", step=3),
            param("seven_character_reads", step=7),
            param("seventeen_character_reads", step=17),
        ]
    )
    def test_produces_identical_lines_regardless_of_pipe_chunk_boundaries(self, _, step: int):
        stream = (
            "Checking input videos...\r\n"
            "\r\n"
            "Extracting audio from a.mp4...\r\n"
            "size=   16KiB time=00:00:00.25\r"
            "size=   32KiB time=00:00:00.50\r\r"
            "[out#0/wav] summary\r\n"
            "Processing text: Welcome dancers\n"
            'Response: {"events": []}\n'
            "Traceback (most recent call last):\n"
            "RuntimeError: CUDA out of memory"
        )
        whole_lines, whole_pending = ballroom_scribe.gui.utils.split_console_lines("", stream)

        chunked_lines: list[str] = []
        chunked_pending = ""
        for index in range(0, len(stream), step):
            lines, chunked_pending = ballroom_scribe.gui.utils.split_console_lines(
                chunked_pending,
                stream[index : index + step],
            )
            chunked_lines.extend(lines)

        self.assertEqual(chunked_lines, whole_lines)
        self.assertEqual(chunked_pending, whole_pending)
        self.assertEqual(
            chunked_lines,
            [
                "Checking input videos...",
                "Extracting audio from a.mp4...",
                "size=   16KiB time=00:00:00.25",
                "size=   32KiB time=00:00:00.50",
                "[out#0/wav] summary",
                "Processing text: Welcome dancers",
                'Response: {"events": []}',
                "Traceback (most recent call last):",
            ],
        )
        # The last line has no line ending yet, so it is still pending.
        self.assertEqual(chunked_pending, "RuntimeError: CUDA out of memory")


if __name__ == "__main__":
    unittest.main()
