import unittest

from parameterized import param, parameterized

try:
    from PySide6 import QtGui
except ImportError as err:
    raise unittest.SkipTest("PySide6 is not installed.") from err

import ballroom_scribe.gui.utils


class TestAppendConsoleOutput(unittest.TestCase):
    """Unit tests for `append_console_output`."""

    @parameterized.expand(
        [
            param("empty_not_pending", initial="old", text="", overwrite=False, expected_text="old", expected_cr=False),
            param("empty_pending", initial="old", text="", overwrite=True, expected_text="old", expected_cr=True),
            param(
                "bare_cr_defers_overwrite",
                initial="old",
                text="\r",
                overwrite=False,
                expected_text="old",
                expected_cr=True,
            ),
            param(
                "repeated_cr_stays_pending",
                initial="old",
                text="\r\r",
                overwrite=True,
                expected_text="old",
                expected_cr=True,
            ),
            param(
                "plain_text_appends",
                initial="old",
                text=" tail",
                overwrite=False,
                expected_text="old tail",
                expected_cr=False,
            ),
            param(
                "plain_text_after_cr_replaces_last_block_only",
                initial="header\nold",
                text="new",
                overwrite=True,
                expected_text="header\nnew",
                expected_cr=False,
            ),
            param(
                "trailing_cr_keeps_text_visible_and_sets_pending",
                initial="Progress: ",
                text="50%\r",
                overwrite=False,
                expected_text="Progress: 50%",
                expected_cr=True,
            ),
            param(
                "multiple_cr_updates_collapse_and_replace_entire_last_block",
                initial="Progress: ",
                text="10%\r50%\r90%\r",
                overwrite=False,
                expected_text="90%",
                expected_cr=True,
            ),
            param(
                "crlf_acts_as_newline",
                initial="first",
                text=" rest\r\nsecond",
                overwrite=False,
                expected_text="first rest\nsecond",
                expected_cr=False,
            ),
            param(
                "lf_after_pending_cr_completes_split_crlf",
                initial="first",
                text="\nsecond",
                overwrite=True,
                expected_text="first\nsecond",
                expected_cr=False,
            ),
            param(
                "cr_on_later_line_does_not_replace_earlier_line",
                initial="first",
                text=" tail\na\rb\r",
                overwrite=False,
                expected_text="first tail\nb",
                expected_cr=True,
            ),
        ]
    )
    def test_applies_single_chunk_transitions(
        self,
        _,
        initial: str,
        text: str,
        overwrite: bool,
        expected_text: str,
        expected_cr: bool,
    ):
        document = QtGui.QTextDocument(initial)

        next_overwrite = ballroom_scribe.gui.utils.append_console_output(
            document,
            text,
            overwrite_line=overwrite,
        )

        self.assertEqual(document.toPlainText(), expected_text)
        self.assertEqual(next_overwrite, expected_cr)

    def test_rewrites_ffmpeg_progress_line_in_place_until_final_newline(self):
        document = QtGui.QTextDocument()
        overwrite = ballroom_scribe.gui.utils.append_console_output(
            document,
            "Extracting audio from clip.mp4...\n",
        )
        self.assertFalse(overwrite)

        overwrite = ballroom_scribe.gui.utils.append_console_output(
            document,
            "size=      32KiB time=00:00:00.53 bitrate= 490.9kbits/s speed=1.06x    \r",
            overwrite_line=overwrite,
        )
        self.assertTrue(overwrite)
        # The in-progress status line remains visible between updates without adding an extra empty block.
        self.assertEqual(
            document.toPlainText(),
            "Extracting audio from clip.mp4...\nsize=      32KiB time=00:00:00.53 bitrate= 490.9kbits/s speed=1.06x    ",
        )
        self.assertEqual(document.blockCount(), 2)

        overwrite = ballroom_scribe.gui.utils.append_console_output(
            document,
            "size=      64KiB time=00:00:01.06 bitrate= 512.0kbits/s speed=1.06x    \r",
            overwrite_line=overwrite,
        )
        self.assertTrue(overwrite)
        self.assertEqual(
            document.toPlainText(),
            "Extracting audio from clip.mp4...\nsize=      64KiB time=00:00:01.06 bitrate= 512.0kbits/s speed=1.06x    ",
        )
        self.assertEqual(document.blockCount(), 2)

        overwrite = ballroom_scribe.gui.utils.append_console_output(
            document,
            "[out#0/wav @ 0x1] video:0KiB audio:70KiB\n"
            "size=      70KiB time=00:00:01.20 bitrate= 512.9kbits/s speed=1.05x    \n",
            overwrite_line=overwrite,
        )
        self.assertFalse(overwrite)
        self.assertEqual(
            document.toPlainText(),
            "Extracting audio from clip.mp4...\n"
            "[out#0/wav @ 0x1] video:0KiB audio:70KiB\n"
            "size=      70KiB time=00:00:01.20 bitrate= 512.9kbits/s speed=1.05x    \n",
        )

    def test_rewrites_leading_carriage_return_progress_bars(self):
        document = QtGui.QTextDocument()
        overwrite = False
        for update in ("\r  0%|          | 0/4", "\r 50%|#####     | 2/4", "\r100%|##########| 4/4\n"):
            overwrite = ballroom_scribe.gui.utils.append_console_output(document, update, overwrite_line=overwrite)

        self.assertFalse(overwrite)
        self.assertEqual(document.toPlainText(), "100%|##########| 4/4\n")

    def test_shorter_replacement_leaves_no_leftover_tail_and_split_line_appends_rest(self):
        document = QtGui.QTextDocument()
        overwrite = ballroom_scribe.gui.utils.append_console_output(
            document,
            "size=   10240KiB time=00:05:00.00 bitrate= 512.0kbits/s speed=120.5x    \r",
        )
        # Next update arrives in two separate pipe reads and is shorter than the previous line.
        overwrite = ballroom_scribe.gui.utils.append_console_output(
            document,
            "Conversion ",
            overwrite_line=overwrite,
        )
        self.assertFalse(overwrite)
        self.assertEqual(document.toPlainText(), "Conversion ")

        overwrite = ballroom_scribe.gui.utils.append_console_output(
            document,
            "failed!\n",
            overwrite_line=overwrite,
        )
        self.assertFalse(overwrite)
        self.assertEqual(document.toPlainText(), "Conversion failed!\n")

    def test_leading_newline_after_progress_preserves_last_progress_line(self):
        document = QtGui.QTextDocument()
        overwrite = ballroom_scribe.gui.utils.append_console_output(document, "size=      32KiB time=00:00:00.53\r")
        overwrite = ballroom_scribe.gui.utils.append_console_output(
            document,
            "\nCanceling pipeline…\n",
            overwrite_line=overwrite,
        )

        self.assertFalse(overwrite)
        self.assertEqual(document.toPlainText(), "size=      32KiB time=00:00:00.53\nCanceling pipeline…\n")

    @parameterized.expand(
        [
            param("single_character_reads", step=1),
            param("two_character_reads", step=2),
            param("three_character_reads", step=3),
            param("seven_character_reads", step=7),
            param("seventeen_character_reads", step=17),
        ]
    )
    def test_produces_identical_output_regardless_of_pipe_chunk_boundaries(self, _, step: int):
        stream = (
            "Checking input videos...\r\n"
            "\r\n"
            "Extracting audio from a.mp4...\r\n"
            "size=   16KiB time=00:00:00.25\r"
            "size=   32KiB time=00:00:00.50\r\r"
            "[out#0/wav] summary\r\n"
            "size=   40KiB time=00:00:00.62\r\n"
            "Processing text: Welcome dancers\n"
            'Response: {"events": []}\n'
            "trailing progress\r"
        )
        whole_doc = QtGui.QTextDocument()
        whole_state = ballroom_scribe.gui.utils.append_console_output(whole_doc, stream)

        chunked_doc = QtGui.QTextDocument()
        chunked_state = False
        for index in range(0, len(stream), step):
            chunked_state = ballroom_scribe.gui.utils.append_console_output(
                chunked_doc,
                stream[index : index + step],
                overwrite_line=chunked_state,
            )

        self.assertEqual(chunked_doc.toPlainText(), whole_doc.toPlainText())
        self.assertEqual(chunked_state, whole_state)
        self.assertTrue(chunked_state)
        self.assertEqual(
            chunked_doc.toPlainText(),
            "Checking input videos...\n"
            "\n"
            "Extracting audio from a.mp4...\n"
            "[out#0/wav] summary\n"
            "size=   40KiB time=00:00:00.62\n"
            "Processing text: Welcome dancers\n"
            'Response: {"events": []}\n'
            "trailing progress",
        )


if __name__ == "__main__":
    unittest.main()
