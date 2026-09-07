import unittest

import tests.helpers.transcript_chunk


class TestTranscriptChunk(unittest.TestCase):
    def test_get_text(self):
        chunk = tests.helpers.transcript_chunk.Maker().make(["Hello,", "world!"])
        self.assertEqual(chunk.get_text(), "Hello, world!")


if __name__ == "__main__":
    unittest.main()
