import pathlib
import unittest
from unittest.mock import call, patch

from parameterized import param, parameterized

import ballroom_scribe.utils

_FINAL_PATH = pathlib.Path("cache/Audio.wav")
_PARTIAL_PATH = pathlib.Path("cache/Audio.partial.wav")


class TestAtomicPath(unittest.TestCase):
    """Unit tests for the `atomic_path` context manager."""

    def setUp(self):
        self.mock_unlink = self.enterContext(patch.object(pathlib.Path, "unlink", autospec=True))
        self.mock_replace = self.enterContext(patch.object(pathlib.Path, "replace", autospec=True))

    @parameterized.expand(
        [
            param("wav", final_path=pathlib.Path("cache/Audio.wav"), expected=pathlib.Path("cache/Audio.partial.wav")),
            param(
                "chunk_json",
                final_path=pathlib.Path("cache/0012.34-0056.78.json"),
                expected=pathlib.Path("cache/0012.34-0056.78.partial.json"),
            ),
            param("no_suffix", final_path=pathlib.Path("cache/Speech"), expected=pathlib.Path("cache/Speech.partial")),
        ]
    )
    def test_yields_partial_path_with_same_directory_and_suffix(self, _, final_path, expected):
        with ballroom_scribe.utils.atomic_path(final_path) as partial_path:
            self.assertEqual(partial_path, expected)

    def test_removes_stale_partial_file_before_yielding(self):
        with ballroom_scribe.utils.atomic_path(_FINAL_PATH):
            self.mock_unlink.assert_called_once_with(_PARTIAL_PATH, missing_ok=True)
            self.mock_replace.assert_not_called()

    def test_renames_partial_file_to_final_path_on_success(self):
        with ballroom_scribe.utils.atomic_path(_FINAL_PATH):
            pass

        self.mock_replace.assert_called_once_with(_PARTIAL_PATH, _FINAL_PATH)
        self.mock_unlink.assert_called_once_with(_PARTIAL_PATH, missing_ok=True)

    @parameterized.expand(
        [
            param("exception", error=RuntimeError("PANN inference failed")),
            param("keyboard_interrupt", error=KeyboardInterrupt()),
        ]
    )
    def test_deletes_partial_file_and_reraises_on_failure(self, _, error):
        with self.assertRaises(type(error)) as context, ballroom_scribe.utils.atomic_path(_FINAL_PATH):
            raise error

        self.assertIs(context.exception, error)
        self.mock_replace.assert_not_called()
        self.assertEqual(self.mock_unlink.call_args_list, [call(_PARTIAL_PATH, missing_ok=True)] * 2)

    def test_reraises_original_exception_when_cleanup_fails(self):
        self.mock_unlink.side_effect = [None, PermissionError("The file is in use")]
        error = RuntimeError("FFmpeg failed")

        with self.assertRaises(RuntimeError) as context, ballroom_scribe.utils.atomic_path(_FINAL_PATH):
            raise error

        self.assertIs(context.exception, error)
        self.mock_replace.assert_not_called()

    def test_keeps_partial_file_when_rename_fails(self):
        self.mock_replace.side_effect = FileNotFoundError("The partial file was never created")

        with self.assertRaises(FileNotFoundError), ballroom_scribe.utils.atomic_path(_FINAL_PATH):
            pass

        self.mock_unlink.assert_called_once_with(_PARTIAL_PATH, missing_ok=True)


if __name__ == "__main__":
    unittest.main()
