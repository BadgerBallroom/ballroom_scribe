import pathlib
import unittest
import urllib.error
import urllib.request
from unittest.mock import ANY, MagicMock, call, mock_open, patch

from parameterized import param, parameterized

import ballroom_scribe.utils

_URL = "https://example.com/model.pth"
_DESTINATION = pathlib.Path("fake_home/panns_data/model.pth")


class TestDownloadIfMissing(unittest.TestCase):
    """Unit tests for the `download_if_missing` utility function."""

    def setUp(self):
        self.mock_is_file = self.enterContext(patch.object(pathlib.Path, "is_file", autospec=True))
        self.mock_is_file.return_value = False

        self.mock_stat = self.enterContext(patch.object(pathlib.Path, "stat", autospec=True))
        self.mock_mkdir = self.enterContext(patch.object(pathlib.Path, "mkdir", autospec=True))

        self.mock_atomic_path = self.enterContext(patch.object(ballroom_scribe.utils, "atomic_path", autospec=True))
        self.mock_partial_path = MagicMock(spec=pathlib.Path)
        self.mock_atomic_path.return_value.__enter__.return_value = self.mock_partial_path
        self.mock_atomic_path.return_value.__exit__.return_value = False

        self.mock_urlopen = self.enterContext(patch.object(urllib.request, "urlopen", autospec=True))
        self.mock_urlopen.return_value.__exit__.return_value = False
        self.mock_response = self.mock_urlopen.return_value.__enter__.return_value

        self.mock_open = mock_open()
        self.enterContext(patch("builtins.open", self.mock_open))

    def test_rejects_negative_min_bytes(self):
        with self.assertRaisesRegex(ValueError, "min_bytes must be non-negative, got -1."):
            ballroom_scribe.utils.download_if_missing(_URL, _DESTINATION, min_bytes=-1)

        self.mock_is_file.assert_not_called()
        self.mock_mkdir.assert_not_called()
        self.mock_atomic_path.assert_not_called()
        self.mock_urlopen.assert_not_called()

    @parameterized.expand(
        [
            param("default_min_bytes", st_size=1, kwargs={}),
            param("custom_min_bytes_exact", st_size=500, kwargs={"min_bytes": 500}),
            param("zero_min_bytes_empty_file", st_size=0, kwargs={"min_bytes": 0}),
        ]
    )
    def test_skips_download_when_valid_file_already_exists(self, _, st_size, kwargs):
        self.mock_is_file.return_value = True
        self.mock_stat.return_value.st_size = st_size

        ballroom_scribe.utils.download_if_missing(_URL, _DESTINATION, **kwargs)

        self.mock_is_file.assert_called_once_with(_DESTINATION)
        self.mock_stat.assert_called_once_with(_DESTINATION)
        self.mock_mkdir.assert_not_called()
        self.mock_atomic_path.assert_not_called()
        self.mock_urlopen.assert_not_called()
        self.mock_open.assert_not_called()

    @parameterized.expand(
        [
            param(
                "missing_file",
                is_file=False,
                st_size=0,
                kwargs={},
                headers={"Content-Length": "11"},
                chunks=[b"hello ", b"world", b""],
            ),
            param(
                "empty_existing_file",
                is_file=True,
                st_size=0,
                kwargs={},
                headers={},
                chunks=[b"hello ", b"world", b""],
            ),
            param(
                "existing_file_below_min_bytes",
                is_file=True,
                st_size=99,
                kwargs={"min_bytes": 100},
                headers={"Content-Length": "100"},
                chunks=[b"x" * 60, b"y" * 40, b""],
            ),
        ]
    )
    def test_downloads_atomically_when_missing_or_too_small(self, _, is_file, st_size, kwargs, headers, chunks):
        self.mock_is_file.return_value = is_file
        self.mock_stat.return_value.st_size = st_size
        self.mock_response.headers = headers
        self.mock_response.read.side_effect = chunks

        ballroom_scribe.utils.download_if_missing(_URL, _DESTINATION, **kwargs)

        self.mock_mkdir.assert_called_once_with(_DESTINATION.parent, parents=True, exist_ok=True)
        self.mock_atomic_path.assert_called_once_with(_DESTINATION)
        self.mock_urlopen.assert_called_once_with(_URL)
        self.mock_open.assert_called_once_with(self.mock_partial_path, "wb")
        self.assertEqual(
            self.mock_open().write.call_args_list,
            [call(chunk) for chunk in chunks if chunk],
        )
        self.mock_atomic_path.return_value.__exit__.assert_called_once_with(None, None, None)

    @parameterized.expand(
        [
            param(
                "truncated_relative_to_content_length",
                kwargs={},
                headers={"Content-Length": "20"},
                chunks=[b"12345", b""],
                expected_regex="truncated: got 5 of 20 bytes",
            ),
            param(
                "smaller_than_min_bytes",
                kwargs={"min_bytes": 50},
                headers={},
                chunks=[b"12345", b""],
                expected_regex=r"unexpectedly small \(5 bytes; expected at least 50\)",
            ),
        ]
    )
    def test_raises_os_error_on_incomplete_or_undersized_payload(self, _, kwargs, headers, chunks, expected_regex):
        self.mock_response.headers = headers
        self.mock_response.read.side_effect = chunks

        with self.assertRaisesRegex(OSError, expected_regex):
            ballroom_scribe.utils.download_if_missing(_URL, _DESTINATION, **kwargs)

        self.mock_atomic_path.return_value.__exit__.assert_called_once_with(OSError, ANY, ANY)

    def test_propagates_network_error_to_atomic_path(self):
        self.mock_urlopen.side_effect = urllib.error.URLError("offline")

        with self.assertRaises(urllib.error.URLError):
            ballroom_scribe.utils.download_if_missing(_URL, _DESTINATION)

        self.mock_open.assert_not_called()
        self.mock_atomic_path.return_value.__exit__.assert_called_once_with(urllib.error.URLError, ANY, ANY)


if __name__ == "__main__":
    unittest.main()
