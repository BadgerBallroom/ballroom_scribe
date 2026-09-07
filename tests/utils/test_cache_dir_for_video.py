import pathlib
import unittest

import ballroom_scribe.utils


class TestCacheDirForVideo(unittest.TestCase):
    """Unit tests for the `cache_dir_for_video` function."""

    def test_returns_video_cache_subdirectory(self):
        result = ballroom_scribe.utils.cache_dir_for_video(
            parent_cache_dir=pathlib.Path("path/to/cache"),
            input_video=pathlib.Path("other/dir/competition.mp4"),
        )
        self.assertEqual(result, pathlib.Path("path/to/cache/competition-Ballroom_Cache"))


if __name__ == "__main__":
    unittest.main()
