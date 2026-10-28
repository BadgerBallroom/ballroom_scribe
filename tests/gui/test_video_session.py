import pathlib
import unittest
from unittest.mock import patch

from parameterized import param, parameterized

import ballroom_scribe.cli.arguments
import ballroom_scribe.gui
import ballroom_scribe.utils


def _fake_resolve(path: pathlib.Path) -> pathlib.Path:
    if path.is_absolute():
        return path
    return pathlib.Path("/workspace") / path


class TestVideoSession(unittest.TestCase):
    """Unit tests for `VideoSession`."""

    def setUp(self):
        self.enterContext(patch.object(pathlib.Path, "resolve", autospec=True, side_effect=_fake_resolve))
        self.mock_count_channels = self.enterContext(
            patch.object(ballroom_scribe.utils, "count_audio_channels", autospec=True, return_value=2)
        )

    def test_add_populates_loaded_video_and_inputs(self):
        session = ballroom_scribe.gui.VideoSession()

        loaded = session.add(
            input_video=pathlib.Path("videos/session1.mp4"),
            input_channel=1,
            parent_cache_dir=pathlib.Path("videos"),
        )

        self.assertEqual(loaded.input_video, pathlib.Path("/workspace/videos/session1.mp4"))
        self.assertEqual(loaded.csv_file, pathlib.Path("/workspace/videos/session1.mp4"))
        self.assertEqual(loaded.cache_dir, pathlib.Path("videos/session1-Ballroom_Cache"))
        self.assertEqual(loaded.channel_count, 2)
        self.assertEqual(len(session), 1)
        self.assertTrue(bool(session))
        self.assertEqual(
            session.inputs(),
            [
                ballroom_scribe.cli.arguments.InputCacheArgs(
                    input_video=pathlib.Path("/workspace/videos/session1.mp4"),
                    input_channel=1,
                    cache_dir=pathlib.Path("videos"),
                )
            ],
        )

    def test_rejects_duplicate_video_and_cache_dir_collision(self):
        session = ballroom_scribe.gui.VideoSession()
        session.add(
            input_video=pathlib.Path("dir_a/clip.mp4"),
            input_channel=0,
            parent_cache_dir=pathlib.Path("shared_cache"),
        )

        with self.assertRaisesRegex(ValueError, "already loaded"):
            session.add(
                input_video=pathlib.Path("/workspace/dir_a/clip.mp4"),
                input_channel=0,
                parent_cache_dir=pathlib.Path("other_cache"),
            )

        with self.assertRaisesRegex(ValueError, "share the cache directory"):
            session.add(
                input_video=pathlib.Path("dir_b/clip.mp4"),
                input_channel=0,
                parent_cache_dir=pathlib.Path("shared_cache"),
            )

    @parameterized.expand(
        [
            param("no_audio_stream", channel_count=0, input_channel=0, expected_regex="no audio stream"),
            param("channel_out_of_range", channel_count=2, input_channel=2, expected_regex="out of range"),
            param("negative_channel", channel_count=2, input_channel=-1, expected_regex="out of range"),
        ]
    )
    def test_rejects_invalid_audio_channel(self, _, channel_count: int, input_channel: int, expected_regex: str):
        self.mock_count_channels.return_value = channel_count
        session = ballroom_scribe.gui.VideoSession()

        with self.assertRaisesRegex(ValueError, expected_regex):
            session.add(
                input_video=pathlib.Path("videos/clip.mp4"),
                input_channel=input_channel,
                parent_cache_dir=pathlib.Path("videos"),
            )

    def test_set_channel_and_set_parent_cache_dir(self):
        session = ballroom_scribe.gui.VideoSession()
        loaded = session.add(
            input_video=pathlib.Path("videos/clip.mp4"),
            input_channel=0,
            parent_cache_dir=pathlib.Path("videos"),
        )

        updated_ch = session.set_channel(loaded.csv_file, 1)
        self.assertEqual(updated_ch.input_channel, 1)

        with self.assertRaises(ValueError):
            session.set_channel(loaded.csv_file, 5)

        updated_cache = session.set_parent_cache_dir(loaded.csv_file, pathlib.Path("alt_cache"))
        self.assertEqual(updated_cache.parent_cache_dir, pathlib.Path("alt_cache"))
        self.assertEqual(updated_cache.cache_dir, pathlib.Path("alt_cache/clip-Ballroom_Cache"))

    def test_find_for_record_matches_exact_relative_and_unique_filename(self):
        session = ballroom_scribe.gui.VideoSession()
        video_a = session.add(
            input_video=pathlib.Path("/media/day1/heat1.mp4"),
            input_channel=0,
            parent_cache_dir=pathlib.Path("/media/day1"),
            csv_file=pathlib.Path("raw/heat1.mp4"),
        )
        video_b = session.add(
            input_video=pathlib.Path("/media/day1/sub/heat2.mp4"),
            input_channel=0,
            parent_cache_dir=pathlib.Path("/media/day1/sub"),
        )

        # Exact `csv_file` match
        self.assertEqual(session.find_for_record(pathlib.Path("raw/heat1.mp4")), video_a)
        # Relative to `csv_dir`
        self.assertEqual(
            session.find_for_record(pathlib.Path("sub/heat2.mp4"), csv_dir=pathlib.Path("/media/day1")),
            video_b,
        )
        # Unique filename fallback
        self.assertEqual(session.find_for_record(pathlib.Path("/moved/elsewhere/heat1.mp4")), video_a)
        # Unknown file
        self.assertIsNone(session.find_for_record(pathlib.Path("missing.mp4")))

        session.remove(video_a.csv_file)
        self.assertEqual(len(session), 1)
        session.clear()
        self.assertEqual(len(session), 0)


class TestInferChannel(unittest.TestCase):
    """Unit tests for `VideoSession.infer_channel`."""

    def setUp(self):
        self.mock_exists = self.enterContext(patch.object(pathlib.Path, "exists", autospec=True))
        self.mock_glob = self.enterContext(patch.object(pathlib.Path, "glob", autospec=True))

    @parameterized.expand(
        [
            param("cache_dir_missing", exists=False, wav_names=[], expected=0),
            param("no_cached_wav", exists=True, wav_names=[], expected=0),
            param("single_cached_channel", exists=True, wav_names=["Audio-Channel-3.wav"], expected=3),
            param(
                "multiple_cached_channels",
                exists=True,
                wav_names=["Audio-Channel-0.wav", "Audio-Channel-1.wav"],
                expected=0,
            ),
        ]
    )
    def test_infer_channel(self, _, exists: bool, wav_names: list[str], expected: int):
        self.mock_exists.return_value = exists
        self.mock_glob.return_value = iter([pathlib.Path(name) for name in wav_names])

        channel = ballroom_scribe.gui.VideoSession.infer_channel(
            input_video=pathlib.Path("clip.mp4"),
            parent_cache_dir=pathlib.Path("."),
        )

        self.assertEqual(channel, expected)


if __name__ == "__main__":
    unittest.main()
