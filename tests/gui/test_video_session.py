import pathlib
import unittest
from unittest.mock import patch

from parameterized import param, parameterized

import ballroom_scribe.cli.arguments
import ballroom_scribe.gui
import ballroom_scribe.gui.models
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


class TestOfflineVideos(unittest.TestCase):
    """Unit tests for `VideoSession`'s offline entries: `add_offline`, `replace`, `offline_match`, and `probe`."""

    def setUp(self):
        self.enterContext(patch.object(pathlib.Path, "resolve", autospec=True, side_effect=_fake_resolve))
        self.mock_count_channels = self.enterContext(
            patch.object(ballroom_scribe.utils, "count_audio_channels", autospec=True, return_value=2)
        )
        self.session = ballroom_scribe.gui.VideoSession()
        self.loaded = self.session.add(
            input_video=pathlib.Path("videos/part1.mp4"),
            input_channel=0,
            parent_cache_dir=pathlib.Path("videos"),
        )
        self.offline = self.session.add_offline(
            pathlib.Path("part2.mp4"),
            pathlib.Path("/comp/part2.mp4"),
            "File not found",
        )
        self.mock_count_channels.reset_mock()

    def test_add_offline_keeps_order_and_is_excluded_from_loaded_videos_and_inputs(self):
        self.assertIsInstance(self.offline, ballroom_scribe.gui.models.OfflineVideo)
        self.assertEqual(self.session.videos, (self.loaded, self.offline))
        self.assertEqual(self.session.loaded_videos, (self.loaded,))
        self.assertEqual(len(self.session), 2)
        self.assertEqual(self.session.get(pathlib.Path("part2.mp4")), self.offline)
        self.assertEqual([entry["input_video"] for entry in self.session.inputs()], [self.loaded.input_video])
        self.mock_count_channels.assert_not_called()  # Offline entries never probe the file.

        # Marking the same path offline again just updates the reason; a loaded key is refused.
        updated = self.session.add_offline(pathlib.Path("part2.mp4"), pathlib.Path("/comp/part2.mp4"), "No audio")
        self.assertEqual(self.session.videos, (self.loaded, updated))
        with self.assertRaisesRegex(ValueError, "cannot be marked offline"):
            self.session.add_offline(self.loaded.csv_file, self.loaded.input_video, "File not found")

    def test_add_replaces_an_offline_entry_with_the_same_key_in_place(self):
        relinked = self.session.add(
            input_video=pathlib.Path("/comp/part2.mp4"),
            input_channel=1,
            parent_cache_dir=pathlib.Path("/comp"),
            csv_file=pathlib.Path("part2.mp4"),
        )

        self.assertEqual(self.session.videos, (self.loaded, relinked))
        self.assertEqual(self.session.loaded_videos, (self.loaded, relinked))
        with self.assertRaisesRegex(ValueError, "already loaded"):
            self.session.add(
                input_video=pathlib.Path("/comp/part2.mp4"),
                input_channel=1,
                parent_cache_dir=pathlib.Path("/comp"),
            )

    def test_probe_validates_without_adding(self):
        probed = self.session.probe(
            pathlib.Path("/found/part2.mp4"),
            1,
            pathlib.Path("/found"),
            csv_file=pathlib.Path("found/part2.mp4"),
        )

        self.assertEqual(probed.csv_file, pathlib.Path("found/part2.mp4"))
        self.assertEqual(probed.input_video, pathlib.Path("/found/part2.mp4"))
        self.assertEqual(probed.channel_count, 2)
        self.assertEqual(self.session.videos, (self.loaded, self.offline))
        with self.assertRaisesRegex(ValueError, "already loaded"):
            self.session.probe(self.loaded.input_video, 0, pathlib.Path("elsewhere"))

    def test_replace_swaps_entries_keeping_the_position_in_both_directions(self):
        later = self.session.add_offline(pathlib.Path("part3.mp4"), pathlib.Path("/comp/part3.mp4"), "File not found")
        found = self.session.probe(
            pathlib.Path("/found/part2.mp4"),
            0,
            pathlib.Path("/found"),
            csv_file=pathlib.Path("found/part2.mp4"),
        )

        self.session.replace(self.offline.csv_file, found)
        self.assertEqual(self.session.videos, (self.loaded, found, later))
        self.assertIsNone(self.session.get(pathlib.Path("part2.mp4")))
        self.assertEqual(self.session.get(pathlib.Path("found/part2.mp4")), found)

        self.session.replace(found.csv_file, self.offline)
        self.assertEqual(self.session.videos, (self.loaded, self.offline, later))

        with self.assertRaises(KeyError):
            self.session.replace(pathlib.Path("missing.mp4"), found)
        with self.assertRaisesRegex(ValueError, "already in the session"):
            self.session.replace(later.csv_file, self.offline)

    def test_offline_match_requires_exactly_one_offline_entry_with_that_name(self):
        self.assertEqual(self.session.offline_match(pathlib.Path("/backup/part2.mp4")), self.offline)
        self.assertIsNone(self.session.offline_match(pathlib.Path("/backup/part1.mp4")))  # Loaded, not offline.
        self.assertIsNone(self.session.offline_match(pathlib.Path("/backup/part9.mp4")))

        self.session.add_offline(pathlib.Path("other/part2.mp4"), pathlib.Path("/other/part2.mp4"), "File not found")
        self.assertIsNone(self.session.offline_match(pathlib.Path("/backup/part2.mp4")))  # Ambiguous.

    def test_channel_and_cache_dir_changes_reject_offline_entries(self):
        with self.assertRaisesRegex(ValueError, "offline"):
            self.session.set_channel(self.offline.csv_file, 0)
        with self.assertRaisesRegex(ValueError, "offline"):
            self.session.set_parent_cache_dir(self.offline.csv_file, pathlib.Path("alt_cache"))

    def test_find_for_record_resolves_rows_to_offline_entries(self):
        self.assertEqual(self.session.find_for_record(pathlib.Path("part2.mp4")), self.offline)
        self.assertEqual(self.session.find_for_record(pathlib.Path("/elsewhere/part2.mp4")), self.offline)
        self.assertEqual(
            self.session.find_for_record(pathlib.Path("part2.mp4"), csv_dir=pathlib.Path("/comp")), self.offline
        )
        self.assertIsNone(self.session.find_for_record(pathlib.Path("part9.mp4")))


class TestReorderAndInsert(unittest.TestCase):
    """Unit tests for `VideoSession.reorder` and `VideoSession.insert`."""

    def setUp(self):
        self.enterContext(patch.object(pathlib.Path, "resolve", autospec=True, side_effect=_fake_resolve))
        self.enterContext(patch.object(ballroom_scribe.utils, "count_audio_channels", autospec=True, return_value=2))
        self.session = ballroom_scribe.gui.VideoSession()
        self.first = self.session.add(
            input_video=pathlib.Path("videos/part1.mp4"),
            input_channel=0,
            parent_cache_dir=pathlib.Path("videos"),
        )
        self.offline = self.session.add_offline(
            pathlib.Path("part2.mp4"),
            pathlib.Path("/comp/part2.mp4"),
            "File not found",
        )
        self.third = self.session.add(
            input_video=pathlib.Path("videos/part3.mp4"),
            input_channel=1,
            parent_cache_dir=pathlib.Path("videos"),
        )

    def test_reorder_applies_a_full_order_and_inputs_follow(self):
        self.session.reorder((self.third.csv_file, self.offline.csv_file, self.first.csv_file))

        self.assertEqual(self.session.videos, (self.third, self.offline, self.first))
        self.assertEqual(
            [entry["input_video"] for entry in self.session.inputs()],
            [self.third.input_video, self.first.input_video],
        )

    @parameterized.expand(
        [
            param("listed_videos_come_first_and_the_rest_keep_their_order", order=("part3",), expected=(2, 0, 1)),
            param("unknown_paths_are_ignored", order=("missing", "part2", "part1"), expected=(1, 0, 2)),
            param("duplicates_count_once", order=("part2", "part2", "part1"), expected=(1, 0, 2)),
            param("empty_order_changes_nothing", order=(), expected=(0, 1, 2)),
        ]
    )
    def test_reorder_tolerates_partial_orders(self, _, order: tuple[str, ...], expected: tuple[int, ...]):
        videos = self.session.videos
        keys = {"part1": self.first.csv_file, "part2": self.offline.csv_file, "part3": self.third.csv_file}

        self.session.reorder(tuple(keys.get(name, pathlib.Path("missing.mp4")) for name in order))

        self.assertEqual(self.session.videos, tuple(videos[index] for index in expected))

    @parameterized.expand(
        [
            param("at_its_old_position", index=1, expected_position=1),
            param("at_the_front", index=0, expected_position=0),
            param("at_the_end", index=2, expected_position=2),
            param("past_the_end_is_clamped", index=99, expected_position=2),
            param("negative_is_clamped_to_the_front", index=-5, expected_position=0),
        ]
    )
    def test_insert_puts_a_removed_video_back(self, _, index: int, expected_position: int):
        self.session.remove(self.offline.csv_file)
        self.assertEqual(self.session.videos, (self.first, self.third))

        self.session.insert(index, self.offline)

        self.assertEqual(self.session.videos[expected_position], self.offline)
        self.assertEqual(len(self.session), 3)
        self.assertEqual(self.session.get(self.offline.csv_file), self.offline)

    def test_insert_rejects_a_video_already_in_the_session(self):
        with self.assertRaisesRegex(ValueError, "already in the session"):
            self.session.insert(0, self.offline)

        self.assertEqual(self.session.videos, (self.first, self.offline, self.third))


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
