import unittest

import numpy as np
from parameterized import param, parameterized

import ballroom_scribe
import ballroom_scribe.models
import ballroom_scribe.music_valley_detector

Classification = ballroom_scribe.models.AudioChunk.Classification


def _make_chunk(
    start_seconds: float,
    duration_seconds: float,
    classification: Classification,
    sample_rate: int = 10,
) -> ballroom_scribe.models.AudioChunk:
    """Helper to create an AudioChunk with sequential float32 data."""
    start_samples = round(start_seconds * sample_rate)
    num_samples = round(duration_seconds * sample_rate)
    return ballroom_scribe.models.AudioChunk(
        data=np.arange(start_samples, start_samples + num_samples, dtype=np.float32),
        sample_rate=sample_rate,
        classification=classification,
        start_samples=start_samples,
    )


class TestExtractSpeechChunks(unittest.TestCase):
    """Unit tests for `_extract_speech_chunks`."""

    @parameterized.expand(
        [
            param(
                "empty",
                chunks=[],
                expected=[],
            ),
            param(
                "filters_non_speech_and_normalizes_speech_music",
                chunks=[
                    _make_chunk(0.0, 2.0, Classification.NONE),
                    _make_chunk(2.0, 3.0, Classification.MUSIC),
                    _make_chunk(5.0, 4.0, Classification.SPEECH),
                    _make_chunk(9.0, 2.0, Classification.SPEECH | Classification.MUSIC),
                ],
                expected=[
                    _make_chunk(5.0, 4.0, Classification.SPEECH),
                    _make_chunk(9.0, 2.0, Classification.SPEECH),
                ],
            ),
        ]
    )
    def test_extract_speech_chunks(self, _, chunks, expected):
        result = list(ballroom_scribe.music_valley_detector._extract_speech_chunks(chunks))
        self.assertEqual(result, expected)


class TestMergeAdjacentChunks(unittest.TestCase):
    """Unit tests for `_merge_adjacent_chunks`."""

    @parameterized.expand(
        [
            param(
                "empty",
                chunks=[],
                expected=[],
            ),
            param(
                "single_chunk",
                chunks=[_make_chunk(0.0, 2.0, Classification.SPEECH)],
                expected=[_make_chunk(0.0, 2.0, Classification.SPEECH)],
            ),
            param(
                "merges_contiguous_and_preserves_gaps",
                chunks=[
                    _make_chunk(0.0, 2.0, Classification.SPEECH),
                    _make_chunk(2.0, 3.0, Classification.SPEECH),
                    _make_chunk(10.0, 4.0, Classification.SPEECH),
                ],
                expected=[
                    _make_chunk(0.0, 5.0, Classification.SPEECH),
                    _make_chunk(10.0, 4.0, Classification.SPEECH),
                ],
            ),
        ]
    )
    def test_merge_adjacent_chunks(self, _, chunks, expected):
        result = list(ballroom_scribe.music_valley_detector._merge_adjacent_chunks(chunks))
        self.assertEqual(result, expected)


class TestFinalizeValleySpeech(unittest.TestCase):
    """Unit tests for `_finalize_valley_speech`."""

    def test_finalize_valley_speech_extracts_merges_and_detaches(self):
        parent = _make_chunk(0.0, 20.0, Classification.SPEECH)
        part1, remainder = parent.split(new_start_samples=50)  # 0.0 - 5.0s
        part2, remainder = remainder.split(new_start_samples=80)  # 5.0 - 8.0s
        _, part3 = remainder.split(new_start_samples=120)  # 12.0 - 20.0s

        chunks = [
            _make_chunk(0.0, 2.0, Classification.NONE),
            part1,
            part2,
            _make_chunk(8.0, 4.0, Classification.MUSIC),
            part3,
        ]

        result = ballroom_scribe.music_valley_detector._finalize_valley_speech(chunks)

        expected = (
            _make_chunk(0.0, 8.0, Classification.SPEECH),
            _make_chunk(12.0, 8.0, Classification.SPEECH),
        )
        self.assertEqual(result, expected)
        for chunk in result:
            self.assertIsNone(chunk.data.base)


class TestMusicValleyDetector(unittest.TestCase):
    """End-to-end unit tests for `MusicValleyDetector`."""

    @parameterized.expand(
        [
            param(
                "empty_stream",
                chunks=[],
                expected=[],
            ),
            param(
                "two_dances_with_fades_dip_and_bumper_music",
                chunks=[
                    # Valley 0: intro speech (0-5s) + fade-in speech on Dance 1 (5-10s)
                    _make_chunk(0.0, 5.0, Classification.SPEECH),
                    _make_chunk(5.0, 5.0, Classification.SPEECH | Classification.MUSIC),
                    # Dance 1 core: 10-30s music + 2s NONE dip (30-32s) + 18s music (32-50s)
                    _make_chunk(10.0, 20.0, Classification.MUSIC),
                    _make_chunk(30.0, 2.0, Classification.NONE),
                    _make_chunk(32.0, 18.0, Classification.MUSIC),
                    # Dance 1 fade-out speech (50-54s)
                    _make_chunk(50.0, 4.0, Classification.SPEECH | Classification.MUSIC),
                    # Valley 1: announcement speech (54-60s) + short bumper music (60-75s) + silence (75-80s)
                    _make_chunk(54.0, 6.0, Classification.SPEECH),
                    _make_chunk(60.0, 15.0, Classification.MUSIC),
                    _make_chunk(75.0, 5.0, Classification.NONE),
                    # Dance 2: pure music (80-120s)
                    _make_chunk(80.0, 40.0, Classification.MUSIC),
                    # Valley 2 (trailing): outro speech (120-125s)
                    _make_chunk(120.0, 5.0, Classification.SPEECH),
                ],
                expected=[
                    ballroom_scribe.models.MusicValley(
                        start_samples=0,
                        sample_rate=10,
                        duration_samples=100,
                        speech_chunks=(_make_chunk(0.0, 10.0, Classification.SPEECH),),
                    ),
                    ballroom_scribe.models.MusicValley(
                        start_samples=500,
                        sample_rate=10,
                        duration_samples=300,
                        speech_chunks=(_make_chunk(50.0, 10.0, Classification.SPEECH),),
                    ),
                    ballroom_scribe.models.MusicValley(
                        start_samples=1200,
                        sample_rate=10,
                        duration_samples=50,
                        speech_chunks=(_make_chunk(120.0, 5.0, Classification.SPEECH),),
                    ),
                ],
            ),
            param(
                "chained_short_valley_chunks_do_not_merge_separate_dances",
                chunks=[
                    # Dance 1 (0-40s)
                    _make_chunk(0.0, 40.0, Classification.MUSIC),
                    # Valley of 3 short chunks totaling 8.3s: 3.0s NONE + 1.8s SPEECH + 3.5s NONE
                    _make_chunk(40.0, 3.0, Classification.NONE),
                    _make_chunk(43.0, 1.8, Classification.SPEECH),
                    _make_chunk(44.8, 3.5, Classification.NONE),
                    # Dance 2 (48.3-88.3s)
                    _make_chunk(48.3, 40.0, Classification.MUSIC),
                ],
                expected=[
                    ballroom_scribe.models.MusicValley(
                        start_samples=400,
                        sample_rate=10,
                        duration_samples=83,
                        speech_chunks=(_make_chunk(43.0, 1.8, Classification.SPEECH),),
                    ),
                ],
            ),
            param(
                "fade_out_speech_followed_by_short_silence_is_not_trapped_in_dance",
                chunks=[
                    # Dance 1: 40s MUSIC + 1.5s fade-out SPEECH|MUSIC
                    _make_chunk(0.0, 40.0, Classification.MUSIC),
                    _make_chunk(40.0, 1.5, Classification.SPEECH | Classification.MUSIC),
                    # Valley: 2.5s NONE followed by 6.0s SPEECH
                    _make_chunk(41.5, 2.5, Classification.NONE),
                    _make_chunk(44.0, 6.0, Classification.SPEECH),
                ],
                expected=[
                    ballroom_scribe.models.MusicValley(
                        start_samples=400,
                        sample_rate=10,
                        duration_samples=100,
                        speech_chunks=(
                            _make_chunk(40.0, 1.5, Classification.SPEECH),
                            _make_chunk(44.0, 6.0, Classification.SPEECH),
                        ),
                    ),
                ],
            ),
            param(
                "stream_ending_with_pure_dance_and_no_trailing_fade",
                chunks=[
                    _make_chunk(0.0, 4.0, Classification.SPEECH),
                    _make_chunk(4.0, 40.0, Classification.MUSIC),
                ],
                expected=[
                    ballroom_scribe.models.MusicValley(
                        start_samples=0,
                        sample_rate=10,
                        duration_samples=40,
                        speech_chunks=(_make_chunk(0.0, 4.0, Classification.SPEECH),),
                    ),
                ],
            ),
            param(
                "stream_ending_with_dance_having_trailing_fade_speech",
                chunks=[
                    _make_chunk(0.0, 40.0, Classification.MUSIC),
                    _make_chunk(40.0, 5.0, Classification.SPEECH | Classification.MUSIC),
                ],
                expected=[
                    ballroom_scribe.models.MusicValley(
                        start_samples=400,
                        sample_rate=10,
                        duration_samples=50,
                        speech_chunks=(_make_chunk(40.0, 5.0, Classification.SPEECH),),
                    ),
                ],
            ),
            param(
                "stream_ending_with_dance_followed_by_short_pending_dip_none_chunk",
                chunks=[
                    _make_chunk(0.0, 40.0, Classification.MUSIC),
                    _make_chunk(40.0, 2.0, Classification.NONE),
                ],
                expected=[
                    ballroom_scribe.models.MusicValley(
                        start_samples=400,
                        sample_rate=10,
                        duration_samples=20,
                        speech_chunks=(),
                    ),
                ],
            ),
        ]
    )
    def test_detect(self, _, chunks, expected):
        detector = ballroom_scribe.MusicValleyDetector(
            min_dance_seconds=35.0,
            max_song_dip_seconds=4.5,
            max_fade_speech_seconds=12.0,
            max_dip_speech_seconds=2.0,
        )
        valleys = list(detector.detect(chunks))
        self.assertEqual(valleys, expected)


class TestSongDipAndDancePredicates(unittest.TestCase):
    """Unit tests for `_is_valid_pending_dip` and `_is_dance_run`."""

    @parameterized.expand(
        [
            param(
                "short_none_is_valid_dip",
                dip_chunks=[_make_chunk(0.0, 4.5, Classification.NONE)],
                expected=True,
            ),
            param(
                "long_none_exceeds_max_song_dip_seconds",
                dip_chunks=[_make_chunk(0.0, 4.6, Classification.NONE)],
                expected=False,
            ),
            param(
                "multiple_chunks_exceeding_cumulative_max_song_dip_seconds",
                dip_chunks=[
                    _make_chunk(0.0, 2.5, Classification.NONE),
                    _make_chunk(2.5, 2.5, Classification.NONE),
                ],
                expected=False,
            ),
            param(
                "brief_speech_under_2s_is_valid_dip",
                dip_chunks=[_make_chunk(0.0, 1.9, Classification.SPEECH)],
                expected=True,
            ),
            param(
                "speech_at_or_above_2s_is_invalid_dip",
                dip_chunks=[_make_chunk(0.0, 2.0, Classification.SPEECH)],
                expected=False,
            ),
            param(
                "multiple_speech_chunks_exceeding_cumulative_2s_is_invalid_dip",
                dip_chunks=[
                    _make_chunk(0.0, 1.2, Classification.SPEECH),
                    _make_chunk(1.2, 1.0, Classification.NONE),
                    _make_chunk(2.2, 1.0, Classification.SPEECH),
                ],
                expected=False,
            ),
        ]
    )
    def test_is_valid_pending_dip(self, _, dip_chunks, expected):
        detector = ballroom_scribe.MusicValleyDetector(
            min_dance_seconds=35.0,
            max_song_dip_seconds=4.5,
            max_fade_speech_seconds=12.0,
            max_dip_speech_seconds=2.0,
        )
        self.assertEqual(
            detector._is_midsong_dip(dip_chunks),
            expected,
        )

    @parameterized.expand(
        [
            param(
                "below_min_dance_seconds",
                chunks=[_make_chunk(0.0, 20.0, Classification.MUSIC), _make_chunk(20.0, 14.9, Classification.MUSIC)],
                expected=False,
            ),
            param(
                "at_min_dance_seconds",
                chunks=[_make_chunk(0.0, 20.0, Classification.MUSIC), _make_chunk(20.0, 15.0, Classification.MUSIC)],
                expected=True,
            ),
        ]
    )
    def test_is_dance_run(self, _, chunks, expected):
        detector = ballroom_scribe.MusicValleyDetector(
            min_dance_seconds=35.0,
            max_song_dip_seconds=4.5,
            max_fade_speech_seconds=12.0,
            max_dip_speech_seconds=2.0,
        )
        self.assertEqual(
            detector._is_dance_run(chunks),
            expected,
        )


class TestPeelFadeEdges(unittest.TestCase):
    """Unit tests for `_count_leading_fade_chunks` and `_peel_fade_edges`."""

    @parameterized.expand(
        [
            param(
                "pure_music_dance",
                dance_chunks=[
                    _make_chunk(10.0, 40.0, Classification.MUSIC),
                ],
                max_fade_speech_seconds=12.0,
                expected=ballroom_scribe.music_valley_detector._PeeledDance(
                    leading_speech=(),
                    core_start_samples=100,
                    core_end_samples=500,
                    sample_rate=10,
                    trailing_speech=(),
                ),
            ),
            param(
                "peels_leading_and_trailing_fade_speech_within_budget",
                dance_chunks=[
                    _make_chunk(10.0, 5.0, Classification.SPEECH | Classification.MUSIC),
                    _make_chunk(15.0, 4.0, Classification.SPEECH | Classification.MUSIC),
                    _make_chunk(19.0, 40.0, Classification.MUSIC),
                    _make_chunk(59.0, 6.0, Classification.SPEECH | Classification.MUSIC),
                ],
                max_fade_speech_seconds=12.0,
                expected=ballroom_scribe.music_valley_detector._PeeledDance(
                    leading_speech=(
                        _make_chunk(10.0, 5.0, Classification.SPEECH | Classification.MUSIC),
                        _make_chunk(15.0, 4.0, Classification.SPEECH | Classification.MUSIC),
                    ),
                    core_start_samples=190,
                    core_end_samples=590,
                    sample_rate=10,
                    trailing_speech=(_make_chunk(59.0, 6.0, Classification.SPEECH | Classification.MUSIC),),
                ),
            ),
            param(
                "stops_peeling_when_exceeding_max_fade_speech_seconds",
                dance_chunks=[
                    _make_chunk(10.0, 8.0, Classification.SPEECH | Classification.MUSIC),
                    _make_chunk(18.0, 6.0, Classification.SPEECH | Classification.MUSIC),
                    _make_chunk(24.0, 30.0, Classification.MUSIC),
                    _make_chunk(54.0, 15.0, Classification.SPEECH | Classification.MUSIC),
                ],
                max_fade_speech_seconds=12.0,
                expected=ballroom_scribe.music_valley_detector._PeeledDance(
                    leading_speech=(_make_chunk(10.0, 8.0, Classification.SPEECH | Classification.MUSIC),),
                    core_start_samples=180,
                    core_end_samples=690,
                    sample_rate=10,
                    trailing_speech=(),
                ),
            ),
            param(
                "all_chunks_peeled_falls_back_to_outer_bounds",
                dance_chunks=[
                    _make_chunk(10.0, 5.0, Classification.SPEECH | Classification.MUSIC),
                    _make_chunk(15.0, 5.0, Classification.SPEECH | Classification.MUSIC),
                ],
                max_fade_speech_seconds=12.0,
                expected=ballroom_scribe.music_valley_detector._PeeledDance(
                    leading_speech=(
                        _make_chunk(10.0, 5.0, Classification.SPEECH | Classification.MUSIC),
                        _make_chunk(15.0, 5.0, Classification.SPEECH | Classification.MUSIC),
                    ),
                    core_start_samples=100,
                    core_end_samples=200,
                    sample_rate=10,
                    trailing_speech=(),
                ),
            ),
        ]
    )
    def test_peel_fade_edges(self, _, dance_chunks, max_fade_speech_seconds, expected):
        detector = ballroom_scribe.MusicValleyDetector(
            min_dance_seconds=35.0,
            max_song_dip_seconds=4.5,
            max_fade_speech_seconds=max_fade_speech_seconds,
            max_dip_speech_seconds=2.0,
        )
        result = detector._peel_fade_edges(dance_chunks)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
