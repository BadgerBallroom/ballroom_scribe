import collections
import pathlib
import unittest
from unittest.mock import ANY, MagicMock, call, patch

import numpy as np
from parameterized import param, parameterized

import ballroom_scribe
import ballroom_scribe.audio_classifier
import ballroom_scribe.models

Classification = ballroom_scribe.models.AudioChunk.Classification
_CLASSIFICATIONS = ballroom_scribe.PannInferencer.CLASSIFICATIONS

_THRESHOLDS = {
    ballroom_scribe.models.AudioChunk.Classification.SPEECH: 0.25,
    ballroom_scribe.models.AudioChunk.Classification.MUSIC: 0.20,
    ballroom_scribe.models.AudioChunk.Classification.NONE: 0.20,
}
_CHANNEL = 1
_CACHE_DIR = pathlib.Path("fake/path")


def _make_audio_chunk(
    start_samples: int,
    num_samples: int,
    sample_rate: int = 10,
    classification: Classification = Classification.UNKNOWN,
) -> ballroom_scribe.models.AudioChunk:
    """Helper to create an AudioChunk with sequential data."""
    return ballroom_scribe.models.AudioChunk(
        data=np.arange(start_samples, start_samples + num_samples, dtype=np.float32),
        sample_rate=sample_rate,
        classification=classification,
        start_samples=start_samples,
    )


class TestCutBeginning(unittest.TestCase):
    """Unit tests for the `_cut_beginning` function."""

    @parameterized.expand(
        [
            param(
                "empty_deque",
                initial=[],
                cutoff_samples=50,
                expected=[],
            ),
            param(
                "cutoff_within_single_chunk",
                initial=[_make_audio_chunk(start_samples=100, num_samples=10, classification=Classification.SPEECH)],
                cutoff_samples=104,
                expected=[_make_audio_chunk(start_samples=104, num_samples=6, classification=Classification.SPEECH)],
            ),
            param(
                "cutoff_drops_earlier_chunks",
                initial=[
                    _make_audio_chunk(start_samples=0, num_samples=10, classification=Classification.SPEECH),
                    _make_audio_chunk(start_samples=10, num_samples=10, classification=Classification.MUSIC),
                    _make_audio_chunk(start_samples=20, num_samples=10, classification=Classification.NONE),
                ],
                cutoff_samples=15,
                expected=[
                    _make_audio_chunk(start_samples=15, num_samples=5, classification=Classification.MUSIC),
                    _make_audio_chunk(start_samples=20, num_samples=10, classification=Classification.NONE),
                ],
            ),
            param(
                "cutoff_at_chunk_start",
                initial=[_make_audio_chunk(start_samples=100, num_samples=10, classification=Classification.SPEECH)],
                cutoff_samples=100,
                expected=[_make_audio_chunk(start_samples=100, num_samples=10, classification=Classification.SPEECH)],
            ),
        ]
    )
    def test_cut_beginning(self, _, initial, cutoff_samples, expected):
        chunks = collections.deque(initial)
        ballroom_scribe.audio_classifier._cut_beginning(chunks, cutoff_samples=cutoff_samples)
        self.assertEqual(chunks, collections.deque(expected))


class TestCutEnd(unittest.TestCase):
    """Unit tests for the `_cut_end` function."""

    @parameterized.expand(
        [
            param(
                "empty_deque",
                initial=[],
                cutoff_samples=50,
                expected=[],
            ),
            param(
                "cutoff_within_single_chunk",
                initial=[_make_audio_chunk(start_samples=100, num_samples=10, classification=Classification.SPEECH)],
                cutoff_samples=106,
                expected=[_make_audio_chunk(start_samples=100, num_samples=6, classification=Classification.SPEECH)],
            ),
            param(
                "cutoff_drops_later_chunks",
                initial=[
                    _make_audio_chunk(start_samples=0, num_samples=10, classification=Classification.SPEECH),
                    _make_audio_chunk(start_samples=10, num_samples=10, classification=Classification.MUSIC),
                    _make_audio_chunk(start_samples=20, num_samples=10, classification=Classification.NONE),
                ],
                cutoff_samples=15,
                expected=[
                    _make_audio_chunk(start_samples=0, num_samples=10, classification=Classification.SPEECH),
                    _make_audio_chunk(start_samples=10, num_samples=5, classification=Classification.MUSIC),
                ],
            ),
            param(
                "cutoff_at_chunk_end",
                initial=[_make_audio_chunk(start_samples=100, num_samples=10, classification=Classification.SPEECH)],
                cutoff_samples=110,
                expected=[_make_audio_chunk(start_samples=100, num_samples=10, classification=Classification.SPEECH)],
            ),
        ]
    )
    def test_cut_end(self, _, initial, cutoff_samples, expected):
        chunks = collections.deque(initial)
        ballroom_scribe.audio_classifier._cut_end(chunks, cutoff_samples=cutoff_samples)
        self.assertEqual(chunks, collections.deque(expected))


class TestInit(unittest.TestCase):
    """Unit tests for the `AudioClassifier.__init__` method."""

    def setUp(self):
        self.mock_pann_inferencer_cls = self.enterContext(
            patch.object(ballroom_scribe, "PannInferencer", autospec=True)
        )
        self.mock_pann_inferencer_cls.CLASSIFICATIONS = _CLASSIFICATIONS

    def test_raises_if_thresholds_is_missing_a_classification(self):
        with self.assertRaisesRegex(ValueError, "Thresholds must include a threshold for .*"):
            ballroom_scribe.AudioClassifier(device="cuda", thresholds={}, min_segment_seconds=1.0)

    def test_initializes_pann_inferencer(self):
        ballroom_scribe.AudioClassifier(device="fake_device", thresholds=_THRESHOLDS, min_segment_seconds=1.0)
        self.mock_pann_inferencer_cls.assert_called_once_with(device="fake_device")


class TestClassify(unittest.TestCase):
    """Unit tests for the `AudioClassifier.classify` method."""

    def setUp(self):
        self.mock_pann_inferencer_cls = self.enterContext(
            patch.object(ballroom_scribe, "PannInferencer", autospec=True)
        )
        self.mock_pann_inferencer_cls.CLASSIFICATIONS = _CLASSIFICATIONS

        self.mock_classify_chunk = self.enterContext(patch.object(ballroom_scribe.AudioClassifier, "_classify_chunk"))
        self.mock_cut_beginning = self.enterContext(patch.object(ballroom_scribe.audio_classifier, "_cut_beginning"))
        self.mock_cut_end = self.enterContext(patch.object(ballroom_scribe.audio_classifier, "_cut_end"))

        # `compute_overlap_cutover` is simple enough that it doesn't need to be mocked.

        self.classifier = ballroom_scribe.AudioClassifier(
            device="cuda",
            thresholds=_THRESHOLDS,
            min_segment_seconds=1.0,
        )

    def test_empty_chunks_iterable_yields_nothing(self):
        result = list(self.classifier.classify([], channel=_CHANNEL, cache_dir=_CACHE_DIR))

        self.assertEqual(result, [])
        self.mock_classify_chunk.assert_not_called()

    @parameterized.expand(
        [
            param(
                "single_chunk",
                input_chunks=[_make_audio_chunk(start_samples=0, num_samples=20, sample_rate=10)],
                classify_returns=[
                    [
                        _make_audio_chunk(
                            start_samples=0, num_samples=20, sample_rate=10, classification=Classification.SPEECH
                        )
                    ]
                ],
                trimmed_ends=[],
                trimmed_beginnings=[],
                expected_cut_calls=[],
                expected_result=[
                    _make_audio_chunk(
                        start_samples=0, num_samples=20, sample_rate=10, classification=Classification.SPEECH
                    )
                ],
            ),
            param(
                "skips_chunks_with_no_classifications",
                input_chunks=[
                    _make_audio_chunk(start_samples=0, num_samples=10, sample_rate=10),
                    _make_audio_chunk(start_samples=0, num_samples=20, sample_rate=10),
                ],
                classify_returns=[
                    [],  # First chunk returns no classifications
                    [
                        _make_audio_chunk(
                            start_samples=0, num_samples=20, sample_rate=10, classification=Classification.SPEECH
                        )
                    ],
                ],
                trimmed_ends=[],
                trimmed_beginnings=[],
                expected_cut_calls=[],
                expected_result=[
                    _make_audio_chunk(
                        start_samples=0, num_samples=20, sample_rate=10, classification=Classification.SPEECH
                    )
                ],
            ),
            param(
                "overlapping_chunks_midpoint_cutover_and_merge",
                input_chunks=[
                    _make_audio_chunk(start_samples=0, num_samples=40, sample_rate=10),
                    _make_audio_chunk(start_samples=20, num_samples=40, sample_rate=10),
                ],
                classify_returns=[
                    [
                        _make_audio_chunk(
                            start_samples=0, num_samples=40, sample_rate=10, classification=Classification.SPEECH
                        )
                    ],
                    [
                        _make_audio_chunk(
                            start_samples=20, num_samples=40, sample_rate=10, classification=Classification.SPEECH
                        )
                    ],
                ],
                trimmed_ends=[
                    [
                        _make_audio_chunk(
                            start_samples=0, num_samples=30, sample_rate=10, classification=Classification.SPEECH
                        )
                    ]
                ],
                trimmed_beginnings=[
                    [
                        _make_audio_chunk(
                            start_samples=30, num_samples=30, sample_rate=10, classification=Classification.SPEECH
                        )
                    ]
                ],
                expected_cut_calls=[30],
                expected_result=[
                    _make_audio_chunk(
                        start_samples=0, num_samples=60, sample_rate=10, classification=Classification.SPEECH
                    )
                ],
            ),
            param(
                "overlapping_chunks_different_classification_not_merged",
                input_chunks=[
                    _make_audio_chunk(start_samples=0, num_samples=40, sample_rate=10),
                    _make_audio_chunk(start_samples=20, num_samples=40, sample_rate=10),
                ],
                classify_returns=[
                    [
                        _make_audio_chunk(
                            start_samples=0, num_samples=40, sample_rate=10, classification=Classification.SPEECH
                        )
                    ],
                    [
                        _make_audio_chunk(
                            start_samples=20, num_samples=40, sample_rate=10, classification=Classification.MUSIC
                        )
                    ],
                ],
                trimmed_ends=[
                    [
                        _make_audio_chunk(
                            start_samples=0, num_samples=30, sample_rate=10, classification=Classification.SPEECH
                        )
                    ]
                ],
                trimmed_beginnings=[
                    [
                        _make_audio_chunk(
                            start_samples=30, num_samples=30, sample_rate=10, classification=Classification.MUSIC
                        )
                    ]
                ],
                expected_cut_calls=[30],
                expected_result=[
                    _make_audio_chunk(
                        start_samples=0, num_samples=30, sample_rate=10, classification=Classification.SPEECH
                    ),
                    _make_audio_chunk(
                        start_samples=30, num_samples=30, sample_rate=10, classification=Classification.MUSIC
                    ),
                ],
            ),
            param(
                "contiguous_non_overlapping_chunks",
                input_chunks=[
                    _make_audio_chunk(start_samples=0, num_samples=30, sample_rate=10),
                    _make_audio_chunk(start_samples=30, num_samples=30, sample_rate=10),
                ],
                classify_returns=[
                    [
                        _make_audio_chunk(
                            start_samples=0, num_samples=30, sample_rate=10, classification=Classification.SPEECH
                        )
                    ],
                    [
                        _make_audio_chunk(
                            start_samples=30, num_samples=30, sample_rate=10, classification=Classification.MUSIC
                        )
                    ],
                ],
                trimmed_ends=[
                    [
                        _make_audio_chunk(
                            start_samples=0, num_samples=30, sample_rate=10, classification=Classification.SPEECH
                        )
                    ]
                ],
                trimmed_beginnings=[
                    [
                        _make_audio_chunk(
                            start_samples=30, num_samples=30, sample_rate=10, classification=Classification.MUSIC
                        )
                    ]
                ],
                expected_cut_calls=[30],
                expected_result=[
                    _make_audio_chunk(
                        start_samples=0, num_samples=30, sample_rate=10, classification=Classification.SPEECH
                    ),
                    _make_audio_chunk(
                        start_samples=30, num_samples=30, sample_rate=10, classification=Classification.MUSIC
                    ),
                ],
            ),
        ]
    )
    def test_classify_chunks(
        self, _, input_chunks, classify_returns, trimmed_ends, trimmed_beginnings, expected_cut_calls, expected_result
    ):
        self.mock_classify_chunk.side_effect = classify_returns

        cut_end_iter = iter(trimmed_ends)
        cut_beg_iter = iter(trimmed_beginnings)

        def cut_end_side_effect(chunks, cutoff):
            del cutoff
            try:
                replacement = next(cut_end_iter)
                chunks.clear()
                chunks.extend(replacement)
            except StopIteration:
                pass

        def cut_beginning_side_effect(chunks, cutoff):
            del cutoff
            try:
                replacement = next(cut_beg_iter)
                chunks.clear()
                chunks.extend(replacement)
            except StopIteration:
                pass

        self.mock_cut_end.side_effect = cut_end_side_effect
        self.mock_cut_beginning.side_effect = cut_beginning_side_effect

        result = list(self.classifier.classify(input_chunks, channel=_CHANNEL, cache_dir=_CACHE_DIR))

        expected_calls = [call(ANY, cutoff) for cutoff in expected_cut_calls]
        self.mock_cut_end.assert_has_calls(expected_calls)
        self.mock_cut_beginning.assert_has_calls(expected_calls)

        self.assertEqual(result, expected_result)


class TestClassifyChunk(unittest.TestCase):
    """Unit tests for the `AudioClassifier._classify_chunk` method."""

    def setUp(self):
        self.mock_pann_inferencer_cls = self.enterContext(
            patch.object(ballroom_scribe, "PannInferencer", autospec=True)
        )
        self.mock_pann_inferencer_cls.CLASSIFICATIONS = _CLASSIFICATIONS
        self.mock_pann_inferencer = self.mock_pann_inferencer_cls.return_value

        self.mock_classify_frames = self.enterContext(patch.object(ballroom_scribe.AudioClassifier, "_classify_frames"))

        self.classifier = ballroom_scribe.AudioClassifier(
            device="cuda",
            thresholds=_THRESHOLDS,
            min_segment_seconds=0.1,
        )

    def test_empty_chunk_returns_immediately(self):
        empty_chunk = ballroom_scribe.models.AudioChunk(
            data=np.array([]),
            sample_rate=10,
            classification=Classification.UNKNOWN,
            start_samples=0,
        )

        result = list(self.classifier._classify_chunk(empty_chunk, channel=_CHANNEL, cache_dir=_CACHE_DIR))

        self.assertEqual(result, [])
        self.mock_pann_inferencer.infer_chunk.assert_not_called()
        self.mock_classify_frames.assert_not_called()

    @parameterized.expand(
        [
            param(
                "splits_chunk_at_classification_boundaries",
                audio_chunk=_make_audio_chunk(start_samples=100, num_samples=40, sample_rate=10),
                sample_classifications_rle=ballroom_scribe.models.RunLengthEncoding.from_lists(
                    values=[
                        Classification.SPEECH.numpy(),
                        (Classification.SPEECH | Classification.MUSIC).numpy(),
                        Classification.MUSIC.numpy(),
                        Classification.NONE.numpy(),
                    ],
                    lengths=[10, 10, 10, 10],
                ),
                expected_chunks=[
                    _make_audio_chunk(
                        start_samples=100, num_samples=10, sample_rate=10, classification=Classification.SPEECH
                    ),
                    _make_audio_chunk(
                        start_samples=110,
                        num_samples=10,
                        sample_rate=10,
                        classification=Classification.SPEECH | Classification.MUSIC,
                    ),
                    _make_audio_chunk(
                        start_samples=120, num_samples=10, sample_rate=10, classification=Classification.MUSIC
                    ),
                    _make_audio_chunk(
                        start_samples=130, num_samples=10, sample_rate=10, classification=Classification.NONE
                    ),
                ],
            ),
            param(
                "single_classification_chunk",
                audio_chunk=_make_audio_chunk(start_samples=0, num_samples=30, sample_rate=10),
                sample_classifications_rle=ballroom_scribe.models.RunLengthEncoding.from_lists(
                    values=[Classification.SPEECH.numpy()],
                    lengths=[30],
                ),
                expected_chunks=[
                    _make_audio_chunk(
                        start_samples=0, num_samples=30, sample_rate=10, classification=Classification.SPEECH
                    )
                ],
            ),
        ]
    )
    def test_classify_chunk(self, _, audio_chunk, sample_classifications_rle, expected_chunks):
        mock_frames = MagicMock(spec=np.ndarray)
        self.mock_pann_inferencer.infer_chunk.return_value = mock_frames
        self.mock_classify_frames.return_value = sample_classifications_rle

        result = list(self.classifier._classify_chunk(audio_chunk, channel=_CHANNEL, cache_dir=_CACHE_DIR))

        self.mock_pann_inferencer.infer_chunk.assert_called_once_with(
            audio_chunk=audio_chunk, channel=_CHANNEL, cache_dir=_CACHE_DIR
        )
        self.mock_classify_frames.assert_called_once_with(
            frames=mock_frames,
            num_samples=audio_chunk.duration_samples,
            sample_rate=audio_chunk.sample_rate,
        )
        self.assertEqual(result, expected_chunks)


class TestClassifyFrames(unittest.TestCase):
    """Unit tests for the `AudioClassifier._classify_frames` method."""

    def setUp(self):
        self.mock_pann_inferencer_cls = self.enterContext(
            patch.object(ballroom_scribe, "PannInferencer", autospec=True)
        )
        self.mock_pann_inferencer_cls.CLASSIFICATIONS = _CLASSIFICATIONS

        self.classifier = ballroom_scribe.AudioClassifier(
            device="cuda",
            thresholds=_THRESHOLDS,
            min_segment_seconds=0.1,
        )

    def test_classification_combinations(self):
        # 3 frames (columns: SPEECH, MUSIC):
        # Frame 0: SPEECH = 0.8, MUSIC = 0.8 -> SPEECH | MUSIC
        # Frame 1: SPEECH = 0.1, MUSIC = 0.8 -> MUSIC
        # Frame 2: all <= thresholds -> NONE
        frames = np.array(
            [
                [0.8, 0.8],
                [0.1, 0.8],
                [0.2, 0.2],
            ],
            dtype=np.float32,
        )

        result = self.classifier._classify_frames(frames=frames, num_samples=30, sample_rate=10)

        expected = ballroom_scribe.models.RunLengthEncoding.from_lists(
            values=[
                (Classification.SPEECH | Classification.MUSIC).numpy(),
                Classification.MUSIC.numpy(),
                Classification.NONE.numpy(),
            ],
            lengths=[10, 10, 10],
        )
        self.assertEqual(result, expected)

    def test_consecutive_frames_with_same_label_merged_by_rle(self):
        # 3 frames, all SPEECH
        frames = np.array(
            [
                [0.9, 0.0],
                [0.9, 0.0],
                [0.9, 0.0],
            ],
            dtype=np.float32,
        )

        result = self.classifier._classify_frames(frames=frames, num_samples=30, sample_rate=10)

        expected = ballroom_scribe.models.RunLengthEncoding.from_lists(
            values=[Classification.SPEECH.numpy()],
            lengths=[30],
        )
        self.assertEqual(result, expected)

    def test_converts_frames_to_samples_with_rounding(self):
        # 25 samples with 3 frames -> 25 / 3 = 8.333 samples per frame
        frames = np.array(
            [
                [0.9, 0.0],
                [0.0, 0.9],
                [0.0, 0.9],
            ],
            dtype=np.float32,
        )

        result = self.classifier._classify_frames(frames=frames, num_samples=25, sample_rate=10)

        # Frame lengths are [1, 2]. Cumulative frame lengths are [1, 3].
        # Scaled and rounded: round(1 * 25/3) = 8, round(3 * 25/3) = 25 -> diff is [8, 17].
        expected = ballroom_scribe.models.RunLengthEncoding.from_lists(
            values=[Classification.SPEECH.numpy(), Classification.MUSIC.numpy()],
            lengths=[8, 17],
        )
        self.assertEqual(result, expected)

    def test_binary_closing_bridges_short_gaps(self):
        # 7 frames, 70 samples at 10 Hz -> 1 frame/sec. min_segment_seconds=2.0 -> min_frames=2.
        # A 1-frame gap in SPEECH between two 2-frame bursts is bridged by binary_closing.
        classifier = ballroom_scribe.AudioClassifier(
            device="cuda",
            thresholds=_THRESHOLDS,
            min_segment_seconds=2.0,
        )
        frames = np.array(
            [
                [0.0, 0.0],
                [0.9, 0.0],
                [0.9, 0.0],
                [0.0, 0.0],
                [0.9, 0.0],
                [0.9, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        )

        result = classifier._classify_frames(frames=frames, num_samples=70, sample_rate=10)

        expected = ballroom_scribe.models.RunLengthEncoding.from_lists(
            values=[
                Classification.NONE.numpy(),
                Classification.SPEECH.numpy(),
                Classification.NONE.numpy(),
            ],
            lengths=[10, 50, 10],
        )
        self.assertEqual(result, expected)

    def test_binary_opening_discards_isolated_short_spikes(self):
        # 5 frames, 50 samples at 10 Hz -> 1 frame/sec. min_segment_seconds=2.0 -> min_frames=2.
        # An isolated 1-frame SPEECH spike is removed by binary_opening.
        classifier = ballroom_scribe.AudioClassifier(
            device="cuda",
            thresholds=_THRESHOLDS,
            min_segment_seconds=2.0,
        )
        frames = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.9, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        )

        result = classifier._classify_frames(frames=frames, num_samples=50, sample_rate=10)

        expected = ballroom_scribe.models.RunLengthEncoding.from_lists(
            values=[Classification.NONE.numpy()],
            lengths=[50],
        )
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
