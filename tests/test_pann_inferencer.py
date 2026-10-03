import pathlib
import unittest
from unittest.mock import ANY, MagicMock, call, patch

import numpy as np
import panns_inference
from parameterized import param, parameterized

import ballroom_scribe
import ballroom_scribe.models
import ballroom_scribe.utils

_CHANNEL = 1
_CACHE_DIR = pathlib.Path("fake/path")


def _make_audio_chunk(
    start_samples: int,
    num_samples: int,
    sample_rate: int = 10,
) -> ballroom_scribe.models.AudioChunk:
    """Helper to create an AudioChunk with sequential data."""
    return ballroom_scribe.models.AudioChunk(
        data=np.arange(start_samples, start_samples + num_samples, dtype=np.float32),
        sample_rate=sample_rate,
        classification=ballroom_scribe.models.AudioChunk.Classification.UNKNOWN,
        start_samples=start_samples,
    )


class TestInit(unittest.TestCase):
    """Unit tests for the `PannInferencer.__init__` method."""

    def setUp(self):
        self.mock_validate_torch_device = self.enterContext(
            patch.object(ballroom_scribe.utils, "validate_torch_device", autospec=True)
        )

    def test_validates_torch_device(self):
        ballroom_scribe.PannInferencer(device="fake_device")
        self.mock_validate_torch_device.assert_called_once_with("fake_device")


class TestInfer(unittest.TestCase):
    """Unit tests for the `PannInferencer.infer` method."""

    def setUp(self):
        self.mock_validate_torch_device = self.enterContext(
            patch.object(ballroom_scribe.utils, "validate_torch_device", autospec=True)
        )
        self.mock_infer_chunk = self.enterContext(patch.object(ballroom_scribe.PannInferencer, "infer_chunk"))

        self.inferencer = ballroom_scribe.PannInferencer(device="cuda")

    def test_empty_chunks_iterable_returns_empty_array(self):
        result = self.inferencer.infer([], channel=_CHANNEL, cache_dir=_CACHE_DIR)

        self.assertEqual(result.shape, (0, len(ballroom_scribe.PannInferencer.CLASSIFICATIONS)))
        self.mock_infer_chunk.assert_not_called()

    def test_only_empty_audio_chunks_returns_empty_array(self):
        empty_chunks = [
            _make_audio_chunk(start_samples=0, num_samples=0, sample_rate=10),
            _make_audio_chunk(start_samples=10, num_samples=0, sample_rate=10),
        ]

        result = self.inferencer.infer(empty_chunks, channel=_CHANNEL, cache_dir=_CACHE_DIR)

        self.assertEqual(result.shape, (0, len(ballroom_scribe.PannInferencer.CLASSIFICATIONS)))
        self.mock_infer_chunk.assert_not_called()

    @parameterized.expand(
        [
            param(
                "single_chunk",
                input_chunks=[_make_audio_chunk(start_samples=0, num_samples=20, sample_rate=10)],
                inference_returns=[
                    np.array([[0.1, 0.2], [0.4, 0.5]], dtype=np.float32),
                ],
                expected_result=np.array([[0.1, 0.2], [0.4, 0.5]], dtype=np.float32),
            ),
            param(
                "skips_empty_chunks",
                input_chunks=[
                    _make_audio_chunk(start_samples=0, num_samples=0, sample_rate=10),
                    _make_audio_chunk(start_samples=0, num_samples=20, sample_rate=10),
                    _make_audio_chunk(start_samples=20, num_samples=0, sample_rate=10),
                ],
                inference_returns=[
                    np.array([[0.1, 0.2], [0.4, 0.5]], dtype=np.float32),
                ],
                expected_result=np.array([[0.1, 0.2], [0.4, 0.5]], dtype=np.float32),
            ),
            param(
                "contiguous_non_overlapping_chunks",
                input_chunks=[
                    _make_audio_chunk(start_samples=0, num_samples=20, sample_rate=10),
                    _make_audio_chunk(start_samples=20, num_samples=20, sample_rate=10),
                ],
                inference_returns=[
                    np.array([[0.1, 0.0], [0.2, 0.0]], dtype=np.float32),
                    np.array([[0.3, 0.0], [0.4, 0.0]], dtype=np.float32),
                ],
                expected_result=np.array(
                    [[0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0]],
                    dtype=np.float32,
                ),
            ),
            param(
                "three_overlapping_chunks_midpoint_cutover",
                # Chunk 0: samples 0..40 (4 frames, 10 samples/frame). Cutover with Chunk 1 is at sample 30 (frame 3).
                # Chunk 1: samples 20..60 (4 frames, 10 samples/frame). Trimmed at both ends to samples 30..50 (frames 1..3).
                # Chunk 2: samples 40..80 (4 frames, 10 samples/frame). Cutover with Chunk 1 is at sample 50 (frame 1).
                input_chunks=[
                    _make_audio_chunk(start_samples=0, num_samples=40, sample_rate=10),
                    _make_audio_chunk(start_samples=20, num_samples=40, sample_rate=10),
                    _make_audio_chunk(start_samples=40, num_samples=40, sample_rate=10),
                ],
                inference_returns=[
                    np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.9, 0.9]], dtype=np.float32),
                    np.array([[0.8, 0.8], [0.3, 0.0], [0.4, 0.0], [0.7, 0.7]], dtype=np.float32),
                    np.array([[0.6, 0.6], [0.5, 0.0], [0.6, 0.0], [0.7, 0.0]], dtype=np.float32),
                ],
                expected_result=np.array(
                    [
                        [0.0, 0.0],
                        [0.1, 0.0],
                        [0.2, 0.0],
                        [0.3, 0.0],
                        [0.4, 0.0],
                        [0.5, 0.0],
                        [0.6, 0.0],
                        [0.7, 0.0],
                    ],
                    dtype=np.float32,
                ),
            ),
            param(
                "overlapping_chunks_with_frame_rounding",
                # 25 samples across 3 frames -> 8.333 samples/frame.
                # Overlap cutover is floor((25 + 15) / 2) = 20.
                # Chunk 0 end frame: round((20 - 0) / (25 / 3)) = round(2.4) = 2.
                # Chunk 1 start frame: round((20 - 15) / (25 / 3)) = round(0.6) = 1.
                input_chunks=[
                    _make_audio_chunk(start_samples=0, num_samples=25, sample_rate=10),
                    _make_audio_chunk(start_samples=15, num_samples=25, sample_rate=10),
                ],
                inference_returns=[
                    np.array([[0.1, 0.0], [0.2, 0.0], [0.9, 0.0]], dtype=np.float32),
                    np.array([[0.8, 0.0], [0.3, 0.0], [0.4, 0.0]], dtype=np.float32),
                ],
                expected_result=np.array(
                    [
                        [0.1, 0.0],
                        [0.2, 0.0],
                        [0.3, 0.0],
                        [0.4, 0.0],
                    ],
                    dtype=np.float32,
                ),
            ),
        ]
    )
    def test_infer(self, _, input_chunks, inference_returns, expected_result):
        self.mock_infer_chunk.side_effect = inference_returns

        result = self.inferencer.infer(input_chunks, channel=_CHANNEL, cache_dir=_CACHE_DIR)

        non_empty_chunks = [chunk for chunk in input_chunks if chunk]
        expected_calls = [call(audio_chunk=chunk, channel=_CHANNEL, cache_dir=_CACHE_DIR) for chunk in non_empty_chunks]
        self.mock_infer_chunk.assert_has_calls(expected_calls)
        np.testing.assert_array_equal(result, expected_result)


class TestInferChunk(unittest.TestCase):
    """Unit tests for the `PannInferencer.infer_chunk` method."""

    def setUp(self):
        self.mock_validate_torch_device = self.enterContext(
            patch.object(ballroom_scribe.utils, "validate_torch_device", autospec=True)
        )

        self.mock_sed_cls = self.enterContext(patch.object(panns_inference, "SoundEventDetection"))
        self.mock_sed = self.mock_sed_cls.return_value

        self.mock_path_exists = self.enterContext(patch.object(pathlib.Path, "exists"))
        self.mock_np_load = self.enterContext(patch.object(np, "load"))
        self.mock_np_save = self.enterContext(patch.object(np, "save"))

        self.mock_atomic_path = self.enterContext(patch.object(ballroom_scribe.utils, "atomic_path", autospec=True))
        self.mock_partial_cache_file = MagicMock(spec=pathlib.Path)
        self.mock_atomic_path.return_value.__enter__.return_value = self.mock_partial_cache_file

        self.inferencer = ballroom_scribe.PannInferencer(device="cuda")

    def test_lazy_loads_sound_event_detection(self):
        self.mock_path_exists.return_value = False
        audio_chunk = _make_audio_chunk(start_samples=100, num_samples=40, sample_rate=10)
        self.mock_sed.inference.return_value = np.zeros((1, 2, 527), dtype=np.float32)

        self.mock_sed_cls.assert_not_called()
        self.inferencer.infer_chunk(audio_chunk=audio_chunk, channel=_CHANNEL, cache_dir=_CACHE_DIR)
        self.inferencer.infer_chunk(audio_chunk=audio_chunk, channel=_CHANNEL, cache_dir=_CACHE_DIR)
        self.mock_sed_cls.assert_called_once_with(device="cuda")

    def test_cache_hit_loads_inference_from_cache(self):
        self.mock_path_exists.return_value = True
        audio_chunk = _make_audio_chunk(start_samples=100, num_samples=40, sample_rate=10)
        cached_frames = np.zeros((4, 2), dtype=np.float32)
        self.mock_np_load.return_value = cached_frames

        result = self.inferencer.infer_chunk(audio_chunk=audio_chunk, channel=_CHANNEL, cache_dir=_CACHE_DIR)

        self.mock_path_exists.assert_called_once_with()
        self.mock_np_load.assert_called_once_with(_CACHE_DIR / f"0000000100-0000000140-{_CHANNEL}.npy")
        self.mock_sed.inference.assert_not_called()
        self.mock_np_save.assert_not_called()
        self.mock_atomic_path.assert_not_called()
        np.testing.assert_array_equal(result, cached_frames)

    def test_cache_miss_computes_and_caches_per_category_max_probabilities(self):
        self.mock_path_exists.return_value = False
        audio_chunk = _make_audio_chunk(start_samples=100, num_samples=40, sample_rate=10)

        # 2 frames across 527 AudioSet labels:
        # Frame 0: SPEECH (indices 0, 3) -> max 0.8; MUSIC (indices 27, 150) -> max 0.6
        # Frame 1: SPEECH (index 1) -> max 0.3; MUSIC (index 30) -> max 0.9
        inference_output = np.zeros((1, 2, 527), dtype=np.float32)
        inference_output[0, 0, 0] = 0.2
        inference_output[0, 0, 3] = 0.8
        inference_output[0, 0, 27] = 0.5
        inference_output[0, 0, 150] = 0.6

        inference_output[0, 1, 1] = 0.3
        inference_output[0, 1, 30] = 0.9
        self.mock_sed.inference.return_value = inference_output

        expected_frames = np.array(
            [
                [0.8, 0.6],
                [0.3, 0.9],
            ],
            dtype=np.float32,
        )

        result = self.inferencer.infer_chunk(audio_chunk=audio_chunk, channel=_CHANNEL, cache_dir=_CACHE_DIR)

        self.mock_path_exists.assert_called_once_with()
        self.mock_np_load.assert_not_called()

        self.mock_sed.inference.assert_called_once()
        call_arg = self.mock_sed.inference.call_args.args[0]
        self.assertEqual(call_arg.shape, (1, 40))
        np.testing.assert_array_equal(call_arg[0], audio_chunk.data)
        self.assertFalse(audio_chunk.data.flags.writeable)

        self.mock_atomic_path.assert_called_once_with(_CACHE_DIR / f"0000000100-0000000140-{_CHANNEL}.npy")
        self.mock_np_save.assert_called_once_with(self.mock_partial_cache_file, ANY)
        np.testing.assert_array_equal(self.mock_np_save.call_args.args[1], expected_frames)
        np.testing.assert_array_equal(result, expected_frames)


if __name__ == "__main__":
    unittest.main()
