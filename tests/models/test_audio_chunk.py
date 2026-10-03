import dataclasses
import unittest
from unittest.mock import MagicMock, patch

import librosa
import numpy as np
from parameterized import param, parameterized

import ballroom_scribe.models


class TestAudioChunk(unittest.TestCase):
    def setUp(self):
        self.classification = MagicMock(spec=ballroom_scribe.models.AudioChunk.Classification)

        # 10 samples at a sample rate of 10 Hz are a duration of 1 second.
        self.chunk = ballroom_scribe.models.AudioChunk(
            data=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            sample_rate=10,
            classification=self.classification,
            start_samples=100,
        )

    def test_duration_samples(self):
        self.assertEqual(self.chunk.duration_samples, 10)

    def test_data_immutable(self):
        with self.assertRaisesRegex(ValueError, "assignment destination is read-only"):
            self.chunk.data[0] = 999

    @parameterized.expand(
        [
            param(
                ballroom_scribe.models.AudioChunk.Classification.SPEECH,
                ballroom_scribe.models.AudioChunk.Classification.SPEECH,
                True,
            ),
            param(
                ballroom_scribe.models.AudioChunk.Classification.SPEECH
                | ballroom_scribe.models.AudioChunk.Classification.MUSIC,
                ballroom_scribe.models.AudioChunk.Classification.SPEECH,
                True,
            ),
            param(
                ballroom_scribe.models.AudioChunk.Classification.MUSIC,
                ballroom_scribe.models.AudioChunk.Classification.SPEECH,
                False,
            ),
        ]
    )
    def test_classification_is(
        self,
        initial: ballroom_scribe.models.AudioChunk.Classification,
        query: ballroom_scribe.models.AudioChunk.Classification,
        expected: bool,
    ):
        chunk = dataclasses.replace(self.chunk, classification=initial)
        self.assertEqual(chunk.classification_is(query), expected)

    def test_split(self):
        chunk = dataclasses.replace(
            self.chunk,
            data=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
        )

        chunk1, chunk2 = chunk.split(new_start_samples=115)

        self.assertEqual(
            chunk1,
            dataclasses.replace(
                self.chunk,
                data=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
            ),
        )
        self.assertEqual(
            chunk2,
            dataclasses.replace(
                self.chunk,
                data=np.array([15, 16, 17, 18, 19]),
                start_samples=115,
            ),
        )

    def test_split_out_of_bounds(self):
        with self.assertRaises(ValueError):
            self.chunk.split(new_start_samples=99)  # Before the start of the chunk

        with self.assertRaises(ValueError):
            self.chunk.split(new_start_samples=111)  # After the end of the chunk

    def test_copy(self):
        copy = self.chunk.copy()

        self.assertEqual(copy, self.chunk)
        self.assertIsNone(copy.data.base)

    def test_copy_if_sharing_memory_when_sharing(self):
        view, _ = self.chunk.split(new_start_samples=105)

        copy = view.copy_if_sharing_memory()

        self.assertIsNot(copy, view)
        self.assertEqual(copy, view)
        self.assertIsNone(copy.data.base)

    def test_copy_if_sharing_memory_when_not_sharing(self):
        result = self.chunk.copy_if_sharing_memory()

        self.assertIs(result, self.chunk)

    @patch.object(librosa, "resample", autospec=True)
    def test_resample(self, mock_resample):
        resampled_chunk = self.chunk.resample(new_sample_rate=20)

        mock_resample.assert_called_once_with(self.chunk.data, orig_sr=self.chunk.sample_rate, target_sr=20)
        self.assertEqual(resampled_chunk.start_samples, 200)
        self.assertEqual(resampled_chunk.data, mock_resample.return_value)
        self.assertEqual(resampled_chunk.sample_rate, 20)

    def test_add(self):
        chunk1 = dataclasses.replace(
            self.chunk,
            data=np.array([0, 1, 2, 3, 4]),
        )
        chunk2 = dataclasses.replace(
            self.chunk,
            data=np.array([5, 6, 7, 8, 9]),
        )

        combined_chunk = chunk1 + chunk2

        self.assertEqual(
            combined_chunk,
            dataclasses.replace(
                self.chunk,
                data=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            ),
        )

    def test_add_wrong_type(self):
        with self.assertRaises(TypeError):
            self.chunk += "not a chunk"

    def test_add_different_classification(self):
        other_chunk = dataclasses.replace(
            self.chunk,
            classification=MagicMock(spec=ballroom_scribe.models.AudioChunk.Classification),
        )

        with self.assertRaises(ValueError):
            self.chunk += other_chunk

    def test_add_different_sample_rate(self):
        other_chunk = dataclasses.replace(
            self.chunk,
            sample_rate=20,
        )

        with self.assertRaises(ValueError):
            self.chunk += other_chunk

    def test_unknown_classification_has_distinct_bitflag(self):
        classification_cls = ballroom_scribe.models.AudioChunk.Classification
        self.assertEqual(classification_cls.UNKNOWN & classification_cls.SPEECH, classification_cls.NONE)
        self.assertEqual(classification_cls.UNKNOWN & classification_cls.MUSIC, classification_cls.NONE)

    def test_composite_classification_numpy_round_trip(self):
        classification_cls = ballroom_scribe.models.AudioChunk.Classification
        combined = classification_cls.SPEECH | classification_cls.MUSIC
        combined_np = combined.numpy()
        self.assertEqual(classification_cls(combined_np), combined)

    def test_bool(self):
        self.assertTrue(self.chunk)
        self.assertFalse(dataclasses.replace(self.chunk, data=np.array([])))


if __name__ == "__main__":
    unittest.main()
