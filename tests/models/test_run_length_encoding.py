import unittest

import numpy as np

import ballroom_scribe.models


class TestRunLengthEncoding(unittest.TestCase):
    def test_data_immutable(self):
        rle = ballroom_scribe.models.RunLengthEncoding.from_lists(values=[1, 2, 3], lengths=[3, 3, 3])

        with self.assertRaisesRegex(ValueError, "assignment destination is read-only"):
            rle.values[0] = 999
        with self.assertRaisesRegex(ValueError, "assignment destination is read-only"):
            rle.lengths[0] = 999

    def test_lengths_must_be_int64(self):
        with self.assertRaisesRegex(TypeError, "The dtype of the lengths array should be int64."):
            ballroom_scribe.models.RunLengthEncoding(
                values=np.array([1, 2, 3]),
                lengths=np.array([3, 3, 3], dtype="int32"),
            )

    def test_iter(self):
        rle = ballroom_scribe.models.RunLengthEncoding.from_lists(values=[1, 2, 3], lengths=[4, 5, 6])

        self.assertEqual(list(rle), [(1, 4), (2, 5), (3, 6)])

    def test_from_ndarray(self):
        a = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3])

        result = ballroom_scribe.models.RunLengthEncoding.from_ndarray(a)

        self.assertEqual(
            result,
            ballroom_scribe.models.RunLengthEncoding.from_lists(values=[0, 1, 2, 3], lengths=[4, 3, 2, 1]),
        )


if __name__ == "__main__":
    unittest.main()
