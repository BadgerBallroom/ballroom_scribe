import unittest

import ballroom_scribe.utils


class TestComputeOverlapCutover(unittest.TestCase):
    """Unit tests for the `compute_overlap_cutover` function."""

    def test_previous_chunk_ends_before_current_chunk(self):
        with self.assertRaisesRegex(ValueError, "The previous chunk ends before the current one starts"):
            ballroom_scribe.utils.compute_overlap_cutover(
                previous_start_samples=100,
                previous_end_samples=90,
                current_start_samples=100,
            )

    def test_current_chunk_starts_before_previous_chunk(self):
        with self.assertRaisesRegex(ValueError, "The current chunk starts before the previous one"):
            ballroom_scribe.utils.compute_overlap_cutover(
                previous_start_samples=100,
                previous_end_samples=100,
                current_start_samples=90,
            )

    def test_computes_cutover(self):
        cutover = ballroom_scribe.utils.compute_overlap_cutover(
            previous_start_samples=100,
            previous_end_samples=110,
            current_start_samples=105,
        )
        self.assertEqual(cutover, 107)


if __name__ == "__main__":
    unittest.main()
