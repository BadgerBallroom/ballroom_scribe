"""Utility for validating overlapping chunk boundaries and computing their midpoint cutover sample."""

import math


def compute_overlap_cutover(
    previous_start_samples: int,
    previous_end_samples: int,
    current_start_samples: int,
) -> int:
    """Validates consecutive chunk boundaries and returns the midpoint sample of their overlap.

    Args:
        previous_start_samples: The index of the sample that started the previous chunk.
        previous_end_samples: The index of the sample that ended the previous chunk.
        current_start_samples: The index of the sample that started the current chunk.

    Returns:
        The midpoint sample of the overlap between the two chunks.

    Raises:
        ValueError: The previous chunk ends before the current one starts.
        ValueError: The current chunk starts before the previous one.
    """
    if previous_end_samples < current_start_samples:
        raise ValueError("The previous chunk ends before the current one starts.")

    if previous_start_samples > current_start_samples:
        raise ValueError("The current chunk starts before the previous one.")

    return math.floor((previous_end_samples + current_start_samples) / 2)
