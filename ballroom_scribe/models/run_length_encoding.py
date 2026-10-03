from dataclasses import dataclass
from typing import Self

import numpy as np


@dataclass(frozen=True, slots=True)
class RunLengthEncoding[T: np.generic]:
    """Represents a 1D NumPy as runs of repeating values.

    See: https://en.wikipedia.org/wiki/Run-length_encoding

    Attributes:
        values: A NumPy array of the runs' values.
        lengths: A NumPy array of the runs' lengths.
    """

    type Length = np.int64

    values: np.typing.NDArray[T]
    lengths: np.typing.NDArray[Length]

    def __post_init__(self):
        # Make the NumPy arrays immutable.
        # Technically, you can set the flag back to `True`, but this is enough to catch accidental attempts at mutation.
        self.values.flags.writeable = False
        self.lengths.flags.writeable = False

        if self.lengths.dtype != "int64":
            raise TypeError("The dtype of the lengths array should be int64.")

    def __iter__(self) -> zip[tuple[T, Length]]:
        return zip(self.values, self.lengths)

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, type(self))
            and np.array_equal(self.values, value.values)
            and np.array_equal(self.lengths, value.lengths)
        )

    @classmethod
    def from_lists(cls, *, values: list, lengths: list[int]):
        return cls(
            values=np.array(values),
            lengths=np.array(lengths, dtype="int64"),
        )

    @classmethod
    def from_ndarray(cls, a: np.typing.NDArray[T]) -> Self:
        """Performs run-length encoding on a 1D NumPy array."""
        if a.ndim != 1:
            raise ValueError("Input array must be 1D.")
        if a.size == 0:
            return cls(values=np.array([]), lengths=np.array([], dtype=np.int64))

        # Find where value changes.
        # Per the NumPy documentation, `np.nonzero` is preferred over `np.where` when only one argument is provided.
        # `np.flatnonzero` then has the benefit of not needing to subscript the result like with `np.nonzero`.
        change_indices = np.flatnonzero(a[:-1] != a[1:]) + 1
        run_boundaries = np.concatenate([np.array([0]), change_indices, np.array([a.size])])

        return cls(
            values=a[run_boundaries[:-1]],
            lengths=np.diff(run_boundaries),
        )
