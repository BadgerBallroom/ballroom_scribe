"""Helper functions to help find a sub-sequence inside a bigger sequence using __eq__ equality."""

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Protocol, Self


class HasEq(Protocol):
    def __eq__(self, other: object, /) -> bool: ...


class _EditOperation(Enum):
    """The lowest-cost operation to reach a cell in the matrix in Sellers' algorithm.

    Attributes:
        NONE: This cell was reached by advancing from the previous to the current elements in both the needle and the
            haystack, and they were equal.
        SUBSTITUTION: This cell was reached by advancing from the previous to the current elements in both the needle
            and the haystack, but they were not equal.
        INSERTION: This cell was reached by advancing from the previous to the current element in the haystack but not
            changing the current element in the needle.
        DELETION: This cell was reached by advancing from the previous to the current element in the needle but not
            changing the current element in the haystack.
    """

    NONE = 0
    SUBSTITUTION = 1
    INSERTION = 2
    DELETION = 3


@dataclass(frozen=True)
class _EditDistance:
    """The value of one cell in the matrix in Sellers' algorithm.

    Attributes:
        last_operation: The operation that was taken to reach this cell.
        distance: At row i column j
    """

    last_operation: _EditOperation
    distance: int

    DEFAULT: ClassVar[Self]

    def __lt__(self, other, /):
        if not isinstance(other, _EditDistance):
            raise TypeError("other must be an _EditDistance")

        return self.distance < other.distance

    def __str__(self):
        return f"{self._get_last_operation_as_arrow()} {self.distance:2d}"

    def _get_last_operation_as_arrow(self):
        match self.last_operation:
            case _EditOperation.NONE | _EditOperation.SUBSTITUTION:
                return "\u2196"  # North west arrow
            case _EditOperation.INSERTION:
                return "\u2190"  # Leftwards arrow
            case _EditOperation.DELETION:
                return "\u2191"  # Upwards arrow
            case _:
                raise RuntimeError(f"Unknown operation {self.last_operation!r}")


_EditDistance.DEFAULT = _EditDistance(last_operation=_EditOperation.NONE, distance=0)


class _SellersMatrixRow[T: HasEq]:
    """Represents one row in the matrix in Sellers' algorithm, corresponding to one element in the needle.

    Each cell corresponds to one element in the haystack.
    """

    def __init__(self, index_in_needle: int, haystack: Sequence[T]):
        """Initializes the instance.

        Args:
            haystack: The sequence in which to search
        """
        # The leftmost column corresponds to one position *before* the start of the haystack.
        # Therefore, it always represents a deletion with a cost that increases by 1 for each row.
        self._cell_negative_one = _EditDistance(last_operation=_EditOperation.DELETION, distance=index_in_needle + 1)

        self._cells = [_EditDistance.DEFAULT] * len(haystack)

    def __getitem__(self, key, /):
        if not isinstance(key, int):
            raise TypeError("key must be int")

        if key == -1:
            return self._cell_negative_one

        if 0 <= key < len(self._cells):
            return self._cells[key]

        raise IndexError("index out of bounds")

    def __setitem__(self, key, value, /):
        if not isinstance(key, int):
            raise TypeError("key must be int")

        if not isinstance(value, _EditDistance):
            raise TypeError("value must be _EditDistance")

        if key == -1:
            raise RuntimeError("The leftmost column should not change")

        if 0 <= key < len(self._cells):
            self._cells[key] = value
            return

        raise IndexError("index out of bounds")

    def __iter__(self):
        return iter(self._cells)

    def __str__(self):
        return f"{self[-1].distance - 1:5d} \u2503\t" + "\t".join(
            str(cell) for cell in itertools.chain([self[-1]], self._cells)
        )


class _SellersMatrixRowNegativeOne[T: HasEq](_SellersMatrixRow):
    """Represents the row at index `-1` in the matrix in Sellers' algorithm.

    Always has an edit distance of 0.
    """

    _distance = _EditDistance(last_operation=_EditOperation.INSERTION, distance=0)

    def __init__(self, haystack: Sequence[T]):
        super().__init__(-1, ())
        self._haystack_length = len(haystack)

    def __getitem__(self, key, /):
        del key
        return super().__getitem__(-1)

    def __setitem__(self, key, value, /):
        del key, value
        raise RuntimeError("Row -1 should not be modified.")

    def __str__(self):
        return (
            "".join(f"\t{j:4d}" for j in range(-1, self._haystack_length))
            + "\n      \u250f\u2501"
            + "".join("\u2501" * 8 for _ in range(self._haystack_length + 1))
            + "\n   -1 \u2503\t   0"
            + "".join(f"\t{self._distance}" for _ in range(self._haystack_length))
        )


class _SellersMatrix[T: HasEq]:
    """Represents the matrix in Sellers' algorithm.

    Each row corresponds to one element in the needle. Each column corresponds to one element in the haystack.
    """

    def __init__(self, needle: Sequence[T], haystack: Sequence[T]):
        """Initializes the instance.

        Args:
            needle: The sub-sequence to search for in `haystack`
            haystack: The sequence in which to search for `needle`
        """
        self._row_negative_one = _SellersMatrixRowNegativeOne(haystack=haystack)
        self._rows = [_SellersMatrixRow(index_in_needle=i, haystack=haystack) for i in range(len(needle))]

    def __getitem__(self, key, /):
        if not isinstance(key, int):
            raise TypeError("key must be int")

        if key == -1:
            return self._row_negative_one

        if 0 <= key < len(self._rows):
            return self._rows[key]

        raise IndexError("index out of bounds")

    def __str__(self):
        return "\n".join(str(row) for row in itertools.chain([self[-1]], self._rows))


def find_closest_matching[T: HasEq](needle: Sequence[T], haystack: Sequence[T]) -> int:
    """Finds the position in `haystack` where the sub-sequence that most closely matches `needle` starts.

    An exact match is not required. For example, given the following sequences:

    *   `needle`: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    *   `haystack`: 2, 3, 5, 6

    The result would be 2, even though **4** is missing from `haystack`.
    Elements can be missing from `needle`, `haystack`, or both.

    If multiple sub-sequences in `haystack` have the same similarity to `needle`, the first one is used.

    Args:
        needle: The sub-sequence to search for in `haystack`
        haystack: The sequence in which to search for `needle`
    """
    if not haystack:
        raise ValueError("The haystack cannot be empty.")

    if not needle:
        return 0

    if len(needle) >= len(haystack):
        return 0

    if needle == haystack:
        return 0

    # This is an implementation of Sellers' algorithm.
    # distances[i][j] stores the edit distance between needle[:i + 1] and the sub-sequence ending at haystack[j].
    distances = _SellersMatrix(needle=needle, haystack=haystack)
    # The first row (distances[-1][j]) is left as 0 so that the match can begin at any position with 0 starting cost.

    for index_in_needle, needle_element in enumerate(needle):
        for index_in_haystack, haystack_element in enumerate(haystack):
            distances[index_in_needle][index_in_haystack] = (
                _EditDistance(
                    last_operation=_EditOperation.NONE,
                    distance=distances[index_in_needle - 1][index_in_haystack - 1].distance,
                )
                if needle_element == haystack_element
                else min(
                    _EditDistance(
                        last_operation=_EditOperation.SUBSTITUTION,
                        distance=distances[index_in_needle - 1][index_in_haystack - 1].distance + 1,
                    ),
                    _EditDistance(
                        last_operation=_EditOperation.INSERTION,
                        distance=distances[index_in_needle][index_in_haystack - 1].distance + 1,
                    ),
                    _EditDistance(
                        last_operation=_EditOperation.DELETION,
                        distance=distances[index_in_needle - 1][index_in_haystack].distance + 1,
                    ),
                )
            )

    # The last row (distances[len(needle) - 1][j]) now contains the edit distances between needle[:len(needle) - 1 + 1],
    # which is the whole needle, and the sub-sequence ending at haystack[j]. First, let's find the j that has the lowest
    # edit distance; this is the position of the end of the needle within the haystack.
    best_match_ends_at, _ = min(enumerate(distances[len(needle) - 1]), key=lambda pair: pair[1])

    # Trace the operations back to the start of the needle.
    index_in_needle = len(needle) - 1
    index_in_haystack = best_match_ends_at
    while index_in_needle > 0:
        current_cell = distances[index_in_needle][index_in_haystack]
        match current_cell.last_operation:
            case _EditOperation.NONE | _EditOperation.SUBSTITUTION:
                index_in_needle -= 1
                index_in_haystack -= 1
            case _EditOperation.INSERTION:
                index_in_haystack -= 1
            case _EditOperation.DELETION:
                index_in_needle -= 1
            case _:
                raise RuntimeError(f"Unknown operation {current_cell.last_operation!r}")
        if index_in_haystack < 0:
            # The needle starts before the haystack.
            return 0

    return index_in_haystack
