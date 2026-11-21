"""Proportional column width distribution with per-column minimums."""

from collections.abc import Sequence


def distribute_column_widths(
    total: int,
    weights: Sequence[float],
    min_widths: Sequence[int],
) -> list[int]:
    """Splits `total` pixels across columns in proportion to `weights`, each at least its `min_widths` entry.

    Columns whose proportional share falls below their minimum are pinned to that minimum so the remaining width goes to
    the rest, and fractional pixels are awarded by largest remainder so equal-weight columns differ by at most `1` px
    and sum to `max(total, sum(min_widths))`.

    Args:
        total: Target total pixel width across all columns.
        weights: Non-negative relative weights for each column (all zeros falls back to equal shares).
        min_widths: Minimum pixel width for each column; must have the same length as `weights`.

    Returns:
        Pixel widths for each column, each `>= min_widths[i]`.

    Raises:
        ValueError: If `weights` and `min_widths` have different lengths, or any weight is negative.
    """
    if len(weights) != len(min_widths):
        raise ValueError(f"weights ({len(weights)}) and min_widths ({len(min_widths)}) must have the same length.")
    if any(weight < 0 for weight in weights):
        raise ValueError("Column weights must be non-negative.")
    if not weights:
        return []
    min_total = sum(min_widths)
    if total <= min_total:
        return list(min_widths)

    widths = [0] * len(weights)
    remaining = total
    active = list(range(len(weights)))
    while active:
        active_weight = sum(weights[i] for i in active)
        if active_weight <= 0:
            shares = {i: remaining / len(active) for i in active}
        else:
            shares = {i: remaining * weights[i] / active_weight for i in active}
        clamped = [i for i in active if shares[i] < min_widths[i]]
        if not clamped:
            floors = {i: int(shares[i]) for i in active}
            for i, width in floors.items():
                widths[i] = width
            leftover = remaining - sum(floors.values())
            by_remainder = sorted(active, key=lambda i: (shares[i] - floors[i], i), reverse=True)
            for i in by_remainder[:leftover]:
                widths[i] += 1
            break
        clamped_set = set(clamped)
        for i in clamped:
            widths[i] = min_widths[i]
            remaining -= min_widths[i]
        active = [i for i in active if i not in clamped_set]
    return widths
