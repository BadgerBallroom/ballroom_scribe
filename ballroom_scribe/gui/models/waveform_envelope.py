"""GUI model representing a precomputed min/max audio waveform envelope."""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class WaveformEnvelope:
    """Stores min/max amplitude peaks at a fixed bin rate for fast waveform rendering.

    Attributes:
        mins: 1-D `float32` array of per-bin minimum sample values.
        maxs: 1-D `float32` array of per-bin maximum sample values, same length as `mins`.
        bins_per_second: Number of envelope bins per second of audio.
    """

    mins: np.typing.NDArray[np.float32]
    maxs: np.typing.NDArray[np.float32]
    bins_per_second: float

    def __post_init__(self) -> None:
        if self.mins.ndim != 1 or self.maxs.ndim != 1:
            raise ValueError("WaveformEnvelope arrays must be 1-dimensional.")
        if self.mins.shape != self.maxs.shape:
            raise ValueError(
                f"WaveformEnvelope mins and maxs must have the same shape, got {self.mins.shape} and {self.maxs.shape}."
            )
        if not math.isfinite(self.bins_per_second) or self.bins_per_second <= 0.0:
            raise ValueError(f"bins_per_second must be a positive finite number, got {self.bins_per_second}.")

        mins_f32 = np.asarray(self.mins, dtype=np.float32)
        maxs_f32 = np.asarray(self.maxs, dtype=np.float32)
        mins_f32.flags.writeable = False
        maxs_f32.flags.writeable = False
        object.__setattr__(self, "mins", mins_f32)
        object.__setattr__(self, "maxs", maxs_f32)

    @property
    def num_bins(self) -> int:
        """Returns the total number of bins in the envelope."""
        return int(self.mins.shape[0])

    @property
    def duration_seconds(self) -> float:
        """Returns the total audio duration covered by the envelope in seconds."""
        return self.num_bins / self.bins_per_second

    def resample(
        self,
        start_seconds: float,
        end_seconds: float,
        count: int,
    ) -> tuple[np.typing.NDArray[np.float32], np.typing.NDArray[np.float32]]:
        """Downsamples or upsamples `[start_seconds, end_seconds]` into `count` pixel columns.

        Columns completely outside `[0, duration_seconds]` are filled with `NaN`.

        Args:
            start_seconds: Left edge of the visible time window in seconds.
            end_seconds: Right edge of the visible time window in seconds. Must exceed `start_seconds`.
            count: Number of output columns (pixels).

        Returns:
            A `(column_mins, column_maxs)` pair of 1-D `float32` arrays of length `count`.

        Raises:
            ValueError: If `count` is negative or `end_seconds <= start_seconds`.
        """
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}.")
        if end_seconds <= start_seconds:
            raise ValueError(f"end_seconds ({end_seconds}) must be greater than start_seconds ({start_seconds}).")
        if count == 0:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

        out_mins = np.full(count, np.nan, dtype=np.float32)
        out_maxs = np.full(count, np.nan, dtype=np.float32)
        if self.num_bins == 0:
            return out_mins, out_maxs

        edges_bins = np.linspace(start_seconds, end_seconds, count + 1) * self.bins_per_second
        b0 = edges_bins[:-1]
        b1 = edges_bins[1:]

        valid_mask = (b1 > 0.0) & (b0 < self.num_bins)
        if not np.any(valid_mask):
            return out_mins, out_maxs

        clamped_b0 = np.clip(b0[valid_mask], 0.0, float(self.num_bins))
        clamped_b1 = np.clip(b1[valid_mask], 0.0, float(self.num_bins))

        lo = np.clip(np.floor(clamped_b0).astype(np.intp), 0, self.num_bins - 1)
        hi = np.clip(np.maximum(lo + 1, np.ceil(clamped_b1 - 1e-9).astype(np.intp)), 1, self.num_bins)

        pairs = np.empty(2 * lo.shape[0], dtype=np.intp)
        pairs[0::2] = lo
        pairs[1::2] = hi

        # Pad by one element so `hi == num_bins` is always a valid in-bounds index for `reduceat`.
        padded_mins = np.append(self.mins, np.float32(0.0))
        padded_maxs = np.append(self.maxs, np.float32(0.0))

        out_mins[valid_mask] = np.minimum.reduceat(padded_mins, pairs)[0::2]
        out_maxs[valid_mask] = np.maximum.reduceat(padded_maxs, pairs)[0::2]
        return out_mins, out_maxs

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, type(self))
            and self.bins_per_second == value.bins_per_second
            and np.array_equal(self.mins, value.mins)
            and np.array_equal(self.maxs, value.maxs)
        )
