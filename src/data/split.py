"""Time-ordered train/val/test splitting.

The split is performed **per source file** on row order (rows are already
time-sorted), so every attack type appears in train, val and test, and no
window ever mixes data across a split boundary. There is no shuffling:
train always precedes val, which always precedes test in time.
"""

from __future__ import annotations

from src.config import SPLIT_FRACTIONS

SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")


def split_boundaries(
    n_rows: int, fractions: tuple[float, float, float] = SPLIT_FRACTIONS
) -> dict[str, tuple[int, int]]:
    """Return ``{split_name: (start_row, end_row)}`` for a time-sorted file.

    Boundaries are computed on row counts; because rows are time-ordered,
    this is equivalent to splitting on time while keeping exact fractions.
    """
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"Split fractions must sum to 1, got {fractions}")
    train_end = int(n_rows * fractions[0])
    val_end = int(n_rows * (fractions[0] + fractions[1]))
    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, n_rows),
    }
