"""Feature engineering for CAN intrusion detection.

Two feature levels are produced from the processed parquet files:

**Per-message features** (causal — each uses only the current and past
messages, so they can be computed on a live stream):

* ``dt_prev`` — time since the previous message on the bus,
* ``dt_same_id`` — time since the previous message with the same CAN ID,
* ``rel_dev`` — relative deviation of ``dt_same_id`` from the ID's expected
  period (median period learned on attack-free TRAIN traffic), clipped to
  ``[-1, UNKNOWN_ID_DEV]``; messages with an ID never seen in attack-free
  train traffic get the maximum value,
* ``is_unknown`` — 1 if the CAN ID was never seen in attack-free train
  traffic.

**Window features** (sliding windows of ``WINDOW_SIZE`` messages, stride
``WINDOW_STRIDE``): ID entropy, unique-ID count, unknown-ID ratio,
mean/std inter-arrival time, mean/mean-abs/max-abs ``rel_dev``.

A window is labeled ``attack`` (label = 1) if it contains **at least one**
injected frame; the window also keeps the attack type and the exact
injected-frame ratio.

Leakage control: expected periods and the known-ID set are learned only on
attack-free frames from the TRAIN portion of every file; windows never
cross a split boundary or a file boundary.

Run the full pipeline::

    python -m src.data.features
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import (
    PROCESSED_DIR,
    REPORTS_DIR,
    WINDOW_SIZE,
    WINDOW_STRIDE,
)
from src.data.split import SPLIT_NAMES, split_boundaries

SOURCES: tuple[str, ...] = ("normal", "dos", "fuzzy", "gear", "rpm")

# rel_dev assigned to messages whose CAN ID never occurs in attack-free
# train traffic (also the clip ceiling for known IDs).
UNKNOWN_ID_DEV: float = 10.0

MESSAGE_FEATURES: list[str] = ["dt_prev", "dt_same_id", "rel_dev", "is_unknown"]
WINDOW_FEATURES: list[str] = [
    "id_entropy", "unique_id_count", "unknown_ratio",
    "mean_dt", "std_dt", "mean_rel_dev", "mean_abs_rel_dev", "max_abs_rel_dev",
]


# ---------------------------------------------------------------------------
# Pass 1 — statistics from attack-free train traffic
# ---------------------------------------------------------------------------

def compute_train_stats(frames: dict[str, pd.DataFrame]) -> dict:
    """Learn the known-ID set and per-ID expected periods.

    Uses only attack-free (label == 0) frames inside the TRAIN portion of
    each file. Periods are medians of same-ID inter-arrival times pooled
    across files.
    """
    pooled: list[pd.DataFrame] = []
    for df in frames.values():
        train_end = split_boundaries(len(df))["train"][1]
        train = df.iloc[:train_end]
        clean = train.loc[train["label"] == 0, ["can_id", "timestamp"]]
        dt = clean.groupby("can_id", observed=True)["timestamp"].diff()
        pooled.append(
            pd.DataFrame({"can_id": clean["can_id"], "dt": dt}).dropna()
        )
    all_dt = pd.concat(pooled, ignore_index=True)
    periods = all_dt.groupby("can_id", observed=True)["dt"].median()
    periods = periods[periods > 0]
    return {
        "known_ids": sorted(int(i) for i in periods.index),
        "period_by_id": {int(i): float(p) for i, p in periods.items()},
        "global_median_period": float(periods.median()),
    }


# ---------------------------------------------------------------------------
# Pass 2 — per-message features (causal)
# ---------------------------------------------------------------------------

def add_message_features(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Append causal per-message timing features to one file's frames."""
    out = df.copy()
    ts = out["timestamp"]

    out["dt_prev"] = ts.diff().fillna(0.0).astype(np.float32)
    out["dt_same_id"] = (
        out.groupby("can_id", observed=True)["timestamp"].diff()
    )

    period_map = {int(k): v for k, v in stats["period_by_id"].items()}
    periods = out["can_id"].map(period_map)
    known = periods.notna()
    out["is_unknown"] = (~known).astype(np.uint8)

    # First occurrence of a known ID: neutral deviation (dt = period).
    dt = out["dt_same_id"].fillna(periods)
    rel_dev = (dt - periods) / periods
    rel_dev = rel_dev.clip(-1.0, UNKNOWN_ID_DEV)
    rel_dev[~known] = UNKNOWN_ID_DEV
    out["rel_dev"] = rel_dev.astype(np.float32)

    out["dt_same_id"] = (
        out["dt_same_id"]
        .fillna(periods)
        .fillna(stats["global_median_period"])
        .astype(np.float32)
    )
    return out


# ---------------------------------------------------------------------------
# Pass 3 — sliding-window features
# ---------------------------------------------------------------------------

def _entropy_and_unique(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shannon entropy (bits) and unique count of IDs per window row.

    Fully vectorised: sort each row, find run boundaries on the flattened
    matrix, then accumulate ``-p log2 p`` per run into its window.
    """
    n_windows, length = codes.shape
    sorted_rows = np.sort(codes, axis=1)
    flat = sorted_rows.ravel()
    row_of = np.repeat(np.arange(n_windows), length)
    boundary = np.ones(flat.shape[0], dtype=bool)
    boundary[1:] = (flat[1:] != flat[:-1]) | (row_of[1:] != row_of[:-1])
    starts = np.nonzero(boundary)[0]
    run_lengths = np.diff(np.append(starts, flat.shape[0]))
    p = run_lengths / length
    contribution = -p * np.log2(p)
    entropy = np.zeros(n_windows)
    np.add.at(entropy, row_of[starts], contribution)
    unique = np.bincount(row_of[starts], minlength=n_windows)
    return entropy.astype(np.float32), unique.astype(np.uint16)


def build_windows(
    segment: pd.DataFrame,
    source: str,
    split: str,
    global_offset: int,
    window: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
) -> pd.DataFrame | None:
    """Aggregate one split segment of one file into sliding windows.

    ``global_offset`` is the absolute row index (in the concatenated
    messages table) of the segment's first row, so each window records
    where its raw messages live.
    """
    n = len(segment)
    if n < window:
        return None

    def view(column: str) -> np.ndarray:
        arr = segment[column].to_numpy()
        return np.lib.stride_tricks.sliding_window_view(arr, window)[::stride]

    ids = view("can_id")
    entropy, unique = _entropy_and_unique(ids)

    dt_prev = view("dt_prev")
    rel_dev = view("rel_dev")
    labels = view("label")
    unknown = view("is_unknown")
    ts = view("timestamp")

    out = pd.DataFrame({
        "id_entropy": entropy,
        "unique_id_count": unique,
        "unknown_ratio": unknown.mean(axis=1, dtype=np.float32),
        "mean_dt": dt_prev.mean(axis=1, dtype=np.float32),
        "std_dt": dt_prev.std(axis=1, dtype=np.float32),
        "mean_rel_dev": rel_dev.mean(axis=1, dtype=np.float32),
        "mean_abs_rel_dev": np.abs(rel_dev).mean(axis=1, dtype=np.float32),
        "max_abs_rel_dev": np.abs(rel_dev).max(axis=1).astype(np.float32),
        "injected_ratio": labels.mean(axis=1, dtype=np.float32),
        "t_start": ts[:, 0],
        "t_end": ts[:, -1],
    })
    out["label"] = (out["injected_ratio"] > 0).astype(np.uint8)
    out["attack_type"] = np.where(out["label"] == 1, source, "normal")
    out["source"] = source
    out["split"] = split
    out["start_row"] = (
        global_offset + np.arange(0, n - window + 1, stride)
    ).astype(np.int64)
    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> dict:
    """Build message- and window-level feature tables for all sources."""
    frames = {
        source: pd.read_parquet(PROCESSED_DIR / f"{source}.parquet")
        for source in SOURCES
    }

    print("Pass 1: learning ID periods from attack-free train traffic ...")
    stats = compute_train_stats(frames)
    print(f"  known IDs: {len(stats['known_ids'])}, "
          f"global median period: {stats['global_median_period']*1000:.2f} ms")
    with open(PROCESSED_DIR / "train_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("Pass 2+3: per-message features and sliding windows ...")
    message_parts: list[pd.DataFrame] = []
    window_parts: list[pd.DataFrame] = []
    offset = 0
    for source, df in frames.items():
        df = add_message_features(df, stats)
        boundaries = split_boundaries(len(df))
        df["split"] = ""
        for split in SPLIT_NAMES:
            start, end = boundaries[split]
            df.iloc[start:end, df.columns.get_loc("split")] = split
            windows = build_windows(
                df.iloc[start:end], source, split, offset + start
            )
            if windows is not None:
                window_parts.append(windows)
        message_parts.append(df)
        print(f"  [{source}] {len(df):,} messages, "
              f"{sum(len(w) for w in window_parts[-3:]):,} windows (3 splits)")
        offset += len(df)

    messages = pd.concat(message_parts, ignore_index=True)
    messages["split"] = messages["split"].astype("category")
    messages.to_parquet(PROCESSED_DIR / "messages.parquet", index=False)

    windows = pd.concat(window_parts, ignore_index=True)
    for column in ("attack_type", "source", "split"):
        windows[column] = windows[column].astype("category")
    windows.to_parquet(PROCESSED_DIR / "windows.parquet", index=False)

    # Class balance of windows per split / source
    summary: dict = {
        "window_size": WINDOW_SIZE, "window_stride": WINDOW_STRIDE,
        "n_messages": int(len(messages)), "n_windows": int(len(windows)),
        "splits": {},
    }
    for split in SPLIT_NAMES:
        split_windows = windows[windows["split"] == split]
        per_source = {}
        for source in SOURCES:
            sw = split_windows[split_windows["source"] == source]
            per_source[source] = {
                "windows": int(len(sw)),
                "attack_windows": int(sw["label"].sum()),
                "attack_ratio": round(float(sw["label"].mean()), 4) if len(sw) else 0.0,
            }
        summary["splits"][split] = per_source

    with open(REPORTS_DIR / "window_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'split':<7} {'source':<8} {'windows':>9} {'attack':>9} {'ratio':>7}")
    for split in SPLIT_NAMES:
        for source in SOURCES:
            s = summary["splits"][split][source]
            print(f"{split:<7} {source:<8} {s['windows']:>9,} "
                  f"{s['attack_windows']:>9,} {s['attack_ratio']:>7.1%}")
    print(f"\nmessages -> {PROCESSED_DIR / 'messages.parquet'}")
    print(f"windows  -> {PROCESSED_DIR / 'windows.parquet'}")
    return summary


if __name__ == "__main__":
    run_pipeline()
