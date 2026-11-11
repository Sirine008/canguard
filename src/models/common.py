"""Shared data access for model training and evaluation.

Loads the window/message tables built in Phase 2 and provides:

* the window-level feature matrices (with named feature subsets for the
  ablation study),
* message-sequence tensors for the LSTM autoencoder (built by gathering
  each window's 64 raw messages and applying fixed, causal transforms).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, WINDOW_SIZE

# Window-level feature subsets (ablation: ID-distribution vs timing vs all)
FEATURE_SETS: dict[str, list[str]] = {
    "id": ["id_entropy", "unique_id_count", "unknown_ratio"],
    "timing": ["mean_dt", "std_dt", "mean_rel_dev",
               "mean_abs_rel_dev", "max_abs_rel_dev"],
}
FEATURE_SETS["all"] = FEATURE_SETS["id"] + FEATURE_SETS["timing"]

# Per-message sequence features for the LSTM-AE (fixed, causal transforms)
SEQUENCE_FEATURES: tuple[str, ...] = (
    "log_dt_prev", "log_dt_same_id", "rel_dev_n",
    "is_unknown", "id_n", "payload_mean",
)
MAX_CAN_ID: float = 0x7FF  # 11-bit identifier space


def load_windows() -> pd.DataFrame:
    """Load the window table (features + labels + split metadata)."""
    return pd.read_parquet(PROCESSED_DIR / "windows.parquet")


def load_message_matrix() -> np.ndarray:
    """Load all messages as an (N, 6) float32 matrix of sequence features.

    Column order follows ``SEQUENCE_FEATURES``. Transforms keep every value
    in a comparable [0, ~8] range without fitting anything (no leakage):
    log1p on millisecond inter-arrival times, ``rel_dev`` scaled by its cap,
    CAN ID scaled by the 11-bit maximum, payload mean scaled to [0, 1].
    """
    columns = ["dt_prev", "dt_same_id", "rel_dev", "is_unknown", "can_id"] + \
              [f"byte{i}" for i in range(8)]
    df = pd.read_parquet(PROCESSED_DIR / "messages.parquet", columns=columns)
    out = np.empty((len(df), len(SEQUENCE_FEATURES)), dtype=np.float32)
    out[:, 0] = np.log1p(df["dt_prev"].to_numpy(np.float32) * 1e3)
    out[:, 1] = np.log1p(df["dt_same_id"].to_numpy(np.float32) * 1e3)
    out[:, 2] = df["rel_dev"].to_numpy(np.float32) / 10.0
    out[:, 3] = df["is_unknown"].to_numpy(np.float32)
    out[:, 4] = df["can_id"].to_numpy(np.float32) / MAX_CAN_ID
    payload = df[[f"byte{i}" for i in range(8)]].to_numpy(np.float32)
    out[:, 5] = payload.mean(axis=1) / 255.0
    return out


def gather_sequences(
    message_matrix: np.ndarray, start_rows: np.ndarray,
    window: int = WINDOW_SIZE,
) -> np.ndarray:
    """Gather (n_windows, window, n_features) sequences by start row."""
    index = start_rows[:, None] + np.arange(window)[None, :]
    return message_matrix[index]


def select(
    windows: pd.DataFrame, split: str, label: int | None = None,
    source: str | None = None,
) -> pd.DataFrame:
    """Convenience filter over the window table."""
    mask = windows["split"] == split
    if label is not None:
        mask &= windows["label"] == label
    if source is not None:
        mask &= windows["source"] == source
    return windows[mask]
