"""Online (streaming) feature computation for the API and simulator.

Mirrors the offline pipeline in :mod:`src.data.features` exactly, but in a
stateful, message-by-message form suitable for a live stream or a single
POSTed window. Uses the same train statistics (``train_stats.json``) — no
leakage: those were learned on attack-free train traffic only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from src.config import PROCESSED_DIR
from src.data.features import UNKNOWN_ID_DEV
from src.models.common import MAX_CAN_ID


def load_train_stats() -> dict:
    with open(PROCESSED_DIR / "train_stats.json", encoding="utf-8") as f:
        stats = json.load(f)
    stats["period_by_id"] = {int(k): v for k, v in stats["period_by_id"].items()}
    return stats


@dataclass
class LiveMessage:
    """One CAN frame with its computed causal features."""

    timestamp: float
    can_id: int
    dlc: int
    payload: list[int]
    dt_prev: float
    dt_same_id: float
    rel_dev: float
    is_unknown: int


class MessageFeaturizer:
    """Stateful per-message feature computation (same math as offline)."""

    def __init__(self, stats: dict | None = None) -> None:
        self.stats = stats or load_train_stats()
        self._last_ts: float | None = None
        self._last_ts_by_id: dict[int, float] = {}

    def feed(self, timestamp: float, can_id: int, dlc: int,
             payload: list[int]) -> LiveMessage:
        period = self.stats["period_by_id"].get(can_id)
        is_unknown = int(period is None)

        dt_prev = 0.0 if self._last_ts is None else timestamp - self._last_ts
        self._last_ts = timestamp

        last_same = self._last_ts_by_id.get(can_id)
        if last_same is None:
            dt_same_id = period if period is not None \
                else self.stats["global_median_period"]
        else:
            dt_same_id = timestamp - last_same
        self._last_ts_by_id[can_id] = timestamp

        if period is None:
            rel_dev = UNKNOWN_ID_DEV
        else:
            rel_dev = float(np.clip((dt_same_id - period) / period,
                                    -1.0, UNKNOWN_ID_DEV))
        return LiveMessage(
            timestamp=timestamp, can_id=can_id, dlc=dlc, payload=payload,
            dt_prev=dt_prev, dt_same_id=dt_same_id,
            rel_dev=rel_dev, is_unknown=is_unknown,
        )


def window_feature_vector(messages: list[LiveMessage]) -> np.ndarray:
    """Window aggregates in ``FEATURE_SETS['all']`` order.

    [id_entropy, unique_id_count, unknown_ratio,
     mean_dt, std_dt, mean_rel_dev, mean_abs_rel_dev, max_abs_rel_dev]
    """
    ids = np.array([m.can_id for m in messages])
    dt_prev = np.array([m.dt_prev for m in messages], dtype=np.float64)
    rel_dev = np.array([m.rel_dev for m in messages], dtype=np.float64)
    unknown = np.array([m.is_unknown for m in messages], dtype=np.float64)

    _, counts = np.unique(ids, return_counts=True)
    p = counts / counts.sum()
    entropy = float(-(p * np.log2(p)).sum())

    return np.array([
        entropy, float(len(counts)), float(unknown.mean()),
        float(dt_prev.mean()), float(dt_prev.std()),
        float(rel_dev.mean()), float(np.abs(rel_dev).mean()),
        float(np.abs(rel_dev).max()),
    ], dtype=np.float32)


def sequence_tensor(messages: list[LiveMessage]) -> np.ndarray:
    """(1, window, 6) tensor in ``SEQUENCE_FEATURES`` order for the AE."""
    out = np.empty((1, len(messages), 6), dtype=np.float32)
    for i, m in enumerate(messages):
        payload_mean = float(np.mean(m.payload)) if m.payload else 0.0
        out[0, i] = (
            np.log1p(m.dt_prev * 1e3),
            np.log1p(m.dt_same_id * 1e3),
            m.rel_dev / 10.0,
            float(m.is_unknown),
            m.can_id / MAX_CAN_ID,
            payload_mean / 255.0,
        )
    return out
