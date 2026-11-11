"""CAN log replay engine.

Replays the TEST segment of a chosen capture as a live stream: virtual time
advances at a configurable speed (1x/2x/5x/10x), with pause/resume and
seeking. Every ``SCORE_EVERY_S`` virtual seconds the engine scores the last
``WINDOW_SIZE`` messages with the selected model, using the same feature
math as the offline evaluation.

The engine is transport-agnostic: :meth:`advance` returns the frames and
detections that happened since the last call, and the WebSocket layer in
:mod:`src.api.main` decides how to ship them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, WINDOW_SIZE
from src.api.models_runtime import get_runtime
from src.models.common import MAX_CAN_ID

SOURCES = ("normal", "dos", "fuzzy", "gear", "rpm")
BYTE_COLUMNS = [f"byte{i}" for i in range(8)]

# CAN 2.0A frame ~ 44 overhead bits + 8 bits per payload byte (+ stuffing,
# ignored): used for the bus-load estimate at 500 kbit/s.
BITS_OVERHEAD = 44
BUS_BITRATE = 500_000.0

# Virtual seconds between consecutive window scorings during replay
SCORE_EVERY_S = 0.25


def load_replay_frame(source: str) -> pd.DataFrame:
    """Test-split messages of one capture, with precomputed features."""
    messages = pd.read_parquet(PROCESSED_DIR / "messages.parquet")
    frame = messages[
        (messages["source"] == source) & (messages["split"] == "test")
    ].reset_index(drop=True)
    frame["rel_time"] = frame["timestamp"] - frame["timestamp"].iloc[0]
    return frame


class ReplayEngine:
    """Stateful replay of one capture's test segment."""

    def __init__(self, source: str = "rpm",
                 model: str = "lstm_autoencoder") -> None:
        if source not in SOURCES:
            raise ValueError(f"unknown source '{source}'")
        self.source = source
        self.model = model
        self.frame = load_replay_frame(source)
        self.duration = float(self.frame["rel_time"].iloc[-1])
        self.speed = 1.0
        self.playing = False
        self.position = 0.0            # virtual seconds from segment start
        self._cursor = 0               # next row index to emit
        self._next_score_t = 0.0       # virtual time of the next scoring
        self._runtime = get_runtime()
        self._times = self.frame["rel_time"].to_numpy()

        # Pre-extract arrays used when building windows (fast slicing)
        f = self.frame
        self._dt_prev = f["dt_prev"].to_numpy(np.float64)
        self._rel_dev = f["rel_dev"].to_numpy(np.float64)
        self._unknown = f["is_unknown"].to_numpy(np.float64)
        self._ids = f["can_id"].to_numpy()
        self._dt_same = f["dt_same_id"].to_numpy(np.float64)
        self._payload_mean = f[BYTE_COLUMNS].to_numpy(np.float32).mean(axis=1)
        self._dlc = f["dlc"].to_numpy()

    # -- controls ----------------------------------------------------------
    def play(self) -> None:
        self.playing = True

    def pause(self) -> None:
        self.playing = False

    def set_speed(self, speed: float) -> None:
        self.speed = float(np.clip(speed, 0.25, 50.0))

    def seek(self, position_s: float) -> None:
        """Jump to a virtual time; scoring restarts from that point."""
        self.position = float(np.clip(position_s, 0.0, self.duration))
        self._cursor = int(np.searchsorted(self._times, self.position))
        self._next_score_t = self.position

    # -- window scoring ----------------------------------------------------
    def _score_window(self, end_row: int) -> dict:
        rows = slice(end_row - WINDOW_SIZE, end_row)
        ids = self._ids[rows]
        _, counts = np.unique(ids, return_counts=True)
        p = counts / counts.sum()
        features = np.array([
            float(-(p * np.log2(p)).sum()), float(len(counts)),
            float(self._unknown[rows].mean()),
            float(self._dt_prev[rows].mean()), float(self._dt_prev[rows].std()),
            float(self._rel_dev[rows].mean()),
            float(np.abs(self._rel_dev[rows]).mean()),
            float(np.abs(self._rel_dev[rows]).max()),
        ], dtype=np.float32)

        sequence = np.empty((1, WINDOW_SIZE, 6), dtype=np.float32)
        sequence[0, :, 0] = np.log1p(self._dt_prev[rows] * 1e3)
        sequence[0, :, 1] = np.log1p(self._dt_same[rows] * 1e3)
        sequence[0, :, 2] = self._rel_dev[rows] / 10.0
        sequence[0, :, 3] = self._unknown[rows]
        sequence[0, :, 4] = self._ids[rows] / MAX_CAN_ID
        sequence[0, :, 5] = self._payload_mean[rows] / 255.0

        result = self._runtime.score_window(features, sequence, self.model)
        result["t"] = round(float(self._times[end_row - 1]), 4)
        # Suspect ID for the direct label: floods/fuzzing inject unknown IDs
        # (most frequent unknown), spoofing compresses the period of a known
        # ID (most negative deviation = arriving far too fast).
        unknown_mask = self._unknown[rows] == 1
        if unknown_mask.any():
            values, counts = np.unique(ids[unknown_mask], return_counts=True)
            suspect = int(values[int(np.argmax(counts))])
        else:
            suspect = int(ids[int(np.argmin(self._rel_dev[rows]))])
        result["suspect_id"] = f"{suspect:03X}"
        # Ground truth for the dashboard's shaded attack periods
        labels = self.frame["label"].to_numpy()[rows]
        result["true_attack"] = bool(labels.any())
        return result

    # -- streaming ---------------------------------------------------------
    def advance(self, wall_dt_s: float, max_frames: int = 400) -> dict:
        """Advance virtual time and return frames + detections since last call."""
        detections: list[dict] = []
        frames: list[dict] = []
        if self.playing:
            self.position = min(self.position + wall_dt_s * self.speed,
                                self.duration)
            end = int(np.searchsorted(self._times, self.position, side="right"))
            new_rows = range(self._cursor, end)

            # Score on a fixed virtual-time cadence (keeps the detection
            # stream ~4 Hz virtual instead of one per 32 messages ~60 Hz)
            while self._next_score_t <= self.position:
                end_row = int(np.searchsorted(
                    self._times, self._next_score_t, side="right"))
                if end_row >= WINDOW_SIZE:
                    detections.append(self._score_window(end_row))
                self._next_score_t += SCORE_EVERY_S

            # Ship at most max_frames raw frames (newest kept), so high
            # speeds don't flood the socket; stats cover ALL new rows.
            visible = list(new_rows)[-max_frames:]
            f = self.frame
            for i in visible:
                row = f.iloc[i]
                frames.append({
                    "t": round(float(row["rel_time"]), 4),
                    "id": f"{int(row['can_id']):03X}",
                    "dlc": int(row["dlc"]),
                    "data": " ".join(f"{int(row[b]):02x}"
                                     for b in BYTE_COLUMNS[:int(row["dlc"])]),
                    "label": int(row["label"]),
                })
            stats = self._bus_stats(self._cursor, end, wall_dt_s)
            heatmap = self._heatmap_bucket(self._cursor, end)
            self._cursor = end
            if self.position >= self.duration:
                self.playing = False
        else:
            stats, heatmap = None, None

        return {
            "frames": frames,
            "detections": detections,
            "stats": stats,
            "heatmap": heatmap,
            "status": {
                "playing": self.playing,
                "position": round(self.position, 3),
                "duration": round(self.duration, 3),
                "speed": self.speed,
                "source": self.source,
                "model": self.model,
            },
        }

    def _bus_stats(self, start: int, end: int, wall_dt_s: float) -> dict | None:
        if end <= start:
            return None
        n = end - start
        span = max(self._times[end - 1] - self._times[start], 1e-6)
        bits = float(BITS_OVERHEAD * n + 8 * self._dlc[start:end].sum())
        return {
            "frames_per_s": round(n / span),
            "bus_load_pct": round(100.0 * bits / span / BUS_BITRATE, 1),
            "active_ids": int(len(np.unique(self._ids[start:end]))),
        }

    def _heatmap_bucket(self, start: int, end: int) -> dict | None:
        """Mean period deviation per CAN ID over the new rows."""
        if end <= start:
            return None
        ids = self._ids[start:end]
        rel = self._rel_dev[start:end]
        known = self._unknown[start:end] == 0
        out: dict[str, float] = {}
        for can_id in np.unique(ids[known]):
            mask = ids == can_id
            out[f"{int(can_id):03X}"] = round(float(np.mean(rel[mask])), 3)
        unknown_count = int((~known).sum())
        return {"t": round(float(self._times[end - 1]), 2),
                "by_id": out, "unknown_frames": unknown_count}
