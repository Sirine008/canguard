"""Parsers for the HCRL Car-Hacking Dataset (Seo et al., 2018).

Two raw formats are handled:

* Attack CSVs (``DoS_dataset.csv`` etc.) — headerless rows of
  ``timestamp, can_id(hex), dlc, data[0..dlc-1], flag`` where the flag
  (``T`` = injected, ``R`` = normal) shifts position for rows with DLC < 8.
* ``normal_run_data.txt`` — a text log of
  ``Timestamp: <ts>  ID: <hex>  000  DLC: <n>  <hex bytes>`` lines.

Each file is normalised into one parquet file with the unified schema::

    timestamp (float64) | can_id (uint16) | dlc (uint8) |
    byte0..byte7 (uint8, zero-filled beyond dlc) |
    label (uint8: 1 = injected) | attack_type (category) | source (category)

Each attack file also contains one very large timestamp gap near its end
(documented by Verma et al. in the ROAD paper); the gap is detected
automatically and everything from the gap onward is dropped.

Run as a module to parse everything::

    python -m src.data.parser
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ATTACK_FILES,
    GAP_THRESHOLD_S,
    NORMAL_FILE,
    PROCESSED_DIR,
    RAW_DIR,
    REPORTS_DIR,
    SUBSAMPLE_FRACTION,
)

BYTE_COLUMNS: list[str] = [f"byte{i}" for i in range(8)]

# Fast hex-string -> int lookup for two-digit payload bytes ("00".."ff").
_HEX_BYTE_MAP: dict[str, int] = {f"{i:02x}": i for i in range(256)}
_HEX_BYTE_MAP.update({f"{i:02X}": i for i in range(256)})


def _hex_series_to_uint8(series: pd.Series) -> np.ndarray:
    """Convert a Series of two-digit hex strings to uint8 (NaN -> 0)."""
    return (
        series.map(_HEX_BYTE_MAP)
        .fillna(0)
        .astype(np.uint8)
        .to_numpy()
    )


def _can_id_to_uint16(series: pd.Series) -> np.ndarray:
    """Convert hex CAN identifiers (e.g. ``"0316"``) to uint16."""
    unique = series.unique()
    mapping = {value: int(value, 16) for value in unique}
    return series.map(mapping).astype(np.uint16).to_numpy()


def detect_timestamp_gap(
    timestamps: np.ndarray, threshold_s: float = GAP_THRESHOLD_S
) -> tuple[int | None, float]:
    """Find the first inter-message gap larger than ``threshold_s``.

    Returns ``(index, gap_seconds)`` where ``index`` is the position of the
    first message *after* the gap (i.e. the file should be trimmed to
    ``[:index]``), or ``(None, 0.0)`` when no such gap exists.
    """
    diffs = np.diff(timestamps)
    gap_positions = np.nonzero(diffs > threshold_s)[0]
    if gap_positions.size == 0:
        return None, 0.0
    first = int(gap_positions[0])
    return first + 1, float(diffs[first])


def _trim_and_subsample(
    df: pd.DataFrame, name: str, stats: dict
) -> pd.DataFrame:
    """Apply gap trimming and optional time-ordered subsampling."""
    trim_index, gap_s = detect_timestamp_gap(df["timestamp"].to_numpy())
    stats["gap_seconds"] = round(gap_s, 1)
    stats["rows_before_trim"] = len(df)
    if trim_index is not None:
        df = df.iloc[:trim_index].reset_index(drop=True)
        print(f"  [{name}] trimmed at a {gap_s:.0f} s timestamp gap: "
              f"{stats['rows_before_trim']:,} -> {len(df):,} rows")
    stats["rows_after_trim"] = len(df)

    if SUBSAMPLE_FRACTION < 1.0:
        keep = int(len(df) * SUBSAMPLE_FRACTION)
        df = df.iloc[:keep].reset_index(drop=True)
        print(f"  [{name}] subsampled (time-ordered) to {len(df):,} rows")
    stats["rows_final"] = len(df)
    return df


def parse_attack_csv(path: Path, attack_type: str) -> tuple[pd.DataFrame, dict]:
    """Parse one headerless attack CSV into the unified schema.

    Rows with DLC < 8 carry fewer payload columns, which shifts the R/T
    flag into an earlier column; the flag is therefore located per row at
    position ``dlc`` within the 9 trailing columns.
    """
    stats: dict = {"file": path.name}
    names = ["timestamp", "can_id", "dlc"] + BYTE_COLUMNS + ["flag"]
    dtypes: dict = {name: str for name in names}
    dtypes["timestamp"] = np.float64
    dtypes["dlc"] = np.uint8
    df = pd.read_csv(
        path, header=None, names=names, dtype=dtypes, skipinitialspace=True
    )

    # Trailing 9 columns as one object matrix: for a row with DLC = k the
    # payload occupies positions 0..k-1 and the flag sits at position k.
    trailing = df[BYTE_COLUMNS + ["flag"]].to_numpy(dtype=object)
    dlc = df["dlc"].to_numpy()
    flags = np.take_along_axis(
        trailing, dlc.astype(np.int64)[:, None], axis=1
    ).ravel()
    # Zero out positions at/after the flag so short payloads are 0-filled.
    column_index = np.arange(8)[None, :]
    byte_matrix = np.where(
        column_index < dlc[:, None], trailing[:, :8], "00"
    )

    out = pd.DataFrame({"timestamp": df["timestamp"].to_numpy()})
    out["can_id"] = _can_id_to_uint16(df["can_id"])
    out["dlc"] = dlc
    for i, column in enumerate(BYTE_COLUMNS):
        out[column] = _hex_series_to_uint8(pd.Series(byte_matrix[:, i]))
    out["label"] = (flags == "T").astype(np.uint8)

    out = _trim_and_subsample(out, attack_type, stats)

    out["attack_type"] = pd.Categorical(
        np.where(out["label"] == 1, attack_type, "normal")
    )
    out["source"] = pd.Categorical([attack_type] * len(out))

    stats["injected_rows"] = int(out["label"].sum())
    stats["injected_ratio"] = round(float(out["label"].mean()), 4)
    stats["duration_s"] = round(
        float(out["timestamp"].iloc[-1] - out["timestamp"].iloc[0]), 1
    )
    stats["unique_ids"] = int(out["can_id"].nunique())
    return out, stats


_NORMAL_LINE_PATTERN = (
    r"Timestamp:\s*(?P<timestamp>[\d.]+)\s+"
    r"ID:\s*(?P<can_id>[0-9a-fA-F]+)\s+\d+\s+"
    r"DLC:\s*(?P<dlc>\d+)\s*(?P<payload>[0-9a-fA-F ]*)"
)


def parse_normal_txt(path: Path) -> tuple[pd.DataFrame, dict]:
    """Parse the attack-free ``normal_run_data.txt`` log."""
    stats: dict = {"file": path.name}
    lines = pd.read_csv(
        path, header=None, names=["line"], sep="\x01", engine="c",
        skip_blank_lines=True, dtype=str,
    )["line"]
    extracted = lines.str.extract(_NORMAL_LINE_PATTERN)
    malformed = int(extracted["timestamp"].isna().sum())
    if malformed:
        print(f"  [normal] dropped {malformed} malformed line(s)")
    extracted = extracted.dropna(subset=["timestamp"]).reset_index(drop=True)
    stats["malformed_lines"] = malformed

    out = pd.DataFrame(
        {"timestamp": extracted["timestamp"].astype(np.float64)}
    )
    out["can_id"] = _can_id_to_uint16(extracted["can_id"])
    out["dlc"] = extracted["dlc"].astype(np.uint8)

    payload = extracted["payload"].str.strip().str.split(" ", expand=True)
    payload = payload.reindex(columns=range(8))
    for i, column in enumerate(BYTE_COLUMNS):
        out[column] = _hex_series_to_uint8(payload[i])
    out["label"] = np.uint8(0)

    out = _trim_and_subsample(out, "normal", stats)

    out["attack_type"] = pd.Categorical(["normal"] * len(out))
    out["source"] = pd.Categorical(["normal"] * len(out))

    stats["injected_rows"] = 0
    stats["injected_ratio"] = 0.0
    stats["duration_s"] = round(
        float(out["timestamp"].iloc[-1] - out["timestamp"].iloc[0]), 1
    )
    stats["unique_ids"] = int(out["can_id"].nunique())
    return out, stats


def parse_all(raw_dir: Path = RAW_DIR, out_dir: Path = PROCESSED_DIR) -> dict:
    """Parse every raw file into ``<out_dir>/<name>.parquet``.

    Returns the per-file statistics dictionary (also written to
    ``reports/parse_summary.json``).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict = {}

    normal_path = raw_dir / NORMAL_FILE
    if not normal_path.exists():
        raise FileNotFoundError(
            f"Missing {normal_path}. Download the HCRL Car-Hacking Dataset "
            "and place its files in data/raw/car_hacking/."
        )

    for attack_type, filename in ATTACK_FILES.items():
        path = raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing raw file: {path}")
        print(f"Parsing {filename} ...")
        start = time.perf_counter()
        df, stats = parse_attack_csv(path, attack_type)
        df.to_parquet(out_dir / f"{attack_type}.parquet", index=False)
        stats["parse_seconds"] = round(time.perf_counter() - start, 1)
        summary[attack_type] = stats
        print(f"  [{attack_type}] {len(df):,} rows "
              f"({stats['injected_ratio']:.1%} injected) "
              f"in {stats['parse_seconds']} s")

    print(f"Parsing {NORMAL_FILE} ...")
    start = time.perf_counter()
    df, stats = parse_normal_txt(normal_path)
    df.to_parquet(out_dir / "normal.parquet", index=False)
    stats["parse_seconds"] = round(time.perf_counter() - start, 1)
    summary["normal"] = stats
    print(f"  [normal] {len(df):,} rows in {stats['parse_seconds']} s")

    with open(REPORTS_DIR / "parse_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {REPORTS_DIR / 'parse_summary.json'}")
    return summary


if __name__ == "__main__":
    parse_all()
