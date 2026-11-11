"""Central configuration for the canguard project.

All paths, seeds and tunable parameters live here so every script and
notebook stays reproducible and consistent.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw" / "car_hacking"
ROAD_DIR: Path = DATA_DIR / "raw" / "road"          # optional second benchmark
PROCESSED_DIR: Path = DATA_DIR / "processed"

MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
KNOWLEDGE_BASE_DIR: Path = PROJECT_ROOT / "knowledge_base"

# Raw file names (HCRL Car-Hacking Dataset, Seo et al. 2018)
ATTACK_FILES: dict[str, str] = {
    "dos": "DoS_dataset.csv",
    "fuzzy": "Fuzzy_dataset.csv",
    "gear": "gear_dataset.csv",
    "rpm": "RPM_dataset.csv",
}
NORMAL_FILE: str = "normal_run_data.txt"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED: int = 42
EVAL_SEEDS: tuple[int, ...] = (42, 43, 44)  # mean +/- std reported over these

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
# Each attack file contains one very large timestamp gap near the end
# (Verma et al., ROAD paper: ~2328 s DoS, ~2466 s Fuzzy, ~1949 s gear,
# ~1952 s RPM). Any inter-message gap above this threshold is treated as
# a recording discontinuity and the file is trimmed just before it.
GAP_THRESHOLD_S: float = 60.0

# Optional time-ordered subsampling (1.0 = keep everything). Reduce if the
# machine runs out of memory: 0.25 keeps the first quarter of each file.
SUBSAMPLE_FRACTION: float = 1.0

# ---------------------------------------------------------------------------
# Features (Phase 2)
# ---------------------------------------------------------------------------
WINDOW_SIZE: int = 64          # messages per sliding window
WINDOW_STRIDE: int = 32        # stride between consecutive windows

# Time-ordered split fractions (train / val / test)
SPLIT_FRACTIONS: tuple[float, float, float] = (0.6, 0.2, 0.2)

# ---------------------------------------------------------------------------
# Attack type encoding (shared across models, figures and the dashboard)
# ---------------------------------------------------------------------------
ATTACK_TYPES: tuple[str, ...] = ("normal", "dos", "fuzzy", "gear", "rpm")
ATTACK_TYPE_TO_ID: dict[str, int] = {name: i for i, name in enumerate(ATTACK_TYPES)}
