"""Isolation Forest baseline (unsupervised, trained on normal windows only).

The model consumes window-level features. Its raw anomaly score is
``-decision_function`` (higher = more anomalous); normalisation to [0, 1]
is handled uniformly for all unsupervised models in :mod:`evaluate`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.models.common import FEATURE_SETS


def train_isolation_forest(
    train_normal: pd.DataFrame, seed: int, feature_set: str = "all",
) -> IsolationForest:
    """Fit an Isolation Forest on attack-free train windows."""
    features = FEATURE_SETS[feature_set]
    model = IsolationForest(
        n_estimators=200,
        max_samples=1024,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(train_normal[features].to_numpy(np.float32))
    return model


def score_windows(
    model: IsolationForest, windows: pd.DataFrame, feature_set: str = "all",
) -> np.ndarray:
    """Raw anomaly score per window (higher = more anomalous)."""
    features = FEATURE_SETS[feature_set]
    return -model.decision_function(windows[features].to_numpy(np.float32))
