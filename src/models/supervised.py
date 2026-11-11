"""Supervised multi-class comparison model (LightGBM).

Classifies each window as normal / DoS / fuzzy / gear / RPM from the
window-level features. LightGBM was chosen over a 1D-CNN for the supervised
baseline because it trains in seconds on CPU for this table size and ships
tiny models; the deep-learning component of the project is the LSTM
autoencoder.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

from src.config import ATTACK_TYPE_TO_ID
from src.models.common import FEATURE_SETS


def encode_labels(windows: pd.DataFrame) -> np.ndarray:
    """Map window attack types to integer class ids (normal = 0)."""
    return windows["attack_type"].astype(str).map(ATTACK_TYPE_TO_ID).to_numpy()


def train_lightgbm(
    train_windows: pd.DataFrame,
    val_windows: pd.DataFrame,
    seed: int,
    feature_set: str = "all",
) -> LGBMClassifier:
    """Train the multi-class model with early stopping on the val split."""
    features = FEATURE_SETS[feature_set]
    model = LGBMClassifier(
        objective="multiclass",
        num_class=len(ATTACK_TYPE_TO_ID),
        n_estimators=300,
        learning_rate=0.1,
        num_leaves=31,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        train_windows[features].to_numpy(np.float32),
        encode_labels(train_windows),
        eval_set=[(val_windows[features].to_numpy(np.float32),
                   encode_labels(val_windows))],
        callbacks=[early_stopping(20, verbose=False), log_evaluation(0)],
    )
    return model


def predict_classes(
    model: LGBMClassifier, windows: pd.DataFrame, feature_set: str = "all",
) -> np.ndarray:
    """Predicted class id per window."""
    features = FEATURE_SETS[feature_set]
    return model.predict(windows[features].to_numpy(np.float32))


def predict_attack_probability(
    model: LGBMClassifier, windows: pd.DataFrame, feature_set: str = "all",
) -> np.ndarray:
    """P(any attack) per window = 1 - P(normal), for ROC comparison."""
    features = FEATURE_SETS[feature_set]
    proba = model.predict_proba(windows[features].to_numpy(np.float32))
    return 1.0 - proba[:, ATTACK_TYPE_TO_ID["normal"]]
