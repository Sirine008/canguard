"""Runtime model registry for the API and simulator.

Loads the trained artifacts once (Isolation Forest + LightGBM via joblib,
the LSTM autoencoder via ONNX Runtime) together with the score
normalisation constants and validation-chosen thresholds, and exposes a
single ``score_window`` call.
"""

from __future__ import annotations

import json
import time
from functools import lru_cache

import joblib
import numpy as np
import onnxruntime as ort

from src.config import ATTACK_TYPES, EVAL_SEEDS, MODELS_DIR

MODEL_VERSIONS = {
    "isolation_forest": "iforest v1.0",
    "lstm_autoencoder": "lstm-ae v1.0",
    "lightgbm": "lightgbm v1.0",
}


class ModelRuntime:
    """All trained models plus thresholds, ready for single-window scoring."""

    def __init__(self) -> None:
        seed = EVAL_SEEDS[0]
        # Project-generated artifacts (trusted; created by src.models.train)
        self.forest = joblib.load(MODELS_DIR / f"isolation_forest_seed{seed}.joblib")
        self.lgbm = joblib.load(MODELS_DIR / f"lightgbm_seed{seed}.joblib")
        self.ae = ort.InferenceSession(
            str(MODELS_DIR / "lstm_ae.onnx"),
            providers=["CPUExecutionProvider"],
        )
        with open(MODELS_DIR / "score_norm.json", encoding="utf-8") as f:
            calibration = json.load(f)
        self.norm = calibration["norm"]
        self.thresholds = {
            name: entry["fpr_target"]
            for name, entry in calibration["thresholds"].items()
        }

    def _normalise(self, name: str, raw: float) -> float:
        lo, hi = self.norm[name]["lo"], self.norm[name]["hi"]
        return float(np.clip((raw - lo) / (hi - lo), 0.0, 1.0))

    def score_window(
        self, features: np.ndarray, sequence: np.ndarray,
        model: str = "lstm_autoencoder",
    ) -> dict:
        """Score one window with the selected detector.

        Returns the normalised anomaly score in [0, 1], the alarm flag
        (validation threshold at 1 % FPR), the LightGBM attack-type
        prediction, and the inference latency in milliseconds.
        """
        start = time.perf_counter()
        if model == "isolation_forest":
            raw = float(-self.forest.decision_function(features[None, :])[0])
            score = self._normalise("isolation_forest", raw)
        elif model == "lstm_autoencoder":
            recon = self.ae.run(None, {"window": sequence})[0]
            raw = float(((recon - sequence) ** 2).mean())
            score = self._normalise("lstm_autoencoder", raw)
        elif model == "lightgbm":
            proba = self.lgbm.predict_proba(features[None, :])[0]
            score = float(1.0 - proba[0])
        else:
            raise ValueError(f"unknown model '{model}'")

        proba = self.lgbm.predict_proba(features[None, :])[0]
        predicted_class = ATTACK_TYPES[int(np.argmax(proba))]
        threshold = self.thresholds.get(model, 0.5)
        latency_ms = (time.perf_counter() - start) * 1e3

        return {
            "model": model,
            "score": round(score, 4),
            "threshold": round(float(threshold), 4),
            "is_attack": bool(score >= threshold),
            "attack_type": predicted_class if score >= threshold else "normal",
            "type_confidence": round(float(np.max(proba)), 3),
            "latency_ms": round(latency_ms, 3),
        }


@lru_cache(maxsize=1)
def get_runtime() -> ModelRuntime:
    return ModelRuntime()
