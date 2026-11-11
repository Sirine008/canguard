"""Train all Phase 3 models over the evaluation seeds.

Trains, for each seed in ``EVAL_SEEDS``:

* Isolation Forest (window features, attack-free train windows only),
* LSTM autoencoder (message sequences, attack-free train windows only),
* LightGBM multi-class supervised comparison (all train windows).

Saves models to ``models/`` plus ``models/train_times.json``, and exports
the first seed's autoencoder to ONNX.

Run::

    python -m src.models.train
"""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
import torch

from src.config import EVAL_SEEDS, MODELS_DIR
from src.models.autoencoder import export_onnx, train_autoencoder
from src.models.common import (
    gather_sequences,
    load_message_matrix,
    load_windows,
    select,
)
from src.models.isolation_forest import train_isolation_forest
from src.models.supervised import train_lightgbm


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    windows = load_windows()
    train_all = select(windows, "train")
    train_normal = select(windows, "train", label=0)
    val_all = select(windows, "val")
    val_normal = select(windows, "val", label=0)

    print(f"train windows: {len(train_all):,} "
          f"({len(train_normal):,} attack-free) | val: {len(val_all):,}")

    print("Loading message sequences for the autoencoder ...")
    message_matrix = load_message_matrix()
    seq_train_normal = gather_sequences(
        message_matrix, train_normal["start_row"].to_numpy()
    )
    seq_val_normal = gather_sequences(
        message_matrix, val_normal["start_row"].to_numpy()
    )
    del message_matrix

    times: dict = {}
    for seed in EVAL_SEEDS:
        print(f"\n=== seed {seed} ===")
        entry: dict = {}

        start = time.perf_counter()
        forest = train_isolation_forest(train_normal, seed)
        entry["isolation_forest_s"] = round(time.perf_counter() - start, 2)
        joblib.dump(forest, MODELS_DIR / f"isolation_forest_seed{seed}.joblib")
        print(f"  isolation forest: {entry['isolation_forest_s']} s")

        start = time.perf_counter()
        booster = train_lightgbm(train_all, val_all, seed)
        entry["lightgbm_s"] = round(time.perf_counter() - start, 2)
        joblib.dump(booster, MODELS_DIR / f"lightgbm_seed{seed}.joblib")
        print(f"  lightgbm: {entry['lightgbm_s']} s "
              f"({booster.best_iteration_} trees)")

        print("  lstm autoencoder:")
        model, stats = train_autoencoder(seq_train_normal, seq_val_normal, seed)
        entry["lstm_ae_s"] = stats["train_time_s"]
        entry["lstm_ae_epochs"] = stats["epochs_run"]
        entry["lstm_ae_best_val_loss"] = round(stats["best_val_loss"], 6)
        torch.save(model.state_dict(), MODELS_DIR / f"lstm_ae_seed{seed}.pt")

        times[str(seed)] = entry

    # ONNX export of the first seed's autoencoder (used by Phases 5/6)
    from src.models.autoencoder import LSTMAutoencoder
    model = LSTMAutoencoder()
    model.load_state_dict(
        torch.load(MODELS_DIR / f"lstm_ae_seed{EVAL_SEEDS[0]}.pt",
                   weights_only=True)
    )
    model.eval()
    export_onnx(model, str(MODELS_DIR / "lstm_ae.onnx"))
    print(f"\nONNX export -> {MODELS_DIR / 'lstm_ae.onnx'}")

    with open(MODELS_DIR / "train_times.json", "w", encoding="utf-8") as f:
        json.dump(times, f, indent=2)
    print(f"training times -> {MODELS_DIR / 'train_times.json'}")


if __name__ == "__main__":
    main()
