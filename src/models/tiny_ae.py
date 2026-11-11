"""Tiny dense autoencoder for embedded deployment (Phase 6).

STM32Cube.AI (X-CUBE-AI) maps small feed-forward networks far more
efficiently than recurrent ones, so the embedded candidate is a dense
autoencoder over the 8 *window-level* features (not the message sequences):
8 -> 16 -> 4 -> 16 -> 8, ~700 parameters. It is trained on the same
attack-free train windows as the big models and evaluated the same way, so
the reported AUC is directly comparable.

Outputs:

* ``models/tiny_ae.onnx``       — float32 export (for X-CUBE-AI import)
* ``models/tiny_ae_int8.onnx``  — dynamically quantized weights
* ``models/tiny_ae_meta.json``  — feature scaling constants + threshold +
  PC-measured test AUC (no on-board numbers are claimed anywhere)

Run (after the Phase 2/3 pipeline)::

    python -m src.models.tiny_ae
"""

from __future__ import annotations

import json

import numpy as np
import torch
from torch import nn

from src.config import EVAL_SEEDS, MODELS_DIR, VAL_TARGET_FPR
from src.models.common import FEATURE_SETS, load_windows, select


class TinyAutoencoder(nn.Module):
    """8 -> 16 -> 4 -> 16 -> 8 dense autoencoder (~700 parameters)."""

    def __init__(self, n_features: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 16), nn.ReLU(),
            nn.Linear(16, 4), nn.ReLU(),
            nn.Linear(4, 16), nn.ReLU(),
            nn.Linear(16, n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    torch.manual_seed(EVAL_SEEDS[0])
    np.random.seed(EVAL_SEEDS[0])
    features = FEATURE_SETS["all"]

    windows = load_windows()
    train = select(windows, "train", label=0)[features].to_numpy(np.float32)
    val = select(windows, "val")
    test = select(windows, "test")

    # Standardize with TRAIN-normal statistics (stored for the C code)
    mean = train.mean(axis=0)
    std = train.std(axis=0) + 1e-6
    scale = lambda x: ((x - mean) / std).astype(np.float32)  # noqa: E731
    x_train = torch.from_numpy(scale(train))

    model = TinyAutoencoder(len(features))
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    criterion = nn.MSELoss()
    for epoch in range(30):
        permutation = torch.randperm(len(x_train))
        for i in range(0, len(x_train), 512):
            batch = x_train[permutation[i:i + 512]]
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
    model.eval()

    @torch.no_grad()
    def scores(frame) -> np.ndarray:
        x = torch.from_numpy(scale(frame[features].to_numpy(np.float32)))
        return ((model(x) - x) ** 2).mean(dim=1).numpy()

    from sklearn.metrics import roc_auc_score
    val_scores = scores(val)
    test_scores = scores(test)
    threshold = float(np.quantile(
        val_scores[val["label"].to_numpy() == 0], 1 - VAL_TARGET_FPR))
    auc = float(roc_auc_score(test["label"].to_numpy(), test_scores))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"tiny AE: {n_params} params | test ROC-AUC {auc:.4f} (PC, "
          f"same split as the main models)")

    dummy = torch.zeros(1, len(features))
    fp32_path = MODELS_DIR / "tiny_ae.onnx"
    # Legacy exporter: the dynamo exporter writes value_info metadata that
    # onnxruntime's quantization shape inference rejects for this graph.
    torch.onnx.export(model, dummy, str(fp32_path),
                      input_names=["features"], output_names=["reconstruction"],
                      opset_version=17, dynamo=False)

    from onnxruntime.quantization import QuantType, quantize_dynamic
    int8_path = MODELS_DIR / "tiny_ae_int8.onnx"
    quantize_dynamic(str(fp32_path), str(int8_path),
                     weight_type=QuantType.QInt8)
    print(f"fp32: {fp32_path.stat().st_size:,} B | "
          f"int8: {int8_path.stat().st_size:,} B")

    meta = {
        "features": features,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "threshold_mse": threshold,
        "threshold_policy": f"val @ {VAL_TARGET_FPR:.0%} FPR",
        "n_parameters": int(n_params),
        "test_roc_auc_pc": auc,
        "note": "AUC measured on PC with the standard test split; no "
                "on-target numbers are claimed until measured on the board.",
    }
    with open(MODELS_DIR / "tiny_ae_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"meta -> {MODELS_DIR / 'tiny_ae_meta.json'}")


if __name__ == "__main__":
    main()
