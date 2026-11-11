"""LSTM autoencoder over message windows (PyTorch, CPU).

A sequence-to-sequence autoencoder: the encoder LSTM compresses a window of
64 messages (6 causal features each) into its final hidden state; the
decoder LSTM reconstructs the sequence from that state repeated at every
step. The anomaly score is the mean squared reconstruction error of the
window; normalisation to [0, 1] and thresholding are handled in
:mod:`evaluate`.

Trained on attack-free TRAIN windows only, with early stopping on the
reconstruction error of attack-free VAL windows.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from src.config import (
    AE_BATCH_SIZE,
    AE_HIDDEN,
    AE_LR,
    AE_MAX_EPOCHS,
    AE_MAX_TRAIN_WINDOWS,
    AE_MAX_VAL_WINDOWS,
    AE_PATIENCE,
    WINDOW_SIZE,
)
from src.models.common import SEQUENCE_FEATURES


class LSTMAutoencoder(nn.Module):
    """Seq2seq LSTM autoencoder (hidden size capped for CPU training)."""

    def __init__(self, n_features: int = len(SEQUENCE_FEATURES),
                 hidden: int = AE_HIDDEN) -> None:
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)
        latent = h[-1]                                    # (B, hidden)
        repeated = latent.unsqueeze(1).expand(-1, x.shape[1], -1)
        decoded, _ = self.decoder(repeated.contiguous())
        return self.head(decoded)


def _subsample(array: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if len(array) <= limit:
        return array
    rng = np.random.default_rng(seed)
    return array[np.sort(rng.choice(len(array), size=limit, replace=False))]


def train_autoencoder(
    train_sequences: np.ndarray,
    val_sequences: np.ndarray,
    seed: int,
) -> tuple[LSTMAutoencoder, dict]:
    """Train on attack-free windows; early-stop on val reconstruction loss.

    Returns the best model (lowest val loss) and a training-stats dict.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    train = _subsample(train_sequences, AE_MAX_TRAIN_WINDOWS, seed)
    val = _subsample(val_sequences, AE_MAX_VAL_WINDOWS, seed)

    model = LSTMAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=AE_LR)
    criterion = nn.MSELoss()
    x_val = torch.from_numpy(val)

    best_loss = float("inf")
    best_state = None
    patience_left = AE_PATIENCE
    history: list[float] = []
    start = time.perf_counter()

    for epoch in range(AE_MAX_EPOCHS):
        model.train()
        permutation = torch.randperm(len(train))
        epoch_loss = 0.0
        for i in range(0, len(train), AE_BATCH_SIZE):
            batch = torch.from_numpy(train[permutation[i:i + AE_BATCH_SIZE].numpy()])
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        epoch_loss /= len(train)

        model.eval()
        with torch.no_grad():
            val_losses = []
            for i in range(0, len(x_val), 2048):
                chunk = x_val[i:i + 2048]
                val_losses.append(
                    ((model(chunk) - chunk) ** 2).mean().item() * len(chunk)
                )
            val_loss = sum(val_losses) / len(x_val)
        history.append(val_loss)
        print(f"    epoch {epoch + 1:2d}: train {epoch_loss:.5f}  "
              f"val {val_loss:.5f}")

        if val_loss < best_loss * 0.999:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = AE_PATIENCE
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"    early stop after epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    stats = {
        "seed": seed,
        "epochs_run": len(history),
        "best_val_loss": best_loss,
        "train_windows": int(len(train)),
        "train_time_s": round(time.perf_counter() - start, 1),
    }
    return model, stats


@torch.no_grad()
def score_sequences(
    model: LSTMAutoencoder, sequences: np.ndarray, batch_size: int = 2048,
) -> np.ndarray:
    """Mean squared reconstruction error per window (raw anomaly score)."""
    model.eval()
    scores = np.empty(len(sequences), dtype=np.float32)
    for i in range(0, len(sequences), batch_size):
        chunk = torch.from_numpy(sequences[i:i + batch_size])
        error = ((model(chunk) - chunk) ** 2).mean(dim=(1, 2))
        scores[i:i + batch_size] = error.numpy()
    return scores


def export_onnx(model: LSTMAutoencoder, path: str) -> None:
    """Export the autoencoder to ONNX (dynamic batch, fixed window)."""
    dummy = torch.zeros(1, WINDOW_SIZE, len(SEQUENCE_FEATURES))
    torch.onnx.export(
        model, dummy, path,
        input_names=["window"], output_names=["reconstruction"],
        dynamic_axes={"window": {0: "batch"}, "reconstruction": {0: "batch"}},
        opset_version=17,
    )
