# Embedded deployment guide (STM32, optional phase)

Goal: run the anomaly detector on an STM32 Nucleo board, scoring windows of
CAN frames received by the CAN peripheral.

**Status: guide + skeleton only.** The model export below was produced and
validated on PC; no inference-time or memory numbers are claimed for the
board — the measurement procedure is described so you can obtain them
honestly on your hardware.

## Why a tiny dense autoencoder (not the LSTM)

X-CUBE-AI maps small feed-forward networks onto Cortex-M very efficiently;
recurrent layers are heavier and more restrictive. The embedded candidate is
therefore `models/tiny_ae.onnx`: a dense autoencoder over the same 8
window-level features used by the Isolation Forest
(8 → 16 → 4 → 16 → 8, ~700 parameters), trained on the same attack-free
train windows and thresholded the same way (val @ 1 % FPR). Its PC-measured
test ROC-AUC is in `models/tiny_ae_meta.json`, directly comparable with the
main models in `reports/metrics.json`. A dynamically quantized variant
(`tiny_ae_int8.onnx`) is provided; for Cube.AI you can also let the tool
apply post-training quantization from the fp32 ONNX.

Produce the artifacts (after Phases 1–3):

```bash
python -m src.models.tiny_ae
```

## Import into STM32CubeIDE

1. Install the **X-CUBE-AI** expansion package from the CubeMX Software
   Packs manager (Core ≥ 8.x).
2. Create a project for your Nucleo board; in CubeMX enable:
   - the CAN peripheral (bxCAN or FDCAN depending on the MCU) at your bus
     bitrate (the HCRL captures correspond to 500 kbit/s),
   - a microsecond timebase (e.g. a 32-bit timer free-running at 1 MHz)
     for frame timestamps,
   - X-CUBE-AI → Add network → name it `canguard`, type ONNX, select
     `models/tiny_ae.onnx`.
3. Use the **Analyze** button: Cube.AI reports MACC, flash and RAM usage of
   the network before you ever flash the board — record these as the static
   footprint.
4. Generate code. Cube.AI emits `canguard.h/.c`, `canguard_data.h/.c`
   (weights) and the `ai_platform` runtime.

## Wire the inference loop

`canguard_inference.c` in this folder is the skeleton:

1. Fill the constant tables at the top from `models/tiny_ae_meta.json`
   (feature mean/std, MSE threshold) and `data/processed/train_stats.json`
   (known IDs + expected periods).
2. Call `canguard_init()` once, then `canguard_on_frame(timestamp_us, id,
   dlc)` from the CAN RX callback. Every 32 frames (stride) it computes the
   8 window features over the last 64 frames and runs the network.
3. Implement `canguard_window_features()` — the required math is small and
   allocation-free; the reference semantics are in
   `src/data/features_live.py` (entropy, unknown-ID ratio, inter-arrival
   stats, period deviation clipped to [-1, +10]).

## Measuring (do this on the board, report only what you measure)

- **Inference time**: wrap `ai_canguard_run` with the DWT cycle counter
  (`DWT->CYCCNT`); convert cycles → µs with the core clock. Report median
  and p95 over ≥1000 windows.
- **Memory**: flash/RAM of the network from the Cube.AI Analyze report plus
  the application `.map` file totals.
- **End-to-end latency**: time from the last frame of a window (CAN RX
  timestamp) to the threshold decision.

## Feeding real frames

Options, in increasing order of realism:

1. Replay the dataset from the PC over a USB-CAN adapter onto a bench bus
   wired to the Nucleo's CAN transceiver.
2. Two Nucleo boards back-to-back (one replaying, one detecting), 120 Ω
   termination at both ends.
3. A real vehicle's OBD-II connector (pins 6/14) — listen-only
   (ensure the transceiver is in silent mode); never transmit on a
   vehicle bus.
