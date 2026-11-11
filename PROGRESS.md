# Progress & decision log

Running log of what was done per phase, non-obvious decisions and why,
anomalies found, and the real numbers behind every reported result.

## Phase 1 — Setup, parsing, EDA

**Done:** project structure, venv, parser for both raw formats, EDA notebook
(5 figures in `reports/figures/`), `reports/parse_summary.json`.

**Decisions**
- Rows with DLC < 8 carry the R/T flag at column position `3 + DLC`; the
  parser locates the flag per row via `np.take_along_axis` and zero-fills the
  missing payload bytes (DLC column keeps the true length).
- Payload columns are read as strings explicitly: a hex column that happens
  to contain only decimal digits would otherwise be auto-parsed as int and
  corrupt values (e.g. "05" -> 5 instead of 0x05).
- 1 malformed trailing line in `normal_run_data.txt` dropped (logged by the
  parser).

**Anomalies**
- The DoS file initially appeared to have no large timestamp gap (threshold
  was 60 s). See Phase 2 for the resolution.

## Phase 2 — Features, split, DoS investigation

**Done:** per-message causal features, sliding-window features (64/32),
per-file time-ordered 60/20/20 split, `messages.parquet` (14.0 M rows),
`windows.parquet` (437,292 windows), `train_stats.json`,
`reports/window_summary.json`, `reports/feature_notes.md`.

**Anomaly found & fixed — DoS tail**
- Injected frames in the DoS capture stop at ~2280 s; a 22.4 s recording
  break sits at ~2328 s (the position reported by Verma et al.); the ~480 s
  after it are 100 % attack-free. A time-ordered split would have left the
  DoS test segment nearly attack-free.
- Fix: gap threshold lowered 60 s -> 10 s. Safe because the second-largest
  gap in any file is 0.71 s, so the 10 s rule isolates exactly the four
  terminal recording breaks (22.4 / 2539 / 5588 / 7745 s). Post-trim
  durations 2328 / 2466 / 1949 / 1952 s match Verma et al.'s figures.

**Decisions**
- Expected per-ID periods + known-ID set learned on attack-free frames from
  the TRAIN portions of ALL five captures (not only normal_run_data), per
  requirement: avoids learning "driving vs stationary" instead of
  "normal vs attack". 38 known IDs, global median period 20 ms.
- Window label = attack if >= 1 injected frame; window also stores exact
  injected ratio and attack type.
- Windows never cross a split or file boundary; per-message features are
  strictly causal (computable on a live stream).
- Unexpected but real: bus-level mean dt RISES during DoS bursts in this
  capture (flood delays legitimate ECUs); detection relies on unknown-ID
  ratio, entropy collapse and period deviation instead.

## Phase 3 — Models

(numbers below filled in after the evaluation run completes)

**Decisions**
- Supervised comparison model = LightGBM (file `src/models/supervised.py`
  instead of the originally sketched `cnn1d.py`): trains in ~3 s on CPU for
  262 k windows vs minutes for a CNN, ships a tiny model, and the deep
  component of the project is already the LSTM autoencoder. Allowed
  explicitly by the project brief ("1D-CNN or LightGBM").
- LSTM-AE: hidden 48 (budget <= 64), 6 causal per-message features,
  trained on 40 k attack-free train windows (random subsample, seeded),
  early stopping on attack-free val reconstruction (patience 2). Converges
  in 6-7 epochs.
- Unsupervised scores normalised to [0, 1] with VAL percentiles (p1/p99.9);
  primary threshold = val attack-free windows @ 1 % FPR; F1-optimal
  threshold reported alongside.
- Replay/live scoring cadence: every 0.25 virtual seconds rather than every
  32 messages (the bus runs ~2000 msg/s, stride-based scoring would mean
  ~60 detections/s flooding the dashboard).
- First training run crashed at ONNX export: torch 2.15's exporter needs
  `onnxscript` (not pulled in automatically). Added to requirements;
  training re-run end-to-end.

**Results (test split, mean over seeds 42/43/44, window level, threshold
= val @ 1 % FPR):**

| model | precision | recall | F1 | ROC-AUC | train | infer/window |
|---|---|---|---|---|---|---|
| Isolation Forest | 0.973 | 0.606 | 0.746 | 0.991 | 3.0 s | 25.4 ms |
| LSTM-AE | 0.987 | 0.990 | 0.989 | 0.997 | 197.5 s | 5.7 ms |
| LightGBM | 1.000 | 0.999 | 0.999 | 1.000 | 4.2 s | 2.1 ms |

Episode level: every attack episode detected by every model (episode recall
1.0; 51-68 episodes per attack type in test); detection latency median
4-148 ms, p95 <= 524 ms. False alarms per hour of attack-free traffic:
LightGBM 30/h, IF 264/h, AE 385/h.

**Sanity-check findings (investigated, kept — all explainable)**
- Near-ceiling supervised scores are a KNOWN property of this benchmark
  (the attacks are blatant at window level; this is exactly the criticism
  that motivated the ROAD dataset). Stated in the README rather than
  presented as a breakthrough.
- IF's low recall (0.61) at the 1 % FPR threshold despite AUC 0.991: IF
  ranks DoS/fuzzy anomalies far above spoofing ones, so the fixed threshold
  catches those but misses ~half of gear/RPM windows; with the F1-optimal
  threshold IF reaches F1 0.978. Threshold placement, not ranking, is IF's
  weakness here — kept both numbers in metrics.json.
- Ablation: ID-features-only beats all-features for IF (gear/rpm F1 0.99 vs
  0.57): the unbounded rel_dev features (cap 10) dominate IF's splits and
  drown the subtle spoofing signal. LightGBM is insensitive to this
  (>= 0.996 everywhere).
- Leave-one-attack-out: LightGBM still recalls ~1.0 on each UNSEEN attack
  type (other attacks teach it "not normal"); unsupervised models unaffected
  by construction (IF only 0.37-0.44 on spoofing at its 1 % FPR threshold,
  consistent with the above; AE 0.98-0.99).
- False alarms/hour are high for the unsupervised models at 1 % window FPR
  (hundreds/h): with ~60 windows/s, window-level FPR is the wrong deployment
  knob; event-level tuning would be needed in practice. Logged as a
  limitation + future-work item.

## Phase 4 — RAG diagnostic assistant

**Done:** `knowledge_base/dtc_codes.json` (55 generic SAE J2012 codes:
P0xxx + C0035-C0050 + U0001/U0073/U01xx + an explicit "B codes are mostly
manufacturer-specific" entry rather than inventing generic B codes), two
project-written reference notes (CAN basics, OBD-II workflow), ChromaDB KB
(63 documents, all-MiniLM-L6-v2, cosine), strictly-grounded assistant on
Ollama llama3.2:3b with similarity-gated refusal (< 0.35 -> "not in
knowledge base" without calling the LLM) and graceful degradation when
Ollama is down (returns raw KB entries, flagged).

**Anomaly found & fixed — semantic sibling confusion**
- Pure semantic retrieval ranked sibling codes above the exact one
  (P0172 before P0171 even when the question named P0171): hit@1 was 0.529.
- Fix: hybrid retrieval — DTC codes written in the question are looked up
  exactly and pinned first. hit@1 0.529 -> 0.588 (only 2/17 eval questions
  name a code explicitly; symptom-phrased questions keep legitimate
  sibling ambiguity).

**Results (reports/rag_eval.json):** retrieval hit-rate @1 0.588 /
@3 0.941 / @5 1.000 over 17 code questions; out-of-scope refusal 1/1;
generation check (llama3.2:3b, 6 questions): expected code cited in 4/6
(missed: P0420 answered via sibling P0430 context, P0442 via the EVAP
family) — logged as a small-model limitation, not hidden.

## Phase 5 — API, simulator, dashboard

**Done:** FastAPI (`/health`, `/metrics`, `/detect`, `/diagnose`,
WS `/stream`, serves `frontend/`), replay engine (speed 1-10x,
pause/seek, window scoring every 0.25 virtual seconds, bus stats, per-ID
timing-deviation buckets), custom HTML/CSS/JS dashboard per the design
spec (no framework, Chart.js vendored locally so the demo works offline),
`run.ps1` one-command start, optional Dockerfile. 22 pytest tests pass
(parser, features, API).

**Decisions**
- Scoring cadence is virtual-time based (0.25 s) instead of per-stride:
  at ~2000 msg/s a stride-32 cadence would emit ~60 detections/s and flood
  the websocket/chart with no added value.
- Live "suspect ID" heuristic for direct labels: most frequent unknown ID
  when the window has unknown IDs (DoS/fuzzy), else the ID with the most
  NEGATIVE period deviation (spoofed frames arrive too fast); max |dev|
  wrongly picked delayed legitimate IDs.
- Chart.js vendored into `frontend/js/vendor/` rather than hot-linked.

**Bugs found during the Playwright run-through (all fixed, re-verified)**
- Boot crash: the initial theme call hit a temporal-dead-zone reference
  (`charts` const used before declaration) and silently killed the app
  before the websocket connect. Added a boot guard.
- Attack-period shading rendered black: hex alpha appended to an `hsl()`
  string is invalid for canvas; replaced with a computed rgba fill.
- ROC panel looked empty: all curves hug fpr~0 at this performance level;
  switched to a log FPR axis (standard for near-perfect IDS ROCs), ticks
  restricted to powers of ten.
- Model colors collided with the attack-type palette (purple=DoS,
  teal=Fuzzy); models now use a separate blue ramp, themed for dark mode.
- Grid blowout at 1280 px (canvas would not shrink): `min-width: 0` on
  grid children.

**Dashboard self-evaluation vs the course grid**

| criterion | verdict |
|---|---|
| 1. Overall view | Good: one screen per tab, no data overload, chart type matches data type (line = score over time, heatmap = ID x time deviation, bars = per-class F1, box = latency distribution), metadata footer present. |
| 2. Colors | Good after fixes: 3 color roles respected (qualitative attack palette identical across all tabs; sequential scale for confusion matrix; diverging for timing deviation; alert orange only on threshold crossings; models on a separate blue ramp). |
| 3. Design | Good: white cards on tinted bg, 4px grid, no shadows/gradients/emoji, stats row without boxes, no duplicated numbers. |
| 4. Text | Good: 2 fonts (Jakarta UI, Roboto Mono for all numbers incl. chart axes via Chart.js defaults), sizes per spec, direct labels on the score spike. |
| 5. Menu | Good: top segmented tabs, source/model pill selectors, CAN-ID + attack-type filters. |
| 6. Interactivity | Good: play/pause/seek/speed, click event -> seek, hover tooltips with exact values, light/dark persisted in localStorage, 150 ms transitions, responsive to 1280 px. |

Remaining known imperfections (accepted, documented): the live anomaly
line sits low on the fixed [0,1] axis (the spec forbids auto-zoom — honest
but visually flat in calm segments); on the RPM replay the model
occasionally labels windows "gear" (the gear/RPM class confusion is a real
model property, visible in the confusion matrix, not a dashboard bug).

## Phase 6 — Embedded export (guide only, no hardware claims)

**Done:** `src/models/tiny_ae.py` — dense autoencoder 8->16->4->16->8
(428 parameters) over the window features, trained on attack-free train
windows, same split/threshold policy as the main models: **test ROC-AUC
0.9996 (PC)**. Exports `models/tiny_ae.onnx` (2.8 KB) +
`tiny_ae_int8.onnx` + `tiny_ae_meta.json` (scaling constants, threshold).
`embedded/README.md` (X-CUBE-AI import, measurement procedure with
DWT->CYCCNT, safe bus-connection options) + `embedded/canguard_inference.c`
skeleton. No on-board numbers are claimed anywhere.

**Findings**
- Dynamic int8 quantization makes this model BIGGER (5.8 KB vs 2.8 KB):
  quantization metadata outweighs 428 parameters. Documented: import the
  fp32 ONNX into Cube.AI.
- torch 2.15 dynamo exporter emits value_info that onnxruntime's
  quantizer rejects (shape-inference clash); the tiny model uses the
  legacy exporter (`dynamo=False`).
