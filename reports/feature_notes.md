# Why each attack is detectable — feature rationale

Median window-feature values (window = 64 messages, stride 32), attack
windows vs the attack-free windows of the same capture. All numbers come from
`data/processed/windows.parquet` built by `python -m src.data.features`.

| feature | normal ref | DoS attack | Fuzzy attack | Gear attack | RPM attack |
|---|---|---|---|---|---|
| id_entropy (bits) | 4.33 | **2.84** | **5.32** | 3.65 | 3.65 |
| unique_id_count | 22 | 18 | **44** | 21 | 21 |
| unknown_ratio | 0.00 | **0.55** | **0.38** | 0.00 | 0.00 |
| mean_abs_rel_dev | 0.04 | **6.70** | **4.51** | **0.49** | **0.47** |
| max_abs_rel_dev | 0.19 | **10.0** | **10.0** | **1.97** | **1.09** |

## Per attack

**DoS (flood on ID `0x000`)** — the injected ID has the highest arbitration
priority and is hammered continuously. Three features light up: (1)
`unknown_ratio` jumps to ~0.55 because `0x000` never occurs in attack-free
traffic, so every flood frame is an unknown ID; (2) `id_entropy` collapses
from 4.33 to 2.84 bits because one ID dominates the window; (3) `rel_dev`
saturates. One subtlety worth knowing: the bus-level `mean_dt` *rises*
(0.5 → 1.0 ms) during flood windows in this capture — the flood wins
arbitration and delays legitimate ECUs, so bursts are followed by quiet
stretches rather than a uniformly faster bus.

**Fuzzy (random IDs + random payloads)** — the mirror image of DoS:
`id_entropy` *increases* (4.37 → 5.32 bits) and `unique_id_count` doubles
(22 → 44) because random 11-bit IDs spray the ID space; `unknown_ratio`
(~0.38) catches every sprayed ID that is not a legitimate one. Payload bytes
are uniformly distributed (see `eda_payload_bytes.png`), whereas normal
payloads are sparse and structured.

**Gear / RPM spoofing (single legitimate ID)** — the hard cases, and the
reason timing features exist. The spoofed IDs (`0x43F`, `0x316`) are
legitimate, so `unknown_ratio` stays 0 and ID-distribution features barely
move. What gives the attack away is **periodicity violation**: a legitimate
ECU sends `0x316` every ~10 ms like clockwork; the attacker's extra frames
land *between* the legitimate ones, so the time since the last same-ID frame
collapses far below the learned period. `mean_abs_rel_dev` rises 0.04 → ~0.5
(12×) and `max_abs_rel_dev` 0.19 → 1–2. This is the classic result that
message-timing features detect masquerade attacks that payload/ID features
miss.

## Leakage control (one-liner for the interview)

Expected per-ID periods and the known-ID set are learned **only on
attack-free frames from the train portion** of each capture (pooled across
driving and stationary captures to avoid learning "driving vs stationary"
instead of "normal vs attack"); splits are time-ordered per file with no
shuffling, and windows never cross a split or file boundary.
