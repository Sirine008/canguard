# canguard

AI-powered CAN bus intrusion detection and vehicle diagnostic assistant.

Modern vehicles exchange thousands of CAN frames per second between ECUs, with
no built-in authentication: any node (or attacker) on the bus can inject
frames. **canguard** detects injection attacks (DoS, fuzzing, spoofing) on real
CAN traffic with classical and deep anomaly-detection models, and ships a
local RAG assistant that explains OBD-II diagnostic trouble codes — all
running on CPU, fully reproducible.

> Work in progress — phases are documented as they land. Final README with
> architecture diagram, full results and screenshots comes with the last phase.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Dataset

Main benchmark: **HCRL Car-Hacking Dataset** (Seo, Song, Kim — *HCRL, Korea
University*, 2018), available at https://ocslab.hksecurity.net/Datasets/car-hacking-dataset.
Place the files in `data/raw/car_hacking/`:

```
DoS_dataset.csv  Fuzzy_dataset.csv  gear_dataset.csv  RPM_dataset.csv  normal_run_data.txt
```

Then parse everything into processed parquet files:

```bash
python -m src.data.parser
```

### Known dataset issues (handled)

- The Fuzzy, gear and RPM files each contain one very large timestamp gap
  (2539 s, 5588 s and 7745 s respectively in this copy — an issue documented
  by Verma et al. in the ROAD dataset paper). The parser detects any
  inter-message gap > 60 s automatically and trims the file just before it;
  the trimmed captures span 2466 s / 1949 s / 1952 s. The DoS file in this
  distribution has no gap above 60 s (largest: 22.4 s) and is kept whole.
- Rows with DLC < 8 carry the R/T flag in an earlier column; the parser
  resolves the flag position per row.
- **Limitation:** the vehicle was likely stationary during the attack
  captures, so "normal" traffic inside attack files differs from
  `normal_run_data.txt` (recorded while driving). Results should be read with
  this domain shift in mind.

## Exploratory data analysis

```bash
jupyter notebook notebooks/01_eda.ipynb
```

Figures are saved to `reports/figures/`.

## License

MIT — see [LICENSE](LICENSE).
