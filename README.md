# Learning to correct SGP4 — and finding out what the error really was

Academic project — IMT Mines Alès, AI & Data Science (December 2025), revisited in 2026
B. Delais, L. Igoho Zephir, L. Vasseur, C. Vial

📄 Original report (PDF): [sgp4.pdf](https://bilaldelais.com/assets/reports/sgp4.pdf)

## Summary

The goal was to improve the **TLE + SGP4** orbital propagator by learning its residual error against JPL Horizons ephemerides, rather than replacing it. The December 2025 report benchmarked MLP, LSTM/GRU and Transformer models on that residual.

Revisiting the pipeline in 2026 showed that **most of the "error" we were learning was an artefact of our own data pipeline**, and that the reference itself was not independent. This repository contains the corrected pipeline, the corrected datasets and the analysis below.

## What went wrong

**1. A time-synchronisation bug.** SGP4 was propagated at a timestamp truncated to the second, while Horizons received the exact timestamp. At orbital speed (~7.7 km/s for the ISS) this created an artificial along-track offset of up to ~7.6 km.

| Median SGP4 − Horizons error | Before fix | After fix |
|---|---|---|
| ISS | 4.10 km | **0.085 km** |
| Hubble | 3.32 km | **0.061 km** |

Hubble RMSE drops from 4.39 km to 0.37 km. ISS RMSE stays around 17 km because it is dominated by a few spikes (up to ~264 km), most likely manoeuvres, not by propagation drift.

**2. Horizons is not independent ground truth here.** After the fix, the SGP4/Horizons gap at the TLE epoch is of the order of a metre, and far from the epoch the Horizons position is reproduced to within tens of metres by SGP4 applied to a neighbouring TLE. The measured "error" therefore looks like the difference between successive TLEs, not SGP4's physical error.

**Consequence:** the models in the original report, including the Transformer that performed best there, were trained largely on the artefact. Their results should not be read as a correction of SGP4. A meaningful version of this problem needs a reference that is independent of TLEs (e.g. precise orbit products from GNSS-tracked satellites) and an explicit treatment of manoeuvres.

## What's in this repository

| Path | Purpose |
|---|---|
| `preproccessing/fetch_tles.py` | Download TLE history from Space-Track (`--norad`, `--count`, `--output`) |
| `preproccessing/sgp4_TLE_horizons.py` | Propagate TLEs with SGP4 and align them with JPL Horizons positions (corrected timing) |
| `preproccessing/fix_time_truncation.py` | Repair existing datasets offline, with validation against the stored positions |
| `data/` | Corrected ISS and Hubble datasets, raw TLE extracts |
| `models/LSTM_TLE.py` | LSTM / GRU sliding-window baseline (PyTorch), trained on the pre-fix data |
| `models/plot_*.py` | Plots used in the original report |
| `tests/` | Pytest checks: TLE parsing, sub-second timing, dataset consistency |

The Transformer code from the original report is not included: it was kept on a local machine and is lost. `data/processed/CHANDRA*.csv` comes from a script no longer in the repo and may be affected by the same timing bug.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env    # fill SPACETRACK_USER / SPACETRACK_PASS — never commit them
set -a; source .env; set +a

python preproccessing/fetch_tles.py --norad 25544 --name ISS --count 200 \
    --output data/raw/iss_last_200_tles_spacetrack.csv
python preproccessing/sgp4_TLE_horizons.py --tles data/raw/iss_last_200_tles_spacetrack.csv \
    --horizons-id -125544 --output data/dataset_iss_sgp4_vs_horizons.csv

python -m pytest tests
```

## Stack

Python · PyTorch · sgp4 · Skyfield · Astropy / Astroquery (JPL Horizons) · Pandas · pytest
