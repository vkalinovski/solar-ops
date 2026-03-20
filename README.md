# solar-ops

M1+ solar flare onset forecasting at horizons 60, 120, 360, and 720 minutes.

Trained on GOES XRS minute-level data (2012–2023, excluding 2018–2019). Calibrated CatBoost ensembles, one per horizon. The bundle in `bundle/` contains trained weights ready for inference.

**Live demo:** [solarflares.space](https://solarflares.space/)

## Install

```bash
pip install .
```

## Usage

```bash
# Train all horizons
python scripts/train.py

# Run live inference against SWPC API
python scripts/infer.py --mode live

# Start API server
python scripts/serve.py

# Evaluate holdout metrics
python scripts/evaluate.py
```

## Docker

```bash
docker compose up --build
```

API available at `http://localhost:8000` — endpoints: `/health`, `/now`, `/history`.

## Structure

```
solarflare_app/   Python package: training, calibration, inference, API
scripts/          Entry points
bundle/           Trained model weights (CatBoost .cbm, 5 seeds × 4 horizons)
backend/          Dockerfile for the API service
data/             Raw, gold, artifacts (gitignored except .gitkeep)
```

## Configuration

Copy `.env.example` to `.env`. Key settings:

| Variable | Default | Description |
|---|---|---|
| `NEG_POS_RATIO` | `3` | Negative-to-positive sampling ratio |
| `CALIB_YEAR` | `2024` | Calibration year |
| `HOLDOUT_YEAR` | `2025` | Holdout year |
| `SEEDS` | `0,1,2,3,4` | Ensemble seeds |
