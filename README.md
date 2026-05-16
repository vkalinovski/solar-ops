# solar-ops

Probabilistic forecasting of **M1+ solar flare onsets** at four horizons: 60, 120, 360, and 720 minutes.

Built on GOES XRS minute-level irradiance data (2012–2023). Each horizon is a calibrated CatBoost ensemble of 5 independently-seeded models. Trained weights are included in `bundle/` — no retraining needed to run inference.

**Live demo:** [solarflares.space](https://solarflares.space/)

---

## How it works

The system predicts the probability that an M1+ solar flare onset will occur within the next H minutes (H ∈ {60, 120, 360, 720}).

**1. Gold dataset** — Raw GOES XRS minute data is joined with daily solar indices (sunspot number, F10.7, X-ray background flux) and SRS active-region data using causal as-of joins (1-day publication delay) to prevent leakage. Features include log-transformed irradiance, rolling statistics at 7 timescales, rate-of-change slopes, and solar-cycle proxies. Binary targets mark each minute as positive if an M1+ onset occurs within the next H minutes.

**2. Training** — Hyperparameters are first found with Optuna TPE search (60–200 trials per horizon, 3-seed mini-ensemble per trial). Final models are trained with 5 seeds and a 1:3 negative-to-positive sampling ratio. `scale_pos_weight` compensates for residual class imbalance after sampling.

**3. Calibration** — The 2024 calibration year is split: Jan–Aug (CAL_A) fits the calibrator, Sep–Dec (CAL_B) selects among Identity / Platt / Isotonic by log-loss. The winning calibrator is frozen into the bundle.

**4. Alerting policy** — An operational threshold is chosen on CAL_B to maximise `0.65·TSS + 0.35·F1` subject to a precision floor and daily alert budget. An alert fires only after the probability exceeds the threshold for `persist_min` consecutive minutes, with a `cooldown_min` quiet period afterward.

---

## Repository structure

```
solar-ops/
├── solarflare_app/          
│   ├── settings.py          
│   ├── data/gold.py        
│   ├── modeling/
│   │   ├── trainer.py       
│   │   ├── calibration.py   
│   │   ├── metrics.py       
│   │   ├── bootstrap.py     
│   │   ├── policy.py        
│   │   └── bundle.py        
│   ├── inference/
│   │   ├── offline.py      
│   │   └── online.py        
│   ├── api/app.py          
│   ├── storage/store.py    
│   └── utils/logging.py     
│
├── scripts/
│   ├── build_dataset.py     
│   ├── tune.py              
│   ├── train.py             
│   ├── train_single.py      
│   ├── test_suite.py        
│   ├── evaluate.py          
│   ├── validate.py          
│   ├── plot.py             
│   ├── infer.py             
│   └── serve.py            
│
├── bundle/                 
│   ├── bundle.json          
│   └── horizons/
│       ├── 60m/            
│       ├── 120m/
│       ├── 360m/
│       └── 720m/
│
├── backend/                
│   ├── Dockerfile
│   └── app/main.py
│
├── data/                    
│   ├── raw/                 
│   ├── gold/                
│   └── artifacts/          
│
├── pyproject.toml
├── Makefile
├── docker-compose.yml
└── .env.example
```

---

## Temporal split

| Set | Years | Purpose |
|-----|-------|---------|
| Train | 2012–2023 | Model fitting |
| CAL_A | Jan–Aug 2024 | Calibrator fitting |
| CAL_B | Sep–Dec 2024 | Calibrator selection + threshold optimisation |
| Holdout | 2025 | Final evaluation (never touched during training) |

---



## Training from scratch

```bash
# 1. Build gold dataset (requires raw data in data/raw/)
python scripts/build_dataset.py

# 2. Tune hyperparameters per horizon (optional — tuned params already in bundle)
python scripts/tune.py --horizon 120 --trials 60

# 3. Train all horizons
python scripts/train.py

# 4. Validate and run test suite
python scripts/validate.py
python scripts/test_suite.py
```

---

## Configuration

Copy `.env.example` → `.env`.

| Variable | Default | Description |
|---|---|---|
| `NEG_POS_RATIO` | `3` | Negative-to-positive sampling ratio |
| `SEEDS` | `0,1,2,3,4` | Ensemble seeds |
| `CALIB_YEAR` | `2024` | Calibration year |
| `HOLDOUT_YEAR` | `2025` | Final evaluation year |
| `CALIBRATION_SPLIT_MONTH` | `9` | CAL_A = months < 9; CAL_B = months ≥ 9 |
| `BUNDLE_DIR` | `./bundle` | Path to trained bundle |
| `UPDATE_EVERY_SEC` | `60` | Live polling interval (API) |

**Live demo:** [solarflares.space](https://solarflares.space/)



## Structure

```
solarflare_app/   Python package: training, calibration, inference, API
scripts/          Entry points
bundle/           Trained model weights (CatBoost .cbm, 5 seeds × 4 horizons)
backend/          Dockerfile for the API service
data/             Raw, gold, artifacts (gitignored except .gitkeep)
```

