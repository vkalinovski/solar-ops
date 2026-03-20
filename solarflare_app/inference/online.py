from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .offline import predict_latest_row


SWPC_XRS_7D_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"


def fetch_xrs_7d(timeout: int = 30) -> pd.DataFrame:
    response = requests.get(SWPC_XRS_7D_URL, timeout=timeout)
    response.raise_for_status()
    df = pd.DataFrame(response.json())
    if df.empty:
        raise RuntimeError("SWPC XRS payload is empty.")

    df["time_tag"] = pd.to_datetime(df["time_tag"], utc=True, errors="coerce")
    df = df.dropna(subset=["time_tag"])

    pivot = df.pivot_table(index="time_tag", columns="energy", values="flux", aggfunc="last").sort_index()
    rename = {}
    for col in pivot.columns:
        name = str(col).lower()
        if "0.1-0.8" in name:
            rename[col] = "xrsb"
        elif "0.05-0.4" in name:
            rename[col] = "xrsa"
    pivot = pivot.rename(columns=rename).reset_index().rename(columns={"time_tag": "ts"})
    return pivot[["ts"] + [c for c in ("xrsa", "xrsb") if c in pivot.columns]]


def build_live_feature_row(df: pd.DataFrame, lookback_min: int = 72 * 60) -> pd.DataFrame:
    data = df.copy()
    data["ts"] = pd.to_datetime(data["ts"], utc=True, errors="coerce")
    data = data.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts", keep="last")
    data = data[data["ts"] >= data["ts"].max() - pd.Timedelta(minutes=int(lookback_min))].copy()

    full = pd.date_range(data["ts"].min(), data["ts"].max(), freq="1min", tz="UTC")
    data = data.set_index("ts").reindex(full).ffill().reset_index().rename(columns={"index": "ts"})

    eps = 1e-12
    data["xrsb"] = pd.to_numeric(data.get("xrsb"), errors="coerce").ffill().fillna(eps).clip(lower=eps)
    data["log_xrsb"] = np.log10(data["xrsb"])

    if "xrsa" in data.columns:
        data["xrsa"] = pd.to_numeric(data["xrsa"], errors="coerce").ffill().fillna(eps).clip(lower=eps)
        data["log_xrsa"] = np.log10(data["xrsa"])

    for lag in (1, 5, 15, 60, 180, 360, 720):
        data[f"log_xrsb_lag{lag}"] = data["log_xrsb"].shift(lag)

    for window in (5, 15, 60, 180, 360, 720):
        data[f"log_xrsb_roll{window}_mean"] = data["log_xrsb"].rolling(window, min_periods=max(2, window // 5)).mean()
        data[f"log_xrsb_roll{window}_std"] = data["log_xrsb"].rolling(window, min_periods=max(2, window // 5)).std()

    return data.tail(1).fillna(0.0)


def run_live_demo(bundle_dir: Path, lookback_min: int = 72 * 60, max_stale_min: float = 60.0) -> dict:
    xrs = fetch_xrs_7d()
    features = build_live_feature_row(xrs, lookback_min=lookback_min)
    preds = predict_latest_row(bundle_dir=bundle_dir, frame=features)

    data_ts = pd.to_datetime(features["ts"].iloc[0], utc=True)
    now = datetime.now(timezone.utc)
    lag_min = float((now - data_ts.to_pydatetime()).total_seconds() / 60.0)
    stale = lag_min > float(max_stale_min)

    return {
        "asof_utc": now.isoformat(),
        "data_ts_utc": data_ts.isoformat(),
        "lag_minutes": lag_min,
        "stale": bool(stale),
        "features_mode": "LIVE_XRS_DEMO_FILL0",
        "predictions": {
            str(item.horizon): {
                "probability": item.probability,
                "threshold": item.threshold,
                "fire": bool(item.fire and not stale),
                "policy": item.policy,
            }
            for item in preds
        },
    }
