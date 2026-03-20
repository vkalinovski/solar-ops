"""
Build the gold dataset: RAW → GOLD (minute-level parquet, partitioned by year).

Input data expected in data/raw/:
  - xrs/           GOES XRS minute CSV files, one per year (columns: ts, xrsb, xrsa)
  - dayind_daily.parquet   Daily solar indices (SSN, F10.7, xray background, ...)
  - srs_daily_agg.parquet  SRS active-region aggregates (n_regions, sum_area, ...)
  - flare_onsets.parquet   M1+ flare onset timestamps (column: onset_time)

Output:
  data/gold/year=YYYY/data.parquet   — one file per year

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --raw-dir data/raw --gold-dir data/gold --years 2012 2024
    python scripts/build_dataset.py --no-overwrite   # skip years already built
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = (60, 120, 360, 720)
FLARE_COUNT_WINDOWS = (1440, 10080)  # 1 day, 1 week
ROLL_WINDOWS = (5, 15, 60, 180, 360, 720, 1440)
MAX_MISSING_XRSB = 0.02
FILL_XRS_LIMIT   = 60


# ── XRS loading ───────────────────────────────────────────────────────────────

def load_xrs_year(year: int, raw_dir: Path) -> pd.DataFrame:
    """Load GOES XRS minute data for one year from data/raw/xrs/."""
    xrs_dir = raw_dir / "xrs"
    candidates = [
        xrs_dir / f"{year}.parquet",
        xrs_dir / f"goes_xrs_{year}.parquet",
        xrs_dir / f"xrs_{year}.csv",
        xrs_dir / f"goes_xrs_{year}.csv",
    ]
    for p in candidates:
        if p.exists():
            if p.suffix == ".parquet":
                return pd.read_parquet(p)
            return pd.read_csv(p, parse_dates=["ts"])
    raise FileNotFoundError(
        f"XRS data for {year} not found. Tried: {[str(c) for c in candidates]}"
    )


# ── minute grid helpers ───────────────────────────────────────────────────────

def _expected_minute_index(year: int) -> pd.DatetimeIndex:
    leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    start = pd.Timestamp(year=year, month=1, day=1)
    end   = pd.Timestamp(year=year, month=12, day=31, hour=23, minute=59)
    return pd.date_range(start, end, freq="min")


def repair_minute_grid(base: pd.DataFrame, year: int) -> pd.DataFrame:
    base = base.copy()
    base["ts"] = pd.to_datetime(base["ts"]).dt.tz_localize(None)
    base = base.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)

    full = pd.DataFrame({"ts": _expected_minute_index(year)})
    merged = full.merge(base, on="ts", how="left")

    for col in ["xrsb", "xrsa"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").ffill(limit=FILL_XRS_LIMIT)

    miss = float(merged["xrsb"].isna().mean()) if "xrsb" in merged.columns else 1.0
    if miss > MAX_MISSING_XRSB:
        raise ValueError(f"Too many missing xrsb after repair for {year}: {miss:.2%}")
    return merged


# ── feature engineering ───────────────────────────────────────────────────────

def _safe_min_periods(w: int) -> int:
    if w <= 2:  return 1
    if w <= 6:  return 2
    if w <= 12: return 3
    return min(w, max(5, w // 10))


def build_xrs_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-12

    df["log_xrsb"] = np.log10(np.clip(pd.to_numeric(df["xrsb"], errors="coerce"), eps, None))
    if "xrsa" in df.columns:
        df["log_xrsa"] = np.log10(np.clip(pd.to_numeric(df["xrsa"], errors="coerce"), eps, None))

    for lag in [1, 5, 15, 60, 180, 360, 720, 1440]:
        df[f"log_xrsb_lag{lag}"] = df["log_xrsb"].shift(lag)
        if "log_xrsa" in df.columns:
            df[f"log_xrsa_lag{lag}"] = df["log_xrsa"].shift(lag)

    for w in ROLL_WINDOWS:
        mp = _safe_min_periods(w)
        s  = df["log_xrsb"]
        df[f"log_xrsb_roll{w}_mean"] = s.rolling(w, min_periods=mp).mean()
        df[f"log_xrsb_roll{w}_std"]  = s.rolling(w, min_periods=mp).std()
        df[f"log_xrsb_roll{w}_min"]  = s.rolling(w, min_periods=mp).min()
        df[f"log_xrsb_roll{w}_max"]  = s.rolling(w, min_periods=mp).max()
        if "log_xrsa" in df.columns:
            a = df["log_xrsa"]
            df[f"log_xrsa_roll{w}_mean"] = a.rolling(w, min_periods=mp).mean()
            df[f"log_xrsa_roll{w}_std"]  = a.rolling(w, min_periods=mp).std()

    def roll_slope(series: pd.Series, w: int, frac: float = 0.2) -> pd.Series:
        mp = _safe_min_periods(w)
        k  = max(1, int(w * frac))
        def _s(arr):
            fin = np.isfinite(arr)
            if fin.sum() < mp: return np.nan
            a_ = arr[:k][np.isfinite(arr[:k])]
            b_ = arr[-k:][np.isfinite(arr[-k:])]
            if a_.size == 0 or b_.size == 0: return np.nan
            return float(b_.mean() - a_.mean())
        return series.rolling(w, min_periods=mp).apply(_s, raw=True)

    for w in [60, 180, 360, 720, 1440]:
        df[f"log_xrsb_roll{w}_slope"] = roll_slope(df["log_xrsb"], w)

    return df


def add_solar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-12

    if "xray_bkgd_flux" in df.columns:
        df["log10_xray_bkgd_flux"] = np.log10(
            np.clip(pd.to_numeric(df["xray_bkgd_flux"], errors="coerce"), eps, None))
        df["log10_xray_bkgd_flux_roll7_mean"] = (
            df["log10_xray_bkgd_flux"].rolling(10080, min_periods=1440).mean())

    for c in ["sunspot_number", "f107"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[f"{c}_d1"] = df[c] - df[c].shift(1440)

    if "sunspot_number" in df.columns:
        s = df["sunspot_number"]
        df["sunspot_roll7_mean"]  = s.rolling(10080, min_periods=1440).mean()
        df["sunspot_roll27_mean"] = s.rolling(38880, min_periods=10080).mean()

    if "f107" in df.columns:
        f = df["f107"]
        df["f107_roll7_mean"]  = f.rolling(10080, min_periods=1440).mean()
        df["f107_roll7_std"]   = f.rolling(10080, min_periods=1440).std()
        df["f107_roll27_mean"] = f.rolling(38880, min_periods=10080).mean()

    return df


# ── causal join (anti-leakage) ────────────────────────────────────────────────

def join_daily_asof(df_min: pd.DataFrame, df_daily: pd.DataFrame,
                    shift_days: int, publish_time: str) -> pd.DataFrame:
    """Left-join daily values onto minute grid using as-of join with publication delay."""
    if df_daily is None or len(df_daily) == 0:
        return df_min

    df_min   = df_min.copy()
    df_daily = df_daily.copy()
    df_min["ts"]      = pd.to_datetime(df_min["ts"]).dt.tz_localize(None)
    df_daily["date"]  = pd.to_datetime(df_daily["date"]).dt.tz_localize(None)

    hh, mm = map(int, publish_time.split(":"))
    df_daily["avail_ts"] = (df_daily["date"]
                            + pd.Timedelta(days=shift_days)
                            + pd.Timedelta(hours=hh, minutes=mm))
    df_daily = df_daily.sort_values("avail_ts").reset_index(drop=True)
    df_min   = df_min.sort_values("ts").reset_index(drop=True)

    merged = pd.merge_asof(
        df_min, df_daily.drop(columns=["date"]),
        left_on="ts", right_on="avail_ts",
        direction="backward", allow_exact_matches=True,
    )
    return merged.drop(columns=["avail_ts"], errors="ignore")


# ── target labelling ──────────────────────────────────────────────────────────

def add_targets(df: pd.DataFrame, flare_df: pd.DataFrame, onset_col: str) -> pd.DataFrame:
    df = df.copy()
    flare = flare_df.copy()
    flare[onset_col] = pd.to_datetime(flare[onset_col]).dt.tz_localize(None)
    onset_times = np.sort(flare[onset_col].values)
    ts_vals = pd.to_datetime(df["ts"]).values

    for h in HORIZONS:
        col   = f"y_onset_m1p_in_{h}m"
        left  = np.searchsorted(onset_times, ts_vals, side="right")
        right = np.searchsorted(onset_times, ts_vals + np.timedelta64(h, "m"), side="right")
        df[col] = (right > left).astype(np.int8)

    for w in FLARE_COUNT_WINDOWS:
        col   = f"flare_cnt_past_{w}m"
        left  = np.searchsorted(onset_times, ts_vals - np.timedelta64(w, "m"), side="right")
        right = np.searchsorted(onset_times, ts_vals, side="right")
        df[col] = (right - left).astype(np.int32)

    return df


# ── per-year build ────────────────────────────────────────────────────────────

def build_year(year: int, raw_dir: Path, gold_dir: Path,
               dayind_df: pd.DataFrame, srs_df: pd.DataFrame,
               flare_df: pd.DataFrame, onset_col: str,
               overwrite: bool = True) -> str:
    out_path = gold_dir / f"year={year}" / "data.parquet"
    ok_marker = out_path.parent / "_SUCCESS.json"

    if not overwrite and out_path.exists() and ok_marker.exists():
        return "skipped"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    base = load_xrs_year(year, raw_dir)
    base = repair_minute_grid(base, year)
    base = build_xrs_features(base)
    base = join_daily_asof(base, dayind_df, shift_days=1, publish_time="00:00")
    base = join_daily_asof(base, srs_df,    shift_days=1, publish_time="00:30")
    base = add_solar_features(base)
    base = add_targets(base, flare_df, onset_col)
    base["year"] = year

    tmp = out_path.with_suffix(".parquet.tmp")
    base.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)

    ok_marker.write_text(json.dumps({
        "year": year, "rows": len(base), "cols": len(base.columns),
    }), encoding="utf-8")

    del base; gc.collect()
    return "built"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build gold dataset from raw sources")
    parser.add_argument("--raw-dir",  type=Path, default=Path("data/raw"))
    parser.add_argument("--gold-dir", type=Path, default=Path("data/gold"))
    parser.add_argument("--years",    type=int, nargs="+",
                        default=list(range(2012, 2026)))
    parser.add_argument("--no-overwrite", action="store_true",
                        help="Skip years that already have _SUCCESS.json")
    parser.add_argument("--onset-col", type=str, default="onset_time")
    args = parser.parse_args()

    raw_dir  = args.raw_dir
    gold_dir = args.gold_dir
    gold_dir.mkdir(parents=True, exist_ok=True)

    dayind_path = raw_dir / "dayind_daily.parquet"
    srs_path    = raw_dir / "srs_daily_agg.parquet"
    flare_path  = raw_dir / "flare_onsets.parquet"

    assert dayind_path.exists(), f"Missing: {dayind_path}"
    assert srs_path.exists(),    f"Missing: {srs_path}"
    assert flare_path.exists(),  f"Missing: {flare_path}"

    dayind_df = pd.read_parquet(dayind_path)
    srs_df    = pd.read_parquet(srs_path)
    flare_df  = pd.read_parquet(flare_path)

    dayind_df["date"] = pd.to_datetime(dayind_df["date"]).dt.tz_localize(None).dt.floor("D")
    srs_df["date"]    = pd.to_datetime(srs_df["date"]).dt.tz_localize(None).dt.floor("D")

    assert args.onset_col in flare_df.columns, (
        f"onset column '{args.onset_col}' not in flare_onsets.parquet. "
        f"Columns: {list(flare_df.columns)}"
    )

    print(f"Building gold for years: {args.years}")
    for year in args.years:
        print(f"  year={year} ... ", end="", flush=True)
        try:
            status = build_year(
                year, raw_dir, gold_dir,
                dayind_df, srs_df, flare_df, args.onset_col,
                overwrite=not args.no_overwrite,
            )
            print(status)
        except Exception as e:
            print(f"FAILED: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
