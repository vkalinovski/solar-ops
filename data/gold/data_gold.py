from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS            = (60, 120, 360, 720)
ROLL_WINDOWS        = (5, 15, 60, 180, 360, 720, 1440)
LAG_MINUTES         = (1, 5, 15, 60, 180, 360, 720, 1440)
FLARE_COUNT_WINDOWS = (1440, 10080)
MAX_MISSING_XRSB    = 0.02
FILL_XRS_LIMIT      = 60
EPS                 = 1e-12


def expected_minute_index(year: int) -> pd.DatetimeIndex:
    return pd.date_range(
        start=pd.Timestamp(year=year, month=1,  day=1),
        end  =pd.Timestamp(year=year, month=12, day=31, hour=23, minute=59),
        freq ="min",
    )


def repair_minute_grid(df: pd.DataFrame, year: int) -> pd.DataFrame:
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(None)
    df = df.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
    full   = pd.DataFrame({"ts": expected_minute_index(year)})
    merged = full.merge(df, on="ts", how="left")
    for col in ("xrsb", "xrsa"):
        if col in merged.columns:
            merged[col] = (pd.to_numeric(merged[col], errors="coerce")
                           .ffill(limit=FILL_XRS_LIMIT))
    miss = float(merged["xrsb"].isna().mean()) if "xrsb" in merged.columns else 1.0
    if miss > MAX_MISSING_XRSB:
        raise ValueError(f"year={year}: xrsb missing {miss:.2%} after repair")
    return merged


def _min_periods(w: int) -> int:
    if w <= 2:  return 1
    if w <= 6:  return 2
    if w <= 12: return 3
    return min(w, max(5, w // 10))


def _roll_slope(series: pd.Series, w: int, frac: float = 0.2) -> pd.Series:
    mp = _min_periods(w)
    k  = max(1, int(w * frac))
    def _slope(arr: np.ndarray) -> float:
        fin = np.isfinite(arr)
        if fin.sum() < mp: return np.nan
        head = arr[:k][np.isfinite(arr[:k])]
        tail = arr[-k:][np.isfinite(arr[-k:])]
        if head.size == 0 or tail.size == 0: return np.nan
        return float(tail.mean() - head.mean())
    return series.rolling(w, min_periods=mp).apply(_slope, raw=True)


def build_xrs_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_xrsb"] = np.log10(np.clip(pd.to_numeric(df["xrsb"], errors="coerce"), EPS, None))
    if "xrsa" in df.columns:
        df["log_xrsa"] = np.log10(np.clip(pd.to_numeric(df["xrsa"], errors="coerce"), EPS, None))
        df["log_xrsb_over_xrsa"] = df["log_xrsb"] - df["log_xrsa"]
    for lag in LAG_MINUTES:
        df[f"log_xrsb_lag{lag}"] = df["log_xrsb"].shift(lag)
        if "log_xrsa" in df.columns:
            df[f"log_xrsa_lag{lag}"] = df["log_xrsa"].shift(lag)
    for w in ROLL_WINDOWS:
        mp = _min_periods(w)
        s  = df["log_xrsb"]
        df[f"log_xrsb_roll{w}_mean"] = s.rolling(w, min_periods=mp).mean()
        df[f"log_xrsb_roll{w}_std"]  = s.rolling(w, min_periods=mp).std()
        df[f"log_xrsb_roll{w}_min"]  = s.rolling(w, min_periods=mp).min()
        df[f"log_xrsb_roll{w}_max"]  = s.rolling(w, min_periods=mp).max()
        if "log_xrsa" in df.columns:
            a = df["log_xrsa"]
            df[f"log_xrsa_roll{w}_mean"] = a.rolling(w, min_periods=mp).mean()
            df[f"log_xrsa_roll{w}_std"]  = a.rolling(w, min_periods=mp).std()
    for w in (60, 180, 360, 720, 1440):
        df[f"log_xrsb_roll{w}_slope"] = _roll_slope(df["log_xrsb"], w)
    df["log_xrsb_d24h"] = df["log_xrsb"] - df["log_xrsb"].shift(1440)
    return df


def add_solar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "xray_bkgd_flux" in df.columns:
        df["log10_xray_bkgd_flux"] = np.log10(
            np.clip(pd.to_numeric(df["xray_bkgd_flux"], errors="coerce"), EPS, None))
        df["log10_xray_bkgd_flux_roll7_mean"] = (
            df["log10_xray_bkgd_flux"].rolling(10080, min_periods=1440).mean())
    for c in ("sunspot_number", "f107"):
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
    if "n_regions" in df.columns:
        df["n_regions"] = pd.to_numeric(df["n_regions"], errors="coerce").fillna(0)
        df["sum_area"]  = pd.to_numeric(df.get("sum_area", 0), errors="coerce").fillna(0)
        df["max_area"]  = pd.to_numeric(df.get("max_area", 0), errors="coerce").fillna(0)
    return df


def join_daily_asof(df_min: pd.DataFrame, df_daily: pd.DataFrame | None,
                    shift_days: int = 1, publish_time: str = "00:00") -> pd.DataFrame:
    if df_daily is None or len(df_daily) == 0:
        return df_min
    df_min   = df_min.copy()
    df_daily = df_daily.copy()
    df_min["ts"]     = pd.to_datetime(df_min["ts"]).dt.tz_localize(None)
    df_daily["date"] = pd.to_datetime(df_daily["date"]).dt.tz_localize(None)
    hh, mm = map(int, publish_time.split(":"))
    df_daily["_avail"] = (df_daily["date"]
                          + pd.Timedelta(days=shift_days)
                          + pd.Timedelta(hours=hh, minutes=mm))
    df_daily = df_daily.sort_values("_avail").reset_index(drop=True)
    df_min   = df_min.sort_values("ts").reset_index(drop=True)
    merged   = pd.merge_asof(df_min, df_daily.drop(columns=["date"]),
                              left_on="ts", right_on="_avail",
                              direction="backward", allow_exact_matches=True)
    return merged.drop(columns=["_avail"], errors="ignore")


def add_targets(df: pd.DataFrame, flare_df: pd.DataFrame,
                onset_col: str = "onset_time") -> pd.DataFrame:
    df     = df.copy()
    flare  = flare_df.copy()
    flare[onset_col] = pd.to_datetime(flare[onset_col]).dt.tz_localize(None)
    onsets = np.sort(flare[onset_col].values.astype("datetime64[ns]"))
    ts_arr = pd.to_datetime(df["ts"]).values.astype("datetime64[ns]")
    for h in HORIZONS:
        left  = np.searchsorted(onsets, ts_arr, side="right")
        right = np.searchsorted(onsets,
                                ts_arr + np.timedelta64(h, "m").astype("timedelta64[ns]"),
                                side="right")
        df[f"y_onset_m1p_in_{h}m"] = (right > left).astype(np.int8)
    for w in FLARE_COUNT_WINDOWS:
        left  = np.searchsorted(onsets,
                                ts_arr - np.timedelta64(w, "m").astype("timedelta64[ns]"),
                                side="right")
        right = np.searchsorted(onsets, ts_arr, side="right")
        df[f"flare_cnt_past_{w}m"] = (right - left).astype(np.int32)
    return df


def build_year(year: int, raw_dir: Path, gold_dir: Path,
               dayind_df: pd.DataFrame, srs_df: pd.DataFrame,
               flare_df: pd.DataFrame, onset_col: str = "onset_time",
               overwrite: bool = True) -> str:
    out_path  = gold_dir / f"year={year}" / "data.parquet"
    ok_marker = out_path.parent / "_SUCCESS.json"
    if not overwrite and out_path.exists() and ok_marker.exists():
        return "skipped"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    xrs_dir    = raw_dir / "xrs"
    candidates = [xrs_dir / f"goes_xrs_{year}.parquet",
                  xrs_dir / f"{year}.parquet",
                  xrs_dir / f"xrs_{year}.csv",
                  xrs_dir / f"goes_xrs_{year}.csv"]
    base = None
    for p in candidates:
        if p.exists():
            base = pd.read_parquet(p) if p.suffix == ".parquet" \
                   else pd.read_csv(p, parse_dates=["ts"])
            break
    if base is None:
        raise FileNotFoundError(f"XRS data for year={year} not found in {xrs_dir}")

    base = repair_minute_grid(base, year)
    base = build_xrs_features(base)
    base = join_daily_asof(base, dayind_df, shift_days=1, publish_time="00:00")
    base = join_daily_asof(base, srs_df,    shift_days=1, publish_time="00:30")
    base = add_solar_features(base)
    base = add_targets(base, flare_df, onset_col=onset_col)
    base["year"] = year

    tmp = out_path.with_suffix(".parquet.tmp")
    base.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)

    ok_marker.write_text(json.dumps({
        "year": year, "rows": len(base), "feature_cols": len(base.columns),
        "target_rates": {f"y_onset_m1p_in_{h}m": float(base[f"y_onset_m1p_in_{h}m"].mean())
                         for h in HORIZONS if f"y_onset_m1p_in_{h}m" in base.columns},
    }), encoding="utf-8")
    del base; gc.collect()
    return "built"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GOLD dataset from RAW")
    parser.add_argument("--raw-dir",      type=Path, default=Path("data/raw"))
    parser.add_argument("--gold-dir",     type=Path, default=Path("data/gold"))
    parser.add_argument("--years",        type=int, nargs="+",
                        default=list(range(2012, 2026)))
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--onset-col",    type=str, default="onset_time")
    args = parser.parse_args()

    raw_dir  = args.raw_dir
    gold_dir = args.gold_dir
    gold_dir.mkdir(parents=True, exist_ok=True)

    for p in (raw_dir / "dayind_daily.parquet",
              raw_dir / "srs_daily_agg.parquet",
              raw_dir / "flare_onsets.parquet"):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}  →  run python data/raw/fetch.py first")

    dayind_df = pd.read_parquet(raw_dir / "dayind_daily.parquet")
    srs_df    = pd.read_parquet(raw_dir / "srs_daily_agg.parquet")
    flare_df  = pd.read_parquet(raw_dir / "flare_onsets.parquet")
    dayind_df["date"] = pd.to_datetime(dayind_df["date"]).dt.tz_localize(None).dt.floor("D")
    srs_df["date"]    = pd.to_datetime(srs_df["date"]).dt.tz_localize(None).dt.floor("D")

    if args.onset_col not in flare_df.columns:
        raise KeyError(f"Column '{args.onset_col}' not found. "
                       f"Available: {list(flare_df.columns)}")

    print(f"Building GOLD for years: {args.years}")
    for year in args.years:
        print(f"  year={year} ... ", end="", flush=True)
        try:
            print(build_year(year, raw_dir, gold_dir, dayind_df, srs_df, flare_df,
                             onset_col=args.onset_col, overwrite=not args.no_overwrite))
        except Exception as exc:
            print(f"FAILED: {exc}")
    print("Done.")


if __name__ == "__main__":
    main()
