from __future__ import annotations

import argparse
import io
import json
import re
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import numpy as np
import pandas as pd

GOES_SATELLITE = {
    2012: 15, 2013: 15, 2014: 15, 2015: 15, 2016: 15,
    2017: 15, 2018: 16, 2019: 16, 2020: 16, 2021: 16,
    2022: 16, 2023: 16, 2024: 16, 2025: 16,
}
GOES_LEGACY  = {13, 14, 15}
GOES_RCLASS  = {16, 17, 18}

NOAA_ARCHIVE_CSV = (
    "https://satdat.ngdc.noaa.gov/sem/goes/data/avg/{year}/{month:02d}/"
    "goes{sat}/csv/g{sat}_xrs_2s_{year}{month:02d}01_{year}{month:02d}{last_day:02d}.csv"
)
GOSR_JSON_PRIMARY = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"
GOSR_THREDDS = (
    "https://thredds.ucar.edu/thredds/fileServer/"
    "satellite/goes16/xrs/1m/{year}/{year}{month:02d}/"
    "OR_SGPS-L2-XRSF-M_G16_{year}{month:02d}01_{year}{month:02d}{ld:02d}.nc"
)
NOAA_F107_URL = (
    "https://services.swpc.noaa.gov/json/"
    "solar-cycle/observed-solar-cycle-indices.json"
)
SILSO_SSN_URL = "http://www.sidc.be/silso/DATA/SN_d_tot_V2.0.txt"
SRS_BASE      = "https://services.swpc.noaa.gov/text/solar-regions.txt"
FLARES_7DAY_URL = "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json"
NOAA_FLARES_URL = (
    "https://www.ngdc.noaa.gov/stp/space-weather/solar-data/"
    "solar-features/solar-flares/x-rays/goes/xrs/"
    "goes-xrs-report_{year}.txt"
)


def _get(url: str, retries: int = 3, pause: float = 2.0) -> bytes:
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "solar-ops/1.0"})
            with urlopen(req, timeout=60) as r:
                return r.read()
        except (HTTPError, URLError):
            if attempt == retries - 1:
                raise
            time.sleep(pause * (attempt + 1))
    raise RuntimeError("unreachable")


def _last_day(year: int, month: int) -> int:
    import calendar
    return calendar.monthrange(year, month)[1]


def _fetch_goes_legacy_year(year: int, sat: int) -> pd.DataFrame:
    frames = []
    for month in range(1, 13):
        ld  = _last_day(year, month)
        url = NOAA_ARCHIVE_CSV.format(year=year, month=month, sat=sat, last_day=ld)
        try:
            raw = _get(url)
        except Exception:
            try:
                raw = _get(url.replace(".csv", ".csv.gz"))
            except Exception:
                continue
        try:
            df = pd.read_csv(io.BytesIO(raw), comment="#", low_memory=False)
        except Exception:
            continue
        col_map = {}
        for c in df.columns:
            cl = c.lower().strip()
            if "b_flux" in cl or "xrsb" in cl:
                col_map[c] = "xrsb"
            elif "a_flux" in cl or "xrsa" in cl:
                col_map[c] = "xrsa"
            elif "time_tag" in cl or cl in ("time", "datetime", "date_time"):
                col_map[c] = "ts"
        df = df.rename(columns=col_map)
        if "ts" not in df.columns:
            df = df.rename(columns={df.columns[0]: "ts"})
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
        df = df.dropna(subset=["ts"])
        for c in ("xrsb", "xrsa"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        frames.append(df[["ts"] + [c for c in ("xrsb", "xrsa") if c in df.columns]])
    if not frames:
        raise RuntimeError(f"No NGDC data for GOES-{sat} year={year}")
    out = (pd.concat(frames, ignore_index=True)
           .sort_values("ts").drop_duplicates("ts", keep="last"))
    out = out.set_index("ts").resample("1min").mean().reset_index()
    out["ts"] = out["ts"].dt.tz_localize(None)
    return out


def _fetch_gosr_from_json(url: str) -> pd.DataFrame:
    data = json.loads(_get(url))
    rows = [{"ts": r.get("time_tag"), "xrsb": r.get("flux")}
            for r in data if "0.1-0.8" in str(r.get("energy", ""))]
    if not rows:
        raise ValueError(f"No XRS-B records in {url}")
    df = pd.DataFrame(rows)
    df["ts"]   = pd.to_datetime(df["ts"], errors="coerce", utc=True).dt.tz_localize(None)
    df["xrsb"] = pd.to_numeric(df["xrsb"], errors="coerce")
    return df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)


def _fetch_gosr_netcdf_year(year: int) -> pd.DataFrame:
    import netCDF4 as nc4
    frames = []
    for month in range(1, 13):
        ld  = _last_day(year, month)
        url = GOSR_THREDDS.format(year=year, month=month, ld=ld)
        try:
            raw = _get(url)
        except Exception:
            continue
        with nc4.Dataset("inmemory.nc", memory=raw) as ds:
            time_var = ds.variables.get("time")
            xrsb_var = ds.variables.get("xrsb") or ds.variables.get("B_FLUX")
            if time_var is None or xrsb_var is None:
                continue
            units = getattr(time_var, "units", "seconds since 2000-01-01 12:00:00")
            times = nc4.num2date(time_var[:], units)
            ts    = pd.to_datetime([str(t) for t in times], errors="coerce", utc=True)
            df    = pd.DataFrame({"ts": ts.tz_localize(None),
                                  "xrsb": np.array(xrsb_var[:], dtype=float)})
            frames.append(df)
    if not frames:
        raise RuntimeError(f"THREDDS fetch failed for year={year}")
    out = pd.concat(frames, ignore_index=True).sort_values("ts")
    return out.set_index("ts").resample("1min").mean().reset_index()


def _fetch_goes_gosr_year(year: int) -> pd.DataFrame:
    try:
        return _fetch_gosr_netcdf_year(year)
    except ImportError:
        pass
    import datetime
    if year == datetime.datetime.utcnow().year:
        df = _fetch_gosr_from_json(GOSR_JSON_PRIMARY)
        df = df[df["ts"].dt.year == year].copy()
        if not df.empty:
            return df
    raise RuntimeError(
        f"Cannot fetch GOES-R year={year} without netCDF4. "
        "Run: pip install netCDF4"
    )


def fetch_xrs_year(year: int, out_dir: Path) -> Path:
    sat     = GOES_SATELLITE.get(year, 16)
    xrs_dir = Path(out_dir) / "xrs"
    xrs_dir.mkdir(parents=True, exist_ok=True)
    out_path = xrs_dir / f"goes_xrs_{year}.parquet"
    print(f"  XRS year={year} GOES-{sat} ... ", end="", flush=True)
    df = _fetch_goes_legacy_year(year, sat) if sat in GOES_LEGACY \
         else _fetch_goes_gosr_year(year)
    for c in ("xrsb", "xrsa"):
        if c not in df.columns:
            df[c] = np.nan
    df = df[["ts", "xrsb", "xrsa"]].sort_values("ts").reset_index(drop=True)
    df.to_parquet(out_path, index=False)
    print(f"{len(df):,} rows → {out_path}")
    return out_path


def fetch_dayind(out_dir: Path) -> Path:
    out_path = Path(out_dir) / "dayind_daily.parquet"
    print("  F10.7 / SSN ... ", end="", flush=True)
    raw      = json.loads(_get(NOAA_F107_URL))
    rows     = [{"date": r.get("time-tag"), "f107": r.get("f10.7"),
                 "ssn_swpc": r.get("ssn")} for r in raw]
    f107_df  = pd.DataFrame(rows)
    f107_df["date"] = pd.to_datetime(f107_df["date"], errors="coerce").dt.floor("D")
    for c in ("f107", "ssn_swpc"):
        f107_df[c] = pd.to_numeric(f107_df[c], errors="coerce")
    f107_df = f107_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    try:
        raw_ssn  = _get(SILSO_SSN_URL).decode("ascii", errors="replace")
        ssn_rows = []
        for line in raw_ssn.splitlines():
            p = line.split()
            if len(p) < 4:
                continue
            try:
                ssn_rows.append({"date": pd.Timestamp(int(p[0]), int(p[1]), int(p[2])),
                                 "sunspot_number": float(p[3]) if p[3] != "-1" else np.nan})
            except (ValueError, IndexError):
                continue
        f107_df = f107_df.merge(pd.DataFrame(ssn_rows).dropna(subset=["date"]),
                                on="date", how="left")
    except Exception:
        f107_df["sunspot_number"] = f107_df.get("ssn_swpc", np.nan)
    try:
        bkgd = json.loads(_get(
            "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json"))
        bkgd_df = (pd.DataFrame([{
            "date": pd.to_datetime(r.get("begin_datetime", "")).floor("D"),
            "xray_bkgd_flux": r.get("max_flux")} for r in bkgd])
            .groupby("date", as_index=False)
            .agg({"xray_bkgd_flux": "min"}))
        bkgd_df["xray_bkgd_flux"] = pd.to_numeric(bkgd_df["xray_bkgd_flux"], errors="coerce")
        f107_df = f107_df.merge(bkgd_df, on="date", how="left")
    except Exception:
        f107_df["xray_bkgd_flux"] = np.nan
    f107_df.to_parquet(out_path, index=False)
    print(f"{len(f107_df):,} days → {out_path}")
    return out_path


def fetch_srs(out_dir: Path) -> Path:
    out_path = Path(out_dir) / "srs_daily_agg.parquet"
    print("  SRS solar regions ... ", end="", flush=True)
    rows = []
    try:
        text = _get(SRS_BASE).decode("utf-8", errors="replace")
        cur_date = None
        for line in text.splitlines():
            dm = re.match(r":Issued:\s+(\d{4}\s+\w+\s+\d{2})", line)
            if dm:
                try: cur_date = pd.to_datetime(dm.group(1))
                except Exception: cur_date = None
            if cur_date is None:
                continue
            rm = re.match(r"\s*(\d{4})\s+\w\s+([\d.]+)\s+([\d.]+)\s+(\d+)", line)
            if rm:
                rows.append({"date": cur_date.floor("D"),
                             "region_id": rm.group(1), "area": float(rm.group(4))})
    except Exception:
        pass
    if rows:
        df  = pd.DataFrame(rows)
        agg = (df.groupby("date")
                 .agg(n_regions=("region_id", "count"),
                      sum_area=("area", "sum"),
                      max_area=("area", "max"))
                 .reset_index())
    else:
        agg = pd.DataFrame(columns=["date", "n_regions", "sum_area", "max_area"])
    agg.to_parquet(out_path, index=False)
    print(f"{len(agg):,} days → {out_path}")
    return out_path


def fetch_flares(out_dir: Path, years: list[int]) -> Path:
    out_path = Path(out_dir) / "flare_onsets.parquet"
    print("  M1+ flare onsets ... ", end="", flush=True)
    all_rows: list[dict] = []
    for year in years:
        try:
            raw = _get(NOAA_FLARES_URL.format(year=year)).decode("utf-8", errors="replace")
        except Exception:
            continue
        for line in raw.splitlines():
            parts = line.split()
            if len(parts) < 7:
                continue
            fc = parts[6].upper()
            if not (fc.startswith("M") or fc.startswith("X")):
                continue
            try:
                onset_ts = pd.to_datetime(
                    f"{parts[0]} {parts[1].zfill(4)}", format="%Y%m%d %H%M", utc=True
                ).tz_localize(None)
                all_rows.append({"onset_time": onset_ts, "flare_class": fc, "year": year})
            except Exception:
                continue
    try:
        for rec in json.loads(_get(FLARES_7DAY_URL)):
            fc = str(rec.get("max_class", "")).upper()
            if not (fc.startswith("M") or fc.startswith("X")):
                continue
            bt = rec.get("begin_datetime") or rec.get("begin_time")
            if not bt:
                continue
            try:
                onset_ts = pd.to_datetime(bt, utc=True).tz_localize(None)
                all_rows.append({"onset_time": onset_ts, "flare_class": fc,
                                 "year": onset_ts.year})
            except Exception:
                continue
    except Exception:
        pass
    if not all_rows:
        raise RuntimeError("Could not fetch any M1+ flare records.")
    df = (pd.DataFrame(all_rows)
          .dropna(subset=["onset_time"])
          .sort_values("onset_time")
          .drop_duplicates("onset_time", keep="first")
          .reset_index(drop=True))
    df["onset_time"] = pd.to_datetime(df["onset_time"])
    df.to_parquet(out_path, index=False)
    print(f"{len(df):,} events → {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw data for solar-ops")
    parser.add_argument("--out-dir",   type=Path, default=Path("data/raw"))
    parser.add_argument("--years",     type=int, nargs="+",
                        default=list(range(2012, 2026)))
    parser.add_argument("--skip-xrs",  action="store_true")
    parser.add_argument("--skip-done", action="store_true")
    args = parser.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "xrs").mkdir(exist_ok=True)

    if not args.skip_xrs:
        for year in args.years:
            dest = out / "xrs" / f"goes_xrs_{year}.parquet"
            if args.skip_done and dest.exists():
                print(f"  Skipping XRS year={year} (exists)")
                continue
            try:
                fetch_xrs_year(year, out)
            except Exception as exc:
                print(f"  FAILED year={year}: {exc}")

    fetch_dayind(out)
    fetch_srs(out)
    fetch_flares(out, args.years)
    print("Done.")


if __name__ == "__main__":
    main()
