from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


META_COLUMNS = {"ts", "year"}
OPTIONAL_META_COLUMNS = {"source", "features_mode"}
TARGET_PREFIX = "y_onset_m1p_in_"


def target_column(horizon: int) -> str:
    return f"{TARGET_PREFIX}{int(horizon)}m"


def gold_year_path(gold_root: Path, year: int) -> Path:
    return Path(gold_root) / f"year={int(year)}" / "data.parquet"


def list_gold_years(gold_root: Path) -> list[int]:
    years = []
    for path in Path(gold_root).glob("year=*"):
        try:
            years.append(int(path.name.split("=")[-1]))
        except ValueError:
            continue
    return sorted(years)


def load_gold_year(gold_root: Path, year: int, columns: Sequence[str] | None = None) -> pd.DataFrame:
    path = gold_year_path(gold_root, year)
    if not path.exists():
        raise FileNotFoundError(f"Missing GOLD parquet for year={year}: {path}")
    df = pd.read_parquet(path, columns=list(columns) if columns is not None else None)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=False)
    return df


def load_gold_years(gold_root: Path, years: Iterable[int], columns: Sequence[str] | None = None) -> pd.DataFrame:
    frames = [load_gold_year(gold_root, year, columns=columns) for year in years]
    if not frames:
        return pd.DataFrame(columns=list(columns or []))
    return pd.concat(frames, axis=0, ignore_index=True)


def detect_feature_columns(df: pd.DataFrame) -> list[str]:
    blocked = set(META_COLUMNS) | set(OPTIONAL_META_COLUMNS)
    blocked |= {c for c in df.columns if c.startswith(TARGET_PREFIX)}
    feature_cols = [c for c in df.columns if c not in blocked]
    return sorted(feature_cols)


def split_calibration_frame(df: pd.DataFrame, split_month: int = 9) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "ts" not in df.columns:
        raise ValueError("Calibration split requires a ts column.")
    ts = pd.to_datetime(df["ts"], errors="coerce")
    mask = ts.dt.month < int(split_month)
    cal_a = df.loc[mask].copy().reset_index(drop=True)
    cal_b = df.loc[~mask].copy().reset_index(drop=True)
    if cal_a.empty or cal_b.empty:
        raise ValueError("CAL_A/CAL_B split is empty. Check calibration year coverage.")
    return cal_a, cal_b


def sample_binary_frame(df: pd.DataFrame, y_col: str, neg_pos_ratio: int = 3, seed: int = 42) -> pd.DataFrame:
    pos = df.loc[df[y_col] == 1]
    neg = df.loc[df[y_col] == 0]
    if pos.empty:
        return df.sample(n=min(len(df), 250_000), random_state=seed).reset_index(drop=True)

    n_neg = min(len(neg), len(pos) * int(neg_pos_ratio))
    neg_sample = neg.sample(n=n_neg, random_state=seed) if n_neg < len(neg) else neg
    return (
        pd.concat([pos, neg_sample], axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


@dataclass(frozen=True)
class TimeSplit:
    train_years: tuple[int, ...]
    calib_year: int
    holdout_year: int
    exclude_years: tuple[int, ...] = (2018, 2019)

    def usable_years(self, available_years: Sequence[int]) -> list[int]:
        return [y for y in available_years if y not in self.exclude_years]

    def default_train_years(self, available_years: Sequence[int]) -> tuple[int, ...]:
        years = [y for y in self.usable_years(available_years) if y < self.calib_year]
        years = [y for y in years if y != self.holdout_year]
        return tuple(years)
