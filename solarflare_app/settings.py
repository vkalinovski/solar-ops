from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from . import DEFAULT_HORIZONS


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return default if raw is None or raw == "" else raw


def _env_csv_int(name: str, default: Sequence[int]) -> tuple[int, ...]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return tuple(default)
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


def _env_csv_str(name: str, default: Sequence[str]) -> list[str]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return list(default)
    return [x.strip() for x in raw.split(",") if x.strip()]


@dataclass(frozen=True)
class RuntimeSettings:
    bundle_dir: Path = Path(_env_str("BUNDLE_DIR", "./bundle"))
    db_path: Path = Path(_env_str("DB_PATH", "./data/artifacts/preds.sqlite"))
    update_every_sec: int = _env_int("UPDATE_EVERY_SEC", 60)
    lookback_min: int = _env_int("LOOKBACK_MIN", 72 * 60)
    max_stale_min: float = _env_float("MAX_STALE_MIN", 60.0)
    host: str = _env_str("HOST", "0.0.0.0")
    port: int = _env_int("PORT", 8000)
    log_level: str = _env_str("LOG_LEVEL", "INFO").upper()
    log_json: bool = _env_str("LOG_JSON", "0").lower() in {"1", "true", "yes"}
    allow_origins: list[str] = field(default_factory=lambda: _env_csv_str("ALLOW_ORIGINS", ["*"]))


@dataclass(frozen=True)
class TrainingSettings:
    gold_root: Path = Path(_env_str("GOLD_ROOT", "./data/gold"))
    bundle_dir: Path = Path(_env_str("BUNDLE_DIR", "./bundle"))
    horizons: tuple[int, ...] = _env_csv_int("HORIZONS", DEFAULT_HORIZONS)
    calib_year: int = _env_int("CALIB_YEAR", 2024)
    holdout_year: int = _env_int("HOLDOUT_YEAR", 2025)
    exclude_years: tuple[int, ...] = _env_csv_int("EXCLUDE_YEARS", (2018, 2019))
    neg_pos_ratio: int = _env_int("NEG_POS_RATIO", 3)
    max_train_rows: int = _env_int("MAX_TRAIN_ROWS", 2_000_000)
    calibration_split_month: int = _env_int("CALIBRATION_SPLIT_MONTH", 9)
    seeds: tuple[int, ...] = _env_csv_int("SEEDS", (0, 1, 2, 3, 4))
