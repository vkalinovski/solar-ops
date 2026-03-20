from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import classification_metrics, tss


@dataclass(frozen=True)
class PolicyConfig:
    min_precision: float
    max_alerts_per_day: float
    persist_min: int
    cooldown_min: int


def alerts_per_day(ts: pd.Series, n_alerts: int) -> float:
    ts = pd.to_datetime(ts, errors="coerce", utc=True).dropna()
    if len(ts) < 2:
        return float("nan")
    days = max((ts.max() - ts.min()).total_seconds() / 86400.0, 1e-9)
    return float(n_alerts / days)


def simulate_alert_times(ts: pd.Series, y_prob: np.ndarray, threshold: float, persist_min: int, cooldown_min: int) -> list[pd.Timestamp]:
    ts = pd.to_datetime(ts, errors="coerce", utc=True)
    prob = np.asarray(y_prob, dtype=float)
    alerts: list[pd.Timestamp] = []

    i = 0
    while i < len(prob):
        if prob[i] >= threshold:
            j = i
            run = 0
            while j < len(prob) and prob[j] >= threshold:
                run += 1
                if run >= max(1, int(persist_min)):
                    stamp = ts.iloc[j]
                    if pd.notna(stamp):
                        alerts.append(pd.Timestamp(stamp))
                    i = j + int(cooldown_min)
                    break
                j += 1
            else:
                i = j + 1
        else:
            i += 1
    return alerts


def event_start_times(ts: pd.Series, y_true: np.ndarray) -> list[pd.Timestamp]:
    ts = pd.to_datetime(ts, errors="coerce", utc=True)
    y = np.asarray(y_true, dtype=np.int8)
    starts = np.where((y == 1) & (np.r_[0, y[:-1]] == 0))[0]
    return [pd.Timestamp(ts.iloc[i]) for i in starts if pd.notna(ts.iloc[i])]


def event_recall(ts: pd.Series, y_true: np.ndarray, y_prob: np.ndarray, threshold: float, horizon_min: int, persist_min: int, cooldown_min: int) -> tuple[float, int, int]:
    alerts = simulate_alert_times(ts, y_prob, threshold, persist_min, cooldown_min)
    starts = event_start_times(ts, y_true)
    if not starts:
        return float("nan"), 0, 0

    caught = 0
    for start in starts:
        left = start - pd.Timedelta(minutes=int(horizon_min))
        ok = any((alert > left) and (alert <= start) for alert in alerts)
        caught += int(ok)
    return float(caught / len(starts)), caught, len(starts)


def lead_times_minutes(ts: pd.Series, y_true: np.ndarray, y_prob: np.ndarray, threshold: float, horizon_min: int, persist_min: int, cooldown_min: int) -> list[float]:
    alerts = simulate_alert_times(ts, y_prob, threshold, persist_min, cooldown_min)
    starts = event_start_times(ts, y_true)
    out: list[float] = []
    for start in starts:
        window_left = start - pd.Timedelta(minutes=int(horizon_min))
        eligible = [alert for alert in alerts if (alert > window_left) and (alert <= start)]
        if eligible:
            out.append(float((start - min(eligible)).total_seconds() / 60.0))
    return out


def choose_threshold(
    ts: pd.Series,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    horizon_min: int,
    policy: PolicyConfig,
    grid: np.ndarray | None = None,
) -> dict:
    p = np.asarray(y_prob, dtype=float)
    y = np.asarray(y_true, dtype=np.int8)
    if grid is None:
        grid = np.unique(np.quantile(p, np.linspace(0.01, 0.995, 500)))
        grid = np.unique(np.r_[grid, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.40])

    best = None
    relaxed = None

    for threshold in grid:
        alerts = simulate_alert_times(ts, p, float(threshold), policy.persist_min, policy.cooldown_min)
        apd = alerts_per_day(ts, len(alerts))
        event_score, caught, total = event_recall(
            ts=ts,
            y_true=y,
            y_prob=p,
            threshold=float(threshold),
            horizon_min=horizon_min,
            persist_min=policy.persist_min,
            cooldown_min=policy.cooldown_min,
        )
        report = classification_metrics(y, p, float(threshold))
        score = 0.55 * report["tss"] + 0.25 * report["f1"] + 0.20 * (event_score if np.isfinite(event_score) else 0.0)
        candidate = {
            "threshold": float(threshold),
            "score": float(score),
            "alerts_per_day": float(apd),
            "event_recall": float(event_score),
            "caught_events": int(caught),
            "total_events": int(total),
            **report,
        }

        if relaxed is None or candidate["score"] > relaxed["score"]:
            relaxed = candidate

        if candidate["precision"] + 1e-12 < policy.min_precision:
            continue
        if np.isfinite(apd) and apd > policy.max_alerts_per_day + 1e-12:
            continue
        if best is None or candidate["score"] > best["score"]:
            best = candidate

    if best is None:
        assert relaxed is not None
        relaxed["selection_reason"] = "relaxed_no_feasible_threshold"
        return relaxed

    best["selection_reason"] = "strict_policy"
    return best
