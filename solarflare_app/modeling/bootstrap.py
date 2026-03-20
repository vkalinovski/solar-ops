from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    n_iterations: int


def build_event_blocks(ts: pd.Series, y_true: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"ts": pd.to_datetime(ts, errors="coerce", utc=True), "y": np.asarray(y_true, dtype=int)})
    frame["block_id"] = (frame["y"].ne(frame["y"].shift(fill_value=0))).cumsum()
    return frame


def block_bootstrap_metric(
    ts: pd.Series,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_iterations: int = 1000,
    random_seed: int = 42,
) -> BootstrapInterval:
    blocks = build_event_blocks(ts, y_true)
    blocks["p"] = np.asarray(y_prob, dtype=float)
    grouped = [g.copy() for _, g in blocks.groupby("block_id", sort=True)]
    rng = np.random.default_rng(random_seed)

    base = float(metric_fn(np.asarray(y_true, dtype=int), np.asarray(y_prob, dtype=float)))
    draws: list[float] = []

    for _ in range(int(n_iterations)):
        sampled = [grouped[idx] for idx in rng.integers(0, len(grouped), size=len(grouped))]
        boot = pd.concat(sampled, axis=0, ignore_index=True)
        draws.append(float(metric_fn(boot["y"].to_numpy(), boot["p"].to_numpy())))

    lower, upper = np.quantile(draws, [0.025, 0.975])
    return BootstrapInterval(
        estimate=base,
        lower=float(lower),
        upper=float(upper),
        n_iterations=int(n_iterations),
    )
