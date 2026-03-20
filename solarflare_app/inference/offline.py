from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..modeling.bundle import load_horizon_bundle


@dataclass
class PredictionBundle:
    horizon: int
    threshold: float
    probability: float
    fire: bool
    policy: dict
    source_ts: str | None


def _align_features(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for col in feature_columns:
        if col in frame.columns:
            out[col] = pd.to_numeric(frame[col], errors="coerce")
        else:
            out[col] = 0.0
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def predict_latest_row(bundle_dir: Path, frame: pd.DataFrame, horizons: list[int] | None = None) -> list[PredictionBundle]:
    root_meta = Path(bundle_dir) / "bundle.json"
    if not root_meta.exists():
        raise FileNotFoundError(f"Missing bundle root metadata: {root_meta}")

    if horizons is None:
        import json
        horizons = json.loads(root_meta.read_text(encoding="utf-8")).get("horizons", [])

    if frame.empty:
        raise ValueError("Input frame is empty.")

    latest = frame.tail(1).copy()
    source_ts = None
    if "ts" in latest.columns:
        source_ts = str(pd.to_datetime(latest["ts"].iloc[0], errors="coerce"))

    predictions: list[PredictionBundle] = []
    for horizon in horizons:
        payload = load_horizon_bundle(bundle_dir, int(horizon))
        meta = payload["metadata"]
        x = _align_features(latest, meta["feature_columns"])
        raw_prob = np.mean([model.predict_proba(x)[:, 1][0] for model in payload["models"]])
        prob = float(payload["calibrator"].predict(np.asarray([raw_prob]))[0])
        threshold = float(meta["threshold"])
        predictions.append(
            PredictionBundle(
                horizon=int(horizon),
                threshold=threshold,
                probability=prob,
                fire=bool(prob >= threshold),
                policy=meta["policy"],
                source_ts=source_ts,
            )
        )
    return predictions
