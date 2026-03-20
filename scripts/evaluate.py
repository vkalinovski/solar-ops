from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from solarflare_app.data.gold import load_gold_year, target_column
from solarflare_app.inference.offline import predict_latest_row
from solarflare_app.modeling.bootstrap import block_bootstrap_metric
from solarflare_app.modeling.metrics import brier_score, ece, logloss, pr_auc, roc_auc_rank
from solarflare_app.modeling.bundle import load_horizon_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", default="./bundle")
    parser.add_argument("--gold-root", default="./data/gold")
    parser.add_argument("--year", type=int, default=2025)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle_root = Path(args.bundle_dir)
    root_meta = json.loads((bundle_root / "bundle.json").read_text(encoding="utf-8"))
    horizons = root_meta["horizons"]

    out = {}
    for horizon in horizons:
        payload = load_horizon_bundle(bundle_root, horizon)
        meta = payload["metadata"]
        frame = load_gold_year(Path(args.gold_root), args.year)
        y_col = target_column(horizon)
        x = frame[meta["feature_columns"]].apply(pd.to_numeric, errors="coerce").replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
        raw = sum(model.predict_proba(x)[:, 1] for model in payload["models"]) / len(payload["models"])
        prob = payload["calibrator"].predict(raw)
        y = frame[y_col].astype(int).to_numpy()

        out[str(horizon)] = {
            "roc_auc": roc_auc_rank(y, prob),
            "pr_auc": pr_auc(y, prob),
            "logloss": logloss(y, prob),
            "brier": brier_score(y, prob),
            "ece": ece(y, prob),
            "bootstrap_roc_auc": block_bootstrap_metric(frame["ts"], y, prob, roc_auc_rank).__dict__,
        }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
