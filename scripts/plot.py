from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", default="./bundle")
    parser.add_argument("--output-dir", default="./data/artifacts/plots")
    return parser.parse_args()


def reliability_points(frame: pd.DataFrame, prob_col: str, y_col: str, bins: int = 12) -> tuple[list[float], list[float]]:
    edges = pd.interval_range(start=0.0, end=1.0, periods=bins)
    xs, ys = [], []
    for interval in edges:
        mask = frame[prob_col].between(interval.left, interval.right, inclusive="right")
        part = frame.loc[mask]
        if part.empty:
            continue
        xs.append(float(part[prob_col].mean()))
        ys.append(float(part[y_col].mean()))
    return xs, ys


def main() -> None:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    root_meta = pd.read_json(bundle_dir / "bundle.json")
    horizons = [int(h) for h in root_meta["horizons"].tolist()]

    for horizon in horizons:
        h_dir = bundle_dir / "horizons" / f"{horizon}m"
        hold = pd.read_parquet(h_dir / "holdout_predictions.parquet")
        y_col = [c for c in hold.columns if c.startswith("y_onset_m1p_in_")][0]

        xs_raw, ys_raw = reliability_points(hold, "raw_prob", y_col)
        xs_cal, ys_cal = reliability_points(hold, "prob", y_col)

        plt.figure(figsize=(6, 5))
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        plt.scatter(xs_raw, ys_raw, label="raw")
        plt.scatter(xs_cal, ys_cal, label="calibrated")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Observed frequency")
        plt.title(f"Calibration before/after {horizon}m")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"calibration_{horizon}m.png", dpi=160)
        plt.close()


if __name__ == "__main__":
    main()
