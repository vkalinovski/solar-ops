from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from solarflare_app.inference.offline import predict_latest_row
from solarflare_app.inference.online import run_live_demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", default="./bundle")
    parser.add_argument("--mode", choices=["live", "table"], default="live")
    parser.add_argument("--input", help="Parquet or CSV file with a gold-like feature table.")
    parser.add_argument("--output", help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir)

    if args.mode == "live":
        payload = run_live_demo(bundle_dir=bundle_dir)
    else:
        if not args.input:
            raise SystemExit("--input is required in table mode.")
        path = Path(args.input)
        if path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(path)
        preds = predict_latest_row(bundle_dir=bundle_dir, frame=frame)
        payload = {
            "mode": "table",
            "predictions": [
                {
                    "horizon": item.horizon,
                    "probability": item.probability,
                    "threshold": item.threshold,
                    "fire": item.fire,
                    "policy": item.policy,
                    "source_ts": item.source_ts,
                }
                for item in preds
            ],
        }

    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
