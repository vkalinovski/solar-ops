from __future__ import annotations

import argparse
import json

from solarflare_app.modeling.trainer import HorizonTrainer
from solarflare_app.settings import TrainingSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=720)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = TrainingSettings()
    settings = TrainingSettings(
        gold_root=base.gold_root,
        bundle_dir=base.bundle_dir,
        horizons=(args.horizon,),
        calib_year=base.calib_year,
        holdout_year=base.holdout_year,
        exclude_years=base.exclude_years,
        neg_pos_ratio=base.neg_pos_ratio,
        max_train_rows=base.max_train_rows,
        calibration_split_month=base.calibration_split_month,
        seeds=base.seeds,
    )
    trainer = HorizonTrainer(settings=settings, bundle_dir=settings.bundle_dir)
    result = trainer.train_horizon(args.horizon)
    print(json.dumps({str(args.horizon): {"threshold": result.threshold, "metrics": result.metrics}}, indent=2))


if __name__ == "__main__":
    main()
