from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from ..data.gold import (
    TimeSplit,
    detect_feature_columns,
    load_gold_year,
    sample_binary_frame,
    split_calibration_frame,
    target_column,
)
from ..settings import TrainingSettings
from .bundle import BundleWriter
from .calibration import choose_best_calibrator
from .metrics import summary_metrics
from .policy import PolicyConfig, choose_threshold, lead_times_minutes


DEFAULT_POLICY_BY_HORIZON = {
    60: PolicyConfig(min_precision=0.08, max_alerts_per_day=3.0, persist_min=3, cooldown_min=30),
    120: PolicyConfig(min_precision=0.10, max_alerts_per_day=3.0, persist_min=3, cooldown_min=60),
    360: PolicyConfig(min_precision=0.12, max_alerts_per_day=5.0, persist_min=5, cooldown_min=120),
    720: PolicyConfig(min_precision=0.15, max_alerts_per_day=5.0, persist_min=5, cooldown_min=240),
}


def _base_catboost_params(horizon: int, params_override: dict | None = None) -> dict:
    horizon = int(horizon)
    depth = 8 if horizon in {60, 120} else 7
    base = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "iterations": 2500,
        "learning_rate": 0.03,
        "depth": depth,
        "l2_leaf_reg": 6.0,
        "random_strength": 1.0,
        "bagging_temperature": 1.0,
        "border_count": 255,
        "task_type": "CPU",
        "allow_writing_files": False,
        "verbose": False,
    }
    if params_override:
        base.update(params_override)
    return base


@dataclass
class TrainArtifacts:
    horizon: int
    threshold: float
    metrics: dict
    diagnostics: dict
    policy: PolicyConfig


class HorizonTrainer:
    def __init__(self, settings: TrainingSettings, bundle_dir: Path):
        self.settings = settings
        self.bundle_writer = BundleWriter(bundle_dir)

    def _load_splits(self, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        target = target_column(horizon)
        split = TimeSplit(
            train_years=tuple(),
            calib_year=self.settings.calib_year,
            holdout_year=self.settings.holdout_year,
            exclude_years=self.settings.exclude_years,
        )
        available = split.usable_years(range(2012, self.settings.holdout_year + 1))
        train_years = split.default_train_years(available)

        train_parts = []
        total_rows = 0
        for year in train_years:
            frame = load_gold_year(self.settings.gold_root, year)
            frame = sample_binary_frame(frame, y_col=target, neg_pos_ratio=self.settings.neg_pos_ratio, seed=year)
            train_parts.append(frame)
            total_rows += len(frame)
            if total_rows >= self.settings.max_train_rows:
                break

        train_df = pd.concat(train_parts, axis=0, ignore_index=True)
        calib_df = load_gold_year(self.settings.gold_root, self.settings.calib_year)
        holdout_df = load_gold_year(self.settings.gold_root, self.settings.holdout_year)
        return train_df, calib_df, holdout_df

    @staticmethod
    def _sanitize_frame(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        out = df[feature_cols].copy()
        for col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    def train_horizon(self, horizon: int) -> TrainArtifacts:
        target = target_column(horizon)
        train_df, calib_df, holdout_df = self._load_splits(horizon)
        cal_a_df, cal_b_df = split_calibration_frame(calib_df, split_month=self.settings.calibration_split_month)
        feature_cols = detect_feature_columns(train_df)

        x_train = self._sanitize_frame(train_df, feature_cols)
        y_train = train_df[target].astype(int).to_numpy()

        x_cal_a = self._sanitize_frame(cal_a_df, feature_cols)
        y_cal_a = cal_a_df[target].astype(int).to_numpy()

        x_cal_b = self._sanitize_frame(cal_b_df, feature_cols)
        y_cal_b = cal_b_df[target].astype(int).to_numpy()

        x_hold = self._sanitize_frame(holdout_df, feature_cols)
        y_hold = holdout_df[target].astype(int).to_numpy()

        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        scale_pos_weight = float(neg / max(pos, 1))

        models: list[CatBoostClassifier] = []
        for seed in self.settings.seeds:
            tuned_params: dict | None = None
            tuned_path = Path("data/artifacts") / f"best_params_{int(horizon)}m.json"
            if tuned_path.exists():
                import json as _json
                with open(tuned_path) as _f:
                    tuned_params = _json.load(_f)
            params = _base_catboost_params(horizon)
            params["random_seed"] = int(seed)
            params["scale_pos_weight"] = scale_pos_weight
            model = CatBoostClassifier(**params)
            model.fit(x_train, y_train, eval_set=(x_cal_a, y_cal_a), use_best_model=True, verbose=False)
            models.append(model)

        def ens_predict(frame: pd.DataFrame) -> np.ndarray:
            probs = [model.predict_proba(frame)[:, 1] for model in models]
            return np.mean(np.vstack(probs), axis=0)

        raw_cal_a = ens_predict(x_cal_a)
        raw_cal_b = ens_predict(x_cal_b)
        raw_hold = ens_predict(x_hold)

        calibrator, calibration_diag = choose_best_calibrator(raw_cal_a, y_cal_a, raw_cal_b, y_cal_b)
        cal_b_prob = calibrator.predict(raw_cal_b)
        hold_prob = calibrator.predict(raw_hold)

        policy = DEFAULT_POLICY_BY_HORIZON[int(horizon)]
        threshold_info = choose_threshold(
            ts=cal_b_df["ts"],
            y_true=y_cal_b,
            y_prob=cal_b_prob,
            horizon_min=int(horizon),
            policy=policy,
        )
        threshold = float(threshold_info["threshold"])
        metrics = summary_metrics(y_hold, hold_prob, threshold=threshold)

        lead_times = lead_times_minutes(
            ts=holdout_df["ts"],
            y_true=y_hold,
            y_prob=hold_prob,
            threshold=threshold,
            horizon_min=int(horizon),
            persist_min=policy.persist_min,
            cooldown_min=policy.cooldown_min,
        )
        metrics["median_lead_time_min"] = float(np.median(lead_times)) if lead_times else float("nan")

        calb_predictions = cal_b_df[["ts", target]].copy()
        calb_predictions["raw_prob"] = raw_cal_b
        calb_predictions["prob"] = cal_b_prob

        hold_predictions = holdout_df[["ts", target]].copy()
        hold_predictions["raw_prob"] = raw_hold
        hold_predictions["prob"] = hold_prob

        self.bundle_writer.write_horizon(
            horizon=horizon,
            feature_columns=feature_cols,
            threshold=threshold,
            policy={
                "min_precision": policy.min_precision,
                "max_alerts_per_day": policy.max_alerts_per_day,
                "persist_min": policy.persist_min,
                "cooldown_min": policy.cooldown_min,
            },
            calibrator=calibrator.to_dict(),
            diagnostics={
                "calibration": calibration_diag,
                "threshold_selection": threshold_info,
                "train_years": [int(y) for y in range(2012, self.settings.calib_year) if y not in self.settings.exclude_years],
                "calibration_year": int(self.settings.calib_year),
                "holdout_year": int(self.settings.holdout_year),
            },
            metrics=metrics,
            models=models,
            calb_predictions=calb_predictions,
            holdout_predictions=hold_predictions,
        )
        return TrainArtifacts(
            horizon=int(horizon),
            threshold=threshold,
            metrics=metrics,
            diagnostics={"calibration": calibration_diag, "threshold_selection": threshold_info},
            policy=policy,
        )

    def train_all(self) -> dict:
        results = {}
        for horizon in self.settings.horizons:
            result = self.train_horizon(int(horizon))
            results[int(horizon)] = {
                "threshold": result.threshold,
                "metrics": result.metrics,
                "policy": {
                    "min_precision": result.policy.min_precision,
                    "max_alerts_per_day": result.policy.max_alerts_per_day,
                    "persist_min": result.policy.persist_min,
                    "cooldown_min": result.policy.cooldown_min,
                },
                "diagnostics": result.diagnostics,
            }

        self.bundle_writer.write_root_metadata(
            {
                "project": "Solar Flare Ops",
                "horizons": [int(h) for h in self.settings.horizons],
                "train_period": "2012-2017, 2020-2023",
                "excluded_years": list(self.settings.exclude_years),
                "calibration_year": int(self.settings.calib_year),
                "calibration_split": "CAL_A=Jan-Aug, CAL_B=Sep-Dec",
                "holdout_year": int(self.settings.holdout_year),
                "ensemble_seeds": list(self.settings.seeds),
                "negative_sampling_ratio": f"1:{int(self.settings.neg_pos_ratio)}",
                "results": results,
            }
        )
        self.bundle_writer.write_description(
            """
            Operational bundle for calibrated M1+ onset forecasting.
            Each horizon directory contains five CatBoost seed models, calibrated prediction dumps,
            horizon metadata, and the final operating threshold selected on CAL_B.
            """
        )
        return results
