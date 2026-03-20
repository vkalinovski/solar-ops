from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

from .calibration import calibrator_from_dict


@dataclass(frozen=True)
class HorizonArtifact:
    horizon: int
    feature_columns: list[str]
    threshold: float
    policy: dict
    calibrator: dict
    diagnostics: dict
    metrics: dict
    model_files: list[str]


class BundleWriter:
    def __init__(self, bundle_dir: Path):
        self.bundle_dir = Path(bundle_dir)
        self.bundle_dir.mkdir(parents=True, exist_ok=True)

    def write_root_metadata(self, payload: dict) -> None:
        (self.bundle_dir / "bundle.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def write_description(self, text: str) -> None:
        (self.bundle_dir / "description.txt").write_text(text.strip() + "\n", encoding="utf-8")

    def write_horizon(
        self,
        horizon: int,
        feature_columns: list[str],
        threshold: float,
        policy: dict,
        calibrator: dict,
        diagnostics: dict,
        metrics: dict,
        models: list[CatBoostClassifier],
        calb_predictions: pd.DataFrame,
        holdout_predictions: pd.DataFrame,
    ) -> HorizonArtifact:
        h_dir = self.bundle_dir / "horizons" / f"{int(horizon)}m"
        h_dir.mkdir(parents=True, exist_ok=True)

        model_files = []
        for seed_idx, model in enumerate(models):
            name = f"seed_{seed_idx}.cbm"
            model.save_model(str(h_dir / name))
            model_files.append(name)

        calb_predictions.to_parquet(h_dir / "calB_predictions.parquet", index=False)
        holdout_predictions.to_parquet(h_dir / "holdout_predictions.parquet", index=False)

        artifact = HorizonArtifact(
            horizon=int(horizon),
            feature_columns=list(feature_columns),
            threshold=float(threshold),
            policy=policy,
            calibrator=calibrator,
            diagnostics=diagnostics,
            metrics=metrics,
            model_files=model_files,
        )
        (h_dir / "metadata.json").write_text(json.dumps(asdict(artifact), indent=2), encoding="utf-8")
        return artifact


def load_horizon_bundle(bundle_dir: Path, horizon: int) -> dict:
    h_dir = Path(bundle_dir) / "horizons" / f"{int(horizon)}m"
    meta = json.loads((h_dir / "metadata.json").read_text(encoding="utf-8"))
    models = []
    for name in meta["model_files"]:
        model = CatBoostClassifier()
        model.load_model(str(h_dir / name))
        models.append(model)
    calibrator = calibrator_from_dict(meta["calibrator"])
    return {
        "metadata": meta,
        "models": models,
        "calibrator": calibrator,
    }


def validate_bundle(bundle_dir: Path) -> list[str]:
    errors: list[str] = []
    bundle_dir = Path(bundle_dir)
    root_meta_path = bundle_dir / "bundle.json"
    if not root_meta_path.exists():
        return ["Missing bundle.json"]

    root_meta = json.loads(root_meta_path.read_text(encoding="utf-8"))
    horizons = root_meta.get("horizons", [])
    for horizon in horizons:
        h_dir = bundle_dir / "horizons" / f"{int(horizon)}m"
        if not h_dir.exists():
            errors.append(f"Missing horizon directory: {h_dir}")
            continue

        meta_path = h_dir / "metadata.json"
        if not meta_path.exists():
            errors.append(f"Missing metadata.json for {horizon}m")
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        model_files = meta.get("model_files", [])
        if len(model_files) != 5:
            errors.append(f"{horizon}m should have 5 seed models, found {len(model_files)}")

        for name in model_files:
            if not (h_dir / name).exists():
                errors.append(f"Missing model file for {horizon}m: {name}")

        for required in ("threshold", "policy", "calibrator", "feature_columns"):
            if required not in meta:
                errors.append(f"{horizon}m metadata missing {required}")

        for parquet_name in ("calB_predictions.parquet", "holdout_predictions.parquet"):
            if not (h_dir / parquet_name).exists():
                errors.append(f"Missing {parquet_name} for {horizon}m")
    return errors
