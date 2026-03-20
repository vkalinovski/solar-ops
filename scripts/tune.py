"""
Optuna hyperparameter search for a single horizon.

Usage:
    python scripts/tune.py --horizon 120 --trials 60 --gold-root data/gold
    python scripts/tune.py --horizon 60 --trials 200 --timeout 3600 --force-cpu
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier

from solarflare_app.data.gold import load_gold_years, sample_binary_frame, split_calibration_frame
from solarflare_app.settings import TrainingSettings

optuna.logging.set_verbosity(optuna.logging.WARNING)

POLICY_DEFAULTS = {
    60:   dict(min_precision=0.08, budget_per_day=2.0, cooldown_min=30,  persist_min=2),
    120:  dict(min_precision=0.10, budget_per_day=2.0, cooldown_min=60,  persist_min=2),
    360:  dict(min_precision=0.12, budget_per_day=1.5, cooldown_min=120, persist_min=3),
    720:  dict(min_precision=0.15, budget_per_day=1.0, cooldown_min=180, persist_min=3),
}


# ── pure-numpy helpers ────────────────────────────────────────────────────────

def _confusion(y, p, thr):
    pred = p >= thr
    tp = int(((y == 1) & pred).sum())
    tn = int(((y == 0) & ~pred).sum())
    fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    return tp, tn, fp, fn


def _tss(tp, tn, fp, fn):
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return float(tpr - fpr)


def _f1(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


def _logloss(y, p, eps=1e-15):
    p = np.clip(p, eps, 1 - eps)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _platt_fit(p, y, iters=50, l2=1e-6):
    z = np.log(np.clip(p, 1e-12, 1 - 1e-12) / (1 - np.clip(p, 1e-12, 1 - 1e-12)))
    a, b = 1.0, 0.0
    for _ in range(iters):
        q = 1 / (1 + np.exp(-(a * z + b)))
        w = q * (1 - q)
        ga = np.sum((q - y) * z) + l2 * a
        gb = np.sum(q - y) + l2 * b
        Haa = np.sum(w * z * z) + l2
        Hab = np.sum(w * z)
        Hbb = np.sum(w) + l2
        det = Haa * Hbb - Hab ** 2
        if abs(det) < 1e-12:
            break
        a -= (Hbb * ga - Hab * gb) / det
        b -= (-Hab * ga + Haa * gb) / det
    return float(a), float(b)


def _platt_predict(p, a, b):
    z = np.log(np.clip(p, 1e-12, 1 - 1e-12) / (1 - np.clip(p, 1e-12, 1 - 1e-12)))
    return np.clip(1 / (1 + np.exp(-(a * z + b))), 0.0, 1.0)


def _simulate_alerts(ts, p, thr, cooldown_min, persist_min):
    alerts = []
    i, n = 0, len(p)
    while i < n:
        if p[i] >= thr:
            cnt, j = 0, i
            while j < n and p[j] >= thr:
                cnt += 1
                if cnt >= persist_min:
                    t = ts.iloc[j] if hasattr(ts, "iloc") else ts[j]
                    alerts.append(t)
                    i = j + cooldown_min
                    break
                j += 1
            else:
                i = j + 1
        else:
            i += 1
    return alerts


def _safe_days(ts):
    try:
        ts = pd.to_datetime(ts, errors="coerce").dropna()
        if len(ts) >= 2:
            d = (ts.max() - ts.min()).total_seconds() / 86400
            if np.isfinite(d) and d > 0:
                return float(d)
    except Exception:
        pass
    return float("nan")


def _pick_threshold(df_calB, y_col, p, policy, w_tss=0.65, w_f1=0.35):
    y = df_calB[y_col].astype(int).values
    ts = df_calB["ts"] if "ts" in df_calB.columns else pd.Series(range(len(df_calB)))

    cooldown = policy["cooldown_min"]
    persist  = policy["persist_min"]
    budget   = policy["budget_per_day"]
    min_prec = policy["min_precision"]

    thr_grid = np.unique(np.r_[
        np.quantile(p, np.linspace(0.01, 0.995, 500)),
        [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40],
    ])

    best = best_relaxed = None
    days = _safe_days(ts)

    for thr in thr_grid:
        tp, tn, fp, fn = _confusion(y, p, thr)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        score = w_tss * _tss(tp, tn, fp, fn) + w_f1 * _f1(tp, fp, fn)

        alerts = _simulate_alerts(ts, p, thr, cooldown, persist)
        apd = len(alerts) / days if (np.isfinite(days) and days > 0) else float("nan")

        cand = dict(thr=float(thr), score=float(score), precision=float(prec), alerts_day=float(apd))

        if best_relaxed is None or score > best_relaxed["score"]:
            best_relaxed = cand.copy()

        if (not math.isnan(apd)) and apd <= budget + 1e-9 and prec >= min_prec - 1e-12:
            if best is None or score > best["score"]:
                best = cand.copy()

    if best is None:
        result = dict(best_relaxed or {}); result["reason"] = "relaxed"
        return result
    best["reason"] = "strict"
    return best


# ── main tuning function ──────────────────────────────────────────────────────

def tune(
    horizon: int,
    gold_root: Path,
    out_dir: Path,
    settings: TrainingSettings,
    n_trials: int = 60,
    timeout_sec: int = 0,
    tune_seeds: list[int] | None = None,
    force_cpu: bool = False,
    w_tss: float = 0.65,
    w_f1: float = 0.35,
) -> dict:
    if tune_seeds is None:
        tune_seeds = [0, 1, 2]

    y_col = f"y_onset_m1p_in_{horizon}m"
    policy = POLICY_DEFAULTS.get(horizon, POLICY_DEFAULTS[120])

    train_years = [y for y in range(2012, settings.calib_year) if y not in set(settings.exclude_years)]
    print(f"[tune] horizon={horizon}m  train_years={train_years}  trials={n_trials}")

    df_train_full = load_gold_years(train_years, root=gold_root)
    df_train = sample_binary_frame(df_train_full, y_col=y_col, neg_pos_ratio=settings.neg_pos_ratio)

    df_calib = load_gold_years([settings.calib_year], root=gold_root)
    df_calA, df_calB = split_calibration_frame(df_calib, split_month=settings.calibration_split_month)

    feat_cols = [c for c in df_train.columns if c not in [y_col, "ts", "year"]]
    X_train = df_train[feat_cols].values
    y_train = df_train[y_col].astype(int).values
    X_calA  = df_calA[feat_cols].values
    y_calA  = df_calA[y_col].astype(int).values
    X_calB  = df_calB[feat_cols].values
    y_calB  = df_calB[y_col].astype(int).values

    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    spw = min(float(neg / max(1, pos)), 200.0)

    base_task = "CPU" if force_cpu else "GPU"

    def objective(trial: optuna.Trial) -> float:
        try:
            params = dict(
                loss_function="Logloss",
                eval_metric="Logloss",
                scale_pos_weight=spw,
                allow_writing_files=False,
                verbose=False,
                od_type="Iter",
                task_type=base_task,
                iterations=trial.suggest_int("iterations", 1500, 6500),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
                depth=trial.suggest_int("depth", 6, 10),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 25.0, log=True),
                random_strength=trial.suggest_float("random_strength", 0.0, 2.0),
                bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 2.0),
                border_count=trial.suggest_int("border_count", 64, 255),
                min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 32, 512, log=True),
                od_wait=trial.suggest_int("od_wait", 150, 700),
            )
            if base_task == "CPU":
                params["rsm"] = trial.suggest_float("rsm", 0.6, 1.0)

            pA_list, pB_list = [], []
            for seed in tune_seeds:
                m = CatBoostClassifier(**{**params, "random_seed": seed})
                m.fit(X_train, y_train, eval_set=(X_calA, y_calA), use_best_model=True, verbose=False)
                pA_list.append(m.predict_proba(X_calA)[:, 1])
                pB_list.append(m.predict_proba(X_calB)[:, 1])

            pA_raw = np.mean(pA_list, axis=0)
            pB_raw = np.mean(pB_list, axis=0)

            a, b = _platt_fit(pA_raw, y_calA)
            pB_cal = _platt_predict(pB_raw, a, b)

            sel = _pick_threshold(df_calB, y_col, pB_cal, policy, w_tss, w_f1)
            score = float(sel.get("score", -1e9))
            if sel.get("reason") != "strict":
                score -= 0.05

            trial.set_user_attr("calB_logloss", _logloss(y_calB, pB_cal))
            for k in ["thr", "precision", "alerts_day", "reason"]:
                trial.set_user_attr(k, sel.get(k))
            return score

        except Exception as e:
            trial.set_user_attr("error", repr(e))
            return -1e9

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True, group=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
        study_name=f"solar_h{horizon}m",
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_sec or None,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    best = study.best_trial
    result = {
        "horizon": horizon,
        "score": float(best.value),
        "params": dict(best.params),
        "user_attrs": {k: best.user_attrs.get(k) for k in ["calB_logloss", "thr", "precision", "alerts_day", "reason"]},
        "policy": policy,
        "tune_seeds": tune_seeds,
        "force_cpu": force_cpu,
        "n_trials": n_trials,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"best_params_{horizon}m.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[tune] saved → {out_path}")
    print(f"[tune] best score={best.value:.4f}  thr={best.user_attrs.get('thr')}  "
          f"logloss={best.user_attrs.get('calB_logloss')}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for one horizon")
    parser.add_argument("--horizon",   type=int,   default=120, choices=[60, 120, 360, 720])
    parser.add_argument("--trials",    type=int,   default=60)
    parser.add_argument("--timeout",   type=int,   default=0,   help="max seconds (0 = unlimited)")
    parser.add_argument("--gold-root", type=Path,  default=Path("data/gold"))
    parser.add_argument("--out-dir",   type=Path,  default=Path("data/artifacts/optuna"))
    parser.add_argument("--tune-seeds", type=int,  nargs="+", default=[0, 1, 2])
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--w-tss",     type=float, default=0.65)
    parser.add_argument("--w-f1",      type=float, default=0.35)
    args = parser.parse_args()

    settings = TrainingSettings()
    tune(
        horizon=args.horizon,
        gold_root=args.gold_root,
        out_dir=args.out_dir,
        settings=settings,
        n_trials=args.trials,
        timeout_sec=args.timeout,
        tune_seeds=args.tune_seeds,
        force_cpu=args.force_cpu,
        w_tss=args.w_tss,
        w_f1=args.w_f1,
    )


if __name__ == "__main__":
    main()
