"""
Statistical test suite for holdout evaluation.

Runs:
  1. Gold dataset QA — row counts, missing values, target rates per year
  2. Artifact completeness — bundle, holdout preds, model files
  3. Metric reproducibility — recompute ROC-AUC / PR-AUC / Brier / ECE from saved preds
  4. Policy verification — alert simulation on holdout with operational threshold
  5. Bootstrap confidence intervals — stationary bootstrap over event-recall and alerts/day
  6. Gold signal sanity — feature mean comparison positives vs negatives

Usage:
    python scripts/test_suite.py --bundle-dir bundle --gold-root data/gold
    python scripts/test_suite.py --no-bootstrap   # skip bootstrap (faster)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = [60, 120, 360, 720]


# ── metrics ───────────────────────────────────────────────────────────────────

def roc_auc(y, p):
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    score = 0
    for i in pos:
        score += (p[i] > p[neg]).sum() + 0.5 * (p[i] == p[neg]).sum()
    return float(score / (len(pos) * len(neg)))


def pr_auc(y, p):
    order = np.argsort(-p)
    y_s = np.asarray(y, dtype=float)[order]
    cum_tp = np.cumsum(y_s)
    prec = cum_tp / (np.arange(len(y_s)) + 1)
    rec  = cum_tp / max(y_s.sum(), 1)
    prec = np.r_[1.0, prec]; rec = np.r_[0.0, rec]
    return float(np.trapz(prec, rec))


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def ece(y, p, n_bins=15):
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]; p = p[mask]
    if len(y) == 0:
        return float("nan")
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    val = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            val += (m.sum() / len(y)) * abs(y[m].mean() - p[m].mean())
    return float(val)


# ── column detection ──────────────────────────────────────────────────────────

def _detect_cols(df: pd.DataFrame, h: int):
    cols = set(df.columns)
    ts_col = next((c for c in ["ts", "timestamp", "time"] if c in cols), None)
    y_col  = next((c for c in [f"y_onset_m1p_in_{h}m", f"y_{h}", "y", "y_true"] if c in cols), None)
    p_col  = next((c for c in [f"p_{h}", f"p_cal_{h}", "p_cal", "p", "proba"] if c in cols), None)
    return ts_col, y_col, p_col


# ── policy simulation ─────────────────────────────────────────────────────────

def _policy_alerts(ts, p, thr, persist, cooldown):
    n = len(p)
    fired = np.zeros(n, dtype=bool)
    last = -10 ** 9
    csum = np.cumsum((p >= thr).astype(int))
    for i in range(persist - 1, n):
        win = csum[i] - (csum[i - persist] if i - persist >= 0 else 0)
        if win == persist and (i - last) > cooldown:
            fired[i] = True
            last = i
    return fired


def _alerts_per_day(fired_ts):
    if len(fired_ts) == 0:
        return 0.0
    d = pd.to_datetime(fired_ts).floor("D")
    return float(len(fired_ts) / max(1, d.nunique()))


def _approx_onsets(ts, y):
    y = np.asarray(y, dtype=int)
    idx = np.where((y[:-1] == 1) & (y[1:] == 0))[0] + 1
    return pd.to_datetime(ts)[idx]


def _event_recall(onsets, fired_ts, h):
    if len(onsets) == 0:
        return float("nan"), 0
    fired_ts = pd.to_datetime(fired_ts)
    hits = sum(
        1 for s in onsets
        if ((fired_ts > s - pd.Timedelta(minutes=h)) & (fired_ts <= s)).any()
    )
    return float(hits / len(onsets)), len(onsets)


# ── bootstrap ─────────────────────────────────────────────────────────────────

def _stationary_bootstrap_idx(n, mean_block, B, seed=42):
    rng = np.random.default_rng(seed)
    p = 1.0 / max(1, mean_block)
    indices = []
    for _ in range(B):
        idx = np.empty(n, dtype=np.int64)
        t = 0
        while t < n:
            start = int(rng.integers(0, n))
            L = max(1, int(rng.geometric(p)))
            for j in range(L):
                if t >= n:
                    break
                idx[t] = (start + j) % n
                t += 1
        indices.append(idx)
    return indices


def _ci(a):
    a = np.asarray(a, dtype=float)
    return (float(np.nanpercentile(a, 5)),
            float(np.nanpercentile(a, 50)),
            float(np.nanpercentile(a, 95)))


# ── gold QA ───────────────────────────────────────────────────────────────────

def _gold_qa(gold_root: Path, horizons: list[int]) -> list[dict]:
    rows = []
    for y_dir in sorted(gold_root.glob("year=*")):
        try:
            year = int(y_dir.name.split("=")[1])
        except ValueError:
            continue
        p = y_dir / "data.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p, columns=(
            ["ts"] +
            [f"y_onset_m1p_in_{h}m" for h in horizons if f"y_onset_m1p_in_{h}m" in pd.read_parquet(p, columns=["ts"]).columns or True] +
            ["xrsb", "sunspot_number", "f107"]
        ), filters=None)
        try:
            df = pd.read_parquet(p)
        except Exception as e:
            rows.append({"year": year, "error": repr(e)})
            continue

        leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        expected = (366 if leap else 365) * 1440
        row: dict = {
            "year": year,
            "rows": len(df),
            "expected": expected,
            "ok": len(df) == expected,
            "ts_monotonic": bool(df["ts"].is_monotonic_increasing) if "ts" in df.columns else None,
        }
        for h in horizons:
            col = f"y_onset_m1p_in_{h}m"
            if col in df.columns:
                row[f"rate_{h}m"] = round(float(df[col].mean()) * 100, 4)
        for c in ["xrsb", "sunspot_number", "f107"]:
            if c in df.columns:
                row[f"null_pct_{c}"] = round(float(df[c].isna().mean()) * 100, 2)
        rows.append(row)
    return rows


# ── main ──────────────────────────────────────────────────────────────────────

def run_suite(
    bundle_dir: Path,
    gold_root: Path,
    out_dir: Path,
    horizons: list[int],
    do_bootstrap: bool = True,
    boot_B: int = 200,
    boot_mean_block_min: int = 1440,
    boot_seed: int = 42,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_run = time.strftime("%Y%m%d_%H%M%S")
    results: dict = {"timestamp": ts_run, "bundle_dir": str(bundle_dir)}

    # 1) Gold QA
    print("\n=== 1. Gold dataset QA ===")
    gold_qa = _gold_qa(gold_root, horizons)
    results["gold_qa"] = gold_qa
    qa_df = pd.DataFrame(gold_qa)
    print(qa_df.to_string(index=False))
    bad = qa_df[~qa_df["ok"].fillna(False)] if "ok" in qa_df.columns else pd.DataFrame()
    if len(bad):
        print(f"[WARN] {len(bad)} years with unexpected row count: {bad['year'].tolist()}")

    # 2-5) Per-horizon checks
    horizon_results = []
    for h in horizons:
        print(f"\n=== horizon {h}m ===")
        hdir = bundle_dir / "horizons" / f"{h}m"
        meta_p = hdir / "metadata.json"
        res: dict = {"h": h, "dir": str(hdir)}

        if not hdir.exists():
            res["status"] = "MISSING_DIR"
            horizon_results.append(res)
            print(f"  [SKIP] {hdir} not found")
            continue

        # find prediction files
        holdout_p = hdir / "holdout_predictions.parquet"
        calb_p    = hdir / "calB_predictions.parquet"
        cbm_files = list(hdir.glob("seed_*.cbm"))

        res["models_found"] = len(cbm_files)
        res["holdout_preds_exists"] = holdout_p.exists()
        res["calb_preds_exists"]    = calb_p.exists()

        if meta_p.exists():
            meta = json.loads(meta_p.read_text())
            res["calibrator"] = meta.get("calibrator", {}).get("type")
            res["threshold"]  = meta.get("threshold")
            thr   = float(meta.get("threshold", 0.20))
            pol   = meta.get("policy", {})
            persist  = int(pol.get("persist_min", 3))
            cooldown = int(pol.get("cooldown_min", 120))
        else:
            print(f"  [WARN] metadata.json missing at {meta_p}")
            thr, persist, cooldown = 0.20, 3, 120

        if not holdout_p.exists():
            res["status"] = "NO_HOLDOUT_PREDS"
            horizon_results.append(res)
            continue

        df = pd.read_parquet(holdout_p)
        ts_col, y_col, p_col = _detect_cols(df, h)

        if y_col is None or p_col is None:
            res["status"] = "COL_DETECT_FAILED"
            res["columns"] = list(df.columns)[:20]
            horizon_results.append(res)
            continue

        y = pd.to_numeric(df[y_col], errors="coerce").fillna(0).astype(int).values
        p = pd.to_numeric(df[p_col], errors="coerce").fillna(0.0).values

        # 3) Metrics
        res["roc_auc"] = round(roc_auc(y, p), 4)
        res["pr_auc"]  = round(pr_auc(y, p), 4)
        res["brier"]   = round(brier(y, p), 4)
        res["ece"]     = round(ece(y, p), 4)
        print(f"  ROC-AUC={res['roc_auc']}  PR-AUC={res['pr_auc']}  "
              f"Brier={res['brier']}  ECE={res['ece']}")

        # 4) Policy verification
        ts_vals = pd.to_datetime(df[ts_col], errors="coerce").values if ts_col else None
        fired = _policy_alerts(ts_vals if ts_vals is not None else np.arange(len(y)),
                               p, thr, persist, cooldown)
        fired_ts = (pd.to_datetime(ts_vals)[fired] if ts_vals is not None
                    else pd.Series(np.where(fired)[0]))

        apd = _alerts_per_day(fired_ts)
        tp = int(((y == 1) & fired).sum())
        fp = int(((y == 0) & fired).sum())
        fn = int(((y == 1) & ~fired).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        res.update(dict(alerts_day=round(apd, 3), minute_precision=round(prec, 3),
                        minute_recall=round(rec, 3), tp=tp, fp=fp, fn=fn))

        if ts_col:
            onsets = _approx_onsets(df[ts_col].values, y)
            er, n_ev = _event_recall(onsets, fired_ts, h)
            res["event_recall"] = round(er, 3) if not math.isnan(er) else None
            res["n_approx_events"] = n_ev
            print(f"  alerts/day={apd:.2f}  prec={prec:.3f}  rec={rec:.3f}  "
                  f"event_recall={res['event_recall']}  events={n_ev}")

        # 5) Bootstrap CI
        if do_bootstrap and ts_col:
            print(f"  Running bootstrap B={boot_B}...")
            idx_list = _stationary_bootstrap_idx(len(y), boot_mean_block_min, boot_B, boot_seed)
            ers, apds = [], []
            for idx in idx_list:
                yb = y[idx]; pb = p[idx]
                ts_b = df[ts_col].iloc[idx] if ts_col else None
                fired_b = _policy_alerts(
                    pd.to_datetime(ts_b, errors="coerce").values if ts_b is not None else idx,
                    pb, thr, persist, cooldown)
                fired_ts_b = (pd.to_datetime(ts_b.values)[fired_b]
                              if ts_b is not None else pd.Series(np.where(fired_b)[0]))
                onsets_b = _approx_onsets(
                    ts_b.values if ts_b is not None else idx, yb)
                er_b, _ = _event_recall(onsets_b, fired_ts_b, h)
                ers.append(er_b)
                apds.append(_alerts_per_day(fired_ts_b))

            res["boot_event_recall_ci"] = _ci(ers)
            res["boot_alerts_day_ci"]   = _ci(apds)
            print(f"  event_recall 90% CI: [{res['boot_event_recall_ci'][0]:.3f}, "
                  f"{res['boot_event_recall_ci'][2]:.3f}]")

        res["status"] = "OK"
        horizon_results.append(res)

    results["horizons"] = horizon_results

    # 6) Signal sanity on latest gold year
    gold_years = sorted([int(p.name.split("=")[1]) for p in gold_root.glob("year=*")
                         if p.is_dir() and "=" in p.name])
    if gold_years:
        latest = gold_years[-1]
        df_g = pd.read_parquet(gold_root / f"year={latest}" / "data.parquet")
        sanity = {}
        for h in horizons:
            col = f"y_onset_m1p_in_{h}m"
            if col in df_g.columns:
                pos_mask = df_g[col] == 1
                for feat in ["log_xrsb_roll360_mean", "xray_bkgd_flux", "f107"]:
                    if feat in df_g.columns:
                        pos_m = float(df_g.loc[pos_mask, feat].dropna().mean())
                        neg_m = float(df_g.loc[~pos_mask, feat].dropna().mean())
                        sanity[f"h{h}_{feat}_pos"] = round(pos_m, 4)
                        sanity[f"h{h}_{feat}_neg"] = round(neg_m, 4)
        results["signal_sanity"] = sanity
        print(f"\n=== Signal sanity (year={latest}) ===")
        for k, v in sanity.items():
            print(f"  {k}: {v}")

    out_path = out_dir / f"test_suite_{ts_run}.json"
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved → {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Statistical test suite")
    parser.add_argument("--bundle-dir",  type=Path, default=Path("bundle"))
    parser.add_argument("--gold-root",   type=Path, default=Path("data/gold"))
    parser.add_argument("--out-dir",     type=Path, default=Path("data/artifacts/tests"))
    parser.add_argument("--horizons",    type=int, nargs="+", default=HORIZONS)
    parser.add_argument("--no-bootstrap", action="store_true")
    parser.add_argument("--boot-B",      type=int, default=200)
    parser.add_argument("--boot-block",  type=int, default=1440,
                        help="mean block length in minutes (default 1440 = 1 day)")
    parser.add_argument("--boot-seed",   type=int, default=42)
    args = parser.parse_args()

    run_suite(
        bundle_dir=args.bundle_dir,
        gold_root=args.gold_root,
        out_dir=args.out_dir,
        horizons=args.horizons,
        do_bootstrap=not args.no_bootstrap,
        boot_B=args.boot_B,
        boot_mean_block_min=args.boot_block,
        boot_seed=args.boot_seed,
    )


if __name__ == "__main__":
    main()
