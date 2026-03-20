from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def roc_auc_rank(y_true: Iterable[int], y_prob: Iterable[float]) -> float:
    y = np.asarray(list(y_true), dtype=np.int8)
    p = np.asarray(list(y_prob), dtype=np.float64)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(p, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1, dtype=np.float64)

    s = p[order]
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s[j] == s[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j

    sum_ranks_pos = float(ranks[y == 1].sum())
    return float((sum_ranks_pos - pos * (pos + 1) / 2.0) / (pos * neg))


def pr_auc(y_true: Iterable[int], y_prob: Iterable[float]) -> float:
    y = np.asarray(list(y_true), dtype=np.int8)
    p = np.asarray(list(y_prob), dtype=np.float64)
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-p, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    return float(np.trapz(precision, recall))


def brier_score(y_true: Iterable[int], y_prob: Iterable[float]) -> float:
    y = np.asarray(list(y_true), dtype=np.float64)
    p = np.asarray(list(y_prob), dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def logloss(y_true: Iterable[int], y_prob: Iterable[float], eps: float = 1e-15) -> float:
    y = np.asarray(list(y_true), dtype=np.float64)
    p = np.clip(np.asarray(list(y_prob), dtype=np.float64), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def ece(y_true: Iterable[int], y_prob: Iterable[float], n_bins: int = 20) -> float:
    y = np.asarray(list(y_true), dtype=np.float64)
    p = np.asarray(list(y_prob), dtype=np.float64)
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    total = 0.0
    n = len(y)
    if n == 0:
        return float("nan")
    for i in range(len(bins) - 1):
        lo = bins[i]
        hi = bins[i + 1]
        mask = (p >= lo) & (p < hi) if i < len(bins) - 2 else (p >= lo) & (p <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        total += (count / n) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(total)


def confusion_at_threshold(y_true: Iterable[int], y_prob: Iterable[float], threshold: float) -> tuple[int, int, int, int]:
    y = np.asarray(list(y_true), dtype=np.int8)
    p = np.asarray(list(y_prob), dtype=np.float64)
    pred = (p >= float(threshold)).astype(np.int8)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    return tp, tn, fp, fn


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return float(precision), float(recall), float(f1)


def tss(tp: int, tn: int, fp: int, fn: int) -> float:
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return float(tpr - fpr)


def classification_metrics(y_true: Iterable[int], y_prob: Iterable[float], threshold: float) -> dict[str, float | int]:
    tp, tn, fp, fn = confusion_at_threshold(y_true, y_prob, threshold)
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    alerts = tp + fp
    support = tp + fn
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "tss": float(tss(tp, tn, fp, fn)),
        "alerts": int(alerts),
        "support": int(support),
    }


def summary_metrics(y_true: Iterable[int], y_prob: Iterable[float], threshold: float) -> dict[str, float | int]:
    out = {
        "roc_auc": roc_auc_rank(y_true, y_prob),
        "pr_auc": pr_auc(y_true, y_prob),
        "logloss": logloss(y_true, y_prob),
        "brier": brier_score(y_true, y_prob),
        "ece": ece(y_true, y_prob),
    }
    out.update(classification_metrics(y_true, y_prob, threshold))
    return out
