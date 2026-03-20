from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.isotonic import IsotonicRegression

from .metrics import logloss


class Calibrator(Protocol):
    name: str

    def predict(self, raw: np.ndarray) -> np.ndarray: ...
    def to_dict(self) -> dict: ...


@dataclass(frozen=True)
class IdentityCalibrator:
    name: str = "identity"

    def predict(self, raw: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(raw, dtype=float), 0.0, 1.0)

    def to_dict(self) -> dict:
        return {"type": self.name, "params": {}}


@dataclass(frozen=True)
class PlattCalibrator:
    a: float
    b: float
    name: str = "platt"

    def predict(self, raw: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(raw, dtype=float), 1e-12, 1.0 - 1e-12)
        z = np.log(p / (1.0 - p))
        return np.clip(1.0 / (1.0 + np.exp(-(self.a * z + self.b))), 0.0, 1.0)

    def to_dict(self) -> dict:
        return {"type": self.name, "params": {"a": float(self.a), "b": float(self.b)}}

    @staticmethod
    def fit(raw: np.ndarray, y: np.ndarray, l2: float = 1e-6, iters: int = 50) -> "PlattCalibrator":
        p = np.clip(np.asarray(raw, dtype=float), 1e-12, 1.0 - 1e-12)
        y = np.asarray(y, dtype=float)
        z = np.log(p / (1.0 - p))
        a = 1.0
        b = 0.0
        for _ in range(int(iters)):
            t = a * z + b
            q = 1.0 / (1.0 + np.exp(-t))
            w = q * (1.0 - q)
            g_a = np.sum((q - y) * z) + l2 * a
            g_b = np.sum(q - y) + l2 * b
            h_aa = np.sum(w * z * z) + l2
            h_ab = np.sum(w * z)
            h_bb = np.sum(w) + l2
            det = h_aa * h_bb - h_ab * h_ab
            if abs(det) < 1e-12:
                break
            da = (h_bb * g_a - h_ab * g_b) / det
            db = (-h_ab * g_a + h_aa * g_b) / det
            a -= da
            b -= db
            if max(abs(da), abs(db)) < 1e-6:
                break
        return PlattCalibrator(a=float(a), b=float(b))


@dataclass(frozen=True)
class IsotonicCalibrator:
    x: tuple[float, ...]
    y: tuple[float, ...]
    name: str = "isotonic"

    def predict(self, raw: np.ndarray) -> np.ndarray:
        values = np.asarray(raw, dtype=float)
        return np.clip(np.interp(values, self.x, self.y, left=self.y[0], right=self.y[-1]), 0.0, 1.0)

    def to_dict(self) -> dict:
        return {"type": self.name, "params": {"x": list(self.x), "y": list(self.y)}}

    @staticmethod
    def fit(raw: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        reg = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        reg.fit(np.asarray(raw, dtype=float), np.asarray(y, dtype=float))
        return IsotonicCalibrator(
            x=tuple(float(v) for v in reg.X_thresholds_),
            y=tuple(float(v) for v in reg.y_thresholds_),
        )


def calibrator_from_dict(payload: dict) -> Calibrator:
    kind = payload["type"]
    params = payload.get("params", {})
    if kind == "identity":
        return IdentityCalibrator()
    if kind == "platt":
        return PlattCalibrator(a=float(params["a"]), b=float(params["b"]))
    if kind == "isotonic":
        return IsotonicCalibrator(
            x=tuple(float(v) for v in params["x"]),
            y=tuple(float(v) for v in params["y"]),
        )
    raise ValueError(f"Unsupported calibrator type: {kind}")


def choose_best_calibrator(
    raw_cal_a: np.ndarray,
    y_cal_a: np.ndarray,
    raw_cal_b: np.ndarray,
    y_cal_b: np.ndarray,
) -> tuple[Calibrator, dict]:
    candidates: list[Calibrator] = [
        IdentityCalibrator(),
        PlattCalibrator.fit(raw_cal_a, y_cal_a),
        IsotonicCalibrator.fit(raw_cal_a, y_cal_a),
    ]

    scored = []
    for calibrator in candidates:
        prob = calibrator.predict(raw_cal_b)
        scored.append(
            {
                "name": calibrator.name,
                "logloss": float(logloss(y_cal_b, prob)),
                "calibrator": calibrator,
            }
        )
    best = min(scored, key=lambda x: x["logloss"])
    diagnostics = {
        "selected": best["name"],
        "candidates": {item["name"]: item["logloss"] for item in scored},
    }
    return best["calibrator"], diagnostics
