from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    asof_utc TEXT PRIMARY KEY,
    data_ts_utc TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    lag_minutes REAL NOT NULL,
    stale INTEGER NOT NULL
);
"""


@dataclass(frozen=True)
class StoredPrediction:
    asof_utc: str
    data_ts_utc: str
    payload: dict
    lag_minutes: float
    stale: bool


class PredictionStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA_SQL)

    def upsert(self, payload: dict) -> None:
        row = (
            payload["asof_utc"],
            payload["data_ts_utc"],
            json.dumps(payload, ensure_ascii=False),
            float(payload["lag_minutes"]),
            int(bool(payload["stale"])),
        )
        self.conn.execute(
            """
            INSERT INTO predictions(asof_utc, data_ts_utc, payload_json, lag_minutes, stale)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(asof_utc) DO UPDATE SET
                data_ts_utc=excluded.data_ts_utc,
                payload_json=excluded.payload_json,
                lag_minutes=excluded.lag_minutes,
                stale=excluded.stale
            """,
            row,
        )
        self.conn.commit()

    def latest(self) -> StoredPrediction | None:
        row = self.conn.execute(
            "SELECT asof_utc, data_ts_utc, payload_json, lag_minutes, stale FROM predictions ORDER BY asof_utc DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return StoredPrediction(
            asof_utc=row["asof_utc"],
            data_ts_utc=row["data_ts_utc"],
            payload=json.loads(row["payload_json"]),
            lag_minutes=float(row["lag_minutes"]),
            stale=bool(row["stale"]),
        )

    def history(self, limit: int = 500) -> list[StoredPrediction]:
        rows = self.conn.execute(
            "SELECT asof_utc, data_ts_utc, payload_json, lag_minutes, stale FROM predictions ORDER BY asof_utc DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [
            StoredPrediction(
                asof_utc=row["asof_utc"],
                data_ts_utc=row["data_ts_utc"],
                payload=json.loads(row["payload_json"]),
                lag_minutes=float(row["lag_minutes"]),
                stale=bool(row["stale"]),
            )
            for row in rows
        ]
