from __future__ import annotations

import threading
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..inference.online import run_live_demo
from ..settings import RuntimeSettings
from ..storage.store import PredictionStore
from ..utils.logging import get_logger, setup_logging


def create_app(settings: RuntimeSettings | None = None) -> FastAPI:
    cfg = settings or RuntimeSettings()
    setup_logging(cfg.log_level, cfg.log_json)
    logger = get_logger("solarflare.api")

    app = FastAPI(title="Solar Flare Ops API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    state_lock = threading.Lock()
    store = PredictionStore(cfg.db_path)
    state = {
        "ready": False,
        "last_error": None,
    }

    def poll_loop() -> None:
        while True:
            try:
                payload = run_live_demo(cfg.bundle_dir, lookback_min=cfg.lookback_min, max_stale_min=cfg.max_stale_min)
                store.upsert(payload)
                with state_lock:
                    state["ready"] = True
                    state["last_error"] = None
            except Exception as exc:
                logger.exception("poll failed")
                with state_lock:
                    state["last_error"] = repr(exc)
            time.sleep(cfg.update_every_sec)

    @app.on_event("startup")
    def startup() -> None:
        thread = threading.Thread(target=poll_loop, daemon=True)
        thread.start()

    @app.get("/health")
    @app.get("/api/health")
    def health() -> dict:
        with state_lock:
            return {
                "status": "ok",
                "ready": state["ready"],
                "bundle_dir": str(Path(cfg.bundle_dir).resolve()),
                "db_path": str(Path(cfg.db_path).resolve()),
                "last_error": state["last_error"],
            }

    @app.get("/now")
    @app.get("/api/now")
    def now() -> dict:
        record = store.latest()
        with state_lock:
            ready = state["ready"]
            error = state["last_error"]
        if record is None:
            return {"status": "warming_up" if ready else "not_ready", "last_error": error}
        payload = dict(record.payload)
        payload["status"] = "ok"
        payload["last_error"] = error
        return payload

    @app.get("/history")
    @app.get("/api/history")
    def history(limit: int = 200) -> dict:
        records = store.history(limit=limit)
        return {
            "status": "ok",
            "items": [record.payload for record in records],
        }

    return app
