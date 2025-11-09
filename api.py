"""FastAPI surface for running backtests and online predictions."""
from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Dict
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from riskmap import RiskMapConfig
from service_utils import DB_PATH, MODEL_DEFAULT, ensure_db, predict_window, run_request

app = FastAPI(title="Quant TCN Risk-Map API", version="1.0")


class RunRequest(BaseModel):
    symbol: str = "SPY"
    start: str
    end: str
    test_start: str
    tau: float = 0.05
    sigma_target: float = 0.02
    wmax: float = 1.0
    blend: float = 0.6
    tau_k: float = 0.75
    prob_threshold: float = 0.0
    stop_loss: float = 0.0
    prob_scale: float = 1.0
    cost_bps: float = 3.0
    model_path: str = str(MODEL_DEFAULT)


class RunResponse(BaseModel):
    run_id: str
    metrics: Dict[str, float]


@app.post("/runs", response_model=RunResponse)
async def create_run(req: RunRequest) -> RunResponse:
    ensure_db()
    loop = asyncio.get_running_loop()
    try:
        metrics = await loop.run_in_executor(
            None,
            run_request,
            req.symbol,
            req.start,
            req.end,
            req.test_start,
            RiskMapConfig(
                tau=req.tau,
                sigma_target=req.sigma_target,
                wmax=req.wmax,
                blend=req.blend,
                tau_k=req.tau_k,
                prob_threshold=req.prob_threshold,
                stop_loss=req.stop_loss,
            ),
            req.prob_scale,
            req.cost_bps,
            Path(req.model_path),
        )
    except Exception as exc:  # pragma: no cover - bubbled via HTTP error
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    run_id = str(uuid4())
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO runs (id, params, metrics) VALUES (?, ?, ?)",
        (run_id, req.json(), json.dumps(metrics)),
    )
    conn.commit()
    conn.close()
    return RunResponse(run_id=run_id, metrics=metrics)


@app.get("/runs/{run_id}")
async def get_run(run_id: str) -> Dict[str, object]:
    ensure_db()
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT params, metrics FROM runs WHERE id=?", (run_id,)).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="run not found")
    params = json.loads(row[0])
    metrics = json.loads(row[1])
    return {"run_id": run_id, "params": params, "metrics": metrics}


class PredictRequest(BaseModel):
    window: list[list[float]]
    sigma_hat: float
    tau: float = 0.05
    sigma_target: float = 0.02
    wmax: float = 1.0
    blend: float = 0.6
    tau_k: float = 0.75
    prob_threshold: float = 0.0
    stop_loss: float = 0.0
    prob_scale: float = 1.0
    model_path: str = str(MODEL_DEFAULT)


@app.post("/predict")
async def predict(req: PredictRequest) -> Dict[str, float]:
    cfg = RiskMapConfig(
        tau=req.tau,
        sigma_target=req.sigma_target,
        wmax=req.wmax,
        blend=req.blend,
        tau_k=req.tau_k,
        prob_threshold=req.prob_threshold,
        stop_loss=req.stop_loss,
    )
    return predict_window(req.window, req.sigma_hat, cfg, req.prob_scale, Path(req.model_path))
