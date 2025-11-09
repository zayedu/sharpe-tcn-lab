"""Shared helpers for the FastAPI service."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

from backtest import run_backtest
from data_loader import load_market_panel
from features import make_feature_tensor
from riskmap import RiskMapConfig, blend_positions
from tuning import scale_probs

DB_PATH = Path("runs.db")
MODEL_DEFAULT = Path("artifacts/tcn.pt")
CALIB_PATH = Path("artifacts/calibration.npz")
_MODEL_CACHE: Dict[Path, torch.jit.ScriptModule] = {}


def ensure_db(path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, params TEXT, metrics TEXT)")
    conn.commit()
    conn.close()


def load_model(path: Path) -> torch.jit.ScriptModule:
    if path not in _MODEL_CACHE:
        _MODEL_CACHE[path] = torch.jit.load(str(path))
    return _MODEL_CACHE[path]


def _regime(close: np.ndarray, dates: np.ndarray) -> np.ndarray:
    series = pd.Series(close, index=dates)
    fast = series.ewm(span=50, adjust=False).mean()
    slow = series.ewm(span=200, adjust=False).mean()
    return np.sign(fast - slow).fillna(0.0).values


def _dynamic_sigma_targets(panel: pd.DataFrame, dates: np.ndarray, base_sigma: float) -> np.ndarray:
    vix = panel.get("^vix_close") or panel.get("vix_close")
    if vix is None:
        return np.full(len(dates), base_sigma)
    vix = vix.reindex(dates, method="ffill").fillna(method="bfill").values
    normalized = (vix - np.median(vix)) / (np.std(vix) + 1e-6)
    scaler = np.clip(1 + 0.05 * normalized, 0.6, 1.4)
    return base_sigma * scaler


def _apply_saved_calibration(probs: np.ndarray) -> np.ndarray:
    if not CALIB_PATH.exists():
        return probs
    data = np.load(CALIB_PATH)
    alpha = float(data.get("alpha", 0.0))
    beta = float(data.get("beta", 1.0))
    clip_p = np.clip(probs, 1e-5, 1 - 1e-5)
    logits = np.log(clip_p / (1 - clip_p))
    logits = beta * logits + alpha
    return np.clip(1.0 / (1.0 + np.exp(-logits)), 1e-6, 1 - 1e-6)


def run_request(
    symbol: str,
    start: str,
    end: str,
    test_start: str,
    cfg: RiskMapConfig,
    prob_scale: float,
    cost_bps: float,
    model_path: Path,
) -> Dict[str, float]:
    panel = load_market_panel([symbol, "^VIX", "^TNX"], start, end)
    tensor = make_feature_tensor(panel, symbol=symbol)
    mask = tensor.dates >= np.datetime64(test_start)
    if not mask.any():
        raise ValueError("No samples for requested window")
    model = load_model(model_path)
    X = torch.from_numpy(tensor.X[mask])
    out = model(X)
    if isinstance(out, tuple):
        probs = out[0].detach().numpy()
        positions = out[2].detach().numpy()
    else:
        probs = out.prob.detach().numpy()
        positions = out.position.detach().numpy()
    probs = _apply_saved_calibration(probs)
    scaled = scale_probs(probs, prob_scale)
    regime = _regime(tensor.close[mask], tensor.dates[mask])
    sigma_override = _dynamic_sigma_targets(panel, tensor.dates, cfg.sigma_target)
    result = run_backtest(
        probs=scaled,
        sigma_hat=tensor.sigma[mask],
        price=tensor.close[mask],
        future_returns=tensor.future_ret[mask],
        dates=tensor.dates[mask],
        positions=positions,
        cfg=cfg,
        regime=regime,
        sigma_override=sigma_override[mask],
        cost_bps=cost_bps,
    )
    return result.metrics


def predict_window(
    window: list[list[float]],
    sigma_hat: float,
    cfg: RiskMapConfig,
    prob_scale: float,
    model_path: Path,
) -> Dict[str, float]:
    model = load_model(model_path)
    with torch.no_grad():
        tensor = torch.tensor([window], dtype=torch.float32)
        out = model(tensor)
    if isinstance(out, tuple):
        prob = float(out[0].item())
        position = float(out[2].item())
    else:
        prob = float(out.prob.item())
        position = float(out.position.item())
    prob = _apply_saved_calibration(np.array([prob]))[0]
    scaled = float(scale_probs(np.array([prob]), prob_scale)[0])
    weight = float(
        blend_positions(
            np.array([position]),
            np.array([scaled]),
            np.array([sigma_hat]),
            cfg,
            np.zeros(1),
        )[0]
    )
    return {"p_hat": scaled, "sigma_hat": sigma_hat, "position": position, "w": weight}
