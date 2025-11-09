"""Vectorized backtester and analytics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from riskmap import RiskMapConfig, blend_positions, volatility_scaled_weights


@dataclass
class BacktestResult:
    metrics: Dict[str, float]
    equity: np.ndarray
    weights: np.ndarray
    pnl: np.ndarray
    dates: Sequence


def _stats(curve: np.ndarray, pnl: np.ndarray) -> Dict[str, float]:
    ret = pnl / curve[:-1]
    std = ret.std() + 1e-12
    downside = ret[ret < 0].std() + 1e-12
    sharpe = np.sqrt(252.0) * ret.mean() / std
    sortino = np.sqrt(252.0) * ret.mean() / downside
    equity_max = np.maximum.accumulate(curve)
    drawdown = (curve - equity_max) / equity_max
    hit_rate = float((ret > 0).mean())
    return {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(drawdown.min()),
        "hit_rate": hit_rate,
    }


def run_backtest(
    probs: np.ndarray | None,
    sigma_hat: np.ndarray,
    price: np.ndarray,
    future_returns: np.ndarray,
    dates: Sequence,
    positions: np.ndarray | None = None,
    cfg: RiskMapConfig | None = None,
    regime: np.ndarray | None = None,
    sigma_override: np.ndarray | None = None,
    cost_bps: float = 3.0,
) -> BacktestResult:
    n = len(future_returns)
    if cfg is None:
        cfg = RiskMapConfig()
    reg = regime if regime is not None else np.zeros_like(sigma_hat)
    if positions is not None and probs is not None:
        weights = blend_positions(positions, probs, sigma_hat, cfg, reg, sigma_override)
    elif positions is not None:
        weights = np.clip(positions, -cfg.wmax, cfg.wmax)
    elif probs is not None:
        weights = volatility_scaled_weights(probs, sigma_hat, cfg, reg, sigma_override)
    else:
        raise ValueError("Either probs or positions must be provided")
    fee = cost_bps * 1e-4
    equity = np.empty(n + 1, dtype=np.float64)
    pnl = np.empty(n, dtype=np.float64)
    equity[0] = 1.0
    cash, inventory = 1.0, 0.0
    entry_price = None
    for t in range(n):
        px = price[t]
        equity_t = cash + inventory * px
        if not np.isclose(equity_t, equity[t], atol=1e-9):
            raise RuntimeError("Equity tracking drift detected")
        target_val = weights[t] * equity_t
        trade_val = target_val - inventory * px
        inventory += trade_val / px if px != 0 else 0.0
        cash -= trade_val + abs(trade_val) * fee
        post_trade_equity = cash + inventory * px
        if not np.isclose(post_trade_equity, equity_t - abs(trade_val) * fee, atol=1e-9):
            raise RuntimeError("Equity accounting mismatch")
        if inventory == 0:
            entry_price = None
        elif entry_price is None or np.sign(inventory) != np.sign(target_val):
            entry_price = px
        if cfg.stop_loss > 0 and inventory != 0 and entry_price is not None:
            loss_frac = (px - entry_price) / entry_price
            triggered = False
            if inventory > 0 and loss_frac <= -cfg.stop_loss:
                triggered = True
            elif inventory < 0 and loss_frac >= cfg.stop_loss:
                triggered = True
            if triggered:
                trade_val = -inventory * px
                cash -= trade_val + abs(trade_val) * fee
                inventory = 0.0
                entry_price = None
                weights[t] = 0.0
        next_price = px * (1.0 + future_returns[t])
        new_equity = cash + inventory * next_price
        pnl[t] = new_equity - equity_t
        equity[t + 1] = new_equity
    metrics = _stats(equity, pnl)
    turnover = np.mean(np.abs(np.diff(np.concatenate([[0.0], weights]))))
    metrics["turnover"] = float(turnover)
    realized = (future_returns > 0).astype(float)
    brier = float(((probs - realized) ** 2).mean())
    metrics["brier"] = brier
    return BacktestResult(metrics=metrics, equity=equity, weights=weights, pnl=pnl, dates=dates)
