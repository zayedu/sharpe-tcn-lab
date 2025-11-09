"""Validation-driven tuning for risk map and probability scaling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from backtest import run_backtest
from riskmap import RiskMapConfig


def _logit(p: np.ndarray) -> np.ndarray:
    eps = 1e-6
    clipped = np.clip(p, eps, 1 - eps)
    return np.log(clipped / (1 - clipped))


def scale_probs(probs: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return probs
    logits = _logit(probs) * scale
    return 1.0 / (1.0 + np.exp(-logits))


@dataclass
class TuningResult:
    config: RiskMapConfig
    prob_scale: float


def _fold_indices(length: int, folds: int = 2) -> list[slice]:
    step = max(1, length // folds)
    return [slice(i * step, None if i == folds - 1 else (i + 1) * step) for i in range(folds)]


def tune_riskmap(
    probs: np.ndarray,
    positions: np.ndarray,
    sigma_hat: np.ndarray,
    regime: np.ndarray,
    future_returns: np.ndarray,
    price: np.ndarray,
    dates: np.ndarray,
    tau_grid: Iterable[float],
    sigma_grid: Iterable[float],
    wmax_grid: Iterable[float],
    blend_grid: Iterable[float],
    tau_k_grid: Iterable[float],
    scale_grid: Iterable[float],
    prob_grid: Iterable[float],
    stop_grid: Iterable[float],
    cost_bps: float = 3.0,
) -> Tuple[TuningResult, dict]:
    best_cfg, best_metrics = None, None
    best_score = float("-inf")
    folds = _fold_indices(len(probs), folds=2)
    for scale in scale_grid:
        scaled_probs = scale_probs(probs, scale)
        for tau in tau_grid:
            for sigma_target in sigma_grid:
                for wmax in wmax_grid:
                    for blend in blend_grid:
                        for tau_k in tau_k_grid:
                            for prob_thr in prob_grid:
                                for stop in stop_grid:
                                    cfg = RiskMapConfig(
                                        tau=tau,
                                        sigma_target=sigma_target,
                                        wmax=wmax,
                                        blend=blend,
                                        tau_k=tau_k,
                                        prob_threshold=prob_thr,
                                        stop_loss=stop,
                                    )
                                    sharpe_scores = []
                                    for fold in folds:
                                        sl = fold
                                        fold_res = run_backtest(
                                            probs=scaled_probs[sl],
                                            sigma_hat=sigma_hat[sl],
                                            price=price[sl],
                                            future_returns=future_returns[sl],
                                            dates=dates[sl],
                                            positions=positions[sl],
                                            cfg=cfg,
                                            regime=regime[sl],
                                            cost_bps=cost_bps,
                                        )
                                        sharpe_scores.append(fold_res.metrics["sharpe"])
                                    score = float(np.nanmean(sharpe_scores))
                                    if score > best_score:
                                        best_score = score
                                        best_cfg = TuningResult(config=cfg, prob_scale=scale)
                                        best_metrics = fold_res.metrics
    if best_cfg is None or best_metrics is None:
        raise RuntimeError("Risk-map tuning failed")
    return best_cfg, best_metrics
