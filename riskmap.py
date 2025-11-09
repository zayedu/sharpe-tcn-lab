"""Volatility-scaled and regime-aware risk mapping."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RiskMapConfig:
    tau: float = 0.05
    sigma_target: float = 0.02
    wmax: float = 1.0
    blend: float = 0.6
    tau_k: float = 0.75
    prob_threshold: float = 0.0  # |p-0.5| threshold for neutrality
    stop_loss: float = 0.0  # expressed as fraction of entry price
    confidence_gamma: float = 1.0  # >1 down-weights low confidence trades


def _apply_prob_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    if threshold <= 0:
        return probs
    delta = np.abs(probs - 0.5)
    mask = delta < threshold
    adjusted = probs.copy()
    adjusted[mask] = 0.5
    return adjusted


def deadband(signal: np.ndarray, tau: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    mask = np.abs(signal) <= tau
    trimmed = np.sign(signal) * (np.abs(signal) - tau)
    denom = np.maximum(1e-9, 1 - tau)
    out = np.where(mask, 0.0, trimmed / denom)
    return np.clip(out, -1.0, 1.0)


def adaptive_tau(sigma_hat: np.ndarray, cfg: RiskMapConfig) -> np.ndarray:
    ratio = np.asarray(sigma_hat) / (cfg.sigma_target + 1e-9)
    adj = cfg.tau * (1 + cfg.tau_k * (ratio - 1))
    return np.clip(adj, cfg.tau * 0.5, cfg.tau * 2.5)


def volatility_scaled_weights(
    probs: np.ndarray,
    sigma_hat: np.ndarray,
    cfg: RiskMapConfig,
    regime: np.ndarray | None = None,
    sigma_override: np.ndarray | None = None,
) -> np.ndarray:
    probs = _apply_prob_threshold(np.asarray(probs, dtype=np.float64), cfg.prob_threshold)
    signal = 2 * probs - 1
    tau = adaptive_tau(sigma_hat, cfg)
    neutral = deadband(signal, tau)
    target_sigma = sigma_override if sigma_override is not None else cfg.sigma_target
    scale = np.asarray(target_sigma) / (np.asarray(sigma_hat) + 1e-9)
    weights = scale * neutral
    if cfg.confidence_gamma != 1.0:
        confidence = np.clip(np.abs(probs - 0.5) * 2, 0.0, 1.0)
        weights *= confidence ** cfg.confidence_gamma
    if regime is not None:
        weights *= 1 + 0.5 * regime
    return np.clip(weights, -cfg.wmax, cfg.wmax)


def blend_positions(
    direct: np.ndarray,
    probs: np.ndarray,
    sigma_hat: np.ndarray,
    cfg: RiskMapConfig,
    regime: np.ndarray,
    sigma_override: np.ndarray | None = None,
) -> np.ndarray:
    vol_scaled = volatility_scaled_weights(probs, sigma_hat, cfg, regime, sigma_override)
    blended = cfg.blend * np.asarray(direct) + (1 - cfg.blend) * vol_scaled
    return np.clip(blended, -cfg.wmax, cfg.wmax)
