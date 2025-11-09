"""Utility helpers shared across research scripts."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import torch

from model_tcn import HybridSignalNet, ModelOutput
from trainer import TrainConfig, train_tcn


@dataclass
class EnsembleBundle:
    base_models: list[HybridSignalNet]
    trend_model: HybridSignalNet | None
    mean_model: HybridSignalNet | None


def seed_all(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


def benchmark(model: HybridSignalNet, sample: torch.Tensor, batches: int = 200) -> Tuple[float, float, float]:
    latencies = []
    model.eval()
    device = next(model.parameters()).device
    sample = sample.to(device)
    with torch.no_grad():
        for _ in range(batches):
            start = time.perf_counter()
            model(sample)
            latencies.append(time.perf_counter() - start)
    lat = np.array(latencies)
    p50 = np.percentile(lat, 50) * 1e3
    p95 = np.percentile(lat, 95) * 1e3
    throughput = sample.shape[0] * batches / lat.sum()
    return p50, p95, throughput


def compute_regime(close: np.ndarray, dates: np.ndarray, fast: int = 50, slow: int = 200) -> np.ndarray:
    series = pd.Series(close, index=pd.to_datetime(dates))
    fast_ma = series.ewm(span=fast, adjust=False).mean()
    slow_ma = series.ewm(span=slow, adjust=False).mean()
    return np.sign((fast_ma - slow_ma).fillna(0.0).to_numpy())


def compute_sample_weights(dates: np.ndarray, half_life_days: int = 504) -> np.ndarray:
    dates_np = np.asarray(dates, dtype="datetime64[ns]")
    max_date = dates_np.max()
    delta = (max_date - dates_np).astype("timedelta64[s]").astype(np.int64) // 86400
    weights = np.exp(-delta / max(1, half_life_days))
    return (weights / weights.mean()).astype(np.float32)


def ensemble_predict(
    bundle: EnsembleBundle,
    data: np.ndarray,
    regime: np.ndarray | None = None,
    trend_weight: float = 0.7,
    mean_weight: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensor = torch.from_numpy(data)
    probs_list, pos_list, sig_list = [], [], []
    for model in bundle.base_models:
        model.eval()
        with torch.no_grad():
            device = next(model.parameters()).device
            out: ModelOutput = model(tensor.to(device))
            probs_list.append(out.prob.cpu().numpy())
            pos_list.append(out.position.cpu().numpy())
            sig_list.append(out.sigma.cpu().numpy())
    probs = np.mean(probs_list, axis=0)
    positions = np.mean(pos_list, axis=0)
    sigma = np.mean(sig_list, axis=0)
    if regime is not None and bundle.trend_model is not None:
        trend_mask = regime > 0
        if trend_mask.any():
            bundle.trend_model.eval()
            with torch.no_grad():
                device = next(bundle.trend_model.parameters()).device
                out = bundle.trend_model(tensor.to(device))
                probs = trend_weight * out.prob.cpu().numpy() * trend_mask + probs * (~trend_mask)
                positions = trend_weight * out.position.cpu().numpy() * trend_mask + positions * (~trend_mask)
    if regime is not None and bundle.mean_model is not None:
        mean_mask = regime <= 0
        if mean_mask.any():
            bundle.mean_model.eval()
            with torch.no_grad():
                device = next(bundle.mean_model.parameters()).device
                out = bundle.mean_model(tensor.to(device))
                probs = mean_weight * out.prob.cpu().numpy() * mean_mask + probs * (~mean_mask)
                positions = mean_weight * out.position.cpu().numpy() * mean_mask + positions * (~mean_mask)
    return probs, positions, sigma


def train_ensemble(
    tensor,
    train_mask,
    val_mask,
    weights: np.ndarray,
    seeds: Iterable[int],
    config: TrainConfig,
    regime: np.ndarray,
) -> EnsembleBundle:
    models: list[HybridSignalNet] = []
    for seed in seeds:
        seed_all(seed)
        model, _ = train_tcn(
            tensor.X[train_mask],
            tensor.y[train_mask],
            tensor.future_ret[train_mask],
            tensor.sigma[train_mask],
            tensor.future_sigma[train_mask],
            tensor.X[val_mask],
            tensor.y[val_mask],
            tensor.future_ret[val_mask],
            tensor.sigma[val_mask],
            tensor.future_sigma[val_mask],
            config=config,
            train_weights=weights,
        )
        models.append(model)
    trend_model = _train_specialized_model(tensor, train_mask, val_mask, config, regime > 0)
    mean_model = _train_specialized_model(tensor, train_mask, val_mask, config, regime <= 0)
    return EnsembleBundle(base_models=models, trend_model=trend_model, mean_model=mean_model)


def _train_specialized_model(
    tensor,
    train_mask,
    val_mask,
    config: TrainConfig,
    mask: np.ndarray,
) -> HybridSignalNet | None:
    if mask[train_mask].sum() < 64 or mask[val_mask].sum() < 32:
        return None
    model, _ = train_tcn(
        tensor.X[train_mask][mask[train_mask]],
        tensor.y[train_mask][mask[train_mask]],
        tensor.future_ret[train_mask][mask[train_mask]],
        tensor.sigma[train_mask][mask[train_mask]],
        tensor.future_sigma[train_mask][mask[train_mask]],
        tensor.X[val_mask][mask[val_mask]],
        tensor.y[val_mask][mask[val_mask]],
        tensor.future_ret[val_mask][mask[val_mask]],
        tensor.sigma[val_mask][mask[val_mask]],
        tensor.future_sigma[val_mask][mask[val_mask]],
        config=config,
        train_weights=compute_sample_weights(tensor.dates[train_mask][mask[train_mask]]),
    )
    return model


def summarize(label: str, metrics_or_result) -> None:
    metrics = metrics_or_result.metrics if hasattr(metrics_or_result, "metrics") else metrics_or_result
    print(
        f"{label} Sharpe={metrics['sharpe']:.2f} | MaxDD={metrics['max_drawdown']:.2%} | "
        f"Turnover={metrics['turnover']:.3f} | HitRate={metrics['hit_rate']:.2f}"
    )


def calibrate_probs(probs: np.ndarray, labels: np.ndarray, iters: int = 200) -> tuple[float, float]:
    clip_p = np.clip(probs, 1e-5, 1 - 1e-5)
    p = torch.from_numpy(clip_p.astype(np.float32))
    y = torch.from_numpy(labels.astype(np.float32))
    logit = torch.logit(p)
    alpha = torch.zeros(1, requires_grad=True)
    beta = torch.ones(1, requires_grad=True)
    optim = torch.optim.Adam([alpha, beta], lr=0.05)
    for _ in range(iters):
        optim.zero_grad(set_to_none=True)
        logits = beta * logit + alpha
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        optim.step()
    return float(alpha.item()), float(beta.item())


def apply_calibration(probs: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    clip_p = np.clip(probs, 1e-5, 1 - 1e-5)
    logit = np.log(clip_p / (1 - clip_p))
    logits = beta * logit + alpha
    calibrated = 1.0 / (1.0 + np.exp(-logits))
    return np.clip(calibrated, 1e-6, 1 - 1e-6)
