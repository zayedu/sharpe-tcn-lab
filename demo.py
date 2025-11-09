"""End-to-end Sharpe-optimized demo for the hybrid model."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from backtest import run_backtest
from data_loader import load_market_panel
from features import make_feature_tensor
from model_tcn import HybridSignalNet, export_jit
from research_utils import (
    benchmark,
    compute_regime,
    compute_sample_weights,
    ensemble_predict,
    apply_calibration,
    calibrate_probs,
    seed_all,
    summarize,
    train_ensemble,
)
from trainer import TrainConfig
from tuning import scale_probs, tune_riskmap


ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)
ENSEMBLE_PATH = ARTIFACTS / "hybrid_ensemble.pt"


def dynamic_sigma_targets(panel, dates, base_sigma: float) -> np.ndarray:
    vix = panel.get("^vix_close")
    if vix is None:
        vix = panel.get("vix_close")
    if vix is None:
        return np.full(len(dates), base_sigma)
    vix = vix.reindex(dates, method="ffill").fillna(method="bfill").values
    normalized = (vix - np.median(vix)) / (np.std(vix) + 1e-6)
    scaler = np.clip(1 + 0.05 * normalized, 0.6, 1.4)
    return base_sigma * scaler


def load_or_train_ensemble(
    tensor,
    train_mask,
    val_mask,
    weights: np.ndarray,
    seeds: tuple[int, ...],
    config: TrainConfig,
    force: bool,
) -> list[HybridSignalNet]:
    if ENSEMBLE_PATH.exists() and not force:
        state = torch.load(ENSEMBLE_PATH, map_location="cpu")
        models: list[HybridSignalNet] = []
        in_features = state.get("in_features", tensor.X.shape[-1])
        for sd in state.get("states", []):
            model = HybridSignalNet(in_features=in_features)
            model.load_state_dict(sd)
            model.eval()
            models.append(model)
        if models:
            print(
                f"Loaded ensemble ({len(models)} models) from {ENSEMBLE_PATH}. "
                "Set FORCE_TRAIN=1 to retrain."
            )
            return models
        print("Checkpoint empty, retraining ensemble...")
    models = train_ensemble(tensor, train_mask, val_mask, weights, seeds, config)
    torch.save(
        {"states": [m.state_dict() for m in models], "in_features": tensor.X.shape[-1]},
        ENSEMBLE_PATH,
    )
    print(f"Saved ensemble checkpoint to {ENSEMBLE_PATH}")
    return models


def main() -> None:
    BASE_SEED = 7
    seed_all(BASE_SEED, deterministic=True)
    panel = load_market_panel(["SPY", "^VIX", "^TNX"], "2004-01-01", "2024-12-31")
    tensor = make_feature_tensor(panel, symbol="SPY", window=96)
    regime = compute_regime(tensor.close, tensor.dates)
    train_cut = np.datetime64("2014-01-01")
    val_cut = np.datetime64("2018-01-01")
    train_mask = tensor.dates < train_cut
    val_mask = (tensor.dates >= train_cut) & (tensor.dates < val_cut)
    test_mask = tensor.dates >= val_cut
    if not (train_mask.any() and val_mask.any() and test_mask.any()):
        raise RuntimeError("Insufficient samples for requested splits")
    train_weights = compute_sample_weights(tensor.dates[train_mask])
    ensemble_seeds = tuple(BASE_SEED + offset for offset in (0, 17, 41))
    config = TrainConfig()
    force_retrain = os.environ.get("FORCE_TRAIN", "0") == "1"
    models = load_or_train_ensemble(
        tensor,
        train_mask,
        val_mask,
        train_weights,
        ensemble_seeds,
        config,
        force_retrain,
    )
    val_probs, val_pos, _ = ensemble_predict(models, tensor.X[val_mask])
    test_probs, test_pos, _ = ensemble_predict(models, tensor.X[test_mask])
    val_brier = float(np.mean((val_probs - tensor.y[val_mask]) ** 2))
    print(f"Validation Brier (ensemble): {val_brier:.4f}")
    alpha, beta = calibrate_probs(val_probs, tensor.y[val_mask])
    val_probs = apply_calibration(val_probs, alpha, beta)
    test_probs = apply_calibration(test_probs, alpha, beta)
    tuning, _ = tune_riskmap(
        probs=val_probs,
        positions=val_pos,
        sigma_hat=tensor.sigma[val_mask],
        regime=regime[val_mask],
        future_returns=tensor.future_ret[val_mask],
        price=tensor.close[val_mask],
        dates=tensor.dates[val_mask],
        tau_grid=(0.03, 0.05, 0.08),
        sigma_grid=(0.01, 0.015, 0.02, 0.03),
        wmax_grid=(0.75, 1.0, 1.25),
        blend_grid=(0.4, 0.6, 0.8),
        tau_k_grid=(0.5, 0.75, 1.0),
        scale_grid=(0.8, 1.0, 1.2, 1.5),
        prob_grid=(0.0, 0.02, 0.04),
        stop_grid=(0.0, 0.03, 0.05),
    )
    cfg = tuning.config
    np.savez(ARTIFACTS / "calibration.npz", alpha=alpha, beta=beta)
    sigma_override = dynamic_sigma_targets(panel, tensor.dates, cfg.sigma_target)
    scaled_val = scale_probs(val_probs, tuning.prob_scale)
    val_bt = run_backtest(
        probs=scaled_val,
        sigma_hat=tensor.sigma[val_mask],
        price=tensor.close[val_mask],
        future_returns=tensor.future_ret[val_mask],
        dates=tensor.dates[val_mask],
        positions=val_pos,
        cfg=cfg,
        regime=regime[val_mask],
        sigma_override=sigma_override[val_mask],
    )
    summarize("Validation", val_bt)
    scaled_test = scale_probs(test_probs, tuning.prob_scale)
    test_bt = run_backtest(
        probs=scaled_test,
        sigma_hat=tensor.sigma[test_mask],
        price=tensor.close[test_mask],
        future_returns=tensor.future_ret[test_mask],
        dates=tensor.dates[test_mask],
        positions=test_pos,
        cfg=cfg,
        regime=regime[test_mask],
        sigma_override=sigma_override[test_mask],
    )
    summarize("Test", test_bt)
    print(
        "Best risk-map config: "
        f"tau={cfg.tau:.2f} sigma_target={cfg.sigma_target:.3f} "
        f"wmax={cfg.wmax:.2f} blend={cfg.blend:.2f} tau_k={cfg.tau_k:.2f} "
        f"prob_scale={tuning.prob_scale:.2f}"
    )
    sample = torch.from_numpy(tensor.X[test_mask][: min(128, len(test_pos))])
    p50, p95, throughput = benchmark(models[0], sample)
    print(
        f"Inference latency p50={p50:.3f}ms p95={p95:.3f}ms "
        f"throughput={throughput:.1f} seq/s"
    )
    example = torch.from_numpy(tensor.X[test_mask][:1])
    export_path = export_jit(models[0], example, ARTIFACTS / "tcn.pt")
    print(f"Exported TorchScript model to {export_path}")


if __name__ == "__main__":
    main()
def dynamic_sigma_targets(panel, dates, base_sigma: float) -> np.ndarray:
    vix = panel.get("^vix_close") or panel.get("vix_close")
    if vix is None:
        return np.full(len(dates), base_sigma)
    vix = vix.reindex(dates, method="ffill").fillna(method="bfill").values
    normalized = (vix - np.median(vix)) / (np.std(vix) + 1e-6)
    scaler = np.clip(1 + 0.05 * normalized, 0.6, 1.4)
    return base_sigma * scaler
