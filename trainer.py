"""Sharpe-aware training orchestration for the hybrid architecture."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn

from model_tcn import HybridSignalNet, ModelOutput
from training_utils import evaluate, make_loader, sharpe_loss, sortino_loss


@dataclass
class TrainConfig:
    max_epochs: int = 160
    batch_size: int = 128
    lr: float = 1.5e-4
    weight_decay: float = 1e-4
    patience: int = 20
    max_grad_norm: float = 1.0
    bce_weight: float = 1.0
    vol_weight: float = 0.3
    sharpe_weight: float = 0.7
    sortino_weight: float = 0.2
    cost_bps: float = 3.0
    turnover_weight: float = 0.1
    stage1_epochs: int = 40
    stage2_bce_mult: float = 0.3
    stage2_sharpe_mult: float = 1.8
    stage2_sortino_mult: float = 2.0


@dataclass
class TrainReport:
    epochs: int
    val_brier: float
    val_sharpe: float
    val_loss: float


def train_tcn(
    train_X: np.ndarray,
    train_y: np.ndarray,
    train_future_ret: np.ndarray,
    train_sigma: np.ndarray,
    train_future_sigma: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    val_future_ret: np.ndarray,
    val_sigma: np.ndarray,
    val_future_sigma: np.ndarray,
    config: TrainConfig | None = None,
    train_weights: np.ndarray | None = None,
) -> Tuple[HybridSignalNet, TrainReport]:
    if config is None:
        config = TrainConfig()
    model = HybridSignalNet(in_features=train_X.shape[-1])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    loader = make_loader(
        train_X,
        train_y,
        train_future_ret,
        train_sigma,
        train_future_sigma,
        train_weights,
        config.batch_size,
        shuffle=True,
    )
    val_loader = make_loader(
        val_X,
        val_y,
        val_future_ret,
        val_sigma,
        val_future_sigma,
        weights=None,
        batch_size=len(val_y),
        shuffle=False,
    )
    pos = float(train_y.mean())
    pos_weight = torch.tensor([(1 - pos) / max(pos, 1e-3)], dtype=torch.float32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    best_state, best_metric = None, float("inf")
    patience = 0
    epochs_run = 0
    cost = config.cost_bps * 1e-4
    for epoch in range(config.max_epochs):
        epochs_run = epoch + 1
        model.train()
        prev_pos = None
        cur_bce_weight = config.bce_weight
        cur_sharpe_weight = config.sharpe_weight
        cur_sortino_weight = config.sortino_weight
        if epoch >= config.stage1_epochs:
            cur_bce_weight *= config.stage2_bce_mult
            cur_sharpe_weight *= config.stage2_sharpe_mult
            cur_sortino_weight *= config.stage2_sortino_mult
        for batch in loader:
            xb, yb, fut_ret, sigma, fut_sigma, *rest = batch
            xb = xb.to(device)
            yb = yb.to(device)
            fut_ret = fut_ret.to(device)
            fut_sigma = fut_sigma.to(device)
            sample_weight = rest[0] if rest else None
            if sample_weight is not None:
                sample_weight = sample_weight.to(device)
            optimizer.zero_grad(set_to_none=True)
            out: ModelOutput = model(xb)
            bce = nn.functional.binary_cross_entropy_with_logits(
                out.logits, yb, weight=sample_weight, pos_weight=pos_weight
            )
            vol_loss = nn.functional.mse_loss(out.sigma, fut_sigma)
            sharpe = sharpe_loss(out.position, fut_ret, cost)
            sortino = sortino_loss(out.position, fut_ret, cost)
            turnover_pen = 0.0
            if prev_pos is not None:
                min_len = min(prev_pos.shape[0], out.position.shape[0])
                turnover_pen = torch.mean(torch.abs(out.position[:min_len] - prev_pos[:min_len]))
            prev_pos = out.position.detach()
            loss = (
                cur_bce_weight * bce
                + config.vol_weight * vol_loss
                + cur_sharpe_weight * sharpe
                + cur_sortino_weight * sortino
                + config.turnover_weight * turnover_pen
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
        val_metrics = evaluate(model, val_loader, cost, device)
        metric = val_metrics["brier"] - val_metrics["sharpe"]
        if metric + 1e-5 < best_metric:
            best_metric = metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    final_metrics = evaluate(model, val_loader, cost, device)
    report = TrainReport(
        epochs=epochs_run,
        val_brier=final_metrics["brier"],
        val_sharpe=final_metrics["sharpe"],
        val_loss=best_metric,
    )
    return model, report
