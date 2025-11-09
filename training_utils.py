"""Shared helpers for Sharpe-aware training."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model_tcn import HybridSignalNet, ModelOutput


def _to_float_array(arr: np.ndarray | list | tuple) -> np.ndarray:
    return np.asarray(arr, dtype=np.float32)


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    future_ret: np.ndarray,
    sigma: np.ndarray,
    future_sigma: np.ndarray,
    weights: np.ndarray | None,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    tensors = [
        torch.from_numpy(_to_float_array(X)),
        torch.from_numpy(_to_float_array(y)),
        torch.from_numpy(_to_float_array(future_ret)),
        torch.from_numpy(_to_float_array(sigma)),
        torch.from_numpy(_to_float_array(future_sigma)),
    ]
    if weights is not None:
        tensors.append(torch.from_numpy(_to_float_array(weights)))
    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def sharpe_loss(positions: torch.Tensor, future_ret: torch.Tensor, cost: float) -> torch.Tensor:
    if len(positions) == 1:
        pnl = positions * future_ret
    else:
        diff = positions[1:] - positions[:-1]
        turnover = torch.zeros_like(positions)
        turnover[1:] = diff.abs()
        pnl = positions * future_ret - cost * turnover
    mean = pnl.mean()
    std = pnl.std(unbiased=False) + 1e-6
    return -(mean / std)


def sortino_loss(positions: torch.Tensor, future_ret: torch.Tensor, cost: float) -> torch.Tensor:
    if len(positions) == 1:
        pnl = positions * future_ret
    else:
        diff = positions[1:] - positions[:-1]
        turnover = torch.zeros_like(positions)
        turnover[1:] = diff.abs()
        pnl = positions * future_ret - cost * turnover
    downside = torch.sqrt((torch.clamp(-pnl, min=0.0) ** 2).mean() + 1e-6)
    return -(pnl.mean() / downside)


def evaluate(model: HybridSignalNet, loader: DataLoader, cost: float, device: torch.device) -> dict:
    model.eval()
    probs, targets, returns, positions = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            xb, yb, fut_ret, _, _, *_ = batch
            xb = xb.to(device)
            yb = yb.to(device)
            fut_ret = fut_ret.to(device)
            out: ModelOutput = model(xb)
            probs.append(out.prob.cpu())
            targets.append(yb.cpu())
            returns.append(fut_ret.cpu())
            positions.append(out.position.cpu())
    probs_t = torch.cat(probs)
    targets_t = torch.cat(targets)
    returns_t = torch.cat(returns)
    positions_t = torch.cat(positions)
    brier = nn.functional.mse_loss(probs_t, targets_t).item()
    sharpe = -sharpe_loss(positions_t, returns_t, cost).item()
    return {"brier": brier, "sharpe": sharpe}
