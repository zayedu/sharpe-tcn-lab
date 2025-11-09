"""Hybrid multi-task forecaster combining TCN and attention."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, NamedTuple

import torch
from torch import nn
import torch.nn.functional as F


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        padding = (kernel_size - 1) * dilation
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self._pad = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = super().forward(x)
        return out[..., : -self._pad] if self._pad else out


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int, kernel: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel, dilation)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_ch)
        self.act = nn.GELU()
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.act(self.norm(out.transpose(1, 2))).transpose(1, 2)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.act(self.norm(out.transpose(1, 2))).transpose(1, 2)
        out = self.dropout(out)
        return out + self.residual(x)


class ModelOutput(NamedTuple):
    prob: torch.Tensor
    logits: torch.Tensor
    position: torch.Tensor
    sigma: torch.Tensor


class HybridSignalNet(nn.Module):
    def __init__(
        self,
        in_features: int,
        channels: Iterable[int] = (32, 64, 96, 128),
        kernel_size: int = 3,
        dropout: float = 0.1,
        attn_heads: int = 4,
    ) -> None:
        super().__init__()
        dims = [in_features, *channels]
        blocks = []
        for idx in range(len(channels)):
            blocks.append(
                TemporalBlock(
                    dims[idx],
                    dims[idx + 1],
                    dilation=2**idx,
                    kernel=kernel_size,
                    dropout=dropout,
                )
            )
        self.tcn = nn.Sequential(*blocks)
        self.attn = nn.MultiheadAttention(channels[-1], attn_heads, dropout=dropout, batch_first=True)
        hidden = channels[-1]
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
        )
        self.cls_head = nn.Linear(hidden, 1)
        self.sigma_head = nn.Linear(hidden, 1)
        self.pos_head = nn.Linear(hidden, 1)
        self.log_temp = nn.Parameter(torch.zeros(1))

    def _temperature(self) -> torch.Tensor:
        return F.softplus(self.log_temp) + 1e-3

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        xt = x.transpose(1, 2)
        feats = self.tcn(xt).transpose(1, 2)
        attn_out, _ = self.attn(feats, feats, feats, need_weights=False)
        pooled = torch.cat([attn_out.mean(dim=1), attn_out[:, -1, :]], dim=1)
        return self.fusion(pooled)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        rep = self._extract(x)
        logits = self.cls_head(rep).squeeze(-1)
        sigma = F.softplus(self.sigma_head(rep).squeeze(-1))
        position = torch.tanh(self.pos_head(rep).squeeze(-1))
        prob = torch.sigmoid(logits / self._temperature())
        return ModelOutput(prob=prob, logits=logits, position=position, sigma=sigma)


def export_jit(model: nn.Module, example: torch.Tensor, path: str | Path) -> Path:
    model.eval()
    traced = torch.jit.trace(model, example, check_trace=False)
    out_path = Path(path)
    traced.save(str(out_path))
    return out_path
