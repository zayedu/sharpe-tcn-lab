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


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / dim))
        pe = torch.zeros(max_len, dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        return self.dropout(x + self.pe[:length].unsqueeze(0))


class SpectralSummary(nn.Module):
    def __init__(self, in_features: int, modes: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.modes = modes
        self.proj = nn.Sequential(
            nn.Linear(max(2 * modes, 4), out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        freq = torch.fft.rfft(x, dim=1)
        freq = freq.mean(dim=2)  # average over feature dimension
        max_modes = min(self.modes, freq.shape[1])
        real = freq.real[:, :max_modes]
        imag = freq.imag[:, :max_modes]
        spec = torch.cat([real, imag], dim=-1)
        needed = self.proj[0].in_features
        if spec.size(-1) < needed:
            spec = F.pad(spec, (0, needed - spec.size(-1)))
        return self.proj(spec)


class GatedResidualUnit(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim_in, dim_out * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sigmoid()
        self.skip = nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()
        self.norm = nn.LayerNorm(dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = self.proj(x)
        gate, value = proj.chunk(2, dim=-1)
        gated = value * self.gate(gate)
        return self.norm(gated + self.skip(x))


class HybridSignalNet(nn.Module):
    def __init__(
        self,
        in_features: int,
        channels: Iterable[int] = (32, 64, 96, 128),
        kernel_size: int = 3,
        dropout: float = 0.1,
        attn_heads: int = 4,
        transformer_layers: int = 2,
        spectral_modes: int = 16,
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
        hidden = channels[-1]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=attn_heads,
            dim_feedforward=hidden * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.pos_encoder = PositionalEncoding(hidden, dropout, max_len=768)
        self.attn = nn.MultiheadAttention(hidden, attn_heads, dropout=dropout, batch_first=True)
        self.spectral = SpectralSummary(in_features, spectral_modes, hidden, dropout)
        fusion_dim = hidden * 3 + hidden
        self.fusion = GatedResidualUnit(fusion_dim, hidden, dropout)
        self.head_dropout = nn.Dropout(dropout)
        self.cls_head = nn.Linear(hidden, 1)
        self.sigma_head = nn.Linear(hidden, 1)
        self.pos_head = nn.Linear(hidden, 1)
        self.log_temp = nn.Parameter(torch.zeros(1))

    def _temperature(self) -> torch.Tensor:
        return F.softplus(self.log_temp) + 1e-3

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        xt = x.transpose(1, 2)
        feats = self.tcn(xt).transpose(1, 2)
        context = self.transformer(self.pos_encoder(feats))
        attn_out, _ = self.attn(context, context, context, need_weights=False)
        pooled = torch.cat(
            [context.mean(dim=1), context[:, -1, :], attn_out.mean(dim=1)],
            dim=1,
        )
        spectral = self.spectral(x)
        fused = torch.cat([pooled, spectral], dim=-1)
        return self.fusion(fused)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        rep = self._extract(x)
        rep = self.head_dropout(rep)
        logits = self.cls_head(rep).squeeze(-1)
        sigma = F.softplus(self.sigma_head(rep).squeeze(-1))
        position = torch.tanh(self.pos_head(rep).squeeze(-1))
        prob = torch.sigmoid(logits / self._temperature())
        return ModelOutput(prob=prob, logits=logits, position=position, sigma=sigma)


def export_jit(model: nn.Module, example: torch.Tensor, path: str | Path) -> Path:
    model.eval()
    device = next(model.parameters()).device
    example = example.to(device)
    with torch.no_grad():
        traced = torch.jit.trace(model, example, check_trace=False)
    traced = traced.to("cpu")
    out_path = Path(path)
    traced.save(str(out_path))
    return out_path
