"""Deterministic Yahoo Finance data access with lightweight caching."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import yfinance as yf

CACHE_DIR = Path("cache")
COLUMNS = ["open", "high", "low", "close", "adj close", "volume"]


def _cache_file(symbol: str, start: str, end: str, interval: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    safe = f"{symbol}_{start}_{end}_{interval}".replace(":", "-")
    return CACHE_DIR / f"{safe}.csv"


def load_ohlcv(
    symbol: str,
    start: str,
    end: str,
    interval: Literal["1d", "1h", "30m"] = "1d",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download OHLCV bars and persist to CSV for reproducibility."""
    path = _cache_file(symbol, start, end, interval)
    if path.exists() and not force_refresh:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        df = yf.download(
            symbol,
            start=start,
            end=end,
            interval=interval,
            progress=False,
            auto_adjust=False,
        )
        if df.empty:
            raise ValueError(f"No data returned for {symbol}")
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.xs(symbol, axis=1, level=-1)
            except (KeyError, IndexError):
                df.columns = df.columns.get_level_values(0)
        df.to_csv(path, index_label="Date")
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(symbol, axis=1, level=-1)
    df = df.rename(columns=str.lower)
    if "adj close" not in df.columns:
        df["adj close"] = df.get("close", df.iloc[:, 0])
    missing = [col for col in COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in data feed: {missing}")
    df = df[COLUMNS].copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().astype(float)
    df.index.name = "date"
    return df.sort_index()


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Augment dataframe with percentage and log returns."""
    work = df.copy()
    work["ret"] = work["close"].pct_change(fill_method=None).fillna(0.0)
    work["log_ret"] = np.log(work["close"]).diff().fillna(0.0)
    return work


def load_market_panel(
    symbols: Iterable[str],
    start: str,
    end: str,
    interval: Literal["1d", "1h", "30m"] = "1d",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download multiple symbols and align them on the index."""
    frames = []
    for sym in symbols:
        frame = load_ohlcv(sym, start, end, interval, force_refresh)
        frame = frame.add_prefix(f"{sym.lower()}_")
        frames.append(frame)
    panel = pd.concat(frames, axis=1).dropna()
    panel.columns = [col.lower() for col in panel.columns]
    panel.index.name = "date"
    return panel
