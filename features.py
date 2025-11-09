"""Feature engineering for the hybrid TCN + transformer stack."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    return (series - mean) / (std.replace(0, np.nan))


def _bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    mid = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    return (series - mid) / (std * num_std + 1e-6)


def _stochastic(df: pd.DataFrame, window: int = 14) -> pd.Series:
    highest = df["high"].rolling(window).max()
    lowest = df["low"].rolling(window).min()
    return (df["close"] - lowest) / (highest - lowest + 1e-6)


def build_feature_frame(panel: pd.DataFrame, symbol: str = "spy") -> tuple[pd.DataFrame, pd.Series]:
    cols = {k.split("_")[-1]: panel[f"{symbol}_{k}"] for k in ("open", "high", "low", "close", "volume")}
    df = pd.DataFrame(cols)
    ret = df["close"].pct_change()
    log_ret = np.log(df["close"]).diff()
    vol = ret.rolling(20).std(ddof=0)
    macd = _ema(df["close"], 12) - _ema(df["close"], 26)
    signal = _ema(macd, 9)
    atr = (_ema(df["high"] - df["low"], 14) + _ema((df["close"] - df["close"].shift()).abs(), 14)) / 2
    rsi_delta = df["close"].diff()
    up = rsi_delta.clip(lower=0)
    down = (-rsi_delta).clip(lower=0)
    rs = _ema(up, 14) / (_ema(down, 14) + 1e-9)
    rsi = 100 - 100 / (1 + rs)
    stoch = _stochastic(df)
    vol_z = _zscore(df["volume"], 63)
    bb = _bollinger(df["close"], 20, 2)
    mom = {f"mom_{w}": df["close"].pct_change(w) for w in (5, 20, 63, 126, 252)}
    vol_windows = {f"vol_{w}": ret.rolling(w).std(ddof=0) for w in (10, 20, 63, 126)}
    sharpe_roll = ret.rolling(63).mean() / (ret.rolling(63).std(ddof=0) + 1e-6)
    skew_roll = ret.rolling(126).skew()
    kurt_roll = ret.rolling(126).kurt()
    macro = {}
    for col in panel.columns:
        lower = col.lower()
        if lower.endswith("close") and not lower.startswith(symbol):
            macro[lower] = panel[col].pct_change().fillna(0.0)
    vix = panel.get("^vix_close", panel.get("vix_close"))
    if vix is not None:
        macro["vix_lvl"] = vix
        macro["vix_chg"] = vix.pct_change().fillna(0.0)
    tnx = panel.get("^tnx_close", panel.get("tnx_close"))
    if tnx is not None:
        macro["yield_curve"] = (df["close"] / (tnx + 1e-6)).pct_change().fillna(0.0)
    fast = _ema(df["close"], 50)
    slow = _ema(df["close"], 200)
    regime = np.sign(fast - slow).fillna(0.0)
    cpd = np.sign(ret.rolling(63).mean() - ret.rolling(252).mean()).fillna(0.0)
    cal = df.index
    season = {
        "season_sin": np.sin(2 * np.pi * cal.dayofyear / 365.25),
        "season_cos": np.cos(2 * np.pi * cal.dayofyear / 365.25),
    }
    feat_dict = {
        "ret_1": ret,
        "log_ret": log_ret,
        "range": (df["high"] - df["low"]) / (df["close"].shift(1) + 1e-6),
        "atr": atr,
        "macd": macd,
        "macd_signal": macd - signal,
        "rsi": rsi / 100,
        "stoch": stoch,
        "bb_pos": bb,
        "vol_z": vol_z,
        "trend_regime": regime,
        "cp_flag": cpd,
        "sharpe_63": sharpe_roll,
        "skew_126": skew_roll,
        "kurt_126": kurt_roll,
    }
    feat_dict.update(mom)
    feat_dict.update(vol_windows)
    feat_dict.update({f"macro_{k}": v for k, v in macro.items()})
    feat_dict.update(season)
    feats = pd.DataFrame(feat_dict).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    z_feats = feats.apply(lambda col: _zscore(col, 128)).fillna(0.0)
    vol_filled = vol.bfill().ffill()
    return z_feats, vol_filled


@dataclass
class FeatureTensor:
    X: np.ndarray
    y: np.ndarray
    future_ret: np.ndarray
    sigma: np.ndarray
    future_sigma: np.ndarray
    close: np.ndarray
    dates: pd.DatetimeIndex


def make_feature_tensor(panel: pd.DataFrame, symbol: str = "SPY", window: int = 64) -> FeatureTensor:
    feats, sigma_hist = build_feature_frame(panel, symbol.lower())
    values = feats.values.astype(np.float32)
    view = sliding_window_view(values, window_shape=window, axis=0)
    X = view[:-1].copy()
    close = panel[f"{symbol.lower()}_close"].values
    returns = pd.Series(close, index=panel.index).pct_change().fillna(0.0)
    y = (returns.shift(-1) > 0).astype(np.float32).values[window - 1 : -1]
    sigma = sigma_hist.shift(1).bfill().ffill().values[window - 1 : -1]
    future_ret = returns.shift(-1).fillna(0.0).values[window - 1 : -1]
    future_sigma = returns.rolling(20).std(ddof=0).shift(-1).ffill().values[
        window - 1 : -1
    ]
    dates = feats.index[window - 1 : -1]
    closes = close[window - 1 : -1]
    return FeatureTensor(
        X=X,
        y=y,
        future_ret=future_ret,
        sigma=sigma,
        future_sigma=future_sigma,
        close=closes,
        dates=dates,
    )
