import numpy as np
import pandas as pd
import torch

from features import make_feature_tensor
from model_tcn import HybridSignalNet


def _toy_panel(rows: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=rows, freq="D")
    base = np.linspace(100, 120, rows)
    volume = np.linspace(1e6, 1.5e6, rows)
    panel = pd.DataFrame(
        {
            "spy_open": base * 0.999,
            "spy_high": base * 1.001,
            "spy_low": base * 0.998,
            "spy_close": base,
            "spy_adj close": base,
            "spy_volume": volume,
            "^vix_close": np.linspace(15, 25, rows),
            "^tnx_close": np.linspace(1.5, 3.0, rows),
        },
        index=idx,
    )
    panel.columns = [c.lower() for c in panel.columns]
    return panel


def test_feature_tensor_alignment():
    panel = _toy_panel()
    tensor = make_feature_tensor(panel, symbol="SPY", window=32)
    close = panel["spy_close"].values
    returns = pd.Series(close, index=panel.index).pct_change().fillna(0.0)
    expected = returns.shift(-1).fillna(0.0).values[31:-1]
    assert np.allclose(tensor.future_ret, expected, atol=1e-12)


def test_model_initialization_deterministic():
    n_features = make_feature_tensor(_toy_panel(), "SPY", 32).X.shape[-1]
    torch.manual_seed(7)
    model_a = HybridSignalNet(in_features=n_features)
    torch.manual_seed(7)
    model_b = HybridSignalNet(in_features=n_features)
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(pa, pb)
