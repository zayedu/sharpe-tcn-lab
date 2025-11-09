import numpy as np

from backtest import run_backtest
from riskmap import RiskMapConfig


def test_backtest_no_signal_no_costs():
    n = 32
    probs = np.full(n, 0.5)
    sigma = np.full(n, 0.02)
    price = np.linspace(100, 101, n)
    future_returns = np.zeros(n)
    dates = np.arange(n)
    cfg = RiskMapConfig(tau=0.2, sigma_target=0.02, wmax=1.0)
    result = run_backtest(
        probs=probs,
        sigma_hat=sigma,
        price=price,
        future_returns=future_returns,
        dates=dates,
        cfg=cfg,
        cost_bps=0.0,
    )
    assert np.allclose(result.equity, 1.0)


def test_pnl_matches_equity_changes():
    n = 40
    probs = np.full(n, 0.7)
    sigma = np.full(n, 0.02)
    price = np.full(n, 100.0)
    future_returns = np.full(n, 0.001)
    dates = np.arange(n)
    cfg = RiskMapConfig(tau=0.0, sigma_target=0.02, wmax=1.0)
    result = run_backtest(
        probs=probs,
        sigma_hat=sigma,
        price=price,
        future_returns=future_returns,
        dates=dates,
        cfg=cfg,
        cost_bps=0.0,
    )
    assert np.isclose(result.equity[-1] - 1.0, result.pnl.sum())
    assert result.metrics["sharpe"] > 0


def test_stop_loss_flattening():
    n = 10
    probs = np.full(n, 0.9)
    sigma = np.full(n, 0.02)
    price = np.linspace(100, 90, n)
    future_returns = np.full(n, -0.01)
    dates = np.arange(n)
    cfg = RiskMapConfig(stop_loss=0.02, tau=0.0, sigma_target=0.02)
    result = run_backtest(
        probs=probs,
        sigma_hat=sigma,
        price=price,
        future_returns=future_returns,
        dates=dates,
        cfg=cfg,
        cost_bps=0.0,
    )
    assert np.all(result.weights[2:] <= result.weights[1])
