import pytest
import pandas as pd
import numpy as np
from backtest import VectorizedBacktester

def test_backtest_accounting():
    # Simple scenario:
    # t0: Price 100. Weight 0.
    # t1: Price 110. Weight 1. (Buy at t1 close? No, weight[t] is for t->t+1)
    # Let's clarify backtester logic again.
    # weights[t] is position held from t to t+1.
    # returns[t+1] is price[t+1]/price[t] - 1.
    # Strategy return[t+1] = weights[t] * returns[t+1].
    
    prices = pd.Series([100, 110, 121], index=pd.date_range("2020-01-01", periods=3))
    # Returns: t1: 0.1, t2: 0.1
    
    weights = pd.Series([1.0, 1.0, 0.0], index=prices.index)
    # w[0]=1.0 -> hold from t0 to t1. Return at t1 = 1.0 * 0.1 = 0.1.
    # w[1]=1.0 -> hold from t1 to t2. Return at t2 = 1.0 * 0.1 = 0.1.
    
    bt = VectorizedBacktester(initial_capital=100.0, transaction_cost_bps=0.0)
    results = bt.run(prices, weights)
    
    equity = results['equity_curve']
    # t0: 100 (initial)
    # t1: 100 * (1 + 0.1) = 110
    # t2: 110 * (1 + 0.1) = 121
    
    assert np.isclose(equity.iloc[1], 110.0)
    assert np.isclose(equity.iloc[2], 121.0)

def test_transaction_costs():
    prices = pd.Series([100, 100], index=pd.date_range("2020-01-01", periods=2))
    weights = pd.Series([1.0, 1.0], index=prices.index)
    # Turnover at t0: |1.0 - 0| = 1.0. Cost = 1.0 * 100bps = 0.01 (if 100bps)
    
    bt = VectorizedBacktester(initial_capital=100.0, transaction_cost_bps=10000.0) # 100% cost
    results = bt.run(prices, weights)
    
    # Cost at t0 reduces return at t0?
    # Backtester implementation:
    # turnover = weights.diff().abs()
    # turnover[0] = abs(weights[0])
    # costs = turnover * cost_rate
    # net_returns = strategy_returns - costs
    
    # strategy_returns[0] is NaN or 0 because returns[0] is NaN (pct_change).
    # returns[0] is NaN. strategy_returns[0] is NaN.
    # net_returns[0] = NaN - cost.
    # equity_curve = cumprod(1 + net_returns).
    
    # If net_returns[0] is negative, equity drops immediately?
    # Usually backtests start with flat equity.
    # Let's check the implementation details.
    # returns = prices.pct_change().fillna(0) -> returns[0] = 0.
    # aligned_weights = weights.shift(1).fillna(0) -> aligned_weights[0] = 0.
    # strategy_returns[0] = 0 * 0 = 0.
    # turnover[0] = 1.0.
    # cost[0] = 1.0.
    # net_return[0] = 0 - 1.0 = -1.0.
    # equity[0] = 100 * (1 - 1.0) = 0.
    
    # This implies we pay cost at t0 to enter the position.
    # This seems correct for a "start of day" or "end of day" trade logic.
    
    assert results['equity_curve'].iloc[0] < 100.0
