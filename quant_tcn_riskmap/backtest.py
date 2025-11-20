import numpy as np
import pandas as pd
from typing import Dict, Any

class VectorizedBacktester:
    def __init__(self, initial_capital: float = 10000.0, transaction_cost_bps: float = 3.0):
        self.initial_capital = initial_capital
        self.tc_bps = transaction_cost_bps
        
    def run(self, prices: pd.Series, weights: pd.Series) -> Dict[str, Any]:
        """
        Runs a vectorized backtest.
        
        Args:
            prices: Series of asset prices (e.g., Close).
            weights: Series of target position weights (from risk map).
                     weights[t] is the position held from t to t+1.
                     
        Returns:
            Dictionary containing metrics and equity curve.
        """
        # Align weights and prices
        # weights[t] determines return at t+1 (price[t+1] / price[t] - 1)
        
        # Calculate returns
        returns = prices.pct_change().fillna(0)
        
        # Shift weights to match returns
        # If weights[t] is decided at time t (using info up to t), it captures return from t to t+1.
        # In pandas, returns[t+1] is (price[t+1]/price[t] - 1).
        # So we want position[t+1] * returns[t+1].
        # Wait, usually weights are calculated at t, and we enter position at t (Close) or t+1 (Open).
        # Let's assume we trade at Close of t. So we hold position from t to t+1.
        # So return at t+1 is driven by weight at t.
        
        # weights series index: t. Value: w_t.
        # returns series index: t. Value: r_t = (p_t / p_{t-1}) - 1.
        
        # Strategy return at t: w_{t-1} * r_t.
        
        # We need to align them.
        aligned_weights = weights.shift(1).fillna(0)
        
        # Gross Strategy Returns
        strategy_returns = aligned_weights * returns
        
        # Transaction Costs
        # Turnover = |w_t - w_{t-1}|
        # We trade at t to go from w_{t-1} to w_t.
        # Cost is incurred at t.
        # But we are calculating daily returns.
        # Let's deduct cost from the return of that day (or the previous day?).
        # If we trade at t, the cost reduces our capital at t.
        # So it affects the PnL at t.
        
        turnover = weights.diff().abs().fillna(0)
        # Initial trade
        turnover.iloc[0] = abs(weights.iloc[0])
        
        costs = turnover * (self.tc_bps / 10000.0)
        
        # Net Strategy Returns
        net_returns = strategy_returns - costs
        
        # Equity Curve
        equity_curve = self.initial_capital * (1 + net_returns).cumprod()
        
        # Metrics
        total_return = equity_curve.iloc[-1] / self.initial_capital - 1
        annualized_return = net_returns.mean() * 252
        annualized_vol = net_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
        
        downside_returns = net_returns[net_returns < 0]
        sortino_ratio = annualized_return / (downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 else 0
        
        # Max Drawdown
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        
        avg_turnover = turnover.mean()
        
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_vol": annualized_vol,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "avg_turnover": avg_turnover,
            "equity_curve": equity_curve,
            "drawdown_curve": drawdown
        }

if __name__ == "__main__":
    # Simple test
    prices = pd.Series([100, 101, 102, 101, 103], index=pd.date_range("2020-01-01", periods=5))
    weights = pd.Series([0.5, 0.5, 1.0, 0.0, 0.0], index=prices.index)
    bt = VectorizedBacktester()
    results = bt.run(prices, weights)
    print(results)
