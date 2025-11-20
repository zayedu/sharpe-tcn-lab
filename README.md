# Sharpe-Optimized TCN Trading System

A production-grade deep learning trading system using Temporal Convolutional Networks (TCN) with custom Sharpe-optimized loss functions for quantitative trading.

## 🎯 Key Results

**QQQ (Nasdaq 100) Performance:**
- **Best Year Sharpe: 2.57** (2023)
- Consistent strong performance: 2019 (2.14), 2020 (1.88), 2021 (1.93)
- Aggregated Return (2018-2024): **10.15%**
- Validation Accuracy: 57-62%

## 🏗️ Architecture

### Core Components
1. **Custom Sharpe Loss**: Direct optimization of risk-adjusted returns with volatility scaling
2. **Hybrid Training**: Two-stage approach (BCE pre-training → Sharpe fine-tuning)
3. **TCN Model**: Shallow architecture with attention mechanism for temporal dependencies
4. **Walk-Forward Validation**: Rigorous out-of-sample testing framework
5. **Risk Map**: Volatility-scaled position sizing with deadband logic

### Technical Features
- Global normalization to preserve regime information
- Early stopping to prevent overfitting
- Transaction cost modeling (3 bps)
- FastAPI backtesting service with SQLite persistence
- Deterministic execution with seed control

## 📊 Performance Summary

| Asset | Sharpe (2023) | Sharpe (Agg) | Return (Agg) | Max DD |
|-------|---------------|--------------|--------------|--------|
| QQQ   | 2.57          | 0.20         | 10.15%       | -26.11% |
| SPY   | 1.83          | 0.03         | -0.06%       | -14.34% |

## 🚀 Quick Start

```bash
# Install dependencies
pip install pandas numpy torch fastapi pydantic yfinance uvicorn

# Run end-to-end demo
python3 quant_tcn_riskmap/demo.py

# Start API server
uvicorn quant_tcn_riskmap.api:app --reload

# Run tests
pytest quant_tcn_riskmap/tests/
```

## 📁 Project Structure

```
quant_tcn_riskmap/
├── data_loader.py      # Yahoo Finance data download
├── features.py         # Technical indicators (RSI, MACD, ADX, ATR, etc.)
├── model_tcn.py        # TCN architecture + Sharpe/Hybrid Loss
├── riskmap.py          # Volatility-scaled position sizing
├── backtest.py         # Vectorized backtester
├── api.py              # FastAPI service
├── demo.py             # Walk-forward validation demo
└── tests/              # Unit tests
```

## 🔬 Key Innovations

1. **Sharpe-Optimized Loss Function**
   - Directly optimizes strategy Sharpe Ratio during training
   - Incorporates volatility scaling and transaction costs
   - Hybrid approach combines BCE stability with Sharpe optimization

2. **Walk-Forward Validation**
   - 7-year rolling window (2018-2024)
   - Separate model training per year
   - Prevents look-ahead bias

3. **Global Normalization**
   - Preserves absolute feature magnitudes
   - Maintains volatility regime information
   - Improves generalization vs rolling z-score

## 📈 Use Cases

- **Quantitative Research**: Framework for testing ML-based trading strategies
- **Academic Projects**: Demonstrates Sharpe optimization in deep learning
- **Portfolio Management**: Volatility-scaled position sizing methodology
- **Backtesting Infrastructure**: Production-ready API with persistence

## ⚠️ Limitations

- Performance is regime-dependent (excellent in bull markets, struggles in bear markets)
- Aggregated Sharpe below institutional targets (0.20 vs target 1.8+)
- Requires regime detection or ensemble methods for all-weather performance

## 🛠️ Future Enhancements

- [ ] Regime detection (bull/bear/sideways classification)
- [ ] Multi-asset ensemble strategies
- [ ] Alternative data integration
- [ ] Real-time inference pipeline
- [ ] Portfolio optimization across multiple signals

## 📝 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

Built with PyTorch, FastAPI, and yfinance. Inspired by modern quantitative trading research.

---

**Note**: This is a research project. Past performance does not guarantee future results. Always validate strategies thoroughly before live deployment.
