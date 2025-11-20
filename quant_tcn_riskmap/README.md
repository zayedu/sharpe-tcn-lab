# Quant TCN Riskmap System

A production-grade quantitative trading system using a Temporal Convolutional Network (TCN) and volatility-scaled risk map.

## Structure
- `data_loader.py`: Yahoo Finance data ingestion.
- `features.py`: Feature engineering (RSI, MACD, Volatility, etc.).
- `model_tcn.py`: PyTorch TCN implementation.
- `riskmap.py`: Volatility scaling and deadband logic.
- `backtest.py`: Vectorized backtester.
- `api.py`: FastAPI service.
- `demo.py`: End-to-end demonstration script.
- `tests/`: Unit tests.

## Usage

### Install Dependencies
```bash
pip install pandas numpy torch fastapi uvicorn yfinance pydantic
```

### Run Demo
```bash
python demo.py
```

### Run Tests
```bash
pytest tests/
```

### Start API
```bash
uvicorn api:app --reload
```
