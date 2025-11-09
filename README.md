# Quant TCN Risk-Map

Deterministic research framework that reproduces a volatility-scaled trading stack inspired by TCN + risk-map designs. The system covers data ingestion, feature engineering, PyTorch modeling, vectorized backtesting, and a FastAPI service with persistence.

## Quickstart

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt  # or pip install pandas numpy torch fastapi uvicorn yfinance pytest
python demo.py
```

`demo.py` downloads SPY+macro series back to 2004, builds a 96×F multi-scale feature tensor (momentum, volatility, macro spreads, changepoint proxies), trains a three-seed ensemble of the hybrid TCN+attention model via `trainer.py` (Sharpe-aware multi-task loss with direct position outputs), performs walk-forward risk-map/probability tuning via `tuning.py`, benchmarks inference latency, exports `artifacts/tcn.pt`, and backtests 2018‑2024 under 3 bps costs. Output covers Sharpe/drawdown/turnover, latency (p50/p95 + throughput), and the tuned adaptive risk parameters. Re-running the script yields identical equity/PnL (<1e-9 tolerance).

## Project Structure

```
quant_tcn_riskmap/
├─ data_loader.py      # deterministic Yahoo Finance ingestion and caching
├─ features.py         # multi-scale technical + macro features (96×F)
├─ model_tcn.py        # hybrid TCN + attention multi-task forecaster
├─ riskmap.py          # adaptive deadband + volatility scaling utilities
├─ backtest.py         # position simulator, analytics, and invariants
├─ trainer.py          # Sharpe-aware PyTorch trainer (multi-task + direct positions)
├─ training_utils.py   # loader / Sharpe helpers shared by the trainer
├─ research_utils.py   # ensemble/benchmark utilities for research scripts
├─ service_utils.py    # FastAPI helper utilities (model cache, DB, predict)
├─ tuning.py           # validation-driven risk-map + probability scaling search
├─ api.py              # FastAPI service with SQLite logging
├─ demo.py             # end-to-end training/backtest/benchmark/export
└─ tests/              # pytest suite for risk-map, backtester, and determinism
```

Each module is importable in isolation (≤150 LOC per file, ≤20 LOC per function where practical) to mirror production code layering.

## API Service

Start the service after running `demo.py` (to produce the TorchScript artifact):

```bash
uvicorn api:app --reload
```

- `POST /runs`: body includes symbol/start/end/test_start/tau/sigma_target/etc. The endpoint runs a deterministic backtest, logs the config+metrics in SQLite (`runs.db`), and returns the metrics plus run_id.
- `GET /runs/{id}`: fetch stored configs + metrics.
- `POST /predict`: submit a single 64×12 feature window and current sigma estimate to receive `{p_hat, sigma_hat, w}` using the deployed risk-map.

## Testing & Benchmarks

```bash
pytest -q
```

Tests cover:

- risk map monotonicity/bounds and deadband neutrality
- backtester invariants (zero-signal behavior, PnL vs. equity reconciliation)
- deterministic feature alignment plus reproducible TCN initialization

The demo emits inference latency percentiles and throughput, satisfying the micro-benchmark requirement.

## Notes

- Python 3.11 only; dependencies limited to pandas, numpy, torch, fastapi, pydantic, yfinance.
- Transaction costs modeled in basis points with turnover penalties applied deterministically.
- Positions respect the equality `equity = cash + inventory × price` every step (hard assertion in the simulator).
- All randomness is seeded via `demo.set_seed`, and TorchScript export enables low-latency inference in the API or external services.
