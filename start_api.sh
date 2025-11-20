#!/bin/bash

# Start the Sharpe TCN Lab API with correct PYTHONPATH
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python3 -m uvicorn quant_tcn_riskmap.api:app --reload --host 0.0.0.0 --port 8000
