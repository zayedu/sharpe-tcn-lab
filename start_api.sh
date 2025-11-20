#!/bin/bash

# Start the Sharpe TCN Lab API with correct PYTHONPATH
cd "$(dirname "$0")"
PYTHONPATH=. uvicorn quant_tcn_riskmap.api:app --reload
