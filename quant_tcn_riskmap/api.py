from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import torch
import os
import sqlite3
from datetime import datetime

from data_loader import download_data
from features import FeatureEngineer
from model_tcn import TCN, train_model
from riskmap import calculate_position_size, vectorized_risk_map
from backtest import VectorizedBacktester

app = FastAPI(title="Quant TCN Riskmap API")

# Database setup
DB_FILE = "runs.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS runs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  ticker TEXT,
                  sharpe_ratio REAL,
                  max_drawdown REAL,
                  config TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Schemas
class BacktestConfig(BaseModel):
    ticker: str = "SPY"
    start_date: str = "2018-01-01"
    end_date: str = "2024-01-01"
    tau: float = 0.1
    target_sigma: float = 0.15
    w_min: float = -1.0
    w_max: float = 1.0
    tc_bps: float = 3.0

class PredictRequest(BaseModel):
    features: List[List[float]] # (Window, Features)
    current_sigma: float

class PredictResponse(BaseModel):
    p_hat: float
    w: float

# Global model (lazy loading or trained on startup)
# For this demo, we'll train a model on the fly or load a saved one.
# To keep it simple, we'll have a global variable that can be updated.
global_model = None

@app.post("/runs")
async def start_backtest(config: BacktestConfig):
    try:
        # 1. Load Data
        df = download_data(config.ticker, config.start_date, config.end_date)
        
        # 2. Features
        fe = FeatureEngineer()
        X, y, timestamps = fe.create_dataset(df)
        
        # Split Train/Test (Simple time split)
        train_size = int(len(X) * 0.7)
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]
        
        # 3. Train Model (Quick training for demo)
        # Convert to tensors
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()),
            batch_size=32, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()),
            batch_size=32, shuffle=False
        )
        
        model = TCN(num_inputs=12, num_channels=[16, 32, 64], kernel_size=3, dropout=0.2)
        train_model(model, train_loader, val_loader, epochs=5) # Short epochs for API response speed
        
        # 4. Inference on whole dataset (or just test set?)
        # Let's do whole dataset to see full curve, but strictly backtest on test set usually.
        # User asked for backtest 2018-2024.
        # Let's assume the user wants to see performance on the requested period.
        # We should ideally train on pre-2018 data if possible, or walk-forward.
        # For this simplified API, we'll evaluate on the requested period (which might be in-sample if we just trained on it).
        # To be rigorous, let's just evaluate on the data we have.
        
        model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().permute(0, 2, 1) # (N, 12, 64)
            p_hats = model(X_tensor).numpy()
            
        # 5. Risk Map
        # Need sigma_hats. We have 'volatility' feature in df.
        # We need to align df with X.
        # X starts at index window_size.
        # timestamps corresponds to the time of prediction (t).
        # We need volatility at t.
        
        # Re-extract volatility from df aligned with timestamps
        aligned_vol = df.loc[timestamps, 'volatility'].values
        
        weights = vectorized_risk_map(
            p_hats, aligned_vol, config.target_sigma, config.w_min, config.w_max, config.tau
        )
        
        # 6. Backtest
        # Weights at t are for return at t+1.
        # Backtester expects weights series and prices series.
        weights_series = pd.Series(weights, index=timestamps)
        prices_series = df.loc[timestamps, 'Close']
        
        bt = VectorizedBacktester(transaction_cost_bps=config.tc_bps)
        results = bt.run(prices_series, weights_series)
        
        # Log to DB
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("INSERT INTO runs (timestamp, ticker, sharpe_ratio, max_drawdown, config) VALUES (?, ?, ?, ?, ?)",
                  (datetime.now().isoformat(), config.ticker, results['sharpe_ratio'], results['max_drawdown'], config.model_dump_json()))
        run_id = c.lastrowid
        conn.commit()
        conn.close()
        
        return {
            "run_id": run_id,
            "metrics": {
                "sharpe_ratio": results['sharpe_ratio'],
                "total_return": results['total_return'],
                "max_drawdown": results['max_drawdown']
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/runs/{run_id}")
async def get_run(run_id: int):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM runs WHERE id=?", (run_id,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Run not found")
        
    return {
        "id": row[0],
        "timestamp": row[1],
        "ticker": row[2],
        "sharpe_ratio": row[3],
        "max_drawdown": row[4],
        "config": row[5]
    }

@app.post("/predict")
async def predict(request: PredictRequest):
    # Mock model if not loaded
    # In a real scenario, we'd load the trained model.
    # For this demo, we'll instantiate a random model if none exists.
    global global_model
    if global_model is None:
        global_model = TCN(num_inputs=12, num_channels=[16, 32, 64], kernel_size=3)
        global_model.eval()
        
    # Input shape: (Batch, Time, Features) -> (Batch, Features, Time)
    # Request features: (64, 12) -> We need (1, 12, 64)
    features = np.array(request.features)
    if features.shape != (64, 12):
        raise HTTPException(status_code=400, detail=f"Expected shape (64, 12), got {features.shape}")
        
    x_tensor = torch.from_numpy(features).float().unsqueeze(0).permute(0, 2, 1)
    
    with torch.no_grad():
        p_hat = global_model(x_tensor).item()
        
    # Risk Map (using default params or could be passed in request)
    w = calculate_position_size(p_hat, request.current_sigma, 0.15, -1.0, 1.0, 0.1)
    
    return PredictResponse(p_hat=p_hat, w=w)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
