import pandas as pd
import numpy as np
import torch
import random
import os
from data_loader import download_data
from features import FeatureEngineer
from model_tcn import TCN, train_model
from riskmap import vectorized_risk_map
from backtest import VectorizedBacktester
import time

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    print("Starting Quant TCN Riskmap Demo...")
    set_seed(42)
    
    # 1. Data
    try:
        df = download_data("SPY", "2010-01-01", "2024-01-01")
    except Exception as e:
        print(f"Failed to download data: {e}")
        return

def main():
    print("Starting Quant TCN Riskmap Demo (Phase 2: Sharpe Optimization)...")
    set_seed(42)
    
    # 1. Data
    try:
        df = download_data("QQQ", "2005-01-01", "2024-01-01")
    except Exception as e:
        print(f"Failed to download data: {e}")
        return

    # 2. Features
    print("Generating features...")
    fe = FeatureEngineer(window_size=64)
    features_df, _ = fe.create_features(df)
    X, y, returns, volatility, timestamps = fe.create_dataset(df)
    
    print(f"Total samples: {len(X)}")
    
    # 3. Walk-Forward Validation
    # Train on 5 years, Test on 1 year, Slide 1 year.
    # Start Test from 2018.
    
    test_start_year = 2018
    test_end_year = 2024
    
    all_preds = []
    all_timestamps = []
    
    # Hyperparams
    target_sigma = 0.15
    
    for year in range(test_start_year, test_end_year + 1):
        print(f"\n--- Walk-Forward: Testing Year {year} ---")
        
        # Define Train/Test masks
        # Train: [Start, Year-1]
        # Test: [Year]
        
        test_mask = (timestamps.year == year)
        train_mask = (timestamps.year < year) & (timestamps.year >= year - 10) # 10 year rolling window
        
        if not np.any(test_mask):
            continue
            
        X_train = X[train_mask]
        y_train = y[train_mask]
        ret_train = returns[train_mask]
        vol_train = volatility[train_mask]
        
        X_test = X[test_mask]
        y_test = y[test_mask]
        ret_test = returns[test_mask]
        vol_test = volatility[test_mask]
        ts_test = timestamps[test_mask]
        
        # Global Normalization
        # Normalize each feature using Train statistics
        mean = X_train.mean(axis=(0, 1), keepdims=True)
        std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
        
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
        
        # Create Loaders
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(X_train).float(), 
                torch.from_numpy(y_train).float(), # Real targets
                torch.from_numpy(ret_train).float(), 
                torch.from_numpy(vol_train).float()
            ),
            batch_size=256, shuffle=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(X_test).float(), 
                torch.from_numpy(y_test).float(), 
                torch.from_numpy(ret_test).float(), 
                torch.from_numpy(vol_test).float()
            ),
            batch_size=256, shuffle=False
        )
        
        # Train Model
        # Shallow model: [32, 32]
        model = TCN(num_inputs=14, num_channels=[32, 32], kernel_size=3, dropout=0.2)
        
        # Stage 1: BCE Pre-training
        print("Stage 1: BCE Pre-training...")
        train_model(model, train_loader, val_loader, epochs=10, lr=0.001, alpha=0.0)
        
        # Stage 2: Sharpe Fine-tuning
        print("Stage 2: Sharpe Fine-tuning...")
        train_model(model, train_loader, val_loader, epochs=20, lr=0.0001, alpha=1.0)
        
        # Inference
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.from_numpy(X_test).float().permute(0, 2, 1)
            # Output is probability [0, 1]
            probs = model(X_test_tensor).squeeze().numpy()
            # Convert to signal [-1, 1]
            signals = (probs - 0.5) * 2
            
        all_preds.append(signals)
        all_timestamps.append(ts_test)
        
    # Concatenate Results
    full_signals = np.concatenate(all_preds)
    full_timestamps = np.concatenate(all_timestamps)
    
    # 4. Backtest
    # Re-align volatility for the full period
    aligned_vol = features_df.loc[full_timestamps, 'volatility'].values
    
    # Apply Volatility Scaling
    # w = signal * (target / vol)
    # Clip leverage
    vol_scalars = np.clip(target_sigma / (aligned_vol + 1e-8), 0, 4.0)
    weights = full_signals * vol_scalars
    
    # Clip final weights
    weights = np.clip(weights, -1.5, 1.5)
    
    weights_series = pd.Series(weights, index=full_timestamps)
    prices_series = df.loc[full_timestamps, 'Close']
    
    bt = VectorizedBacktester(transaction_cost_bps=3.0)
    results = bt.run(prices_series, weights_series)
    
    print("\n=== Final Walk-Forward Results (2018-2024) ===")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Total Return: {results['total_return']*100:.2f}%")
    print(f"Max Drawdown: {results['max_drawdown']*100:.2f}%")
    print(f"Avg Turnover: {results['avg_turnover']:.4f}")
    
    if results['sharpe_ratio'] > 1.8:
        print("\nSUCCESS: Sharpe Ratio > 1.8")
    else:
        print("\nNote: Sharpe Ratio still needs tuning.")

if __name__ == "__main__":
    main()
