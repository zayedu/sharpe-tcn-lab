import pytest
import torch
import numpy as np
import random
from model_tcn import TCN, train_model
from features import FeatureEngineer
import pandas as pd

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test_determinism():
    # Run pipeline twice and compare results
    
    # Mock data
    dates = pd.date_range(start="2020-01-01", periods=100)
    df = pd.DataFrame({
        'Open': np.random.rand(100) * 100,
        'High': np.random.rand(100) * 110,
        'Low': np.random.rand(100) * 90,
        'Close': np.random.rand(100) * 100,
        'Volume': np.random.rand(100) * 1000000
    }, index=dates)
    
    # Run 1
    set_seed(42)
    fe = FeatureEngineer(window_size=10)
    X1, y1, _ = fe.create_dataset(df)
    
    model1 = TCN(num_inputs=12, num_channels=[16], kernel_size=3)
    # Init weights is random, so we need to ensure set_seed works for init
    
    # Forward pass
    with torch.no_grad():
        out1 = model1(torch.from_numpy(X1).float().permute(0, 2, 1))
        
    # Run 2
    set_seed(42)
    fe = FeatureEngineer(window_size=10)
    X2, y2, _ = fe.create_dataset(df)
    
    model2 = TCN(num_inputs=12, num_channels=[16], kernel_size=3)
    
    with torch.no_grad():
        out2 = model2(torch.from_numpy(X2).float().permute(0, 2, 1))
        
    assert np.allclose(X1, X2)
    assert torch.allclose(out1, out2)
