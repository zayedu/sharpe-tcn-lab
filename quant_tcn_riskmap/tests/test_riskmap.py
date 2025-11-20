import pytest
import numpy as np
from riskmap import calculate_position_size, vectorized_risk_map

def test_risk_map_bounds():
    # Test bounds
    w = calculate_position_size(0.9, 0.1, 0.15, -1.0, 1.0, 0.0)
    assert -1.0 <= w <= 1.0
    
    w = calculate_position_size(0.1, 0.1, 0.15, -1.0, 1.0, 0.0)
    assert -1.0 <= w <= 1.0

def test_risk_map_deadband():
    # p_hat = 0.55, tau = 0.2. 2*p - 1 = 0.1. |0.1| < 0.2 -> w should be 0.
    w = calculate_position_size(0.55, 0.1, 0.15, -1.0, 1.0, 0.2)
    assert w == 0.0
    
    # p_hat = 0.65, tau = 0.2. 2*p - 1 = 0.3. |0.3| > 0.2 -> w should be non-zero.
    w = calculate_position_size(0.65, 0.1, 0.15, -1.0, 1.0, 0.2)
    assert w > 0.0

def test_risk_map_vol_scaling():
    # Higher vol -> Lower weight
    w1 = calculate_position_size(0.8, 0.1, 0.15, -1.0, 1.0, 0.0)
    w2 = calculate_position_size(0.8, 0.2, 0.15, -1.0, 1.0, 0.0)
    assert abs(w1) > abs(w2)

def test_vectorized_risk_map():
    p_hats = np.array([0.55, 0.65, 0.8])
    sigma_hats = np.array([0.1, 0.1, 0.2])
    
    w = vectorized_risk_map(p_hats, sigma_hats, 0.15, -1.0, 1.0, 0.2)
    
    assert w[0] == 0.0 # Deadband
    assert w[1] > 0.0
    assert w[2] > 0.0
    assert len(w) == 3
