import numpy as np

def calculate_position_size(p_hat: float, sigma_hat: float, target_sigma: float, w_min: float, w_max: float, tau: float) -> float:
    """
    Calculates the target position size based on the risk map formula.
    
    w_{t+1} = clip( (target_sigma / sigma_hat) * g_tau(2 * p_hat - 1), w_min, w_max )
    
    where g_tau(x) is a deadband function:
    g_tau(x) = x if |x| > tau else 0
    
    Args:
        p_hat: Predicted probability of positive return (0 to 1).
        sigma_hat: Predicted/Current volatility (annualized or daily, must match target_sigma).
        target_sigma: Target volatility.
        w_min: Minimum position size (e.g., -1.0 for short).
        w_max: Maximum position size (e.g., 1.0 for long).
        tau: Deadband threshold (0 to 1).
        
    Returns:
        Target position size (float).
    """
    # Raw signal in [-1, 1]
    raw_signal = 2 * p_hat - 1
    
    # Deadband
    if abs(raw_signal) <= tau:
        signal = 0.0
    else:
        signal = raw_signal
        
    # Volatility scaling
    # Avoid division by zero
    if sigma_hat < 1e-6:
        sigma_hat = 1e-6
        
    vol_scalar = target_sigma / sigma_hat
    
    # Unclipped weight
    w_raw = vol_scalar * signal
    
    # Clipping
    w_final = np.clip(w_raw, w_min, w_max)
    
    return float(w_final)

def vectorized_risk_map(p_hats: np.ndarray, sigma_hats: np.ndarray, target_sigma: float, w_min: float, w_max: float, tau: float) -> np.ndarray:
    """
    Vectorized version of calculate_position_size.
    """
    raw_signals = 2 * p_hats - 1
    
    # Deadband
    signals = np.where(np.abs(raw_signals) <= tau, 0.0, raw_signals)
    
    # Volatility scaling
    sigma_hats = np.maximum(sigma_hats, 1e-6)
    vol_scalars = target_sigma / sigma_hats
    
    w_raw = vol_scalars * signals
    
    w_final = np.clip(w_raw, w_min, w_max)
    
    return w_final

if __name__ == "__main__":
    # Simple test
    w = calculate_position_size(0.8, 0.01, 0.15, -1, 1, 0.1)
    print(f"Position: {w}")
