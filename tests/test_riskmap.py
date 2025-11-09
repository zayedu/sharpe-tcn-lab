import numpy as np

from riskmap import RiskMapConfig, deadband, volatility_scaled_weights


def test_deadband_respects_neutral_zone():
    signal = np.array([-0.01, 0.0, 0.01])
    tau = np.full_like(signal, 0.05)
    output = deadband(signal, tau)
    assert np.allclose(output, 0.0)


def test_weights_monotonic_and_bounded():
    probs = np.linspace(0.2, 0.8, 16)
    sigma = np.full_like(probs, 0.02)
    cfg = RiskMapConfig(tau=0.0, sigma_target=0.02, wmax=0.5, blend=0.0)
    weights = volatility_scaled_weights(probs, sigma, cfg)
    assert np.all(np.diff(weights) >= -1e-9)
    assert weights.min() >= -0.5 - 1e-9
    assert weights.max() <= 0.5 + 1e-9


def test_probability_deadband_zeroes_signal():
    probs = np.array([0.49, 0.5, 0.51, 0.7])
    sigma = np.full_like(probs, 0.02)
    cfg = RiskMapConfig(prob_threshold=0.02)
    weights = volatility_scaled_weights(probs, sigma, cfg)
    assert np.allclose(weights[:3], 0.0)
    assert weights[-1] != 0.0


def test_confidence_gamma_downweights():
    probs = np.array([0.52, 0.9])
    sigma = np.full_like(probs, 0.02)
    cfg = RiskMapConfig(confidence_gamma=2.0, tau=0.0, sigma_target=0.02)
    weights = volatility_scaled_weights(probs, sigma, cfg)
    assert weights[0] < weights[1]
