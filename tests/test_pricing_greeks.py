"""
Unit Tests for Vectorized Black-76 Pricing, Greeks, and IV Solvers
"""

import pytest
import numpy as np
import math
from options_bt.pricing import Black76, ImpliedVolatilitySolver, VolatilitySurfaceMetrics


def test_put_call_parity():
    """Verify that Black-76 strictly adheres to Put-Call Parity: C - P = exp(-rT)*(F - K)"""
    F = 24500.0
    K = 24500.0
    T = 7.0 / 365.0
    r = 0.065
    sigma = 0.135

    call_price = Black76.price(F, K, T, r, sigma, option_type="CE")
    put_price = Black76.price(F, K, T, r, sigma, option_type="PE")

    df = math.exp(-r * T)
    parity_diff = (call_price - put_price) - df * (F - K)
    assert abs(parity_diff) < 1e-6, f"Put-Call parity broken: diff = {parity_diff}"


def test_analytical_greeks_signs():
    """Verify foundational financial properties and signs of options Greeks."""
    F = 24500.0
    K = 24500.0
    T = 3.0 / 365.0
    r = 0.065
    sigma = 0.14

    call_greeks = Black76.greeks(F, K, T, r, sigma, option_type="CE")
    put_greeks = Black76.greeks(F, K, T, r, sigma, option_type="PE")

    # Delta properties
    assert 0.45 <= call_greeks.delta <= 0.55, "ATM Call delta must be ~0.50"
    assert -0.55 <= put_greeks.delta <= -0.45, "ATM Put delta must be ~ -0.50"
    assert abs((call_greeks.delta - put_greeks.delta) - math.exp(-r * T)) < 1e-4

    # Gamma properties
    assert call_greeks.gamma > 0.0, "Gamma must be positive for long option"
    assert abs(call_greeks.gamma - put_greeks.gamma) < 1e-8, "Call and Put gammas must be identical"

    # Vega properties
    assert call_greeks.vega > 0.0, "Vega must be positive"
    assert abs(call_greeks.vega - put_greeks.vega) < 1e-8, "Call and Put vegas must be identical"

    # Theta properties
    assert call_greeks.theta < 0.0, "Theta must be negative for long option"
    assert put_greeks.theta < 0.0, "Theta must be negative for long option"

    # Higher-order Greeks properties
    assert hasattr(call_greeks, "charm"), "GreeksResult must contain charm"
    assert hasattr(call_greeks, "color"), "GreeksResult must contain color"
    assert hasattr(call_greeks, "speed"), "GreeksResult must contain speed"
    assert call_greeks.vanna != 0.0, "Vanna must be non-zero"
    assert abs(call_greeks.volga) < 1e-3, "ATM Volga must be near zero"

    # OTM Volga must be strictly positive
    otm_greeks = Black76.greeks(F, 24650.0, T, r, sigma, option_type="CE")
    assert otm_greeks.volga > 0.0, "OTM Volga must be positive"


def test_greeks_vectorized():
    """Verify vectorized calculation across multiple strikes."""
    F = 24500.0
    strikes = np.array([24300.0, 24400.0, 24500.0, 24600.0, 24700.0])
    sigmas = np.full(5, 0.14)
    T = 4.0 / 365.0
    r = 0.065

    res = Black76.greeks_vector(F, strikes, T, r, sigmas, option_types="CE")
    assert len(res["price"]) == 5
    assert len(res["delta"]) == 5
    assert res["delta"][0] > res["delta"][2] > res["delta"][4], "ITM delta > ATM delta > OTM delta"
    assert res["gamma"][2] == np.max(res["gamma"]), "ATM option must have peak gamma"


def test_iv_solver_precision():
    """Verify that Newton-Raphson / Brent solver accurately recovers known volatility."""
    F = 24550.0
    strikes = [24300.0, 24500.0, 24700.0]
    T = 5.0 / 365.0
    r = 0.065
    target_sigmas = [0.155, 0.138, 0.142]

    for K, true_sigma in zip(strikes, target_sigmas):
        market_price = Black76.price(F, K, T, r, true_sigma, option_type="CE")
        recovered_iv = ImpliedVolatilitySolver.solve_single(
            market_price=market_price,
            F=F,
            K=K,
            T=T,
            r=r,
            option_type="CE"
        )
        assert abs(recovered_iv - true_sigma) < 1e-4, f"IV solver failed for K={K}: got {recovered_iv}, expected {true_sigma}"


def test_advanced_realized_volatilities():
    """Verify Parkinson, Garman-Klass, and Yang-Zhang volatility estimators."""
    np.random.seed(42)
    n_bars = 375
    base = 24500.0
    close = base + np.cumsum(np.random.normal(0, 2.5, n_bars))
    open_px = close + np.random.normal(0, 1.0, n_bars)
    high = np.maximum(open_px, close) + np.random.exponential(2.0, n_bars)
    low = np.minimum(open_px, close) - np.random.exponential(2.0, n_bars)

    park = VolatilitySurfaceMetrics.parkinson_realized_vol(high, low)
    gk = VolatilitySurfaceMetrics.garman_klass_realized_vol(open_px, high, low, close)
    yz = VolatilitySurfaceMetrics.yang_zhang_realized_vol(open_px, high, low, close)

    assert 0.01 < park < 0.50, f"Unreasonable Parkinson vol: {park}"
    assert 0.01 < gk < 0.50, f"Unreasonable Garman-Klass vol: {gk}"
    assert 0.01 < yz < 0.50, f"Unreasonable Yang-Zhang vol: {yz}"

    # Term structure slope (Contango check)
    slope = VolatilitySurfaceMetrics.term_structure_slope(near_iv=0.13, far_iv=0.15, days_near=7, days_far=30)
    assert slope > 0.0, "Near 13% vs Far 15% must show positive Contango slope"
