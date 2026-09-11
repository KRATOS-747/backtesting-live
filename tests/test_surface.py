"""
Unit Tests for 3D Volatility Surface and Smile Engine (options_bt/surface.py)
"""

import os
import glob
import pytest
import numpy as np
import pandas as pd
from options_bt.surface import VolatilitySurfaceEngine, SmileSlice, VolatilitySurfaceMesh
from options_bt.pricing import Black76


def test_smile_fitting_synthetic():
    """Verify parametric polynomial smile fitting and skew extraction on synthetic chain."""
    engine = VolatilitySurfaceEngine()
    F = 24000.0
    T = 10.0 / 365.0
    r = 0.065

    strikes = [23600, 23800, 24000, 24200, 24400]
    # Realistic equity index volatility smile: Put IV > ATM IV > Call IV
    true_ivs = {23600: 0.17, 23800: 0.155, 24000: 0.140, 24200: 0.132, 24400: 0.128}

    call_px = {K: Black76.price(F, K, T, r, true_ivs[K], "CE") for K in strikes}
    put_px = {K: Black76.price(F, K, T, r, true_ivs[K], "PE") for K in strikes}

    smile = engine.fit_smile_slice(
        forward_price=F,
        strikes=strikes,
        call_prices=call_px,
        put_prices=put_px,
        time_to_expiry_years=T,
        r=r,
        timestamp="2025-01-02 09:25:00"
    )

    assert 0.135 < smile.atm_iv < 0.145, f"ATM IV misestimated: {smile.atm_iv}"
    assert smile.atm_strike == 24000
    assert len(smile.points) == 5

    # Check fitted evaluation
    atm_eval = smile.evaluate_fitted_iv(24000)
    assert abs(atm_eval - 0.140) < 0.015

    # Downside Put IV must be higher than upside Call IV (Negative skew)
    put_wing = smile.evaluate_fitted_iv(23600)
    call_wing = smile.evaluate_fitted_iv(24400)
    assert put_wing > call_wing, "Equities must display standard downward-sloping Put Skew"


def test_real_chain_snapshot_extraction():
    """Verify extracting and fitting real market smile from sample_data/options/NIFTY_20250102.csv."""
    options_file = "sample_data/options/NIFTY_20250102.csv"
    if not os.path.exists(options_file):
        pytest.skip("NIFTY_20250102.csv not found.")

    engine = VolatilitySurfaceEngine()
    spot_px = 23750.0
    T = 0.5 / 365.0  # Expiry session

    smile = engine.extract_snapshot_from_file(
        options_file_path=options_file,
        time_str="09:25:00",
        spot_price=spot_px,
        time_to_expiry_years=T,
        r=0.065
    )

    assert smile is not None
    assert smile.atm_strike in (23700, 23750, 23800)
    assert 0.10 <= smile.atm_iv <= 0.45, f"Unreasonable ATM IV: {smile.atm_iv}"
    assert len(smile.points) > 10, f"Expected >10 valid liquid strikes, got {len(smile.points)}"

    # Ensure fitted log-moneyness evaluation works
    iv_atm = smile.evaluate_fitted_iv(smile.atm_strike)
    assert 0.10 <= iv_atm <= 0.45


def test_multi_day_surface_mesh():
    """Verify multi-session 3D Volatility Surface mesh construction and JSON serialization."""
    opt_files = sorted(glob.glob("sample_data/options/NIFTY_2025010*.csv"))[:3]
    spot_file = "sample_data/spot/nifty50_1min_sample.csv"

    if len(opt_files) < 2 or not os.path.exists(spot_file):
        pytest.skip("Insufficient options or spot files for multi-day surface test.")

    spot_df = pd.read_csv(spot_file)
    engine = VolatilitySurfaceEngine()

    mesh = engine.build_surface_from_files(
        options_file_paths=opt_files,
        spot_df=spot_df,
        time_str="09:25:00",
        r=0.065
    )

    assert mesh.iv_matrix.shape[0] == len(opt_files)
    assert mesh.iv_matrix.shape[1] == len(mesh.strikes)
    assert len(mesh.expiries_days) == len(opt_files)

    # Check serialization
    mesh_dict = mesh.to_dict()
    assert "strikes" in mesh_dict
    assert "iv_matrix" in mesh_dict
    assert len(mesh_dict["iv_matrix"]) == len(opt_files)

    # Verify all IV values are positive
    assert np.all(mesh.iv_matrix > 0.0)
