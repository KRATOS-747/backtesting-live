"""
Unit & Integration Tests for OptionsVisualizer Suite
Validates interactive 3D volatility surface generation, Greek waterfall,
intraday straddle trajectory, payoff profiles, and Open Interest distribution dashboards.
"""

import os
import pytest
import numpy as np
import pandas as pd
from options_bt.visualizer import OptionsVisualizer
from options_bt.surface import VolatilitySurfaceMesh
from options_bt.backtester import SessionResult


def test_plot_3d_volatility_surface(tmp_path):
    # Construct synthetic surface mesh
    moneyness = np.linspace(-0.05, 0.05, 10)
    expiries = np.linspace(1, 30, 8)
    strikes = np.linspace(23000, 24000, 10)
    m_grid, e_grid = np.meshgrid(moneyness, expiries)
    iv_matrix = 0.15 + 0.5 * (m_grid ** 2) + 0.001 * e_grid

    mesh = VolatilitySurfaceMesh(
        timestamps=["09:25:00"] * len(expiries),
        expiries_days=expiries,
        strikes=strikes,
        moneyness=moneyness,
        iv_matrix=iv_matrix
    )

    out_file = str(tmp_path / "test_surface_3d.html")
    res_path = OptionsVisualizer.plot_3d_volatility_surface(mesh, save_path=out_file)

    assert os.path.exists(res_path)
    assert os.path.getsize(res_path) > 1000  # Valid non-empty HTML


def test_plot_greek_waterfall(tmp_path):
    attributions = {
        "total_pnl": 5200.0,
        "delta_pnl": 2100.0,
        "gamma_pnl": -800.0,
        "theta_pnl": 3500.0,
        "vega_pnl": -400.0,
        "residual_pnl": 800.0,
    }

    out_file = str(tmp_path / "test_waterfall.html")
    res_path = OptionsVisualizer.plot_greek_waterfall(attributions, save_path=out_file)

    assert os.path.exists(res_path)
    assert os.path.getsize(res_path) > 1000


def test_plot_intraday_straddle_trajectory(tmp_path):
    trade_bars = pd.DataFrame({
        "time_str": ["09:25:00", "09:26:00", "09:27:00"],
        "ce_price": [115.0, 116.0, 114.0],
        "pe_price": [120.0, 118.0, 117.0],
        "ce_oi": [4000000.0, 4050000.0, 4100000.0],
        "pe_oi": [3800000.0, 3850000.0, 3900000.0],
    })

    session = SessionResult(
        date="2025-01-01",
        atm_strike=23650,
        entry_spot=23635.15,
        ce_entry=115.0,
        pe_entry=120.0,
        straddle_entry=235.0,
        ce_exit=114.0,
        pe_exit=117.0,
        ce_reason="EOD_SQUAREOFF",
        pe_reason="EOD_SQUAREOFF",
        gross_points=4.0,
        net_points=2.5,
        gross_pnl_rupees=400.0,
        net_pnl_rupees=250.0,
        statutory_costs_rupees=150.0,
        attributions={}
    )

    out_file = str(tmp_path / "test_trajectory.html")
    res_path = OptionsVisualizer.plot_intraday_straddle_trajectory(trade_bars, session, save_path=out_file)

    assert os.path.exists(res_path)
    assert os.path.getsize(res_path) > 1000


def test_plot_multi_leg_payoff(tmp_path):
    legs = [
        {"strike": 23650, "type": "CE", "side": "SHORT", "premium": 115.0},
        {"strike": 23650, "type": "PE", "side": "SHORT", "premium": 120.0},
    ]

    out_file = str(tmp_path / "test_payoff.html")
    res_path = OptionsVisualizer.plot_multi_leg_payoff("ATM Straddle", legs, save_path=out_file)

    assert os.path.exists(res_path)
    assert os.path.getsize(res_path) > 1000


def test_plot_oi_distribution(tmp_path):
    snap = pd.DataFrame({
        "strike": [23400, 23500, 23600, 23700, 23800],
        "ce_oi": [1000000.0, 2500000.0, 5000000.0, 8000000.0, 4000000.0],
        "pe_oi": [7000000.0, 6000000.0, 4000000.0, 2000000.0, 500000.0],
    })

    out_file = str(tmp_path / "test_oi_dist.html")
    res_path = OptionsVisualizer.plot_oi_distribution(
        snap, spot_price=23635.15, max_pain_strike=23600, save_path=out_file
    )

    assert os.path.exists(res_path)
    assert os.path.getsize(res_path) > 1000
