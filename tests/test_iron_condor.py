"""
Unit & Integration Tests for IronCondorEngine (Defined-Risk 4-Leg Spreads)
Validates Iron Condor and Iron Fly strike selection, net credit calculation,
risk containment, single-session execution with Greeks, and multi-session backtesting.
"""

import os
import pytest
import pandas as pd
from options_bt.iron_condor import (
    IronCondorEngine,
    IronCondorConfig,
    IronCondorSessionResult,
)
from options_bt.backtester import BacktestMetrics


def test_iron_condor_strike_selection():
    # Spot 23635.15 -> ATM 23650
    # Short offset = 100 -> Short Put 23550, Short Call 23750
    # Wing width = 150 -> Long Put 23400, Long Call 23900
    config = IronCondorConfig(structure_type="IRON_CONDOR", short_offset=100.0, wing_width=150.0)
    engine = IronCondorEngine(config)

    strikes = engine.select_strikes(23635.15)
    assert not strikes.is_iron_fly
    assert strikes.short_put == 23550
    assert strikes.long_put == 23400
    assert strikes.short_call == 23750
    assert strikes.long_call == 23900
    assert strikes.wing_width == 150


def test_iron_fly_strike_selection():
    # Spot 23635.15 -> ATM 23650
    # Short Put 23650, Short Call 23650
    # Wing width = 200 -> Long Put 23450, Long Call 23850
    config = IronCondorConfig(structure_type="IRON_FLY", wing_width=200.0)
    engine = IronCondorEngine(config)

    strikes = engine.select_strikes(23635.15)
    assert strikes.is_iron_fly
    assert strikes.short_put == 23650
    assert strikes.short_call == 23650
    assert strikes.long_put == 23450
    assert strikes.long_call == 23850


def test_iron_condor_single_session_real_data():
    spot_df = pd.read_csv("sample_data/spot/nifty50_1min_sample.csv")
    config = IronCondorConfig(structure_type="IRON_CONDOR", short_offset=100.0, wing_width=150.0)
    engine = IronCondorEngine(config)

    res = engine.run_session(
        options_file_path="sample_data/options/NIFTY_20250101.csv",
        spot_df=spot_df,
        date_str="2025-01-01"
    )

    assert res is not None
    assert isinstance(res, IronCondorSessionResult)
    assert res.date == "2025-01-01"
    assert res.structure_type == "IRON_CONDOR"
    assert res.short_put == 23550
    assert res.long_put == 23400
    assert res.short_call == 23750
    assert res.long_call == 23900
    assert res.wing_width == 150
    assert res.net_credit_entry > 0
    assert round(res.max_risk_points + res.net_credit_entry, 1) == 150.0
    assert res.statutory_costs_rupees > 0
    assert res.net_pnl_rupees == round(res.gross_pnl_rupees - res.statutory_costs_rupees, 2)
    assert "delta_pnl" in res.attributions
    assert "gamma_pnl" in res.attributions
    assert "theta_pnl" in res.attributions


def test_iron_fly_single_session_real_data():
    spot_df = pd.read_csv("sample_data/spot/nifty50_1min_sample.csv")
    config = IronCondorConfig(structure_type="IRON_FLY", wing_width=200.0)
    engine = IronCondorEngine(config)

    res = engine.run_session(
        options_file_path="sample_data/options/NIFTY_20250101.csv",
        spot_df=spot_df,
        date_str="2025-01-01"
    )

    assert res is not None
    assert res.structure_type == "IRON_FLY"
    assert res.short_put == 23650
    assert res.short_call == 23650
    assert res.long_put == 23450
    assert res.long_call == 23850
    assert res.net_credit_entry > 0
    assert round(res.max_risk_points + res.net_credit_entry, 1) == 200.0


def test_iron_condor_multi_session_batch_backtest():
    spot_df = pd.read_csv("sample_data/spot/nifty50_1min_sample.csv")
    config = IronCondorConfig(structure_type="IRON_CONDOR", short_offset=100.0, wing_width=150.0)
    engine = IronCondorEngine(config)

    sample_dates = ["2025-01-01", "2025-01-02", "2025-01-03"]
    df_summary, metrics = engine.run_backtest(
        options_dir="sample_data/options",
        spot_csv_or_df=spot_df,
        dates=sample_dates
    )

    assert len(df_summary) == 3
    assert isinstance(metrics, BacktestMetrics)
    assert metrics.total_sessions == 3
    assert metrics.statutory_costs_rupees > 0
    assert metrics.net_pnl_rupees == round(metrics.gross_pnl_rupees - metrics.statutory_costs_rupees, 2)
