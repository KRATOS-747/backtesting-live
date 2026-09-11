"""
Unit & Integration Tests for OptionsBacktester Simulation Engine
Validates single-session execution, Indian statutory friction,
analytical Greek Taylor-series attribution, and multi-session January 2025 batch backtesting.
"""

import os
import pytest
import pandas as pd
from options_bt.backtester import (
    OptionsBacktester,
    SessionResult,
    BacktestMetrics,
)
from options_bt.leg_cutter import LegCutConfig


def test_statutory_costs_calculation():
    bt = OptionsBacktester(lot_size=25, num_lots=4)
    # Total Qty = 100
    # Sell turnover = (100 + 100) * 100 = 20,000
    # Buy turnover = (80 + 80) * 100 = 16,000
    costs = bt.calculate_statutory_costs(sell_turnover=20000.0, buy_turnover=16000.0, num_orders=4)
    
    # Expected:
    # STT = 20000 * 0.0010 = 20.0
    # Exchange = 36000 * 0.00035 = 12.6
    # Brokerage = 4 * 20 = 80.0
    # GST = (12.6 + 80) * 0.18 = 16.668
    # Stamp duty = 16000 * 0.00003 = 0.48
    # SEBI = 36000 * 0.000001 = 0.036
    # Total ~ 129.784
    assert 125.0 <= costs <= 135.0


def test_single_session_real_data():
    spot_df = pd.read_csv("sample_data/spot/nifty50_1min_sample.csv")
    bt = OptionsBacktester(lot_size=25, num_lots=4)

    res = bt.run_session(
        options_file_path="sample_data/options/NIFTY_20250101.csv",
        spot_df=spot_df,
        date_str="2025-01-01",
        entry_time="09:25:00"
    )

    assert res is not None
    assert isinstance(res, SessionResult)
    assert res.date == "2025-01-01"
    assert res.atm_strike == 23650
    assert res.straddle_entry > 200.0
    assert res.ce_reason in ("STATIC_SL", "PROFIT_TARGET", "ROC_ACCELERATION_CUT", "EOD_SQUAREOFF")
    assert res.pe_reason in ("STATIC_SL", "PROFIT_TARGET", "ROC_ACCELERATION_CUT", "EOD_SQUAREOFF")
    assert res.statutory_costs_rupees > 0
    assert res.net_pnl_rupees == round(res.gross_pnl_rupees - res.statutory_costs_rupees, 2)

    # Verify Greek attribution is populated and active
    assert "delta_pnl" in res.attributions
    assert "gamma_pnl" in res.attributions
    assert "theta_pnl" in res.attributions
    assert "vega_pnl" in res.attributions
    assert len(res.bar_attributions) > 0


def test_multi_session_batch_backtest():
    spot_df = pd.read_csv("sample_data/spot/nifty50_1min_sample.csv")
    bt = OptionsBacktester(lot_size=25, num_lots=4)

    # Run batch on the first 3 sessions of January 2025
    sample_dates = ["2025-01-01", "2025-01-02", "2025-01-03"]
    df_summary, metrics = bt.run_backtest(
        options_dir="sample_data/options",
        spot_csv_or_df=spot_df,
        dates=sample_dates,
        entry_time="09:25:00"
    )

    assert len(df_summary) == 3
    assert isinstance(metrics, BacktestMetrics)
    assert metrics.total_sessions == 3
    assert 0.0 <= metrics.win_rate_pct <= 100.0
    assert metrics.gross_pnl_rupees != 0.0
    assert metrics.statutory_costs_rupees > 0.0
    assert metrics.net_pnl_rupees == round(metrics.gross_pnl_rupees - metrics.statutory_costs_rupees, 2)
    assert "delta_pnl" in metrics.greek_attributions
    assert "theta_pnl" in metrics.greek_attributions


def test_missing_data_resilience():
    spot_df = pd.read_csv("sample_data/spot/nifty50_1min_sample.csv")
    bt = OptionsBacktester()

    # Non-existent options file
    res1 = bt.run_session("sample_data/options/NON_EXISTENT.csv", spot_df, "2025-01-01")
    assert res1 is None

    # Date not in spot df
    res2 = bt.run_session("sample_data/options/NIFTY_20250101.csv", spot_df, "1999-01-01")
    assert res2 is None
