"""
Unit & Integration Tests for StrangleEngine Harvester
Validates fixed-points and delta-neutral strike selection,
single-session execution with Greek attribution, and batch multi-day backtesting.
"""

import os
import pytest
import pandas as pd
from options_bt.strangle_engine import (
    StrangleEngine,
    StrangleConfig,
    StrangleSessionResult,
)
from options_bt.backtester import BacktestMetrics
from options_bt.chain_parser import OptionsChainParser


def test_strangle_strike_selection_points():
    config = StrangleConfig(mode="POINTS", otm_points=150.0)
    engine = StrangleEngine(config)

    parser = OptionsChainParser("sample_data/options/NIFTY_20250101.csv")
    ce_k, pe_k = engine.select_strikes(parser, spot=23635.15, T_years=1.25 / 365.0)

    # ATM = 23650 -> CE = 23800 (+150), PE = 23500 (-150)
    assert ce_k == 23800
    assert pe_k == 23500
    assert ce_k > 23635.15 > pe_k


def test_strangle_strike_selection_delta():
    config = StrangleConfig(mode="DELTA", target_delta=0.20)
    engine = StrangleEngine(config)

    parser = OptionsChainParser("sample_data/options/NIFTY_20250101.csv")
    ce_k, pe_k = engine.select_strikes(parser, spot=23635.15, T_years=1.25 / 365.0)

    # For 20-delta on Jan 1 2025, call is around 23900 and put is around 23400
    assert ce_k > 23635.15
    assert pe_k < 23635.15
    assert 23800 <= ce_k <= 24000
    assert 23300 <= pe_k <= 23500


def test_strangle_single_session_real_data():
    spot_df = pd.read_csv("sample_data/spot/nifty50_1min_sample.csv")
    config = StrangleConfig(mode="POINTS", otm_points=150.0, lot_size=25, num_lots=4)
    engine = StrangleEngine(config)

    res = engine.run_session(
        options_file_path="sample_data/options/NIFTY_20250101.csv",
        spot_df=spot_df,
        date_str="2025-01-01"
    )

    assert res is not None
    assert isinstance(res, StrangleSessionResult)
    assert res.date == "2025-01-01"
    assert res.call_strike == 23800
    assert res.put_strike == 23500
    assert res.width_points == 300
    assert res.call_entry > 0
    assert res.put_entry > 0
    assert res.strangle_entry == round(res.call_entry + res.put_entry, 2)
    assert res.statutory_costs_rupees > 0
    assert res.net_pnl_rupees == round(res.gross_pnl_rupees - res.statutory_costs_rupees, 2)
    assert "delta_pnl" in res.attributions
    assert "gamma_pnl" in res.attributions
    assert "theta_pnl" in res.attributions


def test_strangle_multi_session_batch_backtest():
    spot_df = pd.read_csv("sample_data/spot/nifty50_1min_sample.csv")
    config = StrangleConfig(mode="POINTS", otm_points=150.0, lot_size=25, num_lots=4)
    engine = StrangleEngine(config)

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
    assert "delta_pnl" in metrics.greek_attributions
    assert "theta_pnl" in metrics.greek_attributions
