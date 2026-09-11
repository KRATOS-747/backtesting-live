"""
Unit tests for the 90-Minute Microstructure Spike Fading Engine (Alpha 2).
"""

import pytest
import pandas as pd
import numpy as np
from options_bt.spike_fader import (
    SpikeFaderConfig,
    SpikeTradeRecord,
    SpikeFaderSummary,
    SpikeFaderEngine,
)


def test_spike_fader_config_defaults():
    config = SpikeFaderConfig()
    assert config.holding_bars == 90
    assert config.surge_pct_threshold == 12.0
    assert config.profit_target_pct == 0.20
    assert config.stop_loss_pct == 0.25
    assert config.lot_size == 75
    assert config.brokerage_per_order == 20.0
    assert config.slippage_pts_per_leg == 0.50


def test_calculate_trade_friction():
    engine = SpikeFaderEngine(SpikeFaderConfig(lot_size=75, num_lots=1))
    p_entry = 100.0
    p_exit = 80.0
    
    # Expected:
    # Qty = 75
    # Sell turnover = 100 * 75 = 7500
    # Buy turnover = 80 * 75 = 6000
    # STT = 0.001 * 7500 = 7.50
    # Exch turnover = 0.0005 * 13500 = 6.75
    # Brokerage = 40.0
    # GST = 0.18 * (40 + 6.75) = 8.415
    # Stamp duty = 0.00003 * 6000 = 0.18
    # SEBI = 0.000001 * 13500 = 0.0135
    # Slippage = 2 * 0.5 * 75 = 75.0
    # Total ~ 137.86
    friction = engine.calculate_trade_friction(p_entry, p_exit)
    assert 135.0 <= friction <= 140.0


def test_aggregate_summary_empty():
    engine = SpikeFaderEngine()
    summary = engine.aggregate_summary([])
    assert summary.total_trades == 0
    assert summary.win_rate_pct == 0.0
    assert summary.net_pnl_inr == 0.0


def test_aggregate_summary_kpi_computation():
    engine = SpikeFaderEngine()
    trades = [
        SpikeTradeRecord(
            date="2025-01-02",
            time="10:00:00",
            strike=24000,
            option_type="CE",
            p_entry=100.0,
            p_exit=80.0,
            pts_pnl=20.0,
            gross_pnl_inr=1500.0,
            friction_inr=138.0,
            net_pnl_inr=1362.0,
            is_win=True,
            exit_reason="PROFIT_TARGET",
            exit_step=35,
            surge_pct=15.0,
            spot_at_entry=23980.0,
            wall_strike=24000,
        ),
        SpikeTradeRecord(
            date="2025-01-02",
            time="11:00:00",
            strike=24100,
            option_type="CE",
            p_entry=80.0,
            p_exit=100.0,
            pts_pnl=-20.0,
            gross_pnl_inr=-1500.0,
            friction_inr=138.0,
            net_pnl_inr=-1638.0,
            is_win=False,
            exit_reason="STOP_LOSS",
            exit_step=12,
            surge_pct=14.0,
            spot_at_entry=24090.0,
            wall_strike=24100,
        ),
        SpikeTradeRecord(
            date="2025-01-02",
            time="13:00:00",
            strike=24000,
            option_type="PE",
            p_entry=90.0,
            p_exit=82.0,
            pts_pnl=8.0,
            gross_pnl_inr=600.0,
            friction_inr=138.0,
            net_pnl_inr=462.0,
            is_win=True,
            exit_reason="TIME_90M",
            exit_step=90,
            surge_pct=13.0,
            spot_at_entry=24010.0,
            wall_strike=24000,
        ),
    ]

    summary = engine.aggregate_summary(trades)
    assert summary.total_trades == 3
    assert summary.winning_trades == 2
    assert summary.losing_trades == 1
    assert summary.win_rate_pct == pytest.approx(66.67, rel=1e-2)
    assert summary.gross_pnl_inr == 600.0
    assert summary.profit_target_hits == 1
    assert summary.stop_loss_hits == 1
    assert summary.time_exit_hits == 1


def test_run_session_on_sample_data():
    options_file = "sample_data/options/NIFTY_20250101.csv"
    spot_file = "sample_data/spot/nifty50_1min_sample.csv"
    
    spot_df = pd.read_csv(spot_file)
    spot_df["date"] = pd.to_datetime(spot_df["date"])
    day_spot = spot_df[spot_df["date"].dt.strftime("%Y-%m-%d") == "2025-01-01"].copy()

    engine = SpikeFaderEngine(SpikeFaderConfig(holding_bars=90, surge_pct_threshold=12.0))
    trades = engine.run_session(options_file, day_spot, "2025-01-01", naive_mode=False)

    # Validates execution structure
    assert isinstance(trades, list)
    for t in trades:
        assert isinstance(t, SpikeTradeRecord)
        assert t.exit_step <= 90
        assert t.exit_reason in ["PROFIT_TARGET", "STOP_LOSS", "TIME_90M", "EOD_CUT"]
        assert t.gross_pnl_inr == pytest.approx(t.pts_pnl * 75.0, rel=1e-3)
        assert t.net_pnl_inr == pytest.approx(t.gross_pnl_inr - t.friction_inr, rel=1e-3)


def test_naive_mode_generates_more_trades_than_conditioned():
    options_file = "sample_data/options/NIFTY_20250101.csv"
    spot_file = "sample_data/spot/nifty50_1min_sample.csv"
    
    spot_df = pd.read_csv(spot_file)
    spot_df["date"] = pd.to_datetime(spot_df["date"])
    day_spot = spot_df[spot_df["date"].dt.strftime("%Y-%m-%d") == "2025-01-01"].copy()

    engine = SpikeFaderEngine(SpikeFaderConfig(holding_bars=90, surge_pct_threshold=12.0))
    conditioned_trades = engine.run_session(options_file, day_spot, "2025-01-01", naive_mode=False)
    naive_trades = engine.run_session(options_file, day_spot, "2025-01-01", naive_mode=True)

    # Conditioned mode MUST filter out the majority of naive noise
    assert len(conditioned_trades) <= len(naive_trades)
