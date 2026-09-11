"""
Unit Tests for Independent Leg Cutter and ROC Stop Logic
"""

import pytest
import pandas as pd
from options_bt.leg_cutter import LegCutter, LegCutConfig, LegState


def test_profit_target_trigger():
    """Verify that leg exits when profit target points are achieved."""
    config = LegCutConfig(profit_target_points=25.0, static_sl_points=40.0)
    cutter = LegCutter(config)

    leg = LegState(symbol="24500CE", entry_time="09:25:00", entry_price=100.0)
    # Price falls to 70 (unrealized profit = +30 points >= 25 target)
    history = pd.Series([100.0, 90.0, 80.0, 70.0])

    updated = cutter.update_leg(leg, "10:30:00", 70.0, history)
    assert not updated.is_active
    assert updated.exit_reason == "PROFIT_TARGET"
    assert updated.exit_price == 70.0
    assert updated.pnl_points == 30.0


def test_static_sl_trigger():
    """Verify that leg exits when static stop loss barrier is breached."""
    config = LegCutConfig(profit_target_points=25.0, static_sl_points=40.0)
    cutter = LegCutter(config)

    leg = LegState(symbol="24500PE", entry_time="09:25:00", entry_price=100.0)
    # Price rises to 145 (loss = 45 points >= 40 SL)
    history = pd.Series([100.0, 110.0, 125.0, 145.0])

    updated = cutter.update_leg(leg, "11:15:00", 145.0, history)
    assert not updated.is_active
    assert updated.exit_reason == "STATIC_SL"
    assert updated.exit_price == 145.0
    assert updated.pnl_points == -45.0


def test_roc_acceleration_cut():
    """Verify that rapid gamma acceleration triggers emergency ROC leg cut before static SL."""
    config = LegCutConfig(
        profit_target_points=25.0,
        static_sl_points=50.0,
        roc_window=5,
        roc_threshold_pct=0.40  # 40% jump in 5 bars
    )
    cutter = LegCutter(config)

    leg = LegState(symbol="24500CE", entry_time="09:25:00", entry_price=80.0)
    # Price jumps from 75 to 110 (+46% in 4 bars, total loss 30 pts < 50 static SL)
    history = pd.Series([80.0, 78.0, 75.0, 85.0, 95.0, 110.0])

    updated = cutter.update_leg(leg, "13:45:00", 110.0, history)
    assert not updated.is_active
    assert updated.exit_reason == "ROC_ACCELERATION_CUT"
    assert updated.exit_price == 110.0


def test_hard_eod_squareoff():
    """Verify that active legs square off at hard cutoff time."""
    config = LegCutConfig(hard_cutoff_time="15:15:00")
    cutter = LegCutter(config)

    leg = LegState(symbol="24500CE", entry_time="09:25:00", entry_price=80.0)
    history = pd.Series([80.0, 75.0, 72.0])

    updated = cutter.update_leg(leg, "15:15:00", 72.0, history)
    assert not updated.is_active
    assert updated.exit_reason == "EOD_SQUAREOFF"


def test_percentage_sl_trigger():
    """Verify that percentage-based SL triggers at 30% premium expansion."""
    config = LegCutConfig(profit_target_points=None, static_sl_points=None, static_sl_pct=0.30)
    cutter = LegCutter(config)

    leg = LegState(symbol="24500CE", entry_time="09:25:00", entry_price=100.0)
    # Loss = 35 points on 100 entry = 35% >= 30% SL
    history = pd.Series([100.0, 110.0, 135.0])

    updated = cutter.update_leg(leg, "10:00:00", 135.0, history)
    assert not updated.is_active
    assert updated.exit_reason == "STATIC_SL"
    assert updated.pnl_points == -35.0


def test_trailing_sl_trigger():
    """Verify that trailing SL activates after threshold profit and cuts on pullback."""
    config = LegCutConfig(
        profit_target_points=100.0,
        static_sl_points=50.0,
        trail_start_points=30.0,       # Start trailing when profit reaches 30
        trail_distance_points=10.0      # Lock in profit within 10 points of peak
    )
    cutter = LegCutter(config)

    leg = LegState(symbol="24500CE", entry_time="09:25:00", entry_price=100.0)
    
    # Step 1: Price drops to 60 (profit = +40 points, peak = 40, trail stop = 40 - 10 = 30)
    history1 = pd.Series([100.0, 80.0, 60.0])
    cutter.update_leg(leg, "10:30:00", 60.0, history1)
    assert leg.is_active
    assert leg.highest_profit_points == 40.0

    # Step 2: Price bounces back up to 72 (profit drops to +28 points <= 30 trail barrier)
    history2 = pd.Series([100.0, 80.0, 60.0, 72.0])
    updated = cutter.update_leg(leg, "11:00:00", 72.0, history2)
    assert not updated.is_active
    assert updated.exit_reason == "TRAILING_SL"
    assert updated.exit_price == 72.0
    assert updated.pnl_points == 28.0


def test_long_leg_target_and_sl():
    """Verify that long option legs correctly compute positive PnL on price rises."""
    config = LegCutConfig(profit_target_points=30.0, static_sl_points=20.0)
    cutter = LegCutter(config)

    # Long leg: buy at 50, price rises to 85 (+35 pts profit)
    leg = LegState(symbol="24500CE_HEDGE", entry_time="09:25:00", entry_price=50.0, side="LONG")
    history = pd.Series([50.0, 60.0, 85.0])

    updated = cutter.update_leg(leg, "10:15:00", 85.0, history)
    assert not updated.is_active
    assert updated.exit_reason == "PROFIT_TARGET"
    assert updated.pnl_points == 35.0

