"""
Unit & Integration Tests for OpenInterestAnalyzer Engine
Validates buildup classification, Put-Call Ratio (PCR), settlement Max Pain,
OI velocity / institutional surge detection, and real January 2025 chain processing.
"""

import pytest
import numpy as np
import pandas as pd
from options_bt.oi_analyzer import (
    OpenInterestAnalyzer,
    BuildupType,
    BuildupRecord,
    PCRMetrics,
    MaxPainResult,
)
from options_bt.chain_parser import OptionsChainParser


def test_classify_buildup_quadrants():
    # Long Buildup: Price +5%, OI +10%
    b_type, sent = OpenInterestAnalyzer.classify_buildup(105.0, 100.0, 1100.0, 1000.0)
    assert b_type == BuildupType.LONG_BUILDUP.value
    assert sent == "BULLISH"

    # Short Buildup: Price -5%, OI +10%
    b_type, sent = OpenInterestAnalyzer.classify_buildup(95.0, 100.0, 1100.0, 1000.0)
    assert b_type == BuildupType.SHORT_BUILDUP.value
    assert sent == "BEARISH"

    # Long Unwinding: Price -5%, OI -10%
    b_type, sent = OpenInterestAnalyzer.classify_buildup(95.0, 100.0, 900.0, 1000.0)
    assert b_type == BuildupType.LONG_UNWINDING.value
    assert sent == "BEARISH"

    # Short Covering: Price +5%, OI -10%
    b_type, sent = OpenInterestAnalyzer.classify_buildup(105.0, 100.0, 900.0, 1000.0)
    assert b_type == BuildupType.SHORT_COVERING.value
    assert sent == "BULLISH"

    # Sub-threshold noise
    b_type, sent = OpenInterestAnalyzer.classify_buildup(100.05, 100.0, 1002.0, 1000.0)
    assert b_type == BuildupType.NEUTRAL.value
    assert sent == "NEUTRAL"


def test_contract_buildup_series_put_inversion():
    dates = pd.date_range("2025-01-01 09:15:00", periods=4, freq="5min")
    
    # CE contract: price rising with OI rising -> Bullish
    df_ce = pd.DataFrame({
        "DateTime": dates,
        "price": [100.0, 110.0, 120.0, 115.0],
        "oi": [10000.0, 12000.0, 15000.0, 13000.0],
    })
    records_ce = OpenInterestAnalyzer.analyze_contract_buildup_series(
        df_ce, strike=23500, option_type="CE"
    )
    assert len(records_ce) >= 2
    assert records_ce[0].buildup_type == BuildupType.LONG_BUILDUP.value
    assert records_ce[0].sentiment == "BULLISH"

    # PE contract: price rising with OI rising -> Long Buildup in Put -> Bearish market sentiment!
    df_pe = pd.DataFrame({
        "DateTime": dates,
        "price": [100.0, 110.0, 120.0, 115.0],
        "oi": [10000.0, 12000.0, 15000.0, 13000.0],
    })
    records_pe = OpenInterestAnalyzer.analyze_contract_buildup_series(
        df_pe, strike=23500, option_type="PE"
    )
    assert records_pe[0].buildup_type == BuildupType.LONG_BUILDUP.value
    assert records_pe[0].sentiment == "BEARISH"  # Put buying reflects downside protection


def test_calculate_pcr_synthetic():
    snap = pd.DataFrame({
        "strike": [23400, 23500, 23600],
        "ce_oi": [10000.0, 20000.0, 30000.0],  # Total = 60000
        "pe_oi": [30000.0, 25000.0, 15000.0],  # Total = 70000
        "ce_volume": [5000.0, 10000.0, 15000.0],
        "pe_volume": [6000.0, 12000.0, 18000.0],
    })

    pcr = OpenInterestAnalyzer.calculate_pcr(snap, spot_price=23500.0, strike_range=50.0)
    assert round(pcr.oi_pcr, 3) == round(70000.0 / 60000.0, 3)
    assert pcr.volume_pcr is not None
    assert round(pcr.volume_pcr, 3) == round(36000.0 / 30000.0, 3)
    assert pcr.sentiment == "BULLISH_BIAS"  # 1.167 falls in [1.05, 1.40]
    # ATM only strike 23500: pe=25000, ce=20000 -> 1.25
    assert pcr.atm_oi_pcr == 1.25


def test_calculate_max_pain_synthetic():
    # Symmetric distribution around 23500
    # Calls concentrated at 23600, Puts concentrated at 23400
    snap = pd.DataFrame({
        "strike": [23400, 23500, 23600],
        "ce_oi": [1000.0, 5000.0, 20000.0],
        "pe_oi": [20000.0, 5000.0, 1000.0],
    })

    res = OpenInterestAnalyzer.calculate_max_pain(snap, spot_price=23520.0)
    assert isinstance(res, MaxPainResult)
    assert res.max_pain_strike == 23500
    assert res.distance_to_spot == 20.0
    assert res.total_payout_at_max_pain > 0
    assert 23500 in res.strike_payouts


def test_oi_velocity_and_surge_detection():
    # Series with normal small changes, followed by an institutional surge bar
    dates = pd.date_range("2025-01-01 09:15:00", periods=8, freq="1min")
    ois = [10000, 10050, 10100, 10150, 10200, 10250, 25000, 25050]  # Bar 6 has +14750 surge
    df = pd.DataFrame({"DateTime": dates, "oi": ois})

    vel_df = OpenInterestAnalyzer.compute_oi_velocity(df, window_bars=5, surge_zscore_threshold=2.0)
    assert "oi_velocity" in vel_df.columns
    assert "is_surge" in vel_df.columns
    assert vel_df.loc[6, "is_surge"]
    assert vel_df.loc[6, "surge_type"] == "INSTITUTIONAL_ACCUMULATION"


def test_strike_oi_matrix_walls():
    snap = pd.DataFrame({
        "strike": [23400, 23500, 23600, 23700],
        "ce_oi": [5000.0, 15000.0, 30000.0, 80000.0],   # Wall at 23700
        "pe_oi": [70000.0, 20000.0, 10000.0, 2000.0],   # Wall at 23400
    })

    matrix = OpenInterestAnalyzer.build_strike_oi_matrix(snap)
    assert matrix.loc[matrix["strike"] == 23700, "is_call_wall"].values[0]
    assert matrix.loc[matrix["strike"] == 23400, "is_put_wall"].values[0]


def test_real_jan2025_chain_oi_analysis():
    parser = OptionsChainParser("sample_data/options/NIFTY_20250101.csv")
    snap = parser.get_chain_snapshot("09:25:00")

    assert not snap.empty
    assert "strike" in snap.columns

    # 1. PCR
    pcr = OpenInterestAnalyzer.calculate_pcr(snap, spot_price=23625.0)
    assert 0.5 < pcr.oi_pcr < 2.0
    assert pcr.total_call_oi > 0
    assert pcr.total_put_oi > 0

    # 2. Max Pain
    max_pain = OpenInterestAnalyzer.calculate_max_pain(snap, spot_price=23625.0)
    assert 23000 <= max_pain.max_pain_strike <= 24200
    assert max_pain.distance_to_spot is not None

    # 3. Strike Matrix
    walls = OpenInterestAnalyzer.build_strike_oi_matrix(snap)
    assert walls["is_call_wall"].sum() == 1
    assert walls["is_put_wall"].sum() == 1
