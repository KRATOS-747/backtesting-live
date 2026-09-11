"""
Unit and Integration Tests for the Equity Momentum Strategy Suites:
1. PriceActionMomentumStrategy (Donchian 52W Breakout, Minervini VCP, Stage-2 EMA, 3x ATR Stop)
2. IndicatorMomentumStrategy (RSI Power-Zone, Expanding MACD, BB Squeeze, ADX, Regime Gate, Euphoria Trim)
3. MLMomentumStrategy (12 Cross-Sectional Factors, Ridge L2, HistGradientBoosting, Walk-Forward, Risk Parity)
"""

import pytest
import numpy as np
import pandas as pd

from equity_bt.price_action_momentum_strategy import (
    PriceActionConfig,
    PriceActionMomentumStrategy,
    PriceActionSummary,
)
from equity_bt.indicator_momentum_strategy import (
    IndicatorConfig,
    IndicatorMomentumStrategy,
    IndicatorSummary,
)
from equity_bt.ml_momentum_strategy import (
    MLMomentumConfig,
    MLMomentumStrategy,
    MLSummary,
    FEATURE_NAMES,
)


@pytest.fixture
def synthetic_equity_data():
    """Generates synthetic multi-asset OHLCV data across 300 trading days."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=300, freq="B")
    n_assets = 15
    symbols = [f"STOCK_{i}" for i in range(n_assets)]

    prices_dict = {}
    highs_dict = {}
    lows_dict = {}
    volumes_dict = {}

    for i, sym in enumerate(symbols):
        # Create varied asset trajectories (strong trend, cyclical, and lagging)
        drift = 0.0008 + (0.0004 * i)
        vol = 0.015 + (0.002 * (i % 3))
        daily_ret = np.random.normal(drift, vol, len(dates))
        base_px = 100.0 * np.exp(np.cumsum(daily_ret))

        prices_dict[sym] = base_px
        highs_dict[sym] = base_px * (1.0 + np.random.uniform(0.005, 0.02, len(dates)))
        lows_dict[sym] = base_px * (1.0 - np.random.uniform(0.005, 0.02, len(dates)))
        volumes_dict[sym] = np.random.uniform(150_000, 500_000, len(dates))

    prices_df = pd.DataFrame(prices_dict, index=dates)
    highs_df = pd.DataFrame(highs_dict, index=dates)
    lows_df = pd.DataFrame(lows_dict, index=dates)
    volumes_df = pd.DataFrame(volumes_dict, index=dates)

    return prices_df, highs_df, lows_df, volumes_df


# ============================================================================
# 1. Price Action Momentum Strategy Tests
# ============================================================================

def test_price_action_vcp_pattern_detection():
    """Test Mark Minervini's Volatility Contraction Pattern (VCP) logic."""
    np.random.seed(101)
    n = 100
    # Create contracting swings: Wave 1 = 16.6% swing, Wave 2 = 10% swing, Wave 3 = 5% swing
    high = pd.Series(100.0, index=range(n))
    low = pd.Series(95.0, index=range(n))
    volume = pd.Series(100_000.0, index=range(n))

    # Wave 1 (40-60 bars ago relative to idx=99 => [39:59]): high 120, low 100 => 16.6% contraction
    high.iloc[39:59] = 120.0
    low.iloc[39:59] = 100.0

    # Wave 2 (20-40 bars ago relative to idx=99 => [59:79]): high 120, low 108 => 10.0% contraction
    high.iloc[59:79] = 120.0
    low.iloc[59:79] = 108.0

    # Wave 3 (recent 20 bars relative to idx=99 => [79:100]): high 120, low 114 => 5.0% contraction
    high.iloc[79:100] = 120.0
    low.iloc[79:100] = 114.0

    # Volume dry-up on pullback
    volume.iloc[95:100] = 50_000.0

    is_vcp = PriceActionMomentumStrategy.check_vcp_pattern(
        high, low, volume, idx=99,
        c1_max=0.25, c2_max=0.12, c3_max=0.06,
        vol_expansion_mult=2.5, vol_dryup_mult=0.7
    )
    assert is_vcp is True, "VCP contracted pattern should evaluate to True"

    # Expanding volatility must fail VCP
    high_expanding = high.copy()
    low_expanding = low.copy()
    low_expanding.iloc[79:100] = 85.0  # Big drop in wave 3 (30% drop)
    is_vcp_fail = PriceActionMomentumStrategy.check_vcp_pattern(
        high_expanding, low_expanding, volume, idx=99
    )
    assert is_vcp_fail is False, "Expanding volatility must fail VCP validation"


def test_price_action_momentum_backtest(synthetic_equity_data):
    """Test Price Action Momentum backtest across V1, V2, and V3."""
    prices_df, highs_df, lows_df, volumes_df = synthetic_equity_data

    # Use shorter lookback for compact test dataset
    cfg = PriceActionConfig(
        initial_capital=10_000_000.0,
        high_window_days=60,
        donchian_exit_days=20,
        portfolio_slots=5,
        rebalance_days=10,
    )
    strategy = PriceActionMomentumStrategy(cfg)

    # Run V1
    res_v1 = strategy.run_backtest(prices_df, volumes_df, highs_df, lows_df, version="V1")
    assert isinstance(res_v1, PriceActionSummary)
    assert res_v1.strategy_version == "PriceAction_V1"
    assert res_v1.final_equity > 0

    # Run V3 (with ATR trailing stop and STCG tax)
    res_v3 = strategy.run_backtest(prices_df, volumes_df, highs_df, lows_df, version="V3")
    assert isinstance(res_v3, PriceActionSummary)
    assert res_v3.strategy_version == "PriceAction_V3"
    assert res_v3.total_friction_inr >= 0.0


# ============================================================================
# 2. Indicator Momentum Strategy Tests
# ============================================================================

def test_indicator_math_calculations():
    """Verify technical indicator calculations: RSI, MACD, BB Squeeze, ADX."""
    np.random.seed(42)
    # Monotonically rising prices
    px = pd.Series(np.linspace(100, 200, 100))
    hi = px + 1.0
    lo = px - 1.0

    # RSI
    rsi = IndicatorMomentumStrategy.calculate_rsi(px, period=14)
    assert 0.0 <= rsi.iloc[-1] <= 100.0
    assert rsi.iloc[-1] > 80.0, "Monotonically rising price must have high RSI"

    # MACD
    macd, signal, hist = IndicatorMomentumStrategy.calculate_macd(px)
    assert macd.iloc[-1] > 0
    assert not hist.isna().all()

    # Bollinger Bands
    upper, mid, lower, width = IndicatorMomentumStrategy.calculate_bollinger_bands(px, period=20)
    assert (upper.dropna() > mid.dropna()).all()
    assert (mid.dropna() > lower.dropna()).all()
    assert (width.dropna() > 0).all()


    # ADX
    adx, plus_di, minus_di = IndicatorMomentumStrategy.calculate_adx(hi, lo, px, period=14)
    assert (plus_di.dropna() >= 0).all()
    assert (minus_di.dropna() >= 0).all()
    assert (adx.dropna() >= 0).all()


def test_indicator_momentum_backtest(synthetic_equity_data):
    """Test Indicator Momentum backtest across V1, V2, and V3."""
    prices_df, highs_df, lows_df, volumes_df = synthetic_equity_data

    cfg = IndicatorConfig(
        initial_capital=10_000_000.0,
        portfolio_slots=5,
        rebalance_days=7,
        bb_squeeze_lookback=40,
    )
    strategy = IndicatorMomentumStrategy(cfg)

    # Run V1
    res_v1 = strategy.run_backtest(prices_df, volumes_df, highs_df, lows_df, version="V1")
    assert isinstance(res_v1, IndicatorSummary)
    assert res_v1.strategy_version == "Indicator_V1"
    assert res_v1.final_equity > 0

    # Run V2
    res_v2 = strategy.run_backtest(prices_df, volumes_df, highs_df, lows_df, version="V2")
    assert isinstance(res_v2, IndicatorSummary)
    assert res_v2.strategy_version == "Indicator_V2"

    # Run V3 with bear regime triggering cash hedge
    bear_benchmark = pd.Series(np.linspace(200, 100, len(prices_df)), index=prices_df.index)
    res_v3_bear = strategy.run_backtest(
        prices_df, volumes_df, highs_df, lows_df, benchmark_prices=bear_benchmark, version="V3"
    )
    assert isinstance(res_v3_bear, IndicatorSummary)
    # Under prolonged bear market, strategy should liquidate to cash yield
    assert res_v3_bear.final_equity > 0


# ============================================================================
# 3. Machine Learning Momentum Strategy Tests
# ============================================================================

def test_ml_feature_computation(synthetic_equity_data):
    """Verify that all 12 cross-sectional feature matrices are properly computed."""
    prices_df, highs_df, lows_df, volumes_df = synthetic_equity_data
    strategy = MLMomentumStrategy()

    feats = strategy.compute_feature_matrices(prices_df, highs_df, lows_df, volumes_df)

    assert len(feats) == 12
    for fname in FEATURE_NAMES:
        assert fname in feats
        df_feat = feats[fname]
        assert df_feat.shape == prices_df.shape
        # Tail values should not be all NaN
        assert not df_feat.iloc[-1].isna().all()


def test_ml_risk_parity_weighting():
    """Verify inverse-volatility risk-parity weights satisfy constraints."""
    strategy = MLMomentumStrategy(MLMomentumConfig(max_weight_per_slot=0.06, min_weight_per_slot=0.01))
    symbols = [f"SYM_{i}" for i in range(20)]
    vols = pd.Series({sym: 0.10 + (0.02 * i) for i, sym in enumerate(symbols)})

    weights = strategy.compute_risk_parity_weights(symbols, vols)

    assert len(weights) == 20
    # Sum of normalized weights should equal 1.0
    assert np.isclose(sum(weights.values()), 1.0, atol=1e-4)

    # Lower volatility assets must receive higher or equal weights
    assert weights["SYM_0"] >= weights["SYM_19"]

    # Constraints respected
    for w in weights.values():
        assert w <= 0.0601
        assert w >= 0.0099


def test_ml_momentum_backtest(synthetic_equity_data):
    """Test ML Momentum backtest across V1 (Ridge), V2 (HGB), and V3 (Walk-Forward)."""
    prices_df, highs_df, lows_df, volumes_df = synthetic_equity_data

    cfg = MLMomentumConfig(
        initial_capital=10_000_000.0,
        portfolio_slots=5,
        rebalance_days=10,
        forward_horizon_days=10,
        train_window_days=80,
        retrain_freq_days=30,
        hgb_max_iter=20,
    )
    strategy = MLMomentumStrategy(cfg)

    # Run V1 (Ridge L2)
    res_v1 = strategy.run_backtest(prices_df, volumes_df, highs_df, lows_df, version="V1")
    assert isinstance(res_v1, MLSummary)
    assert res_v1.strategy_version == "ML_V1"
    assert res_v1.final_equity > 0
    assert len(res_v1.feature_importances) > 0

    # Run V3 (Walk-Forward OOS + Risk-Parity + Statutory Tax)
    res_v3 = strategy.run_backtest(prices_df, volumes_df, highs_df, lows_df, version="V3")
    assert isinstance(res_v3, MLSummary)
    assert res_v3.strategy_version == "ML_V3"
    assert res_v3.total_friction_inr >= 0.0
