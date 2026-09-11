"""
Unit Tests for Equity Factor Engine, Survivorship Bias Defense, and Tax Models
"""

import pytest
import numpy as np
import pandas as pd
from equity_bt.pit_universe import PointInTimeUniverse
from equity_bt.costs_tax import EquityCostModel
from equity_bt.pca_factor_model import PCAResidualMomentumEngine, PCAStrategyConfig


def test_equity_cost_and_tax():
    """Verify Indian statutory charges, slippage, and STCG tax math."""
    cost_model = EquityCostModel(stcg_tax_rate=0.20)

    # ₹10 Lakh trade turnover
    cost = cost_model.calculate_turnover_friction(1_000_000.0)
    assert cost.stt == 1000.0  # 0.1% of 10L = 1000
    assert cost.total_statutory > 1000.0
    assert cost.slippage_cost == 1500.0  # 0.15% of 10L = 1500

    # Tax on ₹50,000 profit
    tax = cost_model.apply_annual_stcg_tax(50_000.0)
    assert tax == 10_000.0  # 20% of 50k = 10k

    # Tax on loss is zero
    assert cost_model.apply_annual_stcg_tax(-20_000.0) == 0.0


def test_pit_universe_survivorship_audit():
    """Verify that survivorship bias audit correctly identifies survivor-only names."""
    static_universe = {"RELIANCE", "TCS", "INFY", "HDFCBANK", "NEW_WINNER"}
    pit_historical = {"RELIANCE", "TCS", "INFY", "HDFCBANK", "DELISTED_FAIL"}

    audit = PointInTimeUniverse.audit_survivorship_bias(static_universe, pit_historical)
    assert audit["total_pit_count"] == 5
    assert audit["overlap_count"] == 4
    assert audit["survivor_bias_stocks_count"] == 1
    assert "NEW_WINNER" in audit["survivor_bias_stocks"]
    assert "DELISTED_FAIL" in audit["historical_excluded_sample"]


def test_pca_residual_extraction():
    """Verify PCA successfully strips common systematic factor from returns."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    market = np.random.normal(0.001, 0.015, len(dates))

    data = {}
    for i in range(10):
        beta = 1.0 + (i * 0.05)
        idio = np.random.normal(0.0, 0.008, len(dates))
        data[f"STOCK_{i}"] = beta * market + idio

    returns_df = pd.DataFrame(data, index=dates)

    engine = PCAResidualMomentumEngine(PCAStrategyConfig(n_components=1))
    residuals = engine.extract_residual_returns(returns_df, n_components=1)

    # Residual returns must have significantly lower correlation with market factor
    raw_corr = returns_df.corrwith(pd.Series(market, index=dates)).mean()
    res_corr = residuals.corrwith(pd.Series(market, index=dates)).mean()

    assert abs(res_corr) < abs(raw_corr) * 0.35, "PCA failed to strip market beta"
