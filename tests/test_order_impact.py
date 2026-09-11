"""
Unit Tests for OMS Execution Router, Order Book Sweep, and Pre-Trade Risk
"""

import pytest
from oms.order_router import OrderBookRouter
from oms.multi_leg_router import MultiLegRouter, SingleLegOrder
from oms.pre_trade_risk import PreTradeRiskManager
from oms.consensus_engine import AlphaConsensusEngine


def test_order_book_clean_sweep():
    """Verify that order book sweep accurately calculates VWAP and allows clean fills."""
    # 5-level Ask book
    asks = {
        100.0: 50.0,
        100.5: 50.0,
        101.0: 100.0,
        101.5: 100.0
    }
    # Sweep 75 units: 50 @ 100.0 + 25 @ 100.5 = 5000 + 2512.5 = 7512.5 / 75 = 100.1667
    res = OrderBookRouter.check_liquidity_impact(
        bids_or_asks=asks,
        side="LONG",
        order_qty=75.0,
        best_price=100.0,
        max_slip_bps=25.0
    )
    assert res.is_safe
    assert round(res.vwap_price, 2) == 100.17
    assert res.estimated_impact_bps < 25.0


def test_order_book_thin_liquidity_rejection():
    """Verify that orders exceeding total available depth are blocked."""
    asks = {100.0: 10.0, 100.5: 10.0}
    res = OrderBookRouter.check_liquidity_impact(
        bids_or_asks=asks,
        side="LONG",
        order_qty=50.0,
        best_price=100.0
    )
    assert not res.is_safe
    assert "Insufficient order book depth" in res.reason


def test_multi_leg_atomic_fill():
    """Verify clean dual-leg straddle execution."""
    router = MultiLegRouter(legging_policy="AUTO_UNWIND")
    ce = SingleLegOrder(symbol="24500CE", side="SELL", qty=50, limit_price=120.0)
    pe = SingleLegOrder(symbol="24500PE", side="SELL", qty=50, limit_price=115.0)

    report = router.execute_dual_leg(ce, pe, simulated_fill_ce=True, simulated_fill_pe=True)
    assert report.status == "COMPLETE"
    assert report.net_premium == 235.0


def test_multi_leg_partial_fill_auto_unwind():
    """Verify that legging risk triggers automatic emergency unwind of the filled leg."""
    router = MultiLegRouter(legging_policy="AUTO_UNWIND")
    ce = SingleLegOrder(symbol="24500CE", side="SELL", qty=50, limit_price=120.0)
    pe = SingleLegOrder(symbol="24500PE", side="SELL", qty=50, limit_price=115.0)

    # CE fills, PE fails
    report = router.execute_dual_leg(ce, pe, simulated_fill_ce=True, simulated_fill_pe=False)
    assert report.status == "PARTIAL_UNWOUND"
    assert "EMERGENCY_UNWIND" in report.legging_action


def test_pre_trade_risk_delta_clamp():
    """Verify pre-trade risk blocks orders pushing net delta past limits."""
    manager = PreTradeRiskManager(max_net_delta=100.0)
    res = manager.evaluate_order(
        current_delta=80.0,
        current_gamma=5.0,
        order_delta=30.0,  # Projected = +110 > 100 limit
        order_gamma=2.0,
        current_margin_used=500_000,
        order_margin_required=100_000,
        total_account_capital=2_000_000
    )
    assert not res.is_approved
    assert "desk delta clamp" in res.rejection_reason


def test_alpha_consensus_quorum():
    """Verify multi-signal consensus engine."""
    engine = AlphaConsensusEngine(required_quorum=2)
    votes = {
        "iv_rv_spread": "SHORT",
        "oi_velocity": "SHORT",
        "trend_filter": "HOLD"
    }
    decision = engine.evaluate_votes(votes)
    assert decision.is_quorum_met
    assert decision.action == "SHORT"
