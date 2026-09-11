"""
Comprehensive Unit Tests for the Institutional Order Management System (OMS)
Validates multi-tier isolation, order lifecycle state machine, error handling,
real-time MTM roll-up, and audit ledger crash recovery.
"""

import os
import tempfile
import pytest

from oms.models import (
    OrderSide,
    OrderType,
    OrderStatus,
    RejectReason,
    Order,
    Fill,
    Position,
)
from oms.position_book import HierarchicalPositionBook
from oms.pre_trade_risk import PreTradeRiskManager
from oms.audit_logger import AuditLogger
from oms.engine import OMSEngine


@pytest.fixture
def temp_audit_logger():
    """Provides an AuditLogger writing to a temporary file."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
        temp_path = tf.name
    logger = AuditLogger(ledger_path=temp_path)
    yield logger
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def oms_setup(temp_audit_logger):
    """Initializes a full OMS stack with position book and risk manager."""
    pos_book = HierarchicalPositionBook()
    risk_mgr = PreTradeRiskManager(max_net_delta=150.0, max_net_gamma=25.0, max_margin_utilization_pct=85.0)
    engine = OMSEngine(position_book=pos_book, risk_manager=risk_mgr, audit_logger=temp_audit_logger)
    return engine, pos_book, risk_mgr, temp_audit_logger


def test_multi_tier_isolation(oms_setup):
    """Verifies strict isolation across Accounts, Strategies, and Instances on the same exchange token."""
    engine, pos_book, _, _ = oms_setup

    # Trade 1: Account UPSTOX_01, Strategy ALPHA1, Instance NIFTY_40PT
    o1 = engine.create_order(
        account_id="UPSTOX_01",
        strategy_id="ALPHA1_STRADDLE",
        instance_id="NIFTY_40PT",
        exchange_token="26000",
        symbol="NIFTY25JAN24000CE",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        qty=75,
    )
    engine.validate_and_submit(o1)
    engine.process_acknowledgment(o1.client_order_id, "BRK_001")
    engine.process_execution_report(o1.client_order_id, fill_qty=75, fill_price=100.0)

    # Trade 2: Account UPSTOX_01, Strategy ALPHA2, Instance NIFTY_90M (same token, different strategy)
    o2 = engine.create_order(
        account_id="UPSTOX_01",
        strategy_id="ALPHA2_SPIKE_FADE",
        instance_id="NIFTY_90M",
        exchange_token="26000",
        symbol="NIFTY25JAN24000CE",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        qty=75,
    )
    engine.validate_and_submit(o2)
    engine.process_acknowledgment(o2.client_order_id, "BRK_002")
    engine.process_execution_report(o2.client_order_id, fill_qty=75, fill_price=110.0)

    # Trade 3: Account ZERODHA_02 (different account, same token)
    o3 = engine.create_order(
        account_id="ZERODHA_02",
        strategy_id="ALPHA1_STRADDLE",
        instance_id="NIFTY_40PT",
        exchange_token="26000",
        symbol="NIFTY25JAN24000CE",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        qty=150,
    )
    engine.validate_and_submit(o3)
    engine.process_acknowledgment(o3.client_order_id, "BRK_003")
    engine.process_execution_report(o3.client_order_id, fill_qty=150, fill_price=105.0)

    # Assert individual isolation
    p1 = pos_book.get_position("UPSTOX_01", "ALPHA1_STRADDLE", "NIFTY_40PT", "26000")
    assert p1 is not None
    assert p1.net_qty == 75
    assert p1.avg_buy_price == 100.0

    p2 = pos_book.get_position("UPSTOX_01", "ALPHA2_SPIKE_FADE", "NIFTY_90M", "26000")
    assert p2 is not None
    assert p2.net_qty == -75
    assert p2.avg_sell_price == 110.0

    p3 = pos_book.get_position("ZERODHA_02", "ALPHA1_STRADDLE", "NIFTY_40PT", "26000")
    assert p3 is not None
    assert p3.net_qty == 150
    assert p3.avg_buy_price == 105.0


def test_order_lifecycle_state_machine(oms_setup):
    """Tests clean progression: PENDING_RISK -> SUBMITTED -> ACKNOWLEDGED -> PARTIALLY_FILLED -> FILLED."""
    engine, pos_book, _, _ = oms_setup

    order = engine.create_order(
        account_id="UPSTOX_PROP",
        strategy_id="ALPHA1",
        instance_id="INST_1",
        exchange_token="26050",
        symbol="NIFTY25JAN24050CE",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        qty=100,
        limit_price=85.0,
    )
    assert order.status == OrderStatus.PENDING_RISK
    assert order.leaves_qty == 100

    # Risk Check
    approved, order = engine.validate_and_submit(order)
    assert approved
    assert order.status == OrderStatus.SUBMITTED

    # Broker Acknowledgment
    order = engine.process_acknowledgment(order.client_order_id, broker_order_id="BROKER_ORD_991")
    assert order.status == OrderStatus.ACKNOWLEDGED
    assert order.broker_order_id == "BROKER_ORD_991"

    # Partial Fill 1: 40 units @ 85.0
    ok, fill1, order = engine.process_execution_report(order.client_order_id, fill_qty=40, fill_price=85.0)
    assert ok
    assert order.status == OrderStatus.PARTIALLY_FILLED
    assert order.filled_qty == 40
    assert order.leaves_qty == 60
    assert order.avg_fill_price == 85.0

    # Partial Fill 2: 60 units @ 84.50 (Completes order)
    ok, fill2, order = engine.process_execution_report(order.client_order_id, fill_qty=60, fill_price=84.50)
    assert ok
    assert order.status == OrderStatus.FILLED
    assert order.filled_qty == 100
    assert order.leaves_qty == 0
    # Weighted avg: (40*85 + 60*84.5) / 100 = (3400 + 5070) / 100 = 84.70
    assert order.avg_fill_price == pytest.approx(84.70, rel=1e-3)


def test_pre_trade_risk_rejection(oms_setup):
    """Tests rejection when projected portfolio delta or margin ceiling is breached."""
    engine, _, _, _ = oms_setup

    order = engine.create_order(
        account_id="UPSTOX_PROP",
        strategy_id="ALPHA1",
        instance_id="INST_1",
        exchange_token="26050",
        symbol="NIFTY25JAN24050CE",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        qty=500,
    )

    # Order delta = +200.0 breaches desk limit of max_net_delta = 150.0
    approved, order = engine.validate_and_submit(
        order,
        current_portfolio_delta=0.0,
        order_delta=200.0,
        total_account_capital=1_000_000.0,
    )

    assert not approved
    assert order.status == OrderStatus.REJECTED_RISK
    assert order.error_code == RejectReason.PRE_TRADE_RISK_BREACH
    assert "delta" in order.rejection_reason.lower()


def test_exchange_rms_rejection(oms_setup):
    """Verifies exchange circuit / RMS rejection handling."""
    engine, _, _, _ = oms_setup

    order = engine.create_order(
        account_id="UPSTOX_PROP",
        strategy_id="ALPHA1",
        instance_id="INST_1",
        exchange_token="26050",
        symbol="NIFTY25JAN24050CE",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        qty=75,
    )
    engine.validate_and_submit(order)

    # Exchange rejects for insufficient margin
    rejected_order = engine.process_rejection(
        order.client_order_id,
        error_code=RejectReason.INSUFFICIENT_MARGIN,
        reason="RMS: Margin deficit of ₹24,500",
    )
    assert rejected_order.status == OrderStatus.REJECTED_EXCHANGE
    assert rejected_order.error_code == RejectReason.INSUFFICIENT_MARGIN


def test_in_flight_ambiguity_and_reconciliation(oms_setup):
    """Tests network disconnect / timeout handling and subsequent order reconciliation."""
    engine, _, _, _ = oms_setup

    order = engine.create_order(
        account_id="UPSTOX_PROP",
        strategy_id="ALPHA1",
        instance_id="INST_1",
        exchange_token="26050",
        symbol="NIFTY25JAN24050CE",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        qty=75,
    )
    engine.validate_and_submit(order)

    # Socket disconnect while awaiting response
    amb_order = engine.process_timeout_or_disconnect(order.client_order_id, reason="Gateway socket timeout (5000ms)")
    assert amb_order.status == OrderStatus.IN_FLIGHT_AMBIGUOUS
    assert amb_order.error_code == RejectReason.NETWORK_TIMEOUT

    # OMS queries broker order book and confirms order was filled at exchange
    resolved = engine.reconcile_ambiguous_order(
        order.client_order_id,
        confirmed_status=OrderStatus.FILLED,
        confirmed_filled_qty=75,
        avg_fill_price=92.50,
        note="Order confirmed filled via Upstox order book reconciliation",
    )
    assert resolved.status == OrderStatus.FILLED
    assert resolved.filled_qty == 75
    assert resolved.avg_fill_price == 92.50


def test_real_time_mtm_pnl_rollup(oms_setup):
    """Tests Realized P&L, Unrealized MTM, and roll-ups across all 4 tiers."""
    engine, pos_book, _, _ = oms_setup

    # Open Long 100 qty @ ₹100.0 (fee ₹40)
    o1 = engine.create_order("ACC1", "STRAT1", "INST1", "TOK_A", "SYM_A", OrderSide.BUY, OrderType.MARKET, 100)
    engine.validate_and_submit(o1)
    engine.process_acknowledgment(o1.client_order_id, "B1")
    engine.process_execution_report(o1.client_order_id, fill_qty=100, fill_price=100.0, statutory_fees=40.0)

    p = pos_book.get_position("ACC1", "STRAT1", "INST1", "TOK_A")
    assert p.net_qty == 100
    assert p.avg_buy_price == 100.0
    assert p.realized_pnl == -40.0

    # Market moves up: LTP = 110.0
    pos_book.update_market_price("TOK_A", 110.0)
    assert p.unrealized_pnl == 1000.0  # 100 * (110 - 100)
    assert p.total_mtm == 960.0        # 1000 - 40

    # Close 50 qty @ ₹115.0 (fee ₹20)
    o2 = engine.create_order("ACC1", "STRAT1", "INST1", "TOK_A", "SYM_A", OrderSide.SELL, OrderType.MARKET, 50)
    engine.validate_and_submit(o2)
    engine.process_acknowledgment(o2.client_order_id, "B2")
    engine.process_execution_report(o2.client_order_id, fill_qty=50, fill_price=115.0, statutory_fees=20.0)

    # Realized from closing 50: 50 * (115 - 100) - 20 = 750 - 20 = +730
    # Total Realized = -40 + 730 = +690
    assert p.realized_pnl == 690.0
    assert p.net_qty == 50

    # Roll-up verification at Instance, Strategy, Account, and Firm levels
    inst_pnl = pos_book.get_instance_pnl("ACC1", "STRAT1", "INST1")
    assert inst_pnl.realized_pnl == 690.0

    strat_pnl = pos_book.get_strategy_pnl("ACC1", "STRAT1")
    assert strat_pnl.realized_pnl == 690.0

    acc_pnl = pos_book.get_account_pnl("ACC1")
    assert acc_pnl.realized_pnl == 690.0

    firm_pnl = pos_book.get_firm_pnl()
    assert firm_pnl["total_realized_pnl"] == 690.0


def test_audit_logger_crash_replay_and_hydration(oms_setup):
    """Tests disaster recovery: replaying the audit ledger reconstructs the exact position book."""
    engine, pos_book, _, audit_logger = oms_setup

    # Execute 2 trades across 2 different accounts
    o1 = engine.create_order("ACC_ALPHA", "STRAT1", "INST1", "TOK_1", "NIFTY24000CE", OrderSide.BUY, OrderType.MARKET, 75)
    engine.validate_and_submit(o1)
    engine.process_acknowledgment(o1.client_order_id, "B_1")
    engine.process_execution_report(o1.client_order_id, fill_qty=75, fill_price=120.0, statutory_fees=45.0)

    o2 = engine.create_order("ACC_BETA", "STRAT2", "INST2", "TOK_2", "NIFTY24000PE", OrderSide.SELL, OrderType.MARKET, 75)
    engine.validate_and_submit(o2)
    engine.process_acknowledgment(o2.client_order_id, "B_2")
    engine.process_execution_report(o2.client_order_id, fill_qty=75, fill_price=95.0, statutory_fees=45.0)

    p1_orig = pos_book.get_position("ACC_ALPHA", "STRAT1", "INST1", "TOK_1")
    p2_orig = pos_book.get_position("ACC_BETA", "STRAT2", "INST2", "TOK_2")

    # SIMULATE SYSTEM CRASH: Wipe out in-memory position book
    new_position_book = HierarchicalPositionBook()
    assert len(new_position_book.get_all_positions()) == 0

    # HYDRATE FROM AUDIT LEDGER
    replayed_count = audit_logger.replay_and_hydrate(new_position_book)
    assert replayed_count == 2

    p1_hydrated = new_position_book.get_position("ACC_ALPHA", "STRAT1", "INST1", "TOK_1")
    p2_hydrated = new_position_book.get_position("ACC_BETA", "STRAT2", "INST2", "TOK_2")

    assert p1_hydrated is not None
    assert p1_hydrated.net_qty == p1_orig.net_qty == 75
    assert p1_hydrated.avg_buy_price == p1_orig.avg_buy_price == 120.0
    assert p1_hydrated.realized_pnl == p1_orig.realized_pnl == -45.0

    assert p2_hydrated is not None
    assert p2_hydrated.net_qty == p2_orig.net_qty == -75
    assert p2_hydrated.avg_sell_price == p2_orig.avg_sell_price == 95.0
    assert p2_hydrated.realized_pnl == p2_orig.realized_pnl == -45.0
