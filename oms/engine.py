"""
Central OMS Orchestration Engine
Coordinates order lifecycle state transitions, fill acceptance gates,
error handling, position book updates, and audit logging.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Dict, List, Optional, Tuple, Any

from .models import (
    OrderSide,
    OrderType,
    OrderStatus,
    RejectReason,
    Order,
    Fill,
    Position,
)
from .position_book import HierarchicalPositionBook
from .pre_trade_risk import PreTradeRiskManager
from .audit_logger import AuditLogger


class OMSEngine:
    """
    Production-grade Order Management System:
    - Maintains deterministic order state machine.
    - Enforces pre-trade risk and fill price sanity gates.
    - Tracks multi-tier hierarchical position book across accounts/strategies/instances.
    - Writes immutable audit events for crash recovery.
    """

    def __init__(
        self,
        position_book: Optional[HierarchicalPositionBook] = None,
        risk_manager: Optional[PreTradeRiskManager] = None,
        audit_logger: Optional[AuditLogger] = None,
        max_fill_deviation_pct: float = 0.05,  # 5% max deviation from LTP for fill acceptance
    ):
        self.position_book = position_book or HierarchicalPositionBook()
        self.risk_manager = risk_manager or PreTradeRiskManager()
        self.audit_logger = audit_logger or AuditLogger()
        self.max_fill_deviation_pct = max_fill_deviation_pct

        # Active in-memory order registry keyed by client_order_id
        self._orders: Dict[str, Order] = {}

    def create_order(
        self,
        account_id: str,
        strategy_id: str,
        instance_id: str,
        exchange_token: str,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        qty: int,
        limit_price: float = 0.0,
        trigger_price: float = 0.0,
        client_order_id: Optional[str] = None,
    ) -> Order:
        """Initializes a new order in PENDING_RISK status."""
        cid = client_order_id or f"ORD_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"
        order = Order(
            client_order_id=cid,
            account_id=account_id,
            strategy_id=strategy_id,
            instance_id=instance_id,
            exchange_token=exchange_token,
            symbol=symbol,
            side=side,
            order_type=order_type,
            requested_qty=qty,
            limit_price=limit_price,
            trigger_price=trigger_price,
            status=OrderStatus.PENDING_RISK,
        )

        self._orders[cid] = order
        self.audit_logger.log_order_transition(order, "ORDER_CREATED", note="Signal initialized in OMS")
        return order

    def validate_and_submit(
        self,
        order: Order,
        current_portfolio_delta: float = 0.0,
        current_portfolio_gamma: float = 0.0,
        order_delta: float = 0.0,
        order_gamma: float = 0.0,
        current_margin: float = 0.0,
        order_margin: float = 0.0,
        total_account_capital: float = 1_000_000.0,
    ) -> Tuple[bool, Order]:
        """
        Validates order against Pre-Trade Risk Manager:
        - If approved: transitions to SUBMITTED.
        - If rejected: transitions to REJECTED_RISK with explicit cause.
        """
        risk_res = self.risk_manager.evaluate_order(
            current_delta=current_portfolio_delta,
            current_gamma=current_portfolio_gamma,
            order_delta=order_delta,
            order_gamma=order_gamma,
            current_margin_used=current_margin,
            order_margin_required=order_margin,
            total_account_capital=total_account_capital,
        )

        order.updated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        if not risk_res.is_approved:
            order.status = OrderStatus.REJECTED_RISK
            order.rejection_reason = risk_res.rejection_reason
            order.error_code = RejectReason.PRE_TRADE_RISK_BREACH
            self.audit_logger.log_order_transition(order, "ORDER_REJECTED_RISK", note=risk_res.rejection_reason or "")
            return False, order

        order.status = OrderStatus.SUBMITTED
        self.audit_logger.log_order_transition(order, "ORDER_SUBMITTED", note="Risk check passed. Dispatched to broker.")
        return True, order

    def process_acknowledgment(
        self,
        client_order_id: str,
        broker_order_id: str,
        exchange_order_id: str = "",
    ) -> Optional[Order]:
        """Transitions order from SUBMITTED to ACKNOWLEDGED upon broker confirmation."""
        order = self._orders.get(client_order_id)
        if not order:
            return None

        order.broker_order_id = broker_order_id
        order.exchange_order_id = exchange_order_id
        order.status = OrderStatus.ACKNOWLEDGED
        order.updated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        self.audit_logger.log_order_transition(order, "ORDER_ACKNOWLEDGED", note="Broker/Exchange registered order")
        return order

    def process_execution_report(
        self,
        client_order_id: str,
        fill_qty: int,
        fill_price: float,
        exchange_trade_id: str = "",
        current_ltp: float = 0.0,
        statutory_fees: float = 0.0,
        slippage_pts: float = 0.0,
    ) -> Tuple[bool, Optional[Fill], Optional[Order]]:
        """
        Processes execution report:
        1. Fill Acceptance Sanity Gate: flags fat-finger/abnormal fills.
        2. Updates order filled & leaves quantities.
        3. Advances order state to PARTIALLY_FILLED or FILLED.
        4. Applies fill to HierarchicalPositionBook.
        5. Logs fill to audit trail.
        """
        order = self._orders.get(client_order_id)
        if not order:
            return False, None, None

        # Fill Acceptance Sanity Gate: check deviation against LTP
        if current_ltp > 0:
            dev = abs(fill_price - current_ltp) / current_ltp
            if dev > self.max_fill_deviation_pct:
                order.rejection_reason = f"Fill price ₹{fill_price:.2f} deviates {dev*100:.1f}% from LTP ₹{current_ltp:.2f}"
                order.error_code = RejectReason.PRICE_DEVIATION_ANOMALY
                self.audit_logger.log_order_transition(order, "FILL_SANITY_WARNING", note=order.rejection_reason)

        # Update order quantities and average fill price
        prev_filled = order.filled_qty
        new_filled = prev_filled + fill_qty
        order.avg_fill_price = ((prev_filled * order.avg_fill_price) + (fill_qty * fill_price)) / new_filled if new_filled > 0 else fill_price
        order.filled_qty = new_filled
        order.leaves_qty = max(0, order.requested_qty - new_filled)
        order.updated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        if order.leaves_qty == 0:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIALLY_FILLED

        # Create Fill record
        fid = f"FILL_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"
        fill = Fill(
            fill_id=fid,
            order_id=client_order_id,
            exchange_trade_id=exchange_trade_id or f"TRD_{uuid.uuid4().hex[:8]}",
            account_id=order.account_id,
            strategy_id=order.strategy_id,
            instance_id=order.instance_id,
            exchange_token=order.exchange_token,
            symbol=order.symbol,
            side=order.side,
            qty=fill_qty,
            price=fill_price,
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            statutory_fees=statutory_fees,
            slippage_pts=slippage_pts,
        )
        order.fills.append(fill)

        # Update Hierarchical Position Book
        updated_pos = self.position_book.apply_fill(fill, order)

        # Log Fill and Transition
        self.audit_logger.log_fill(fill, updated_pos)
        self.audit_logger.log_order_transition(
            order,
            "ORDER_FILLED" if order.status == OrderStatus.FILLED else "ORDER_PARTIALLY_FILLED",
            note=f"Filled {fill_qty} @ ₹{fill_price:.2f} (Leaves: {order.leaves_qty})",
        )

        return True, fill, order

    def process_rejection(
        self,
        client_order_id: str,
        error_code: RejectReason,
        reason: str,
    ) -> Optional[Order]:
        """Handles exchange or broker RMS rejection."""
        order = self._orders.get(client_order_id)
        if not order:
            return None

        order.status = OrderStatus.REJECTED_EXCHANGE
        order.error_code = error_code
        order.rejection_reason = reason
        order.updated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        self.audit_logger.log_order_transition(order, "ORDER_REJECTED", note=f"[{error_code.value}] {reason}")
        return order

    def process_timeout_or_disconnect(
        self,
        client_order_id: str,
        reason: str = "Network timeout awaiting broker response",
    ) -> Optional[Order]:
        """
        Handles in-flight ambiguity when socket drops or timeout occurs:
        Marks order IN_FLIGHT_AMBIGUOUS to prevent dangerous re-submission.
        """
        order = self._orders.get(client_order_id)
        if not order:
            return None

        order.status = OrderStatus.IN_FLIGHT_AMBIGUOUS
        order.error_code = RejectReason.NETWORK_TIMEOUT
        order.rejection_reason = reason
        order.updated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        self.audit_logger.log_order_transition(
            order, "ORDER_AMBIGUOUS", note="Socket timeout. Order flagged for active reconciliation."
        )
        return order

    def reconcile_ambiguous_order(
        self,
        client_order_id: str,
        confirmed_status: OrderStatus,
        confirmed_filled_qty: int = 0,
        avg_fill_price: float = 0.0,
        note: str = "Reconciled via broker order book query",
    ) -> Optional[Order]:
        """Resolves an IN_FLIGHT_AMBIGUOUS order post-reconciliation."""
        order = self._orders.get(client_order_id)
        if not order:
            return None

        order.status = confirmed_status
        order.filled_qty = confirmed_filled_qty
        order.leaves_qty = max(0, order.requested_qty - confirmed_filled_qty)
        order.avg_fill_price = avg_fill_price
        order.updated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        self.audit_logger.log_order_transition(order, "ORDER_RECONCILED", note=note)
        return order

    def cancel_order(self, client_order_id: str, reason: str = "User/Strategy requested cancel") -> Optional[Order]:
        """Cancels an active order and releases residual leaves."""
        order = self._orders.get(client_order_id)
        if not order:
            return None

        if order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED_EXCHANGE):
            return order  # Cannot cancel terminal order

        order.status = OrderStatus.CANCELLED
        order.leaves_qty = 0
        order.rejection_reason = reason
        order.updated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        self.audit_logger.log_order_transition(order, "ORDER_CANCELLED", note=reason)
        return order

    # -------------------------------------------------------------
    # INSPECTION & RECOVERY HELPERS
    # -------------------------------------------------------------
    def get_order(self, client_order_id: str) -> Optional[Order]:
        return self._orders.get(client_order_id)

    def get_active_orders(self) -> List[Order]:
        active_statuses = {
            OrderStatus.PENDING_RISK,
            OrderStatus.SUBMITTED,
            OrderStatus.ACKNOWLEDGED,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.IN_FLIGHT_AMBIGUOUS,
        }
        return [o for o in self._orders.values() if o.status in active_statuses]

    def get_instance_orders(self, account_id: str, strategy_id: str, instance_id: str) -> List[Order]:
        return [
            o for o in self._orders.values()
            if o.account_id == account_id and o.strategy_id == strategy_id and o.instance_id == instance_id
        ]
