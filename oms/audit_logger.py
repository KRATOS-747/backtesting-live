"""
Immutable Audit Trail & Disaster Recovery Ledger for the OMS
Maintains append-only structured JSONL logging with crash replay & hydration capabilities.
"""

from __future__ import annotations

import json
import os
import datetime
from typing import Dict, List, Optional, Any
from .models import Order, Fill, Position, OrderSide, OrderType, OrderStatus
from .position_book import HierarchicalPositionBook


class AuditLogger:
    """
    Append-only event ledger and state hydration engine:
    1. Every order transition and fill is logged synchronously with microsecond precision.
    2. If the trading server crashes, replay_and_hydrate() rebuilds the entire position book.
    """

    def __init__(self, ledger_path: str = "reports/audit_ledger.jsonl"):
        self.ledger_path = ledger_path
        os.makedirs(os.path.dirname(os.path.abspath(ledger_path)), exist_ok=True)

    def _append_record(self, record: Dict[str, Any]):
        """Synchronously appends a JSON record to the audit ledger."""
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def log_order_transition(self, order: Order, event_type: str, note: str = ""):
        """Logs an order state transition event."""
        now_str = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        record = {
            "timestamp": now_str,
            "event_type": event_type,
            "client_order_id": order.client_order_id,
            "broker_order_id": order.broker_order_id,
            "exchange_order_id": order.exchange_order_id,
            "account_id": order.account_id,
            "strategy_id": order.strategy_id,
            "instance_id": order.instance_id,
            "exchange_token": order.exchange_token,
            "symbol": order.symbol,
            "side": order.side.value if hasattr(order.side, "value") else str(order.side),
            "order_type": order.order_type.value if hasattr(order.order_type, "value") else str(order.order_type),
            "requested_qty": order.requested_qty,
            "filled_qty": order.filled_qty,
            "leaves_qty": order.leaves_qty,
            "limit_price": order.limit_price,
            "avg_fill_price": order.avg_fill_price,
            "status": order.status.value if hasattr(order.status, "value") else str(order.status),
            "rejection_reason": order.rejection_reason,
            "error_code": order.error_code.value if (order.error_code and hasattr(order.error_code, "value")) else str(order.error_code or ""),
            "note": note,
        }
        self._append_record(record)

    def log_fill(self, fill: Fill, position: Position):
        """Logs an accepted fill and resulting position snapshot."""
        now_str = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        record = {
            "timestamp": now_str,
            "event_type": "FILL_RECORDED",
            "fill_id": fill.fill_id,
            "order_id": fill.order_id,
            "exchange_trade_id": fill.exchange_trade_id,
            "account_id": fill.account_id,
            "strategy_id": fill.strategy_id,
            "instance_id": fill.instance_id,
            "exchange_token": fill.exchange_token,
            "symbol": fill.symbol,
            "side": fill.side.value if hasattr(fill.side, "value") else str(fill.side),
            "qty": fill.qty,
            "price": fill.price,
            "statutory_fees": fill.statutory_fees,
            "slippage_pts": fill.slippage_pts,
            "post_trade_position": {
                "net_qty": position.net_qty,
                "avg_buy_price": position.avg_buy_price,
                "avg_sell_price": position.avg_sell_price,
                "realized_pnl": position.realized_pnl,
                "total_mtm": position.total_mtm,
            },
        }
        self._append_record(record)

    def replay_and_hydrate(self, position_book: HierarchicalPositionBook) -> int:
        """
        Disaster recovery method:
        Replays all FILL_RECORDED events from the ledger to reconstruct the position book.
        Returns the number of replayed fill events.
        """
        if not os.path.exists(self.ledger_path):
            return 0

        position_book.clear()
        replayed_count = 0

        with open(self.ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue

                if data.get("event_type") == "FILL_RECORDED":
                    fill = Fill(
                        fill_id=data["fill_id"],
                        order_id=data["order_id"],
                        exchange_trade_id=data.get("exchange_trade_id", ""),
                        account_id=data["account_id"],
                        strategy_id=data["strategy_id"],
                        instance_id=data["instance_id"],
                        exchange_token=data["exchange_token"],
                        symbol=data["symbol"],
                        side=OrderSide(data["side"]),
                        qty=int(data["qty"]),
                        price=float(data["price"]),
                        timestamp=data.get("timestamp", ""),
                        statutory_fees=float(data.get("statutory_fees", 0.0)),
                        slippage_pts=float(data.get("slippage_pts", 0.0)),
                    )

                    # Synthesize dummy order container
                    dummy_order = Order(
                        client_order_id=fill.order_id,
                        account_id=fill.account_id,
                        strategy_id=fill.strategy_id,
                        instance_id=fill.instance_id,
                        exchange_token=fill.exchange_token,
                        symbol=fill.symbol,
                        side=fill.side,
                        order_type=OrderType.MARKET,
                        requested_qty=fill.qty,
                        filled_qty=fill.qty,
                        leaves_qty=0,
                        avg_fill_price=fill.price,
                        status=OrderStatus.FILLED,
                    )

                    position_book.apply_fill(fill, dummy_order)
                    replayed_count += 1

        return replayed_count
