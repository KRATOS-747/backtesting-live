"""
Canonical Data Models & Enums for the Order Management System (OMS)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import datetime


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL_LIMIT = "SL_LIMIT"
    SL_MARKET = "SL_MARKET"


class OrderStatus(str, Enum):
    PENDING_RISK = "PENDING_RISK"
    REJECTED_RISK = "REJECTED_RISK"
    SUBMITTED = "SUBMITTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED_EXCHANGE = "REJECTED_EXCHANGE"
    IN_FLIGHT_AMBIGUOUS = "IN_FLIGHT_AMBIGUOUS"
    EXPIRED = "EXPIRED"


class RejectReason(str, Enum):
    INSUFFICIENT_MARGIN = "INSUFFICIENT_MARGIN"
    CIRCUIT_LIMIT_BREACH = "CIRCUIT_LIMIT_BREACH"
    FREEZE_QUANTITY_EXCEEDED = "FREEZE_QUANTITY_EXCEEDED"
    NETWORK_TIMEOUT = "NETWORK_TIMEOUT"
    PRICE_DEVIATION_ANOMALY = "PRICE_DEVIATION_ANOMALY"
    PRE_TRADE_RISK_BREACH = "PRE_TRADE_RISK_BREACH"
    INVALID_INSTRUMENT = "INVALID_INSTRUMENT"


@dataclass
class Fill:
    """Represents a single verified execution report from the exchange/broker."""
    fill_id: str
    order_id: str
    exchange_trade_id: str
    account_id: str
    strategy_id: str
    instance_id: str
    exchange_token: str
    symbol: str
    side: OrderSide
    qty: int
    price: float
    timestamp: str
    statutory_fees: float = 0.0
    slippage_pts: float = 0.0


@dataclass
class Order:
    """Complete lifecycle order record tracked through the OMS state machine."""
    client_order_id: str
    account_id: str
    strategy_id: str
    instance_id: str
    exchange_token: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    requested_qty: int
    limit_price: float = 0.0
    trigger_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING_RISK
    broker_order_id: Optional[str] = None
    exchange_order_id: Optional[str] = None
    filled_qty: int = 0
    leaves_qty: int = 0
    avg_fill_price: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
    updated_at: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
    rejection_reason: Optional[str] = None
    error_code: Optional[RejectReason] = None
    fills: List[Fill] = field(default_factory=list)

    def __post_init__(self):
        if self.leaves_qty == 0 and self.filled_qty == 0:
            self.leaves_qty = self.requested_qty


@dataclass
class Position:
    """Real-time position tracking net quantities, average prices, and MTM P&L."""
    account_id: str
    strategy_id: str
    instance_id: str
    exchange_token: str
    symbol: str
    net_qty: int = 0
    buy_qty: int = 0
    buy_val: float = 0.0
    sell_qty: int = 0
    sell_val: float = 0.0
    avg_buy_price: float = 0.0
    avg_sell_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_mtm: float = 0.0
    current_ltp: float = 0.0
    last_updated: str = ""


@dataclass
class AccountPnL:
    """Aggregated financial metrics for a specific broker account."""
    account_id: str
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_mtm: float = 0.0
    total_fees: float = 0.0
    net_capital_used: float = 0.0
    active_positions_count: int = 0


@dataclass
class StrategyPnL:
    """Aggregated financial metrics for a specific strategy across instances."""
    strategy_id: str
    account_id: str
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_mtm: float = 0.0
    active_positions_count: int = 0


@dataclass
class InstancePnL:
    """Aggregated financial metrics for an individual portfolio instance."""
    instance_id: str
    strategy_id: str
    account_id: str
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_mtm: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)
