"""Order Management System Package"""

from .order_router import OrderBookRouter, LiquidityImpactResult
from .multi_leg_router import MultiLegRouter, SingleLegOrder, MultiLegExecutionReport
from .pre_trade_risk import PreTradeRiskManager, RiskCheckResult
from .consensus_engine import AlphaConsensusEngine, ConsensusDecision
from .models import (
    OrderSide,
    OrderType,
    OrderStatus,
    RejectReason,
    Order,
    Fill,
    Position,
    AccountPnL,
    StrategyPnL,
    InstancePnL,
)
from .position_book import HierarchicalPositionBook
from .audit_logger import AuditLogger
from .engine import OMSEngine

__all__ = [
    "OrderBookRouter",
    "LiquidityImpactResult",
    "MultiLegRouter",
    "SingleLegOrder",
    "MultiLegExecutionReport",
    "PreTradeRiskManager",
    "RiskCheckResult",
    "AlphaConsensusEngine",
    "ConsensusDecision",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "RejectReason",
    "Order",
    "Fill",
    "Position",
    "AccountPnL",
    "StrategyPnL",
    "InstancePnL",
    "HierarchicalPositionBook",
    "AuditLogger",
    "OMSEngine",
]
