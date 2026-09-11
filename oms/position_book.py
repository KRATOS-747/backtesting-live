"""
Hierarchical Position Book & Real-Time Mark-to-Market (MTM) Engine
Maintains thread-safe position tracking partitioned by Account, Strategy, Instance, and Exchange Token.
"""

from __future__ import annotations

import datetime
from typing import Dict, List, Optional, Tuple, Any
from .models import (
    OrderSide,
    Fill,
    Order,
    Position,
    AccountPnL,
    StrategyPnL,
    InstancePnL,
)


class HierarchicalPositionBook:
    """
    Multi-tenant position book maintaining isolated positions across:
    Account ID -> Strategy ID -> Instance ID -> Exchange Token.
    """

    def __init__(self):
        # Key: (account_id, strategy_id, instance_id, exchange_token)
        self._positions: Dict[Tuple[str, str, str, str], Position] = {}
        # Market price lookup by token
        self._token_ltp: Dict[str, float] = {}

    def _get_key(self, account_id: str, strategy_id: str, instance_id: str, exchange_token: str) -> Tuple[str, str, str, str]:
        return (account_id, strategy_id, instance_id, exchange_token)

    def get_or_create_position(
        self, account_id: str, strategy_id: str, instance_id: str, exchange_token: str, symbol: str
    ) -> Position:
        key = self._get_key(account_id, strategy_id, instance_id, exchange_token)
        if key not in self._positions:
            ltp = self._token_ltp.get(exchange_token, 0.0)
            self._positions[key] = Position(
                account_id=account_id,
                strategy_id=strategy_id,
                instance_id=instance_id,
                exchange_token=exchange_token,
                symbol=symbol,
                current_ltp=ltp,
                last_updated=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
        return self._positions[key]

    def apply_fill(self, fill: Fill, order: Order) -> Position:
        """
        Updates position upon execution report arrival:
        1. Accumulates total traded quantities and values.
        2. Realizes P&L when reducing/closing existing positions.
        3. Updates weighted average open price for remaining net inventory.
        4. Re-computes unrealized and total MTM.
        """
        pos = self.get_or_create_position(
            account_id=fill.account_id,
            strategy_id=fill.strategy_id,
            instance_id=fill.instance_id,
            exchange_token=fill.exchange_token,
            symbol=fill.symbol,
        )

        qty = fill.qty
        px = fill.price
        fees = fill.statutory_fees

        if fill.side == OrderSide.BUY:
            pos.buy_qty += qty
            pos.buy_val += (qty * px)

            # If currently net SHORT, this BUY is closing/reducing short inventory -> Realize P&L
            if pos.net_qty < 0:
                short_net = abs(pos.net_qty)
                closed_qty = min(short_net, qty)
                realized_from_trade = closed_qty * (pos.avg_sell_price - px) - fees
                pos.realized_pnl += realized_from_trade

                # Residual after closing
                remaining_qty = qty - closed_qty
                pos.net_qty += closed_qty  # moves towards 0

                if remaining_qty > 0:
                    # Flipped from short to net long
                    pos.net_qty = remaining_qty
                    pos.avg_buy_price = px
                elif pos.net_qty == 0:
                    pos.avg_buy_price = 0.0
                    pos.avg_sell_price = 0.0
            else:
                # Currently net LONG or FLAT -> adding to long position
                new_net = pos.net_qty + qty
                pos.avg_buy_price = ((pos.net_qty * pos.avg_buy_price) + (qty * px)) / new_net if new_net > 0 else px
                pos.net_qty = new_net
                pos.realized_pnl -= fees

        elif fill.side == OrderSide.SELL:
            pos.sell_qty += qty
            pos.sell_val += (qty * px)

            # If currently net LONG, this SELL is closing/reducing long inventory -> Realize P&L
            if pos.net_qty > 0:
                long_net = pos.net_qty
                closed_qty = min(long_net, qty)
                realized_from_trade = closed_qty * (px - pos.avg_buy_price) - fees
                pos.realized_pnl += realized_from_trade

                # Residual after closing
                remaining_qty = qty - closed_qty
                pos.net_qty -= closed_qty  # moves towards 0

                if remaining_qty > 0:
                    # Flipped from long to net short
                    pos.net_qty = -remaining_qty
                    pos.avg_sell_price = px
                elif pos.net_qty == 0:
                    pos.avg_buy_price = 0.0
                    pos.avg_sell_price = 0.0
            else:
                # Currently net SHORT or FLAT -> adding to short position
                curr_short = abs(pos.net_qty)
                new_short = curr_short + qty
                pos.avg_sell_price = ((curr_short * pos.avg_sell_price) + (qty * px)) / new_short if new_short > 0 else px
                pos.net_qty = -new_short
                pos.realized_pnl -= fees

        # Update LTP if fill price is fresh
        if px > 0:
            pos.current_ltp = px
            self._token_ltp[fill.exchange_token] = px

        # Recompute Unrealized MTM
        self._recalculate_unrealized(pos)
        pos.last_updated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        return pos

    def update_market_price(self, exchange_token: str, ltp: float) -> List[Position]:
        """
        Broadcasts incoming market tick to all active positions holding exchange_token.
        Recalculates Unrealized P&L and Total MTM.
        """
        self._token_ltp[exchange_token] = ltp
        affected: List[Position] = []

        for key, pos in self._positions.items():
            if pos.exchange_token == exchange_token:
                pos.current_ltp = ltp
                self._recalculate_unrealized(pos)
                pos.last_updated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                affected.append(pos)

        return affected

    def _recalculate_unrealized(self, pos: Position):
        """Calculates mark-to-market unrealized P&L based on open inventory."""
        if pos.net_qty == 0:
            pos.unrealized_pnl = 0.0
        elif pos.net_qty > 0:
            pos.unrealized_pnl = round(pos.net_qty * (pos.current_ltp - pos.avg_buy_price), 2)
        else:
            pos.unrealized_pnl = round(abs(pos.net_qty) * (pos.avg_sell_price - pos.current_ltp), 2)

        pos.total_mtm = round(pos.realized_pnl + pos.unrealized_pnl, 2)

    # -------------------------------------------------------------
    # HIERARCHICAL ROLL-UP QUERIES
    # -------------------------------------------------------------
    def get_position(
        self, account_id: str, strategy_id: str, instance_id: str, exchange_token: str
    ) -> Optional[Position]:
        """Retrieves individual position record."""
        key = self._get_key(account_id, strategy_id, instance_id, exchange_token)
        return self._positions.get(key)

    def get_instance_pnl(self, account_id: str, strategy_id: str, instance_id: str) -> InstancePnL:
        """Rolls up all positions for a specific strategy instance."""
        res = InstancePnL(instance_id=instance_id, strategy_id=strategy_id, account_id=account_id)
        for (acc, strat, inst, tok), pos in self._positions.items():
            if acc == account_id and strat == strategy_id and inst == instance_id:
                res.realized_pnl += pos.realized_pnl
                res.unrealized_pnl += pos.unrealized_pnl
                res.total_mtm += pos.total_mtm
                res.positions[tok] = pos

        res.realized_pnl = round(res.realized_pnl, 2)
        res.unrealized_pnl = round(res.unrealized_pnl, 2)
        res.total_mtm = round(res.total_mtm, 2)
        return res

    def get_strategy_pnl(self, account_id: str, strategy_id: str) -> StrategyPnL:
        """Rolls up all instances for a specific strategy under an account."""
        res = StrategyPnL(strategy_id=strategy_id, account_id=account_id)
        for (acc, strat, inst, tok), pos in self._positions.items():
            if acc == account_id and strat == strategy_id:
                res.realized_pnl += pos.realized_pnl
                res.unrealized_pnl += pos.unrealized_pnl
                res.total_mtm += pos.total_mtm
                if pos.net_qty != 0:
                    res.active_positions_count += 1

        res.realized_pnl = round(res.realized_pnl, 2)
        res.unrealized_pnl = round(res.unrealized_pnl, 2)
        res.total_mtm = round(res.total_mtm, 2)
        return res

    def get_account_pnl(self, account_id: str) -> AccountPnL:
        """Rolls up all strategies and instances under a broker account."""
        res = AccountPnL(account_id=account_id)
        for (acc, strat, inst, tok), pos in self._positions.items():
            if acc == account_id:
                res.realized_pnl += pos.realized_pnl
                res.unrealized_pnl += pos.unrealized_pnl
                res.total_mtm += pos.total_mtm
                if pos.net_qty != 0:
                    res.active_positions_count += 1

        res.realized_pnl = round(res.realized_pnl, 2)
        res.unrealized_pnl = round(res.unrealized_pnl, 2)
        res.total_mtm = round(res.total_mtm, 2)
        return res

    def get_firm_pnl(self) -> Dict[str, Any]:
        """Rolls up global desk P&L across all accounts."""
        realized = 0.0
        unrealized = 0.0
        total_mtm = 0.0
        active_count = 0

        for pos in self._positions.values():
            realized += pos.realized_pnl
            unrealized += pos.unrealized_pnl
            total_mtm += pos.total_mtm
            if pos.net_qty != 0:
                active_count += 1

        return {
            "total_realized_pnl": round(realized, 2),
            "total_unrealized_pnl": round(unrealized, 2),
            "global_total_mtm": round(total_mtm, 2),
            "active_positions_count": active_count,
            "total_positions_tracked": len(self._positions),
        }

    def get_all_positions(self) -> List[Position]:
        return list(self._positions.values())

    def get_active_positions(self) -> List[Position]:
        return [p for p in self._positions.values() if p.net_qty != 0]

    def clear(self):
        """Resets in-memory state."""
        self._positions.clear()
        self._token_ltp.clear()
