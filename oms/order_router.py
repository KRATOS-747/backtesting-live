"""
L2 Order Book Sweeper & Execution Market Impact Engine
Simulates sweeping live 50-level order books to calculate VWAP fill price and market impact in basis points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class LiquidityImpactResult:
    is_safe: bool
    estimated_impact_bps: float
    vwap_price: float
    best_price: float
    fill_qty: float
    reason: str


class OrderBookRouter:
    """
    Pre-trade liquidity impact gatekeeper and execution router:
    1. Sweeps through L2 order book levels up to required order_qty.
    2. Computes cumulative VWAP.
    3. Calculates basis point deviation from top-of-book best price.
    4. Blocks orders if impact > max_slip_bps or available depth < order_qty.
    """

    @staticmethod
    def check_liquidity_impact(
        bids_or_asks: Dict[float, float],
        side: str,
        order_qty: float,
        best_price: float,
        max_slip_bps: float = 15.0
    ) -> LiquidityImpactResult:
        """
        Sweeps order book side:
        - LONG order sweeps asks (sorted ascending)
        - SHORT order sweeps bids (sorted descending)
        """
        if not bids_or_asks:
            return LiquidityImpactResult(
                is_safe=False,
                estimated_impact_bps=999.0,
                vwap_price=0.0,
                best_price=best_price,
                fill_qty=0.0,
                reason="Order book is empty"
            )

        if best_price <= 0:
            return LiquidityImpactResult(
                is_safe=False,
                estimated_impact_bps=999.0,
                vwap_price=0.0,
                best_price=0.0,
                fill_qty=0.0,
                reason="Invalid best price in order book"
            )

        # Sort price levels
        reverse = (side.upper() in ("SHORT", "SELL"))
        sorted_levels = sorted(bids_or_asks.items(), key=lambda x: x[0], reverse=reverse)

        cum_qty = 0.0
        cum_val = 0.0

        for price, qty in sorted_levels:
            fill_step = min(qty, order_qty - cum_qty)
            cum_qty += fill_step
            cum_val += (price * fill_step)

            if cum_qty >= order_qty:
                break

        if cum_qty < order_qty:
            return LiquidityImpactResult(
                is_safe=False,
                estimated_impact_bps=999.0,
                vwap_price=0.0,
                best_price=best_price,
                fill_qty=cum_qty,
                reason=f"Insufficient order book depth (Available: {cum_qty:.2f} < Req: {order_qty:.2f})"
            )

        vwap = cum_val / cum_qty

        # Basis point impact from top of book
        if side.upper() in ("LONG", "BUY"):
            impact_bps = ((vwap - best_price) / best_price) * 10000.0
        else:
            impact_bps = ((best_price - vwap) / best_price) * 10000.0

        if impact_bps > max_slip_bps:
            return LiquidityImpactResult(
                is_safe=False,
                estimated_impact_bps=round(impact_bps, 2),
                vwap_price=round(vwap, 4),
                best_price=best_price,
                fill_qty=cum_qty,
                reason=f"Market impact ({impact_bps:.2f} bps) exceeds ceiling ({max_slip_bps:.2f} bps)"
            )

        return LiquidityImpactResult(
            is_safe=True,
            estimated_impact_bps=round(impact_bps, 2),
            vwap_price=round(vwap, 4),
            best_price=best_price,
            fill_qty=cum_qty,
            reason=f"Clean sweep verified ({impact_bps:.2f} bps <= {max_slip_bps:.2f} bps)"
        )
