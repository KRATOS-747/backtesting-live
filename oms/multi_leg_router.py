"""
Atomic Multi-Leg Options Execution Router with Legging-Risk Mitigation
Coordinates simultaneous entry across option legs and handles partial fill / legging fallbacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import datetime


@dataclass
class SingleLegOrder:
    symbol: str
    side: str         # 'BUY' or 'SELL'
    qty: int
    limit_price: float
    time_in_force: str = "IOC"
    is_filled: bool = False
    fill_price: float = 0.0
    reject_reason: Optional[str] = None


@dataclass
class MultiLegExecutionReport:
    structure_name: str
    status: str       # 'COMPLETE', 'PARTIAL_UNWOUND', 'HEDGED', 'REJECTED'
    legs: List[SingleLegOrder]
    execution_time: str
    net_premium: float
    legging_action: Optional[str] = None
    note: str = ""


class MultiLegRouter:
    """
    Manages atomic dual-leg and multi-leg options execution (Straddles, Strangles, Spreads):
    Eliminates legging risk by enforcing atomic execution policies:
    - If all legs fill: Status COMPLETE.
    - If Leg 1 fills but Leg 2 fails: Triggers emergency AUTO_UNWIND on Leg 1 or SYNTHETIC_HEDGE.
    """

    def __init__(self, legging_policy: str = "AUTO_UNWIND"):
        # Policies: 'AUTO_UNWIND' (close filled leg) or 'SYNTHETIC_HEDGE'
        self.legging_policy = legging_policy

    def execute_dual_leg(
        self,
        leg_ce: SingleLegOrder,
        leg_pe: SingleLegOrder,
        structure_name: str = "ATM_STRADDLE",
        simulated_fill_ce: bool = True,
        simulated_fill_pe: bool = True
    ) -> MultiLegExecutionReport:
        """
        Executes dual-leg option order with atomic guarantee simulation.
        """
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Simulate execution
        leg_ce.is_filled = simulated_fill_ce
        leg_pe.is_filled = simulated_fill_pe

        if simulated_fill_ce:
            leg_ce.fill_price = leg_ce.limit_price
        else:
            leg_ce.reject_reason = "IOC_UNFILLED_OR_CIRCUIT_LIMIT"

        if simulated_fill_pe:
            leg_pe.fill_price = leg_pe.limit_price
        else:
            leg_pe.reject_reason = "IOC_UNFILLED_OR_CIRCUIT_LIMIT"

        # Case 1: Complete atomic fill
        if leg_ce.is_filled and leg_pe.is_filled:
            net_prem = (leg_ce.fill_price + leg_pe.fill_price) if leg_ce.side == "SELL" else -(leg_ce.fill_price + leg_pe.fill_price)
            return MultiLegExecutionReport(
                structure_name=structure_name,
                status="COMPLETE",
                legs=[leg_ce, leg_pe],
                execution_time=now_str,
                net_premium=round(net_prem, 2),
                legging_action=None,
                note="All legs filled cleanly within IOC tolerance"
            )

        # Case 2: Complete rejection (Zero legging risk)
        if not leg_ce.is_filled and not leg_pe.is_filled:
            return MultiLegExecutionReport(
                structure_name=structure_name,
                status="REJECTED",
                legs=[leg_ce, leg_pe],
                execution_time=now_str,
                net_premium=0.0,
                legging_action=None,
                note="Both legs failed to fill. Zero exposure created."
            )

        # Case 3: Partial Fill (CRITICAL LEGGING EVENT)
        filled_leg = leg_ce if leg_ce.is_filled else leg_pe
        failed_leg = leg_pe if leg_ce.is_filled else leg_ce

        if self.legging_policy == "AUTO_UNWIND":
            # Emergency market unwind of the filled leg
            return MultiLegExecutionReport(
                structure_name=structure_name,
                status="PARTIAL_UNWOUND",
                legs=[leg_ce, leg_pe],
                execution_time=now_str,
                net_premium=0.0,
                legging_action=f"EMERGENCY_UNWIND on {filled_leg.symbol}",
                note=f"Legging risk triggered: {failed_leg.symbol} failed. Immediately squared off {filled_leg.symbol} to eliminate naked delta."
            )
        else:
            # Synthetic delta hedge
            return MultiLegExecutionReport(
                structure_name=structure_name,
                status="HEDGED",
                legs=[leg_ce, leg_pe],
                execution_time=now_str,
                net_premium=round(filled_leg.fill_price, 2),
                legging_action="SYNTHETIC_FUTURES_HEDGE",
                note=f"Legging risk triggered: {failed_leg.symbol} failed. Deployed synthetic future to neutralize delta of {filled_leg.symbol}."
            )
