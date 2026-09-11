"""
Indian Equity Statutory Costs, Execution Slippage, and Tax Engine
Calculates STT, exchange turnover fees, SEBI charges, stamp duty, and STCG tax liability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class EquityTradeCost:
    turnover: float
    stt: float
    exchange_charges: float
    sebi_charges: float
    stamp_duty: float
    gst: float
    total_statutory: float
    slippage_cost: float
    net_cost: float


class EquityCostModel:
    """
    Simulates actual execution frictions and statutory levies for Indian equities:
    - Delivery STT: 0.10% buy & sell
    - Intraday STT: 0.025% sell only
    - Exchange Turnover Fees: 0.00297%
    - SEBI Turnover Fees: ₹10 per crore (0.0001%)
    - Stamp Duty: 0.015% on buy turnover
    - GST: 18% on (brokerage + exchange charges + SEBI charges)
    - Short-Term Capital Gains Tax (STCG): 20% on net realized annual profits
    """

    def __init__(
        self,
        default_statutory_rate: float = 0.0011,  # 0.11% blended statutory rate
        fixed_slippage_rate: float = 0.0015,     # 0.15% half-spread slippage
        stcg_tax_rate: float = 0.20             # 20% STCG
    ):
        self.default_statutory_rate = default_statutory_rate
        self.fixed_slippage_rate = fixed_slippage_rate
        self.stcg_tax_rate = stcg_tax_rate

    def calculate_turnover_friction(self, traded_value: float) -> EquityTradeCost:
        """Calculates total friction per trade."""
        stt = traded_value * 0.0010
        exchange_charges = traded_value * 0.0000297
        sebi_charges = traded_value * 0.0000010
        stamp_duty = traded_value * 0.000075  # Averaged across buy/sell
        gst = (exchange_charges + sebi_charges) * 0.18

        total_statutory = stt + exchange_charges + sebi_charges + stamp_duty + gst
        slippage = traded_value * self.fixed_slippage_rate

        return EquityTradeCost(
            turnover=traded_value,
            stt=round(stt, 2),
            exchange_charges=round(exchange_charges, 2),
            sebi_charges=round(sebi_charges, 2),
            stamp_duty=round(stamp_duty, 2),
            gst=round(gst, 2),
            total_statutory=round(total_statutory, 2),
            slippage_cost=round(slippage, 2),
            net_cost=round(total_statutory + slippage, 2)
        )

    def apply_annual_stcg_tax(self, net_realized_profit: float) -> float:
        """Applies 20% STCG tax on positive net annual profits."""
        if net_realized_profit <= 0:
            return 0.0
        return round(net_realized_profit * self.stcg_tax_rate, 2)
