"""
Inverted Retail Long-Buildup Fade Alpha
Fades retail intraday option buying surges when Price and Open Interest expand simultaneously.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


@dataclass
class OIFaderSignal:
    timestamp: str
    strike: int
    option_type: str
    price: float
    oi: float
    price_change_pct: float
    oi_change_pct: float
    signal: str  # 'FADE_SHORT' or 'NEUTRAL'


class OIFaderEngine:
    """
    Detects retail FOMO buying bursts on 5-minute option bars:
    Condition: Price_t > Price_{t-1} AND OI_t > OI_{t-1} with velocity gate.
    Action: Fade the move by shorting the inflated premium or initiating a credit spread.
    """

    def __init__(
        self,
        min_price_change_pct: float = 0.03,  # Minimum 3% price rise in 5m
        min_oi_change_pct: float = 0.05,     # Minimum 5% OI expansion in 5m
        stop_loss_mult: float = 1.60         # 60% stop loss barrier (hardened from 2.0x)
    ):
        self.min_price_change_pct = min_price_change_pct
        self.min_oi_change_pct = min_oi_change_pct
        self.stop_loss_mult = stop_loss_mult

    def generate_signals(self, resampled_5m_df: pd.DataFrame, strike: int, option_type: str) -> List[OIFaderSignal]:
        """
        Scans resampled 5-minute DataFrame (with 'DateTime', 'price', 'oi') for long buildup fade setups.
        """
        signals: List[OIFaderSignal] = []
        if len(resampled_5m_df) < 3:
            return signals

        df = resampled_5m_df.copy()
        df["price_ret"] = df["price"].pct_change()
        df["oi_ret"] = df["oi"].pct_change()

        for idx, row in df.dropna().iterrows():
            t_str = row["DateTime"].strftime("%H:%M:%S") if isinstance(row["DateTime"], pd.Timestamp) else str(row["DateTime"])
            
            p_ret = row["price_ret"]
            oi_ret = row["oi_ret"]

            # Institutional filter: price rise AND genuine OI buildup (not just 1 contract)
            if p_ret >= self.min_price_change_pct and oi_ret >= self.min_oi_change_pct:
                signals.append(
                    OIFaderSignal(
                        timestamp=t_str,
                        strike=strike,
                        option_type=option_type,
                        price=row["price"],
                        oi=row["oi"],
                        price_change_pct=round(p_ret * 100.0, 2),
                        oi_change_pct=round(oi_ret * 100.0, 2),
                        signal="FADE_SHORT"
                    )
                )

        return signals
