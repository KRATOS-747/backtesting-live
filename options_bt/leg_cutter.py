"""
Dynamic ROC & Points-Based Independent Leg-Cutting Engine
Manages independent stop-losses, trailing profit locks, and rapid Rate-of-Change (ROC)
exit triggers for option legs (Short option premium decay & Long hedge wings).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class LegState:
    symbol: str
    entry_time: str
    entry_price: float
    side: str = "SHORT"                  # 'SHORT' or 'LONG'
    is_active: bool = True
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_points: float = 0.0
    highest_profit_points: float = 0.0   # High-water mark for trailing stop-loss


@dataclass
class LegCutConfig:
    profit_target_points: Optional[float] = 25.0
    profit_target_pct: Optional[float] = None        # e.g. 0.50 for 50% decay
    static_sl_points: Optional[float] = 40.0
    static_sl_pct: Optional[float] = None           # e.g. 0.35 for 35% SL
    trail_start_points: Optional[float] = None      # Profit barrier to activate trailing SL
    trail_distance_points: Optional[float] = None   # Distance from high-water mark
    roc_window: int = 5                             # Rolling N bars for gamma explosion check
    roc_threshold_pct: float = 0.35                 # Rapid expansion threshold
    min_roc_loss_points: float = 10.0               # Minimum loss required before ROC cut triggers
    hard_cutoff_time: str = "15:15:00"


class LegCutter:
    """
    Monitors intraday price trajectories of option legs independently.
    Protects against 0DTE gamma spikes and runaway delta moves using multi-tier barriers:
    1. Static points / percentage stop loss
    2. Dynamic ROC (Rate of Change) acceleration stop
    3. High-water mark trailing profit lock
    4. Fixed profit target
    5. Hard intraday EOD square-off
    """

    def __init__(self, config: Optional[LegCutConfig] = None):
        self.config = config or LegCutConfig()

    def check_exit(
        self,
        leg: LegState,
        current_time_str: str,
        current_price: float,
        price_history: pd.Series
    ) -> Tuple[bool, Optional[str], float]:
        """
        Evaluates whether an active leg should be exited on the current bar.
        Returns: (should_exit, exit_reason, exit_price)
        """
        if not leg.is_active:
            return False, None, 0.0

        # 1. Hard EOD Square-Off
        if current_time_str >= self.config.hard_cutoff_time:
            return True, "EOD_SQUAREOFF", current_price

        # Direction-aware PnL and Loss
        is_short = (leg.side.upper() == "SHORT")
        if is_short:
            unrealized_points = leg.entry_price - current_price
            loss_points = current_price - leg.entry_price
        else:
            unrealized_points = current_price - leg.entry_price
            loss_points = leg.entry_price - current_price

        # Update high-water mark
        if unrealized_points > leg.highest_profit_points:
            leg.highest_profit_points = unrealized_points

        # 2. Profit Target (Points or Percentage)
        if self.config.profit_target_points is not None and unrealized_points >= self.config.profit_target_points:
            return True, "PROFIT_TARGET", current_price

        if self.config.profit_target_pct is not None and leg.entry_price > 0:
            if (unrealized_points / leg.entry_price) >= self.config.profit_target_pct:
                return True, "PROFIT_TARGET", current_price

        # 3. Static Stop Loss Barrier (Points or Percentage)
        if self.config.static_sl_points is not None and loss_points >= self.config.static_sl_points:
            return True, "STATIC_SL", current_price

        if self.config.static_sl_pct is not None and leg.entry_price > 0:
            if (loss_points / leg.entry_price) >= self.config.static_sl_pct:
                return True, "STATIC_SL", current_price

        # 4. Trailing Stop Loss
        if (
            self.config.trail_start_points is not None
            and self.config.trail_distance_points is not None
            and leg.highest_profit_points >= self.config.trail_start_points
        ):
            trail_stop_profit = leg.highest_profit_points - self.config.trail_distance_points
            if unrealized_points <= trail_stop_profit:
                return True, "TRAILING_SL", current_price

        # 5. Rate-of-Change (ROC) Acceleration Barrier
        # For short legs, detects sudden gamma explosion in premium
        if is_short and len(price_history) >= self.config.roc_window:
            window_slice = price_history.iloc[-self.config.roc_window:]
            min_recent_price = float(window_slice.min())
            if min_recent_price > 0:
                recent_roc = (current_price - min_recent_price) / min_recent_price
                if recent_roc >= self.config.roc_threshold_pct and loss_points > self.config.min_roc_loss_points:
                    return True, "ROC_ACCELERATION_CUT", current_price

        return False, None, 0.0

    def update_leg(
        self,
        leg: LegState,
        current_time_str: str,
        current_price: float,
        price_history: pd.Series
    ) -> LegState:
        """Evaluates and updates the state and PnL of a single leg."""
        should_exit, reason, exit_px = self.check_exit(leg, current_time_str, current_price, price_history)
        if should_exit:
            leg.is_active = False
            leg.exit_time = current_time_str
            leg.exit_price = round(exit_px, 2)
            leg.exit_reason = reason
            
            is_short = (leg.side.upper() == "SHORT")
            pnl = (leg.entry_price - exit_px) if is_short else (exit_px - leg.entry_price)
            leg.pnl_points = round(pnl, 2)
        return leg
