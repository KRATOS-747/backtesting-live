"""
Taylor Series Greek P&L Attribution Engine
Decomposes realized options P&L into Delta, Gamma, Theta, Vega, and Unexplained Residual.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
import numpy as np


@dataclass
class BarAttribution:
    timestamp: str
    total_pnl: float
    delta_pnl: float
    gamma_pnl: float
    theta_pnl: float
    vega_pnl: float
    residual_pnl: float


class GreekPnLAttributor:
    """
    Computes bar-by-bar Taylor series attribution for an options position:
    dP = Delta * dS + 0.5 * Gamma * (dS)^2 + Theta * dt + Vega * dVol + Residual
    """

    @staticmethod
    def attribute_bar(
        delta_pos: float,
        gamma_pos: float,
        theta_pos_daily: float,
        vega_pos_per_vol: float,
        dS: float,
        dVol: float,
        dt_days: float,
        actual_pnl: float,
        timestamp: str = ""
    ) -> BarAttribution:
        """
        Decomposes PnL for a single bar.
        
        Args:
            delta_pos: Net portfolio delta (in underlying units)
            gamma_pos: Net portfolio gamma
            theta_pos_daily: Net portfolio 1-day theta decay
            vega_pos_per_vol: Net portfolio vega per 1.0 vol point (0.01 change)
            dS: Spot price change (S_t - S_{t-1})
            dVol: Implied vol change in percentage points (e.g. +0.5 for +0.5% IV)
            dt_days: Elapsed time in days (e.g. 1 min = 1 / (375 * 252))
            actual_pnl: Net realized change in position mark-to-market
        """
        delta_pnl = delta_pos * dS
        gamma_pnl = 0.5 * gamma_pos * (dS ** 2)
        theta_pnl = theta_pos_daily * dt_days
        vega_pnl = vega_pos_per_vol * dVol

        model_pnl = delta_pnl + gamma_pnl + theta_pnl + vega_pnl
        residual = actual_pnl - model_pnl

        return BarAttribution(
            timestamp=timestamp,
            total_pnl=round(actual_pnl, 2),
            delta_pnl=round(delta_pnl, 2),
            gamma_pnl=round(gamma_pnl, 2),
            theta_pnl=round(theta_pnl, 2),
            vega_pnl=round(vega_pnl, 2),
            residual_pnl=round(residual, 2)
        )

    @staticmethod
    def aggregate_attribution(attributions: List[BarAttribution]) -> Dict[str, float]:
        """Sums attribution across all session bars."""
        if not attributions:
            return {"total_pnl": 0.0, "delta_pnl": 0.0, "gamma_pnl": 0.0, "theta_pnl": 0.0, "vega_pnl": 0.0, "residual_pnl": 0.0}

        df = pd.DataFrame([vars(a) for a in attributions])
        return {
            "total_pnl": round(float(df["total_pnl"].sum()), 2),
            "delta_pnl": round(float(df["delta_pnl"].sum()), 2),
            "gamma_pnl": round(float(df["gamma_pnl"].sum()), 2),
            "theta_pnl": round(float(df["theta_pnl"].sum()), 2),
            "vega_pnl": round(float(df["vega_pnl"].sum()), 2),
            "residual_pnl": round(float(df["residual_pnl"].sum()), 2),
        }
