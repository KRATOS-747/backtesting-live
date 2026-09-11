"""
Dynamic Strike Matrix & Multi-Leg Structure Grid Engine
Constructs ATM Straddles, OTM Strangles, Iron Condors, and Iron Flies around intraday spot anchors.
Supports Indian index specifications: NIFTY (50), BANKNIFTY (100), FINNIFTY (25), SENSEX (100).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


# =====================================================================
# DATA MODELS
# =====================================================================
@dataclass
class StrikeProfile:
    strike: int
    option_type: str  # 'CE' or 'PE'
    moneyness: str    # 'ITM2', 'ITM1', 'ATM', 'OTM1', 'OTM2'


@dataclass
class StraddleStrikes:
    atm_strike: int
    ce_strike: int
    pe_strike: int


@dataclass
class StrangleStrikes:
    call_strike: int
    put_strike: int
    width_points: int
    atm_strike: int


@dataclass
class IronCondorStrikes:
    short_put: int
    long_put: int       # Protective Put wing
    short_call: int
    long_call: int      # Protective Call wing
    wing_width: int
    is_iron_fly: bool


# =====================================================================
# STRIKE MATRIX & MULTI-LEG ENGINE
# =====================================================================
class StrikeMatrixEngine:
    """
    Manages strike selection, moneyness mapping, and multi-leg option structures.
    """

    # Standard exchange strike intervals
    INDEX_INTERVALS: Dict[str, float] = {
        "NIFTY": 50.0,
        "BANKNIFTY": 100.0,
        "FINNIFTY": 25.0,
        "MIDCPNIFTY": 25.0,
        "SENSEX": 100.0,
        "BANKEX": 100.0
    }

    def __init__(self, step_size: float = 50.0, index_name: str = "NIFTY"):
        self.index_name = index_name.upper()
        self.step_size = step_size

    @classmethod
    def for_index(cls, index_name: str) -> StrikeMatrixEngine:
        """Factory constructor using exchange preset strike intervals."""
        idx = index_name.upper()
        step = cls.INDEX_INTERVALS.get(idx, 50.0)
        return cls(step_size=step, index_name=idx)

    def get_atm_strike(self, spot_price: float) -> int:
        """Rounds spot price to the nearest strike interval (half-up convention)."""
        return int(np.floor(spot_price / self.step_size + 0.5) * self.step_size)

    def build_straddle_strikes(self, spot_price: float) -> StraddleStrikes:
        """Generates ATM Call and Put strikes for a Straddle."""
        atm = self.get_atm_strike(spot_price)
        return StraddleStrikes(atm_strike=atm, ce_strike=atm, pe_strike=atm)

    def build_strangle_strikes(self, spot_price: float, otm_points: Optional[float] = None, otm_steps: int = 3) -> StrangleStrikes:
        """
        Generates OTM Strangle strikes (e.g. ATM + 150 CE, ATM - 150 PE).
        If otm_points is not provided, defaults to otm_steps * step_size (e.g. 3 * 50 = 150 pts).
        """
        atm = self.get_atm_strike(spot_price)
        offset = int(otm_points if otm_points is not None else (otm_steps * self.step_size))
        # Ensure offset is aligned with strike step
        offset = int(round(offset / self.step_size) * self.step_size)

        call_k = atm + offset
        put_k = atm - offset
        return StrangleStrikes(
            call_strike=call_k,
            put_strike=put_k,
            width_points=offset,
            atm_strike=atm
        )

    def build_iron_condor_strikes(
        self,
        spot_price: float,
        short_offset: Optional[float] = None,
        wing_width: Optional[float] = None
    ) -> IronCondorStrikes:
        """
        Generates 4-leg defined-risk Iron Condor:
        - Short Put: ATM - short_offset
        - Long Put (Wing): Short Put - wing_width
        - Short Call: ATM + short_offset
        - Long Call (Wing): Short Call + wing_width
        """
        atm = self.get_atm_strike(spot_price)
        s_off = int(short_offset if short_offset is not None else (2 * self.step_size))
        w_w = int(wing_width if wing_width is not None else (3 * self.step_size))

        short_p = atm - s_off
        long_p = short_p - w_w
        short_c = atm + s_off
        long_c = short_c + w_w

        return IronCondorStrikes(
            short_put=short_p,
            long_put=long_p,
            short_call=short_c,
            long_call=long_c,
            wing_width=w_w,
            is_iron_fly=False
        )

    def build_iron_fly_strikes(
        self,
        spot_price: float,
        wing_width: Optional[float] = None
    ) -> IronCondorStrikes:
        """
        Generates 4-leg defined-risk Iron Fly (Short ATM Straddle + Long OTM Wings):
        - Short Put: ATM
        - Long Put: ATM - wing_width
        - Short Call: ATM
        - Long Call: ATM + wing_width
        """
        atm = self.get_atm_strike(spot_price)
        w_w = int(wing_width if wing_width is not None else (4 * self.step_size))

        return IronCondorStrikes(
            short_put=atm,
            long_put=atm - w_w,
            short_call=atm,
            long_call=atm + w_w,
            wing_width=w_w,
            is_iron_fly=True
        )

    def build_5strike_matrix(self, spot_price: float) -> List[StrikeProfile]:
        """
        Builds the canonical 5-moneyness matrix around spot (ITM2, ITM1, ATM, OTM1, OTM2).
        """
        atm = self.get_atm_strike(spot_price)
        step = int(self.step_size)

        return [
            # Call side
            StrikeProfile(strike=atm - 2 * step, option_type="CE", moneyness="ITM2"),
            StrikeProfile(strike=atm - step,     option_type="CE", moneyness="ITM1"),
            StrikeProfile(strike=atm,            option_type="CE", moneyness="ATM"),
            StrikeProfile(strike=atm + step,     option_type="CE", moneyness="OTM1"),
            StrikeProfile(strike=atm + 2 * step, option_type="CE", moneyness="OTM2"),

            # Put side
            StrikeProfile(strike=atm + 2 * step, option_type="PE", moneyness="ITM2"),
            StrikeProfile(strike=atm + step,     option_type="PE", moneyness="ITM1"),
            StrikeProfile(strike=atm,            option_type="PE", moneyness="ATM"),
            StrikeProfile(strike=atm - step,     option_type="PE", moneyness="OTM1"),
            StrikeProfile(strike=atm - 2 * step, option_type="PE", moneyness="OTM2"),
        ]

    def categorize_moneyness(self, spot_price: float, strike: int, option_type: str) -> str:
        """Categorizes option contract moneyness relative to spot."""
        diff = strike - spot_price
        opt = option_type.upper()

        if opt in ("CE", "CALL", "C"):
            if abs(diff) <= (self.step_size / 2.0):
                return "ATM"
            elif diff < -self.step_size * 2:
                return "DEEP_ITM"
            elif diff < 0:
                return "ITM"
            elif diff > self.step_size * 2:
                return "DEEP_OTM"
            else:
                return "OTM"
        else:
            if abs(diff) <= (self.step_size / 2.0):
                return "ATM"
            elif diff > self.step_size * 2:
                return "DEEP_ITM"
            elif diff > 0:
                return "ITM"
            elif diff < -self.step_size * 2:
                return "DEEP_OTM"
            else:
                return "OTM"

    @staticmethod
    def find_anchor_spot(spot_df: pd.DataFrame, target_time: str = "09:25:00") -> float:
        """
        Extracts benchmark spot index price at the desk anchor time.
        Expects spot_df with 'date' and 'close' columns.
        """
        if spot_df.empty:
            raise ValueError("Spot DataFrame is empty")

        df = spot_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"])

        target_t = pd.to_datetime(target_time).time()
        matching = df[df["date"].dt.time >= target_t]
        if matching.empty:
            return float(df["close"].iloc[0])

        return float(matching["close"].iloc[0])
