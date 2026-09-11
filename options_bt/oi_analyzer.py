"""
Institutional Open Interest (OI) & Microstructure Sentiment Engine
Analyzes intraday OI dynamics, 4-quadrant buildup classification,
Put-Call Ratio (PCR), and settlement Max Pain mechanics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


# =====================================================================
# DATA MODELS & ENUMS
# =====================================================================
class BuildupType(str, Enum):
    LONG_BUILDUP = "LONG_BUILDUP"       # Price Up, OI Up (Fresh buyers, Bullish)
    SHORT_BUILDUP = "SHORT_BUILDUP"     # Price Down, OI Up (Fresh sellers, Bearish)
    LONG_UNWINDING = "LONG_UNWINDING"   # Price Down, OI Down (Long liquidation, Bearish)
    SHORT_COVERING = "SHORT_COVERING"   # Price Up, OI Down (Short squeeze, Bullish)
    NEUTRAL = "NEUTRAL"                 # Sub-threshold price or OI variation


@dataclass
class BuildupRecord:
    timestamp: str
    strike: int
    option_type: str  # 'CE' or 'PE'
    price: float
    price_change: float
    price_change_pct: float
    oi: float
    oi_change: float
    oi_change_pct: float
    buildup_type: str
    sentiment: str    # 'BULLISH', 'BEARISH', or 'NEUTRAL'


@dataclass
class PCRMetrics:
    oi_pcr: float
    volume_pcr: Optional[float]
    total_call_oi: float
    total_put_oi: float
    total_call_volume: float
    total_put_volume: float
    sentiment: str
    atm_oi_pcr: Optional[float] = None


@dataclass
class MaxPainResult:
    max_pain_strike: int
    spot_price: Optional[float]
    distance_to_spot: Optional[float]
    total_payout_at_max_pain: float
    strike_payouts: Dict[int, float] = field(default_factory=dict)


@dataclass
class StrikeOISummary:
    strike: int
    ce_oi: float
    pe_oi: float
    total_oi: float
    net_oi: float           # pe_oi - ce_oi (>0 means Put heavy / Support)
    strike_pcr: float
    is_call_wall: bool      # Major ceiling resistance
    is_put_wall: bool       # Major floor support


# =====================================================================
# OPEN INTEREST ANALYZER ENGINE
# =====================================================================
class OpenInterestAnalyzer:
    """
    Core engine for derivative chain sentiment, strike concentration,
    and intraday OI velocity tracking.
    """

    @staticmethod
    def classify_buildup(
        price_now: float,
        price_prev: float,
        oi_now: float,
        oi_prev: float,
        price_threshold_pct: float = 0.002,  # 0.2% minimum price delta
        oi_threshold_pct: float = 0.01       # 1.0% minimum OI delta
    ) -> Tuple[str, str]:
        """
        Classifies single-period movement into 4 standard derivatives quadrants:
        Returns: (BuildupType, Sentiment)
        """
        if price_prev <= 0 or oi_prev <= 0:
            return BuildupType.NEUTRAL.value, "NEUTRAL"

        p_pct = (price_now - price_prev) / price_prev
        oi_pct = (oi_now - oi_prev) / oi_prev

        # Dead-band filter
        if abs(p_pct) < price_threshold_pct or abs(oi_pct) < oi_threshold_pct:
            return BuildupType.NEUTRAL.value, "NEUTRAL"

        if p_pct > 0 and oi_pct > 0:
            return BuildupType.LONG_BUILDUP.value, "BULLISH"
        elif p_pct < 0 and oi_pct > 0:
            return BuildupType.SHORT_BUILDUP.value, "BEARISH"
        elif p_pct < 0 and oi_pct < 0:
            return BuildupType.LONG_UNWINDING.value, "BEARISH"
        elif p_pct > 0 and oi_pct < 0:
            return BuildupType.SHORT_COVERING.value, "BULLISH"

        return BuildupType.NEUTRAL.value, "NEUTRAL"

    @classmethod
    def analyze_contract_buildup_series(
        cls,
        contract_df: pd.DataFrame,
        strike: int,
        option_type: str,
        price_col: str = "price",
        oi_col: str = "oi",
        time_col: str = "DateTime",
        min_oi_filter: float = 1000.0
    ) -> List[BuildupRecord]:
        """
        Processes a full intraday time series for an individual contract,
        generating discrete buildup transitions per bar.
        """
        records: List[BuildupRecord] = []
        if contract_df.empty or len(contract_df) < 2:
            return records

        df = contract_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col])

        prices = df[price_col].values
        ois = df[oi_col].values
        times = df[time_col].dt.strftime("%H:%M:%S").values

        for i in range(1, len(df)):
            p_curr = float(prices[i])
            p_prev = float(prices[i - 1])
            oi_curr = float(ois[i])
            oi_prev = float(ois[i - 1])

            if oi_curr < min_oi_filter:
                continue

            b_type, sentiment = cls.classify_buildup(p_curr, p_prev, oi_curr, oi_prev)
            if b_type != BuildupType.NEUTRAL.value:
                p_chg = p_curr - p_prev
                p_chg_pct = (p_chg / p_prev) * 100.0 if p_prev > 0 else 0.0
                oi_chg = oi_curr - oi_prev
                oi_chg_pct = (oi_chg / oi_prev) * 100.0 if oi_prev > 0 else 0.0

                # Invert sentiment if this is a Put contract:
                # E.g. Long buildup in Put = Bearish market bias
                market_sentiment = sentiment
                if option_type.upper() == "PE":
                    if sentiment == "BULLISH":
                        market_sentiment = "BEARISH"
                    elif sentiment == "BEARISH":
                        market_sentiment = "BULLISH"

                records.append(
                    BuildupRecord(
                        timestamp=times[i],
                        strike=strike,
                        option_type=option_type.upper(),
                        price=round(p_curr, 2),
                        price_change=round(p_chg, 2),
                        price_change_pct=round(p_chg_pct, 2),
                        oi=round(oi_curr, 0),
                        oi_change=round(oi_chg, 0),
                        oi_change_pct=round(oi_chg_pct, 2),
                        buildup_type=b_type,
                        sentiment=market_sentiment,
                    )
                )

        return records

    @staticmethod
    def calculate_pcr(
        chain_snapshot_df: pd.DataFrame,
        spot_price: Optional[float] = None,
        strike_range: Optional[float] = 500.0
    ) -> PCRMetrics:
        """
        Calculates Put-Call Ratio (PCR) for total chain and ATM proximity.
        Expects snapshot columns: 'strike', 'ce_oi', 'pe_oi', optional 'ce_volume', 'pe_volume'.
        """
        if chain_snapshot_df.empty:
            return PCRMetrics(
                oi_pcr=1.0,
                volume_pcr=None,
                total_call_oi=0.0,
                total_put_oi=0.0,
                total_call_volume=0.0,
                total_put_volume=0.0,
                sentiment="NEUTRAL"
            )

        df = chain_snapshot_df.copy().fillna(0.0)

        ce_oi = float(df["ce_oi"].sum()) if "ce_oi" in df.columns else 0.0
        pe_oi = float(df["pe_oi"].sum()) if "pe_oi" in df.columns else 0.0

        ce_vol = float(df["ce_volume"].sum()) if "ce_volume" in df.columns else 0.0
        pe_vol = float(df["pe_volume"].sum()) if "pe_volume" in df.columns else 0.0

        oi_pcr = pe_oi / ce_oi if ce_oi > 0 else 1.0
        vol_pcr = (pe_vol / ce_vol) if ce_vol > 0 else None

        # Proximity ATM PCR
        atm_pcr = None
        if spot_price is not None and strike_range is not None and "strike" in df.columns:
            atm_df = df[abs(df["strike"] - spot_price) <= strike_range]
            if not atm_df.empty:
                atm_ce = float(atm_df["ce_oi"].sum())
                atm_pe = float(atm_df["pe_oi"].sum())
                atm_pcr = atm_pe / atm_ce if atm_ce > 0 else 1.0

        # Sentiment interpretation
        # Institutional Indian derivatives thresholds:
        # PCR > 1.4: Extreme Put writing (Bullish support, but nearing overbought)
        # 1.0 <= PCR <= 1.4: Healthy Bullish bias
        # 0.7 <= PCR < 1.0: Bearish bias
        # PCR < 0.7: Extreme Call writing (Bearish ceiling, but nearing oversold)
        if oi_pcr > 1.40:
            sentiment = "OVERBOUGHT_BULLISH_REVERSAL"
        elif oi_pcr >= 1.05:
            sentiment = "BULLISH_BIAS"
        elif oi_pcr >= 0.85:
            sentiment = "NEUTRAL"
        elif oi_pcr >= 0.65:
            sentiment = "BEARISH_BIAS"
        else:
            sentiment = "OVERSOLD_BEARISH_REVERSAL"

        return PCRMetrics(
            oi_pcr=round(oi_pcr, 4),
            volume_pcr=round(vol_pcr, 4) if vol_pcr is not None else None,
            total_call_oi=ce_oi,
            total_put_oi=pe_oi,
            total_call_volume=ce_vol,
            total_put_volume=pe_vol,
            sentiment=sentiment,
            atm_oi_pcr=round(atm_pcr, 4) if atm_pcr is not None else None
        )

    @staticmethod
    def calculate_max_pain(
        chain_snapshot_df: pd.DataFrame,
        spot_price: Optional[float] = None
    ) -> MaxPainResult:
        """
        Calculates Max Pain strike where option buyers lose maximum value
        and option sellers face the minimal aggregate payout liability.
        """
        if chain_snapshot_df.empty or "strike" not in chain_snapshot_df.columns:
            raise ValueError("Invalid chain snapshot for Max Pain calculation")

        df = chain_snapshot_df.dropna(subset=["strike"]).copy().fillna(0.0)
        strikes = df["strike"].astype(int).values
        ce_ois = np.nan_to_num(df["ce_oi"].values, nan=0.0)
        pe_ois = np.nan_to_num(df["pe_oi"].values, nan=0.0)

        strike_payouts: Dict[int, float] = {}

        for S in strikes:
            # Call loss: if settlement is S, calls below S are in the money
            call_loss = np.sum(np.maximum(0, S - strikes) * ce_ois)
            # Put loss: if settlement is S, puts above S are in the money
            put_loss = np.sum(np.maximum(0, strikes - S) * pe_ois)
            strike_payouts[int(S)] = float(call_loss + put_loss)

        if not strike_payouts:
            return MaxPainResult(
                max_pain_strike=0,
                spot_price=spot_price,
                distance_to_spot=None,
                total_payout_at_max_pain=0.0,
                strike_payouts={}
            )

        max_pain_k = min(strike_payouts, key=strike_payouts.get)
        min_payout = strike_payouts[max_pain_k]
        dist = round(spot_price - max_pain_k, 2) if spot_price is not None else None

        return MaxPainResult(
            max_pain_strike=max_pain_k,
            spot_price=spot_price,
            distance_to_spot=dist,
            total_payout_at_max_pain=min_payout,
            strike_payouts=strike_payouts
        )

    @staticmethod
    def compute_oi_velocity(
        contract_df: pd.DataFrame,
        time_col: str = "DateTime",
        oi_col: str = "oi",
        window_bars: int = 5,
        surge_zscore_threshold: float = 2.0
    ) -> pd.DataFrame:
        """
        Calculates continuous rate-of-change (velocity) and acceleration of OI.
        Flags institutional accumulation/unwinding surges via rolling Z-scores.
        """
        if contract_df.empty:
            return pd.DataFrame()

        df = contract_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col])

        # Velocity = 1-bar change in OI (contracts / minute)
        df["oi_diff"] = df[oi_col].diff().fillna(0.0)
        df["oi_velocity"] = df["oi_diff"]

        # Acceleration = change in velocity
        df["oi_acceleration"] = df["oi_velocity"].diff().fillna(0.0)

        # Rolling baseline statistics (computed over preceding bars to prevent outlier contamination)
        prev_vel = df["oi_velocity"].shift(1)
        rolling_mean = prev_vel.rolling(window=window_bars, min_periods=2).mean()
        rolling_std = prev_vel.rolling(window=window_bars, min_periods=2).std()
        
        # Floor standard deviation at 1.0 or 5% of mean to avoid division by near-zero
        floor_std = np.maximum(rolling_mean.abs() * 0.05, 1.0)
        rolling_std = rolling_std.fillna(floor_std).clip(lower=floor_std)

        df["velocity_zscore"] = (df["oi_velocity"] - rolling_mean) / rolling_std
        df["velocity_zscore"] = df["velocity_zscore"].fillna(0.0)

        df["is_surge"] = df["velocity_zscore"].abs() >= surge_zscore_threshold
        df["surge_type"] = np.where(
            df["is_surge"] & (df["oi_velocity"] > 0),
            "INSTITUTIONAL_ACCUMULATION",
            np.where(df["is_surge"] & (df["oi_velocity"] < 0), "INSTITUTIONAL_UNWINDING", "NORMAL")
        )

        return df

    @staticmethod
    def build_strike_oi_matrix(chain_snapshot_df: pd.DataFrame) -> pd.DataFrame:
        """
        Constructs ranked strike distribution identifying the Call Wall (Major Resistance)
        and Put Wall (Major Support) across the entire options surface.
        """
        if chain_snapshot_df.empty or "strike" not in chain_snapshot_df.columns:
            return pd.DataFrame()

        df = chain_snapshot_df.copy().fillna(0.0)
        df = df.sort_values("strike").reset_index(drop=True)

        ce_col = "ce_oi" if "ce_oi" in df.columns else None
        pe_col = "pe_oi" if "pe_oi" in df.columns else None

        if ce_col and pe_col:
            df["total_oi"] = df[ce_col] + df[pe_col]
            df["net_oi"] = df[pe_col] - df[ce_col]  # Positive = Bullish support dominance
            df["strike_pcr"] = np.where(df[ce_col] > 0, df[pe_col] / df[ce_col], np.nan)

            max_ce_oi = df[ce_col].max()
            max_pe_oi = df[pe_col].max()

            df["is_call_wall"] = df[ce_col] == max_ce_oi
            df["is_put_wall"] = df[pe_col] == max_pe_oi
        else:
            df["total_oi"] = 0.0
            df["net_oi"] = 0.0
            df["strike_pcr"] = np.nan
            df["is_call_wall"] = False
            df["is_put_wall"] = False

        return df
