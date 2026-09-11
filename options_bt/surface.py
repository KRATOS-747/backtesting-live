"""
Implied Volatility Surface & Smile Modeling Engine
Fits parametric and cross-sectional IV surfaces across strikes (moneyness) and time-to-expiry.
Prepares structured 3D surface mesh data for institutional visualizations.
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import polars as pl

from .pricing import Black76, ImpliedVolatilitySolver, VolatilitySurfaceMetrics


# =====================================================================
# DATA MODELS
# =====================================================================
@dataclass
class SmilePoint:
    strike: int
    option_type: str        # 'CE' or 'PE'
    moneyness: float        # K / F
    log_moneyness: float    # ln(K / F)
    market_price: float
    implied_vol: float
    delta: float
    vega: float


@dataclass
class SmileSlice:
    timestamp: str
    time_to_expiry_years: float
    forward_price: float
    atm_strike: int
    atm_iv: float
    skew_25d: float         # 25-Delta Put IV - 25-Delta Call IV
    curvature: float        # Smile convexity parameter (c in a + b*m + c*m^2)
    coefficients: Tuple[float, float, float]  # (a, b, c) in sigma(m) = a + b*m + c*m^2
    points: List[SmilePoint]

    def evaluate_fitted_iv(self, strike: float) -> float:
        """Evaluates parametric smile at arbitrary strike using fitted log-moneyness curve."""
        if self.forward_price <= 0:
            return self.atm_iv
        m = math.log(strike / self.forward_price)
        a, b, c = self.coefficients
        sig = a + (b * m) + (c * (m ** 2))
        return float(np.clip(sig, 0.05, 1.50))


@dataclass
class VolatilitySurfaceMesh:
    timestamps: List[str]
    expiries_days: np.ndarray
    strikes: np.ndarray
    moneyness: np.ndarray
    iv_matrix: np.ndarray   # 2D array: rows = expiries/timestamps, cols = strikes

    def to_dict(self) -> Dict[str, any]:
        """Serializes surface into JSON-compatible dictionary for Plotly 3D visualizers."""
        return {
            "timestamps": self.timestamps,
            "expiries_days": self.expiries_days.tolist(),
            "strikes": self.strikes.tolist(),
            "moneyness": [round(float(m), 4) for m in self.moneyness],
            "iv_matrix": [[round(float(v), 4) for v in row] for row in self.iv_matrix]
        }


# =====================================================================
# VOLATILITY SURFACE & SMILE ENGINE
# =====================================================================
class VolatilitySurfaceEngine:
    """
    Constructs cross-sectional Implied Volatility Smiles and 3D Surfaces from NSE options chain snapshots.
    Uses OTM liquidity rules (OTM Puts for K < F, OTM Calls for K > F) to prevent early-exercise/dividend bias.
    """

    def __init__(self, min_iv: float = 0.05, max_iv: float = 1.50):
        self.min_iv = min_iv
        self.max_iv = max_iv

    def fit_smile_slice(
        self,
        forward_price: float,
        strikes: List[int],
        call_prices: Dict[int, float],
        put_prices: Dict[int, float],
        time_to_expiry_years: float,
        r: float = 0.065,
        timestamp: str = ""
    ) -> SmileSlice:
        """
        Inverts market prices into IVs across strikes and fits parametric polynomial smile:
        sigma(m) = a + b * m + c * m^2 where m = ln(K / F).
        """
        F = forward_price
        T = max(time_to_expiry_years, 1e-5)
        atm_strike = int(round(F / 50.0) * 50)

        points: List[SmilePoint] = []
        log_m_list = []
        iv_list = []
        weights = []

        for K in sorted(strikes):
            m = K / F
            log_m = math.log(m)

            # Use OTM contracts: Puts for K < F, Calls for K >= F
            if K >= F:
                px = call_prices.get(K, 0.0)
                opt_type = "CE"
            else:
                px = put_prices.get(K, 0.0)
                opt_type = "PE"

            if px <= 0.50:
                continue

            iv = ImpliedVolatilitySolver.solve_single(px, F, K, T, r, opt_type)
            if math.isnan(iv) or iv < self.min_iv or iv > self.max_iv:
                continue

            greeks = Black76.greeks(F, K, T, r, iv, opt_type)

            points.append(SmilePoint(
                strike=K,
                option_type=opt_type,
                moneyness=round(m, 4),
                log_moneyness=round(log_m, 4),
                market_price=round(px, 2),
                implied_vol=round(iv, 4),
                delta=round(greeks.delta, 4),
                vega=round(greeks.vega, 4)
            ))

            log_m_list.append(log_m)
            iv_list.append(iv)
            # Vega-weighted least squares (ATM liquid options get highest weight)
            weights.append(max(greeks.vega, 0.05))

        if len(points) < 4:
            # Fallback baseline smile if insufficient liquid quotes
            default_iv = 0.15
            return SmileSlice(
                timestamp=timestamp,
                time_to_expiry_years=T,
                forward_price=F,
                atm_strike=atm_strike,
                atm_iv=default_iv,
                skew_25d=0.0,
                curvature=0.0,
                coefficients=(default_iv, 0.0, 0.0),
                points=points
            )

        # Fit quadratic curve: sigma(m) = a + b*m + c*m^2
        m_arr = np.array(log_m_list)
        iv_arr = np.array(iv_list)
        w_arr = np.array(weights)
        w_norm = w_arr / np.sum(w_arr)

        # Polynomial fit (degree 2)
        poly_coeffs = np.polyfit(m_arr, iv_arr, deg=2, w=w_norm)
        c, b, a = poly_coeffs  # np.polyfit returns [highest_power ... lowest_power]

        atm_iv = float(a)
        curvature = float(c)

        # Calculate 25-Delta Skew: (Put 25Δ IV - Call 25Δ IV)
        put_25d_iv = atm_iv
        call_25d_iv = atm_iv
        for pt in points:
            if pt.option_type == "PE" and abs(pt.delta - (-0.25)) < 0.08:
                put_25d_iv = pt.implied_vol
            elif pt.option_type == "CE" and abs(pt.delta - 0.25) < 0.08:
                call_25d_iv = pt.implied_vol

        skew_25d = round(put_25d_iv - call_25d_iv, 4)

        return SmileSlice(
            timestamp=timestamp,
            time_to_expiry_years=round(T, 5),
            forward_price=round(F, 2),
            atm_strike=atm_strike,
            atm_iv=round(atm_iv, 4),
            skew_25d=skew_25d,
            curvature=round(curvature, 4),
            coefficients=(round(float(a), 5), round(float(b), 5), round(float(c), 5)),
            points=points
        )

    def extract_snapshot_from_file(
        self,
        options_file_path: str,
        time_str: str = "09:25:00",
        spot_price: float = 23750.0,
        time_to_expiry_years: float = 1.0 / 365.0,
        r: float = 0.065
    ) -> Optional[SmileSlice]:
        """
        Extracts complete options chain snapshot from historical 1-minute CSV at a specific minute bar.
        """
        if not os.path.exists(options_file_path):
            return None

        # Load wide options file
        df = pl.read_csv(options_file_path).to_pandas()
        df["DateTime"] = pd.to_datetime(df["DateTime"]).dt.tz_localize(None)

        filtered = df[df["DateTime"].dt.strftime("%H:%M:%S") >= time_str]
        if filtered.empty:
            filtered = df.iloc[-1:]
        row = filtered.iloc[0]

        timestamp = str(row["DateTime"])
        call_prices = {}
        put_prices = {}
        strikes = set()

        for col in df.columns:
            if "_close" in col:
                if "CE_close" in col:
                    s_str = col.split("CE")[0]
                    if s_str.isdigit():
                        s = int(s_str)
                        call_prices[s] = float(row[col])
                        strikes.add(s)
                elif "PE_close" in col:
                    s_str = col.split("PE")[0]
                    if s_str.isdigit():
                        s = int(s_str)
                        put_prices[s] = float(row[col])
                        strikes.add(s)

        # Use ATM Put-Call Parity to deduce forward price F
        atm_strike = int(round(spot_price / 50.0) * 50)
        c_atm = call_prices.get(atm_strike, 0.0)
        p_atm = put_prices.get(atm_strike, 0.0)

        df_rate = math.exp(-r * time_to_expiry_years)
        if c_atm > 0 and p_atm > 0:
            F = atm_strike + ((c_atm - p_atm) / df_rate)
        else:
            F = spot_price

        return self.fit_smile_slice(
            forward_price=F,
            strikes=sorted(list(strikes)),
            call_prices=call_prices,
            put_prices=put_prices,
            time_to_expiry_years=time_to_expiry_years,
            r=r,
            timestamp=timestamp
        )

    def build_surface_from_files(
        self,
        options_file_paths: List[str],
        spot_df: pd.DataFrame,
        time_str: str = "09:25:00",
        r: float = 0.065
    ) -> VolatilitySurfaceMesh:
        """
        Rolls through multiple historical session files and constructs a multi-day 3D Volatility Surface.
        """
        slices: List[SmileSlice] = []
        days_to_expiry_list = []

        spot_df_clean = spot_df.copy()
        spot_df_clean["date_str"] = pd.to_datetime(spot_df_clean["date"]).dt.strftime("%Y-%m-%d")

        for idx, f in enumerate(sorted(options_file_paths)):
            filename = os.path.basename(f)
            date_raw = filename.split("_")[-1].replace(".csv", "")
            formatted_date = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:]}"

            day_spot_rows = spot_df_clean[spot_df_clean["date_str"] == formatted_date]
            if day_spot_rows.empty:
                continue

            spot_px = float(day_spot_rows.iloc[0]["close"])
            days_to_exp = max(0.5, (len(options_file_paths) - idx))  # Synthetic decaying term structure
            T = days_to_exp / 365.0

            smile = self.extract_snapshot_from_file(
                options_file_path=f,
                time_str=time_str,
                spot_price=spot_px,
                time_to_expiry_years=T,
                r=r
            )
            if smile:
                slices.append(smile)
                days_to_expiry_list.append(days_to_exp)

        if not slices:
            raise ValueError("No valid smile slices extracted from provided files.")

        # Determine common strike grid (Moneyness: 0.95 to 1.05)
        ref_spot = slices[0].forward_price
        atm_ref = int(round(ref_spot / 50.0) * 50)
        strike_grid = np.arange(atm_ref - 500, atm_ref + 550, 50)
        moneyness_grid = strike_grid / ref_spot

        iv_matrix = np.zeros((len(slices), len(strike_grid)))
        timestamps = []

        for i, s in enumerate(slices):
            timestamps.append(s.timestamp[:10])
            for j, K in enumerate(strike_grid):
                iv_matrix[i, j] = s.evaluate_fitted_iv(K)

        return VolatilitySurfaceMesh(
            timestamps=timestamps,
            expiries_days=np.array(days_to_expiry_list),
            strikes=strike_grid,
            moneyness=moneyness_grid,
            iv_matrix=iv_matrix
        )

    def detect_calendar_arbitrage(self, mesh: VolatilitySurfaceMesh) -> List[Dict]:
        """
        Checks for total variance monotonicity across term structure:
        w(K, T) = sigma^2 * T must be non-decreasing in T (otherwise calendar spread arbitrage exists).
        """
        arbitrage_events = []
        n_exp, n_strikes = mesh.iv_matrix.shape

        for j in range(n_strikes):
            K = mesh.strikes[j]
            for i in range(n_exp - 1):
                t1 = mesh.expiries_days[i] / 365.0
                t2 = mesh.expiries_days[i + 1] / 365.0
                sig1 = mesh.iv_matrix[i, j]
                sig2 = mesh.iv_matrix[i + 1, j]

                var1 = (sig1 ** 2) * t1
                var2 = (sig2 ** 2) * t2

                if var1 > var2:
                    arbitrage_events.append({
                        "strike": int(K),
                        "near_days": mesh.expiries_days[i],
                        "far_days": mesh.expiries_days[i + 1],
                        "near_iv": round(sig1, 4),
                        "far_iv": round(sig2, 4),
                        "variance_drop": round(var1 - var2, 6)
                    })

        return arbitrage_events
