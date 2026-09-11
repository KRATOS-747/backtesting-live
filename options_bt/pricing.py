"""
Vectorized Black-76 Options Pricing, Analytical Greeks, and Implied Volatility Solvers
Tailored for European Index Options traded on NSE (NIFTY, BANKNIFTY, FINNIFTY) and BSE (SENSEX).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple, Union, Optional
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


@dataclass
class GreeksResult:
    price: float
    delta: float
    gamma: float
    theta: float      # 1-day theta decay (₹ / point decay per day)
    vega: float       # Change per 1.0 vol point (1% change in IV)
    rho: float        # Sensitivity to 1% change in risk-free rate
    vanna: float      # d(Delta)/d(Vol) cross Greek
    volga: float      # d(Vega)/d(Vol) cross Greek (Vomma)
    charm: float      # d(Delta)/dt decay of delta over time
    color: float      # d(Gamma)/dt decay of gamma over time
    speed: float      # d(Gamma)/dF 3rd-order price sensitivity


class Black76:
    """
    Standard Black-76 Model for European Options on Futures / Forward Index Price.
    Put-Call Parity: C - P = exp(-r * T) * (F - K)
    """

    @staticmethod
    def d1_d2(F: np.ndarray, K: np.ndarray, T: np.ndarray, sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized computation of d1 and d2."""
        sqrt_T = np.sqrt(np.maximum(T, 1e-6))
        sigma_safe = np.maximum(sigma, 1e-4)
        d1 = (np.log(F / K) + 0.5 * (sigma_safe ** 2) * T) / (sigma_safe * sqrt_T)
        d2 = d1 - sigma_safe * sqrt_T
        return d1, d2

    @classmethod
    def price(
        cls,
        F: Union[float, np.ndarray],
        K: Union[float, np.ndarray],
        T: Union[float, np.ndarray],
        r: float,
        sigma: Union[float, np.ndarray],
        option_type: str = "CE"
    ) -> Union[float, np.ndarray]:
        """
        Computes Black-76 European Option Price.
        
        Args:
            F: Forward / Synthetic Futures price
            K: Strike price
            T: Time to expiry in years (e.g. 1 day = 1/365.0)
            r: Annualized risk-free rate (e.g. 0.065 for 6.5% MIBOR)
            sigma: Annualized implied volatility (e.g. 0.14 for 14%)
            option_type: 'CE' for Call or 'PE' for Put
        """
        is_scalar = np.isscalar(F) and np.isscalar(K) and np.isscalar(T) and np.isscalar(sigma)
        F_arr = np.atleast_1d(np.asarray(F, dtype=float))
        K_arr = np.atleast_1d(np.asarray(K, dtype=float))
        T_arr = np.atleast_1d(np.asarray(T, dtype=float))
        sig_arr = np.atleast_1d(np.asarray(sigma, dtype=float))

        # Handle zero time to expiry (intrinsic value)
        zero_mask = T_arr <= 1e-7
        df = np.exp(-r * T_arr)
        d1, d2 = cls.d1_d2(F_arr, K_arr, T_arr, sig_arr)

        opt_upper = option_type.upper()
        if opt_upper in ("CE", "CALL", "C"):
            prices = df * (F_arr * norm.cdf(d1) - K_arr * norm.cdf(d2))
            intrinsic = np.maximum(F_arr - K_arr, 0.0)
        elif opt_upper in ("PE", "PUT", "P"):
            prices = df * (K_arr * norm.cdf(-d2) - F_arr * norm.cdf(-d1))
            intrinsic = np.maximum(K_arr - F_arr, 0.0)
        else:
            raise ValueError(f"Unknown option_type: {option_type}")

        prices[zero_mask] = intrinsic[zero_mask]
        return float(prices[0]) if is_scalar else prices

    @classmethod
    def greeks(
        cls,
        F: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "CE"
    ) -> GreeksResult:
        """
        Analytical calculation of complete 1st, 2nd, and 3rd-order Greeks suite.
        Theta is expressed as 1 calendar day decay.
        Vega is expressed as change per 1 percentage point change in IV (0.01).
        Rho is expressed as change per 1 percentage point change in interest rate (0.01).
        """
        opt_upper = option_type.upper()
        is_call = opt_upper in ("CE", "CALL", "C")

        if T <= 1e-6:
            intrinsic = max(F - K, 0.0) if is_call else max(K - F, 0.0)
            delta = 1.0 if (is_call and F > K) else (-1.0 if (not is_call and F < K) else 0.0)
            return GreeksResult(
                price=intrinsic, delta=delta, gamma=0.0, theta=0.0, vega=0.0,
                rho=0.0, vanna=0.0, volga=0.0, charm=0.0, color=0.0, speed=0.0
            )

        df = math.exp(-r * T)
        sqrt_T = math.sqrt(T)
        sigma_safe = max(sigma, 1e-4)
        d1 = (math.log(F / K) + 0.5 * (sigma_safe ** 2) * T) / (sigma_safe * sqrt_T)
        d2 = d1 - sigma_safe * sqrt_T

        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_d2 = norm.cdf(d2)

        # 1. Price, Delta, Theta, Rho
        if is_call:
            price = df * (F * cdf_d1 - K * cdf_d2)
            delta = df * cdf_d1
            theta_annual = -(F * df * pdf_d1 * sigma_safe) / (2.0 * sqrt_T) - r * df * (F * cdf_d1 - K * cdf_d2)
            rho = -T * price * 0.01
            charm_annual = df * (pdf_d1 * (r / (sigma_safe * sqrt_T) - d2 / (2.0 * T)) - r * cdf_d1)
        else:
            price = df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
            delta = -df * norm.cdf(-d1)
            theta_annual = -(F * df * pdf_d1 * sigma_safe) / (2.0 * sqrt_T) + r * df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
            rho = -T * price * 0.01
            charm_annual = df * (-pdf_d1 * (r / (sigma_safe * sqrt_T) - d2 / (2.0 * T)) - r * norm.cdf(-d1))

        # 2. Gamma & Vega
        gamma = (df * pdf_d1) / (F * sigma_safe * sqrt_T)
        vega_total = F * df * sqrt_T * pdf_d1
        vega = vega_total * 0.01  # Normalized to 1% IV move
        theta_1day = theta_annual / 365.0
        charm_1day = charm_annual / 365.0

        # 3. Higher-order Cross Greeks
        vanna = (-df * pdf_d1 * (d2 / sigma_safe)) * 0.01
        volga = (vega_total * (d1 * d2 / sigma_safe)) * 0.0001
        speed = -(gamma / F) * (d1 / (sigma_safe * sqrt_T) + 1.0)
        color_annual = -(gamma / (2.0 * T)) * (2.0 * r * T + 1.0 + (d1 * (2.0 * r * T - d2 * sigma_safe * sqrt_T)) / (sigma_safe * sqrt_T))
        color_1day = color_annual / 365.0

        return GreeksResult(
            price=price,
            delta=delta,
            gamma=gamma,
            theta=theta_1day,
            vega=vega,
            rho=rho,
            vanna=vanna,
            volga=volga,
            charm=charm_1day,
            color=color_1day,
            speed=speed
        )

    @classmethod
    def greeks_vector(
        cls,
        F: float,
        strikes: np.ndarray,
        T: float,
        r: float,
        sigmas: np.ndarray,
        option_types: Union[str, List[str]] = "CE"
    ) -> Dict[str, np.ndarray]:
        """
        Fast vectorized computation of all Greeks across an entire options chain simultaneously.
        """
        n = len(strikes)
        types = [option_types] * n if isinstance(option_types, str) else option_types
        res = {
            "price": np.zeros(n), "delta": np.zeros(n), "gamma": np.zeros(n),
            "theta": np.zeros(n), "vega": np.zeros(n), "rho": np.zeros(n),
            "vanna": np.zeros(n), "volga": np.zeros(n)
        }

        for i in range(n):
            g = cls.greeks(F, strikes[i], T, r, sigmas[i], types[i])
            res["price"][i] = g.price
            res["delta"][i] = g.delta
            res["gamma"][i] = g.gamma
            res["theta"][i] = g.theta
            res["vega"][i] = g.vega
            res["rho"][i] = g.rho
            res["vanna"][i] = g.vanna
            res["volga"][i] = g.volga

        return res


class ImpliedVolatilitySolver:
    """
    High-speed vectorized Newton-Raphson solver with Brentq fallback
    for robust IV inversion across wide strikes and fast market conditions.
    """

    @classmethod
    def solve_single(
        cls,
        market_price: float,
        F: float,
        K: float,
        T: float,
        r: float,
        option_type: str = "CE",
        initial_guess: float = 0.20,
        max_iter: int = 25,
        tolerance: float = 1e-4
    ) -> float:
        """Calculates Black-76 implied volatility for a single contract."""
        if T <= 1e-6 or market_price <= 0.05:
            return 0.0

        df = math.exp(-r * T)
        is_call = option_type.upper() in ("CE", "CALL", "C")
        intrinsic = max(df * (F - K), 0.0) if is_call else max(df * (K - F), 0.0)
        
        if market_price < intrinsic:
            return 0.0  # Violation of no-arbitrage bounds

        # Newton-Raphson loop
        sigma = initial_guess
        for _ in range(max_iter):
            price = Black76.price(F, K, T, r, sigma, option_type)
            diff = price - market_price
            if abs(diff) < tolerance:
                return round(sigma, 6)

            # Vega
            sqrt_T = math.sqrt(T)
            d1 = (math.log(F / K) + 0.5 * (sigma ** 2) * T) / (sigma * sqrt_T)
            vega = F * df * sqrt_T * norm.pdf(d1)

            if vega < 1e-6:
                break  # Fall back to root-finding
            sigma = sigma - (diff / vega)
            if sigma <= 0.001 or sigma > 5.0:
                break

        # Robust Brentq fallback if Newton diverges
        def objective(sig: float) -> float:
            return Black76.price(F, K, T, r, sig, option_type) - market_price

        try:
            return brentq(objective, 0.01, 4.0, xtol=tolerance)
        except Exception:
            return np.nan

    @classmethod
    def solve_vector(
        cls,
        market_prices: np.ndarray,
        F: float,
        strikes: np.ndarray,
        T: float,
        r: float,
        option_types: Union[str, np.ndarray] = "CE"
    ) -> np.ndarray:
        """Vectorized wrapper across option chain arrays."""
        n = len(strikes)
        ivs = np.full(n, np.nan)
        types = [option_types] * n if isinstance(option_types, str) else option_types

        for i in range(n):
            ivs[i] = cls.solve_single(
                market_price=market_prices[i],
                F=F,
                K=strikes[i],
                T=T,
                r=r,
                option_type=types[i]
            )
        return ivs


class VolatilitySurfaceMetrics:
    """
    Extracts structural surface signals:
    - 25-Delta Put/Call Skew
    - Smile Curvature
    - Realized vs Implied Volatility (IV-RV Spread)
    """

    @staticmethod
    def parkinson_realized_vol(high: np.ndarray, low: np.ndarray, periods_per_year: float = 252.0 * 375.0) -> float:
        """
        Parkinson High-Low intraday realized volatility estimator.
        More efficient than close-to-close for intraday data.
        """
        valid = (high > 0) & (low > 0) & (high >= low)
        h = high[valid]
        l = low[valid]
        if len(h) < 10:
            return 0.0
        log_hl = np.log(h / l)
        factor = 1.0 / (4.0 * math.log(2.0))
        rv = math.sqrt(factor * np.mean(log_hl ** 2) * periods_per_year)
        return rv

    @staticmethod
    def calculate_skew_25delta(
        atm_iv: float,
        put_25d_iv: float,
        call_25d_iv: float
    ) -> Dict[str, float]:
        """
        Standard institutional 25-delta skew metrics:
        - Absolute Skew: Put IV - Call IV
        - Normalized Skew: (Put IV - Call IV) / ATM IV
        - Smile Butterfly: 0.5 * (Put IV + Call IV) - ATM IV
        """
        abs_skew = put_25d_iv - call_25d_iv
        norm_skew = abs_skew / max(atm_iv, 1e-4)
        butterfly = 0.5 * (put_25d_iv + call_25d_iv) - atm_iv

        return {
            "atm_iv": atm_iv,
            "put_25d_iv": put_25d_iv,
            "call_25d_iv": call_25d_iv,
            "abs_skew_25d": abs_skew,
            "norm_skew_25d": norm_skew,
            "butterfly_25d": butterfly
        }

    @staticmethod
    def garman_klass_realized_vol(
        open_px: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        periods_per_year: float = 252.0 * 375.0
    ) -> float:
        """
        Garman-Klass OHLC realized volatility estimator.
        Incorporates open, high, low, and close prices for up to 8x statistical efficiency.
        """
        valid = (open_px > 0) & (high > 0) & (low > 0) & (close > 0) & (high >= low)
        o, h, l, c = open_px[valid], high[valid], low[valid], close[valid]
        if len(o) < 10:
            return 0.0

        log_hl = np.log(h / l)
        log_co = np.log(c / o)
        term1 = 0.5 * (log_hl ** 2)
        term2 = (2.0 * math.log(2.0) - 1.0) * (log_co ** 2)
        variance = np.mean(term1 - term2) * periods_per_year
        return math.sqrt(max(variance, 0.0))

    @staticmethod
    def yang_zhang_realized_vol(
        open_px: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        periods_per_year: float = 252.0 * 375.0,
        k: float = 0.34
    ) -> float:
        """
        Yang-Zhang volatility estimator: handles overnight jumps and intraday drift.
        Minimum variance unbiased estimator.
        """
        valid = (open_px > 0) & (high > 0) & (low > 0) & (close > 0) & (high >= low)
        o, h, l, c = open_px[valid], high[valid], low[valid], close[valid]
        n = len(o)
        if n < 10:
            return 0.0

        prev_c = np.roll(c, 1)
        prev_c[0] = o[0]

        # Overnight variance
        log_oc = np.log(o / prev_c)
        var_overnight = np.var(log_oc, ddof=1)

        # Open-to-close variance
        log_co = np.log(c / o)
        var_open_to_close = np.var(log_co, ddof=1)

        # Rogers-Satchell variance
        log_ho = np.log(h / o)
        log_lo = np.log(l / o)
        log_hc = np.log(h / c)
        log_lc = np.log(l / c)
        var_rs = np.mean(log_ho * log_hc + log_lo * log_lc)

        # Yang-Zhang linear combination
        total_var = (var_overnight + k * var_open_to_close + (1.0 - k) * var_rs) * periods_per_year
        return math.sqrt(max(total_var, 0.0))

    @staticmethod
    def term_structure_slope(near_iv: float, far_iv: float, days_near: float, days_far: float) -> float:
        """
        Calculates annualized term structure slope between near-week and far-week expiries:
        Slope > 0 indicates Contango (normal), Slope < 0 indicates Backwardation (panic/event).
        """
        dt = (days_far - days_near) / 365.0
        if dt <= 1e-4:
            return 0.0
        return (far_iv - near_iv) / dt
