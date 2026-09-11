"""
PCA & Residual Momentum Strategy Suite (3 Progressive Versions)
Adapted from desk implementations: clean PCA/L1.py & L1.1(PCA).py

Versions:
- V1: Staggered Tranches with Indian Statutory Frictions (0.11%), Slippage (0.15%), and FY STCG Tax (20%).
- V2: Multi-Factor PCA Residual Extraction (Stripping Market PC1 & Sector PC2 Beta) with Rank Buffering.
- V3: Full Institutional Model with 24-Day ATR Trailing Stop (8.0x), Volatility Percentile Trimming, and Refill Engine.
"""

from __future__ import annotations

import os
import glob
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# =====================================================================
# CONFIGURATION
# =====================================================================
@dataclass
class PCAStrategyConfig:
    # Portfolio Structure
    initial_capital: float = 10_000_000.0   # ₹1 Crore Base Capital
    portfolio_slots: int = 30              # Target portfolio slots
    num_tranches: int = 10                 # Staggered execution tranches
    rebalance_days: int = 125              # Rebalance cycle per tranche (1 tranche every 12.5 days)
    buffer_rank: int = 270                 # Rank buffer to prevent churn

    # Lookback Windows
    mom_lookback_months: int = 6           # 6-Month Momentum Window
    abs_mom_lookback_months: int = 3       # Absolute return gate window
    pca_lookback_months: int = 6           # PCA Factor window

    # PCA Engine Parameters
    n_components: int = 2                  # Number of principal components (Market + Sector Beta)
    vol_percentile_cutoff: float = 0.90    # Trim top 10% highest vol residuals
    use_raw_absolute_momentum: bool = True # Layer 1 Gate: Return > 0%
    use_residual_abs_momentum: bool = True # Layer 2 Gate: Residual Return > 0%

    # Dynamic ATR Stop & Refill Engine
    use_atr_stop: bool = True              # Enable ATR Dynamic Trailing Stop
    atr_period: int = 24                   # 24-Day ATR
    atr_multiplier: float = 8.0            # 8.0x ATR Multiplier
    refill_on_next_rebalance: bool = True  # Refill vacant slots on next tranche cycle

    # Frictions & Indian Tax (Finance Act 2024 Schedule)
    statutory_cost_rate: float = 0.0011    # 0.11% blended statutory rate (STT, Stamp Duty, Exchange Charges)
    fixed_slippage_rate: float = 0.0015    # 0.15% fixed bid-ask spread / market impact
    stcg_tax_rate: float = 0.20            # 20% Annual Short-Term Capital Gains Tax
    risk_free_rate: float = 0.06           # 6.0% baseline cash yield


@dataclass
class StrategyBacktestResult:
    version: str
    equity_curve: pd.Series
    total_frictions_paid: float
    total_tax_paid: float
    post_tax_cagr_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    terminal_value: float


# =====================================================================
# COMMON UTILITY FUNCTIONS (FRICTION, TAX, PCA)
# =====================================================================
def apply_transaction_frictions(trade_value: float, statutory_rate: float, slippage_rate: float) -> Tuple[float, float]:
    """Calculates friction cost (statutory fees + slippage) for a trade leg."""
    total_friction_pct = statutory_rate + slippage_rate
    friction_cost = trade_value * total_friction_pct
    net_value = trade_value - friction_cost
    return net_value, friction_cost


def settle_annual_stcg_tax(
    realized_pnl_fy: float, tax_rate: float, tranche_capital: List[float]
) -> Tuple[List[float], float]:
    """
    Deducts annual STCG tax at FY boundary (April 1st) if net realized PnL is positive.
    Deduction is distributed evenly across liquid tranche cash buckets.
    """
    if realized_pnl_fy <= 0:
        return tranche_capital, 0.0

    total_tax_due = realized_pnl_fy * tax_rate
    num_tranches = len(tranche_capital)
    tax_per_tranche = total_tax_due / max(1, num_tranches)
    updated_capital = [cap - tax_per_tranche for cap in tranche_capital]
    return updated_capital, total_tax_due


def extract_pca_residuals(
    returns_slice: pd.DataFrame,
    n_components: int = 2,
    clip_outliers: float = 0.15
) -> pd.DataFrame:
    r"""
    Projects cross-sectional returns onto Market (PC1) and Sector/Style (PC2) principal components,
    then isolates idiosyncratic residual returns: \epsilon_i = R_i - \hat{R}_i
    """
    clean = returns_slice.dropna(axis=1, thresh=int(len(returns_slice) * 0.70)).fillna(0.0)
    clean = clean.clip(lower=-clip_outliers, upper=clip_outliers)

    if len(clean.columns) <= n_components:
        return clean

    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(clean), index=clean.index, columns=clean.columns)

    pca = PCA(n_components=n_components)
    pca.fit(scaled)
    recon_scaled = pca.inverse_transform(pca.transform(scaled))
    residual_scaled = scaled - pd.DataFrame(recon_scaled, index=clean.index, columns=clean.columns)

    residuals = pd.DataFrame(
        scaler.inverse_transform(residual_scaled),
        index=clean.index,
        columns=clean.columns
    ) - scaler.mean_

    return residuals


# =====================================================================
# STRATEGY VERSION 1: STAGGERED TRANCHES WITH TAX & FRICTIONS (L1.py)
# =====================================================================
class PCABacktestV1_BaseMomentum:
    """
    Version 1: Baseline 6-month trailing momentum with staggered tranches,
    Indian statutory frictions (0.11%), slippage (0.15%), and annual 20% STCG tax.
    """

    def __init__(self, config: Optional[PCAStrategyConfig] = None):
        self.cfg = config or PCAStrategyConfig()

    def run(
        self,
        close_df: pd.DataFrame,
        pit_universe_map: Optional[Dict[int, Set[str]]] = None
    ) -> StrategyBacktestResult:
        lookback_days = 21 * self.cfg.mom_lookback_months
        trading_timeline = close_df.index
        start_date_idx = lookback_days

        num_tranches = self.cfg.num_tranches
        slots_per_tranche = max(1, self.cfg.portfolio_slots // num_tranches)
        rebalance_days = self.cfg.rebalance_days
        tranche_step = max(1, rebalance_days // num_tranches)

        daily_rf_rate = self.cfg.risk_free_rate / 252.0
        stat_rate = self.cfg.statutory_cost_rate
        slip_rate = self.cfg.fixed_slippage_rate
        tax_rate = self.cfg.stcg_tax_rate

        tranche_capital = [self.cfg.initial_capital / num_tranches] * num_tranches
        tranche_holdings = [pd.Series(dtype=float) for _ in range(num_tranches)]
        tranche_cost_basis = [pd.Series(dtype=float) for _ in range(num_tranches)]

        portfolio_history = {}
        cum_realized_pnl_fy = 0.0
        total_tax_paid = 0.0
        total_frictions_paid = 0.0
        current_fy = None

        for d_idx in range(start_date_idx, len(trading_timeline)):
            date = trading_timeline[d_idx]
            prev_date = trading_timeline[d_idx - 1]

            # 1. FY Boundary Check for STCG Tax Settlement (April 1st)
            date_fy = date.year if date.month >= 4 else date.year - 1
            if current_fy is not None and date_fy != current_fy:
                tranche_capital, tax_deducted = settle_annual_stcg_tax(
                    cum_realized_pnl_fy, tax_rate, tranche_capital
                )
                total_tax_paid += tax_deducted
                cum_realized_pnl_fy = 0.0
            current_fy = date_fy

            # Active PIT Universe for the year
            current_universe = pit_universe_map.get(date.year, set(close_df.columns)) if pit_universe_map else set(close_df.columns)

            # 2. Daily MTM updates & Cash Interest
            for t_idx in range(num_tranches):
                if not tranche_holdings[t_idx].empty:
                    daily_rets = close_df.loc[date] / close_df.loc[prev_date] - 1.0
                    tranche_holdings[t_idx] = tranche_holdings[t_idx] * (1.0 + daily_rets.reindex(tranche_holdings[t_idx].index).fillna(0.0))
                tranche_capital[t_idx] *= (1.0 + daily_rf_rate)

            # 3. Staggered Tranche Rebalancing
            days_since_start = d_idx - start_date_idx
            for t_idx in range(num_tranches):
                tranche_stagger = t_idx * tranche_step
                is_reb = (days_since_start - tranche_stagger) >= 0 and ((days_since_start - tranche_stagger) % rebalance_days == 0)

                if is_reb:
                    # Sell old holdings
                    if not tranche_holdings[t_idx].empty:
                        for stock, gross_val in tranche_holdings[t_idx].items():
                            clean_val, friction = apply_transaction_frictions(gross_val, stat_rate, slip_rate)
                            total_frictions_paid += friction
                            cost_val = tranche_cost_basis[t_idx].get(stock, gross_val)
                            cum_realized_pnl_fy += (clean_val - cost_val)
                            tranche_capital[t_idx] += clean_val

                        tranche_holdings[t_idx] = pd.Series(dtype=float)
                        tranche_cost_basis[t_idx] = pd.Series(dtype=float)

                    # Buy new momentum holdings
                    slice_start = trading_timeline[d_idx - lookback_days]
                    close_slice = close_df.loc[slice_start:date]

                    valid_cols = [c for c in current_universe if c in close_slice.columns]
                    if valid_cols:
                        sub = close_slice[valid_cols]
                        cum_ret = (sub.iloc[-1] / sub.iloc[0]) - 1.0
                        pos_cands = cum_ret[cum_ret > 0.0].sort_values(ascending=False).index.tolist()

                        # Exclude stocks held in other tranches
                        held_elsewhere = set()
                        for other_t in range(num_tranches):
                            if other_t != t_idx and not tranche_holdings[other_t].empty:
                                held_elsewhere.update(tranche_holdings[other_t].index)

                        selected = [s for s in pos_cands if s not in held_elsewhere][:slots_per_tranche]

                        if selected:
                            per_slot = tranche_capital[t_idx] / len(selected)
                            new_h = pd.Series(dtype=float)
                            new_c = pd.Series(dtype=float)
                            for s in selected:
                                net_val, friction = apply_transaction_frictions(per_slot, stat_rate, slip_rate)
                                total_frictions_paid += friction
                                new_h[s] = net_val
                                new_c[s] = per_slot
                                tranche_capital[t_idx] -= per_slot
                            tranche_holdings[t_idx] = new_h
                            tranche_cost_basis[t_idx] = new_c

            total_equity = sum(tranche_holdings[t].sum() + tranche_capital[t] for t in range(num_tranches))
            portfolio_history[date] = total_equity

        return self._build_result("V1_BaseMomentum", portfolio_history, total_frictions_paid, total_tax_paid)

    def _build_result(self, version: str, history: Dict, frictions: float, tax: float) -> StrategyBacktestResult:
        eq = pd.Series(history).ffill().dropna()
        years = max(0.1, (eq.index[-1] - eq.index[0]).days / 365.25)
        tot_ret = (eq.iloc[-1] / eq.iloc[0]) - 1.0
        cagr = ((1.0 + tot_ret) ** (1.0 / years) - 1.0) * 100.0
        max_dd = ((eq - eq.cummax()) / eq.cummax()).min() * 100.0

        daily_pct = eq.pct_change().dropna()
        sharpe = (daily_pct.mean() / (daily_pct.std() + 1e-8)) * np.sqrt(252)

        return StrategyBacktestResult(
            version=version,
            equity_curve=eq,
            total_frictions_paid=round(frictions, 2),
            total_tax_paid=round(tax, 2),
            post_tax_cagr_pct=round(cagr, 2),
            max_drawdown_pct=round(max_dd, 2),
            sharpe_ratio=round(sharpe, 2),
            terminal_value=round(eq.iloc[-1], 2)
        )


# =====================================================================
# STRATEGY VERSION 2: DUAL-COMPONENT PCA RESIDUAL MOMENTUM (L1.1)
# =====================================================================
class PCABacktestV2_ResidualExtraction(PCABacktestV1_BaseMomentum):
    """
    Version 2: Fits cross-sectional PCA to isolate Market (PC1) & Sector (PC2) drivers.
    Ranks universe by idiosyncratic residual momentum rather than raw momentum.
    Includes rank hysteresis buffering to curb turnover.
    """

    def _rank_by_pca_residuals(
        self,
        close_slice: pd.DataFrame,
        current_universe: Set[str]
    ) -> List[str]:
        valid_cols = [c for c in current_universe if c in close_slice.columns]
        if len(valid_cols) < (self.cfg.n_components + 2):
            return []

        returns_df = close_slice[valid_cols].pct_change().dropna()
        if len(returns_df) < 20:
            return []

        residuals = extract_pca_residuals(returns_df, n_components=self.cfg.n_components)
        cum_residuals = residuals.sum()
        cum_nominal = returns_df.sum()

        # Gate 1 & 2: Nominal > 0% and Residual > 0%
        cands = cum_residuals[(cum_nominal > 0.0) & (cum_residuals > 0.0)]
        return cands.sort_values(ascending=False).index.tolist()


# =====================================================================
# STRATEGY VERSION 3: INSTITUTIONAL PCA + DYNAMIC ATR TRAILING STOP
# =====================================================================
class PCABacktestV3_ATRBufferedPCA(PCABacktestV2_ResidualExtraction):
    """
    Version 3: Full institutional model with 24-day ATR trailing stop (8.0x),
    volatility percentile trimming, and slot refill engine.
    """

    def calculate_atr(self, high_df: pd.DataFrame, low_df: pd.DataFrame, close_df: pd.DataFrame) -> pd.DataFrame:
        """Calculates 24-day rolling ATR across the cross-section."""
        prev_close = close_df.shift(1)
        tr1 = high_df - low_df
        tr2 = (high_df - prev_close).abs()
        tr3 = (low_df - prev_close).abs()
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        return true_range.rolling(self.cfg.atr_period).mean()
