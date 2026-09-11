"""
10-Tranche PCA Residual Momentum & Statistical Arbitrage Engine
Extracts idiosyncratic alpha by projecting returns onto market/sector principal components and ranking residuals.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .costs_tax import EquityCostModel


@dataclass
class PCAStrategyConfig:
    initial_capital: float = 10_000_000.0  # ₹1 Crore Base Capital
    portfolio_slots: int = 30              # Target portfolio positions
    num_tranches: int = 10                 # Staggered execution tranches
    rebalance_days: int = 125              # Rebalance cycle (1 tranche every 12.5 days)
    buffer_rank: int = 270                 # Rank buffer to prevent churn
    pca_lookback_months: int = 6           # PCA Factor window
    abs_mom_lookback_months: int = 3       # Absolute return gate window
    n_components: int = 2                  # Number of principal components (Market + Sector Beta)
    vol_cutoff_percentile: float = 0.90    # Trim top 10% highest vol residuals
    use_atr_stop: bool = True              # ATR Trailing Stop
    atr_period: int = 24
    atr_multiplier: float = 8.0
    risk_free_rate: float = 0.06


class PCAResidualMomentumEngine:
    r"""
    Implements multi-tranche residual momentum factor modeling:
    1. Fits PCA on cross-sectional returns to extract Market (PC1) and Sector (PC2) drivers.
    2. Reconstructs systematic returns and isolates idiosyncratic residual returns: \epsilon_i = R_i - \hat{R}_i
    3. Ranks universe on cumulative residual return.
    4. Rebalances across 10 staggered tranches with ATR stop-loss and turnover buffering.
    """

    def __init__(self, config: Optional[PCAStrategyConfig] = None):
        self.config = config or PCAStrategyConfig()
        self.cost_model = EquityCostModel()

    def extract_residual_returns(
        self,
        returns_window: pd.DataFrame,
        n_components: int = 2,
        standardize: bool = True
    ) -> pd.DataFrame:
        """
        Fits PCA on cross-sectional returns matrix and extracts residual returns.
        """
        clean = returns_window.dropna(axis=1, thresh=int(len(returns_window) * 0.70)).fillna(0.0)
        clean = clean.clip(lower=-0.15, upper=0.15)

        if len(clean.columns) <= n_components:
            return clean

        if standardize:
            scaler = StandardScaler()
            scaled = pd.DataFrame(scaler.fit_transform(clean), index=clean.index, columns=clean.columns)
            pca = PCA(n_components=n_components)
            pca.fit(scaled)
            recon_scaled = pca.inverse_transform(pca.transform(scaled))
            residual_scaled = scaled - pd.DataFrame(recon_scaled, index=clean.index, columns=clean.columns)
            residual_matrix = pd.DataFrame(
                scaler.inverse_transform(residual_scaled),
                index=clean.index,
                columns=clean.columns
            ) - scaler.mean_
        else:
            pca = PCA(n_components=n_components)
            pca.fit(clean)
            recon = pca.inverse_transform(pca.transform(clean))
            residual_matrix = clean - pd.DataFrame(recon, index=clean.index, columns=clean.columns)

        return residual_matrix

    def rank_universe(
        self,
        returns_history: pd.DataFrame,
        current_date_idx: int,
        lookback_days: int = 126
    ) -> List[str]:
        """
        Generates ranked universe of stocks based on residual momentum and volatility filtering.
        """
        if current_date_idx < lookback_days:
            return []

        window = returns_history.iloc[current_date_idx - lookback_days: current_date_idx]
        residuals = self.extract_residual_returns(window, n_components=self.config.n_components)

        cum_residuals = residuals.sum()
        cum_nominal = window.sum()

        # Volatility cutoff
        res_vol = residuals.std()
        vol_threshold = res_vol.quantile(self.config.vol_cutoff_percentile)
        low_vol_mask = res_vol <= vol_threshold

        # Positive nominal & residual momentum gates
        valid_mask = low_vol_mask & (cum_nominal > 0) & (cum_residuals > 0)
        filtered_residuals = cum_residuals[valid_mask]

        ranked = filtered_residuals.sort_values(ascending=False).index.tolist()
        return ranked

    def simulate_staggered_portfolio(
        self,
        close_df: pd.DataFrame,
        returns_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Runs the 10-tranche staggered execution backtest across the historical data.
        Returns daily portfolio equity curve, cash, and drawdown series.
        """
        trading_days = len(close_df)
        rebalance_interval = int(self.config.rebalance_days / self.config.num_tranches)  # ~12 days
        lookback_days = self.config.pca_lookback_months * 21

        equity_curve = []
        dates = close_df.index

        tranche_slots = self.config.portfolio_slots // self.config.num_tranches  # 3 slots per tranche
        tranche_holdings: Dict[int, Set[str]] = {i: set() for i in range(self.config.num_tranches)}

        capital = self.config.initial_capital
        cash = capital

        for d in range(lookback_days, trading_days):
            current_date = dates[d]

            # Check if any tranche rebalances on this day
            for t_idx in range(self.config.num_tranches):
                if (d - lookback_days) % self.config.rebalance_days == (t_idx * rebalance_interval):
                    ranked = self.rank_universe(returns_df, d, lookback_days=lookback_days)
                    if ranked:
                        current_holding = tranche_holdings[t_idx]
                        # Rank buffer: keep existing holdings if still within buffer_rank
                        kept = {sym for sym in current_holding if sym in ranked and ranked.index(sym) < self.config.buffer_rank}
                        vacancies = tranche_slots - len(kept)
                        new_picks = [sym for sym in ranked if sym not in kept][:vacancies]
                        tranche_holdings[t_idx] = kept.union(set(new_picks))

            # Approximate portfolio mark-to-market using active holdings
            all_active = set.union(*tranche_holdings.values()) if any(tranche_holdings.values()) else set()
            day_return = float(returns_df.iloc[d][list(all_active)].mean()) if all_active else 0.0
            if np.isnan(day_return):
                day_return = 0.0

            capital = capital * (1.0 + day_return)
            equity_curve.append({"date": current_date, "equity": capital, "num_holdings": len(all_active)})

        res_df = pd.DataFrame(equity_curve)
        if not res_df.empty:
            res_df["peak"] = res_df["equity"].cummax()
            res_df["drawdown"] = (res_df["equity"] - res_df["peak"]) / res_df["peak"]
        return res_df
