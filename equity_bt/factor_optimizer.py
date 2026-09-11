"""
Factor Portfolio Optimizer using SciPy
Optimizes capital allocation across factor indices (Multi-Factor Quality, Alpha Low-Vol, Momentum).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import scipy.optimize as sco


@dataclass
class FactorPortfolioWeights:
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float


class FactorOptimizer:
    """
    Solves for optimal factor index weights using convex quadratic programming:
    Max Sharpe Ratio or Min Variance subject to portfolio constraints.
    """

    def __init__(self, returns_df: pd.DataFrame, risk_free_rate: float = 0.065):
        self.returns_df = returns_df.dropna()
        self.risk_free_rate = risk_free_rate
        self.factor_names = list(self.returns_df.columns)
        self.num_factors = len(self.factor_names)

        # Annualized statistics
        self.mean_returns = self.returns_df.mean() * 252.0
        self.cov_matrix = self.returns_df.cov() * 252.0

    def portfolio_stats(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """Calculates expected return, volatility, and Sharpe ratio for given weight vector."""
        ret = np.sum(self.mean_returns * weights)
        vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe = (ret - self.risk_free_rate) / max(vol, 1e-6)
        return ret, vol, sharpe

    def optimize_max_sharpe(self, max_factor_weight: float = 0.45) -> FactorPortfolioWeights:
        """Solves for Maximum Sharpe Ratio portfolio."""
        def neg_sharpe(weights):
            return -self.portfolio_stats(weights)[2]

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = tuple((0.0, max_factor_weight) for _ in range(self.num_factors))
        init_weights = np.full(self.num_factors, 1.0 / self.num_factors)

        opt = sco.minimize(neg_sharpe, init_weights, method="SLSQP", bounds=bounds, constraints=constraints)
        opt_weights = opt.x

        ret, vol, sharpe = self.portfolio_stats(opt_weights)
        weight_dict = {name: round(float(w), 4) for name, w in zip(self.factor_names, opt_weights)}

        return FactorPortfolioWeights(
            weights=weight_dict,
            expected_return=round(ret, 4),
            expected_volatility=round(vol, 4),
            sharpe_ratio=round(sharpe, 4)
        )
