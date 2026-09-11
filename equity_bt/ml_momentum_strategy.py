"""
Machine Learning Momentum Strategy Suite (3 Progressive Versions)
Adapted for Indian Equities (NSE NIFTY 500 / Midcap 150).

Versions:
- V1: Cross-Sectional L2 Ridge Alpha Factor Ranker across 12 proprietary momentum/liquidity/volatility signals.
- V2: Tree-Based Non-Linear Interaction Model (HistGradientBoosting) capturing multi-factor non-linearities.
- V3: Full Institutional Out-of-Sample Walk-Forward Retraining Engine with Inverse-Volatility Risk-Parity
      Position Sizing, Dynamic 3.0x ATR Trailing Stops, Statutory Costs (0.11%), Slippage (0.15%), and STCG Tax (20%).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor


FEATURE_NAMES = [
    "ret_1m",           # 1-Month Return (21 trading days)
    "ret_3m",           # 3-Month Return (63 trading days)
    "ret_6m",           # 6-Month Return (126 trading days)
    "ret_12m_1m",       # 12M Return excluding last month (Jegadeesh-Titman 252 - 21)
    "realized_vol_63",  # 63-Day Realized Annualized Volatility
    "downside_vol_63",  # 63-Day Downside Semi-Deviation
    "amihud_illiq_21",  # 21-Day Amihud Illiquidity Ratio
    "vol_velocity_20",  # Volume Velocity (20D MA / 126D MA volume)
    "dist_52w_high",    # Distance to 52-Week High (Price - Max252) / Max252
    "zscore_20",        # 20-Day Mean Reversion Z-Score (Price - SMA20) / Std20
    "trend_slope_20",   # 20-Day Log Price Linear Regression Slope
    "parkinson_vol_21", # 21-Day High-Low Parkinson Volatility
]


@dataclass
class MLMomentumConfig:
    # Portfolio Structure
    initial_capital: float = 10_000_000.0     # ₹1 Crore Starting Capital
    portfolio_slots: int = 25                # 25 Target Positions
    rebalance_days: int = 21                 # Monthly review cycle (21 trading days)
    forward_horizon_days: int = 21           # Forward return target horizon for ML labels
    train_window_days: int = 504             # 2-Year Rolling Training Window (504 trading days)
    retrain_freq_days: int = 63              # Quarterly Retraining Frequency (63 trading days)
    min_adtv_inr: float = 20_000_000.0       # Minimum ADTV ₹2 Crore (Liquidity Gate)

    # Model Hyperparameters
    ridge_alpha: float = 10.0
    hgb_max_iter: int = 100
    hgb_max_leaf_nodes: int = 15
    hgb_min_samples_leaf: int = 20
    hgb_l2_reg: float = 1.0
    random_state: int = 42

    # Risk Parity Position Sizing (V3)
    enable_risk_parity: bool = True
    max_weight_per_slot: float = 0.06        # Max 6.0% allocation cap per asset
    min_weight_per_slot: float = 0.01        # Min 1.0% allocation floor per asset

    # ATR Trailing Stop
    enable_atr_stop: bool = True
    atr_period: int = 24
    atr_multiplier: float = 3.0

    # Indian Statutory Frictions & Taxes (Finance Act 2024 Schedule)
    statutory_cost_rate: float = 0.0011      # 0.11% blended turnover tax (STT, Stamp Duty, Exchange charges)
    slippage_rate: float = 0.0015            # 0.15% fixed bid-ask spread / market impact
    stcg_tax_rate: float = 0.20              # 20% Annual Short-Term Capital Gains Tax
    risk_free_rate: float = 0.06             # 6.0% baseline yield on unallocated cash


@dataclass
class MLTrade:
    symbol: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    qty: int
    pnl_gross: float
    friction: float
    pnl_net: float
    return_pct: float
    holding_days: int
    exit_reason: str                         # 'REBALANCE', 'ATR_STOP'


@dataclass
class MLSummary:
    strategy_version: str
    initial_capital: float
    final_equity: float
    total_net_pnl: float
    cagr_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    win_rate_pct: float
    total_trades: int
    total_friction_inr: float
    total_stcg_tax_inr: float
    profit_factor: float
    feature_importances: Dict[str, float] = field(default_factory=dict)
    trades: List[MLTrade] = field(default_factory=list)


class MLMomentumStrategy:
    """
    Implements 3 progressive tiers of institutional Machine Learning Momentum:
    - V1: Cross-Sectional L2 Ridge Alpha Factor Ranker across 12 signals
    - V2: Tree-Based Non-Linear Interaction Model (HistGradientBoostingRegressor)
    - V3: Walk-Forward OOS Retraining Engine with Risk-Parity Allocation & Tax Controls
    """

    def __init__(self, config: Optional[MLMomentumConfig] = None):
        self.config = config or MLMomentumConfig()

    @staticmethod
    def calculate_atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 24
    ) -> pd.Series:
        """Vectorized ATR calculation."""
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period, min_periods=period).mean()

    def compute_feature_matrices(
        self,
        prices_df: pd.DataFrame,
        highs_df: pd.DataFrame,
        lows_df: pd.DataFrame,
        volumes_df: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """
        Precomputes the 12 cross-sectional feature matrices across all symbols and dates.
        Returns a dictionary mapping feature name -> pd.DataFrame (index=dates, columns=symbols).
        """
        feats: Dict[str, pd.DataFrame] = {}

        # 1. 1-Month Return (21 trading days)
        feats["ret_1m"] = prices_df.pct_change(21)

        # 2. 3-Month Return (63 trading days)
        feats["ret_3m"] = prices_df.pct_change(63)

        # 3. 6-Month Return (126 trading days)
        feats["ret_6m"] = prices_df.pct_change(126)

        # 4. 12-Month Return excluding last month (Jegadeesh-Titman 252 - 21)
        feats["ret_12m_1m"] = (prices_df.shift(21) - prices_df.shift(252)) / (prices_df.shift(252) + 1e-9)

        # 5. Realized Volatility (63-day rolling daily std annualized)
        daily_ret = prices_df.pct_change()
        feats["realized_vol_63"] = daily_ret.rolling(window=63, min_periods=20).std() * np.sqrt(252)

        # 6. Downside Semi-Deviation (63 days)
        neg_ret = daily_ret.clip(upper=0.0)
        feats["downside_vol_63"] = neg_ret.rolling(window=63, min_periods=20).std() * np.sqrt(252)

        # 7. Amihud Illiquidity Ratio (21-day rolling mean)
        turnover_val = prices_df * volumes_df
        amihud_daily = daily_ret.abs() / (turnover_val + 1e-9)
        feats["amihud_illiq_21"] = amihud_daily.rolling(window=21, min_periods=10).mean()

        # 8. Volume Velocity (20D MA / 126D MA volume)
        vol_20 = volumes_df.rolling(window=20, min_periods=10).mean()
        vol_126 = volumes_df.rolling(window=126, min_periods=30).mean()
        feats["vol_velocity_20"] = vol_20 / (vol_126 + 1e-9)

        # 9. Distance to 52-Week High
        high_52w = prices_df.rolling(window=252, min_periods=60).max()
        feats["dist_52w_high"] = (prices_df - high_52w) / (high_52w + 1e-9)

        # 10. 20-Day Mean Reversion Z-Score
        sma_20 = prices_df.rolling(window=20, min_periods=10).mean()
        std_20 = prices_df.rolling(window=20, min_periods=10).std()
        feats["zscore_20"] = (prices_df - sma_20) / (std_20 + 1e-9)

        # 11. Exponential Trend Slope (20-day log-linear slope via rolling dot product)
        weights = (np.arange(20) - 9.5) / 665.0
        log_px = np.log(prices_df.clip(lower=1e-4))
        feats["trend_slope_20"] = log_px.rolling(window=20, min_periods=20).apply(
            lambda s: np.dot(s, weights), raw=True
        )

        # 12. Parkinson High-Low Volatility (21 days)
        log_hl = np.log((highs_df / (lows_df + 1e-9)).clip(lower=1.0))
        park_var = (log_hl ** 2) / (4.0 * np.log(2.0))
        feats["parkinson_vol_21"] = np.sqrt(park_var.rolling(window=21, min_periods=10).mean()) * np.sqrt(252)

        return feats

    def build_cross_sectional_dataset(
        self,
        feature_dict: Dict[str, pd.DataFrame],
        prices_df: pd.DataFrame,
        start_idx: int,
        end_idx: int,
        step: int = 14,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Builds a cross-sectionally normalized dataset for model training.
        For each sampling date in [start_idx, end_idx - horizon], z-scores features
        across available stocks and computes forward returns as targets.
        """
        fwd_h = self.config.forward_horizon_days
        symbols = prices_df.columns
        feature_keys = FEATURE_NAMES

        x_rows = []
        y_rows = []

        max_sample_idx = end_idx - fwd_h
        sample_indices = list(range(start_idx, max_sample_idx, step))
        if not sample_indices and max_sample_idx > start_idx:
            sample_indices = [start_idx]

        for s_idx in sample_indices:
            # Extract cross-sectional features for all symbols at bar s_idx
            raw_feats = np.column_stack([
                feature_dict[k].iloc[s_idx].values for k in feature_keys
            ])  # Shape: (n_symbols, 12)

            # Compute forward returns: (P_{t+h} - P_t) / P_t
            px_t = prices_df.iloc[s_idx].values
            px_th = prices_df.iloc[s_idx + fwd_h].values
            valid_px = (px_t > 0) & (px_th > 0) & (~np.isnan(px_t)) & (~np.isnan(px_th))
            fwd_ret = np.where(valid_px, (px_th - px_t) / px_t, np.nan)

            # Identify valid non-NaN rows
            valid_mask = ~np.isnan(raw_feats).any(axis=1) & ~np.isnan(fwd_ret)
            if np.sum(valid_mask) < 5:
                continue

            sub_x = raw_feats[valid_mask]
            sub_y = fwd_ret[valid_mask]

            # Cross-sectional z-score standardization
            x_mean = np.mean(sub_x, axis=0)
            x_std = np.std(sub_x, axis=0) + 1e-9
            sub_x_norm = (sub_x - x_mean) / x_std

            # Clip extreme feature outliers (-4.0 to +4.0)
            sub_x_norm = np.clip(sub_x_norm, -4.0, 4.0)

            x_rows.append(sub_x_norm)
            y_rows.append(sub_y)

        if not x_rows:
            return np.empty((0, len(feature_keys))), np.empty((0,))

        return np.vstack(x_rows), np.concatenate(y_rows)

    def extract_predict_matrix(
        self,
        feature_dict: Dict[str, pd.DataFrame],
        t_idx: int,
        symbols: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extracts and cross-sectionally standardizes features at the current test timestamp.
        """
        feature_keys = FEATURE_NAMES
        raw_feats = np.column_stack([
            feature_dict[k].iloc[t_idx].values for k in feature_keys
        ])

        valid_mask = ~np.isnan(raw_feats).any(axis=1)
        valid_symbols = [symbols[i] for i in range(len(symbols)) if valid_mask[i]]

        if np.sum(valid_mask) < 2:
            return np.empty((0, len(feature_keys))), []

        sub_x = raw_feats[valid_mask]
        x_mean = np.mean(sub_x, axis=0)
        x_std = np.std(sub_x, axis=0) + 1e-9
        sub_x_norm = np.clip((sub_x - x_mean) / x_std, -4.0, 4.0)

        return sub_x_norm, valid_symbols

    def compute_risk_parity_weights(
        self,
        selected_symbols: List[str],
        realized_vols: pd.Series,
    ) -> Dict[str, float]:
        """
        Calculates inverse-volatility risk-parity weights subject to min/max concentration constraints.
        """
        if not selected_symbols:
            return {}

        n_assets = len(selected_symbols)
        vols_arr = np.array([max(realized_vols.get(s, 0.20), 0.05) for s in selected_symbols])
        inv_vols = 1.0 / vols_arr
        sum_inv = np.sum(inv_vols)

        if sum_inv <= 0:
            equal_w = 1.0 / n_assets
            return {s: equal_w for s in selected_symbols}

        weights = inv_vols / sum_inv

        # Iterative projection to enforce [min_weight_per_slot, max_weight_per_slot] while summing to 1.0
        min_w = self.config.min_weight_per_slot
        max_w = self.config.max_weight_per_slot

        effective_max = max(max_w, 1.0 / n_assets)
        effective_min = min(min_w, 1.0 / n_assets)

        for _ in range(15):
            weights = np.clip(weights, effective_min, effective_max)
            s = weights.sum()
            if np.isclose(s, 1.0, atol=1e-5):
                break
            unclamped = (weights > effective_min + 1e-6) & (weights < effective_max - 1e-6)
            if not unclamped.any():
                weights = weights / s
                break
            excess = 1.0 - s
            weights[unclamped] += excess * (weights[unclamped] / (weights[unclamped].sum() + 1e-9))

        return dict(zip(selected_symbols, weights.tolist()))


    def run_backtest(
        self,
        prices_df: pd.DataFrame,
        volumes_df: Optional[pd.DataFrame] = None,
        highs_df: Optional[pd.DataFrame] = None,
        lows_df: Optional[pd.DataFrame] = None,
        version: str = "V3",
    ) -> MLSummary:
        """
        Executes ML portfolio backtest.
        prices_df: index=Datetime, columns=Symbols (Daily Close)
        """
        dates = prices_df.index
        symbols = list(prices_df.columns)
        n_dates = len(dates)

        if highs_df is None:
            highs_df = prices_df * 1.01
        if lows_df is None:
            lows_df = prices_df * 0.99
        if volumes_df is None:
            volumes_df = pd.DataFrame(100_000.0, index=dates, columns=symbols)

        # 1. Precompute features and ATR
        feature_dict = self.compute_feature_matrices(prices_df, highs_df, lows_df, volumes_df)

        atr_dict = {}
        for sym in symbols:
            atr_dict[sym] = self.calculate_atr(highs_df[sym], lows_df[sym], prices_df[sym], self.config.atr_period)

        # Determine train and warmup window
        # For production: 504 days training; adapt down if shorter test fixture
        effective_train_window = min(self.config.train_window_days, max(60, int(n_dates * 0.45)))
        warmup_idx = effective_train_window + self.config.forward_horizon_days + 10

        if warmup_idx >= n_dates - 10:
            warmup_idx = min(60, n_dates // 2)
            effective_train_window = max(30, warmup_idx - self.config.forward_horizon_days - 5)

        # Portfolio Tracking State
        cash = self.config.initial_capital
        holdings: Dict[str, Dict] = {}  # sym -> {qty, entry_px, entry_date, entry_idx, stop_px}
        trade_records: List[MLTrade] = []
        equity_curve: List[float] = []

        total_friction_inr = 0.0
        stcg_tax_inr = 0.0
        curr_fy_realized_gains = 0.0

        daily_rf = (1.0 + self.config.risk_free_rate) ** (1.0 / 252.0) - 1.0

        # Model instance and last training timestamp
        current_model = None
        last_trained_idx = -999
        feature_importances: Dict[str, float] = {}

        for t_idx in range(warmup_idx, n_dates):
            current_date = dates[t_idx]
            is_rebalance_bar = (t_idx % self.config.rebalance_days == 0)

            # 1. Update Holdings & Process ATR Trailing Stops (V3)
            exit_symbols = []
            for sym, pos in list(holdings.items()):
                curr_px = prices_df[sym].iloc[t_idx]
                if pd.isna(curr_px) or curr_px <= 0:
                    continue

                if version == "V3" and self.config.enable_atr_stop:
                    atr_val = atr_dict[sym].iloc[t_idx]
                    if pd.isna(atr_val) or atr_val <= 0:
                        atr_val = 0.02 * curr_px
                    new_stop = curr_px - (self.config.atr_multiplier * atr_val)
                    pos["stop_px"] = max(pos["stop_px"], new_stop)

                    if curr_px <= pos["stop_px"]:
                        exit_symbols.append((sym, curr_px, "ATR_STOP"))

            # Process ATR stop exits
            for sym, exit_px, reason in exit_symbols:
                if sym not in holdings:
                    continue
                pos = holdings.pop(sym)
                qty = pos["qty"]
                proceeds = qty * exit_px
                turnover = (qty * pos["entry_px"]) + proceeds
                friction = turnover * (self.config.statutory_cost_rate + self.config.slippage_rate)
                gross_pnl = qty * (exit_px - pos["entry_px"])
                net_pnl = gross_pnl - friction

                cash += (proceeds - friction)
                total_friction_inr += friction
                curr_fy_realized_gains += max(0.0, net_pnl)

                trade_records.append(MLTrade(
                    symbol=sym,
                    entry_date=str(pos["entry_date"]),
                    exit_date=str(current_date),
                    entry_price=pos["entry_px"],
                    exit_price=exit_px,
                    qty=qty,
                    pnl_gross=round(gross_pnl, 2),
                    friction=round(friction, 2),
                    pnl_net=round(net_pnl, 2),
                    return_pct=round((exit_px - pos["entry_px"]) / pos["entry_px"] * 100.0, 2),
                    holding_days=(t_idx - pos["entry_idx"]),
                    exit_reason=reason,
                ))

            # 2. Walk-Forward Model Retraining & Inference on Rebalance Bars
            if is_rebalance_bar or current_model is None:
                # Check if model needs retraining
                need_retrain = (
                    current_model is None or
                    (version == "V3" and (t_idx - last_trained_idx >= self.config.retrain_freq_days))
                )

                if need_retrain:
                    train_start = max(0, t_idx - effective_train_window)
                    train_end = t_idx - self.config.forward_horizon_days  # Prevent data leakage

                    x_train, y_train = self.build_cross_sectional_dataset(
                        feature_dict, prices_df, train_start, train_end, step=14
                    )

                    if len(x_train) >= 30:
                        if version == "V1":
                            model = Ridge(alpha=self.config.ridge_alpha)
                            model.fit(x_train, y_train)
                            current_model = model
                            # Approximate feature importance from normalized coefficients
                            coef_abs = np.abs(model.coef_)
                            coef_sum = np.sum(coef_abs) + 1e-9
                            feature_importances = dict(zip(FEATURE_NAMES, (coef_abs / coef_sum).round(4)))
                        else:
                            # V2 & V3: Non-linear Tree Ensemble
                            model = HistGradientBoostingRegressor(
                                max_iter=self.config.hgb_max_iter,
                                max_leaf_nodes=self.config.hgb_max_leaf_nodes,
                                min_samples_leaf=self.config.hgb_min_samples_leaf,
                                l2_regularization=self.config.hgb_l2_reg,
                                random_state=self.config.random_state,
                            )
                            model.fit(x_train, y_train)
                            current_model = model

                        last_trained_idx = t_idx

                # If we have a trained model, run cross-sectional inference
                if current_model is not None and is_rebalance_bar:
                    x_pred, pred_symbols = self.extract_predict_matrix(feature_dict, t_idx, symbols)

                    if len(x_pred) > 0:
                        predicted_returns = current_model.predict(x_pred)
                        ranked_indices = np.argsort(-predicted_returns)

                        target_picks = [
                            pred_symbols[idx]
                            for idx in ranked_indices[: self.config.portfolio_slots]
                        ]
                        target_picks_set = set(target_picks)

                        # Rebalance: Exit holdings that dropped out of top picks
                        for sym in list(holdings.keys()):
                            if sym not in target_picks_set:
                                pos = holdings.pop(sym)
                                curr_px = prices_df[sym].iloc[t_idx]
                                qty = pos["qty"]
                                proceeds = qty * curr_px
                                turnover = (qty * pos["entry_px"]) + proceeds
                                friction = turnover * (self.config.statutory_cost_rate + self.config.slippage_rate)
                                gross_pnl = qty * (curr_px - pos["entry_px"])
                                net_pnl = gross_pnl - friction

                                cash += (proceeds - friction)
                                total_friction_inr += friction
                                curr_fy_realized_gains += max(0.0, net_pnl)

                                trade_records.append(MLTrade(
                                    symbol=sym,
                                    entry_date=str(pos["entry_date"]),
                                    exit_date=str(current_date),
                                    entry_price=pos["entry_px"],
                                    exit_price=curr_px,
                                    qty=qty,
                                    pnl_gross=round(gross_pnl, 2),
                                    friction=round(friction, 2),
                                    pnl_net=round(net_pnl, 2),
                                    return_pct=round((curr_px - pos["entry_px"]) / pos["entry_px"] * 100.0, 2),
                                    holding_days=(t_idx - pos["entry_idx"]),
                                    exit_reason="REBALANCE",
                                ))

                        # Compute Target Allocations
                        current_equity = cash + sum(
                            h["qty"] * prices_df[s].iloc[t_idx] for s, h in holdings.items()
                        )

                        if version == "V3" and self.config.enable_risk_parity:
                            curr_vols = feature_dict["realized_vol_63"].iloc[t_idx]
                            weight_map = self.compute_risk_parity_weights(target_picks, curr_vols)
                        else:
                            equal_w = 1.0 / len(target_picks) if target_picks else 0.0
                            weight_map = {s: equal_w for s in target_picks}

                        # Allocate to vacant / new target picks
                        new_picks = [s for s in target_picks if s not in holdings]
                        for sym in new_picks:
                            entry_px = prices_df[sym].iloc[t_idx]
                            if pd.isna(entry_px) or entry_px <= 0:
                                continue

                            target_weight = weight_map.get(sym, 1.0 / self.config.portfolio_slots)
                            target_capital = current_equity * target_weight
                            alloc_cash = min(cash, target_capital)

                            if alloc_cash < 1000.0:
                                break

                            qty = int(alloc_cash / entry_px)
                            if qty <= 0:
                                continue

                            cost = qty * entry_px
                            friction = cost * (self.config.statutory_cost_rate + self.config.slippage_rate)

                            if cash >= (cost + friction):
                                cash -= (cost + friction)
                                total_friction_inr += friction

                                atr_val = atr_dict[sym].iloc[t_idx]
                                if pd.isna(atr_val) or atr_val <= 0:
                                    atr_val = 0.02 * entry_px
                                init_stop = entry_px - (self.config.atr_multiplier * atr_val) if version == "V3" else (entry_px * 0.90)

                                holdings[sym] = {
                                    "qty": qty,
                                    "entry_px": entry_px,
                                    "entry_date": current_date,
                                    "entry_idx": t_idx,
                                    "stop_px": init_stop,
                                }

            # Daily cash yield on unallocated cash
            if cash > 0:
                cash *= (1.0 + daily_rf)

            # Year-End STCG Tax Deduction (every 252 days)
            if version == "V3" and (t_idx % 252 == 0) and curr_fy_realized_gains > 0:
                fy_tax = curr_fy_realized_gains * self.config.stcg_tax_rate
                cash -= fy_tax
                stcg_tax_inr += fy_tax
                curr_fy_realized_gains = 0.0

            # Daily Equity Snap
            tot_val = cash + sum(h["qty"] * prices_df[s].iloc[t_idx] for s, h in holdings.items())
            equity_curve.append(tot_val)

        # Final portfolio valuation
        final_equity = equity_curve[-1] if equity_curve else self.config.initial_capital
        total_net_pnl = final_equity - self.config.initial_capital

        # KPI Computations
        sim_days = max(1, n_dates - warmup_idx)
        years = sim_days / 252.0
        cagr = ((final_equity / self.config.initial_capital) ** (1.0 / max(years, 0.1)) - 1.0) * 100.0

        eq_series = pd.Series(equity_curve)
        daily_returns = eq_series.pct_change().dropna()

        rf_daily = (1.0 + self.config.risk_free_rate) ** (1.0 / 252.0) - 1.0
        excess_returns = daily_returns - rf_daily

        sharpe = (excess_returns.mean() / excess_returns.std() * np.sqrt(252)) if excess_returns.std() > 0 else 0.0

        downside_std = daily_returns[daily_returns < 0].std()
        sortino = (excess_returns.mean() / downside_std * np.sqrt(252)) if (downside_std and downside_std > 0) else 0.0

        cummax = eq_series.cummax()
        drawdowns = (eq_series - cummax) / cummax
        max_dd = abs(float(drawdowns.min())) * 100.0 if not drawdowns.empty else 0.0
        calmar = (cagr / max_dd) if max_dd > 0 else 0.0

        n_trades = len(trade_records)
        wins = sum(1 for t in trade_records if t.pnl_net > 0)
        win_rate = (wins / n_trades * 100.0) if n_trades > 0 else 0.0

        gross_wins = sum(t.pnl_net for t in trade_records if t.pnl_net > 0)
        gross_losses = abs(sum(t.pnl_net for t in trade_records if t.pnl_net < 0))
        pf = (gross_wins / gross_losses) if gross_losses > 0 else 999.0

        return MLSummary(
            strategy_version=f"ML_{version}",
            initial_capital=self.config.initial_capital,
            final_equity=round(final_equity, 2),
            total_net_pnl=round(total_net_pnl, 2),
            cagr_pct=round(cagr, 2),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            max_drawdown_pct=round(max_dd, 2),
            calmar_ratio=round(calmar, 2),
            win_rate_pct=round(win_rate, 2),
            total_trades=n_trades,
            total_friction_inr=round(total_friction_inr, 2),
            total_stcg_tax_inr=round(stcg_tax_inr, 2),
            profit_factor=round(pf, 2),
            feature_importances=feature_importances,
            trades=trade_records,
        )
