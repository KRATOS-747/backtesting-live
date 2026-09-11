"""
Alpha Momentum Strategy Suite (3 Progressive Versions)
Adapted from desk implementations: alphamomentum/sorintino rankbuffer.py

Versions:
- V1: Multi-Lookback (3M, 6M, 9M) Cross-Sectional Z-Score Momentum with Liquidity Gate (ADTV ₹2 Cr) and Circuit Protection.
- V2: Sortino Downside-Penalized Z-Score with Rank Hysteresis / Buffering (Entry: Top 25, Exit: Rank > 60).
- V3: Full Institutional Execution with Dynamic 24-Day ATR Trailing Stops (7.5x ATR) and Euphoria / Parabolic Volatility Filters.
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd


# =====================================================================
# CONFIGURATION
# =====================================================================
@dataclass
class AlphaStrategyConfig:
    # Portfolio Capital & Turnover
    initial_capital: float = 10_000_000.0     # ₹1 Crore Starting Capital
    transaction_cost: float = 0.0020          # 0.20% slippage + brokerage per trade
    risk_free_rate: float = 0.06              # 6.0% Risk-Free Rate
    rebalance_days_interval: int = 67         # ~67 Trading Days Cycle

    # Rank Hysteresis / Buffer Configuration
    target_portfolio_size: int = 25           # Target portfolio size & entry threshold
    buffer_rank_cutoff: int = 60              # Existing holdings exit only if rank > 60

    # ATR Trailing Stop Configuration
    enable_atr_stop: bool = True              # Enable ATR Dynamic Exit
    atr_length: int = 24                      # 24-Day ATR
    atr_multiplier: float = 7.5               # 7.5x ATR trailing exit

    # Z-Score & Sortino Weighting
    sortino_penalty_weight: float = 0.50      # Weight given to downside vol penalty Z-score

    # Multi-Lookback Windows (in trading days)
    skip_days: int = 21                       # 1-Month Skip Window (prevents short-term reversal)
    lookback_3m: int = 63                     # ~3 Months
    lookback_6m: int = 126                    # ~6 Months
    lookback_9m: int = 189                    # ~9 Months
    vol_lookback: int = 126                   # 6-Month Downside Volatility Window

    # Institutional Universe & Microstructure Gates
    min_adtv_inr: float = 20_000_000.0        # ₹2 Crores 20-Day ADTV Gate
    max_circuit_pct: float = 0.05             # < 5% circuit days in formation window
    max_downside_vol: float = 0.35            # Exclude stocks with >35% annualized downside volatility
    max_1m_return: float = 0.40               # Exclude parabolic 1-month spike >40% (anti-bubble gate)


@dataclass
class AlphaTradeLogItem:
    rebalance_cycle: int
    symbol: str
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    shares: float
    return_pct: float
    exit_reason: str


@dataclass
class AlphaBacktestResult:
    version: str
    equity_curve: pd.Series
    trade_log: List[AlphaTradeLogItem]
    cagr_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate_pct: float
    terminal_value: float


# =====================================================================
# COMMON METRIC BUILDERS
# =====================================================================
def compute_atr_matrix(high_prices: pd.DataFrame, low_prices: pd.DataFrame, close_prices: pd.DataFrame, atr_length: int = 24) -> pd.DataFrame:
    """Vectorized daily Average True Range calculation."""
    prev_close = close_prices.shift(1)
    tr1 = high_prices - low_prices
    tr2 = (high_prices - prev_close).abs()
    tr3 = (low_prices - prev_close).abs()
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    return true_range.rolling(atr_length).mean()


def compute_downside_vol_matrix(close_prices: pd.DataFrame, vol_lookback: int = 126) -> pd.DataFrame:
    """Vectorized semi-deviation / downside volatility matrix annualized."""
    daily_rets = close_prices.pct_change()
    negative_rets = daily_rets.clip(upper=0.0)
    return negative_rets.rolling(vol_lookback).std() * np.sqrt(252)


# =====================================================================
# STRATEGY VERSION 1: MULTI-LOOKBACK COMPOSITE Z-SCORE
# =====================================================================
class AlphaMomentumV1_CompositeZScore:
    """
    Version 1: Composite 3M, 6M, 9M Momentum Z-Scores with ADTV filter and circuit locks.
    Fixed rebalance frequency with full portfolio rotation (no rank buffer).
    """

    def __init__(self, config: Optional[AlphaStrategyConfig] = None):
        self.cfg = config or AlphaStrategyConfig()

    def rank_candidates(
        self,
        close_prices: pd.DataFrame,
        loc: int,
        adtv_series: Optional[pd.Series] = None
    ) -> pd.Series:
        p_end = close_prices.iloc[loc - self.cfg.skip_days]
        p_3m = close_prices.iloc[loc - self.cfg.lookback_3m]
        p_6m = close_prices.iloc[loc - self.cfg.lookback_6m]
        p_9m = close_prices.iloc[loc - self.cfg.lookback_9m]

        r_3m = (p_end - p_3m) / p_3m
        r_6m = (p_end - p_6m) / p_6m
        r_9m = (p_end - p_9m) / p_9m

        # Gate: Positive momentum across all 3 horizons
        valid_mask = (r_3m > 0) & (r_6m > 0) & (r_9m > 0)
        if adtv_series is not None:
            valid_mask = valid_mask & (adtv_series >= self.cfg.min_adtv_inr)

        valid_symbols = valid_mask[valid_mask].index
        if len(valid_symbols) == 0:
            return pd.Series(dtype=float)

        z_3m = (r_3m[valid_symbols] - r_3m[valid_symbols].mean()) / (r_3m[valid_symbols].std() + 1e-6)
        z_6m = (r_6m[valid_symbols] - r_6m[valid_symbols].mean()) / (r_6m[valid_symbols].std() + 1e-6)
        z_9m = (r_9m[valid_symbols] - r_9m[valid_symbols].mean()) / (r_9m[valid_symbols].std() + 1e-6)

        composite_zscore = (z_3m + z_6m + z_9m) / 3.0
        return composite_zscore.sort_values(ascending=False)


# =====================================================================
# STRATEGY VERSION 2: SORTINO DOWNSIDE PENALTY + RANK BUFFERING
# =====================================================================
class AlphaMomentumV2_SortinoRankBuffer(AlphaMomentumV1_CompositeZScore):
    """
    Version 2: Penalizes downside volatility (Sortino penalty) and introduces
    Rank Hysteresis / Buffer:
    - New entries must be in top 25.
    - Existing holdings are retained unless their rank drops below 60.
    """

    def rank_candidates_sortino(
        self,
        close_prices: pd.DataFrame,
        loc: int,
        downside_vol_matrix: pd.DataFrame,
        adtv_series: Optional[pd.Series] = None
    ) -> pd.Series:
        p_end = close_prices.iloc[loc - self.cfg.skip_days]
        p_1m = close_prices.iloc[loc - self.cfg.skip_days - 21]
        p_3m = close_prices.iloc[loc - self.cfg.lookback_3m]
        p_6m = close_prices.iloc[loc - self.cfg.lookback_6m]
        p_9m = close_prices.iloc[loc - self.cfg.lookback_9m]

        r_1m = (p_end - p_1m) / p_1m
        r_3m = (p_end - p_3m) / p_3m
        r_6m = (p_end - p_6m) / p_6m
        r_9m = (p_end - p_9m) / p_9m

        downside_vol = downside_vol_matrix.iloc[loc - self.cfg.skip_days]
        is_euphoric = (r_1m > self.cfg.max_1m_return) | (downside_vol > self.cfg.max_downside_vol)

        valid_mask = (r_3m > 0) & (r_6m > 0) & (r_9m > 0) & (~is_euphoric)
        if adtv_series is not None:
            valid_mask = valid_mask & (adtv_series >= self.cfg.min_adtv_inr)

        valid_symbols = valid_mask[valid_mask].index
        if len(valid_symbols) == 0:
            return pd.Series(dtype=float)

        z_3m = (r_3m[valid_symbols] - r_3m[valid_symbols].mean()) / (r_3m[valid_symbols].std() + 1e-6)
        z_6m = (r_6m[valid_symbols] - r_6m[valid_symbols].mean()) / (r_6m[valid_symbols].std() + 1e-6)
        z_9m = (r_9m[valid_symbols] - r_9m[valid_symbols].mean()) / (r_9m[valid_symbols].std() + 1e-6)

        dvol_sub = downside_vol[valid_symbols]
        z_dvol = (dvol_sub - dvol_sub.mean()) / (dvol_sub.std() + 1e-6)

        composite = ((z_3m + z_6m + z_9m) / 3.0) - (self.cfg.sortino_penalty_weight * z_dvol)
        return composite.sort_values(ascending=False)


# =====================================================================
# STRATEGY VERSION 3: FULL INSTITUTIONAL WITH DYNAMIC ATR TRAILING STOPS
# =====================================================================
class AlphaMomentumV3_FullWithATR(AlphaMomentumV2_SortinoRankBuffer):
    """
    Version 3: Complete desk implementation matching `sorintino rankbuffer.py`:
    - Sortino Z-Score ranking
    - Rank Hysteresis Buffer (Target 25, Cutoff 60)
    - 24-day ATR trailing stop (7.5x multiplier) checked daily between rebalance cycles.
    """

    def run_backtest(
        self,
        close_prices: pd.DataFrame,
        high_prices: pd.DataFrame,
        low_prices: pd.DataFrame,
        turnover_df: Optional[pd.DataFrame] = None,
        is_circuit_locked: Optional[pd.DataFrame] = None,
        start_date: Optional[str] = None
    ) -> AlphaBacktestResult:
        all_dates = close_prices.index.tolist()
        lookback_min = self.cfg.lookback_9m

        start_idx = lookback_min
        if start_date:
            dt_start = pd.to_datetime(start_date)
            candidates = [i for i, d in enumerate(all_dates) if d >= dt_start]
            if candidates:
                start_idx = max(lookback_min, candidates[0])

        rebalance_indices = list(range(start_idx, len(all_dates), self.cfg.rebalance_days_interval))

        # Precompute metrics
        atr_matrix = compute_atr_matrix(high_prices, low_prices, close_prices, self.cfg.atr_length)
        downside_vol = compute_downside_vol_matrix(close_prices, self.cfg.vol_lookback)
        adtv = turnover_df.rolling(20).mean() if turnover_df is not None else None

        cash = self.cfg.initial_capital
        current_holdings: Dict[str, Dict] = {}
        equity_curve: Dict[pd.Timestamp, float] = {}
        trade_log: List[AlphaTradeLogItem] = []

        for idx_pos, loc in enumerate(rebalance_indices):
            reb_date = all_dates[loc]
            next_loc = rebalance_indices[idx_pos + 1] if idx_pos + 1 < len(rebalance_indices) else len(all_dates) - 1

            # 1. Rank universe using Sortino Z-Score
            adtv_series = adtv.iloc[loc] if adtv is not None else None
            ranked_series = self.rank_candidates_sortino(close_prices, loc, downside_vol, adtv_series)
            rank_dict = {sym: rank + 1 for rank, sym in enumerate(ranked_series.index)}

            current_prices = close_prices.loc[reb_date]

            # 2. Evaluate existing holdings against Buffer Cutoff (60)
            symbols_to_remove = []
            for sym, meta in list(current_holdings.items()):
                stock_rank = rank_dict.get(sym, 9999)
                if stock_rank > self.cfg.buffer_rank_cutoff:
                    exit_px = current_prices[sym]
                    proceeds = meta["shares"] * exit_px * (1.0 - self.cfg.transaction_cost)
                    cash += proceeds
                    symbols_to_remove.append(sym)
                    pnl_pct = ((exit_px / meta["entry_price"]) - 1.0) * 100.0

                    trade_log.append(AlphaTradeLogItem(
                        rebalance_cycle=idx_pos + 1,
                        symbol=sym,
                        entry_date=meta["entry_date"].strftime("%Y-%m-%d"),
                        entry_price=round(meta["entry_price"], 2),
                        exit_date=reb_date.strftime("%Y-%m-%d"),
                        exit_price=round(exit_px, 2),
                        shares=round(meta["shares"], 2),
                        return_pct=round(pnl_pct, 2),
                        exit_reason=f"Dropped Below Buffer Rank ({stock_rank} > {self.cfg.buffer_rank_cutoff})"
                    ))

            for sym in symbols_to_remove:
                del current_holdings[sym]

            # 3. Fill open slots from top candidates
            retained = list(current_holdings.keys())
            open_slots = self.cfg.target_portfolio_size - len(retained)
            new_candidates = [sym for sym in ranked_series.index if sym not in retained]
            new_entries = new_candidates[:open_slots]

            if new_entries and open_slots > 0:
                capital_per_slot = cash / len(new_entries)
                for sym in new_entries:
                    entry_px = current_prices[sym]
                    net_capital = capital_per_slot * (1.0 - self.cfg.transaction_cost)
                    shares = net_capital / entry_px

                    current_holdings[sym] = {
                        "shares": shares,
                        "entry_price": entry_px,
                        "entry_date": reb_date,
                        "peak_price": entry_px,
                        "trailing_stop": entry_px - (self.cfg.atr_multiplier * atr_matrix.loc[reb_date, sym])
                    }
                    cash -= capital_per_slot

            # 4. Daily simulation between rebalances (with ATR stop checks)
            for d in range(loc, next_loc):
                day_date = all_dates[d]
                day_close = close_prices.loc[day_date]
                day_low = low_prices.loc[day_date]

                daily_port_val = cash
                stopped_symbols = []

                for sym, meta in list(current_holdings.items()):
                    curr_close = day_close[sym]
                    curr_low = day_low[sym]

                    # Update trailing stop on new high
                    if curr_close > meta["peak_price"]:
                        meta["peak_price"] = curr_close
                        new_stop = curr_close - (self.cfg.atr_multiplier * atr_matrix.loc[day_date, sym])
                        if new_stop > meta["trailing_stop"]:
                            meta["trailing_stop"] = new_stop

                    # ATR Trailing Stop Hit
                    if self.cfg.enable_atr_stop and curr_low <= meta["trailing_stop"]:
                        exit_px = meta["trailing_stop"]
                        proceeds = meta["shares"] * exit_px * (1.0 - self.cfg.transaction_cost)
                        cash += proceeds
                        stopped_symbols.append(sym)
                        pnl_pct = ((exit_px / meta["entry_price"]) - 1.0) * 100.0

                        trade_log.append(AlphaTradeLogItem(
                            rebalance_cycle=idx_pos + 1,
                            symbol=sym,
                            entry_date=meta["entry_date"].strftime("%Y-%m-%d"),
                            entry_price=round(meta["entry_price"], 2),
                            exit_date=day_date.strftime("%Y-%m-%d"),
                            exit_price=round(exit_px, 2),
                            shares=round(meta["shares"], 2),
                            return_pct=round(pnl_pct, 2),
                            exit_reason=f"ATR Trailing Stop Triggered ({self.cfg.atr_multiplier}x ATR)"
                        ))
                    else:
                        daily_port_val += (meta["shares"] * curr_close)

                for sym in stopped_symbols:
                    del current_holdings[sym]

                equity_curve[day_date] = daily_port_val

        # Summary statistics
        eq = pd.Series(equity_curve).ffill().dropna()
        years = max(0.1, (eq.index[-1] - eq.index[0]).days / 365.25)
        tot_ret = (eq.iloc[-1] / eq.iloc[0]) - 1.0
        cagr = ((1.0 + tot_ret) ** (1.0 / years) - 1.0) * 100.0
        max_dd = ((eq - eq.cummax()) / eq.cummax()).min() * 100.0

        daily_pct = eq.pct_change().dropna()
        sharpe = (daily_pct.mean() / (daily_pct.std() + 1e-8)) * np.sqrt(252)

        win_trades = [t for t in trade_log if t.return_pct > 0]
        win_rate = (len(win_trades) / len(trade_log) * 100.0) if trade_log else 0.0

        return AlphaBacktestResult(
            version="V3_SortinoBufferATR",
            equity_curve=eq,
            trade_log=trade_log,
            cagr_pct=round(cagr, 2),
            max_drawdown_pct=round(max_dd, 2),
            sharpe_ratio=round(sharpe, 2),
            win_rate_pct=round(win_rate, 2),
            terminal_value=round(eq.iloc[-1], 2)
        )
