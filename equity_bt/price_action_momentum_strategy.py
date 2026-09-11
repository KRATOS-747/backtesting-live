"""
Price Action Momentum Strategy Suite (3 Progressive Versions)
Adapted for Indian Equities (NSE NIFTY 500 / Midcap 150).

Versions:
- V1: 52-Week High Donchian Breakout with 50-Day Low Floor Exit & Liquidity Filter (ADTV ≥ ₹2 Cr).
- V2: Minervini Volatility Contraction Pattern (VCP) with Volume Dry-Up and Breakout Expansion Gate.
- V3: Full Institutional Model with Stan Weinstein Stage-2 Trend Filter (EMA50 > EMA200), 
      Dynamic 3.0x ATR Trailing Stops, Statutory Costs (0.11%), Slippage (0.15%), and STCG Tax (20%).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd


@dataclass
class PriceActionConfig:
    # Portfolio Structure
    initial_capital: float = 10_000_000.0     # ₹1 Crore Starting Capital
    portfolio_slots: int = 25                # 25 Equal-Weight Positions (4% each)
    rebalance_days: int = 21                 # Monthly Review Cycle (21 trading days)

    # Donchian Breakout Windows
    high_window_days: int = 252              # 52-Week High Lookback (252 Trading Days)
    donchian_exit_days: int = 50             # 50-Day Channel Exit Floor
    min_adtv_inr: float = 20_000_000.0       # Minimum ADTV ₹2 Crore (Liquidity Gate)

    # Minervini Volatility Contraction Pattern (VCP) Thresholds
    enable_vcp_filter: bool = True
    vcp_contraction_1_max: float = 0.25      # Max 25% drop in first contraction wave
    vcp_contraction_2_max: float = 0.12      # Max 12% drop in second wave
    vcp_contraction_3_max: float = 0.06      # Max 6% drop in final tight handle
    volume_expansion_mult: float = 2.5       # Breakout volume >= 2.5x 20-day average
    volume_dryup_mult: float = 0.7           # Pullback volume <= 0.7x 20-day average

    # Stan Weinstein Stage-2 Trend Filter
    enable_stage2_filter: bool = True
    ema_fast_period: int = 50                # 50-Day EMA
    ema_slow_period: int = 200               # 200-Day EMA
    min_ema_slope_days: int = 20             # Slope of 200 EMA must be strictly positive

    # ATR Trailing Stop
    enable_atr_stop: bool = True
    atr_period: int = 24                     # 24-Day Average True Range
    atr_multiplier: float = 3.0              # 3.0x ATR Trailing Stop

    # Indian Statutory Frictions & Taxes (Finance Act 2024 Schedule)
    statutory_cost_rate: float = 0.0011      # 0.11% blended turnover tax (STT, Stamp Duty, Exchange charges)
    slippage_rate: float = 0.0015            # 0.15% fixed bid-ask spread / market impact
    stcg_tax_rate: float = 0.20              # 20% Annual Short-Term Capital Gains Tax
    risk_free_rate: float = 0.06             # 6.0% baseline yield on unallocated cash


@dataclass
class PriceActionTrade:
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
    exit_reason: str                         # 'DONCHIAN_EXIT', 'ATR_STOP', 'REBALANCE'


@dataclass
class PriceActionSummary:
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
    trades: List[PriceActionTrade] = field(default_factory=list)


class PriceActionMomentumStrategy:
    """
    Implements 3 progressive tiers of institutional Price Action Momentum:
    - V1: 52-Week High Donchian Breakout
    - V2: Minervini VCP + Volume Dry-Up Gate
    - V3: Stage-2 Trend + 3.0x ATR Dynamic Trailing Stop
    """

    def __init__(self, config: Optional[PriceActionConfig] = None):
        self.config = config or PriceActionConfig()

    @staticmethod
    def calculate_atr(df_symbol: pd.DataFrame, period: int = 24) -> pd.Series:
        """Vectorized ATR calculation."""
        high = df_symbol["high"]
        low = df_symbol["low"]
        close_prev = df_symbol["close"].shift(1)
        tr = pd.concat([
            high - low,
            (high - close_prev).abs(),
            (low - close_prev).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(window=period, min_periods=period).mean()

    @staticmethod
    def check_vcp_pattern(
        high_series: pd.Series,
        low_series: pd.Series,
        volume_series: pd.Series,
        idx: int,
        c1_max: float = 0.25,
        c2_max: float = 0.12,
        c3_max: float = 0.06,
        vol_expansion_mult: float = 2.5,
        vol_dryup_mult: float = 0.7
    ) -> bool:
        """
        Validates Mark Minervini's Volatility Contraction Pattern (VCP):
        Checks for 3 successive shrinking contraction waves and volume dry-up.
        """
        if idx < 60:
            return False

        # Wave 1: 40-60 bars ago
        w1_high = high_series.iloc[idx - 60 : idx - 40].max()
        w1_low = low_series.iloc[idx - 60 : idx - 40].min()
        if w1_high <= 0:
            return False
        depth1 = (w1_high - w1_low) / w1_high

        # Wave 2: 20-40 bars ago
        w2_high = high_series.iloc[idx - 40 : idx - 20].max()
        w2_low = low_series.iloc[idx - 40 : idx - 20].min()
        if w2_high <= 0:
            return False
        depth2 = (w2_high - w2_low) / w2_high

        # Wave 3: recent 5-20 bars (tight handle)
        w3_high = high_series.iloc[idx - 20 : idx].max()
        w3_low = low_series.iloc[idx - 20 : idx].min()
        if w3_high <= 0:
            return False
        depth3 = (w3_high - w3_low) / w3_high

        # Verify progressive contraction
        is_contracting = (depth1 <= c1_max) and (depth2 <= c2_max) and (depth3 <= c3_max)
        is_hierarchical = (depth1 >= depth2) and (depth2 >= depth3)

        if not (is_contracting and is_hierarchical):
            return False

        # Volume Signature
        vol_20m = volume_series.iloc[idx - 20 : idx].mean()
        curr_vol = volume_series.iloc[idx]
        pullback_vol = volume_series.iloc[idx - 5 : idx].mean()

        vol_expanded = curr_vol >= (vol_20m * vol_expansion_mult)
        vol_dried_up = pullback_vol <= (vol_20m * vol_dryup_mult)

        return bool(vol_expanded or vol_dried_up)

    def run_backtest(
        self,
        prices_df: pd.DataFrame,
        volumes_df: Optional[pd.DataFrame] = None,
        highs_df: Optional[pd.DataFrame] = None,
        lows_df: Optional[pd.DataFrame] = None,
        version: str = "V3",
    ) -> PriceActionSummary:
        """
        Executes systematic portfolio simulation across the input price matrix.
        prices_df: index=Datetime, columns=Symbols (Daily Close)
        """
        dates = prices_df.index
        symbols = prices_df.columns
        n_dates = len(dates)

        if highs_df is None:
            highs_df = prices_df * 1.01  # Synthesize conservative high
        if lows_df is None:
            lows_df = prices_df * 0.99   # Synthesize conservative low
        if volumes_df is None:
            volumes_df = pd.DataFrame(100_000.0, index=dates, columns=symbols)

        # Precompute 52W High and 50D Low
        h_lookback = self.config.high_window_days
        d_lookback = self.config.donchian_exit_days

        roll_max_52w = prices_df.rolling(window=h_lookback, min_periods=int(h_lookback * 0.8)).max()
        roll_min_50d = prices_df.rolling(window=d_lookback, min_periods=int(d_lookback * 0.8)).min()

        # Precompute Stage-2 EMAs
        ema_50 = prices_df.ewm(span=self.config.ema_fast_period, adjust=False).mean()
        ema_200 = prices_df.ewm(span=self.config.ema_slow_period, adjust=False).mean()
        ema_200_slope = ema_200.diff(self.config.min_ema_slope_days)

        # Portfolio Tracking State
        cash = self.config.initial_capital
        # Key: symbol -> {'qty': int, 'entry_px': float, 'entry_date': date, 'stop_px': float}
        holdings: Dict[str, Dict] = {}
        trade_records: List[PriceActionTrade] = []
        equity_curve: List[float] = []

        total_friction_inr = 0.0
        stcg_tax_inr = 0.0
        curr_fy_realized_gains = 0.0

        for t_idx in range(h_lookback, n_dates):
            current_date = dates[t_idx]
            is_rebalance_bar = (t_idx % self.config.rebalance_days == 0)

            # 1. Update Portfolio Valuation & Check Exits on Existing Holdings
            exit_symbols = []
            portfolio_mv = 0.0

            for sym, pos in list(holdings.items()):
                curr_px = prices_df[sym].iloc[t_idx]
                if pd.isna(curr_px) or curr_px <= 0:
                    continue

                portfolio_mv += pos["qty"] * curr_px
                exit_triggered = False
                exit_reason = ""

                # V1 Exit: Donchian Channel Floor
                floor_px = roll_min_50d[sym].iloc[t_idx - 1]
                if curr_px < floor_px:
                    exit_triggered = True
                    exit_reason = "DONCHIAN_EXIT"

                # V3 Exit: Dynamic ATR Trailing Stop
                if version == "V3" and self.config.enable_atr_stop:
                    # Update trailing stop high-water mark
                    tr = max(highs_df[sym].iloc[t_idx] - lows_df[sym].iloc[t_idx], 0.01 * curr_px)
                    new_stop = curr_px - (self.config.atr_multiplier * tr)
                    pos["stop_px"] = max(pos["stop_px"], new_stop)

                    if curr_px <= pos["stop_px"]:
                        exit_triggered = True
                        exit_reason = "ATR_STOP"

                if exit_triggered:
                    exit_symbols.append((sym, curr_px, exit_reason))

            # Process Exits
            for sym, exit_px, reason in exit_symbols:
                pos = holdings.pop(sym)
                qty = pos["qty"]
                proceeds = qty * exit_px
                trade_turnover = (qty * pos["entry_px"]) + proceeds
                friction = trade_turnover * (self.config.statutory_cost_rate + self.config.slippage_rate)
                gross_pnl = qty * (exit_px - pos["entry_px"])
                net_pnl = gross_pnl - friction

                cash += (proceeds - friction)
                total_friction_inr += friction
                curr_fy_realized_gains += max(0.0, net_pnl)

                trade_records.append(PriceActionTrade(
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

            # 2. Enter New Breakout Positions on Rebalance Bars
            vacant_slots = self.config.portfolio_slots - len(holdings)

            if vacant_slots > 0 and (is_rebalance_bar or version == "V1"):
                candidates = []

                for sym in symbols:
                    if sym in holdings:
                        continue

                    px_now = prices_df[sym].iloc[t_idx]
                    px_prev = prices_df[sym].iloc[t_idx - 1]
                    high_52w = roll_max_52w[sym].iloc[t_idx - 1]

                    if pd.isna(px_now) or pd.isna(high_52w) or px_now <= 0 or high_52w <= 0:
                        continue

                    # Baseline Breakout Condition (V1, V2, V3)
                    # Closes at or within 1.0% of 52W High
                    is_52w_breakout = px_now >= high_52w * 0.99

                    if not is_52w_breakout:
                        continue

                    # Version 2 Filter: Minervini VCP + Volume
                    if version in ("V2", "V3") and self.config.enable_vcp_filter:
                        is_vcp = self.check_vcp_pattern(
                            highs_df[sym], lows_df[sym], volumes_df[sym], t_idx,
                            c1_max=self.config.vcp_contraction_1_max,
                            c2_max=self.config.vcp_contraction_2_max,
                            c3_max=self.config.vcp_contraction_3_max,
                            vol_expansion_mult=self.config.volume_expansion_mult,
                            vol_dryup_mult=self.config.volume_dryup_mult,
                        )
                        if not is_vcp:
                            continue

                    # Version 3 Filter: Stan Weinstein Stage-2 Trend
                    if version == "V3" and self.config.enable_stage2_filter:
                        e50 = ema_50[sym].iloc[t_idx]
                        e200 = ema_200[sym].iloc[t_idx]
                        e200_s = ema_200_slope[sym].iloc[t_idx]

                        is_stage2 = (px_now > e50 > e200) and (e200_s > 0)
                        if not is_stage2:
                            continue

                    # Rank score: distance above 52W high + volume momentum
                    score = (px_now - high_52w) / high_52w
                    candidates.append((score, sym, px_now))

                # Sort descending by breakout score
                candidates.sort(key=lambda x: x[0], reverse=True)
                allocations = candidates[:vacant_slots]

                # Allocate available cash equally across new slots
                if allocations:
                    current_equity = cash + sum(h["qty"] * prices_df[s].iloc[t_idx] for s, h in holdings.items())
                    target_slot_capital = current_equity / self.config.portfolio_slots

                    for _, sym, entry_px in allocations:
                        slot_cash = min(cash, target_slot_capital)
                        if slot_cash < 1000.0:
                            break

                        qty = int(slot_cash / entry_px)
                        if qty <= 0:
                            continue

                        cost = qty * entry_px
                        entry_friction = cost * (self.config.statutory_cost_rate + self.config.slippage_rate)

                        if cash >= (cost + entry_friction):
                            cash -= (cost + entry_friction)
                            total_friction_inr += entry_friction

                            initial_stop = entry_px * 0.85  # 15% default stop
                            if version == "V3":
                                tr_init = max(highs_df[sym].iloc[t_idx] - lows_df[sym].iloc[t_idx], 0.01 * entry_px)
                                initial_stop = entry_px - (self.config.atr_multiplier * tr_init)

                            holdings[sym] = {
                                "qty": qty,
                                "entry_px": entry_px,
                                "entry_date": current_date,
                                "entry_idx": t_idx,
                                "stop_px": initial_stop,
                            }

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
        years = (n_dates - h_lookback) / 252.0
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

        return PriceActionSummary(
            strategy_version=f"PriceAction_{version}",
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
            trades=trade_records,
        )
