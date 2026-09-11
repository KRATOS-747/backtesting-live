"""
Indicator Momentum Strategy Suite (3 Progressive Versions)
Adapted for Indian Equities (NSE NIFTY 500 / Midcap 150).

Versions:
- V1: Oscillator Momentum Baseline (RSI 14 in 60-75 Power Zone + Expanding MACD Histogram).
- V2: Multi-Horizon Composite RSI (9D, 14D, 21D) + Bollinger Bandwidth Squeeze Expansion + ADX Trend Gate.
- V3: Full Institutional Model with NIFTY 200-EMA Regime Gate, Parabolic Euphoria 50% Trimming (RSI > 82),
      Dynamic 3.0x ATR Trailing Stops, Statutory Costs (0.11%), Slippage (0.15%), and STCG Tax (20%).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd


@dataclass
class IndicatorConfig:
    # Portfolio Structure
    initial_capital: float = 10_000_000.0     # ₹1 Crore Starting Capital
    portfolio_slots: int = 25                # 25 Equal-Weight Positions (4% each)
    rebalance_days: int = 14                 # Bi-weekly review cycle (14 trading days)
    min_adtv_inr: float = 20_000_000.0       # Minimum ADTV ₹2 Crore (Liquidity Gate)

    # RSI Parameters
    rsi_fast_period: int = 9                 # 9-Day Fast RSI
    rsi_mid_period: int = 14                 # 14-Day Standard RSI
    rsi_slow_period: int = 21                # 21-Day Slow RSI
    rsi_min_entry: float = 60.0              # Momentum power-zone floor
    rsi_max_entry: float = 75.0              # Upper limit for entry (avoid buying extreme tops)
    rsi_exit_level: float = 50.0             # Exit when momentum drops below midline
    rsi_euphoria_trim: float = 82.0          # Parabolic euphoria trim threshold

    # MACD Parameters
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    macd_hist_expansion_bars: int = 3        # Number of consecutive expanding bars required

    # Bollinger Bands & Squeeze Parameters
    bb_period: int = 20
    bb_std: float = 2.0
    bb_squeeze_pctile: float = 0.20          # 20th percentile bandwidth defines squeeze
    bb_squeeze_lookback: int = 126           # 6-month lookback for bandwidth distribution

    # ADX Trend Strength Parameters
    adx_period: int = 14
    adx_threshold: float = 25.0              # Trend strength threshold

    # Rank Hysteresis
    top_n_enter: int = 25                    # Enter top N by momentum score
    exit_rank_threshold: int = 60            # Hold existing positions until rank falls below this

    # Benchmark Regime Gate
    enable_regime_filter: bool = True
    benchmark_ema_period: int = 200          # 200-Day EMA on Benchmark / NIFTY 500

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
class IndicatorTrade:
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
    exit_reason: str                         # 'RSI_DROP', 'MACD_REVERSAL', 'RANK_DROP', 'ATR_STOP', 'REGIME_CASH', 'EUPHORIA_TRIM'


@dataclass
class IndicatorSummary:
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
    trades: List[IndicatorTrade] = field(default_factory=list)


class IndicatorMomentumStrategy:
    """
    Implements 3 progressive tiers of institutional Indicator Momentum:
    - V1: Oscillator Momentum Baseline (RSI 14 in 60-75 + Expanding MACD Histogram)
    - V2: Multi-Horizon Composite RSI + Bollinger Squeeze Expansion + ADX Trend Gate
    - V3: Full Institutional Model with Regime Gate, Euphoria Trim, 3.0x ATR Stop, and STCG Tax
    """

    def __init__(self, config: Optional[IndicatorConfig] = None):
        self.config = config or IndicatorConfig()

    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Vectorized Wilder's RSI calculation."""
        delta = series.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        alpha = 1.0 / period
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def calculate_macd(
        series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculates MACD Line, Signal Line, and MACD Histogram."""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        macd_hist = macd_line - signal_line
        return macd_line, signal_line, macd_hist

    @staticmethod
    def calculate_bollinger_bands(
        series: pd.Series, period: int = 20, num_std: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Calculates Upper Band, Middle Band, Lower Band, and Bandwidth."""
        middle = series.rolling(window=period, min_periods=period).mean()
        std = series.rolling(window=period, min_periods=period).std()
        upper = middle + (num_std * std)
        lower = middle - (num_std * std)
        bandwidth = (upper - lower) / (middle + 1e-9)
        return upper, middle, lower, bandwidth

    @staticmethod
    def calculate_adx(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculates ADX, +DI, and -DI with exponential smoothing."""
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        alpha = 1.0 / period
        tr_smooth = pd.Series(tr, index=close.index).ewm(alpha=alpha, adjust=False).mean()
        plus_dm_smooth = pd.Series(plus_dm, index=close.index).ewm(alpha=alpha, adjust=False).mean()
        minus_dm_smooth = pd.Series(minus_dm, index=close.index).ewm(alpha=alpha, adjust=False).mean()

        plus_di = 100.0 * (plus_dm_smooth / (tr_smooth + 1e-9))
        minus_di = 100.0 * (minus_dm_smooth / (tr_smooth + 1e-9))

        dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9))
        adx = dx.ewm(alpha=alpha, adjust=False).mean()
        return adx, plus_di, minus_di

    @staticmethod
    def calculate_atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 24
    ) -> pd.Series:
        """Calculates Average True Range (ATR)."""
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period, min_periods=period).mean()

    def run_backtest(
        self,
        prices_df: pd.DataFrame,
        volumes_df: Optional[pd.DataFrame] = None,
        highs_df: Optional[pd.DataFrame] = None,
        lows_df: Optional[pd.DataFrame] = None,
        benchmark_prices: Optional[pd.Series] = None,
        version: str = "V3",
    ) -> IndicatorSummary:
        """
        Executes systematic portfolio simulation across the input price matrix.
        prices_df: index=Datetime, columns=Symbols (Daily Close)
        """
        dates = prices_df.index
        symbols = prices_df.columns
        n_dates = len(dates)

        if highs_df is None:
            highs_df = prices_df * 1.01
        if lows_df is None:
            lows_df = prices_df * 0.99
        if volumes_df is None:
            volumes_df = pd.DataFrame(100_000.0, index=dates, columns=symbols)

        # Benchmark setup (for V3 regime gating)
        if benchmark_prices is None:
            benchmark_prices = prices_df.mean(axis=1)
        benchmark_ema200 = benchmark_prices.ewm(
            span=self.config.benchmark_ema_period, adjust=False
        ).mean()

        # Warmup period needed for indicators (at least 60 days)
        warmup = max(60, self.config.bb_squeeze_lookback // 2)

        # Precompute indicators across all symbols
        rsi_14 = {}
        rsi_9 = {}
        rsi_21 = {}
        macd_hist_dict = {}
        bb_upper_dict = {}
        bb_width_dict = {}
        bb_squeeze_thresh_dict = {}
        adx_dict = {}
        plus_di_dict = {}
        minus_di_dict = {}
        atr_dict = {}

        for sym in symbols:
            px = prices_df[sym]
            hi = highs_df[sym]
            lo = lows_df[sym]

            # RSI
            rsi_14[sym] = self.calculate_rsi(px, self.config.rsi_mid_period)
            rsi_9[sym] = self.calculate_rsi(px, self.config.rsi_fast_period)
            rsi_21[sym] = self.calculate_rsi(px, self.config.rsi_slow_period)

            # MACD
            _, _, m_hist = self.calculate_macd(
                px, self.config.macd_fast, self.config.macd_slow, self.config.macd_signal
            )
            macd_hist_dict[sym] = m_hist

            # Bollinger Bands & Squeeze Threshold
            bb_up, _, _, bb_w = self.calculate_bollinger_bands(
                px, self.config.bb_period, self.config.bb_std
            )
            bb_upper_dict[sym] = bb_up
            bb_width_dict[sym] = bb_w
            # Rolling 20th percentile of bandwidth
            bb_squeeze_thresh_dict[sym] = bb_w.rolling(
                window=self.config.bb_squeeze_lookback, min_periods=40
            ).quantile(self.config.bb_squeeze_pctile)

            # ADX & Directional Movement
            adx_val, p_di, m_di = self.calculate_adx(hi, lo, px, self.config.adx_period)
            adx_dict[sym] = adx_val
            plus_di_dict[sym] = p_di
            minus_di_dict[sym] = m_di

            # ATR
            atr_dict[sym] = self.calculate_atr(hi, lo, px, self.config.atr_period)

        # Portfolio Tracking State
        cash = self.config.initial_capital
        # Key: symbol -> {'qty': int, 'entry_px': float, 'entry_date': date, 'stop_px': float, 'trimmed': bool}
        holdings: Dict[str, Dict] = {}
        trade_records: List[IndicatorTrade] = []
        equity_curve: List[float] = []

        total_friction_inr = 0.0
        stcg_tax_inr = 0.0
        curr_fy_realized_gains = 0.0

        daily_rf = (1.0 + self.config.risk_free_rate) ** (1.0 / 252.0) - 1.0

        for t_idx in range(warmup, n_dates):
            current_date = dates[t_idx]
            is_rebalance_bar = (t_idx % self.config.rebalance_days == 0)

            # Check benchmark regime in V3
            is_market_bullish = True
            if version == "V3" and self.config.enable_regime_filter:
                curr_bm = benchmark_prices.iloc[t_idx]
                curr_bm_ema = benchmark_ema200.iloc[t_idx]
                if pd.notna(curr_bm) and pd.notna(curr_bm_ema) and curr_bm < curr_bm_ema:
                    is_market_bullish = False

            # 1. Update Holdings & Process Exits / Trims
            exit_symbols = []

            # If bear regime triggered in V3, liquidate all positions to 100% cash
            if not is_market_bullish and holdings:
                for sym, pos in list(holdings.items()):
                    curr_px = prices_df[sym].iloc[t_idx]
                    exit_symbols.append((sym, curr_px, "REGIME_CASH"))
            else:
                for sym, pos in list(holdings.items()):
                    curr_px = prices_df[sym].iloc[t_idx]
                    if pd.isna(curr_px) or curr_px <= 0:
                        continue

                    exit_triggered = False
                    exit_reason = ""

                    r14 = rsi_14[sym].iloc[t_idx]
                    m_hist_now = macd_hist_dict[sym].iloc[t_idx]

                    # V1 Exit: RSI drops below 50 OR MACD histogram turns negative
                    if r14 < self.config.rsi_exit_level:
                        exit_triggered = True
                        exit_reason = "RSI_DROP"
                    elif m_hist_now < 0:
                        exit_triggered = True
                        exit_reason = "MACD_REVERSAL"

                    # V3 Exit: Dynamic ATR Trailing Stop
                    if version == "V3" and self.config.enable_atr_stop:
                        atr_val = atr_dict[sym].iloc[t_idx]
                        if pd.isna(atr_val) or atr_val <= 0:
                            atr_val = 0.02 * curr_px
                        new_stop = curr_px - (self.config.atr_multiplier * atr_val)
                        pos["stop_px"] = max(pos["stop_px"], new_stop)

                        if curr_px <= pos["stop_px"]:
                            exit_triggered = True
                            exit_reason = "ATR_STOP"

                    # V3 Euphoria Trim: Parabolic run with RSI > 82 trims 50%
                    if version == "V3" and not exit_triggered and not pos.get("trimmed", False):
                        if r14 >= self.config.rsi_euphoria_trim:
                            trim_qty = pos["qty"] // 2
                            if trim_qty > 0:
                                pos["qty"] -= trim_qty
                                pos["trimmed"] = True

                                proceeds = trim_qty * curr_px
                                turnover = (trim_qty * pos["entry_px"]) + proceeds
                                friction = turnover * (self.config.statutory_cost_rate + self.config.slippage_rate)
                                gross_pnl = trim_qty * (curr_px - pos["entry_px"])
                                net_pnl = gross_pnl - friction

                                cash += (proceeds - friction)
                                total_friction_inr += friction
                                curr_fy_realized_gains += max(0.0, net_pnl)

                                trade_records.append(IndicatorTrade(
                                    symbol=sym,
                                    entry_date=str(pos["entry_date"]),
                                    exit_date=str(current_date),
                                    entry_price=pos["entry_px"],
                                    exit_price=curr_px,
                                    qty=trim_qty,
                                    pnl_gross=round(gross_pnl, 2),
                                    friction=round(friction, 2),
                                    pnl_net=round(net_pnl, 2),
                                    return_pct=round((curr_px - pos["entry_px"]) / pos["entry_px"] * 100.0, 2),
                                    holding_days=(t_idx - pos["entry_idx"]),
                                    exit_reason="EUPHORIA_TRIM",
                                ))

                    if exit_triggered:
                        exit_symbols.append((sym, curr_px, exit_reason))

            # Process Complete Exits
            for sym, exit_px, reason in exit_symbols:
                if sym not in holdings:
                    continue
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

                trade_records.append(IndicatorTrade(
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

            # 2. Rebalance & Candidate Ranking
            if is_market_bullish and (is_rebalance_bar or not holdings):
                # Calculate scores for all candidate symbols
                scored_candidates = []
                for sym in symbols:
                    px_now = prices_df[sym].iloc[t_idx]
                    if pd.isna(px_now) or px_now <= 0:
                        continue

                    r14_val = rsi_14[sym].iloc[t_idx]
                    r9_val = rsi_9[sym].iloc[t_idx]
                    r21_val = rsi_21[sym].iloc[t_idx]
                    m_hist_now = macd_hist_dict[sym].iloc[t_idx]

                    if pd.isna(r14_val) or pd.isna(m_hist_now):
                        continue

                    # V1 Conditions: 14D RSI in [60, 75] + expanding MACD Histogram
                    if version == "V1":
                        if not (self.config.rsi_min_entry <= r14_val <= self.config.rsi_max_entry):
                            continue
                        if t_idx < 3:
                            continue
                        h1 = macd_hist_dict[sym].iloc[t_idx - 1]
                        h2 = macd_hist_dict[sym].iloc[t_idx - 2]
                        # Positive and expanding over 3 bars
                        if not (m_hist_now > h1 > h2 and m_hist_now > 0):
                            continue

                        score = r14_val + (m_hist_now / px_now * 100.0)
                        scored_candidates.append((score, sym, px_now))

                    # V2 & V3 Conditions: Multi-Horizon RSI + Squeeze Expansion + ADX Trend Gate
                    else:
                        # Composite RSI
                        r_comp = (0.25 * r9_val) + (0.50 * r14_val) + (0.25 * r21_val)
                        if not (self.config.rsi_min_entry <= r_comp <= self.config.rsi_max_entry):
                            continue

                        # ADX Trend Strength & Direction
                        adx_v = adx_dict[sym].iloc[t_idx]
                        p_di_v = plus_di_dict[sym].iloc[t_idx]
                        m_di_v = minus_di_dict[sym].iloc[t_idx]

                        if pd.isna(adx_v) or adx_v < self.config.adx_threshold or p_di_v <= m_di_v:
                            continue

                        # Bollinger Band Squeeze & Breakout
                        # Either currently squeezing or just broke out above upper band from recent squeeze
                        bb_thresh = bb_squeeze_thresh_dict[sym].iloc[t_idx]
                        bb_up = bb_upper_dict[sym].iloc[t_idx]

                        recent_squeeze = False
                        if pd.notna(bb_thresh):
                            # Squeeze present recently (past 5 bars)
                            w_slice = bb_width_dict[sym].iloc[max(0, t_idx - 5) : t_idx + 1]
                            th_val = bb_thresh
                            if (w_slice <= th_val).any():
                                recent_squeeze = True

                        # Must be at or breaking above upper Bollinger Band
                        is_bb_breakout = px_now >= (bb_up * 0.995)

                        if not (recent_squeeze and is_bb_breakout):
                            continue

                        # Ranking Score: Composite RSI boosted by ADX strength
                        score = r_comp * (1.0 + (adx_v / 100.0))
                        scored_candidates.append((score, sym, px_now))

                # Sort candidates descending by momentum score
                scored_candidates.sort(key=lambda x: x[0], reverse=True)
                ranked_syms = [s for _, s, _ in scored_candidates]

                # V2 / V3 Rank Hysteresis:
                # If an existing holding's rank falls beyond exit_rank_threshold, exit it
                if version in ("V2", "V3") and is_rebalance_bar:
                    for sym in list(holdings.keys()):
                        curr_px = prices_df[sym].iloc[t_idx]
                        if sym in ranked_syms:
                            rank = ranked_syms.index(sym)
                            if rank > self.config.exit_rank_threshold:
                                pos = holdings.pop(sym)
                                qty = pos["qty"]
                                proceeds = qty * curr_px
                                trade_turnover = (qty * pos["entry_px"]) + proceeds
                                friction = trade_turnover * (self.config.statutory_cost_rate + self.config.slippage_rate)
                                gross_pnl = qty * (curr_px - pos["entry_px"])
                                net_pnl = gross_pnl - friction

                                cash += (proceeds - friction)
                                total_friction_inr += friction
                                curr_fy_realized_gains += max(0.0, net_pnl)

                                trade_records.append(IndicatorTrade(
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
                                    exit_reason="RANK_DROP",
                                ))
                        else:
                            # Not in ranked list at all; if rebalance bar, close position
                            pos = holdings.pop(sym)
                            qty = pos["qty"]
                            proceeds = qty * curr_px
                            trade_turnover = (qty * pos["entry_px"]) + proceeds
                            friction = trade_turnover * (self.config.statutory_cost_rate + self.config.slippage_rate)
                            gross_pnl = qty * (curr_px - pos["entry_px"])
                            net_pnl = gross_pnl - friction

                            cash += (proceeds - friction)
                            total_friction_inr += friction
                            curr_fy_realized_gains += max(0.0, net_pnl)

                            trade_records.append(IndicatorTrade(
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
                                exit_reason="RANK_DROP",
                            ))

                # Allocate to new positions in vacant slots
                vacant_slots = self.config.portfolio_slots - len(holdings)
                if vacant_slots > 0 and scored_candidates:
                    top_picks = [c for c in scored_candidates if c[1] not in holdings][:vacant_slots]

                    if top_picks:
                        current_equity = cash + sum(
                            h["qty"] * prices_df[s].iloc[t_idx] for s, h in holdings.items()
                        )
                        target_slot_capital = current_equity / self.config.portfolio_slots

                        for _, sym, entry_px in top_picks:
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

                                # Initial Stop Setup
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
                                    "trimmed": False,
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
        sim_days = max(1, n_dates - warmup)
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

        return IndicatorSummary(
            strategy_version=f"Indicator_{version}",
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
