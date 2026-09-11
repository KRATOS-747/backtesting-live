"""
Institutional Microstructure Spike Fading Engine (Alpha 2)
Fades short-term option price surges conditioned on Primary OI Wall resistance
and underlying spot momentum deceleration, held for up to 90 minutes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import polars as pl


@dataclass
class SpikeFaderConfig:
    """Configuration parameters for the 90-Minute Spike Fading Engine."""
    surge_pct_threshold: float = 12.0       # Minimum 5-minute price surge percentage
    min_premium: float = 30.0               # Minimum option premium to avoid penny options
    max_moneyness: int = 200                # Strike search boundary around ATM (± points)
    oi_wall_proximity: float = 35.0         # Max spot distance to Call/Put Wall for confluence
    max_spot_roc1: float = 0.02             # Max spot 1-min ROC (exhaustion filter: <= 0.02% for CE, >= -0.02% for PE)
    holding_bars: int = 90                  # Primary holding horizon (90 minutes)
    profit_target_pct: float = 0.20         # Take-profit threshold (20% decay from entry)
    stop_loss_pct: float = 0.25             # Stop-loss threshold (25% adverse price surge)
    lot_size: int = 75                      # NIFTY lot size (75 qty)
    num_lots: int = 1                       # Number of lots per trade
    brokerage_per_order: float = 20.0       # Flat brokerage per executed leg (₹20)
    slippage_pts_per_leg: float = 0.50      # Conservative bid-ask spread cross per execution (pts)


@dataclass
class SpikeTradeRecord:
    """Audit record capturing individual trade lifecycle and financial metrics."""
    date: str
    time: str
    strike: int
    option_type: str                        # 'CE' or 'PE'
    p_entry: float
    p_exit: float
    pts_pnl: float                          # Points gained: p_entry - p_exit (short position)
    gross_pnl_inr: float                    # pts_pnl * qty
    friction_inr: float                     # Total statutory friction + slippage
    net_pnl_inr: float                      # gross_pnl_inr - friction_inr
    is_win: bool
    exit_reason: str                        # 'PROFIT_TARGET', 'STOP_LOSS', 'TIME_TIMEOUT', 'EOD_CUT'
    exit_step: int                          # Minutes held before exit
    surge_pct: float                        # Magnitude of the triggering surge
    spot_at_entry: float
    wall_strike: int                        # Opposing institutional OI barrier strike


@dataclass
class SpikeFaderSummary:
    """Aggregated portfolio KPIs across tested sessions."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    gross_pnl_inr: float
    total_friction_inr: float
    net_pnl_inr: float
    net_to_gross_conversion_pct: float
    avg_gross_per_trade_inr: float
    avg_net_per_trade_inr: float
    avg_points_per_trade: float
    profit_target_hits: int
    profit_target_rate_pct: float
    stop_loss_hits: int
    stop_loss_rate_pct: float
    time_exit_hits: int
    time_exit_rate_pct: float
    profit_factor: float
    trades: List[SpikeTradeRecord] = field(default_factory=list)


class SpikeFaderEngine:
    """
    Production-grade vectorized engine for detecting, filtering,
    and executing the 90-Minute Microstructure Spike Fade alpha.
    """

    def __init__(self, config: Optional[SpikeFaderConfig] = None):
        self.config = config or SpikeFaderConfig()

    def calculate_trade_friction(self, p_entry: float, p_exit: float) -> float:
        """
        Calculates Indian statutory derivatives friction for a single-leg short trade:
        - STT: 0.10% on option sell turnover (entry)
        - Exchange turnover charges: 0.05% on total turnover (buy + sell)
        - Brokerage: ₹20 entry + ₹20 exit = ₹40 flat
        - GST: 18% on (Brokerage + Exchange turnover)
        - Stamp Duty: 0.003% on buy turnover (exit)
        - Slippage: 0.50 pts each way = 1.0 pt round-trip * total_qty
        """
        qty = self.config.lot_size * self.config.num_lots
        turnover_sell = p_entry * qty
        turnover_buy = p_exit * qty

        stt = 0.0010 * turnover_sell
        exch_turnover = 0.0005 * (turnover_sell + turnover_buy)
        brokerage = 2.0 * self.config.brokerage_per_order
        gst = 0.18 * (brokerage + exch_turnover)
        stamp_duty = 0.00003 * turnover_buy
        sebi_charges = 0.000001 * (turnover_sell + turnover_buy)
        slippage = 2.0 * self.config.slippage_pts_per_leg * qty

        return float(stt + exch_turnover + brokerage + gst + stamp_duty + sebi_charges + slippage)

    def run_session(
        self,
        options_file_path: str,
        spot_day_df: pd.DataFrame,
        day_str: str,
        naive_mode: bool = False,
    ) -> List[SpikeTradeRecord]:
        """
        Processes a single trading session across all 1-minute chain snapshots.
        """
        trades: List[SpikeTradeRecord] = []
        if spot_day_df.empty:
            return trades

        # Pre-index spot data
        spot_sorted = spot_day_df.copy().sort_values("date").reset_index(drop=True)
        if "time_str" not in spot_sorted.columns:
            spot_sorted["time_str"] = spot_sorted["date"].dt.strftime("%H:%M:%S")
        if "spot_roc_1m" not in spot_sorted.columns:
            spot_sorted["spot_roc_1m"] = spot_sorted["close"].pct_change(1) * 100.0

        spot_map = dict(zip(spot_sorted["time_str"], spot_sorted["close"]))
        roc_map = dict(zip(spot_sorted["time_str"], spot_sorted["spot_roc_1m"]))

        # Load options chain snapshot via Polars
        pldf = pl.read_csv(options_file_path, ignore_errors=True)
        pdf = pldf.to_pandas()
        if "DateTime" not in pdf.columns:
            return trades

        pdf["DateTime"] = pd.to_datetime(pdf["DateTime"]).dt.tz_localize(None)
        pdf["time_str"] = pdf["DateTime"].dt.strftime("%H:%M:%S")
        pdf = pdf.sort_values("DateTime").reset_index(drop=True)

        times = pdf["time_str"].values
        n_bars = len(pdf)
        h_bars = self.config.holding_bars

        # Extract strike matrices
        ce_close_cols = [c for c in pdf.columns if c.endswith("CE_close") and c.replace("CE_close", "").isdigit()]
        pe_close_cols = [c for c in pdf.columns if c.endswith("PE_close") and c.replace("PE_close", "").isdigit()]
        ce_oi_cols = [c for c in pdf.columns if c.endswith("CE_OI") and c.replace("CE_OI", "").isdigit()]
        pe_oi_cols = [c for c in pdf.columns if c.endswith("PE_OI") and c.replace("PE_OI", "").isdigit()]

        ce_strikes = np.array([int(c.replace("CE_close", "")) for c in ce_close_cols])
        pe_strikes = np.array([int(c.replace("PE_close", "")) for c in pe_close_cols])
        ce_oi_strikes = np.array([int(c.replace("CE_OI", "")) for c in ce_oi_cols])
        pe_oi_strikes = np.array([int(c.replace("PE_OI", "")) for c in pe_oi_cols])

        ce_close_mat = pdf[ce_close_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        pe_close_mat = pdf[pe_close_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        ce_oi_mat = pdf[ce_oi_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        pe_oi_mat = pdf[pe_oi_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)

        # Vectorized Primary Call Wall and Put Wall identification
        call_wall_idx = np.argmax(ce_oi_mat, axis=1)
        call_walls = ce_oi_strikes[call_wall_idx] if len(ce_oi_strikes) > 0 else np.zeros(n_bars)
        put_wall_idx = np.argmax(pe_oi_mat, axis=1)
        put_walls = pe_oi_strikes[put_wall_idx] if len(pe_oi_strikes) > 0 else np.zeros(n_bars)

        ce_k_to_idx = {k: i for i, k in enumerate(ce_strikes)}
        pe_k_to_idx = {k: i for i, k in enumerate(pe_strikes)}

        qty = self.config.lot_size * self.config.num_lots

        for idx in range(5, n_bars - h_bars):
            t_str = times[idx]
            if not ("09:30:00" <= t_str <= "14:45:00"):
                continue

            spot_px = spot_map.get(t_str)
            spot_roc1 = roc_map.get(t_str, 0.0)
            if spot_px is None or pd.isna(spot_px):
                continue

            call_wall = call_walls[idx]
            put_wall = put_walls[idx]
            atm_k = int(round(spot_px / 50.0) * 50)
            target_strikes = [
                atm_k + offset
                for offset in range(-self.config.max_moneyness, self.config.max_moneyness + 50, 50)
            ]

            # 1. CE Evaluation
            for k in target_strikes:
                col_idx = ce_k_to_idx.get(k)
                if col_idx is None:
                    continue

                p_now = ce_close_mat[idx, col_idx]
                p_prev5 = ce_close_mat[idx - 5, col_idx]

                if p_now < self.config.min_premium or p_prev5 <= 0:
                    continue

                surge_pct = ((p_now - p_prev5) / p_prev5) * 100.0
                if surge_pct >= self.config.surge_pct_threshold:
                    # Confluence condition
                    if not naive_mode:
                        is_at_wall = abs(spot_px - call_wall) <= self.config.oi_wall_proximity
                        is_stalled = spot_roc1 <= self.config.max_spot_roc1
                        if not (is_at_wall and is_stalled):
                            continue

                    p_entry = p_now
                    p_exit = ce_close_mat[idx + h_bars, col_idx]
                    exit_reason = f"TIME_{h_bars}M"
                    actual_step = h_bars

                    # Multi-bar exit path
                    for step in range(1, h_bars + 1):
                        px = ce_close_mat[idx + step, col_idx]
                        if px <= 0:
                            continue
                        # Stop Loss check
                        if (px - p_entry) / p_entry >= self.config.stop_loss_pct:
                            p_exit = px
                            exit_reason = "STOP_LOSS"
                            actual_step = step
                            break
                        # Profit Target check
                        if (p_entry - px) / p_entry >= self.config.profit_target_pct:
                            p_exit = px
                            exit_reason = "PROFIT_TARGET"
                            actual_step = step
                            break

                    pts_pnl = p_entry - p_exit
                    gross_pnl = pts_pnl * qty
                    friction = self.calculate_trade_friction(p_entry, p_exit)
                    net_pnl = gross_pnl - friction

                    trades.append(
                        SpikeTradeRecord(
                            date=day_str,
                            time=t_str,
                            strike=k,
                            option_type="CE",
                            p_entry=p_entry,
                            p_exit=p_exit,
                            pts_pnl=pts_pnl,
                            gross_pnl_inr=gross_pnl,
                            friction_inr=friction,
                            net_pnl_inr=net_pnl,
                            is_win=net_pnl > 0,
                            exit_reason=exit_reason,
                            exit_step=actual_step,
                            surge_pct=surge_pct,
                            spot_at_entry=spot_px,
                            wall_strike=int(call_wall),
                        )
                    )

            # 2. PE Evaluation
            for k in target_strikes:
                col_idx = pe_k_to_idx.get(k)
                if col_idx is None:
                    continue

                p_now = pe_close_mat[idx, col_idx]
                p_prev5 = pe_close_mat[idx - 5, col_idx]

                if p_now < self.config.min_premium or p_prev5 <= 0:
                    continue

                surge_pct = ((p_now - p_prev5) / p_prev5) * 100.0
                if surge_pct >= self.config.surge_pct_threshold:
                    if not naive_mode:
                        is_at_wall = abs(spot_px - put_wall) <= self.config.oi_wall_proximity
                        is_stalled = spot_roc1 >= -self.config.max_spot_roc1
                        if not (is_at_wall and is_stalled):
                            continue

                    p_entry = p_now
                    p_exit = pe_close_mat[idx + h_bars, col_idx]
                    exit_reason = f"TIME_{h_bars}M"
                    actual_step = h_bars

                    for step in range(1, h_bars + 1):
                        px = pe_close_mat[idx + step, col_idx]
                        if px <= 0:
                            continue
                        if (px - p_entry) / p_entry >= self.config.stop_loss_pct:
                            p_exit = px
                            exit_reason = "STOP_LOSS"
                            actual_step = step
                            break
                        if (p_entry - px) / p_entry >= self.config.profit_target_pct:
                            p_exit = px
                            exit_reason = "PROFIT_TARGET"
                            actual_step = step
                            break

                    pts_pnl = p_entry - p_exit
                    gross_pnl = pts_pnl * qty
                    friction = self.calculate_trade_friction(p_entry, p_exit)
                    net_pnl = gross_pnl - friction

                    trades.append(
                        SpikeTradeRecord(
                            date=day_str,
                            time=t_str,
                            strike=k,
                            option_type="PE",
                            p_entry=p_entry,
                            p_exit=p_exit,
                            pts_pnl=pts_pnl,
                            gross_pnl_inr=gross_pnl,
                            friction_inr=friction,
                            net_pnl_inr=net_pnl,
                            is_win=net_pnl > 0,
                            exit_reason=exit_reason,
                            exit_step=actual_step,
                            surge_pct=surge_pct,
                            spot_at_entry=spot_px,
                            wall_strike=int(put_wall),
                        )
                    )

        return trades

    def aggregate_summary(self, trades: List[SpikeTradeRecord]) -> SpikeFaderSummary:
        """Computes institutional KPIs across all executed trades."""
        if not trades:
            return SpikeFaderSummary(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate_pct=0.0,
                gross_pnl_inr=0.0,
                total_friction_inr=0.0,
                net_pnl_inr=0.0,
                net_to_gross_conversion_pct=0.0,
                avg_gross_per_trade_inr=0.0,
                avg_net_per_trade_inr=0.0,
                avg_points_per_trade=0.0,
                profit_target_hits=0,
                profit_target_rate_pct=0.0,
                stop_loss_hits=0,
                stop_loss_rate_pct=0.0,
                time_exit_hits=0,
                time_exit_rate_pct=0.0,
                profit_factor=0.0,
                trades=[],
            )

        df = pd.DataFrame([t.__dict__ for t in trades])
        n = len(df)
        wins = int(df["is_win"].sum())
        losses = n - wins
        wr = (wins / n) * 100.0

        gross = float(df["gross_pnl_inr"].sum())
        fric = float(df["friction_inr"].sum())
        net = float(df["net_pnl_inr"].sum())
        conv = (net / gross * 100.0) if gross > 0 else 0.0

        pt_hits = int((df["exit_reason"] == "PROFIT_TARGET").sum())
        sl_hits = int((df["exit_reason"] == "STOP_LOSS").sum())
        time_hits = n - pt_hits - sl_hits

        # Profit Factor
        gross_wins = float(df[df["net_pnl_inr"] > 0]["net_pnl_inr"].sum())
        gross_losses = abs(float(df[df["net_pnl_inr"] < 0]["net_pnl_inr"].sum()))
        pf = (gross_wins / gross_losses) if gross_losses > 0 else 999.0

        return SpikeFaderSummary(
            total_trades=n,
            winning_trades=wins,
            losing_trades=losses,
            win_rate_pct=round(wr, 2),
            gross_pnl_inr=round(gross, 2),
            total_friction_inr=round(fric, 2),
            net_pnl_inr=round(net, 2),
            net_to_gross_conversion_pct=round(conv, 2),
            avg_gross_per_trade_inr=round(gross / n, 2),
            avg_net_per_trade_inr=round(net / n, 2),
            avg_points_per_trade=round(float(df["pts_pnl"].mean()), 2),
            profit_target_hits=pt_hits,
            profit_target_rate_pct=round((pt_hits / n) * 100.0, 2),
            stop_loss_hits=sl_hits,
            stop_loss_rate_pct=round((sl_hits / n) * 100.0, 2),
            time_exit_hits=time_hits,
            time_exit_rate_pct=round((time_hits / n) * 100.0, 2),
            profit_factor=round(pf, 2),
            trades=trades,
        )
