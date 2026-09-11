"""
Institutional OTM Strangle Harvester & Delta-Neutral Adjustment Engine
Simulates systematic OTM strangles with delta-targeted strike selection,
volatility skew compensation, independent ROC leg cutting, and Greek P&L attribution.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from .pricing import Black76, ImpliedVolatilitySolver
from .chain_parser import OptionsChainParser
from .strike_matrix import StrikeMatrixEngine, StrangleStrikes
from .leg_cutter import LegCutter, LegCutConfig, LegState
from .pnl_attribution import GreekPnLAttributor, BarAttribution
from .backtester import BacktestMetrics


@dataclass
class StrangleConfig:
    mode: str = "POINTS"                          # 'POINTS' or 'DELTA'
    otm_points: float = 150.0                    # Strike offset in points (e.g. ATM +/- 150)
    target_delta: float = 0.20                   # Target delta (e.g. 0.20 delta Call and Put)
    rebalance_delta_imbalance: Optional[float] = 0.25  # Max net delta imbalance before alert
    leg_config: LegCutConfig = field(default_factory=LegCutConfig)
    lot_size: int = 25
    num_lots: int = 4
    index_name: str = "NIFTY"
    entry_time: str = "09:25:00"


@dataclass
class StrangleSessionResult:
    date: str
    entry_spot: float
    call_strike: int
    put_strike: int
    width_points: int
    call_entry: float
    put_entry: float
    strangle_entry: float
    call_exit: float
    put_exit: float
    call_reason: str
    put_reason: str
    gross_points: float
    net_points: float
    gross_pnl_rupees: float
    net_pnl_rupees: float
    statutory_costs_rupees: float
    attributions: Dict[str, float]
    bar_attributions: List[BarAttribution] = field(default_factory=list)


class StrangleEngine:
    """
    Backtests systematic OTM Strangle strategies across historical NSE 1-minute options chains.
    Supports both fixed-offset (Points) and skew-aware Delta-neutral strike selection.
    """

    def __init__(self, config: Optional[StrangleConfig] = None):
        self.config = config or StrangleConfig()
        self.lot_size = self.config.lot_size
        self.num_lots = self.config.num_lots
        self.total_qty = self.lot_size * self.num_lots
        self.strike_engine = StrikeMatrixEngine.for_index(self.config.index_name)
        self.leg_cutter = LegCutter(self.config.leg_config)

    def calculate_statutory_costs(self, sell_turnover: float, buy_turnover: float, num_orders: int = 4) -> float:
        """Indian statutory friction calculator."""
        total_turnover = sell_turnover + buy_turnover
        stt = sell_turnover * 0.0010
        exchange_charges = total_turnover * 0.00035
        brokerage = num_orders * 20.0
        gst = (exchange_charges + brokerage) * 0.18
        stamp_duty = buy_turnover * 0.00003
        sebi_charges = total_turnover * 0.000001
        return float(stt + exchange_charges + brokerage + gst + stamp_duty + sebi_charges)

    def select_strikes(
        self,
        parser: OptionsChainParser,
        spot: float,
        T_years: float,
        r: float = 0.065
    ) -> Tuple[int, int]:
        """
        Determines Call and Put strikes based on configured selection mode:
        1. 'POINTS': Fixed distance from spot (e.g. ATM + 150 CE, ATM - 150 PE).
        2. 'DELTA': Skew-adjusted strike selection targeting specific Call and Put deltas.
        """
        if self.config.mode.upper() == "DELTA":
            snap = parser.get_chain_snapshot(self.config.entry_time)
            if not snap.empty and "strike" in snap.columns:
                call_deltas: Dict[int, float] = {}
                put_deltas: Dict[int, float] = {}

                for _, row in snap.iterrows():
                    K = int(row["strike"])
                    ce_px = float(row.get("ce_close", 0.0) or 0.0)
                    pe_px = float(row.get("pe_close", 0.0) or 0.0)

                    if ce_px > 0.5 and K >= spot:
                        iv_c = ImpliedVolatilitySolver.solve_single(ce_px, spot, K, T_years, r, "CE")
                        if iv_c > 0:
                            call_deltas[K] = Black76.greeks(spot, K, T_years, r, iv_c, "CE").delta

                    if pe_px > 0.5 and K <= spot:
                        iv_p = ImpliedVolatilitySolver.solve_single(pe_px, spot, K, T_years, r, "PE")
                        if iv_p > 0:
                            put_deltas[K] = Black76.greeks(spot, K, T_years, r, iv_p, "PE").delta

                if call_deltas and put_deltas:
                    best_ce = min(call_deltas, key=lambda k: abs(call_deltas[k] - self.config.target_delta))
                    best_pe = min(put_deltas, key=lambda k: abs(abs(put_deltas[k]) - self.config.target_delta))
                    return best_ce, best_pe

        # Fallback to Points mode
        strangle_s = self.strike_engine.build_strangle_strikes(spot, otm_points=self.config.otm_points)
        return strangle_s.call_strike, strangle_s.put_strike

    def run_session(
        self,
        options_file_path: str,
        spot_df: pd.DataFrame,
        date_str: str,
        r: float = 0.065
    ) -> Optional[StrangleSessionResult]:
        """Runs a complete 1-minute simulation for a single trading session."""
        if not os.path.exists(options_file_path):
            return None

        # 1. Spot data setup
        df_spot = spot_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_spot["date"]):
            df_spot["date"] = pd.to_datetime(df_spot["date"])

        day_spot = df_spot[df_spot["date"].dt.strftime("%Y-%m-%d") == date_str].copy().sort_values("date")
        if day_spot.empty:
            return None

        day_spot["time_str"] = day_spot["date"].dt.strftime("%H:%M:%S")
        spot_map = dict(zip(day_spot["time_str"], day_spot["close"]))

        entry_time = self.config.entry_time
        anchor_spot = self.strike_engine.find_anchor_spot(day_spot, entry_time)

        # Time to expiry estimate
        curr_dt = pd.to_datetime(f"{date_str} {entry_time}")
        days_to_thursday = (3 - curr_dt.weekday()) % 7
        T_years = max((days_to_thursday + 1.0) / 365.0, 1e-4)

        # 2. Select Strangle Strikes
        parser = OptionsChainParser(options_file_path)
        call_strike, put_strike = self.select_strikes(parser, anchor_spot, T_years, r)

        # 3. Fast batch fetch Call and Put series
        contracts = [(call_strike, "CE"), (put_strike, "PE")]
        batch_df = parser.get_multiple_contracts(contracts)
        if batch_df.empty:
            return None

        ce_col = f"{call_strike}CE_price"
        pe_col = f"{put_strike}PE_price"
        if ce_col not in batch_df.columns or pe_col not in batch_df.columns:
            return None

        batch_df["time_str"] = batch_df["DateTime"].dt.strftime("%H:%M:%S")
        trade_bars = batch_df[batch_df["time_str"] >= entry_time].copy().reset_index(drop=True)
        if trade_bars.empty:
            return None

        # 4. Entry execution
        entry_row = trade_bars.iloc[0]
        ce_entry_px = float(entry_row[ce_col])
        pe_entry_px = float(entry_row[pe_col])

        ce_leg = LegState(symbol=f"{call_strike}CE", entry_time=entry_row["time_str"], entry_price=ce_entry_px)
        pe_leg = LegState(symbol=f"{put_strike}PE", entry_time=entry_row["time_str"], entry_price=pe_entry_px)

        ce_history = [ce_entry_px]
        pe_history = [pe_entry_px]
        bar_attributions: List[BarAttribution] = []

        prev_spot = anchor_spot
        prev_ce = ce_entry_px
        prev_pe = pe_entry_px
        prev_iv = 0.20

        # 5. Intraday execution loop
        for i in range(1, len(trade_bars)):
            bar = trade_bars.iloc[i]
            t_str = bar["time_str"]
            ce_px = float(bar[ce_col])
            pe_px = float(bar[pe_col])

            ce_history.append(ce_px)
            pe_history.append(pe_px)

            if ce_leg.is_active:
                ce_leg = self.leg_cutter.update_leg(ce_leg, t_str, ce_px, pd.Series(ce_history))
            if pe_leg.is_active:
                pe_leg = self.leg_cutter.update_leg(pe_leg, t_str, pe_px, pd.Series(pe_history))

            curr_spot = float(spot_map.get(t_str, prev_spot))
            dS = curr_spot - prev_spot
            d_pnl_points = (prev_ce - ce_px if ce_leg.is_active else 0.0) + (prev_pe - pe_px if pe_leg.is_active else 0.0)
            actual_bar_pnl = d_pnl_points * self.total_qty

            hour_val = int(t_str[:2]) + int(t_str[3:5]) / 60.0
            day_fraction_left = max((15.5 - hour_val) / 6.25, 0.01)
            T_current = max((days_to_thursday + day_fraction_left) / 365.0, 1e-4)

            ce_iv = ImpliedVolatilitySolver.solve_single(ce_px, curr_spot, call_strike, T_current, r, "CE") if (ce_leg.is_active and ce_px > 0) else prev_iv
            pe_iv = ImpliedVolatilitySolver.solve_single(pe_px, curr_spot, put_strike, T_current, r, "PE") if (pe_leg.is_active and pe_px > 0) else prev_iv
            curr_avg_iv = (ce_iv + pe_iv) / 2.0 if (ce_iv > 0 and pe_iv > 0) else prev_iv
            dVol = curr_avg_iv - prev_iv

            delta_pos = 0.0
            gamma_pos = 0.0
            theta_daily_pos = 0.0
            vega_pos = 0.0

            if ce_leg.is_active and ce_iv > 0:
                ce_g = Black76.greeks(curr_spot, call_strike, T_current, r, ce_iv, "CE")
                delta_pos += (-ce_g.delta) * self.total_qty
                gamma_pos += (-ce_g.gamma) * self.total_qty
                theta_daily_pos += (-ce_g.theta / 252.0) * self.total_qty
                vega_pos += (-ce_g.vega) * self.total_qty

            if pe_leg.is_active and pe_iv > 0:
                pe_g = Black76.greeks(curr_spot, put_strike, T_current, r, pe_iv, "PE")
                delta_pos += (-pe_g.delta) * self.total_qty
                gamma_pos += (-pe_g.gamma) * self.total_qty
                theta_daily_pos += (-pe_g.theta / 252.0) * self.total_qty
                vega_pos += (-pe_g.vega) * self.total_qty

            bar_attr = GreekPnLAttributor.attribute_bar(
                delta_pos=delta_pos,
                gamma_pos=gamma_pos,
                theta_pos_daily=theta_daily_pos,
                vega_pos_per_vol=vega_pos,
                dS=dS,
                dVol=dVol,
                dt_days=1.0 / 375.0,
                actual_pnl=actual_bar_pnl,
                timestamp=t_str
            )
            bar_attributions.append(bar_attr)

            prev_spot = curr_spot
            prev_ce = ce_px
            prev_pe = pe_px
            prev_iv = curr_avg_iv

            if not ce_leg.is_active and not pe_leg.is_active:
                break

        # 6. Final settlement & metrics
        ce_exit_px = ce_leg.exit_price if ce_leg.exit_price is not None else ce_history[-1]
        pe_exit_px = pe_leg.exit_price if pe_leg.exit_price is not None else pe_history[-1]

        ce_pnl_pts = ce_entry_px - ce_exit_px
        pe_pnl_pts = pe_entry_px - pe_exit_px
        gross_pts = ce_pnl_pts + pe_pnl_pts
        gross_rupees = gross_pts * self.total_qty

        sell_turnover = (ce_entry_px + pe_entry_px) * self.total_qty
        buy_turnover = (ce_exit_px + pe_exit_px) * self.total_qty
        statutory_costs = self.calculate_statutory_costs(sell_turnover, buy_turnover, num_orders=4)

        net_rupees = gross_rupees - statutory_costs
        net_pts = net_rupees / self.total_qty if self.total_qty > 0 else 0.0
        aggregated_attr = GreekPnLAttributor.aggregate_attribution(bar_attributions)

        return StrangleSessionResult(
            date=date_str,
            entry_spot=anchor_spot,
            call_strike=call_strike,
            put_strike=put_strike,
            width_points=int(call_strike - put_strike),
            call_entry=round(ce_entry_px, 2),
            put_entry=round(pe_entry_px, 2),
            strangle_entry=round(ce_entry_px + pe_entry_px, 2),
            call_exit=round(ce_exit_px, 2),
            put_exit=round(pe_exit_px, 2),
            call_reason=ce_leg.exit_reason or "RUNNING",
            put_reason=pe_leg.exit_reason or "RUNNING",
            gross_points=round(gross_pts, 2),
            net_points=round(net_pts, 2),
            gross_pnl_rupees=round(gross_rupees, 2),
            net_pnl_rupees=round(net_rupees, 2),
            statutory_costs_rupees=round(statutory_costs, 2),
            attributions=aggregated_attr,
            bar_attributions=bar_attributions
        )

    def run_backtest(
        self,
        options_dir: str,
        spot_csv_or_df: Union[str, pd.DataFrame],
        dates: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, BacktestMetrics]:
        """Runs multi-session Strangle backtest across all available dates."""
        if isinstance(spot_csv_or_df, str):
            spot_df = pd.read_csv(spot_csv_or_df)
        else:
            spot_df = spot_csv_or_df.copy()

        if not pd.api.types.is_datetime64_any_dtype(spot_df["date"]):
            spot_df["date"] = pd.to_datetime(spot_df["date"])

        pattern = os.path.join(options_dir, "NIFTY_*.csv")
        files = sorted(glob.glob(pattern))
        results: List[StrangleSessionResult] = []

        for fpath in files:
            fname = os.path.basename(fpath)
            raw_date = fname.replace("NIFTY_", "").replace(".csv", "")
            if len(raw_date) == 8:
                date_str = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
            else:
                continue

            if dates is not None and date_str not in dates:
                continue

            res = self.run_session(fpath, spot_df, date_str)
            if res is not None:
                results.append(res)

        if not results:
            empty_metrics = BacktestMetrics(
                total_sessions=0, winning_sessions=0, losing_sessions=0, win_rate_pct=0.0,
                gross_pnl_rupees=0.0, statutory_costs_rupees=0.0, net_pnl_rupees=0.0,
                profit_factor=0.0, max_drawdown_rupees=0.0, max_drawdown_pct=0.0,
                sharpe_ratio=0.0, avg_trade_pnl_rupees=0.0, max_win_rupees=0.0,
                max_loss_rupees=0.0, greek_attributions={}
            )
            return pd.DataFrame(), empty_metrics

        records = []
        for r in results:
            records.append({
                "date": r.date,
                "entry_spot": r.entry_spot,
                "call_strike": r.call_strike,
                "put_strike": r.put_strike,
                "width_points": r.width_points,
                "strangle_entry": r.strangle_entry,
                "call_exit": r.call_exit,
                "put_exit": r.put_exit,
                "call_reason": r.call_reason,
                "put_reason": r.put_reason,
                "gross_points": r.gross_points,
                "net_points": r.net_points,
                "gross_pnl": r.gross_pnl_rupees,
                "statutory_costs": r.statutory_costs_rupees,
                "net_pnl": r.net_pnl_rupees,
                "delta_pnl": r.attributions.get("delta_pnl", 0.0),
                "gamma_pnl": r.attributions.get("gamma_pnl", 0.0),
                "theta_pnl": r.attributions.get("theta_pnl", 0.0),
                "vega_pnl": r.attributions.get("vega_pnl", 0.0),
                "residual_pnl": r.attributions.get("residual_pnl", 0.0),
            })

        df_summary = pd.DataFrame(records)
        total_sessions = len(df_summary)
        wins = df_summary[df_summary["net_pnl"] > 0]
        losses = df_summary[df_summary["net_pnl"] <= 0]
        win_rate = (len(wins) / total_sessions) * 100.0 if total_sessions > 0 else 0.0

        total_gross = float(df_summary["gross_pnl"].sum())
        total_costs = float(df_summary["statutory_costs"].sum())
        total_net = float(df_summary["net_pnl"].sum())

        sum_wins = float(wins["net_pnl"].sum()) if not wins.empty else 0.0
        sum_losses = abs(float(losses["net_pnl"].sum())) if not losses.empty else 0.0
        profit_factor = (sum_wins / sum_losses) if sum_losses > 0 else (999.0 if sum_wins > 0 else 0.0)

        df_summary["cum_net_pnl"] = df_summary["net_pnl"].cumsum()
        df_summary["peak"] = df_summary["cum_net_pnl"].cummax()
        df_summary["drawdown"] = df_summary["peak"] - df_summary["cum_net_pnl"]
        max_dd = float(df_summary["drawdown"].max())

        daily_pnls = df_summary["net_pnl"].values
        sharpe = (float(np.mean(daily_pnls)) / float(np.std(daily_pnls))) * np.sqrt(252.0) if len(daily_pnls) > 1 and np.std(daily_pnls) > 0 else 0.0

        total_greeks = {
            "total_pnl": round(total_net, 2),
            "delta_pnl": round(float(df_summary["delta_pnl"].sum()), 2),
            "gamma_pnl": round(float(df_summary["gamma_pnl"].sum()), 2),
            "theta_pnl": round(float(df_summary["theta_pnl"].sum()), 2),
            "vega_pnl": round(float(df_summary["vega_pnl"].sum()), 2),
            "residual_pnl": round(float(df_summary["residual_pnl"].sum()), 2),
        }

        metrics = BacktestMetrics(
            total_sessions=total_sessions,
            winning_sessions=len(wins),
            losing_sessions=len(losses),
            win_rate_pct=round(win_rate, 2),
            gross_pnl_rupees=round(total_gross, 2),
            statutory_costs_rupees=round(total_costs, 2),
            net_pnl_rupees=round(total_net, 2),
            profit_factor=round(profit_factor, 2),
            max_drawdown_rupees=round(max_dd, 2),
            max_drawdown_pct=0.0,
            sharpe_ratio=round(sharpe, 2),
            avg_trade_pnl_rupees=round(total_net / total_sessions, 2) if total_sessions > 0 else 0.0,
            max_win_rupees=round(float(df_summary["net_pnl"].max()), 2) if not df_summary.empty else 0.0,
            max_loss_rupees=round(float(df_summary["net_pnl"].min()), 2) if not df_summary.empty else 0.0,
            greek_attributions=total_greeks
        )

        return df_summary, metrics
