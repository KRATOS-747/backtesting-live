"""
Defined-Risk 4-Leg Iron Condor & Iron Fly Quantitative Harvester
Simulates capital-efficient multi-leg credit structures with protective wings,
asymmetric margin relief, Greek Taylor-series attribution, and statutory friction.
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
from .strike_matrix import StrikeMatrixEngine, IronCondorStrikes
from .leg_cutter import LegCutter, LegCutConfig, LegState
from .pnl_attribution import GreekPnLAttributor, BarAttribution
from .backtester import BacktestMetrics


@dataclass
class IronCondorConfig:
    structure_type: str = "IRON_CONDOR"  # 'IRON_CONDOR' or 'IRON_FLY'
    short_offset: float = 100.0          # For IC: distance from ATM to short strikes
    wing_width: float = 150.0            # Distance from short strike to outer hedge wing
    leg_config: LegCutConfig = field(default_factory=LegCutConfig)
    lot_size: int = 25
    num_lots: int = 4
    index_name: str = "NIFTY"
    entry_time: str = "09:25:00"


@dataclass
class IronCondorSessionResult:
    date: str
    entry_spot: float
    structure_type: str
    short_put: int
    long_put: int
    short_call: int
    long_call: int
    wing_width: int
    net_credit_entry: float
    max_risk_points: float
    gross_points: float
    net_points: float
    gross_pnl_rupees: float
    net_pnl_rupees: float
    statutory_costs_rupees: float
    short_put_reason: str
    short_call_reason: str
    attributions: Dict[str, float]
    bar_attributions: List[BarAttribution] = field(default_factory=list)


class IronCondorEngine:
    """
    Simulates institutional 4-leg defined-risk credit spreads (Iron Condor & Iron Fly)
    with strict downside risk containment, exchange margin rules, and Greek attribution.
    """

    def __init__(self, config: Optional[IronCondorConfig] = None):
        self.config = config or IronCondorConfig()
        self.lot_size = self.config.lot_size
        self.num_lots = self.config.num_lots
        self.total_qty = self.lot_size * self.num_lots
        self.strike_engine = StrikeMatrixEngine.for_index(self.config.index_name)
        self.leg_cutter = LegCutter(self.config.leg_config)

    def calculate_statutory_costs(self, sell_turnover: float, buy_turnover: float, num_orders: int = 8) -> float:
        """Indian statutory friction for 4-leg structure (entry 4 legs + exit 4 legs = 8 orders)."""
        total_turnover = sell_turnover + buy_turnover
        stt = sell_turnover * 0.0010
        exchange_charges = total_turnover * 0.00035
        brokerage = num_orders * 20.0
        gst = (exchange_charges + brokerage) * 0.18
        stamp_duty = buy_turnover * 0.00003
        sebi_charges = total_turnover * 0.000001
        return float(stt + exchange_charges + brokerage + gst + stamp_duty + sebi_charges)

    def select_strikes(self, spot: float) -> IronCondorStrikes:
        """Constructs 4 strikes based on structure type (Iron Condor vs Iron Fly)."""
        if self.config.structure_type.upper() == "IRON_FLY":
            return self.strike_engine.build_iron_fly_strikes(spot, wing_width=self.config.wing_width)
        else:
            return self.strike_engine.build_iron_condor_strikes(
                spot, short_offset=self.config.short_offset, wing_width=self.config.wing_width
            )

    def run_session(
        self,
        options_file_path: str,
        spot_df: pd.DataFrame,
        date_str: str,
        r: float = 0.065
    ) -> Optional[IronCondorSessionResult]:
        """Runs a complete 1-minute 4-leg simulation for a single trading session."""
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

        # 2. Build 4-leg strike structure
        strikes = self.select_strikes(anchor_spot)

        # 3. Fast batch fetch 4 contracts in 1 single pass
        parser = OptionsChainParser(options_file_path)
        contracts = [
            (strikes.short_put, "PE"),
            (strikes.long_put, "PE"),
            (strikes.short_call, "CE"),
            (strikes.long_call, "CE"),
        ]
        batch_df = parser.get_multiple_contracts(contracts)
        if batch_df.empty:
            return None

        sp_col = f"{strikes.short_put}PE_price"
        lp_col = f"{strikes.long_put}PE_price"
        sc_col = f"{strikes.short_call}CE_price"
        lc_col = f"{strikes.long_call}CE_price"

        for col in (sp_col, lp_col, sc_col, lc_col):
            if col not in batch_df.columns:
                return None

        batch_df["time_str"] = batch_df["DateTime"].dt.strftime("%H:%M:%S")
        trade_bars = batch_df[batch_df["time_str"] >= entry_time].copy().reset_index(drop=True)
        if trade_bars.empty:
            return None

        # 4. Entry execution
        entry_row = trade_bars.iloc[0]
        sp_entry = float(entry_row[sp_col])
        lp_entry = float(entry_row[lp_col])
        sc_entry = float(entry_row[sc_col])
        lc_entry = float(entry_row[lc_col])

        # Net credit = (Sold Put + Sold Call) - (Bought Put Wing + Bought Call Wing)
        net_credit_pts = (sp_entry + sc_entry) - (lp_entry + lc_entry)
        max_risk_pts = float(strikes.wing_width) - net_credit_pts

        sp_leg = LegState(symbol=f"{strikes.short_put}PE", entry_time=entry_row["time_str"], entry_price=sp_entry, side="SHORT")
        lp_leg = LegState(symbol=f"{strikes.long_put}PE", entry_time=entry_row["time_str"], entry_price=lp_entry, side="LONG")
        sc_leg = LegState(symbol=f"{strikes.short_call}CE", entry_time=entry_row["time_str"], entry_price=sc_entry, side="SHORT")
        lc_leg = LegState(symbol=f"{strikes.long_call}CE", entry_time=entry_row["time_str"], entry_price=lc_entry, side="LONG")

        sp_hist = [sp_entry]
        sc_hist = [sc_entry]
        bar_attributions: List[BarAttribution] = []

        curr_dt = pd.to_datetime(f"{date_str} {entry_time}")
        days_to_thursday = (3 - curr_dt.weekday()) % 7

        prev_spot = anchor_spot
        prev_sp = sp_entry
        prev_lp = lp_entry
        prev_sc = sc_entry
        prev_lc = lc_entry
        prev_iv = 0.20

        # 5. Intraday execution loop
        for i in range(1, len(trade_bars)):
            bar = trade_bars.iloc[i]
            t_str = bar["time_str"]
            sp_px = float(bar[sp_col])
            lp_px = float(bar[lp_col])
            sc_px = float(bar[sc_col])
            lc_px = float(bar[lc_col])

            sp_hist.append(sp_px)
            sc_hist.append(sc_px)

            # Update short legs with active stop-loss monitoring
            if sp_leg.is_active:
                sp_leg = self.leg_cutter.update_leg(sp_leg, t_str, sp_px, pd.Series(sp_hist))
            if sc_leg.is_active:
                sc_leg = self.leg_cutter.update_leg(sc_leg, t_str, sc_px, pd.Series(sc_hist))

            # If hard EOD reached, close long wings too
            if t_str >= self.config.leg_config.hard_cutoff_time:
                if lp_leg.is_active:
                    lp_leg.is_active = False
                    lp_leg.exit_price = lp_px
                    lp_leg.exit_reason = "EOD_SQUAREOFF"
                if lc_leg.is_active:
                    lc_leg.is_active = False
                    lc_leg.exit_price = lc_px
                    lc_leg.exit_reason = "EOD_SQUAREOFF"

            curr_spot = float(spot_map.get(t_str, prev_spot))
            dS = curr_spot - prev_spot

            # Bar PnL across 4 legs
            d_sp = (prev_sp - sp_px) if sp_leg.is_active else 0.0
            d_lp = (lp_px - prev_lp) if lp_leg.is_active else 0.0
            d_sc = (prev_sc - sc_px) if sc_leg.is_active else 0.0
            d_lc = (lc_px - prev_lc) if lc_leg.is_active else 0.0

            d_pnl_points = d_sp + d_lp + d_sc + d_lc
            actual_bar_pnl = d_pnl_points * self.total_qty

            hour_val = int(t_str[:2]) + int(t_str[3:5]) / 60.0
            day_fraction_left = max((15.5 - hour_val) / 6.25, 0.01)
            T_current = max((days_to_thursday + day_fraction_left) / 365.0, 1e-4)

            # Greek decomposition
            delta_pos = 0.0
            gamma_pos = 0.0
            theta_daily_pos = 0.0
            vega_pos = 0.0

            # Short Call (-1)
            if sc_leg.is_active and sc_px > 0:
                iv = ImpliedVolatilitySolver.solve_single(sc_px, curr_spot, strikes.short_call, T_current, r, "CE") or prev_iv
                g = Black76.greeks(curr_spot, strikes.short_call, T_current, r, iv, "CE")
                delta_pos += (-g.delta) * self.total_qty
                gamma_pos += (-g.gamma) * self.total_qty
                theta_daily_pos += (-g.theta / 252.0) * self.total_qty
                vega_pos += (-g.vega) * self.total_qty

            # Long Call (+1)
            if lc_leg.is_active and lc_px > 0:
                iv = ImpliedVolatilitySolver.solve_single(lc_px, curr_spot, strikes.long_call, T_current, r, "CE") or prev_iv
                g = Black76.greeks(curr_spot, strikes.long_call, T_current, r, iv, "CE")
                delta_pos += (+g.delta) * self.total_qty
                gamma_pos += (+g.gamma) * self.total_qty
                theta_daily_pos += (+g.theta / 252.0) * self.total_qty
                vega_pos += (+g.vega) * self.total_qty

            # Short Put (-1)
            if sp_leg.is_active and sp_px > 0:
                iv = ImpliedVolatilitySolver.solve_single(sp_px, curr_spot, strikes.short_put, T_current, r, "PE") or prev_iv
                g = Black76.greeks(curr_spot, strikes.short_put, T_current, r, iv, "PE")
                delta_pos += (-g.delta) * self.total_qty
                gamma_pos += (-g.gamma) * self.total_qty
                theta_daily_pos += (-g.theta / 252.0) * self.total_qty
                vega_pos += (-g.vega) * self.total_qty

            # Long Put (+1)
            if lp_leg.is_active and lp_px > 0:
                iv = ImpliedVolatilitySolver.solve_single(lp_px, curr_spot, strikes.long_put, T_current, r, "PE") or prev_iv
                g = Black76.greeks(curr_spot, strikes.long_put, T_current, r, iv, "PE")
                delta_pos += (+g.delta) * self.total_qty
                gamma_pos += (+g.gamma) * self.total_qty
                theta_daily_pos += (+g.theta / 252.0) * self.total_qty
                vega_pos += (+g.vega) * self.total_qty

            bar_attr = GreekPnLAttributor.attribute_bar(
                delta_pos=delta_pos,
                gamma_pos=gamma_pos,
                theta_pos_daily=theta_daily_pos,
                vega_pos_per_vol=vega_pos,
                dS=dS,
                dVol=0.0,
                dt_days=1.0 / 375.0,
                actual_pnl=actual_bar_pnl,
                timestamp=t_str
            )
            bar_attributions.append(bar_attr)

            prev_spot = curr_spot
            prev_sp = sp_px
            prev_lp = lp_px
            prev_sc = sc_px
            prev_lc = lc_px

            if not sp_leg.is_active and not sc_leg.is_active and not lp_leg.is_active and not lc_leg.is_active:
                break

        # 6. Final settlement & metrics
        sp_exit = sp_leg.exit_price if sp_leg.exit_price is not None else sp_hist[-1]
        lp_exit = lp_leg.exit_price if lp_leg.exit_price is not None else float(trade_bars.iloc[-1][lp_col])
        sc_exit = sc_leg.exit_price if sc_leg.exit_price is not None else sc_hist[-1]
        lc_exit = lc_leg.exit_price if lc_leg.exit_price is not None else float(trade_bars.iloc[-1][lc_col])

        sp_pnl = sp_entry - sp_exit
        lp_pnl = lp_exit - lp_entry
        sc_pnl = sc_entry - sc_exit
        lc_pnl = lc_exit - lc_entry

        gross_pts = sp_pnl + lp_pnl + sc_pnl + lc_pnl
        gross_rupees = gross_pts * self.total_qty

        sell_turnover = (sp_entry + sc_entry + lp_exit + lc_exit) * self.total_qty
        buy_turnover = (lp_entry + lc_entry + sp_exit + sc_exit) * self.total_qty
        statutory_costs = self.calculate_statutory_costs(sell_turnover, buy_turnover, num_orders=8)

        net_rupees = gross_rupees - statutory_costs
        net_pts = net_rupees / self.total_qty if self.total_qty > 0 else 0.0
        aggregated_attr = GreekPnLAttributor.aggregate_attribution(bar_attributions)

        return IronCondorSessionResult(
            date=date_str,
            entry_spot=anchor_spot,
            structure_type=self.config.structure_type.upper(),
            short_put=strikes.short_put,
            long_put=strikes.long_put,
            short_call=strikes.short_call,
            long_call=strikes.long_call,
            wing_width=strikes.wing_width,
            net_credit_entry=round(net_credit_pts, 2),
            max_risk_points=round(max_risk_pts, 2),
            gross_points=round(gross_pts, 2),
            net_points=round(net_pts, 2),
            gross_pnl_rupees=round(gross_rupees, 2),
            net_pnl_rupees=round(net_rupees, 2),
            statutory_costs_rupees=round(statutory_costs, 2),
            short_put_reason=sp_leg.exit_reason or "RUNNING",
            short_call_reason=sc_leg.exit_reason or "RUNNING",
            attributions=aggregated_attr,
            bar_attributions=bar_attributions
        )

    def run_backtest(
        self,
        options_dir: str,
        spot_csv_or_df: Union[str, pd.DataFrame],
        dates: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, BacktestMetrics]:
        """Runs multi-session Iron Condor / Iron Fly backtest across available dates."""
        if isinstance(spot_csv_or_df, str):
            spot_df = pd.read_csv(spot_csv_or_df)
        else:
            spot_df = spot_csv_or_df.copy()

        if not pd.api.types.is_datetime64_any_dtype(spot_df["date"]):
            spot_df["date"] = pd.to_datetime(spot_df["date"])

        pattern = os.path.join(options_dir, "NIFTY_*.csv")
        files = sorted(glob.glob(pattern))
        results: List[IronCondorSessionResult] = []

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
                "structure_type": r.structure_type,
                "short_put": r.short_put,
                "long_put": r.long_put,
                "short_call": r.short_call,
                "long_call": r.long_call,
                "wing_width": r.wing_width,
                "net_credit": r.net_credit_entry,
                "max_risk": r.max_risk_points,
                "gross_points": r.gross_points,
                "net_points": r.net_points,
                "gross_pnl": r.gross_pnl_rupees,
                "statutory_costs": r.statutory_costs_rupees,
                "net_pnl": r.net_pnl_rupees,
                "short_put_reason": r.short_put_reason,
                "short_call_reason": r.short_call_reason,
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
