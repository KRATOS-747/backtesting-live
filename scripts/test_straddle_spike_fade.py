"""
Empirical Audit: Event-Driven Straddle Selling at Spike Rejections
Instead of selling naked legs, sell ATM Straddle (both CE + PE) when
underlying tests an OI Wall and momentum stalls.
Audited across all 23 trading sessions of January 2025.
"""

import glob
import os
import re
import numpy as np
import pandas as pd
import polars as pl

# 1. Load spot
spot_path = "sample_data/spot/nifty50_1min_sample.csv"
spot_df = pd.read_csv(spot_path)
spot_df["date"] = pd.to_datetime(spot_df["date"])
spot_df["day_str"] = spot_df["date"].dt.strftime("%Y-%m-%d")
spot_df["time_str"] = spot_df["date"].dt.strftime("%H:%M:%S")
spot_df = spot_df.sort_values("date").reset_index(drop=True)
spot_df["spot_roc_1m"] = spot_df["close"].pct_change(1) * 100.0
spot_df["spot_roc_5m"] = spot_df["close"].pct_change(5) * 100.0

spot_lookup = {}
for _, row in spot_df.iterrows():
    spot_lookup[(row["day_str"], row["time_str"])] = {
        "close": row["close"],
        "roc_1m": row["spot_roc_1m"],
        "roc_5m": row["spot_roc_5m"],
    }

files = sorted(glob.glob("sample_data/options/NIFTY_*.csv"))

def run_event_straddle_audit(holding_bars=30, profit_target_pct=0.12, stop_loss_pct=0.15):
    trades = []

    for fpath in files:
        fname = os.path.basename(fpath)
        m = re.search(r"NIFTY_(\d{4})(\d{2})(\d{2})", fname)
        if not m:
            continue
        day_str = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

        pldf = pl.read_csv(fpath, ignore_errors=True)
        pdf = pldf.to_pandas()
        if "DateTime" not in pdf.columns:
            continue
        pdf["DateTime"] = pd.to_datetime(pdf["DateTime"]).dt.tz_localize(None)
        pdf["time_str"] = pdf["DateTime"].dt.strftime("%H:%M:%S")
        pdf = pdf.sort_values("DateTime").reset_index(drop=True)

        times = pdf["time_str"].values
        n_bars = len(pdf)

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

        call_wall_idx = np.argmax(ce_oi_mat, axis=1)
        call_walls = ce_oi_strikes[call_wall_idx] if len(ce_oi_strikes) > 0 else np.zeros(n_bars)

        put_wall_idx = np.argmax(pe_oi_mat, axis=1)
        put_walls = pe_oi_strikes[put_wall_idx] if len(pe_oi_strikes) > 0 else np.zeros(n_bars)

        ce_k_to_idx = {k: i for i, k in enumerate(ce_strikes)}
        pe_k_to_idx = {k: i for i, k in enumerate(pe_strikes)}

        active_until_idx = -1

        for idx in range(5, n_bars - holding_bars):
            t_str = times[idx]
            if not ("09:30:00" <= t_str <= "14:45:00"):
                continue

            if idx < active_until_idx:
                continue

            spot_info = spot_lookup.get((day_str, t_str))
            if not spot_info or pd.isna(spot_info["close"]):
                continue

            spot_px = spot_info["close"]
            spot_roc1 = spot_info["roc_1m"]
            call_wall = call_walls[idx]
            put_wall = put_walls[idx]

            # Spike condition: Spot testing Call Wall with momentum stall OR Spot testing Put Wall with momentum stall
            is_call_surge_stall = (abs(spot_px - call_wall) <= 35) and (spot_roc1 <= 0.02) and (spot_info["roc_5m"] > 0.15)
            is_put_surge_stall = (abs(spot_px - put_wall) <= 35) and (spot_roc1 >= -0.02) and (spot_info["roc_5m"] < -0.15)

            if not (is_call_surge_stall or is_put_surge_stall):
                continue

            # Sell ATM Straddle at this moment
            atm_k = int(round(spot_px / 50.0) * 50)
            ce_idx = ce_k_to_idx.get(atm_k)
            pe_idx = pe_k_to_idx.get(atm_k)

            if ce_idx is None or pe_idx is None:
                continue

            ce_entry = ce_close_mat[idx, ce_idx]
            pe_entry = pe_close_mat[idx, pe_idx]

            if ce_entry <= 5 or pe_entry <= 5:
                continue

            straddle_entry = ce_entry + pe_entry

            # Path simulation across holding_bars
            straddle_exit = ce_close_mat[idx + holding_bars, ce_idx] + pe_close_mat[idx + holding_bars, pe_idx]
            exit_reason = f"TIME_{holding_bars}M"
            actual_bars = holding_bars

            for step in range(1, holding_bars + 1):
                cur_ce = ce_close_mat[idx + step, ce_idx]
                cur_pe = pe_close_mat[idx + step, pe_idx]
                cur_straddle = cur_ce + cur_pe

                # Stop loss check (straddle premium increased)
                if (cur_straddle - straddle_entry) / straddle_entry >= stop_loss_pct:
                    straddle_exit = cur_straddle
                    exit_reason = "STOP_LOSS"
                    actual_bars = step
                    break

                # Profit target check (straddle premium decayed)
                if (straddle_entry - cur_straddle) / straddle_entry >= profit_target_pct:
                    straddle_exit = cur_straddle
                    exit_reason = "PROFIT_TARGET"
                    actual_bars = step
                    break

            pts_pnl = straddle_entry - straddle_exit
            gross_pnl = pts_pnl * 75.0

            # Friction for 2-leg Straddle round trip:
            # 4 legs total: Sell CE, Sell PE, Buy CE, Buy PE
            # Brokerage: 4 * 20 = 80
            # STT: 0.1% on sell side total premium
            turnover_sell = straddle_entry * 75.0
            turnover_buy = straddle_exit * 75.0
            stt = 0.001 * turnover_sell
            exch_turnover = 0.0005 * (turnover_sell + turnover_buy)
            brokerage = 80.0
            gst = 0.18 * (brokerage + exch_turnover)
            stamp = 0.00003 * turnover_buy
            slippage = 2.0 * 75.0  # 0.5 pt per leg across 4 executions = 2.0 pts = 150 INR
            friction = stt + exch_turnover + brokerage + gst + stamp + slippage
            net_pnl = gross_pnl - friction

            trade = {
                "date": day_str,
                "time": t_str,
                "strike": atm_k,
                "spot": spot_px,
                "straddle_entry": straddle_entry,
                "straddle_exit": straddle_exit,
                "pts_pnl": pts_pnl,
                "gross_pnl_inr": gross_pnl,
                "friction_inr": friction,
                "net_pnl_inr": net_pnl,
                "is_win": net_pnl > 0,
                "exit_reason": exit_reason,
                "holding_bars": actual_bars,
            }
            trades.append(trade)
            active_until_idx = idx + actual_bars

    return pd.DataFrame(trades)

if __name__ == "__main__":
    df_trades = run_event_straddle_audit(holding_bars=30, profit_target_pct=0.12, stop_loss_pct=0.15)
    print("=======================================================")
    print("   EMPIRICAL AUDIT: EVENT-DRIVEN STRADDLE SELLING     ")
    print("   (Selling ATM Straddle at OI Wall Spike Rejections)  ")
    print("=======================================================")
    print(f"Total Trades:           {len(df_trades)}")
    if not df_trades.empty:
        wr = (df_trades['is_win'].sum() / len(df_trades)) * 100.0
        gross = df_trades['gross_pnl_inr'].sum()
        fric = df_trades['friction_inr'].sum()
        net = df_trades['net_pnl_inr'].sum()
        conv = (net / gross * 100.0) if gross > 0 else 0.0
        avg_gross = gross / len(df_trades)
        avg_fric = fric / len(df_trades)
        avg_net = net / len(df_trades)
        print(f"Win Rate:               {wr:.1f}%")
        print(f"Gross P&L:              ₹{gross:,.2f}")
        print(f"Total Friction:         ₹{fric:,.2f}")
        print(f"Net Realized P&L:       ₹{net:,.2f}")
        print(f"Net-to-Gross Conv:      {conv:.1f}%")
        print(f"Avg Straddle Entry Px:  ₹{df_trades['straddle_entry'].mean():.2f}")
        print(f"Avg Gross / Trade:      ₹{avg_gross:,.2f} ({avg_gross/75:.2f} pts)")
        print(f"Avg Friction / Trade:   ₹{avg_fric:,.2f} ({avg_fric/75:.2f} pts)")
        print(f"Avg Net / Trade:        ₹{avg_net:,.2f} ({avg_net/75:.2f} pts)")
        print(f"\nExit Breakdown:\n{df_trades['exit_reason'].value_counts().to_string()}")
    print("=======================================================")
