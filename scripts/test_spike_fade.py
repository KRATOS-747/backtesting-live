"""
High-Performance Empirical Research Audit: Spike Fading Hypothesis & Polish
Comparing:
1. Naive Spike Fade (Raw Edge)
2. Conditioned Spike Fade (Without Debounce)
3. Option 1: Debounced Single-Position Engine (Max 1 concurrent position, highest OI strike)
Across all 23 trading days of January 2025 on 1-minute NIFTY option chains.
"""

import glob
import os
import re
import numpy as np
import pandas as pd
import polars as pl

# 1. Load spot 1-min data
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

def run_spike_fade_audit(surge_pct_threshold=12.0, min_premium=30.0, max_moneyness=200):
    naive_trades = []
    eng_raw_trades = []
    option1_debounced_trades = []

    print(f"Starting Vectorized Spike Fade Audit (including Option 1 Debounce) across {len(files)} sessions...", flush=True)

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
        ce_oi_k_to_idx = {k: i for i, k in enumerate(ce_oi_strikes)}
        pe_oi_k_to_idx = {k: i for i, k in enumerate(pe_oi_strikes)}

        # Tracking active position for Option 1 (Debounced Single-Position)
        active_position_until_idx = -1

        for idx in range(5, n_bars - 15):
            t_str = times[idx]
            if not ("09:30:00" <= t_str <= "14:45:00"):
                continue

            spot_info = spot_lookup.get((day_str, t_str))
            if not spot_info or pd.isna(spot_info["close"]):
                continue

            spot_px = spot_info["close"]
            spot_roc1 = spot_info["roc_1m"]
            call_wall = call_walls[idx]
            put_wall = put_walls[idx]

            atm_strike = int(round(spot_px / 50.0) * 50)
            target_strikes = [atm_strike + offset for offset in range(-max_moneyness, max_moneyness + 50, 50)]

            candidates_at_this_bar = []

            # 1. CE Checks
            for k in target_strikes:
                col_idx = ce_k_to_idx.get(k)
                if col_idx is None:
                    continue

                p_now = ce_close_mat[idx, col_idx]
                p_prev5 = ce_close_mat[idx - 5, col_idx]

                if p_now < min_premium or p_prev5 <= 0:
                    continue

                surge_pct = ((p_now - p_prev5) / p_prev5) * 100.0
                if surge_pct >= surge_pct_threshold:
                    p_entry = p_now
                    p_exit = ce_close_mat[idx + 15, col_idx]
                    exit_reason = "TIME_15M"
                    actual_exit_step = 15

                    future_pxs = ce_close_mat[idx + 1 : idx + 16, col_idx]
                    for step_i, px in enumerate(future_pxs):
                        if px <= 0:
                            continue
                        if (px - p_entry) / p_entry >= 0.25:
                            p_exit = px
                            exit_reason = "STOP_LOSS"
                            actual_exit_step = step_i + 1
                            break
                        if (p_entry - px) / p_entry >= 0.20:
                            p_exit = px
                            exit_reason = "PROFIT_TARGET"
                            actual_exit_step = step_i + 1
                            break

                    pts_pnl = p_entry - p_exit
                    gross_pnl = pts_pnl * 75.0
                    turnover_sell = p_entry * 75.0
                    turnover_buy = p_exit * 75.0
                    stt = 0.001 * turnover_sell
                    exch_turnover = 0.0005 * (turnover_sell + turnover_buy)
                    brokerage = 40.0
                    gst = 0.18 * (brokerage + exch_turnover)
                    stamp = 0.00003 * turnover_buy
                    slippage = 1.0 * 75.0
                    friction = stt + exch_turnover + brokerage + gst + stamp + slippage
                    net_pnl = gross_pnl - friction

                    trade = {
                        "date": day_str,
                        "time": t_str,
                        "strike": k,
                        "type": "CE",
                        "p_entry": p_entry,
                        "p_exit": p_exit,
                        "surge_pct": surge_pct,
                        "pts_pnl": pts_pnl,
                        "gross_pnl_inr": gross_pnl,
                        "friction_inr": friction,
                        "net_pnl_inr": net_pnl,
                        "is_win": net_pnl > 0,
                        "exit_reason": exit_reason,
                        "exit_step": actual_exit_step,
                    }
                    naive_trades.append(trade)

                    # Conditioned: Spot within 35 pts of Call Wall & Spot 1m ROC <= 0.02%
                    if abs(spot_px - call_wall) <= 35 and spot_roc1 <= 0.02:
                        eng_raw_trades.append(trade)
                        
                        # Get contract OI for ranking candidates
                        oi_idx = ce_oi_k_to_idx.get(k)
                        oi_val = ce_oi_mat[idx, oi_idx] if oi_idx is not None else 0.0
                        candidates_at_this_bar.append((oi_val, trade))

            # 2. PE Checks
            for k in target_strikes:
                col_idx = pe_k_to_idx.get(k)
                if col_idx is None:
                    continue

                p_now = pe_close_mat[idx, col_idx]
                p_prev5 = pe_close_mat[idx - 5, col_idx]

                if p_now < min_premium or p_prev5 <= 0:
                    continue

                surge_pct = ((p_now - p_prev5) / p_prev5) * 100.0
                if surge_pct >= surge_pct_threshold:
                    p_entry = p_now
                    p_exit = pe_close_mat[idx + 15, col_idx]
                    exit_reason = "TIME_15M"
                    actual_exit_step = 15

                    future_pxs = pe_close_mat[idx + 1 : idx + 16, col_idx]
                    for step_i, px in enumerate(future_pxs):
                        if px <= 0:
                            continue
                        if (px - p_entry) / p_entry >= 0.25:
                            p_exit = px
                            exit_reason = "STOP_LOSS"
                            actual_exit_step = step_i + 1
                            break
                        if (p_entry - px) / p_entry >= 0.20:
                            p_exit = px
                            exit_reason = "PROFIT_TARGET"
                            actual_exit_step = step_i + 1
                            break

                    pts_pnl = p_entry - p_exit
                    gross_pnl = pts_pnl * 75.0
                    turnover_sell = p_entry * 75.0
                    turnover_buy = p_exit * 75.0
                    stt = 0.001 * turnover_sell
                    exch_turnover = 0.0005 * (turnover_sell + turnover_buy)
                    brokerage = 40.0
                    gst = 0.18 * (brokerage + exch_turnover)
                    stamp = 0.00003 * turnover_buy
                    slippage = 1.0 * 75.0
                    friction = stt + exch_turnover + brokerage + gst + stamp + slippage
                    net_pnl = gross_pnl - friction

                    trade = {
                        "date": day_str,
                        "time": t_str,
                        "strike": k,
                        "type": "PE",
                        "p_entry": p_entry,
                        "p_exit": p_exit,
                        "surge_pct": surge_pct,
                        "pts_pnl": pts_pnl,
                        "gross_pnl_inr": gross_pnl,
                        "friction_inr": friction,
                        "net_pnl_inr": net_pnl,
                        "is_win": net_pnl > 0,
                        "exit_reason": exit_reason,
                        "exit_step": actual_exit_step,
                    }
                    naive_trades.append(trade)

                    if abs(spot_px - put_wall) <= 35 and spot_roc1 >= -0.02:
                        eng_raw_trades.append(trade)
                        oi_idx = pe_oi_k_to_idx.get(k)
                        oi_val = pe_oi_mat[idx, oi_idx] if oi_idx is not None else 0.0
                        candidates_at_this_bar.append((oi_val, trade))

            # Option 1: Debounce Logic
            # Only enter if not currently in an active trade
            if idx >= active_position_until_idx and candidates_at_this_bar:
                # Sort candidates by OI descending -> pick the single strike at the heart of the institutional wall
                candidates_at_this_bar.sort(key=lambda x: x[0], reverse=True)
                best_trade = candidates_at_this_bar[0][1]
                option1_debounced_trades.append(best_trade)
                # Lock until trade has fully exited
                active_position_until_idx = idx + best_trade["exit_step"]

    print("Audit computation completed successfully!", flush=True)
    return pd.DataFrame(naive_trades), pd.DataFrame(eng_raw_trades), pd.DataFrame(option1_debounced_trades)

if __name__ == "__main__":
    df_naive, df_eng, df_opt1 = run_spike_fade_audit()
    print("\n=======================================================")
    print("      EMPIRICAL RESEARCH AUDIT: SPIKE FADING           ")
    print("      (All 23 Sessions of January 2025 - NIFTY)        ")
    print("=======================================================")
    
    for name, df in [
        ("1. NAIVE SPIKE FADING (Raw Edge)", df_naive),
        ("2. CONDITIONED SPIKE FADING (No Debounce)", df_eng),
        ("3. OPTION 1: DEBOUNCED SINGLE-POSITION (Polish)", df_opt1)
    ]:
        print(f"\n{name}:")
        print(f"   Total Trades:       {len(df)}")
        if not df.empty:
            wr = (df['is_win'].sum() / len(df)) * 100.0
            gross = df['gross_pnl_inr'].sum()
            fric = df['friction_inr'].sum()
            net = df['net_pnl_inr'].sum()
            conv = (net / gross * 100.0) if gross > 0 else 0.0
            print(f"   Win Rate:           {wr:.1f}%")
            print(f"   Gross P&L:          ₹{gross:,.2f}")
            print(f"   Total Friction:     ₹{fric:,.2f}")
            print(f"   Net Realized P&L:   ₹{net:,.2f}")
            print(f"   Net-to-Gross Conv:  {conv:.1f}%")
            print(f"   Avg Net per Trade:  ₹{df['net_pnl_inr'].mean():,.2f}")
            print(f"   Exit Breakdown:\n{df['exit_reason'].value_counts().to_string()}")
    print("=======================================================")
