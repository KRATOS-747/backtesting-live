"""
High-Performance Options Chain Parser using Polars & NumPy
Efficiently reads, pivots, batches, and structures NSE weekly options chain snapshots.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Union
import polars as pl
import pandas as pd
import numpy as np


class OptionsChainParser:
    """
    Parses and indexes wide-format NSE intraday options chain CSVs.
    Features:
    - Zero-copy single-pass Polars scanning.
    - Multi-contract batch extraction (e.g. 5-strike grids or straddles in 1 query).
    - Single-minute full chain snapshot extraction across all strikes.
    - In-memory DataFrame caching for fast backtest loops.
    """

    def __init__(self, file_path: str, enable_cache: bool = True):
        self.file_path = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Options chain file not found: {file_path}")
        self.enable_cache = enable_cache
        self._cached_pldf: Optional[pl.DataFrame] = None
        self._available_columns: Optional[List[str]] = None

    def _get_columns(self) -> List[str]:
        if self._available_columns is None:
            df_header = pl.read_csv(self.file_path, n_rows=1)
            self._available_columns = df_header.columns
        return self._available_columns

    def load_polars_raw(self) -> pl.DataFrame:
        """Loads the raw options CSV into a Polars DataFrame with optimized schema."""
        if self._cached_pldf is not None:
            return self._cached_pldf
        pldf = pl.read_csv(self.file_path, infer_schema_length=10000, ignore_errors=True)
        if self.enable_cache:
            self._cached_pldf = pldf
        return pldf

    def extract_available_strikes(self) -> Tuple[List[int], List[int]]:
        """
        Discovers all available Call and Put strikes present in the file headers.
        Returns: (call_strikes_sorted, put_strikes_sorted)
        """
        columns = self._get_columns()
        call_strikes: Set[int] = set()
        put_strikes: Set[int] = set()

        for col in columns:
            if "CE_close" in col or "CE_OI" in col:
                strike_str = col.split("CE")[0]
                if strike_str.isdigit():
                    call_strikes.add(int(strike_str))
            elif "PE_close" in col or "PE_OI" in col:
                strike_str = col.split("PE")[0]
                if strike_str.isdigit():
                    put_strikes.add(int(strike_str))

        return sorted(list(call_strikes)), sorted(list(put_strikes))

    def get_contract_series(self, strike: int, option_type: str = "CE") -> pd.DataFrame:
        """
        Extracts a clean, aligned 1-minute time series of price and OI for a single contract.
        Returns Pandas DataFrame with 'DateTime', 'price', and 'oi'.
        """
        opt_suffix = option_type.upper()
        close_col = f"{strike}{opt_suffix}_close"
        oi_col = f"{strike}{opt_suffix}_OI"
        cols_available = set(self._get_columns())

        if close_col not in cols_available:
            return pd.DataFrame(columns=["DateTime", "price", "oi"])

        select_cols = ["DateTime", close_col]
        has_oi = oi_col in cols_available
        if has_oi:
            select_cols.append(oi_col)

        try:
            if self._cached_pldf is not None:
                pldf = self._cached_pldf.select([c for c in select_cols if c in self._cached_pldf.columns])
            else:
                pldf = pl.read_csv(self.file_path, columns=select_cols, ignore_errors=True)

            pldf = pldf.drop_nulls()
            df = pldf.to_pandas()
            df["DateTime"] = pd.to_datetime(df["DateTime"]).dt.tz_localize(None)
            rename_map = {close_col: "price"}
            if has_oi:
                rename_map[oi_col] = "oi"
            else:
                df["oi"] = 0.0
            df = df.rename(columns=rename_map)
            return df.sort_values("DateTime").reset_index(drop=True)
        except Exception:
            return pd.DataFrame(columns=["DateTime", "price", "oi"])

    def get_multiple_contracts(self, contracts: List[Tuple[int, str]]) -> pd.DataFrame:
        """
        High-performance single-pass batch extraction for multiple option contracts simultaneously.
        contracts: List of (strike, 'CE' or 'PE') e.g. [(23700, 'CE'), (23700, 'PE'), (23750, 'CE'), ...]
        Returns DataFrame with 'DateTime' and columns '{strike}{type}_price' and '{strike}{type}_oi'.
        """
        available_cols = set(self._get_columns())
        select_cols = ["DateTime"]
        rename_map = {}

        for strike, opt_type in contracts:
            suffix = opt_type.upper()
            close_col = f"{strike}{suffix}_close"
            oi_col = f"{strike}{suffix}_OI"

            if close_col in available_cols:
                select_cols.append(close_col)
                rename_map[close_col] = f"{strike}{suffix}_price"
            if oi_col in available_cols:
                select_cols.append(oi_col)
                rename_map[oi_col] = f"{strike}{suffix}_oi"

        if len(select_cols) <= 1:
            return pd.DataFrame(columns=["DateTime"])

        try:
            if self._cached_pldf is not None:
                pldf = self._cached_pldf.select([c for c in select_cols if c in self._cached_pldf.columns])
            else:
                pldf = pl.read_csv(self.file_path, columns=select_cols, ignore_errors=True)

            df = pldf.drop_nulls().to_pandas()
            df["DateTime"] = pd.to_datetime(df["DateTime"]).dt.tz_localize(None)
            df = df.rename(columns=rename_map)
            return df.sort_values("DateTime").reset_index(drop=True)
        except Exception:
            return pd.DataFrame(columns=["DateTime"])

    def get_straddle_series(self, atm_strike: int) -> pd.DataFrame:
        """
        Convenience method: extracts aligned 1-minute Call & Put series for an ATM Straddle in 1 pass.
        Returns DataFrame with: ['DateTime', 'time_str', 'ce_price', 'pe_price', 'straddle_price', 'ce_oi', 'pe_oi', 'total_oi']
        """
        df_batch = self.get_multiple_contracts([(atm_strike, "CE"), (atm_strike, "PE")])
        if df_batch.empty:
            return pd.DataFrame()

        ce_px_col = f"{atm_strike}CE_price"
        pe_px_col = f"{atm_strike}PE_price"
        ce_oi_col = f"{atm_strike}CE_oi"
        pe_oi_col = f"{atm_strike}PE_oi"

        if ce_px_col not in df_batch.columns or pe_px_col not in df_batch.columns:
            return pd.DataFrame()

        df_batch["time_str"] = df_batch["DateTime"].dt.strftime("%H:%M:%S")
        df_batch["ce_price"] = df_batch[ce_px_col]
        df_batch["pe_price"] = df_batch[pe_px_col]
        df_batch["straddle_price"] = df_batch["ce_price"] + df_batch["pe_price"]

        df_batch["ce_oi"] = df_batch[ce_oi_col] if ce_oi_col in df_batch.columns else 0.0
        df_batch["pe_oi"] = df_batch[pe_oi_col] if pe_oi_col in df_batch.columns else 0.0
        df_batch["total_oi"] = df_batch["ce_oi"] + df_batch["pe_oi"]

        keep_cols = ["DateTime", "time_str", "ce_price", "pe_price", "straddle_price", "ce_oi", "pe_oi", "total_oi"]
        return df_batch[keep_cols].copy()

    def get_chain_snapshot(self, time_str: str = "09:25:00") -> pd.DataFrame:
        """
        Extracts complete options chain snapshot across ALL available strikes at a specific minute bar.
        Returns DataFrame with: ['strike', 'ce_close', 'pe_close', 'ce_oi', 'pe_oi', 'total_oi', 'straddle_price']
        """
        pldf = self.load_polars_raw()
        df = pldf.to_pandas()
        df["DateTime"] = pd.to_datetime(df["DateTime"]).dt.tz_localize(None)

        filtered = df[df["DateTime"].dt.strftime("%H:%M:%S") >= time_str]
        if filtered.empty:
            filtered = df.iloc[-1:]
        row = filtered.iloc[0]

        ce_strikes, pe_strikes = self.extract_available_strikes()
        all_strikes = sorted(list(set(ce_strikes).union(pe_strikes)))

        snapshot_rows = []
        for K in all_strikes:
            ce_val = row.get(f"{K}CE_close", 0.0)
            pe_val = row.get(f"{K}PE_close", 0.0)
            ce_oi_val = row.get(f"{K}CE_OI", 0.0)
            pe_oi_val = row.get(f"{K}PE_OI", 0.0)

            ce_c = float(ce_val) if pd.notna(ce_val) else 0.0
            pe_c = float(pe_val) if pd.notna(pe_val) else 0.0
            ce_oi = float(ce_oi_val) if pd.notna(ce_oi_val) else 0.0
            pe_oi = float(pe_oi_val) if pd.notna(pe_oi_val) else 0.0

            snapshot_rows.append({
                "strike": K,
                "ce_close": ce_c,
                "pe_close": pe_c,
                "ce_oi": ce_oi,
                "pe_oi": pe_oi,
                "total_oi": ce_oi + pe_oi,
                "straddle_price": round(ce_c + pe_c, 2) if (ce_c > 0 and pe_c > 0) else np.nan
            })

        return pd.DataFrame(snapshot_rows)

    def resample_bars(self, contract_df: pd.DataFrame, timeframe: str = "5min") -> pd.DataFrame:
        """Resamples 1-minute contract price and OI into discrete N-minute candles."""
        if contract_df.empty:
            return contract_df

        df = contract_df.copy().set_index("DateTime")
        resampled_price = df["price"].resample(timeframe).last()
        resampled_oi = df["oi"].resample(timeframe).last() if "oi" in df.columns else pd.Series(0.0, index=resampled_price.index)

        resampled = pd.DataFrame({"price": resampled_price, "oi": resampled_oi}).dropna().reset_index()
        return resampled
