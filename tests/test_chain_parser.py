"""
Unit Tests for High-Performance Options Chain Parser (options_bt/chain_parser.py)
"""

import os
import pytest
import pandas as pd
from options_bt.chain_parser import OptionsChainParser


@pytest.fixture
def parser():
    csv_path = "sample_data/options/NIFTY_20250102.csv"
    if not os.path.exists(csv_path):
        pytest.skip(f"Test options CSV missing: {csv_path}")
    return OptionsChainParser(csv_path, enable_cache=True)


def test_strike_discovery(parser):
    """Verify strike discovery across wide headers."""
    ce_strikes, pe_strikes = parser.extract_available_strikes()
    assert len(ce_strikes) > 20, f"Expected >20 Call strikes, found {len(ce_strikes)}"
    assert len(pe_strikes) > 20, f"Expected >20 Put strikes, found {len(pe_strikes)}"
    assert 23750 in ce_strikes, "ATM strike 23750 must be present in CE strikes"
    assert 23750 in pe_strikes, "ATM strike 23750 must be present in PE strikes"
    assert ce_strikes == sorted(ce_strikes), "Discovered strikes must be sorted ascending"


def test_single_contract_series(parser):
    """Verify 1-minute series extraction for single contract."""
    df_ce = parser.get_contract_series(23750, "CE")
    assert not df_ce.empty
    assert len(df_ce) == 375, f"Expected 375 intraday 1-min bars, got {len(df_ce)}"
    assert "DateTime" in df_ce.columns
    assert "price" in df_ce.columns
    assert "oi" in df_ce.columns
    assert df_ce["price"].iloc[0] > 0.0, "Option price must be positive"


def test_batch_multiple_contracts(parser):
    """Verify single-pass multi-contract batch extraction."""
    contracts = [(23700, "CE"), (23700, "PE"), (23800, "CE"), (23800, "PE")]
    df_batch = parser.get_multiple_contracts(contracts)

    assert not df_batch.empty
    assert len(df_batch) == 375
    assert "23700CE_price" in df_batch.columns
    assert "23700PE_price" in df_batch.columns
    assert "23800CE_price" in df_batch.columns
    assert "23800PE_price" in df_batch.columns
    assert "DateTime" in df_batch.columns


def test_straddle_series_builder(parser):
    """Verify ATM straddle series alignment and math."""
    atm = 23750
    df_straddle = parser.get_straddle_series(atm)

    assert not df_straddle.empty
    assert len(df_straddle) == 375
    assert "straddle_price" in df_straddle.columns

    # Verify straddle price is exact sum of CE and PE
    row0 = df_straddle.iloc[0]
    expected_straddle = row0["ce_price"] + row0["pe_price"]
    assert abs(row0["straddle_price"] - expected_straddle) < 1e-4

    # Verify total OI is exact sum
    expected_oi = row0["ce_oi"] + row0["pe_oi"]
    assert abs(row0["total_oi"] - expected_oi) < 1e-4


def test_full_chain_snapshot(parser):
    """Verify single-minute snapshot across all strikes."""
    snap = parser.get_chain_snapshot("09:25:00")
    assert not snap.empty
    assert "strike" in snap.columns
    assert "ce_close" in snap.columns
    assert "pe_close" in snap.columns
    assert "total_oi" in snap.columns

    # Verify ATM strike exists in snapshot
    atm_row = snap[snap["strike"] == 23750]
    assert len(atm_row) == 1
    assert float(atm_row.iloc[0]["ce_close"]) > 0.0
    assert float(atm_row.iloc[0]["pe_close"]) > 0.0


def test_resample_bars(parser):
    """Verify resampling 1-minute bars to 5-minute candles."""
    df_ce = parser.get_contract_series(23750, "CE")
    df_5m = parser.resample_bars(df_ce, timeframe="5min")

    assert not df_5m.empty
    assert len(df_5m) == 75, f"Expected 75 5-min candles for 375-min session, got {len(df_5m)}"
    assert "price" in df_5m.columns
