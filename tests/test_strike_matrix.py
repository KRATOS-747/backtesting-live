"""
Unit tests for StrikeMatrixEngine & Multi-Leg Structure Generator
"""

import pytest
import pandas as pd
from options_bt.strike_matrix import (
    StrikeMatrixEngine,
    StrikeProfile,
    StraddleStrikes,
    StrangleStrikes,
    IronCondorStrikes,
)


def test_index_factory_and_intervals():
    nifty = StrikeMatrixEngine.for_index("NIFTY")
    assert nifty.step_size == 50.0
    assert nifty.index_name == "NIFTY"

    bn = StrikeMatrixEngine.for_index("BANKNIFTY")
    assert bn.step_size == 100.0
    assert bn.index_name == "BANKNIFTY"

    fn = StrikeMatrixEngine.for_index("FINNIFTY")
    assert fn.step_size == 25.0

    sx = StrikeMatrixEngine.for_index("SENSEX")
    assert sx.step_size == 100.0


def test_atm_strike_rounding():
    engine = StrikeMatrixEngine.for_index("NIFTY")
    assert engine.get_atm_strike(23524.8) == 23500
    assert engine.get_atm_strike(23525.0) == 23550
    assert engine.get_atm_strike(23549.9) == 23550

    bn_engine = StrikeMatrixEngine.for_index("BANKNIFTY")
    assert bn_engine.get_atm_strike(50149.0) == 50100
    assert bn_engine.get_atm_strike(50151.0) == 50200


def test_straddle_strikes():
    engine = StrikeMatrixEngine.for_index("NIFTY")
    straddle = engine.build_straddle_strikes(23510.0)
    assert isinstance(straddle, StraddleStrikes)
    assert straddle.atm_strike == 23500
    assert straddle.ce_strike == 23500
    assert straddle.pe_strike == 23500


def test_strangle_strikes():
    engine = StrikeMatrixEngine.for_index("NIFTY")
    # Default 3 steps = 150 points
    strangle = engine.build_strangle_strikes(23500.0)
    assert isinstance(strangle, StrangleStrikes)
    assert strangle.atm_strike == 23500
    assert strangle.call_strike == 23650
    assert strangle.put_strike == 23350
    assert strangle.width_points == 150

    # Custom otm_points = 200
    strangle_custom = engine.build_strangle_strikes(23500.0, otm_points=200)
    assert strangle_custom.call_strike == 23700
    assert strangle_custom.put_strike == 23300
    assert strangle_custom.width_points == 200


def test_iron_condor_and_iron_fly_strikes():
    engine = StrikeMatrixEngine.for_index("NIFTY")
    # Iron Condor with short_offset=100 (2 steps), wing_width=150 (3 steps)
    ic = engine.build_iron_condor_strikes(23500.0, short_offset=100, wing_width=150)
    assert isinstance(ic, IronCondorStrikes)
    assert not ic.is_iron_fly
    assert ic.short_call == 23600
    assert ic.long_call == 23750
    assert ic.short_put == 23400
    assert ic.long_put == 23250
    assert ic.wing_width == 150

    # Iron Fly (Short ATM + Wings)
    fly = engine.build_iron_fly_strikes(23500.0, wing_width=200)
    assert fly.is_iron_fly
    assert fly.short_call == 23500
    assert fly.short_put == 23500
    assert fly.long_call == 23700
    assert fly.long_put == 23300
    assert fly.wing_width == 200


def test_5strike_matrix():
    engine = StrikeMatrixEngine.for_index("NIFTY")
    matrix = engine.build_5strike_matrix(23500.0)
    assert len(matrix) == 10  # 5 calls + 5 puts

    calls = [p for p in matrix if p.option_type == "CE"]
    puts = [p for p in matrix if p.option_type == "PE"]

    assert len(calls) == 5
    assert len(puts) == 5

    # Calls: ITM2(23400), ITM1(23450), ATM(23500), OTM1(23550), OTM2(23600)
    assert calls[0].moneyness == "ITM2" and calls[0].strike == 23400
    assert calls[2].moneyness == "ATM" and calls[2].strike == 23500
    assert calls[4].moneyness == "OTM2" and calls[4].strike == 23600

    # Puts: ITM2(23600), ITM1(23550), ATM(23500), OTM1(23450), OTM2(23400)
    assert puts[0].moneyness == "ITM2" and puts[0].strike == 23600
    assert puts[2].moneyness == "ATM" and puts[2].strike == 23500
    assert puts[4].moneyness == "OTM2" and puts[4].strike == 23400


def test_categorize_moneyness():
    engine = StrikeMatrixEngine.for_index("NIFTY")
    spot = 23500.0

    # CE
    assert engine.categorize_moneyness(spot, 23500, "CE") == "ATM"
    assert engine.categorize_moneyness(spot, 23550, "CE") == "OTM"
    assert engine.categorize_moneyness(spot, 23700, "CE") == "DEEP_OTM"
    assert engine.categorize_moneyness(spot, 23450, "CE") == "ITM"
    assert engine.categorize_moneyness(spot, 23300, "CE") == "DEEP_ITM"

    # PE
    assert engine.categorize_moneyness(spot, 23500, "PE") == "ATM"
    assert engine.categorize_moneyness(spot, 23450, "PE") == "OTM"
    assert engine.categorize_moneyness(spot, 23300, "PE") == "DEEP_OTM"
    assert engine.categorize_moneyness(spot, 23550, "PE") == "ITM"
    assert engine.categorize_moneyness(spot, 23700, "PE") == "DEEP_ITM"


def test_find_anchor_spot():
    spot_df = pd.DataFrame({
        "date": [
            "2025-01-01 09:15:00",
            "2025-01-01 09:20:00",
            "2025-01-01 09:25:00",
            "2025-01-01 09:30:00",
        ],
        "close": [23600.0, 23610.0, 23625.0, 23630.0],
    })

    anchor = StrikeMatrixEngine.find_anchor_spot(spot_df, target_time="09:25:00")
    assert anchor == 23625.0
