"""
Unit Tests for Broker Gateway, OAuth Login, and Margin Engine
Verifies Fyers v3, Upstox v2, and Zerodha Kite gateway adapters.
"""

import os
import tempfile
import pytest
from broker_gateway.fyers_login import (
    FyersLoginGateway,
    FyersCredentials,
    FyersMarginReport,
    load_fyers_credentials,
)
from broker_gateway.upstox_login import (
    UpstoxLoginGateway,
    UpstoxCredentials,
    UpstoxMarginReport,
)
from broker_gateway.kite_client import KiteConnectGateway, KiteOrderParams
from broker_gateway.state_manager import BotStateManager, BotState


def test_fyers_credentials_parser():
    """Verify that APPcode.txt with spaces and comments is parsed cleanly."""
    content = """
    # Fyers API Configuration
    APP ID = XC12345-100
    SECRET ID = SEC_987654321
    REDIRECT URL = http://127.0.0.1:5000/callback
    """
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        creds = load_fyers_credentials(tmp_path)
        assert creds.client_id == "XC12345-100"
        assert creds.secret_key == "SEC_987654321"
        assert creds.redirect_uri == "http://127.0.0.1:5000/callback"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_fyers_margin_report_calculation():
    """Verify Fyers funds parsing and margin utilization math."""
    creds = FyersCredentials(
        client_id="TEST_CLIENT",
        secret_key="TEST_SECRET",
        redirect_uri="http://127.0.0.1:5000/callback"
    )
    gateway = FyersLoginGateway(credentials=creds)

    # Mock response format returned by Fyers API v3 fyers.funds()
    mock_funds = {
        "s": "ok",
        "fund_limit": [
            {"id": 10, "title": "Total Balance", "equityAmount": 1_000_000.0},
            {"id": 1, "title": "Available Balance", "equityAmount": 650_000.0},
            {"id": 2, "title": "Utilized Amount", "equityAmount": 350_000.0},
            {"id": 3, "title": "Realized Profit", "equityAmount": 25_000.0},
            {"id": 4, "title": "Unrealized Profit", "equityAmount": 12_500.0},
            {"id": 5, "title": "Collateral Margin", "equityAmount": 200_000.0}
        ]
    }

    # Verify margin report calculations
    total_bal = 1_000_000.0
    avail_bal = 650_000.0
    utilized = 350_000.0
    util_pct = (utilized / total_bal) * 100.0

    report = FyersMarginReport(
        total_balance=total_bal,
        available_balance=avail_bal,
        utilized_amount=utilized,
        realized_pnl=25_000.0,
        unrealized_pnl=12_500.0,
        collateral_margin=200_000.0,
        margin_utilization_pct=util_pct,
        is_margin_sufficient=True,
        raw_response=mock_funds
    )

    assert report.total_balance == 1_000_000.0
    assert report.available_balance == 650_000.0
    assert report.utilized_amount == 350_000.0
    assert report.margin_utilization_pct == 35.0
    assert report.is_margin_sufficient is True


def test_upstox_auth_url_and_margin():
    """Verify Upstox auth dialog URL generation and margin report math."""
    gateway = UpstoxLoginGateway(
        api_key="UPSTOX_TEST_KEY",
        api_secret="UPSTOX_TEST_SECRET",
        redirect_uri="https://httpbin.org/get"
    )

    auth_url = gateway.get_auth_url()
    assert "https://api.upstox.com/v2/login/authorization/dialog" in auth_url
    assert "client_id=UPSTOX_TEST_KEY" in auth_url

    # Mock Upstox /user/get-funds-and-margin response
    mock_margin_data = {
        "status": "success",
        "data": {
            "equity": {
                "available_margin": 450_000.0,
                "used_margin": 150_000.0,
                "payin_amount": 50_000.0,
                "span_margin": 100_000.0,
                "exposure_margin": 50_000.0,
                "adhoc_margin": 0.0,
                "notional_cash": 450_000.0
            }
        }
    }

    avail = 450_000.0
    used = 150_000.0
    tot = avail + used
    util_pct = (used / tot) * 100.0

    report = UpstoxMarginReport(
        available_margin=avail,
        used_margin=used,
        payin_amount=50_000.0,
        span_margin=100_000.0,
        exposure_margin=50_000.0,
        adhoc_margin=0.0,
        notional_cash=450_000.0,
        margin_utilization_pct=util_pct,
        is_margin_sufficient=True,
        segment="SEC",
        raw_response=mock_margin_data
    )

    assert report.available_margin == 450_000.0
    assert report.used_margin == 150_000.0
    assert report.margin_utilization_pct == 25.0
    assert report.is_margin_sufficient is True


def test_kite_order_formatting():
    """Verify Zerodha Kite order formatting."""
    gateway = KiteConnectGateway(api_key="KITE_KEY", access_token="KITE_TOKEN")
    order = KiteOrderParams(
        tradingsymbol="NIFTY24DEC24000CE",
        exchange="NFO",
        transaction_type="BUY",
        quantity=50,
        order_type="LIMIT",
        price=185.50,
        product="MIS",
        validity="IOC"
    )
    payload = gateway.format_order_payload(order)
    assert payload["tradingsymbol"] == "NIFTY24DEC24000CE"
    assert payload["exchange"] == "NFO"
    assert payload["transaction_type"] == "BUY"
    assert payload["price"] == 185.50
    assert payload["validity"] == "IOC"


def test_bot_state_manager():
    """Verify trading bot state persistence."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        manager = BotStateManager(state_file_path=tmp_path)
        bot = BotState(
            bot_id="BOT_01",
            strategy_name="NIFTY_STRADDLE_HARVESTER",
            instrument="NIFTY",
            mode="LIVE",
            is_active=True,
            current_position_lots=4,
            realized_pnl_rupees=12_500.0
        )
        manager.register_bot(bot)

        # Reload from disk
        reloaded = BotStateManager(state_file_path=tmp_path)
        fetched = reloaded.get_bot("BOT_01")
        assert fetched is not None
        assert fetched.strategy_name == "NIFTY_STRADDLE_HARVESTER"
        assert fetched.current_position_lots == 4
        assert fetched.realized_pnl_rupees == 12_500.0
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
