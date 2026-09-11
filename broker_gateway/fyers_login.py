"""
Fyers API v3 Broker Authentication & Margin Fetching Gateway
Adapted from desk implementation: loginfyersakshay.py

Capabilities:
1. Flexible Credentials Parser: Parses APPcode.txt or environment variables.
2. OAuth 2.0 Flow: Local Flask redirect listener & manual auth code fallback.
3. Access Token Persistence: Saves token to access_token.txt.
4. Profile & Connectivity Verification: Fetches user profile via Fyers API.
5. Real-Time Margin & Funds Engine: Fetches available funds, utilized margin, collateral, and checks pre-trade risk.
"""

from __future__ import annotations

import os
import sys
import json
import time
import threading
import webbrowser
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    from fyers_apiv3 import fyersModel
    FYERS_SDK_AVAILABLE = True
except ImportError:
    FYERS_SDK_AVAILABLE = False

try:
    from flask import Flask, request
    from werkzeug.serving import make_server
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False


# =====================================================================
# DATA MODELS
# =====================================================================
@dataclass
class FyersCredentials:
    client_id: str
    secret_key: str
    redirect_uri: str


@dataclass
class FyersMarginReport:
    total_balance: float
    available_balance: float
    utilized_amount: float
    realized_pnl: float
    unrealized_pnl: float
    collateral_margin: float
    margin_utilization_pct: float
    is_margin_sufficient: bool
    raw_response: Dict


# =====================================================================
# CREDENTIALS PARSER
# =====================================================================
def load_fyers_credentials(config_file: str = "APPcode.txt") -> FyersCredentials:
    """
    Parses key-value pairs from APPcode.txt cleanly even if keys contain spaces
    (e.g., 'APP ID = ...' becomes 'APP_ID').
    Also checks environment variables as fallback.
    """
    client_id = os.getenv("FYERS_CLIENT_ID") or os.getenv("FYERS_APP_ID", "")
    secret_key = os.getenv("FYERS_SECRET_KEY") or os.getenv("FYERS_SECRET_ID", "")
    redirect_uri = os.getenv("FYERS_REDIRECT_URI") or os.getenv("FYERS_REDIRECT_URL", "")

    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    parts = line.split("=", 1)
                    key = parts[0].strip().upper().replace(" ", "_")
                    value = parts[1].strip().strip('"').strip("'")
                    if key in ("APP_ID", "CLIENT_ID") and not client_id:
                        client_id = value
                    elif key in ("SECRET_ID", "SECRET_KEY") and not secret_key:
                        secret_key = value
                    elif key in ("REDIRECT_URL", "REDIRECT_URI") and not redirect_uri:
                        redirect_uri = value

    if not redirect_uri:
        redirect_uri = "http://127.0.0.1:5000/callback"

    return FyersCredentials(
        client_id=client_id,
        secret_key=secret_key,
        redirect_uri=redirect_uri
    )


# =====================================================================
# FYERS GATEWAY & MARGIN FETCHER
# =====================================================================
class FyersLoginGateway:
    """
    Manages Fyers v3 session authentication, token refresh, and real-time margin queries.
    """

    def __init__(
        self,
        config_path: str = "APPcode.txt",
        token_path: str = "fyers_token.txt",
        credentials: Optional[FyersCredentials] = None
    ):
        self.config_path = config_path
        self.token_path = token_path
        self.creds = credentials or load_fyers_credentials(config_path)
        self.access_token: Optional[str] = self.load_saved_token()
        self.fyers_client = None

        if self.access_token and self.creds.client_id and FYERS_SDK_AVAILABLE:
            self._init_fyers_client(self.access_token)

    def load_saved_token(self) -> Optional[str]:
        """Loads access token from disk if present."""
        if os.path.exists(self.token_path):
            try:
                with open(self.token_path, "r") as f:
                    token = f.read().strip()
                    if token:
                        return token
            except Exception:
                pass
        return None

    def save_token(self, token: str):
        """Persists access token to disk."""
        self.access_token = token
        with open(self.token_path, "w") as f:
            f.write(token)

    def _init_fyers_client(self, token: str):
        if FYERS_SDK_AVAILABLE:
            self.fyers_client = fyersModel.FyersModel(
                token=token,
                client_id=self.creds.client_id,
                log_path=""
            )

    def get_auth_url(self) -> str:
        """Generates Fyers OAuth authorization URL."""
        if FYERS_SDK_AVAILABLE:
            session = fyersModel.SessionModel(
                client_id=self.creds.client_id,
                secret_key=self.creds.secret_key,
                redirect_uri=self.creds.redirect_uri,
                response_type="code",
                grant_type="authorization_code"
            )
            return session.generate_authcode()
        else:
            base_url = "https://api-t1.fyers.in/api/v3/generate-authcode"
            return f"{base_url}?client_id={self.creds.client_id}&redirect_uri={self.creds.redirect_uri}&response_type=code&state=sample_state"

    def exchange_code_for_token(self, auth_code: str) -> str:
        """Exchanges authorization code for an active API access token."""
        if not FYERS_SDK_AVAILABLE:
            raise RuntimeError("fyers_apiv3 package is required to exchange auth_code for token.")

        session = fyersModel.SessionModel(
            client_id=self.creds.client_id,
            secret_key=self.creds.secret_key,
            redirect_uri=self.creds.redirect_uri,
            response_type="code",
            grant_type="authorization_code"
        )
        session.set_token(auth_code)
        response = session.generate_token()

        if response.get("s") == "ok" or "access_token" in response:
            token = response["access_token"]
            self.save_token(token)
            self._init_fyers_client(token)
            return token
        else:
            raise RuntimeError(f"Fyers token generation failed: {response}")

    def fetch_profile(self) -> Dict:
        """Fetches active user profile from Fyers."""
        if not self.fyers_client:
            raise RuntimeError("Fyers client not authenticated. Please log in first.")
        return self.fyers_client.get_profile()

    def fetch_margins(self) -> FyersMarginReport:
        """
        Fetches live account funds & margin limits from Fyers:
        - Total Balance (Capital + Collateral)
        - Available Cash Balance
        - Utilized Margin
        - Realized & Unrealized P&L
        - Margin Utilization Percentage
        """
        if not self.fyers_client:
            raise RuntimeError("Fyers client not authenticated. Please log in first.")

        response = self.fyers_client.funds()
        fund_limits = response.get("fund_limit", [])

        total_bal = 0.0
        avail_bal = 0.0
        utilized = 0.0
        realized_pnl = 0.0
        unrealized_pnl = 0.0
        collateral = 0.0

        for item in fund_limits:
            title = str(item.get("title", "")).strip().lower()
            equity_amt = float(item.get("equityAmount", 0.0))

            if "total balance" in title:
                total_bal = equity_amt
            elif "available balance" in title:
                avail_bal = equity_amt
            elif "utilized amount" in title:
                utilized = equity_amt
            elif "realized profit" in title:
                realized_pnl = equity_amt
            elif "unrealized profit" in title:
                unrealized_pnl = equity_amt
            elif "collateral" in title:
                collateral = equity_amt

        util_pct = (utilized / total_bal * 100.0) if total_bal > 0 else 0.0

        return FyersMarginReport(
            total_balance=round(total_bal, 2),
            available_balance=round(avail_bal, 2),
            utilized_amount=round(utilized, 2),
            realized_pnl=round(realized_pnl, 2),
            unrealized_pnl=round(unrealized_pnl, 2),
            collateral_margin=round(collateral, 2),
            margin_utilization_pct=round(util_pct, 2),
            is_margin_sufficient=(avail_bal > 0),
            raw_response=response
        )

    def check_margin_availability(self, required_margin: float) -> Tuple[bool, str]:
        """Pre-trade risk check: ensures desk has sufficient margin before placing order."""
        try:
            report = self.fetch_margins()
            if report.available_balance >= required_margin:
                return True, f"Margin OK (Available: ₹{report.available_balance:,.2f} >= Req: ₹{required_margin:,.2f})"
            else:
                return False, f"INSUFFICIENT_MARGIN (Available: ₹{report.available_balance:,.2f} < Req: ₹{required_margin:,.2f})"
        except Exception as e:
            return False, f"Failed to check margin: {e}"


# =====================================================================
# CLI RUNNER FOR FYERS LOGIN
# =====================================================================
def run_fyers_login_cli(config_file: str = "APPcode.txt"):
    """
    Executes interactive OAuth login flow:
    1. Opens browser with auth URL.
    2. Listens for callback or accepts manual code.
    3. Fetches account profile & margin summary.
    """
    gateway = FyersLoginGateway(config_path=config_file)
    auth_url = gateway.get_auth_url()

    print("\n" + "=" * 70)
    print("🚀 FYERS API v3 INTERACTIVE AUTHENTICATION")
    print("=" * 70)
    print(f"🔑 Client ID    : {gateway.creds.client_id or 'Not Configured'}")
    print(f"🌐 Redirect URI : {gateway.creds.redirect_uri}")
    print(f"🔗 Auth URL     :\n   {auth_url}\n")

    if FLASK_AVAILABLE:
        app = Flask(__name__)
        server_instance = None

        @app.route("/callback")
        def callback():
            nonlocal server_instance
            code = request.args.get("auth_code")
            if code:
                try:
                    token = gateway.exchange_code_for_token(code)
                    print(f"🎉 Login Successful! Token saved.")
                    margin = gateway.fetch_margins()
                    print(f"💰 Available Margin: ₹{margin.available_balance:,.2f}")
                    threading.Thread(target=lambda: (time.sleep(1), server_instance.shutdown())).start()
                    return "<h2>Login Successful! You can close this browser tab.</h2>"
                except Exception as ex:
                    return f"<h2>Auth Error: {ex}</h2>", 400
            return "<h2>No auth_code in callback.</h2>", 400

        try:
            webbrowser.open(auth_url, new=1)
            print("⏳ Listening for redirect on http://127.0.0.1:5000/callback ...")
            server_instance = make_server("127.0.0.1", 5000, app)
            server_instance.serve_forever()
        except Exception as e:
            print(f"⚠️ Flask listener exception: {e}")
    else:
        print("💡 Flask not installed. Paste the auth_code from your browser URL:")
        code = input("Enter auth_code: ").strip()
        if code:
            token = gateway.exchange_code_for_token(code)
            print(f"🎉 Success! Token generated: {token[:15]}...")


if __name__ == "__main__":
    run_fyers_login_cli()
