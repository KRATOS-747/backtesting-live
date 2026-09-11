"""
Upstox API v2 Broker Authentication & Margin Fetching Gateway
Adapted from desk implementation: login_up_coursedes.py

Capabilities:
1. OAuth 2.0 Flow: Generates authorization dialog URL and exchanges auth code for access token.
2. Token Persistence: Saves token to disk (access_token.txt).
3. Profile & Account Info: Fetches user profile from https://api.upstox.com/v2/user/profile.
4. Live Margin & Funds Engine: Fetches equity & commodity margin limits via /user/get-funds-and-margin.
5. Pre-Trade Risk Gate: Evaluates available funds against required order margin before routing to OMS.
"""

from __future__ import annotations

import os
import sys
import json
import urllib.request
import urllib.parse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# =====================================================================
# DATA MODELS
# =====================================================================
@dataclass
class UpstoxCredentials:
    api_key: str
    api_secret: str
    redirect_uri: str


@dataclass
class UpstoxMarginReport:
    available_margin: float
    used_margin: float
    payin_amount: float
    span_margin: float
    exposure_margin: float
    adhoc_margin: float
    notional_cash: float
    margin_utilization_pct: float
    is_margin_sufficient: bool
    segment: str
    raw_response: Dict


# =====================================================================
# HTTP REQUEST HELPER (Zero External Dependency Fallback)
# =====================================================================
def _http_request(url: str, method: str = "GET", data: Optional[Dict] = None, headers: Optional[Dict] = None) -> Dict:
    """Executes HTTP request via requests if available, else urllib.request."""
    headers = headers or {}
    headers.setdefault("Accept", "application/json")

    if REQUESTS_AVAILABLE:
        if method.upper() == "POST":
            resp = requests.post(url, data=data, headers=headers)
        else:
            resp = requests.get(url, headers=headers)
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code} Error: {resp.text}")
        return resp.json()
    else:
        # Standard library urllib fallback
        req_data = urllib.parse.urlencode(data).encode("utf-8") if data else None
        req = urllib.request.Request(url, data=req_data, headers=headers, method=method.upper())
        try:
            with urllib.request.urlopen(req) as response:
                body = response.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            raise RuntimeError(f"HTTP {e.code} Error: {err_body}")


# =====================================================================
# UPSTOX GATEWAY & MARGIN FETCHER
# =====================================================================
class UpstoxLoginGateway:
    """
    Manages Upstox v2 session authentication, profile retrieval, and real-time margin queries.
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        redirect_uri: str = "https://httpbin.org/get",
        token_path: str = "access_token.txt"
    ):
        self.api_key = api_key or os.getenv("UPSTOX_API_KEY", "679a51e4-ff5c-445d-9001-a22238419906")
        self.api_secret = api_secret or os.getenv("UPSTOX_API_SECRET", "q81m29knx1")
        self.redirect_uri = redirect_uri or os.getenv("UPSTOX_REDIRECT_URI", "https://httpbin.org/get")
        self.token_path = token_path
        self.access_token: Optional[str] = self.load_saved_token()

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
        print(f"✅ Access token saved successfully to {self.token_path}")

    def get_auth_url(self) -> str:
        """Generates Upstox OAuth authorization dialog URL."""
        base_url = "https://api.upstox.com/v2/login/authorization/dialog"
        params = {
            "response_type": "code",
            "client_id": self.api_key,
            "redirect_uri": self.redirect_uri,
            "state": "nifty_quant_workbench",
        }
        return f"{base_url}?{urllib.parse.urlencode(params)}"

    def fetch_access_token(self, auth_code: str) -> str:
        """Exchanges authorization code for an active API access token."""
        token_url = "https://api.upstox.com/v2/login/authorization/token"
        payload = {
            "grant_type": "authorization_code",
            "code": auth_code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.api_key,
            "client_secret": self.api_secret,
        }
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }

        resp = _http_request(token_url, method="POST", data=payload, headers=headers)
        access_token = resp.get("access_token")
        if not access_token:
            raise RuntimeError(f"Failed to extract access_token from response: {resp}")

        self.save_token(access_token)
        return access_token

    def fetch_profile(self) -> Dict:
        """Fetches active user profile from Upstox."""
        if not self.access_token:
            raise RuntimeError("Access token not set. Please authenticate first.")

        url = "https://api.upstox.com/v2/user/profile"
        headers = {
            "Authorization": f"Bearer {self.access_token}"
        }
        return _http_request(url, method="GET", headers=headers)

    def fetch_margins(self, segment: str = "SEC") -> UpstoxMarginReport:
        """
        Fetches live margin and funds from Upstox API:
        Endpoint: https://api.upstox.com/v2/user/get-funds-and-margin
        segment: 'SEC' for Equity/Derivatives or 'COM' for Commodity.
        """
        if not self.access_token:
            raise RuntimeError("Access token not set. Please authenticate first.")

        url = f"https://api.upstox.com/v2/user/get-funds-and-margin?segment={segment}"
        headers = {
            "Authorization": f"Bearer {self.access_token}"
        }

        resp = _http_request(url, method="GET", headers=headers)
        data = resp.get("data", {})
        equity_data = data.get("equity", {})

        avail_margin = float(equity_data.get("available_margin", 0.0))
        used_margin = float(equity_data.get("used_margin", 0.0))
        payin = float(equity_data.get("payin_amount", 0.0))
        span = float(equity_data.get("span_margin", 0.0))
        exposure = float(equity_data.get("exposure_margin", 0.0))
        adhoc = float(equity_data.get("adhoc_margin", 0.0))
        notional_cash = float(equity_data.get("notional_cash", 0.0))

        total_margin = avail_margin + used_margin
        util_pct = (used_margin / total_margin * 100.0) if total_margin > 0 else 0.0

        return UpstoxMarginReport(
            available_margin=round(avail_margin, 2),
            used_margin=round(used_margin, 2),
            payin_amount=round(payin, 2),
            span_margin=round(span, 2),
            exposure_margin=round(exposure, 2),
            adhoc_margin=round(adhoc, 2),
            notional_cash=round(notional_cash, 2),
            margin_utilization_pct=round(util_pct, 2),
            is_margin_sufficient=(avail_margin > 0),
            segment=segment,
            raw_response=resp
        )

    def check_margin_availability(self, required_margin: float) -> Tuple[bool, str]:
        """Pre-trade risk check: ensures desk has sufficient margin before placing order."""
        try:
            report = self.fetch_margins()
            if report.available_margin >= required_margin:
                return True, f"Margin OK (Available: ₹{report.available_margin:,.2f} >= Req: ₹{required_margin:,.2f})"
            else:
                return False, f"INSUFFICIENT_MARGIN (Available: ₹{report.available_margin:,.2f} < Req: ₹{required_margin:,.2f})"
        except Exception as e:
            return False, f"Failed to check margin: {e}"


# =====================================================================
# CLI RUNNER FOR UPSTOX LOGIN
# =====================================================================
def run_upstox_login_cli():
    """CLI interactive authentication flow."""
    gateway = UpstoxLoginGateway()
    auth_url = gateway.get_auth_url()

    print("\n" + "=" * 70)
    print("🚀 UPSTOX API v2 INTERACTIVE AUTHENTICATION")
    print("=" * 70)
    print("Visit this URL in your browser to authorize:")
    print(f"\n   {auth_url}\n")
    print("After login, you will be redirected. Copy the 'code' parameter from the URL bar.")

    auth_code = input("Enter the authorization code: ").strip()
    if auth_code:
        try:
            token = gateway.fetch_access_token(auth_code)
            print(f"🎉 Access Token: {token[:20]}...")
            profile = gateway.fetch_profile()
            print(f"👤 Profile: {profile.get('data', {}).get('user_name', 'Upstox User')}")
            margin = gateway.fetch_margins()
            print(f"💰 Available Margin: ₹{margin.available_margin:,.2f}")
        except Exception as e:
            print(f"❌ Error during authentication: {e}")


if __name__ == "__main__":
    run_upstox_login_cli()
