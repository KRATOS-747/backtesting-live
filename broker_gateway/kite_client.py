"""
Zerodha KiteConnect Broker Gateway Adapter
Handles session authentication, option chain subscription, and live order placement.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class KiteOrderParams:
    tradingsymbol: str
    exchange: str       # 'NFO' or 'BFO'
    transaction_type: str  # 'BUY' or 'SELL'
    quantity: int
    order_type: str     # 'MARKET', 'LIMIT', 'SL', 'SL-M'
    price: float = 0.0
    product: str = "MIS" # MIS for intraday margin, NRML for overnight
    validity: str = "IOC"


class KiteConnectGateway:
    """
    Standardized adapter for Zerodha KiteConnect v3 API.
    """

    def __init__(self, api_key: str = "", access_token: str = ""):
        self.api_key = api_key
        self.access_token = access_token
        self.is_authenticated = bool(access_token)

    def format_order_payload(self, params: KiteOrderParams) -> Dict[str, any]:
        """Formats order into standardized Kite API JSON request."""
        return {
            "tradingsymbol": params.tradingsymbol,
            "exchange": params.exchange,
            "transaction_type": params.transaction_type.upper(),
            "quantity": params.quantity,
            "order_type": params.order_type.upper(),
            "price": params.price if params.order_type.upper() == "LIMIT" else None,
            "product": params.product.upper(),
            "validity": params.validity.upper()
        }
