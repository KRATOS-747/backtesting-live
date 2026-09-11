"""
Upstox v2 Broker Gateway & Binary Protocol Buffers Streaming Market Feeder
Parses ultra-low-latency binary streaming market quotes for NSE/BSE options.
"""

from __future__ import annotations

import os
import json
import asyncio
from typing import Callable, Dict, List, Optional
try:
    import websockets
except ImportError:
    websockets = None

try:
    from . import MarketDataFeed_pb2 as pb
except Exception:
    try:
        import MarketDataFeed_pb2 as pb
    except Exception:
        pb = None


class UpstoxGateway:
    """
    Interfaces with Upstox v2 API and decodes streaming binary Google Protocol Buffers market data.
    """

    def __init__(self, api_key: str = "", access_token: str = ""):
        self.api_key = api_key
        self.access_token = access_token
        self.ws_url = "wss://api.upstox.com/v3/market-quote/socket"
        self.is_connected = False
        self.subscribed_keys: List[str] = []

    def parse_protobuf_feed(self, binary_payload: bytes) -> Dict[str, Dict]:
        """
        Decodes binary Protocol Buffer stream into structured quote dictionaries.
        """
        feed_response = pb.FeedResponse()
        feed_response.ParseFromString(binary_payload)

        parsed_quotes = {}
        for instrument_key, feed_data in feed_response.feeds.items():
            ltp = 0.0
            volume = 0
            oi = 0.0

            # Inspect feed contents
            if feed_data.HasField("ltpc"):
                ltp = feed_data.ltpc.ltp
            elif feed_data.HasField("fullFeed"):
                full = feed_data.fullFeed
                if full.HasField("marketFF"):
                    ltp = full.marketFF.ltpc.ltp
                    volume = full.marketFF.vtt
                    oi = full.marketFF.oi

            parsed_quotes[instrument_key] = {
                "instrument_key": instrument_key,
                "ltp": ltp,
                "volume": volume,
                "oi": oi
            }

        return parsed_quotes
