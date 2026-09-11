"""
Point-in-Time (PIT) Nifty 500 Universe Engine
Eliminates survivorship bias by tracking exact historical index constituents on any trade date.
"""

from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Dict, List, Set, Optional
import pandas as pd


class PointInTimeUniverse:
    """
    Maintains historical inclusion and exclusion events for the Nifty 500.
    Ensures backtests only select stocks that were active constituents on that specific date.
    """

    def __init__(self, pit_json_path: str):
        self.pit_json_path = pit_json_path
        self.events: List[Dict] = []
        self._load_pit_data()

    def _load_pit_data(self):
        if not os.path.exists(self.pit_json_path):
            raise FileNotFoundError(f"PIT universe JSON not found at: {self.pit_json_path}")

        with open(self.pit_json_path, "r") as f:
            data = json.load(f)

        # Parse events and normalize dates
        if isinstance(data, list):
            self.events = data
        elif isinstance(data, dict):
            self.events = data.get("changes", []) or data.get("events", [])

    def get_universe_on_date(self, target_date: str, base_constituents: Optional[Set[str]] = None) -> Set[str]:
        """
        Reconstructs the exact Nifty 500 constituent set as of target_date (YYYY-MM-DD).
        """
        target_dt = pd.to_datetime(target_date).date()
        active_set = set(base_constituents) if base_constituents else set()

        # Sort events chronologically up to target_date
        applicable_events = []
        for ev in self.events:
            ev_date_str = ev.get("Parsed Date") or ev.get("event_date") or ev.get("date")
            if not ev_date_str:
                continue
            ev_dt = pd.to_datetime(ev_date_str).date()
            if ev_dt <= target_dt:
                applicable_events.append((ev_dt, ev))

        applicable_events.sort(key=lambda x: x[0])

        for _, ev in applicable_events:
            action = str(ev.get("Action", "")).strip().upper()
            symbol = str(ev.get("Symbol") or ev.get("symbol", "")).strip().upper()

            if not symbol:
                continue

            if "INCL" in action or "ADD" in action:
                active_set.add(symbol)
            elif "EXCL" in action or "DEL" in action or "REMOVE" in action:
                active_set.discard(symbol)

        return active_set

    @classmethod
    def audit_survivorship_bias(cls, static_universe: Set[str], pit_universe: Set[str]) -> Dict[str, any]:
        """
        Quantifies the discrepancy between naive static universe and true PIT universe.
        """
        survivor_only = static_universe - pit_universe
        delisted_or_excluded = pit_universe - static_universe
        overlap = static_universe.intersection(pit_universe)

        overlap_pct = (len(overlap) / len(pit_universe) * 100.0) if pit_universe else 100.0

        return {
            "total_pit_count": len(pit_universe),
            "total_static_count": len(static_universe),
            "overlap_count": len(overlap),
            "overlap_pct": round(overlap_pct, 2),
            "survivor_bias_stocks_count": len(survivor_only),
            "survivor_bias_stocks": sorted(list(survivor_only))[:15],
            "historical_excluded_count": len(delisted_or_excluded),
            "historical_excluded_sample": sorted(list(delisted_or_excluded))[:15]
        }
