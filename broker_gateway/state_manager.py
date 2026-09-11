"""
Trading Bot State & Position Persistence Manager
Maintains live execution states, active bots, and audit logs.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


@dataclass
class BotState:
    bot_id: str
    strategy_name: str
    instrument: str
    mode: str           # 'PAPER' or 'LIVE'
    is_active: bool
    current_position_lots: int = 0
    realized_pnl_rupees: float = 0.0
    unrealized_pnl_rupees: float = 0.0
    last_update_time: str = ""


class BotStateManager:
    """
    Manages operational states of active desk strategies and bots.
    Persists configuration and position tracking to JSON disk storage.
    """

    def __init__(self, state_file_path: str = "active_bots.json"):
        self.state_file_path = state_file_path
        self.bots: Dict[str, BotState] = {}
        self.load_state()

    def load_state(self):
        if not os.path.exists(self.state_file_path):
            return
        try:
            with open(self.state_file_path, "r") as f:
                data = json.load(f)
                for item in data:
                    bot = BotState(**item)
                    self.bots[bot.bot_id] = bot
        except Exception:
            pass

    def save_state(self):
        try:
            with open(self.state_file_path, "w") as f:
                json.dump([asdict(b) for b in self.bots.values()], f, indent=2)
        except Exception:
            pass

    def register_bot(self, bot: BotState):
        self.bots[bot.bot_id] = bot
        self.save_state()

    def get_bot(self, bot_id: str) -> Optional[BotState]:
        return self.bots.get(bot_id)
