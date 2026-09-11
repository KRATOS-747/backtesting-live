"""
Multi-Alpha Consensus & Quorum Engine
Aggregates votes across independent micro-alphas to make high-conviction execution decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import pandas as pd


@dataclass
class ConsensusDecision:
    action: str            # 'LONG', 'SHORT', 'HOLD'
    is_quorum_met: bool
    votes_long: int
    votes_short: int
    total_active_signals: int
    required_quorum: int
    vote_summary: str
    rationale: str


class AlphaConsensusEngine:
    """
    Evaluates multi-signal consensus across diverse micro-alphas:
    Requires at least k out of N micro-alphas to agree before triggering capital deployment.
    """

    def __init__(self, required_quorum: int = 2, direction_mode: str = "BI_DIRECTIONAL"):
        self.required_quorum = required_quorum
        self.direction_mode = direction_mode.upper()

    def evaluate_votes(self, signal_votes: Dict[str, str]) -> ConsensusDecision:
        """
        Evaluates votes dictionary: {signal_name: 'LONG' | 'SHORT' | 'HOLD'}
        """
        votes = [v.upper() for v in signal_votes.values() if v.upper() in ("LONG", "SHORT")]
        total_active = len(signal_votes)

        votes_long = votes.count("LONG")
        votes_short = votes.count("SHORT")
        summary_str = ", ".join([f"{k}:{v}" for k, v in signal_votes.items()])

        target_k = min(self.required_quorum, total_active)

        # 1. Long Consensus
        if votes_long >= target_k:
            if self.direction_mode in ("LONG_ONLY", "BI_DIRECTIONAL"):
                return ConsensusDecision(
                    action="LONG",
                    is_quorum_met=True,
                    votes_long=votes_long,
                    votes_short=votes_short,
                    total_active_signals=total_active,
                    required_quorum=target_k,
                    vote_summary=summary_str,
                    rationale=f"Long Quorum Met ({votes_long}/{total_active} >= {target_k}): {summary_str}"
                )

        # 2. Short Consensus
        if votes_short >= target_k:
            if self.direction_mode in ("SHORT_ONLY", "BI_DIRECTIONAL"):
                return ConsensusDecision(
                    action="SHORT",
                    is_quorum_met=True,
                    votes_long=votes_long,
                    votes_short=votes_short,
                    total_active_signals=total_active,
                    required_quorum=target_k,
                    vote_summary=summary_str,
                    rationale=f"Short Quorum Met ({votes_short}/{total_active} >= {target_k}): {summary_str}"
                )

        return ConsensusDecision(
            action="HOLD",
            is_quorum_met=False,
            votes_long=votes_long,
            votes_short=votes_short,
            total_active_signals=total_active,
            required_quorum=target_k,
            vote_summary=summary_str,
            rationale=f"Quorum Not Met (L:{votes_long}, S:{votes_short}, Req:{target_k}): {summary_str}"
        )
