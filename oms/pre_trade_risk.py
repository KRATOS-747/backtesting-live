"""
Pre-Trade Greeks and Margin Risk Gatekeeper
Validates that incoming orders do not breach desk portfolio Greek limits or peak margin thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class RiskCheckResult:
    is_approved: bool
    current_net_delta: float
    projected_net_delta: float
    current_net_gamma: float
    projected_net_gamma: float
    margin_utilization_pct: float
    rejection_reason: Optional[str] = None


class PreTradeRiskManager:
    """
    Enforces institutional intraday risk boundaries before any order hits the exchange:
    1. Net Portfolio Delta Clamping
    2. Net Portfolio Gamma Ceiling (especially critical on 0DTE expiry afternoon)
    3. Peak Margin Utilization Threshold (SPAN + Exposure)
    """

    def __init__(
        self,
        max_net_delta: float = 150.0,         # Maximum net unhedged delta units
        max_net_gamma: float = 25.0,          # Maximum net gamma ceiling
        max_margin_utilization_pct: float = 85.0  # Max 85% margin utilization
    ):
        self.max_net_delta = max_net_delta
        self.max_net_gamma = max_net_gamma
        self.max_margin_utilization_pct = max_margin_utilization_pct

    def evaluate_order(
        self,
        current_delta: float,
        current_gamma: float,
        order_delta: float,
        order_gamma: float,
        current_margin_used: float,
        order_margin_required: float,
        total_account_capital: float
    ) -> RiskCheckResult:
        """
        Evaluates projected portfolio Greeks and margin post-fill.
        """
        proj_delta = current_delta + order_delta
        proj_gamma = current_gamma + order_gamma

        total_margin_projected = current_margin_used + order_margin_required
        margin_pct = (total_margin_projected / max(total_account_capital, 1.0)) * 100.0

        # Check 1: Margin Utilization
        if margin_pct > self.max_margin_utilization_pct:
            return RiskCheckResult(
                is_approved=False,
                current_net_delta=current_delta,
                projected_net_delta=proj_delta,
                current_net_gamma=current_gamma,
                projected_net_gamma=proj_gamma,
                margin_utilization_pct=round(margin_pct, 2),
                rejection_reason=f"Projected margin utilization ({margin_pct:.1f}%) breaches safety ceiling ({self.max_margin_utilization_pct:.1f}%)"
            )

        # Check 2: Delta Clamp
        if abs(proj_delta) > self.max_net_delta:
            return RiskCheckResult(
                is_approved=False,
                current_net_delta=current_delta,
                projected_net_delta=proj_delta,
                current_net_gamma=current_gamma,
                projected_net_gamma=proj_gamma,
                margin_utilization_pct=round(margin_pct, 2),
                rejection_reason=f"Projected net delta ({proj_delta:.1f}) breaches desk delta clamp (±{self.max_net_delta:.1f})"
            )

        # Check 3: Gamma Ceiling
        if abs(proj_gamma) > self.max_net_gamma:
            return RiskCheckResult(
                is_approved=False,
                current_net_delta=current_delta,
                projected_net_delta=proj_delta,
                current_net_gamma=current_gamma,
                projected_net_gamma=proj_gamma,
                margin_utilization_pct=round(margin_pct, 2),
                rejection_reason=f"Projected net gamma ({proj_gamma:.2f}) breaches desk gamma ceiling (±{self.max_net_gamma:.2f})"
            )

        return RiskCheckResult(
            is_approved=True,
            current_net_delta=current_delta,
            projected_net_delta=proj_delta,
            current_net_gamma=current_gamma,
            projected_net_gamma=proj_gamma,
            margin_utilization_pct=round(margin_pct, 2),
            rejection_reason=None
        )
