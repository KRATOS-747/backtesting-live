"""
Institutional Options Visualization Suite
Generates publication-quality dark-mode interactive Plotly dashboards and static figures:
1. 3D Volatility Surface & Smile Slices
2. Taylor Series Greek P&L Waterfall Attribution
3. Intraday Multi-Leg Premium & Trajectory Tracker
4. Expiry Payoff Diagrams (Straddles, Strangles, Iron Condors, Iron Flies)
5. Open Interest Distribution & Max Pain / Resistance Walls
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .surface import VolatilitySurfaceMesh
from .backtester import SessionResult


class OptionsVisualizer:
    """
    Renders institutional-grade visual analytics for quantitative derivative strategies.
    Supports both interactive Plotly HTML export and publication-quality PNGs.
    """

    # Institutional dark-palette styling
    COLOR_BG = "#0d1117"
    COLOR_PANEL = "#161b22"
    COLOR_GRID = "#21262d"
    COLOR_TEXT = "#c9d1d9"
    COLOR_CALL = "#238636"      # Green
    COLOR_PUT = "#da3633"       # Red
    COLOR_STRADDLE = "#a371f7"  # Purple / Indigo
    COLOR_ACCENT = "#58a6ff"    # Blue
    COLOR_GOLD = "#d29922"      # Gold / Warning

    @classmethod
    def plot_3d_volatility_surface(
        cls,
        surface_mesh: VolatilitySurfaceMesh,
        save_path: Optional[str] = None,
        title: str = "NIFTY 3D Implied Volatility Surface"
    ) -> str:
        """
        Renders interactive 3D Plotly surface of Implied Volatility across Moneyness and Expiry.
        """
        x_val = getattr(surface_mesh, "moneyness_grid", getattr(surface_mesh, "moneyness", None))
        y_val = getattr(surface_mesh, "expiry_grid", getattr(surface_mesh, "expiries_days", None))
        z_val = getattr(surface_mesh, "iv_grid", getattr(surface_mesh, "iv_matrix", None))
        z_pct = np.asarray(z_val, dtype=float) * 100.0 if z_val is not None else np.zeros((1, 1))

        fig = go.Figure(data=[
            go.Surface(
                x=x_val,
                y=y_val,
                z=z_pct,
                colorscale="Viridis",
                colorbar=dict(title="IV (%)", tickfont=dict(color=cls.COLOR_TEXT)),
                contours=dict(
                    z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
                )
            )
        ])

        fig.update_layout(
            title=dict(text=title, font=dict(color=cls.COLOR_TEXT, size=18)),
            template="plotly_dark",
            paper_bgcolor=cls.COLOR_BG,
            plot_bgcolor=cls.COLOR_PANEL,
            scene=dict(
                xaxis=dict(title="Log-Moneyness ln(K/F)", backgroundcolor=cls.COLOR_PANEL, gridcolor=cls.COLOR_GRID),
                yaxis=dict(title="Days to Expiry (DTE)", backgroundcolor=cls.COLOR_PANEL, gridcolor=cls.COLOR_GRID),
                zaxis=dict(title="Implied Vol (%)", backgroundcolor=cls.COLOR_PANEL, gridcolor=cls.COLOR_GRID),
            ),
            margin=dict(l=20, r=20, b=20, t=50),
            width=900,
            height=600
        )

        out_path = save_path or "volatility_surface_3d.html"
        fig.write_html(out_path)
        return out_path

    @classmethod
    def plot_greek_waterfall(
        cls,
        attributions: Dict[str, float],
        save_path: Optional[str] = None,
        title: str = "Taylor Series Greek P&L Waterfall Attribution"
    ) -> str:
        """
        Constructs Greek waterfall decomposition:
        Net PnL = Delta PnL + Gamma PnL + Theta PnL + Vega PnL + Residual
        """
        components = ["delta_pnl", "gamma_pnl", "theta_pnl", "vega_pnl", "residual_pnl"]
        labels = ["Delta (ΔS)", "Gamma (½ΓΔS²)", "Theta (ΘΔt)", "Vega (νΔσ)", "Residual"]
        values = [attributions.get(c, 0.0) for c in components]
        total_pnl = attributions.get("total_pnl", sum(values))

        fig = go.Figure(go.Waterfall(
            name="Greek Attribution",
            orientation="v",
            measure=["relative"] * len(values) + ["total"],
            x=labels + ["Net Model P&L"],
            y=values + [total_pnl],
            connector=dict(line=dict(color=cls.COLOR_GRID)),
            increasing=dict(marker=dict(color=cls.COLOR_CALL)),
            decreasing=dict(marker=dict(color=cls.COLOR_PUT)),
            totals=dict(marker=dict(color=cls.COLOR_STRADDLE)),
            textposition="outside",
            text=[f"₹{v:+,.0f}" for v in values] + [f"₹{total_pnl:+,.0f}"]
        ))

        fig.update_layout(
            title=dict(text=f"{title} (Total: ₹{total_pnl:,.2f})", font=dict(color=cls.COLOR_TEXT, size=18)),
            template="plotly_dark",
            paper_bgcolor=cls.COLOR_BG,
            plot_bgcolor=cls.COLOR_PANEL,
            yaxis=dict(title="P&L Contribution (₹)", gridcolor=cls.COLOR_GRID),
            xaxis=dict(gridcolor=cls.COLOR_GRID),
            margin=dict(l=40, r=40, b=40, t=60),
            width=900,
            height=500
        )

        out_path = save_path or "greek_waterfall.html"
        fig.write_html(out_path)
        return out_path

    @classmethod
    def plot_intraday_straddle_trajectory(
        cls,
        trade_bars_df: pd.DataFrame,
        session_result: SessionResult,
        save_path: Optional[str] = None
    ) -> str:
        """
        Plots 1-minute intraday Call, Put, and Combined Straddle premium trajectories
        alongside open interest evolution and stop-loss / target exit markers.
        """
        df = trade_bars_df.copy()
        if "time_str" not in df.columns and "DateTime" in df.columns:
            df["time_str"] = pd.to_datetime(df["DateTime"]).dt.strftime("%H:%M:%S")

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Intraday Straddle Premium Trajectory (₹)", "Combined Open Interest Evolution"),
            row_heights=[0.7, 0.3]
        )

        # 1. Premium Trajectories
        fig.add_trace(go.Scatter(
            x=df["time_str"], y=df["ce_price"], name=f"{session_result.atm_strike} CE",
            line=dict(color=cls.COLOR_CALL, width=1.8)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df["time_str"], y=df["pe_price"], name=f"{session_result.atm_strike} PE",
            line=dict(color=cls.COLOR_PUT, width=1.8)
        ), row=1, col=1)

        straddle_curve = df["ce_price"] + df["pe_price"]
        fig.add_trace(go.Scatter(
            x=df["time_str"], y=straddle_curve, name="Combined Straddle",
            line=dict(color=cls.COLOR_STRADDLE, width=2.5)
        ), row=1, col=1)

        # Baseline entry marker
        fig.add_hline(
            y=session_result.straddle_entry, line_dash="dash", line_color=cls.COLOR_GOLD,
            annotation_text=f"Entry: ₹{session_result.straddle_entry:.1f}", row=1, col=1
        )

        # 2. Open Interest
        if "ce_oi" in df.columns and "pe_oi" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["time_str"], y=df["ce_oi"], name="Call OI",
                line=dict(color=cls.COLOR_CALL, dash="dot", width=1.2)
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=df["time_str"], y=df["pe_oi"], name="Put OI",
                line=dict(color=cls.COLOR_PUT, dash="dot", width=1.2)
            ), row=2, col=1)

        fig.update_layout(
            title=dict(
                text=f"NIFTY {session_result.atm_strike} ATM Straddle ({session_result.date}) | Net PnL: ₹{session_result.net_pnl_rupees:,.2f}",
                font=dict(color=cls.COLOR_TEXT, size=16)
            ),
            template="plotly_dark",
            paper_bgcolor=cls.COLOR_BG,
            plot_bgcolor=cls.COLOR_PANEL,
            margin=dict(l=40, r=40, b=40, t=60),
            hovermode="x unified",
            width=950,
            height=650
        )

        out_path = save_path or f"straddle_trajectory_{session_result.date}.html"
        fig.write_html(out_path)
        return out_path

    @classmethod
    def plot_multi_leg_payoff(
        cls,
        structure_name: str,
        legs: List[Dict[str, Union[int, str, float]]],
        lot_size: int = 25,
        num_lots: int = 4,
        spot_center: float = 23650.0,
        span_points: float = 600.0,
        save_path: Optional[str] = None
    ) -> str:
        """
        Renders institutional expiration payoff diagram for multi-leg strategies:
        legs format: [{'strike': 23650, 'type': 'CE', 'side': 'SHORT', 'premium': 115.0}, ...]
        """
        qty = lot_size * num_lots
        spot_range = np.linspace(spot_center - span_points, spot_center + span_points, 300)
        payoff = np.zeros_like(spot_range)

        for leg in legs:
            k = float(leg["strike"])
            opt = str(leg["type"]).upper()
            side = str(leg["side"]).upper()
            prem = float(leg["premium"])

            is_call = opt in ("CE", "CALL", "C")
            intrinsic = np.maximum(0, spot_range - k) if is_call else np.maximum(0, k - spot_range)

            if side in ("SHORT", "SELL", "S"):
                leg_pnl = (prem - intrinsic) * qty
            else:
                leg_pnl = (intrinsic - prem) * qty

            payoff += leg_pnl

        fig = go.Figure()

        # Shaded Profit/Loss regions
        fig.add_trace(go.Scatter(
            x=spot_range, y=payoff, mode="lines", name="Expiry Payoff",
            line=dict(color=cls.COLOR_ACCENT, width=3)
        ))

        # Breakeven line
        fig.add_hline(y=0.0, line_color=cls.COLOR_TEXT, line_width=1)

        # Anchor spot marker
        fig.add_vline(x=spot_center, line_dash="dot", line_color=cls.COLOR_GOLD, annotation_text="Spot Anchor")

        # Compute max profit / loss
        max_p = np.max(payoff)
        min_p = np.min(payoff)

        fig.update_layout(
            title=dict(
                text=f"{structure_name} Expiry Payoff Profile (Max Profit: ₹{max_p:+,.0f} | Max Loss: ₹{min_p:+,.0f})",
                font=dict(color=cls.COLOR_TEXT, size=16)
            ),
            template="plotly_dark",
            paper_bgcolor=cls.COLOR_BG,
            plot_bgcolor=cls.COLOR_PANEL,
            xaxis=dict(title="Index Settlement Price (₹)", gridcolor=cls.COLOR_GRID),
            yaxis=dict(title="Realized P&L at Expiry (₹)", gridcolor=cls.COLOR_GRID),
            margin=dict(l=40, r=40, b=40, t=60),
            width=900,
            height=500
        )

        out_path = save_path or f"payoff_{structure_name.lower().replace(' ', '_')}.html"
        fig.write_html(out_path)
        return out_path

    @classmethod
    def plot_oi_distribution(
        cls,
        chain_snapshot_df: pd.DataFrame,
        spot_price: Optional[float] = None,
        max_pain_strike: Optional[int] = None,
        strike_range: float = 600.0,
        save_path: Optional[str] = None
    ) -> str:
        """
        Renders grouped strike Open Interest distribution highlighting:
        - Major Call Wall (Resistance)
        - Major Put Wall (Support)
        - Settlement Max Pain strike
        - Current spot index level
        """
        df = chain_snapshot_df.copy().fillna(0.0)
        if "strike" not in df.columns:
            raise ValueError("Snapshot must contain 'strike' column")

        if spot_price is not None:
            df = df[abs(df["strike"] - spot_price) <= strike_range]

        df = df.sort_values("strike")

        fig = go.Figure()

        # Put OI (Support) on left/negative or grouped
        fig.add_trace(go.Bar(
            x=df["strike"], y=df.get("pe_oi", 0.0),
            name="Put OI (Support)",
            marker_color=cls.COLOR_PUT,
            opacity=0.85
        ))

        # Call OI (Resistance)
        fig.add_trace(go.Bar(
            x=df["strike"], y=df.get("ce_oi", 0.0),
            name="Call OI (Resistance)",
            marker_color=cls.COLOR_CALL,
            opacity=0.85
        ))

        if spot_price is not None:
            fig.add_vline(
                x=spot_price, line_dash="solid", line_color=cls.COLOR_ACCENT, line_width=2,
                annotation_text=f"Spot: {spot_price:.1f}", annotation_position="top left"
            )

        if max_pain_strike is not None:
            fig.add_vline(
                x=max_pain_strike, line_dash="dash", line_color=cls.COLOR_GOLD, line_width=2,
                annotation_text=f"Max Pain: {max_pain_strike}", annotation_position="top right"
            )

        fig.update_layout(
            title=dict(text="Options Chain Open Interest (OI) & Structural Support/Resistance Walls", font=dict(color=cls.COLOR_TEXT, size=16)),
            barmode="group",
            template="plotly_dark",
            paper_bgcolor=cls.COLOR_BG,
            plot_bgcolor=cls.COLOR_PANEL,
            xaxis=dict(title="Strike Price", gridcolor=cls.COLOR_GRID),
            yaxis=dict(title="Open Interest (Contracts)", gridcolor=cls.COLOR_GRID),
            margin=dict(l=40, r=40, b=40, t=60),
            width=950,
            height=500
        )

        out_path = save_path or "oi_distribution.html"
        fig.write_html(out_path)
        return out_path
