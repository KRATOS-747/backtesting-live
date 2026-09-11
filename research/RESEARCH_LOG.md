# Desk Quantitative Research Logbook (`RESEARCH_LOG.md`)
**Desk / Author**: Systematic Derivatives & Quantitative Research  
**Primary Focus**: NSE/BSE Weekly Options, Greeks Dynamics, Expiry Microstructure, and Residual Factor Models  
**Time Horizon**: 2023 – Present  

This document tracks hypotheses, empirical tests, live deployment iterations, and post-mortems across market regimes.

---

### [2023-Q2] Hypothesis: Does Intraday Open Interest (OI) Build-up Predict 15-Minute Spot Direction?
* **Hypothesis**: Rapid increases in Call OI indicate institutional writing (resistance), predicting downward mean-reversion in spot Nifty within the next 15–30 minutes.
* **Empirical Test**: Computed 5-min rolling $\Delta \text{OI}$ and Call/Put OI velocity across 180 trading sessions. Evaluated forward Information Coefficient (IC) at $t+5\text{m}, t+15\text{m}, t+30\text{m}$.
* **Findings**:
  * $IC(15\text{m}) = +0.024$ with $t\text{-stat} = 1.18$ (statistically insignificant).
  * In fact, during strong momentum days, heavy Call writing gets aggressively short-squeezed, leading to catastrophic losses if traded directionally.
* **Conclusion & Decision**: **ABANDONED** raw OI as a directional predictor. Repurposed OI change as an **overbought/oversold volatility fader**—shorting the inflated premium of retail "long buildup" rather than taking directional spot futures risk.

---

### [2023-Q4] Expiry Shift: Bank Nifty Expiry Migration (Thursday $\to$ Wednesday)
* **Market Event**: NSE shifted Bank Nifty weekly expiries from Thursday to Wednesday.
* **Impact Analysis**:
  * Capital recycling opportunity: Two distinct high-gamma expiry days per week (Wed for Bank Nifty, Thu for Nifty).
  * Observed increased intraday IV crush on Wednesday afternoon post-13:30, with Bank Nifty ATM straddles losing up to 45% of value between 13:30 and 15:15.
* **Implementation**: Built dedicated Wednesday mean-reversion module (`options_bt/`). Calibrated separate strike step parameters (100 pts for Bank Nifty vs 50 pts for Nifty).

---

### [2024-Q1] Survivorship Bias Audit: Naive Static Nifty 500 vs. Point-in-Time (PIT)
* **Audit Motivation**: When testing medium-frequency factor strategies over 2015–2023, using the current 2024 Nifty 500 constituent list produced an annualized Sharpe of 2.38.
* **Investigation**: Built `PointInTimeUniverse` parser tracking every inclusion and exclusion event from `IndexInclExcl 2.xls`.
* **Findings**:
  * Testing with true PIT constituents dropped the naive Sharpe from 2.38 to 1.74.
  * Over 40% of the apparent "alpha" in the naive backtest came from survivorship bias (holding winners that made it into the index while ignoring failed/delisted names).
* **Action**: Enforced mandatory PIT universe resolution for all equity factor backtests (`equity_bt/pit_universe.py`).

---

### [2024-Q2] Post-Mortem & Parameter Evolution: The 2.0x SL Failure on Trend Days
* **Problem**: In early iterations of `OIlongbuilup.py`, short option legs utilized a static `STOP_LOSS_MULTIPLIER = 2.0` (stop hits if premium doubles).
* **Observed Flaw**: On fast-trending breakout days, premium doubles within 2–3 bars, incurring full -100% loss. Furthermore, slippage on market stop-orders in volatile options pushed effective exits to 2.4x–2.8x.
* **Research Solution**:
  * Formulated **Rate-of-Change (ROC) Leg-Cutting Engine** (`options_bt/leg_cutter.py`).
  * Instead of waiting for a 100% price move, we compute the 5-bar ROC: if option premium expands $>35\%$ over 5 bars and shows loss $>10$ points, the leg is cut immediately.
  * Result: Reduced average losing leg drawdown by 38% and eliminated fat-tail slippage on breakout days.

---

### [2024-Q3] Factor Modeling: Why Raw Momentum Whipsaws vs. PCA Residual Momentum
* **Hypothesis**: Stripping systematic Market Beta (PC1) and Sector/Style Beta (PC2) from Nifty 500 stock returns isolates true idiosyncratic alpha, cutting portfolio drawdowns during broad market corrections.
* **Empirical Results**:
  * Raw 12-Month Momentum: Max Drawdown = -31.4% (whipsawed heavily during sector rotations).
  * 10-Tranche PCA Residual Momentum: Max Drawdown = -16.2%, Calmar Ratio improved from 0.82 to 1.45.
* **Deployment**: Staggered execution into 10 tranches (rebalancing 1 tranche every 12.5 days) paired with rank buffering (slot 30, buffer 270) to suppress turnover friction.

---

### [2025-Q1] Legging Risk & Pre-Trade Microstructure Controls
* **Desk Observation**: During high-volatility opens (09:15–09:25), submitting independent Call and Put market orders resulted in 150–350ms execution delay between legs, leaving the desk briefly exposed to naked delta.
* **Engineering Solution**: Developed `MultiLegRouter` in `oms/`:
  * Enforces atomic dual-leg IOC execution.
  * If one leg fails to fill, automatically triggers an immediate `EMERGENCY_UNWIND` on the filled leg or deploys a synthetic future hedge within the same event loop.

---

### [2025-Q1] Empirical Study: Edge vs. Alpha (The Rolling Straddle Friction Trap)
* **Core Premise**: Testing whether a popular retail "edge" (selling ATM straddles and rolling whenever spot moves $\pm 40$ points) survives Indian statutory frictions.
* **Empirical Audit (January 2025 - All 23 Sessions)**:
  * **The Raw Edge (Flat 40pt Roll)**: Executed **258 rolls in 23 days** (11.2 rolls/day). While generating +₹31,665.00 in gross returns, it paid **₹35,118.41 in statutory taxes, GST, and brokerage**, finishing with a **net loss of -₹3,453.41** (Win Rate: 43.5%).
  * **The Engineered Alpha (Spot-ROC Adaptive)**: Conditioned the roll trigger on 5-minute Spot Rate-of-Change ($ROC_{5\text{m}}$). Widening the threshold during low-velocity drift ($<0.08\%$) and freezing rolls during momentum impulses ($>0.18\%$) cut churn rolls by **37%** (down to 163 rolls), saved **₹11,730 in fees**, and boosted Net Realized P&L to **+₹20,221.57** (Win Rate: 60.9%).
* **Artifacts & Notebook**: Detailed in `research/01_edge_vs_alpha_rolling_straddles.ipynb` and visualized in `reports/visuals/edge_vs_alpha_equity_friction.html`.

---

### [2025-Q1] Empirical Study: Microstructure Spike Fading (Alpha 2) & The 90-Minute Sweet Spot
* **Core Premise**: Does fading sharp option price surges (12%+ in 5 minutes) yield positive expectancy, or is it a retail illusion crushed by trend breakouts and statutory friction?
* **Empirical Audit (January 2025 - All 23 Sessions - NIFTY 1 Lot / 75 Qty)**:
  * **The Naive Fallacy**: Shorting all liquid near-the-money spikes triggered **6,546 trades**, generated +₹80,148.75 gross, but surrendered **₹9,34,237.76 in statutory taxes, fees, and slippage**, ending in a catastrophic **-₹8,54,089.01 net loss** (-1065% conversion).
  * **The Confluence Filter**: Requiring the surge to test the Primary Call/Put OI Wall (within 35 pts) with underlying 1-min spot momentum stalling ($ROC_{1\text{m}} \le 0.02\%$) cut noise by **96.2%** (down to 246 trades), delivering **+₹11,875.12 net profit** (26.4% conversion).
  * **The 90-Minute Horizon Discovery**: Premature 15-minute exits strangulated positions during the initial "dead-band" consolidation. Extending the holding horizon to **90 minutes** allowed full long unwinding and intraday Theta decay to take effect:
    * Total Trades: **165 trades**
    * Win Rate: **70.3%** (vs 55.3% at 15m)
    * Profit Target Rate: **66.7%** (vs 30.1% at 15m)
    * Gross P&L: **+₹98,658.75**
    * Statutory Friction: **-₹22,274.98**
    * **Net Realized P&L**: **+₹76,383.77**
    * **Net-to-Gross Conversion**: **77.4%** (reaching the institutional 70%–80% target).
  * **The Double-Leg Straddle Penalty**: Replacing the single naked leg with a 2-leg ATM Straddle on the same triggers expanded gross P&L by +23% (+₹55.5k), but doubled brokerage and slippage to -₹64.5k, resulting in a **-₹9,022.71 net loss**. Conclusion: Straddles are macro Theta harvesters (Alpha 1); event scalps require single-leg precision (Alpha 2).
* **Artifacts & Notebook**: Fully detailed in `research/02_microstructure_spike_fading.ipynb`, implemented in `options_bt/spike_fader.py`, and visualized in `reports/visuals/spike_fade_equity_curve.html` and `reports/visuals/holding_horizon_sensitivity.html`.


