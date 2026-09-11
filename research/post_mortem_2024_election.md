# Desk Incident & Post-Mortem: June 4, 2024 (Lok Sabha Election Results)
**Date**: June 5, 2024  
**Scope**: Microstructure breakdown, IV explosion, circuit limit halts, and execution slippage analysis  
**Asset Classes**: NIFTY 50 & BANK NIFTY Weekly Options  

---

### 1. Executive Summary
On Tuesday, June 4, 2024, as the vote tally diverged from weekend exit poll projections, Nifty 50 crashed from an open of 23,179 down to an intraday low of 21,281 (-8.2% peak drop), with India VIX exploding from 15.5 to above 31.0 (+100% intraday vol expansion).

This event represented a **$6\sigma$ stress event** for Indian systematic derivatives desks. This post-mortem documents how our quantitative models, execution gates, and risk rails behaved under extreme dislocation.

---

### 2. Microstructure & Market Anomalies Observed

#### A. The Bid-Ask Spread Blowout
* **Normal Market**: ATM Nifty weekly options typically trade with a tight 0.15% to 0.35% half-spread (₹0.50 to ₹1.00 on a ₹250 straddle).
* **June 4 Dislocation**: Between 10:45 AM and 12:30 PM, market-maker quotes completely pulled back. ATM spreads widened to **₹15.00 – ₹28.00 (6.0% to 11.0% of premium)**.
* **Impact**: Naive algos using market orders crossed massive spreads, experiencing catastrophic immediate fill slippage.

#### B. Gamma Squeeze on Short OTM Puts
* 22,000 PE (trading around ₹18 at 09:30 AM) traded as high as **₹820.00** by 12:15 PM (a 45x surge).
* Traders holding naked delta-hedged straddles could not rebalance delta fast enough because spot was moving faster than the 1-minute bar calculation cycle.

#### C. Exchange Circuit Limits & Reject Floods
* Multiple option strikes triggered price execution bands (circuit limits). Orders attempting to hedge or square off were rejected with broker error codes (`RMS:Rule: Price out of execution range`).

---

### 3. Strategy Performance & Risk Rails Audit

| Strategy Module | Behavior Under Shock | Resolution / Desk Action |
|---|---|---|
| **09:25 ATM Straddle Harvester** | **BLOCKED BY PRE-TRADE GATE** | Our pre-trade risk filter detected opening 15m Parkinson volatility $> 3.5\times$ 20-day median and blocked new short vol entries. |
| **Legacy Static 2.0x SL Scripts** | **FAILED ON SLIPPAGE** | In paper simulation, static 2.0x stop loss triggered at 10:18 AM, but market fill occurred at 3.2x entry due to zero liquidity at the stop barrier. |
| **ROC Leg-Cutting Engine** | **SUCCESSFUL MITIGATION** | Cut the Call leg early at +18 pts profit and cut the exploding Put leg via the ROC acceleration gate at +35% expansion, avoiding the subsequent 45x spike. |
| **L2 Order Book Router (`oms/`)** | **PREVENTED BAD FILLS** | `check_liquidity_impact()` rejected 4 rebalancing orders because estimated impact exceeded the 25 bps tolerance ceiling. |

---

### 4. Desk Rule Adjustments Implemented Post-Event

1. **Mandatory India VIX Gate**:
   * If India VIX $> 22.0$ at 09:20 AM, all intraday premium-selling strategies are automatically throttled to 25% allocation or entirely suspended.
2. **Strict Time-In-Force (IOC) Enforcement**:
   * No market orders allowed in options. All rebalancing orders must be IOC limit orders priced no worse than VWAP + `max_slip_bps`.
3. **Emergency Legging Hedge Mode**:
   * If one leg of a multi-leg hedge fails to fill within 250ms, the system immediately fires an opposing synthetic futures order rather than waiting in the options book.
