---
description: MQL5 Elite Ops - Daily Market Intelligence & Risk Adjustment
---

# MQL5 Elite Ops: Market Intelligence Scan

This workflow runs a daily scan to adjust the EA's risk parameters based on real-world conditions.

## Step 1: Deep Researcher Scan
**Agent:** `Deep Researcher`
**Directives:**
1.  **News Scan:** Identify High-Impact events for the day (USD, XAU).
2.  **Sentiment Analysis:** Analyze current market sentiment (Risk-On vs Risk-Off).
3.  **Volatility Check:** Check VIX or ATR levels.
4.  **Output:** A "Daily Market Briefing".

## Step 2: Quantum Strategist Risk Adjustment
**Agent:** `Quantum Strategist`
**Directives:**
1.  Based on the Briefing, determine the **Risk Mode**:
    - 🟢 **Normal:** Standard Lots, Standard SL.
    - 🟡 **Caution:** Reduced Lots (e.g., 50%), Tight SL.
    - 🔴 **Survival:** Trading Paused or "Close Only".
2.  **Output:** Specific parameter overrides for the EA (e.g., `RiskPercent = 0.5`).

## Step 3: Code Artisan Deployment (Simulated)
**Agent:** `Code Artisan`
**Directives:**
1.  Confirm how these parameters would be updated in the **Local Cache** (e.g., updating `trading_params.json`).
2.  **Output:** Confirmation of the update mechanism.

---
**Next Steps:**
- User confirms the risk mode.
- (Optional) Generate the JSON config file.
