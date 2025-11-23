---
description: MQL5 Elite Ops - Comprehensive Strategy Refinement & Review
---

# MQL5 Elite Ops: Strategy Refinement Protocol

This workflow orchestrates the "Council of Agents" to rigorously analyze, refine, and validate a trading strategy.

## Step 1: Strategy Intake
**User Action:** Please provide the full details of the strategy you want to refine. This can be:
- A description of the logic (Entry, Exit, Risk).
- An existing `.mq5` file path.
- A rough idea or hypothesis.

## Step 2: Quantum Strategist Analysis
**Agent:** `Quantum Strategist`
**Directives:**
1.  Analyze the strategy for **Logical Flaws** (e.g., repainting, curve fitting).
2.  Evaluate compliance with **Prop Firm Rules** (Drawdown, News Trading).
3.  **Survival Mode Check:** Does the strategy have a defined "Safe State" if Python disconnects?
4.  **Output:** A "Strategy Stress Report" with Pass/Fail grades.

## Step 3: MQL5 Architect Review
**Agent:** `MQL5 Architect`
**Directives:**
1.  Review the proposed implementation structure.
2.  **Async Verification:** Ensure no blocking calls are required.
3.  **Heartbeat Check:** Confirm where the heartbeat/watchdog logic fits in.
4.  **Output:** An "Architectural Blueprint" or "Refactoring Plan".

## Step 4: Deep Researcher Context (Optional)
**Agent:** `Deep Researcher`
**Directives:**
1.  Assess if this strategy fits the **Current Market Regime** (e.g., High Volatility, War, Inflation).
2.  Identify "Kill Zones" (News events that would destroy this specific strategy).
3.  **Output:** A "Market Viability Score".

## Step 5: Code Artisan Implementation
**Agent:** `Code Artisan`
**Directives:**
1.  Propose the **Code Structure** (Classes, Functions).
2.  **Local Persistence:** Define what data needs to be cached (e.g., `NewsCalendar.csv`).
3.  **Output:** Pseudocode or actual MQL5 snippets for the critical logic.

## Step 6: Backtest Commander Protocol
**Agent:** `Backtest Commander`
**Directives:**
1.  Define the **Validation Protocol** (Dates, Symbols, Delays, Slippage).
2.  Define **Robustness Tests** (Monte Carlo, Walk-Forward).
3.  **Output:** A "Validation Checklist".

---
**Next Steps:**
- Ask the user which part of the plan they want to execute first.
