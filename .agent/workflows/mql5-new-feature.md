---
description: MQL5 Elite Ops - End-to-End Feature Implementation
---

# MQL5 Elite Ops: New Feature Implementation

This workflow guides the creation of a new feature (e.g., "Trailing Stop based on ATR") from concept to code.

## Step 1: Feature Definition
**User Action:** Describe the feature you want to add.

## Step 2: Quantum Strategist Requirements (PRD)
**Agent:** `Quantum Strategist`
**Directives:**
1.  Define the **Logic** (Mathematical formula, Trigger conditions).
2.  Define **Parameters** (Inputs exposed to the user).
3.  Define **Edge Cases** (What happens if ATR is 0?).
4.  **Output:** Mini-PRD for the feature.

## Step 3: MQL5 Architect Design
**Agent:** `MQL5 Architect`
**Directives:**
1.  Determine **Where** this feature lives (Which Class/Module?).
2.  Define the **Interface** (Public methods).
3.  **Output:** Class Diagram or Header file snippet.

## Step 4: Code Artisan Implementation
**Agent:** `Code Artisan`
**Directives:**
1.  Write the **Actual MQL5 Code**.
2.  Ensure **Optimization** (No heavy calculations inside loops).
3.  **Output:** The complete code block.

## Step 5: Backtest Commander Verification
**Agent:** `Backtest Commander`
**Directives:**
1.  How do we **Verify** this specific feature? (Visual Mode? Print logs?).
2.  **Output:** A quick verification test.

---
**Next Steps:**
- Copy the code to the project.
- Compile and Test.
