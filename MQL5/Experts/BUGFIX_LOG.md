# BUGFIX LOG - EA_SCALPER_XAUUSD

## Purpose
Central log for all bug fixes in MQL5 and Python code. Every agent fixing bugs MUST document here.

---

## 2025-12-07 (FORGE - ORACLE audit fixes)

### Module: nautilus_gold_scalper/src/strategies/gold_scalper_strategy.py
**Bug #2 (ORACLE):** execution_threshold set to 65 instead of 70 (TIER_B_MIN)
- **Symptom:** Strategy accepted TIER-C signals (60-69 score) when it should only accept TIER-B+ (70+)
- **Impact:** Poor win rate due to low-quality signal acceptance
- **Root Cause:** Threshold lowered from MQL5 standard (70) during migration
- **Fix:** Line 67 changed from 65 to 70 to match MQL5 TIER_B_MIN
- **Evidence:** MQL5/Include/EA_SCALPER/Scoring/CConfluenceScorer.mqh uses TIER_B_MIN = 70

### Module: nautilus_gold_scalper/src/strategies/gold_scalper_strategy.py
**Bug #4 (ORACLE):** ConfluenceScorer instantiated without min_score_to_trade config
- **Symptom:** Config parameter `execution_threshold` defined but never passed to scorer
- **Impact:** Scorer used default TIER_INVALID (60) instead of config value (70)
- **Root Cause:** Constructor call missing parameter during migration
- **Fix:** Line 162-164 now passes `min_score_to_trade=float(self.config.execution_threshold)` to ConfluenceScorer constructor
- **Evidence:** confluence_scorer.py line 951 checks `self.min_score_to_trade` but it was never set from config

**Result:** Both bugs fixed. Strategy now enforces TIER-B minimum (70) consistently with MQL5.

---
