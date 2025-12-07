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

## 2025-12-07 (POST-VALIDATION import tests)

### Module: nautilus_gold_scalper/src/ml/model_trainer.py
**Bug:** NameError when onnxruntime import fails
- **Symptom:** `NameError: name 'ort' is not defined` at class definition time (line 637)
- **Impact:** ML module completely unusable - couldn't even import the file
- **Root Cause:** Type hint `-> ort.InferenceSession` evaluated at class definition time, but `ort` imported in try/except block and may not exist
- **Fix:** Added `from __future__ import annotations` at top of file to defer type hint evaluation until runtime
- **Why missed:** Audit 001 checked file existence but didn't run actual import tests

### Module: nautilus_gold_scalper/src/ml/ensemble_predictor.py
**Bug:** NameError when onnxruntime import fails (same as model_trainer.py)
- **Symptom:** `NameError: name 'ort' is not defined` at class definition time (line 635)
- **Impact:** ML ensemble prediction completely unusable
- **Root Cause:** Type hint `-> ort.InferenceSession` in method signature but `ort` may not exist
- **Fix:** Added `from __future__ import annotations` at top of file
- **Detection:** Discovered during post-audit import validation testing

**Result:** All 11 critical modules (8 core + 3 ML) now import successfully. Validates importance of running actual import tests, not just file checks.

---

## 2025-12-07 (FORGE - Metrics telemetry implementation)

### Module: nautilus_gold_scalper/src/utils/metrics.py (NEW)
**Issue:** No Sharpe/Sortino/Calmar/SQN telemetry for GO/NO-GO decisions
- **Symptom:** Backtests completed but lacked performance metrics needed for validation plan Phase 6
- **Impact:** Cannot make GO/NO-GO decision without core metrics (Audit 003 P0 blocker)
- **Root Cause:** No metrics calculation module existed, only basic PnL tracking
- **Fix:** Created comprehensive MetricsCalculator with:
  - Sharpe Ratio (annualized, risk-adjusted return)
  - Sortino Ratio (downside deviation only)
  - Calmar Ratio (CAGR / Max DD)
  - SQN - System Quality Number (Van Tharp)
  - Supporting metrics: Win Rate, Profit Factor, Expectancy, Max DD
  - Handles edge cases: zero std_dev (perfect consistency) → inf Sharpe/SQN
- **Testing:** 9 unit tests (100% pass), validates edge cases (all wins, all losses, empty, high Sharpe)
- **Evidence:** Prompt 005 requires these metrics for GO/NO-GO framework decision tree

### Module: nautilus_gold_scalper/src/strategies/gold_scalper_strategy.py
**Integration:** Metrics calculation into strategy lifecycle
- **Added:** `_trade_pnl_history` list to track all realized PnLs
- **Added:** `_metrics_calculator` instance (risk_free_rate=0.05, 252 trading days)
- **Added:** `_calculate_and_emit_metrics()` method - calculates metrics and emits to telemetry
- **Hooked:** `_on_strategy_stop()` calls metrics calculation on backtest completion
- **Output:** Metrics emitted to `logs/telemetry.jsonl` in structured JSON format

### Module: nautilus_gold_scalper/src/strategies/base_strategy.py
**Integration:** PnL tracking in position close event
- **Added:** PnL tracking in `on_position_closed()` after circuit breaker update
- **Logic:** Appends `net_pnl` to `_trade_pnl_history` if attribute exists
- **Safe:** Uses `hasattr()` check for backward compatibility

**Result:** Metrics telemetry P0 blocker RESOLVED. Backtests now output Sharpe/Sortino/Calmar/SQN for GO/NO-GO validation.

---
