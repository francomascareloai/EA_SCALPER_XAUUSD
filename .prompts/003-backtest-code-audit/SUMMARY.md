# Backtest Code Audit - Summary

## One-Liner
P0s cleared (risk engine on, slippage applied, cutoff/overnight enforced); **status CONDITIONAL** — can run internal tests now, but 9 P1 gaps block production GO (modules unused, no E2E tests, metrics missing, <20% coverage).

## Version
v4 - Complete gap analysis (2025-12-07)

## Key Findings

### ✅ **COMPLETED (P0 - Critical)**
- Risk engine enforced; slippage applied to ticks; threshold aligned to 70
- Apex: 4:59 PM ET flatten + no-overnight; PropFirmManager daily reset via `on_new_day`
- PositionSizer receives SL in pips; SpreadMonitor affects score/size
- Commission netted in backtest summary

### ⚠️ **PENDING (P1 - Blocks Production)**

**Module Integration** (4 gaps):
1. CircuitBreaker: exists but NOT called in live path
2. StrategySelector: bypassed (no dynamic selection)
3. EntryOptimizer: not wired into strategy
4. SpreadMonitor: used but no telemetry/logging

**Configuration** (1 gap):
5. YAML realism knobs (slippage/commission/latency) NOT loaded by runners

**GENIUS v4.2 Logic** (2 gaps):
6. Phase 1 multipliers MISSING (no bandit/phase multipliers)
7. Session weight profiles NOT applied (only Asia blocking)

**Testing & Metrics** (3 gaps):
8. Test coverage <20% (no E2E tests for backtest runs)
9. Metrics MISSING: Sharpe/Sortino/Calmar/SQN/DD%

**Execution Realism Score: 5/10** (no latency/partial fills/rejections)

### 📊 **TESTING READINESS**

**Minimal Path** (can start testing with current code):
- Add E2E test for tick backtest (validates P0 fixes)
- Add basic metrics output (Sharpe + max DD%)
- Fix DrawdownTracker to use backtest clock (not wall-clock)
**Effort: 1-2 days** → Gets to **testable but not production-ready**

**Full Path** (production GO):
- Complete minimal path above
- Load YAML realism knobs + CLI overrides
- Integrate CircuitBreaker + StrategySelector + EntryOptimizer
- Add latency/partial-fill model
- Complete metrics suite (Sharpe/Sortino/Calmar/SQN/DD%)
- Implement 30% Apex daily consistency check
- Achieve >50% test coverage
**Effort: 5-7 days** → Production-ready with full validation

## Decisions Needed

1. **Start testing now** with minimal path (1-2 days) OR wait for full path (5-7 days)?
2. Approve effort for P1 integration (CircuitBreaker/Selector/Optimizer)?
3. ML stack (ensemble_predictor, feature_engineering) — remove or integrate?

## Blockers

**For starting tests**: None (can start with minimal path)
**For production GO**: 9 P1 items above must be completed

## Next Step

**RECOMMENDED**: Implement minimal path first (E2E test + metrics + DrawdownTracker fix), validate P0 fixes work, then decide on full P1 integration based on backtest results.
