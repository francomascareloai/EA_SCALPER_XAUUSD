# Backtest Code Complete Audit Report

<metadata>
<confidence>MEDIUM</confidence>
<modules_audited>45/45</modules_audited>
<bugs_found>11</bugs_found>
<gaps_found>9</gaps_found>
<execution_realism_score>3/10</execution_realism_score>
</metadata>

## Executive Summary

The current Nautilus-based backtest executes the `GoldScalperStrategy`, but several realism and compliance controls are missing or mis-wired. Risk is materially understated because the backtest bypasses Nautilus risk engine, applies no commissions or slippage, and ignores Apex operational rules (daily reset, 4:59 PM ET flat, no overnight). Prop firm daily loss never resets in backtests (manager lacks `on_new_day`), drawdown tracking relies on wall‑clock time, and spread management is reduced to a static point cap with the full SpreadMonitor left unused. Execution thresholds are inconsistent: strategy config is set to 70, tick runner defaults to 65, and bar runner to 50, producing non‑comparable results and masking weak signals. Strategy sizing passes raw price distance to a pip-based sizer, inflating lots when XAUUSD decimals differ. Data loading trusts parquet/CSV without gap or timezone validation. Metrics are minimal (PnL, win rate only); Sharpe/DD bugs noted in the plan remain unaddressed. Tests cover modules in isolation; no end-to-end backtest or Apex rule coverage, confirming “não testa 100%”. Verdict: **NO-GO** until P0/P1 items below are fixed.

## Module Integration Matrix

| Module | Imported? | Used in backtest flow? | Configured? | Issues | Status |
|--------|-----------|-----------------------|-------------|--------|--------|
| strategies/gold_scalper_strategy.py | ✅ | ✅ tick/backtest runners | Partial (hardcoded) | Threshold drift vs runners (70 vs 50/65); sizing unit mismatch | ⚠️ |
| strategies/base_strategy.py | ✅ | ✅ | Yes | Uses wall-clock reset hooks | ⚠️ |
| strategies/strategy_selector.py | ✅ tests only | ❌ | N/A | Not called by runners; selector logic unused | ⚠️ |
| signals/confluence_scorer.py | ✅ | ✅ | Defaults | No major issue seen | ✅ |
| signals/entry_optimizer.py | ✅ tests | ❌ | N/A | Not wired into strategy | ⚠️ |
| signals/mtf_manager.py | ✅ | ✅ | Defaults | Ok | ✅ |
| signals/news_calendar.py | ✅ | ✅ | Hardcoded calendar | 2026+ events missing | ⚠️ |
| signals/news_trader.py | ✅ | ❌ | N/A | Not integrated | ⚠️ |
| indicators/session_filter.py | ✅ | ✅ | allow_asian=False | OK | ✅ |
| indicators/regime_detector.py | ✅ | ✅ | Defaults | OK | ✅ |
| indicators/structure_analyzer.py | ✅ | ✅ | Defaults | OK | ✅ |
| indicators/footprint_analyzer.py | ✅ | ✅ when use_footprint | Defaults | Score weighting not validated | ⚠️ |
| indicators/order_block_detector.py | ✅ | ✅ | Defaults | OK | ✅ |
| indicators/fvg_detector.py | ✅ | ✅ | Defaults | OK | ✅ |
| indicators/liquidity_sweep.py | ✅ | ✅ | Defaults | OK | ✅ |
| indicators/amd_cycle_tracker.py | ✅ | ✅ | Defaults | OK | ✅ |
| indicators/mtf_manager.py (dup) | ✅ | ✅ | Defaults | OK | ✅ |
| context/holiday_detector.py | ✅ tests | ❌ | N/A | Not used in runners | ⚠️ |
| risk/prop_firm_manager.py | ✅ | ✅ | Partial | No daily reset hook; trailing OK | ⚠️ |
| risk/position_sizer.py | ✅ | ✅ | Partial | Expects pips; receives price distance (unit bug) | ⚠️ |
| risk/drawdown_tracker.py | ✅ | ✅ | Partial | Uses wall-clock for daily reset | ⚠️ |
| risk/spread_monitor.py | ✅ | ❌ | N/A | Not wired; only max_spread_points check used | ⚠️ |
| risk/circuit_breaker.py | ✅ tests | ❌ | N/A | Not integrated | ⚠️ |
| risk/var_calculator.py | ✅ | ❌ | N/A | Unused | ⚠️ |
| execution/trade_manager.py | ✅ | ❌ | N/A | Not connected to strategy/orders | ⚠️ |
| execution/base_adapter.py | ✅ | ❌ | N/A | Stub | ⚠️ |
| execution/mt5_adapter.py | ✅ | ❌ | N/A | Stub | ⚠️ |
| execution/ninjatrader_adapter.py | ✅ | ❌ | N/A | Stub | ⚠️ |
| execution/_archive/apex_adapter.py | ✅ | ❌ | N/A | Legacy only | ⚠️ |
| ml/ensemble_predictor.py | ✅ | ❌ | N/A | Not used | ⚠️ |
| ml/feature_engineering.py | ✅ | ❌ | N/A | Not used | ⚠️ |
| ml/model_trainer.py | ✅ | ❌ | N/A | Not used | ⚠️ |
| signals/news_calendar events | ✅ | ✅ | Partial | 2026+ missing | ⚠️ |
| signals/confluence weights (config) | ✅ | ✅ | Hardcoded | YAML not loaded | ⚠️ |
| core/data_types.py | ✅ | ✅ | N/A | Metrics structs unused in runners | ⚠️ |
| core/definitions.py | ✅ | ✅ | N/A | OK | ✅ |
| core/exceptions.py | ✅ | ✅ | N/A | OK | ✅ |
| context/holiday_detector.py | ✅ | ❌ | N/A | Unused | ⚠️ |
| signals/news_trader.py | ✅ | ❌ | N/A | Unused | ⚠️ |
| utils/__init__.py | ✅ | ✅ | N/A | N/A | ✅ |

**Call graph (tick runner)**  
`run_backtest.py` → `GoldScalperStrategy` → {SessionFilter, RegimeDetector, StructureAnalyzer, FootprintAnalyzer (opt), OrderBlockDetector, FVGDetector, LiquiditySweepDetector, AMDCycleTracker, MTFManager, ConfluenceScorer, NewsCalendar, PropFirmManager, PositionSizer, DrawdownTracker}. Modules not in flow: SpreadMonitor, CircuitBreaker, TradeManager, StrategySelector, NewsTrader, ML stack, adapters.

## Strategy Implementation Validation

### GENIUS v4.2 Logic
- 7 ICT sequential steps: **PARTIAL** – OB/FVG/sweep/AMD/structure present; footprint optional; entry optimizer & selector unused, so execution gate reduced. Evidence: no EntryOptimizer call in `gold_scalper_strategy.py`.  
- Session-specific weights: **PARTIAL** – SessionFilter blocks Asia, but session weight profiles from YAML not loaded anywhere.  
- Phase 1 multipliers: **MISSING** – no bandit/phase multipliers applied in `_calculate_position_size` or confluence scorer.  
- strategy_selector.py bypassed; only single strategy runs (no dynamic selection).

### Test Coverage
- Unit tests exist for indicators/risk, but **no test covers `run_backtest.py` or full strategy execution** (see `tests/test_integration/test_strategy_flow.py` uses mocks, not Nautilus engine). Coverage of strategy logic estimated <20%.

## Execution Realism Audit

- **Risk engine bypassed** (`RiskEngineConfig(bypass=True)` in `run_backtest.py`:200) → no margin/commission/slippage enforcement.  
- **Slippage/commission/latency configs ignored**: values in `configs/strategy_config.yaml` never loaded; runners don’t apply any slippage model.  
- **Spreads**: strategy only checks `max_spread_points` against instantaneous spread; statistical SpreadMonitor unused.  
- **Fills**: immediate market orders, no partial fills or rejects; no latency model; no book depth.  
- **Order rejections**: none for margin or volatility.  
- **Apex-specific**:  
  - Trailing DD tracked, but daily loss reset missing (PropFirmManager lacks `on_new_day`; `GoldScalperStrategy` lines 254-259 guarded by hasattr).  
  - **4:59 PM ET flat/no-overnight not implemented** (no time checks in strategy or runners).  
  - 30% daily consistency rule not checked.  
- **Execution Realism Score: 3/10** (basic bid/ask ticks are used; everything else idealized).

## Data Pipeline Validation

- `load_tick_data` (`run_backtest.py`:63) trusts parquet, only tz-converts; **no gap, monotonicity, duplicate, or checksum validation**; no spread sanity checks.  
- Bars aggregated from mid-price only; volume set to tick count, ignoring real sizes.  
- Bar runner loads CSV without timezone awareness beyond UTC parse; no outlier handling.  
- No handling for missing days/holidays; no per-session filtering in loader (only in strategy).

## Metrics Calculation Audit

- Runners print PnL/trade counts only; **no Sharpe/Sortino/Calmar/SQN** computed.  
- Known bugs from plan remain: Sharpe annualization and max_drawdown_pct not implemented (`mass_backtest.py` keeps TODO, lines 228-229).  
- Threshold mismatch: strategy threshold 70 (line 41) but tick runner default 65 (lines 183/442) and bar runner 50 (line 167), breaking comparability and understating false positives.

## Risk Management Validation

- PropFirmManager trailing DD uses HWM and includes unrealized via tick MTM, but **daily loss never resets per backtest day** (no `on_new_day` method; `GoldScalperStrategy` only resets DrawdownTracker).  
- DrawdownTracker daily reset uses `datetime.now()` (line 316) → wall-clock, not simulation clock → resets won’t align with backtest dates.  
- No circuit breaker integration; no size throttling on losing streaks beyond PositionSizer’s internal throttle.  
- Position sizing unit bug: `_calculate_position_size` passes `sl_distance` (price units) to `PositionSizer.calculate_lot(stop_loss_pips=...)` which expects pips → lot sizes mis-scaled when price increment ≠ pip (lines 448-470).  
- Max trades/day enforced (15) but no end-of-day flattening.

## Code Quality Issues

### Critical (P0)
1) Daily loss not reset in prop-firm mode → risk blocks never recover after first drawdown day (GoldScalperStrategy 254-259; PropFirmManager lacks hook).  
2) Risk engine bypass + zero slippage/commissions → materially overstates performance (run_backtest.py:200).  
3) PositionSizer units mismatch (GoldScalperStrategy `_calculate_position_size` lines ~448-470).  
4) No end-of-day flat / overnight ban (search shows no “4:59”/“overnight”).  

### Important (P1)
1) Execution threshold drift (70 in config vs 65/50 in runners).  
2) SpreadMonitor, CircuitBreaker, TradeManager, StrategySelector, EntryOptimizer not integrated → promised safety/entry logic inert.  
3) DrawdownTracker uses wall-clock time; daily reset misaligned in backtests (drawdown_tracker.py:316).  
4) Config YAML not loaded anywhere → slippage/latency/commission knobs dead.  
5) Data loaders lack validation (run_backtest.py:63-95, nautilus_backtest.py:38-66).  

### Minor (P2)
1) News calendar fixed to 2025; 2026+ empty (news_calendar.py TODO 218).  
2) Parameter sweep reuses runner instances without freeing on exception; potential resource leak in long sweeps.  

**Code smells**: 5 TODOs, no silent `except: pass`, many unused modules.  

## Gap Analysis

### Missing Features
1) Slippage/commission/latency modeling tied to config.  
2) SpreadMonitor and CircuitBreaker gating in strategy loop.  
3) Apex operational rules: 4:59 ET flatten, no overnight, 30% daily consistency.  
4) End-of-day resets for PropFirmManager using backtest clock.  
5) StrategySelector / EntryOptimizer integration for promised GENIUS v4.2 pipeline.  

### Missing Tests
1) End-to-end tick/backtest run asserting trades, PnL, DD vs fixtures.  
2) Tests for daily reset and trailing DD across multi-day scenarios.  
3) Tests for slippage/commission application and spread gating.  

### Missing Documentation
1) How to enable realism knobs (slippage/latency) – unused config.  
2) Apex rule coverage and expected behaviors.  

## GO/NO-GO Assessment

**Current Status**: ❌ NO-GO  

**Blockers (P0)**  
1) Daily loss reset missing (prop-firm compliance).  
2) No end-of-day/overnight enforcement for Apex.  
3) Risk engine bypass & no slippage/commissions; results not trustworthy.  
4) Sizing unit mismatch inflates/deflates exposure.  

## Recommendations

### Immediate Actions (P0)
1) Add backtest-clock daily reset hook to PropFirmManager and wire into strategy start/each bar; enforce 4:59 PM ET flatten + no overnight.  
2) Load `configs/strategy_config.yaml` and apply slippage/latency/commission to fills; disable `RiskEngineConfig(bypass=True)`.  
3) Fix sizing units: convert `sl_distance` to pips before calling PositionSizer; align pip_value with XAUUSD tick size.  
4) Harmonize execution thresholds to 70 (or target value) across all runners; add CLI flag default.  

### Important Actions (P1)
1) Integrate SpreadMonitor + CircuitBreaker into signal gate and size multiplier.  
2) Validate data on load (monotonic timestamps, gap detection, NaN/outlier rejection); fail fast if bad.  
3) Implement slippage/commission Monte Carlo toggles for sweeps; record applied costs in telemetry.  
4) Add end-to-end pytest covering run_backtest + multi-day Apex rules.  

### Nice-to-Have (P2)
1) Use EntryOptimizer for placement; integrate StrategySelector for regime-based handoff.  
2) Expand NewsCalendar to rolling API/CSV loader for 2026+.  

## Next Steps

1) Implement P0 fixes (risk resets, slippage, sizing units, threshold alignment) and rerun tick backtest.  
2) Add data validation layer before engine ingest; log rejected spans.  
3) Wire SpreadMonitor/CircuitBreaker into strategy and backtest realism knobs.  
4) Add end-to-end tests + telemetry metrics (Sharpe/Sortino/Calmar/SQN, DD%).  

<open_questions>
- Target execution threshold: confirm 70 vs 65?  
- Required slippage/commission values for Apex realism?  
- Should bar runner remain supported or migrate fully to tick runner?  
</open_questions>

<assumptions>
- XAUUSD tick size 0.01 and pip_value $10 used unless otherwise specified.  
- Apex trailing DD 10% of HWM including unrealized.  
</assumptions>

<dependencies>
- Need updated tick/bar datasets with integrity checks.  
- Confirmation of Apex operational cutoffs to code exact times (ET).  
</dependencies>
