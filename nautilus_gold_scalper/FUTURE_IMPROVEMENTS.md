# Future Improvements - Nautilus Gold Scalper

**Purpose:** Repository for optimization ideas and enhancements. Ideas are captured when insights emerge, NOT during active implementation.

**Status Legend:**
- 💡 **IDEA** - Captured concept, needs evaluation
- 📋 **PLANNED** - Added to roadmap, prioritized
- 🚧 **IN PROGRESS** - Currently being implemented
- ✅ **DONE** - Implemented (moved to archive)
- ❌ **REJECTED** - Decided against (moved to archive with reason)

---

## Active Ideas (Backtest Full Path - P1 Nice-to-Have)

### 1. StrategySelector Integration

**Status:** 💡 IDEA  
**Priority:** P2 (high value, conditional on multi-strategy need)

**WHY:**  
Currently single-strategy system. If/when we expand to multiple strategies (SMC + ICT + OrderBlock variations), need intelligent routing based on market regime/volatility/liquidity. Without StrategySelector, all strategies would trade simultaneously (position sizing conflicts, contradictory signals).

**WHAT:**  
- Implement StrategySelector Actor (NautilusTrader pattern)
- Subscribe to regime detector signals (RANDOM_WALK, TRENDING, MEAN_REVERTING)
- Route market data to appropriate strategy based on regime
- Handle strategy switching logic (close positions on regime change? gradual transition?)
- Config: strategy_priority_map (regime → strategy)

**IMPACT:**  
- **Adaptability:** +30-50% win rate in multi-regime markets
- **Risk:** Reduced whipsaws from wrong strategy in wrong regime
- **Complexity:** +200 LOC, +1 Actor, moderate maintenance

**EFFORT:** 8-12 hours  
**Dependencies:** None (regime detector already implemented)

---

### 2. EntryOptimizer Integration

**Status:** 💡 IDEA  
**Priority:** P3 (nice-to-have, incremental improvement)

**WHY:**  
Current entry logic is fixed threshold (score > 70 → enter). EntryOptimizer could improve fill quality by micro-analyzing tick patterns, spread, order book pressure (if available), and optimal entry timing within bar. Could reduce slippage by 0.5-1 pip per trade.

**WHAT:**  
- Implement EntryOptimizer Actor
- Subscribe to strategy signals (pending entries)
- Analyze 5-10 ticks before entry for optimal timing
- Criteria: tight spread, favorable tick direction, momentum confirmation
- Emit optimized entry signal with adjusted price/timing

**IMPACT:**  
- **Performance:** +0.5-1 pip per entry (slippage reduction)
- **Win Rate:** Potential +2-5% from better fills
- **Effort ROI:** Low (small gains, but cumulative over hundreds of trades)

**EFFORT:** 4-6 hours  
**Dependencies:** Tick data feed with sub-second resolution

---

### 3. YAML Realism Knobs Loader

**Status:** 💡 IDEA  
**Priority:** P3 (convenience, developer experience)

**WHY:**  
Realism parameters (slippage, commission, latency, partial fills) currently hardcoded in run_backtest.py. When tuning for different brokers (Apex vs FTMO vs Tradovate), need to edit Python code. YAML config would enable quick parameter sweeps without code changes.

**WHAT:**  
- Create `config/backtest_realism.yaml`
- Schema: slippage (min/avg/max), commission, latency_ms, partial_fill_probability, spread_multiplier
- Loader function in `src/utils/config_loader.py`
- Update BacktestRunner to accept config dict
- Validation: ensure values within realistic ranges

**IMPACT:**  
- **DX:** Much faster parameter tuning (seconds vs minutes)
- **Testing:** Easier A/B testing of realism scenarios
- **Maintenance:** Config changes don't require code recompilation

**EFFORT:** 6-8 hours (includes YAML schema design + validation + tests)  
**Dependencies:** PyYAML or similar (check if already installed)

---

### 4. Latency & Partial Fill Model

**Status:** 💡 IDEA  
**Priority:** P2 (realism improvement, reduces backtest optimism)

**WHY:**  
Current backtest assumes instant fills at exact price (unrealistic). Real trading has 20-100ms latency (order → exchange → fill), and large orders (>0.5 lot XAUUSD) can get partial fills during fast moves. Ignoring this inflates backtest Sharpe by 10-20%.

**WHAT:**  
- **Latency simulation:**
  - Add configurable delay (e.g., 50ms avg, 20ms std dev)
  - Price can move during delay → fill at worse price if against us
  - Use actual tick data to determine slippage during latency window
- **Partial fill model:**
  - Orders >0.3 lot: 30% chance of partial fill (fill 50-80% immediately, rest delayed 100-500ms)
  - Market orders during high volatility (ATR >0.8): 50% partial fill chance
  - Track unfilled portion, re-submit or cancel based on strategy

**IMPACT:**  
- **Realism:** +20-30% more realistic P&L
- **Risk Management:** Better understanding of worst-case execution
- **Strategy Tuning:** May reveal that large positions are too risky → optimize sizing

**EFFORT:** 8-12 hours (complex logic, needs extensive testing)  
**Dependencies:** High-resolution tick data (1-tick granularity)

---

### 5. Phase 1 Multipliers (GENIUS v4.2 Bandit Feature)

**Status:** 💡 IDEA  
**Priority:** P2 (alpha generation, if edge proven)

**WHY:**  
GENIUS v4.2 introduced adaptive multipliers that increase position size after consecutive wins (exploitation) and reduce after losses (exploration). Multi-armed bandit (MAB) logic balances risk (Kelly) with momentum (trend in win streak). Could boost Sharpe by 15-25% if implemented correctly, but DANGEROUS if overfitted.

**WHAT:**  
- Implement MAB logic in RiskManager
- Track win/loss streaks
- Multiplier formula: `size = base_size * (1 + 0.1 * win_streak) * (1 - 0.15 * loss_streak)` (example)
- Constraints: max_multiplier = 1.5 (never >150% base), min_multiplier = 0.5 (never <50% base)
- Reset multiplier daily or after max drawdown >2%
- **CRITICAL:** Validate with WFA (Walk-Forward Analysis) - if WFE <0.6, REJECT immediately (overfitting)

**IMPACT:**  
- **Alpha:** Potential +15-25% Sharpe if edge is real
- **Risk:** HIGH - can amplify losses if logic flawed or overfitted
- **Validation Required:** ORACLE GO/NO-GO with WFE ≥0.6, Monte Carlo 95th DD <4%

**EFFORT:** 4-6 hours (implementation simple, validation time-consuming)  
**Dependencies:** ORACLE validation (non-negotiable)

**WARNINGS:**  
- ⚠️ DO NOT deploy without ORACLE validation (WFE ≥0.6)
- ⚠️ Can violate Apex 5% trailing DD if multipliers too aggressive
- ⚠️ Requires conservative base sizing (0.5% max risk per trade, not 1%)

---

### 6. Session Weight Profiles

**Status:** 💡 IDEA  
**Priority:** P3 (fine-tuning, incremental edge)

**WHY:**  
XAUUSD has distinct behavior in different sessions:
- **Asian:** Low volatility, ranging (win rate 45-50%)
- **London:** High volatility, trending (win rate 60-65%)
- **NY AM:** Explosive moves, news-driven (win rate 55-60% but high variance)
- **NY PM:** Choppy, institutional profit-taking (win rate 40-45%)

Currently all sessions treated equally. Weighting signals by session could improve risk-adjusted returns by 5-10%.

**WHAT:**  
- Create `config/session_weights.yaml`:
  ```yaml
  sessions:
    asian: {weight: 0.7, max_trades: 2}
    london: {weight: 1.2, max_trades: 5}
    ny_am: {weight: 1.0, max_trades: 3}
    ny_pm: {weight: 0.5, max_trades: 1}
  ```
- Implement session detector (already have time utils)
- In strategy: `adjusted_score = raw_score * session_weight`
- Only trade if adjusted_score > threshold

**IMPACT:**  
- **Selectivity:** Fewer bad trades in low-edge sessions (Asian/NY PM)
- **Alpha:** More capital deployed in high-edge sessions (London)
- **Sharpe:** Potential +5-10% from better capital allocation

**EFFORT:** 3-4 hours (simple, mostly config + basic logic)  
**Dependencies:** Session detection (already have via time MCP)

---

## Implemented (Archive)

_Empty - will be populated as ideas move from 🚧 IN PROGRESS → ✅ DONE_

---

## Rejected (Archive)

_Empty - will be populated as ideas are evaluated and rejected with rationale_

---

## Notes

- **Philosophy:** This is an **ideas repository**, NOT a backlog. Items here may never be implemented.
- **Capture When:** Research findings, post-backtest insights, "aha" moments during implementation (but finish current work first!)
- **Never Add:** Vague ideas without clear WHY/WHAT/IMPACT, already-implemented features, mid-implementation thoughts
- **Status Transitions:** 💡 IDEA → 📋 PLANNED → 🚧 IN PROGRESS → ✅ DONE / ❌ REJECTED

**Last Updated:** 2025-12-08 (minimal path complete, 6 full-path ideas captured)
