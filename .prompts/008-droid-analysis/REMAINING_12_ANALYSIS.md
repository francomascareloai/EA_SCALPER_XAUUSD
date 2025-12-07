# Remaining 12 Droids Analysis (LAYER 2)

## Executive Summary
- **Total size**: ~193KB (50% of ecosystem, NOT in TOP 5)
- **Estimated redundancy**: ~130KB (67% similar to TOP 5 pattern)
- **Potential savings**: ~130KB (67% reduction)
- **Critical droids**: orchestrator (MAESTRO candidate), onnx-model-builder (ML pipeline)
- **Low-priority droids**: personal droids (if not used for project)

---

## Critical Droids (MUST REFACTOR)

### ea-scalper-xauusd-orchestrator ⚠️ **MAESTRO CANDIDATE**
- **Size**: ~20KB
- **Redundancy**: ~60% (generic orchestration patterns)
- **Specialization**: 
  - Coordinates 6 droids (CRUCIBLE, SENTINEL, FORGE, ORACLE, ARGUS, NAUTILUS)
  - Routes based on trigger keywords
  - Enforces Apex Trading constraints
  - BUILD>PLAN philosophy enforcement
- **Priority**: **CRITICAL** (should be MAESTRO)
- **Recommendation**: **REFACTOR NOW + ELEVATE TO MAESTRO**
  - Add workflow DAG knowledge (3 workflows: Strategy Dev, Code Review, Research→Design)
  - Implement automatic invocation based on dependency graph
  - Add decision_hierarchy awareness (7 levels)
  - Elevate to priority 7 in AGENTS.md authority hierarchy
- **Current issues**:
  - Coordination is manual (user must invoke correct droid)
  - No automatic workflow execution
  - No conflict resolution logic
- **Proposed enhancements**:
  - Auto-detect workflow from user intent
  - Invoke droids in correct DAG order
  - Handle conditional steps (e.g., "only if ORACLE says NO-GO")
  - Loop when needed (fix → re-test → validate)
  - Report progress transparently

---

### onnx-model-builder ⚠️ **ML CRITICAL**
- **Size**: ~20KB
- **Redundancy**: ~65%
- **Specialization**:
  - ML model design (LSTM, GRU, CNN, Transformer)
  - Feature engineering for trading
  - ONNX export for MQL5 integration
  - Regime detection models (Hurst, Entropy, HMM)
  - WFA validation (WFE ≥0.6 required)
- **Priority**: **CRITICAL** (ML is core to strategy)
- **Recommendation**: **REFACTOR NOW**
- **Integration gaps**:
  - Does NOT auto-hand off to ORACLE for validation (should)
  - Does NOT integrate with FORGE for code review (should)
  - No dependency on ARGUS for research (could benefit)
- **Proposed workflow**:
  - ARGUS researches ML approach → ONNX-MODEL-BUILDER trains model → ORACLE validates WFE → FORGE integrates into strategy → NAUTILUS backtests

---

## High Priority Droids (REFACTOR SOON)

### crucible-gold-strategist
- **Size**: ~10KB
- **Redundancy**: ~70%
- **Specialization**:
  - XAUUSD-specific strategy analysis
  - SMC patterns (Order Blocks, FVG, Liquidity)
  - Backtest quality validation (realism gates)
  - Prop firm rules (Apex, Tradovate)
- **Priority**: **HIGH**
- **Overlap**: **ORACLE** (both validate backtests)
  - CRUCIBLE: Setup quality + backtest realism
  - ORACLE: Statistical validation (WFA, Monte Carlo)
  - Resolution: Keep both, CRUCIBLE → ORACLE handoff
- **Recommendation**: **REFACTOR LATER** (after TOP 5 + orchestrator + onnx)

---

### argus-quant-researcher
- **Size**: ~15KB
- **Redundancy**: ~70%
- **Specialization**:
  - Quant/trading-specific research
  - Multi-source triangulation
  - arXiv/SSRN paper analysis
  - Evidence-based synthesis
- **Priority**: **HIGH**
- **Overlap**: **research-analyst-pro** (similar triangulation methodology)
  - ARGUS: Trading/quant-specific
  - research-analyst-pro: General multi-source research
  - Resolution: Keep both OR add domain parameter
- **Recommendation**: **REFACTOR LATER** (after CRITICAL droids)

---

### code-architect-reviewer
- **Size**: ~12KB
- **Redundancy**: ~75%
- **Specialization**:
  - Architecture/design review
  - High-level code structure
  - System design patterns
- **Priority**: **MEDIUM**
- **Overlap**: **senior-code-reviewer** (personal droid) + **FORGE**
  - code-architect-reviewer: Architecture focus
  - senior-code-reviewer: General best practices
  - FORGE: Python/Nautilus-specific
  - Resolution: **MERGE** code-architect-reviewer + senior-code-reviewer → generic-code-reviewer
- **Recommendation**: **MERGE + REFACTOR** (Phase 2)

---

### project-reader
- **Size**: ~8KB
- **Redundancy**: ~65%
- **Specialization**:
  - Project structure analysis
  - Codebase orientation
  - Documentation navigation
- **Priority**: **MEDIUM**
- **Recommendation**: **REFACTOR LATER** (Phase 2)
- **Usage**: Infrequent (onboarding, initial project exploration)

---

### trading-project-documenter
- **Size**: ~10KB
- **Redundancy**: ~70%
- **Specialization**:
  - Trading system documentation
  - MQL5 EA documentation
  - Multi-strategy architecture docs
- **Priority**: **MEDIUM**
- **Recommendation**: **REFACTOR LATER** (Phase 2)
- **Usage**: Infrequent (after major milestones)

---

### deep-researcher
- **Size**: ~12KB
- **Redundancy**: ~70%
- **Specialization**:
  - Deep multi-layer research
  - Complex trading/quant topics
  - Triangulation (academic + industry + empirical)
- **Priority**: **MEDIUM**
- **Overlap**: **argus-quant-researcher** + **research-analyst-pro**
  - All 3 do multi-source research with triangulation
  - deep-researcher: Most comprehensive (v2.1 enhanced)
  - ARGUS: Quant-focused
  - research-analyst-pro: General
  - Resolution: Consider **MERGING** into single research droid with complexity parameter
- **Recommendation**: **EVALUATE MERGE** (Phase 3)

---

## NANO Versions (Special Case)

### nautilus-nano
- **Size**: ~4KB (compact version of nautilus-trader-architect)
- **Purpose**: Party Mode efficiency
- **Status**: Already optimized
- **Recommendation**: **KEEP AS-IS** (serves specific use case)

---

## Secondary Droids (Lower Priority)

### sentinel-ftmo-guardian
- **Size**: ~15KB
- **Redundancy**: ~70%
- **Specialization**: FTMO prop firm rules (10% fixed DD, different from Apex)
- **Priority**: **LOW** (project uses Apex, not FTMO)
- **Recommendation**: **ARCHIVE** or keep if planning FTMO challenge later

---

### git-guardian
- **Size**: ~6KB
- **Redundancy**: ~60%
- **Specialization**: Git workflow management, commit templates
- **Priority**: **LOW**
- **Recommendation**: **REFACTOR LATER** (Phase 3) or merge into generic devops droid

---

### bmad-builder
- **Size**: ~8KB
- **Redundancy**: ~65%
- **Specialization**: BMAD methodology builder (project-specific)
- **Priority**: **LOW-MEDIUM**
- **Recommendation**: **REFACTOR LATER** (Phase 3)

---

## Personal Droids (7 total, NOT analyzed)

**Context**: These are in `.factory/droids/` but marked as `(personal)`:
- ai-engineer
- backend-typescript-architect
- business-analyst
- database-optimizer
- senior-code-reviewer
- ui-engineer
- (others)

**Question**: Are these used for THIS trading project or generic?

**Recommendation**:
- **IF used for project**: Refactor in Phase 3
- **IF NOT used for project**: **LEAVE AS-IS** (don't prioritize)

---

## Priority Matrix

| Droid | Size | Redundancy | Priority | Recommendation | Phase |
|-------|------|------------|----------|----------------|-------|
| **orchestrator** | 20KB | 60% | CRITICAL | REFACTOR + MAESTRO | 1 |
| **onnx-model-builder** | 20KB | 65% | CRITICAL | REFACTOR NOW | 1 |
| crucible | 10KB | 70% | HIGH | REFACTOR LATER | 2 |
| argus | 15KB | 70% | HIGH | REFACTOR LATER | 2 |
| code-architect-reviewer | 12KB | 75% | MEDIUM | MERGE + REFACTOR | 2 |
| project-reader | 8KB | 65% | MEDIUM | REFACTOR LATER | 2 |
| trading-documenter | 10KB | 70% | MEDIUM | REFACTOR LATER | 2 |
| deep-researcher | 12KB | 70% | MEDIUM | EVALUATE MERGE | 3 |
| nautilus-nano | 4KB | N/A | N/A | KEEP AS-IS | N/A |
| sentinel-ftmo | 15KB | 70% | LOW | ARCHIVE or LATER | 3 |
| git-guardian | 6KB | 60% | LOW | LATER or MERGE | 3 |
| bmad-builder | 8KB | 65% | LOW-MEDIUM | LATER | 3 |
| personal droids | ~80KB | ~70% | TBD | IF USED: Phase 3 | 3 |

---

## Refactoring Roadmap

### Phase 1 (WITH TOP 5 - Week 1)
- **orchestrator** (MAESTRO role) ← CRITICAL
- **onnx-model-builder** (ML pipeline) ← CRITICAL

### Phase 2 (AFTER TOP 5 - Week 2)
- crucible
- argus
- code-architect-reviewer (merge with senior-code-reviewer)
- project-reader
- trading-documenter

### Phase 3 (Lower Priority - Week 3)
- deep-researcher (evaluate merge with argus + research-analyst-pro)
- sentinel-ftmo (if FTMO challenge planned)
- git-guardian (or merge into devops droid)
- bmad-builder
- personal droids (if used for project)

---

## Total Savings Estimate

**Before**:
- Remaining 12 (project): ~120KB
- Personal droids: ~80KB (if not used, ignore)
- **Total**: 120-200KB depending on personal usage

**After** (if all refactored):
- Remaining 12 (project): ~40KB (67% reduction)
- Personal droids: ~27KB (if refactored)
- **Savings**: 80-173KB

**Combined with TOP 5**:
- Before: 196KB (TOP 5) + 120KB (remaining 12) = **316KB**
- After: 61KB (TOP 5) + 40KB (remaining 12) = **101KB**
- **Total savings**: **215KB (68% reduction)**

---

## Key Insights

1. **Orchestrator is underutilized** - Has coordination logic but not automatic, should be MAESTRO
2. **ML pipeline incomplete** - onnx-model-builder doesn't hand off to ORACLE/FORGE automatically
3. **Research droids overlap** - 3 droids (ARGUS, research-analyst-pro, deep-researcher) do similar triangulation
4. **Code review droids redundant** - 3 droids (code-architect-reviewer, senior-code-reviewer, FORGE) review code
5. **FTMO droid unused** - Project uses Apex, not FTMO (sentinel-ftmo-guardian low priority)

---

**Next**: Execute LAYER 3 (Gap Analysis) to identify 5 critical droids MISSING from ecosystem.
