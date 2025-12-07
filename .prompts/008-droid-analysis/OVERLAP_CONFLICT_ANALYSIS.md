# Overlap & Conflict Analysis (LAYER 4)

## Executive Summary
- **Overlaps identified**: 4 major areas (code architecture, code review, research, backtest)
- **Conflicts possible**: Multiple scenarios where droids could disagree
- **Resolution framework**: Authority hierarchy (7 levels) + conflict resolution protocol
- **Recommendation**: Update AGENTS.md with expanded `<decision_hierarchy>` and `<conflict_resolution>` sections

---

## Overlap 1: Code Architecture (FORGE vs NAUTILUS) ⚙️

### Droids Involved
- **FORGE** (Python code implementation, debugging, testing)
- **NAUTILUS** (NautilusTrader architecture, migration, high-level design)

### Overlap Area
Both work with NautilusTrader Python code, causing confusion: "Which droid do I invoke for Python task?"

### Confusion Factor
**HIGH** - Users don't know whether to call FORGE or NAUTILUS for Python work

### Current State
- FORGE: Low-level Python implementation, anti-patterns, debugging, testing
- NAUTILUS: High-level architecture (Strategy vs Actor decision), MQL5 → Python migration, backtest setup

### Resolution
**CLEAR SEPARATION**:
- **NAUTILUS**: HIGH-LEVEL
  - Strategy vs Actor vs Indicator decision tree
  - MQL5 → Nautilus migration mappings
  - Backtest configuration (ParquetDataCatalog, BacktestNode)
  - Event-driven architecture patterns
  
- **FORGE**: LOW-LEVEL
  - Python implementation details
  - Debugging (Deep Debug protocol)
  - Testing (pytest fixtures)
  - Code quality (type hints, anti-patterns)

**HANDOFF PROTOCOL**:
```
User: "I need Actor for divergence detection"
  ↓
NAUTILUS: Designs Actor (what it does, pub/sub pattern)
  ↓
FORGE: Implements Actor (Python code, tests, anti-pattern checks)
  ↓
NAUTILUS: Validates backtest with new Actor
  ↓
ORACLE: Validates statistical performance
```

### Document In
`<handoffs>` section of AGENTS.md (or ORCHESTRATOR.md)

---

## Overlap 2: Code Review (FORGE vs code-architect-reviewer vs senior-code-reviewer) 👀

### Droids Involved
- **FORGE** (/review trigger for Python/Nautilus code)
- **code-architect-reviewer** (architecture/design review)
- **senior-code-reviewer** (personal droid, general best practices)

### Overlap Area
**ALL THREE do code review** - causing confusion: "Which reviewer do I invoke?"

### Confusion Factor
**HIGH** - Three different code review droids with unclear specialization

### Current State
- FORGE: Python/Nautilus-specific patterns, anti-patterns (AP-01 through AP-12)
- code-architect-reviewer: High-level architecture, system design patterns
- senior-code-reviewer: General best practices, security, performance (generic)

### Resolution
**MERGE + SPECIALIZE**:

1. **MERGE**: code-architect-reviewer + senior-code-reviewer → **generic-code-reviewer**
   - Rationale: Both do general code review, high overlap
   - New droid handles: Architecture, design, general best practices, security

2. **KEEP**: FORGE (Nautilus-specific)
   - Domain-specific patterns (NautilusTrader event-driven architecture)
   - Python/Nautilus anti-patterns
   - Integration with NAUTILUS for architecture validation

**DECISION MATRIX**:

| Code Type | Use Droid | Rationale |
|-----------|-----------|-----------|
| Python trading code (Nautilus) | **FORGE** | Domain-specific (event-driven, Actor/Strategy patterns) |
| TypeScript backend | **generic-code-reviewer** | General best practices |
| React frontend | **generic-code-reviewer** + ui-engineer | UI-specific + general |
| High-level architecture | **generic-code-reviewer** | System design focus |
| Performance-critical code | **FORGE** + **performance-optimizer** | Domain + optimization |

### Document In
Decision matrix in AGENTS.md or ORCHESTRATOR routing table

---

## Overlap 3: Research (ARGUS vs research-analyst-pro vs deep-researcher) 🔬

### Droids Involved
- **ARGUS** (quant/trading-specific research, arXiv/SSRN papers)
- **research-analyst-pro** (general multi-source research, triangulation)
- **deep-researcher** (comprehensive multi-layer research, deep dive)

### Overlap Area
**ALL THREE do multi-source research with triangulation** - high overlap

### Confusion Factor
**MEDIUM** - Similar methodology but different scope/depth

### Current State
- ARGUS: Trading/quant-specific, financial papers, backtest validation
- research-analyst-pro: General multi-source, confidence levels, source credibility
- deep-researcher: Most comprehensive, academic + industry + empirical triangulation

### Resolution Options

**Option A: KEEP ALL THREE** (status quo)
- ARGUS: Trading/quant-specific only
- research-analyst-pro: General non-trading research
- deep-researcher: Complex topics requiring deep analysis
- **Decision rule**: Topic determines which to invoke

**Option B: MERGE INTO ONE** (recommended)
- Single research droid with **domain parameter**:
  ```
  research-master.md
    ├─ domain=trading → Use ARGUS methodology
    ├─ domain=general → Use research-analyst-pro methodology
    └─ depth=deep → Use deep-researcher methodology
  ```
- **Pros**: Single source of truth, no confusion
- **Cons**: Larger droid (but can use inheritance to reduce size)

**Option C: KEEP ARGUS, MERGE OTHER TWO**
- ARGUS: Trading-specific (keep separate)
- research-analyst-pro + deep-researcher → research-general.md
- **Pros**: Clear trading vs general separation
- **Cons**: Still have two research droids

### Recommendation
**Option C** (pragmatic):
- ARGUS: Keep separate (trading-specific is valuable specialization)
- Merge research-analyst-pro + deep-researcher → research-general.md

### Document In
Routing rules in ORCHESTRATOR (when to invoke which research droid)

---

## Overlap 4: Backtest (ORACLE vs CRUCIBLE vs NAUTILUS) 📊

### Droids Involved
- **ORACLE** (statistical validation: WFA, Monte Carlo, GO/NO-GO)
- **CRUCIBLE** (backtest realism: slippage, spread, fills, Apex constraints)
- **NAUTILUS** (backtest setup: ParquetDataCatalog, BacktestNode, configs)

### Overlap Area
**All three touch backtesting** - but at different stages

### Confusion Factor
**MEDIUM** - Overlap exists but roles are distinct (less confusing than other overlaps)

### Current State
- NAUTILUS: Backtest SETUP (configuration, data loading, engine setup)
- CRUCIBLE: Backtest REALISM (25 realism gates, execution modeling)
- ORACLE: Backtest VALIDATION (statistical significance, WFA, Monte Carlo)

### Resolution
**CLEAR PIPELINE** (no merge needed, just document handoffs):

```
User: "Backtest new strategy"
  ↓
NAUTILUS: Sets up backtest
  - Loads Parquet data
  - Configures BacktestNode (VenueConfig, DataConfig)
  - Configures FillModel, LatencyModel
  - Runs backtest → produces results.json
  ↓
CRUCIBLE: Validates realism
  - Checks 25 realism gates (slippage, spread, fills realistic?)
  - Verifies Apex constraints enforced (no overnight, DD tracking)
  - If unrealistic → BLOCK, request fixes
  ↓
ORACLE: Validates statistics
  - WFA (WFE ≥ 0.6?)
  - Monte Carlo (95th percentile DD < 5%?)
  - GO/NO-GO decision
  ↓
If NO-GO:
  FORGE: Implements fixes based on ORACLE/CRUCIBLE feedback
  → Loop back to NAUTILUS for re-test
```

**AUTHORITY**:
- **CRUCIBLE** can BLOCK if backtest is unrealistic (realism authority)
- **ORACLE** has final GO/NO-GO authority (validation authority)
- **NAUTILUS** executes setup but doesn't make GO/NO-GO decisions

### Document In
`<dependency_graph>` in AGENTS.md (backtest workflow)

---

## Conflict Resolution Framework

### Authority Hierarchy (7 Levels)

**Expand AGENTS.md `<decision_hierarchy>` from 3 levels → 7 levels**:

```xml
<decision_hierarchy>
  <description>
    When droids disagree, this hierarchy determines who has final authority.
    Lower priority number = higher authority (priority 1 beats priority 2).
  </description>
  
  <level priority="1" domain="risk_management">
    <droid>SENTINEL</droid>
    <authority>VETO on any trade if DD >9%, time <30min to 4:59 PM, or consistency >30%</authority>
    <cannot_override>No other droid can override SENTINEL veto</cannot_override>
    <example>CRUCIBLE says "trade this setup (9/10)" → SENTINEL blocks (DD 8.9%) → SENTINEL WINS</example>
  </level>
  
  <level priority="2" domain="validation">
    <droid>ORACLE</droid>
    <authority>GO/NO-GO decision on backtests (NO-GO if WFE <0.6 or DSR ≤0)</authority>
    <cannot_override>SENTINEL can veto (priority 1) but no one else</cannot_override>
    <example>CRUCIBLE says "backtest looks good" → ORACLE says NO-GO (WFE 0.52) → ORACLE WINS</example>
  </level>
  
  <level priority="3" domain="realism">
    <droid>CRUCIBLE</droid>
    <authority>Backtest realism validation (25 gates, can BLOCK if unrealistic)</authority>
    <can_be_overridden_by>SENTINEL (risk), ORACLE (statistical validation)</can_be_overridden_by>
    <example>NAUTILUS runs backtest → CRUCIBLE blocks (unrealistic slippage) → BLOCK WINS</example>
  </level>
  
  <level priority="4" domain="architecture">
    <droid>NAUTILUS</droid>
    <authority>High-level architecture decisions (Strategy vs Actor, event-driven patterns)</authority>
    <can_be_overridden_by>FORGE (if implementation concerns), SENTINEL (if performance risk)</can_be_overridden_by>
    <example>NAUTILUS chooses Actor pattern → FORGE implements Actor → NO CONFLICT</example>
  </level>
  
  <level priority="5" domain="implementation">
    <droid>FORGE</droid>
    <authority>Low-level implementation, code quality, testing, anti-patterns</authority>
    <can_be_overridden_by>NAUTILUS (if architecture conflict), SENTINEL (if performance risk)</can_be_overridden_by>
    <example>FORGE suggests inline code for speed → NAUTILUS overrides (architecture) → NAUTILUS WINS</example>
  </level>
  
  <level priority="6" domain="research">
    <droid>ARGUS</droid>
    <authority>Research findings, source credibility, academic validation</authority>
    <can_be_overridden_by>ORACLE (if contradicts backtest), SENTINEL (if risk concern)</can_be_overridden_by>
    <example>ARGUS research says "RSI divergence 70% accurate" → ORACLE backtests 52% → ORACLE WINS</example>
  </level>
  
  <level priority="7" domain="orchestration">
    <droid>ORCHESTRATOR</droid>
    <authority>Workflow coordination, droid invocation order, routing</authority>
    <note>Coordinates but doesn't override domain authorities</note>
    <example>ORCHESTRATOR routes "review code" to FORGE (not generic-reviewer) → ROUTING DECISION ONLY</example>
  </level>
</decision_hierarchy>
```

### Conflict Resolution Protocol

**Add to AGENTS.md `<conflict_resolution>` section**:

```xml
<conflict_resolution>
  <protocol name="Droid Disagreement">
    <trigger>Two or more droids provide contradictory recommendations</trigger>
    
    <steps>
      <step number="1">Identify which domain the conflict is in (risk, validation, realism, architecture, implementation, research, orchestration)</step>
      <step number="2">Check decision_hierarchy for domain authority</step>
      <step number="3">Domain authority droid has final say</step>
      <step number="4">If still unclear (cross-domain conflict), escalate to ORCHESTRATOR for mediation</step>
      <step number="5">If ORCHESTRATOR can't resolve, escalate to USER with steel-man analysis of both options</step>
    </steps>
    
    <examples>
      <example name="Risk vs Strategy">
        <scenario>CRUCIBLE recommends trade (confluence 9/10) but SENTINEL blocks (DD 8.9%)</scenario>
        <step1>Domain = risk_management</step1>
        <step2>SENTINEL has priority 1 (highest authority)</step2>
        <step3>SENTINEL veto WINS</step3>
        <outcome>Trade blocked, no escalation needed</outcome>
      </example>
      
      <example name="Validation vs Realism">
        <scenario>ORACLE says GO (WFE 0.72) but CRUCIBLE blocks (unrealistic slippage)</scenario>
        <step1>Domain conflict: validation (priority 2) vs realism (priority 3)</step1>
        <step2>ORACLE higher authority BUT CRUCIBLE realism block is pre-validation gate</step2>
        <step3>CRUCIBLE must pass BEFORE ORACLE validation</step3>
        <outcome>Fix realism issues first, then re-validate with ORACLE</outcome>
        <note>Sequential gates: CRUCIBLE → ORACLE, not parallel</note>
      </example>
      
      <example name="Architecture vs Implementation">
        <scenario>NAUTILUS chooses Actor pattern, FORGE suggests inline for performance</scenario>
        <step1>Domain = architecture (priority 4) vs implementation (priority 5)</step1>
        <step2>NAUTILUS higher authority</step2>
        <step3>NAUTILUS decision WINS (Actor pattern)</step3>
        <outcome>FORGE implements Actor pattern (follows NAUTILUS architecture)</outcome>
      </example>
      
      <example name="Cross-Domain Conflict">
        <scenario>ARGUS research says "use LSTM" but performance-optimizer says "too slow (>50ms)"</scenario>
        <step1>Domain conflict: research (priority 6) vs performance constraint (hard limit)</step1>
        <step2>Not in decision_hierarchy → cross-domain issue</step2>
        <step3>Escalate to ORCHESTRATOR</step3>
        <step4>ORCHESTRATOR mediates: "Use xLSTM (faster variant) or simplify architecture"</step4>
        <outcome>Compromise solution that satisfies both constraints</outcome>
      </example>
    </examples>
    
    <special_cases>
      <case name="Sequential Gates">
        <description>Some droids are gates that must pass BEFORE others can evaluate</description>
        <example>CRUCIBLE (realism) → ORACLE (validation) → SENTINEL (risk) → DEPLOY</example>
        <rule>Gate failures BLOCK pipeline, not conflict with downstream droids</rule>
      </case>
      
      <case name="Performance Constraints">
        <description>Hard performance limits (OnTick <50ms) override design preferences</description>
        <rule>If performance-optimizer says "too slow", ANY droid must adapt</rule>
      </case>
      
      <case name="Security/Compliance">
        <description>Security-compliance droid has implicit veto on unsafe/non-compliant code</description>
        <rule>Security issues BLOCK deployment, similar authority to SENTINEL for risk</rule>
      </case>
    </special_cases>
  </protocol>
</conflict_resolution>
```

---

## Decision Matrix

| Conflict Scenario | Droid A (Recommendation) | Droid B (Recommendation) | Winner | Reason |
|-------------------|--------------------------|--------------------------|--------|--------|
| Trade decision | CRUCIBLE (GO - 9/10 setup) | SENTINEL (BLOCK - DD 8.9%) | **SENTINEL** | Priority 1 > Priority 3 |
| Backtest validation | ORACLE (NO-GO - WFE 0.52) | CRUCIBLE (LOOKS GOOD - realism OK) | **ORACLE** | Priority 2 (GO/NO-GO authority) |
| Backtest realism | CRUCIBLE (BLOCK - unrealistic) | NAUTILUS (SETUP DONE) | **CRUCIBLE** | Realism gate must pass first |
| Architecture | NAUTILUS (Actor pattern) | FORGE (inline for speed) | **NAUTILUS** | Priority 4 > Priority 5 |
| Research vs backtest | ARGUS (70% accuracy) | ORACLE (52% in backtest) | **ORACLE** | Priority 2 > Priority 6 (empirical > research) |
| Code review | FORGE (Nautilus-specific) | generic-code-reviewer (general) | **FORGE** | Domain-specific beats generic |
| Performance constraint | ARGUS (use LSTM) | performance-optimizer (too slow) | **COMPROMISE** | Hard constraint (escalate to ORCHESTRATOR) |

---

## Recommendations

### 1. Update AGENTS.md
- Expand `<decision_hierarchy>` to 7 levels (add CRUCIBLE, NAUTILUS, FORGE, ARGUS, ORCHESTRATOR)
- Add comprehensive `<conflict_resolution>` protocol with examples
- Add special cases (sequential gates, performance constraints, security vetos)

### 2. Document Handoffs
- **NAUTILUS → FORGE**: Design → Implement
- **NAUTILUS → ORACLE**: Backtest setup → Validation
- **CRUCIBLE → ORACLE**: Realism check → Statistical validation
- **ORACLE → FORGE**: Validation feedback → Fixes
- **ARGUS → onnx-model-builder**: Research → ML implementation

### 3. Merge Overlapping Droids
- **code-architect-reviewer** + **senior-code-reviewer** → **generic-code-reviewer**
- **research-analyst-pro** + **deep-researcher** → **research-general**
- Keep **FORGE** (Nautilus-specific), **ARGUS** (trading-specific) as specialists

### 4. Add Decision Matrix to ORCHESTRATOR
- When routing user request, check for potential conflicts
- Apply decision matrix proactively
- "Review Python trading code" → FORGE (not generic-code-reviewer)

---

## Key Insights

1. **Overlaps are common** - 4 major areas identified (code architecture, code review, research, backtest)
2. **Authority hierarchy resolves most conflicts** - Clear priority levels prevent circular disagreements
3. **Sequential gates exist** - Some droids are validation gates (CRUCIBLE → ORACLE → SENTINEL)
4. **Specialization matters** - Domain-specific droids (FORGE, ARGUS) beat generic ones
5. **ORCHESTRATOR needs upgrade** - Must understand conflicts and mediate

---

**Next**: Execute LAYER 5 (Ecosystem Health Framework) to establish versioning, quality gates, observability, and dependency graph.
