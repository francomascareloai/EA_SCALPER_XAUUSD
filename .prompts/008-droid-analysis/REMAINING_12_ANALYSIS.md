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

## Personal/Global Droids (18 total, DETAILED ANALYSIS)

**Context**: These are in `~/.factory/droids/` (personal/global scope, not project-specific)

**Total size estimated**: ~180-200KB (based on pattern analysis)

### Analysis by Category

#### 🤖 AI/ML Engineering (3 droids)
**ai-engineer** (~10KB)
- **Specialization**: LLM applications, RAG systems, prompt pipelines, vector search
- **Redundancy**: ~65% (generic AI patterns)
- **Project relevance**: **LOW** (project uses ONNX models, not LLM/RAG)
- **Recommendation**: **LEAVE AS-IS** (not used for trading project)

**mcp-testing-engineer** (~12KB)
- **Specialization**: MCP server testing, JSON schema validation, protocol compliance
- **Redundancy**: ~60%
- **Project relevance**: **MEDIUM** (project uses MCPs: Twelve-Data, memory, time, etc)
- **Recommendation**: **EVALUATE** - Could be useful for MCP validation

**prompt-optimizer** (~8KB)
- **Specialization**: Optimize prompts using 23 principles
- **Redundancy**: ~70%
- **Project relevance**: **LOW** (not prompt-heavy project)
- **Recommendation**: **LEAVE AS-IS**

---

#### 💻 Backend Engineering (3 droids)
**python-backend-engineer** (~15KB)
- **Specialization**: Python backend (FastAPI, Django, SQLAlchemy, uv tooling)
- **Redundancy**: ~70%
- **Project relevance**: **MEDIUM** (project is Python backend for Nautilus)
- **Overlap**: FORGE (also does Python) but less backend-focused
- **Recommendation**: **EVALUATE FOR MERGE** with FORGE or keep if backend-specific tasks needed

**backend-typescript-architect** (~15KB)
- **Specialization**: TypeScript backend with Bun runtime
- **Redundancy**: ~70%
- **Project relevance**: **NONE** (project is Python/MQL5, not TypeScript)
- **Recommendation**: **LEAVE AS-IS** (not used for trading project)

**network-engineer** (~10KB)
- **Specialization**: Network connectivity, DNS, SSL/TLS, CDN, load balancers
- **Redundancy**: ~65%
- **Project relevance**: **LOW** (trading system doesn't need network engineering)
- **Recommendation**: **LEAVE AS-IS**

---

#### 🗄️ Database (2 droids)
**database-optimizer** (~12KB)
- **Specialization**: SQL optimization, query tuning, indexing, migrations
- **Redundancy**: ~70%
- **Project relevance**: **NONE** (project uses Parquet files, not SQL databases)
- **Recommendation**: **LEAVE AS-IS**

**database-optimization** (~12KB)
- **Specialization**: Database performance tuning (appears to be duplicate?)
- **Redundancy**: ~70%
- **Project relevance**: **NONE**
- **Recommendation**: **REMOVE** (duplicate of database-optimizer?)

---

#### 👨‍💻 Code Review (1 droid)
**senior-code-reviewer** (~15KB)
- **Specialization**: Comprehensive code review (fullstack, security, performance)
- **Redundancy**: ~75%
- **Project relevance**: **MEDIUM** (general code review could help)
- **Overlap**: code-architect-reviewer (project droid)
- **Recommendation**: **MERGE** with code-architect-reviewer → generic-code-reviewer (already recommended in LAYER 4)

---

#### 🎨 Frontend (1 droid)
**ui-engineer** (~12KB)
- **Specialization**: Frontend UI components, React, responsive design
- **Redundancy**: ~70%
- **Project relevance**: **NONE** (trading system has no UI, only backend)
- **Recommendation**: **LEAVE AS-IS**

---

#### 📊 Business/Analytics (1 droid)
**business-analyst** (~10KB)
- **Specialization**: Metrics, KPIs, reports, dashboards, revenue models
- **Redundancy**: ~65%
- **Project relevance**: **LOW** (trading project doesn't need business analytics)
- **Recommendation**: **LEAVE AS-IS**

---

#### ☁️ Cloud/DevOps (2 droids)
**cloud-architect** (~15KB)
- **Specialization**: AWS/Azure/GCP infrastructure, Terraform, auto-scaling
- **Redundancy**: ~70%
- **Project relevance**: **LOW** (trading system runs locally or on VPS, not cloud)
- **Recommendation**: **LEAVE AS-IS**

**command-expert** (~8KB)
- **Specialization**: CLI commands, automation, tooling
- **Redundancy**: ~60%
- **Project relevance**: **LOW**
- **Recommendation**: **LEAVE AS-IS**

---

#### 🛠️ Utilities (5 droids)
**markdown-syntax-formatter** (~6KB)
- **Specialization**: Fix markdown formatting, convert text to markdown
- **Redundancy**: ~55%
- **Project relevance**: **LOW** (documentation formatting, not critical)
- **Recommendation**: **LEAVE AS-IS**

**subagent-auditor** (~8KB)
- **Specialization**: Audit subagent configurations
- **Redundancy**: ~60%
- **Project relevance**: **MEDIUM** (project has many droids/subagents)
- **Recommendation**: **EVALUATE** - Could help with droid quality

**slash-command-auditor** (~7KB)
- **Specialization**: Audit slash commands
- **Redundancy**: ~60%
- **Project relevance**: **LOW**
- **Recommendation**: **LEAVE AS-IS**

**skill-auditor** (~8KB)
- **Specialization**: Audit skill configurations
- **Redundancy**: ~60%
- **Project relevance**: **MEDIUM** (project has 6 skills)
- **Recommendation**: **EVALUATE** - Could help with skill quality

**jailbreak** (~5KB)
- **Specialization**: (Purpose unclear, possibly testing/security)
- **Redundancy**: ~50%
- **Project relevance**: **UNKNOWN**
- **Recommendation**: **REVIEW** - Unclear purpose, may be deprecated

---

### Summary: Personal Droids

| Category | Droids | Total Size | Project Relevance | Recommendation |
|----------|--------|------------|-------------------|----------------|
| AI/ML | 3 | ~30KB | LOW-MEDIUM | Evaluate mcp-testing, leave others |
| Backend | 3 | ~40KB | MEDIUM | Evaluate python-backend-engineer |
| Database | 2 | ~24KB | NONE | Leave as-is (not used) |
| Code Review | 1 | ~15KB | MEDIUM | **MERGE** with code-architect-reviewer |
| Frontend | 1 | ~12KB | NONE | Leave as-is (no UI) |
| Business | 1 | ~10KB | LOW | Leave as-is |
| Cloud/DevOps | 2 | ~23KB | LOW | Leave as-is |
| Utilities | 5 | ~34KB | LOW-MEDIUM | Evaluate auditors, leave others |
| **TOTAL** | **18** | **~188KB** | **Mostly LOW** | **Selective refactoring** |

### Project-Relevant Personal Droids (Potential Refactoring)

**MEDIUM Priority (Consider for Phase 3)**:
1. **senior-code-reviewer** → Already recommended for merge with code-architect-reviewer
2. **python-backend-engineer** → Evaluate merge with FORGE or keep if backend-specific needed
3. **mcp-testing-engineer** → Could validate Twelve-Data MCP, memory MCP, time MCP
4. **subagent-auditor** → Could audit droid configurations (meta-analysis)
5. **skill-auditor** → Could audit 6 skills (argus, crucible, forge, oracle, sentinel, nautilus)

**Estimated savings IF refactored**:
- 5 project-relevant droids: ~60KB current → ~20KB after (40KB saved)
- Remaining 13 non-relevant: Leave as-is (no savings, no effort wasted)

### Updated Recommendation

**Personal droids strategy**:
- **DO NOT refactor all 18** (13 are not project-relevant, waste of effort)
- **Selectively refactor 5 project-relevant** in Phase 3 (40KB savings)
- **Total ecosystem** with personal droids considered:
  - Before: 389KB (project) + 188KB (personal) = **577KB total**
  - After: 101KB (project) + 148KB (personal, 5 refactored) = **249KB total**
  - **Savings: 328KB (57% reduction)** across full ecosystem

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

### Project Droids Only (Original Scope)
**Before**:
- TOP 5: 196KB
- Remaining 12 (project): ~120KB
- **Total project**: 316KB

**After** (all project droids refactored):
- TOP 5: 61KB (135KB saved)
- Remaining 12: 40KB (80KB saved)
- **Total project**: 101KB
- **Savings**: **215KB (68% reduction)**

---

### Full Ecosystem (Project + Personal Droids)
**Before**:
- Project droids (17): 389KB
- Personal droids (18): ~188KB
- **Total ecosystem**: **577KB**

**After** (selective refactoring):
- Project droids (17 refactored): 101KB
- Personal droids (5 project-relevant refactored, 13 left as-is):
  - 5 refactored: ~20KB (from ~60KB, 40KB saved)
  - 13 not refactored: ~128KB (unchanged)
  - Subtotal: 148KB
- **Total ecosystem**: **249KB**
- **Savings**: **328KB (57% reduction)**

---

### Token Impact
**Project droids only**:
- Before: 97,250 tokens
- After: 25,250 tokens
- **Savings**: 72,000 tokens (74%)

**Full ecosystem**:
- Before: 144,250 tokens (577KB / 4)
- After: 62,250 tokens (249KB / 4)
- **Savings**: 82,000 tokens (57%)

**Party Mode impact**:
- Before overhead: 144,250 tokens (full ecosystem loaded)
- After overhead: 62,250 tokens
- **Freed budget**: +82,000 tokens (57% improvement)

---

## Key Insights

1. **Orchestrator is underutilized** - Has coordination logic but not automatic, should be MAESTRO
2. **ML pipeline incomplete** - onnx-model-builder doesn't hand off to ORACLE/FORGE automatically
3. **Research droids overlap** - 3 droids (ARGUS, research-analyst-pro, deep-researcher) do similar triangulation
4. **Code review droids redundant** - 3 droids (code-architect-reviewer, senior-code-reviewer, FORGE) review code
5. **FTMO droid unused** - Project uses Apex, not FTMO (sentinel-ftmo-guardian low priority)

---

**Next**: Execute LAYER 3 (Gap Analysis) to identify 5 critical droids MISSING from ecosystem.
