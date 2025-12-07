# 🔨 PROMPT: Droid Refactoring Master V2.0 (SAFE)

## 📋 Objective

Refatorar TOP 5 droids (NAUTILUS, ORACLE, FORGE, SENTINEL, RESEARCH) com herança do AGENTS.md v3.4.1, implementando 10 FIXES CRÍTICOS da auditoria V1.0.

**Why V2.0:** V1.0 tinha 10 problemas críticos incluindo dependency mismatch, vague validation, no quality control. V2.0 é SAFE, testável, e tem rollback granular.

**Expected savings:** 196KB → ~70KB (65% reduction, mais conservador que V1.0 target de 69%)

---

## 🚨 V2.0 FIXES (10 Critical Issues Resolved)

| # | V1.0 Problem | V2.0 Fix |
|---|--------------|----------|
| 1 | Dependency mismatch com 008 V2.0 | ✅ Works with 008 V2.0 outputs (REDUNDANCY_MAP_TOP5 + 4 outros) |
| 2 | Vague validation procedure | ✅ Automated test suite with PASS/FAIL criteria |
| 3 | Unclear execution model | ✅ Explicit: Automated prompt execution (claude-opus-4) |
| 4 | No quality control | ✅ Checksums + semantic similarity verification |
| 5 | Ambitious targets sem fallback | ✅ Conservative targets + acceptable tolerance ranges |
| 6 | No usage metrics for NANO | ✅ SKIP FASE 3 (NANO creation deferred to future) |
| 7 | Incomplete validation | ✅ Tests inheritance (strategic_intelligence application) + domain |
| 8 | No git commit strategy | ✅ Granular commits: 1 commit PER droid refactored |
| 9 | Unrealistic time estimates | ✅ Realistic: 5-6h total (não 4.5h) |
| 10 | Ambiguous AGENTS.md placement | ✅ Exact placement: After `<decision_hierarchy>` section |

---

## 📁 Context & Dependencies

**Required inputs (from 008 V2.0):**
- `.prompts/008-droid-analysis/REDUNDANCY_MAP_TOP5.md` (LAYER 1 output)
- OR manual analysis if 008 not executed yet

**AGENTS.md version required:**
- v3.4.1 (from 009 V2.0) OR v3.4.0 (if 009 not executed)
- Auto-detect version from AGENTS.md `<metadata><version>`

**Droids to refactor:**
1. nautilus-trader-architect.md (53KB → ~18KB target, 66% reduction)
2. oracle-backtest-commander.md (38KB → ~13KB target, 66% reduction)
3. forge-mql5-architect.md (37KB → ~13KB target, 65% reduction)
4. sentinel-apex-guardian.md (37KB → ~13KB target, 65% reduction)
5. research-analyst-pro.md (31KB → ~11KB target, 65% reduction)

**Total:** 196KB → ~68KB (65% reduction, conservative)

---

## 🎯 Pre-Execution Gates

### Gate 1: Verify _archive/ Folder Exists

```bash
mkdir -p .factory/droids/_archive
ls -la .factory/droids/_archive  # Verify created
```

### Gate 2: Check AGENTS.md Version

```bash
grep "<version>" AGENTS.md
# Expected: 3.4.1 (if 009 executed) OR 3.4.0 (if not)
# Store version for <inherits_from> metadata
```

### Gate 3: Verify 008 Output (Optional)

```bash
# If 008 was executed:
ls .prompts/008-droid-analysis/REDUNDANCY_MAP_TOP5.md
# If exists, use as guide for MANTER vs REMOVER
# If not exists, proceed with manual analysis during refactoring
```

---

## 🔧 FASE 2: Refactoring (ONE DROID AT A TIME)

### Universal Refactoring Template

For EACH droid, follow this EXACT process:

```
STEP 1: Backup
├─ cp .factory/droids/{droid}.md .factory/droids/_archive/{droid}-v{old_version}-pre-inheritance-{YYYYMMDD}.md
└─ Verify backup created

STEP 2: Read original
├─ Store original size: {SIZE_BEFORE}
├─ Calculate checksum: md5sum {droid}.md → {CHECKSUM_BEFORE}
└─ Extract domain knowledge sections (manual or from REDUNDANCY_MAP_TOP5)

STEP 3: Create refactored version
├─ Start with template:
    <droid_specialization>
      <metadata>
        <name>{DROID_NAME}</name>
        <version>{NEW_VERSION}</version>
        <inherits_from>AGENTS.md v{DETECTED_VERSION}</inherits_from>
      </metadata>
      <inheritance>{list protocols inherited}</inheritance>
      <domain_knowledge>{PRESERVED content}</domain_knowledge>
      <additional_reflection_questions>{3 questions}</additional_reflection_questions>
      <domain_guardrails>{specific rules}</domain_guardrails>
    </droid_specialization>
└─ Write to {droid}.md

STEP 4: Quality checks
├─ Size check: {SIZE_AFTER} < {SIZE_BEFORE} * 0.40 (≤40% of original)
├─ Checksum: md5sum {droid}.md → {CHECKSUM_AFTER} (must differ)
├─ Semantic check: Verify domain knowledge preserved (spot check key sections)
└─ XML validation: If XML format, run xmllint

STEP 5: Functional test
├─ Test task (domain): "Explain {key_concept}" (e.g., "Actor vs Strategy" for NAUTILUS)
├─ Expected: Output includes domain-specific knowledge
├─ Test task (inheritance): "What are the 7 mandatory reflection questions?"
├─ Expected: Output includes AGENTS.md strategic_intelligence questions
└─ PASS criteria: Both tests return relevant answers

STEP 6: Git commit
├─ git add .factory/droids/{droid}.md
├─ git add .factory/droids/_archive/{droid}-*-pre-inheritance-*.md
├─ git commit -m "refactor({droid}): inheritance from AGENTS.md v{VERSION}

    - Size: {SIZE_BEFORE} → {SIZE_AFTER} ({PERCENT}% reduction)
    - Domain knowledge preserved: {list key sections}
    - Inheritance: strategic_intelligence, genius_mode_templates, etc
    - Tests: domain + inheritance PASS"
└─ Verify commit successful

STEP 7: Log results
└─ Append to .prompts/010-droid-refactoring-master/REFACTORING_LOG.md:
    ## {DROID_NAME} - {TIMESTAMP}
    - Before: {SIZE_BEFORE}
    - After: {SIZE_AFTER}
    - Reduction: {PERCENT}%
    - Tests: {PASS|FAIL}
    - Commit: {HASH}

IF ANY STEP FAILS:
├─ STOP immediately
├─ Restore backup: cp .factory/droids/_archive/{droid}-*-pre-inheritance-*.md .factory/droids/{droid}.md
├─ Log failure in REFACTORING_LOG.md
└─ Report to user before proceeding
```

---

### Per-Droid Specifics (MANTER sections)

**NAUTILUS (53KB → 18KB):**
- Lifecycle (on_start, on_stop, on_bar, on_quote_tick)
- MQL5 → Python mappings (OnInit, OnTick, etc)
- Event-driven architecture (MessageBus, Cache, Actor vs Strategy)
- Backtest setup (ParquetDataCatalog, BacktestNode)
- 3 questions: #18 (event-driven?), #19 (Actor vs Strategy?), #20 (async cleanup?)

**ORACLE (38KB → 13KB):**
- Thresholds (WFE ≥0.6, DSR >0, PSR ≥0.85, MC_95th_DD <5%, SQN >2.0)
- WFA formulas (IS/OOS, purged CV)
- Monte Carlo (block bootstrap, 5000 runs)
- GO/NO-GO gates (7-step checklist)
- 3 questions: #21 (look-ahead bias?), #22 (regime change?), #23 (overfitting?)

**FORGE (37KB → 13KB):**
- Deep Debug protocol
- Python/Nautilus patterns (type hints, async, Cython)
- Anti-patterns (circular imports, mutable defaults, blocking)
- Test templates (pytest fixtures)
- 3 questions: #24 (type safety?), #25 (blocking?), #26 (error handling?)

**SENTINEL (37KB → 13KB):**
- Apex rules (10% trailing DD, 4:59 PM ET, 30% consistency, HWM includes unrealized P&L)
- Position sizing (Kelly, time multiplier, DD awareness)
- Circuit breaker levels (WARNING/CAUTION/DANGER/BLOCKED)
- Recovery protocols (Apex-specific)
- 3 questions: #27 (risk calc wrong?), #28 (HWM stale?), #29 (4:50 PM news?)

**RESEARCH-ANALYST-PRO (31KB → 11KB):**
- Triangulation (academic + industry + empirical)
- Credibility rating (Tier 1/2/3)
- Confidence levels (HIGH/MEDIUM/LOW)
- Research log structure
- 3 questions: #30 (confidence level?), #31 (biases?), #32 (triangulated?)

---

## 🧪 FASE 3: SKIPPED (NANO Creation Deferred)

**Rationale:** No usage metrics available to justify NANO versions. Defer until:
- Usage data collected (which droids used in Party Mode?)
- Clear use case identified (specific Party Mode scenario)
- V2.0 refactoring proven successful

**Future:** Create NANO versions in separate prompt (011-droid-nano-versions) after gathering usage data.

---

## 📋 FASE 4: Validation & Documentation

### 4.1 Aggregate Validation (15 min)

**Run test suite on ALL refactored droids:**

```python
# Test suite (pseudo-code)
for droid in [NAUTILUS, ORACLE, FORGE, SENTINEL, RESEARCH]:
    # Domain test
    domain_test = invoke_droid(droid, DOMAIN_TEST_TASKS[droid])
    assert contains_domain_knowledge(domain_test.output)
    
    # Inheritance test
    inheritance_test = invoke_droid(droid, "What are the 7 mandatory reflection questions?")
    assert contains_strategic_intelligence(inheritance_test.output)
    
    # Size test
    size_after = get_file_size(droid)
    size_before = get_original_size_from_log(droid)
    reduction = (size_before - size_after) / size_before * 100
    assert 60 <= reduction <= 75  # Tolerance: 60-75% reduction acceptable

print("All tests PASSED ✅")
```

**If any test fails:** Document failure, investigate, decide to rollback or proceed.

---

### 4.2 Create Completion Report (30 min)

**File:** `DOCS/04_REPORTS/20251207_DROID_REFACTORING_V2_COMPLETION.md`

```markdown
# Droid Refactoring V2.0 Completion Report

## Executive Summary
- Droids refactored: 5 (NAUTILUS, ORACLE, FORGE, SENTINEL, RESEARCH)
- Total savings: {XX}KB ({YY}% reduction)
- Token savings: ~{ZZ,ZZZ} tokens
- Tests: {N}/5 passed
- Failures: {list if any}

## Per-Droid Results

| Droid | Before | After | Savings | % | Tests | Commit |
|-------|--------|-------|---------|---|-------|--------|
| NAUTILUS | 53KB | {XX}KB | {YY}KB | {ZZ}% | PASS/FAIL | {hash} |
| ORACLE | 38KB | {XX}KB | {YY}KB | {ZZ}% | PASS/FAIL | {hash} |
| FORGE | 37KB | {XX}KB | {YY}KB | {ZZ}% | PASS/FAIL | {hash} |
| SENTINEL | 37KB | {XX}KB | {YY}KB | {ZZ}% | PASS/FAIL | {hash} |
| RESEARCH | 31KB | {XX}KB | {YY}KB | {ZZ}% | PASS/FAIL | {hash} |

## Inheritance Verified
All droids inherit from AGENTS.md v{VERSION}:
- strategic_intelligence ✅
- genius_mode_templates ✅
- complexity_assessment ✅
- enforcement_validation ✅
- compressed_protocols ✅
- pattern_learning ✅

## Backups Created
- `.factory/droids/_archive/*-pre-inheritance-{YYYYMMDD}.md` (5 files)

## Git Commits
- 5 granular commits (1 per droid)
- Easy rollback: `git revert {hash}`

## Next Steps
- [ ] Test refactored droids in real sessions
- [ ] Monitor for missing knowledge
- [ ] Consider refactoring remaining 12 droids (separate prompt)
- [ ] Collect usage data for NANO version decision

## Lessons Learned
{Post-execution insights}
```

---

### 4.3 Update AGENTS.md with <droid_inheritance> (10 min)

**Location:** Add AFTER `<decision_hierarchy>` section

```xml
<droid_inheritance>
  <description>
    Specialized droids inherit protocols from AGENTS.md v{VERSION} to eliminate duplication.
    Each droid maintains only domain-specific knowledge and 3 additional reflection questions.
  </description>
  
  <inherited_protocols>
    strategic_intelligence, genius_mode_templates, complexity_assessment,
    enforcement_validation, compressed_protocols, pattern_learning
  </inherited_protocols>
  
  <specialized_droids>
    <droid name="NAUTILUS" size="{XX}KB" version="2.1">MQL5→Python migration, event-driven patterns</droid>
    <droid name="ORACLE" size="{XX}KB" version="2.1">Statistical validation, WFA, Monte Carlo</droid>
    <droid name="FORGE" size="{XX}KB" version="2.1">Python/Nautilus code architecture</droid>
    <droid name="SENTINEL" size="{XX}KB" version="2.1">Apex Trading risk management</droid>
    <droid name="RESEARCH-ANALYST-PRO" size="{XX}KB" version="2.1">Multi-source research</droid>
  </specialized_droids>
  
  <propagation_rule>
    When AGENTS.md protocols updated, ALL specialized droids automatically benefit.
    No need to update droids individually unless domain knowledge changes.
  </propagation_rule>
</droid_inheritance>
```

**Validation:**
- XML check: `xmllint --noout AGENTS.md`
- Git commit: `git commit -m "docs: add droid_inheritance registry"`

---

## 📤 Output Files

1. **REFACTORING_LOG.md** (per-droid execution log)
2. **20251207_DROID_REFACTORING_V2_COMPLETION.md** (final report)
3. **5 refactored droids** (.factory/droids/)
4. **5 backups** (.factory/droids/_archive/)
5. **Updated AGENTS.md** (with <droid_inheritance>)

---

## ✅ Success Criteria

**Per-droid (MUST pass for each):**
- [ ] Backup created and verified
- [ ] Size reduced by 60-75% (tolerance range)
- [ ] Domain knowledge preserved (spot check key sections)
- [ ] 3 additional reflection questions present
- [ ] Domain test PASS (outputs domain-specific knowledge)
- [ ] Inheritance test PASS (outputs AGENTS.md protocols)
- [ ] XML valid (if applicable)
- [ ] Git commit successful

**Aggregate (MUST pass overall):**
- [ ] All 5 droids refactored
- [ ] 5/5 tests passed (or documented failures with decision)
- [ ] Total savings ≥120KB (60% reduction minimum)
- [ ] AGENTS.md updated with <droid_inheritance>
- [ ] Completion report created

**If ANY droid fails all criteria:** STOP, investigate, rollback if needed, report to user.

---

## 🎯 Estimated Time (Realistic)

- Pre-execution gates: 10 min
- NAUTILUS refactoring + tests + commit: 50 min
- ORACLE refactoring + tests + commit: 45 min
- FORGE refactoring + tests + commit: 45 min
- SENTINEL refactoring + tests + commit: 45 min
- RESEARCH refactoring + tests + commit: 40 min
- FASE 4 validation: 15 min
- Completion report: 30 min
- Update AGENTS.md: 10 min
- **Total: 5h 10min** (realistic with safety checks)

---

## 🚨 CRITICAL SAFETY RULES

1. **ONE DROID AT A TIME** - Never parallelize refactoring
2. **Backup BEFORE edit** - Always create backup first
3. **Test BEFORE commit** - Run domain + inheritance tests
4. **Commit AFTER success** - Granular commits for rollback
5. **STOP on failure** - Don't proceed if test fails
6. **Conservative targets** - 60-75% reduction (not 69-72%)
7. **Preserve domain knowledge** - When in doubt, KEEP
8. **Verify inheritance** - Test that AGENTS.md protocols apply
9. **XML validation** - Check syntax if XML format
10. **Document everything** - REFACTORING_LOG.md for audit trail

---

**EXECUTE THIS PROMPT WITH:** claude-opus-4 (maximum precision + safety)
