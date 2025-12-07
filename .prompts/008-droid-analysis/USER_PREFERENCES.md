# User Preferences for Droid Refactoring

## Date: 2025-12-07

### Droids to EXCLUDE from Refactoring

**DO NOT touch these droids** (per user request):
1. **subagent-auditor** - Leave AS-IS
2. **skill-auditor** - Leave AS-IS

**Rationale**: These auditors are working well and should not be modified.

---

### Updated Project-Relevant Personal Droids (Phase 3)

**REFACTOR these 3 droids**:
1. **senior-code-reviewer** (~15KB → ~5KB)
   - Action: MERGE with code-architect-reviewer → generic-code-reviewer
   - Savings: ~10KB

2. **python-backend-engineer** (~15KB → ~5KB)  
   - Action: EVALUATE merge with FORGE or refactor standalone
   - Savings: ~10KB

3. **mcp-testing-engineer** (~8KB → ~3KB)
   - Action: Refactor for MCP validation (Twelve-Data, memory, time MCPs)
   - Savings: ~5KB

**Total savings** (3 droids): ~25KB (down from 40KB with auditors)

---

### Updated Ecosystem Totals

**Full Ecosystem** (35 droids):
- Before: 577KB (389 project + 188 personal)
- After: 264KB (101 project + 163 personal)
  - Project: 101KB (all 17 refactored)
  - Personal: 163KB (3 refactored, 2 auditors untouched, 13 non-relevant untouched)
- **Savings**: 313KB (54% reduction)
- **Token savings**: ~78,000 tokens

**Strategy**:
- ✅ Refactor all 17 project droids (Phase 1-2)
- ✅ Refactor 3 project-relevant personal droids (Phase 3)
- ❌ **DO NOT touch** 2 auditors (per user)
- ❌ Leave 13 non-relevant personal droids AS-IS (no effort wasted)

---

## Next Actions (Prioritized)

### Phase 1 (Week 1) - CRITICAL
1. CREATE security-compliance droid (CRITICAL)
2. CREATE performance-optimizer droid (HIGH)
3. Refactor TOP 5 (NAUTILUS, ORACLE, FORGE, SENTINEL, RESEARCH)
4. Refactor orchestrator → MAESTRO
5. Refactor onnx-model-builder

### Phase 2 (Week 2) - HIGH
1. Refactor remaining 12 project droids
2. MERGE code-architect-reviewer + senior-code-reviewer → generic-code-reviewer
3. CREATE data-pipeline, monitoring-alerting droids
4. Update AGENTS.md (decision_hierarchy, DAG, quality_monitoring)

### Phase 3 (Week 3) - SELECTIVE PERSONAL
1. **EVALUATE** python-backend-engineer (merge with FORGE or standalone)
2. Refactor mcp-testing-engineer
3. ~~subagent-auditor~~ → **SKIP** (user request)
4. ~~skill-auditor~~ → **SKIP** (user request)
5. Remaining project droids (crucible, argus, deep-researcher merge)
6. REMOVE database-optimization (duplicate)
