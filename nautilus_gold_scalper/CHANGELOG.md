# Nautilus Gold Scalper - Code Change Log

**Purpose:** Log COMPLETED work units (features, improvements, breaking changes, config)  
**Owner:** FORGE, NAUTILUS  
**Format:** Chronological (newest first)  
**When:** ONLY when work unit fully complete (all edits done, tests passing). NOT during individual edits.  
**Usage:** Understand what changed, why, and impact. Prevent getting lost in codebase evolution.

---

## Template (copy for new entries)

```markdown
## [Module] - YYYY-MM-DD HH:MM (AGENT)

### 🐛 BUGFIX | 🚀 IMPROVEMENT | ✨ FEATURE | ⚠️ BREAKING | ⚙️ CONFIG

**What:** Brief description (1 line)  
**Why:** Problem solved / motivation / context  
**Impact:** What changed (behavior, API, performance, dependencies)  
**Files:**
- path/to/file1.py
- path/to/file2.py

**Validation:** Tests passed, compilation status, quality gates  
**Commit:** [hash if committed]
```

---

## 2025-12-08 18:15 (FORGE)

### ✨ FEATURE

**What:** Created FUTURE_IMPROVEMENTS.md brainstorming repository for optimization ideas  
**Why:** Franco requested systematic way to capture ideas for future enhancements without losing context - "base de ideias" for when he wants to improve  
**Impact:** Organized repository for all agents to add optimization ideas (research findings, backtest insights, bottlenecks). Includes WHY/WHAT/IMPACT/EFFORT for each idea. Priority matrix (P1-P4) helps decide what to implement when bandwidth available.  
**Files:**
- nautilus_gold_scalper/FUTURE_IMPROVEMENTS.md (created - brainstorming base)
- AGENTS.md (updated future_improvements_tracking section with triggers + format)

**Validation:** Template complete with 15+ example ideas spanning strategy, risk, ML, architecture. Update protocol defined.  
**Commit:** pending

---

## 2025-12-08 18:00 (FORGE)

### ⚙️ CONFIG

**What:** Created CHANGELOG.md tracking system for COMPLETED work units + BUGFIX_LOG.md for discovered bugs  
**Why:** Franco requested systematic logging to prevent losing context, but ONLY when work complete (not individual edits)  
**Impact:** FORGE/NAUTILUS logs when work unit DONE (e.g., 10 edits = 1 log entry), bugs logged immediately when discovered  
**Files:**
- nautilus_gold_scalper/CHANGELOG.md (this file)
- nautilus_gold_scalper/BUGFIX_LOG.md (created)
- MQL5/Experts/CHANGELOG.md (created)
- MQL5/Experts/BUGFIX_LOG.md (created)
- AGENTS.md (updated code_change_tracking + git_workflow + forge_rule)

**Validation:** Documentation complete, enforcement rules added to AGENTS.md, philosophy: completion-based, NOT edit-based  
**Commit:** pending
