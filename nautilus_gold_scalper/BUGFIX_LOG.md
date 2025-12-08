# Nautilus Gold Scalper - Bug Fix Log

**Purpose:** Track bugs and fixes with ROOT CAUSE analysis to prevent recurrence  
**Owner:** FORGE, NAUTILUS  
**Format:** Structured Markdown (newest first)  
**Usage:** Debugging, pattern recognition, post-mortem analysis

**CRITICAL bugs (account risk, Apex violations):** MUST include 5 Whys + Prevention (AGENTS.md updates)

---

## Template for Standard Bugs

```markdown
## YYYY-MM-DD HH:MM [AGENT] - Module

**Bug:** Brief description  
**Impact:** What broke / consequences  
**Root Cause:** Why it happened (1-2 sentences)  
**Fix:** Solution applied  
**Files:** List of modified files  
**Validation:** Tests added/passed  
**Commit:** hash
```

---

## Template for CRITICAL Bugs (🚨 Account Risk / Apex Violations)

```markdown
## 🚨 YYYY-MM-DD HH:MM [AGENT] - CRITICAL

**Module:** src/path/to/module.py  
**Severity:** CRITICAL (Account survival - $50k risk) | HIGH (Trading logic) | MEDIUM  
**Bug:** Brief description  
**Impact:** Specific consequences (would violate Apex? lose money?)  

**Root Cause (5 Whys):**
1. Why? [First level]
2. Why? [Deeper]
3. Why? [Process issue]
4. Why? [Missing validation]
5. Why? [Root cause]

**Fix:** Solution applied  

**Prevention (MANDATORY - Protocol Updates):**
- ✅ Updated AGENTS.md: [which section, what added]
- ✅ Added test: [coverage added]
- ✅ Added automation: [pre-commit hook, CI check]
- ✅ Updated complexity: [if escalation needed]

**Files:**
- path/to/file1.py (fixed)
- path/to/file2.py (test)
- AGENTS.md (protocol update)

**Validation:** [proof fix works]  
**Commit:** hash
```

---

## Log Entries

### 2025-12-08 18:00 [FORGE] - BUGFIX_LOG.md

**Bug:** No structured bug tracking system  
**Impact:** Bugs not analyzed for root cause, patterns not learned  
**Root Cause:** Missing systematic logging protocol with prevention enforcement  
**Fix:** Created BUGFIX_LOG.md with mandatory Root Cause + Prevention for CRITICAL bugs  
**Files:** BUGFIX_LOG.md  
**Validation:** Template complete with 🚨 CRITICAL protocol  
**Commit:** pending
