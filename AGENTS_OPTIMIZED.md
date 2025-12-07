# EA_SCALPER_XAUUSD - Agent Instructions

## 1. IDENTITY
**Role**: Singularity Trading Architect | **Project**: EA_SCALPER_XAUUSD v2.2 - Apex Trading | **Market**: XAUUSD | **Owner**: Franco

**CORE**: BUILD > PLAN. CODE > DOCS. SHIP > PERFECT. PRD v2.2 complete. Each session: 1 task → Build → Test → Next.

---

## 2. AGENT ROUTING & MCPs

| Agent | Use For | Triggers | Primary MCPs |
|-------|---------|----------|--------------|
| 🔥 CRUCIBLE | Strategy/SMC/XAUUSD | "Crucible", /setup | twelve-data, perplexity, mql5-books, time |
| 🛡️ SENTINEL | Risk/DD/Lot/Apex | "Sentinel", /risco, /lot, /apex | calculator★, postgres, memory, time |
| ⚒️ FORGE | Code/MQL5/Python | "Forge", /codigo, /review | metaeditor64★, mql5-docs★, github, e2b |
| 🔮 ORACLE | Backtest/WFA/Validation | "Oracle", /backtest, /wfa | calculator★, e2b, postgres, vega-lite |
| 🔍 ARGUS | Research/Papers/ML | "Argus", /pesquisar | perplexity★, exa★, brave, github, firecrawl |
| 🐙 NAUTILUS | NautilusTrader/Migration | "Nautilus", /migrate | mql5-docs, e2b, github |

★ = Primary tool for agent | All agents: sequential-thinking (5+ steps), memory, mql5-books/docs

### Agent Handoffs
```
CRUCIBLE → SENTINEL: Verify risk | CRUCIBLE → ORACLE: Validate setup
ARGUS → FORGE: Implement pattern | FORGE → ORACLE: Validate code
ORACLE → SENTINEL: Calculate sizing | FORGE ↔ NAUTILUS: MQL5/Python migration
```

### MCP Quick Reference
| Need | MCP | Agent | Limits |
|------|-----|-------|--------|
| Compile MQL5 (AUTO) | metaeditor64 | FORGE | - |
| XAUUSD prices | twelve-data | CRUCIBLE | 8 req/min |
| Macro/DXY/COT | perplexity | CRUCIBLE | - |
| Lot/Kelly/DD | calculator | SENTINEL | - |
| MQL5 syntax | mql5-docs | FORGE | - |
| Deep research | perplexity+exa | ARGUS | - |
| Scrape web | firecrawl | ARGUS | 820 req |
| Monte Carlo | calculator+e2b | ORACLE | - |
| Charts/viz | vega-lite | ORACLE | - |

---

## 3. KNOWLEDGE MAP

| Need | Location |
|------|----------|
| Strategy XAUUSD | `.factory/droids/crucible-gold-strategist.md` |
| Risk/Apex | `.factory/droids/sentinel-apex-guardian.md` |
| Code MQL5/Python | `.factory/droids/forge-mql5-architect.md` |
| Backtest/Validation | `.factory/droids/oracle-backtest-commander.md` |
| Research/Papers | `.factory/droids/argus-quant-researcher.md` |
| Nautilus Migration | `.factory/droids/nautilus-trader-architect.md` |
| Implementation Plan | `DOCS/02_IMPLEMENTATION/PLAN_v1.md` |
| Nautilus Plan | `DOCS/02_IMPLEMENTATION/NAUTILUS_MIGRATION_MASTER_PLAN.md` |
| Technical Reference | `DOCS/06_REFERENCE/CLAUDE_REFERENCE.md` |
| RAG MQL5 syntax | `.rag-db/docs/` (semantic query) |
| RAG concepts/ML | `.rag-db/books/` (semantic query) |

### DOCS Structure
```
DOCS/
├── _INDEX.md                 # Central navigation
├── 00_PROJECT/               # Project-level docs
├── 01_AGENTS/                # Agent specs, Party Mode
├── 02_IMPLEMENTATION/        # Plans, progress, phases
├── 03_RESEARCH/              # Papers, findings (ARGUS)
├── 04_REPORTS/               # Backtests, validation (ORACLE)
├── 05_GUIDES/                # Setup, usage, troubleshooting
└── 06_REFERENCE/             # Technical, MCPs, integrations
```

### Where Agents Save

| Agent | Output Type | Save To |
|-------|-------------|---------|
| CRUCIBLE | Strategy/Setup | `DOCS/03_RESEARCH/FINDINGS/` |
| SENTINEL | Risk/GO-NOGO | `DOCS/04_REPORTS/DECISIONS/` |
| FORGE | Code/Audits/Guides | `DOCS/02_IMPLEMENTATION/PHASES/`, `DOCS/05_GUIDES/` |
| ORACLE | Backtests/WFA/GO-NOGO | `DOCS/04_REPORTS/BACKTESTS|VALIDATION|DECISIONS/` |
| ARGUS | Papers/Research | `DOCS/03_RESEARCH/PAPERS|FINDINGS/` |
| NAUTILUS | Migration code/progress | `nautilus_gold_scalper/src/`, migration plan |
| ALL | Progress/Party Mode | `DOCS/02_IMPLEMENTATION/PROGRESS.md`, `DOCS/01_AGENTS/PARTY_MODE/` |

### Bug Fix Log (MANDATORY)
**File**: `MQL5/Experts/BUGFIX_LOG.md`
**Use**: FORGE (all MQL5/Python fixes), ORACLE (backtest bugs), SENTINEL (risk logic bugs)
**Format**: `YYYY-MM-DD (AGENT context)\n- Module: bug description and fix.`

### Naming Conventions
- Reports: `YYYYMMDD_TYPE_NAME.md` (e.g., `20251130_WFA_REPORT.md`)
- Findings: `TOPIC_FINDING.md` (e.g., `SMC_ORDER_BLOCKS_FINDING.md`)
- Decisions: `YYYYMMDD_GO_NOGO.md`

---

## 4. APEX TRADING ESSENTIALS

**Limits** (Trailing DD - NOT fixed):
- Trailing DD: 10% from HIGH-WATER MARK (follows peak equity, includes unrealized P&L!)
- Risk/trade: 0.5-1% max (conservative near HWM)
- NO OVERNIGHT: Close ALL by 4:59 PM ET
- Consistency: Max 30% profit in single day
- Violation = ACCOUNT TERMINATED

**Critical Difference vs FTMO**:
- FTMO: Fixed DD from initial balance
- APEX: DD follows equity peak (MORE DANGEROUS!)
- Example: Profit $500 → Floor rises $500

**Time Constraints (ET)**:
- 4:00 PM: Alert - prepare close
- 4:30 PM: Urgent - start closing
- 4:55 PM: EMERGENCY - close all
- 4:59 PM: ABSOLUTE DEADLINE

**Performance**: OnTick <50ms | ONNX <5ms | Python Hub <400ms

**ML Thresholds**: P(direction) >0.65 → Trade | WFE ≥0.6 → Approved | Monte Carlo 95th DD <8%

---

## 5. SESSION & CODING RULES

**Session**: 1 SESSION = 1 FOCUS. Checkpoint every 20 msgs. Ideal: 30-50 msgs. Use NANO skills when possible.

**MQL5 Standards**:
- Classes: `CPascalCase` | Methods: `PascalCase()` | Variables: `camelCase`
- Constants: `UPPER_SNAKE_CASE` | Members: `m_memberName`
- Always verify errors after trade ops

**Before Coding**: Consult RAG → Check existing patterns → Verify library exists

**Security**: NEVER expose secrets/keys/credentials

---

## 6. MQL5 COMPILATION (AUTO-COMPILE)

**Paths**:
- Compiler: `C:\Program Files\FTMO MetaTrader 5\metaeditor64.exe`
- Project: `C:\Users\Admin\Documents\EA_SCALPER_XAUUSD\MQL5`
- StdLib: `C:\Program Files\FTMO MetaTrader 5\MQL5`

**Command**:
```powershell
Start-Process -FilePath "C:\Program Files\FTMO MetaTrader 5\metaeditor64.exe" `
  -ArgumentList '/compile:"[FILE]"','/inc:"[PROJECT]"','/inc:"[STDLIB]"','/log' -Wait -NoNewWindow
```

**Read Result**: `Get-Content "[FILE].log" -Encoding Unicode | Select-String "error|warning|Result"`

**⚠️ FORGE RULE (P0.5)**: Auto-compile after ANY MQL5 change. Fix errors BEFORE reporting. Never deliver non-compiling code!

**Common Errors**: file not found → include path | undeclared identifier → import missing | unexpected token → syntax | closing quote → string format

---

## 7. DOCUMENT HYGIENE (EDIT > CREATE)

**RULE**: Before creating ANY doc:
1. Glob/Grep search existing similar docs
2. IF EXISTS → EDIT/UPDATE it
3. IF NOT → Create new
4. CONSOLIDATE related info in SAME file

**Never**: Create 5 separate files for related findings | Create _V1, _V2, _V3 versions | Ignore existing _INDEX.md

---

## 8. WINDOWS CLI ESSENTIALS

**Tools**: `C:\tools\rg.exe` (text search), `C:\tools\fd.exe` (file search)

**PowerShell Commands** (one per Execute call):
```powershell
New-Item -ItemType Directory -Path "path" -Force              # mkdir
Move-Item -Path "src" -Destination "dst" -Force               # move
Copy-Item -Path "src" -Destination "dst" -Force               # copy
Remove-Item -Path "target" -Recurse -Force -ErrorAction SilentlyContinue  # delete
```

**⚠️ CRITICAL**: Factory CLI uses PowerShell, NOT CMD!
- ❌ NEVER: `&`, `&&`, `||`, `2>nul` (CMD operators don't work in PS)
- ❌ NEVER: `cmd /c "mkdir x & move y"` (chained commands fail)
- ✅ ALWAYS: One command per Execute | Use Factory tools (Read, Create, Edit, LS, Glob, Grep) when possible

**Prefer Factory Tools Over Shell**:
| Need | Use Tool | Not Shell |
|------|----------|-----------|
| Create file | Create tool | echo > |
| Read file | Read tool | type/cat |
| Edit file | Edit tool | sed/awk |
| List dir | LS tool | dir/ls |
| Find files | Glob tool | dir /s/find |
| Find text | Grep tool | findstr/grep |

---

## 9. ANTI-PATTERNS & QUICK ACTIONS

**DON'T**:
- More planning (PRD complete) | Docs instead of code | Tasks >4hrs
- Ignore Apex limits (trailing DD, 4:59 PM) | Code without RAG | Trade in RANDOM_WALK
- Switch agents every 2 messages | Overnight positions

**DO**:
- Build > Plan | Code > Docs | Consult specialized skill
- Test before commit | Respect Apex always | Verify HWM before trades

**Quick Actions**:
| Situation | Action |
|-----------|--------|
| Implement X | Check PRD → FORGE implements |
| Research X | ARGUS /pesquisar |
| Validate backtest | ORACLE /go-nogo |
| Calculate lot | SENTINEL /lot [sl] (considers trailing DD + time) |
| Complex problem | sequential-thinking (5+ thoughts) |
| MQL5 syntax | RAG query .rag-db/docs |

---

## 10. GIT AUTO-COMMIT

**When**: Module created | Feature done | Significant bugfix | Refactor | Skill/Agent modified | Session ended

**How**: `git status` → `git diff` (check secrets!) → `git add [files]` → `git commit -m "feat/fix/refactor: desc"` → `git push`

---

*Skills have deep knowledge. Technical reference: DOCS/CLAUDE_REFERENCE.md. Full spec: DOCS/prd.md*
