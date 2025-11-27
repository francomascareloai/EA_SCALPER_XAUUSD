# CLAUDE.md - EA_SCALPER_XAUUSD Development Guide

This file provides guidance for Factory Droid (Claude Opus 4.5) when working with this trading system repository.

---

## CRITICAL: Builder Mode First

**STOP PLANNING. START BUILDING.**

This project has a complete PRD v2.1 (`DOCS/prd.md`). No more planning is needed. Every session should focus on:
1. Pick ONE small task from the PRD
2. Build it
3. Test it
4. Move to next task

If you catch yourself writing more documentation instead of code → STOP and redirect to implementation.

---

## Project Overview

**EA_SCALPER_XAUUSD** - Institutional-grade Expert Advisor for XAUUSD scalping with:
- MQL5 execution engine (high-speed, FTMO-compliant)
- Python Agent Hub for advanced analysis (ML, sentiment, fundamentals)
- Multi-agent scoring system (TechScore, FundScore, SentScore)
- Full explainability and reasoning for every trade

**Target**: FTMO $100k Challenge compliance (10% max DD, 5% daily DD)

---

## Factory Droid Toolbox

### Skills (Auto-Triggered)

Skills are loaded INTO my context and I execute them directly with full MCP access.

| Skill | Trigger Phrases | What It Does |
|-------|-----------------|--------------|
| `web-research` | "deep research", "find repositories", "investigate" | Multi-source research with triangulation |
| `scientific-critical-thinking` | "evaluate methodology", "assess evidence" | Rigorous validation of claims |
| `prompt-optimizer` | "optimize prompt", "improve this prompt", "elevate this" | Apply 23 principles to enhance any prompt |
| `skill-creator` | "create a skill", "new skill" | Guide for creating new skills |
| `mcp-builder` | "create MCP server", "build MCP" | Guide for MCP server development |

**To invoke**: Just say the trigger phrase naturally. Example: "I need deep research on LSTM for gold prediction"

### Droids (Subagents via Task Tool)

Droids are separate agents I can launch for complex, autonomous tasks.

| Droid | Use For | Invoke With |
|-------|---------|-------------|
| `deep-researcher` | Complex multi-source research, academic papers | Task tool with `deep-researcher` |
| `project-reader` | Codebase analysis, architecture understanding | Task tool with `project-reader` |
| `research-analyst-pro` | Decision-oriented research with recommendations | Task tool with `research-analyst-pro` |
| `trading-project-documenter` | Comprehensive trading system documentation | Task tool with `trading-project-documenter` |

**Note**: Droids do NOT have MCP access. For research requiring MCPs, use the `web-research` skill instead.

### Slash Commands

Quick workflows I can execute when you type `/command`:

#### BMAD Method Commands
| Command | Description |
|---------|-------------|
| `/bmad-analyze` | BMAD analysis workflow |
| `/bmad-brainstorm` | Structured brainstorming session |
| `/bmad-tech-spec` | Create technical specification |
| `/bmad-refine-strategy` | Council of Agents strategy refinement |
| `/bmad-new-feature` | End-to-end feature implementation |
| `/bmad-market-scan` | Daily market intelligence scan |

#### Development Commands
| Command | Description |
|---------|-------------|
| `/architect` | MQL5 architecture review |
| `/code-review` | Trading code review |
| `/strategy` | Strategy analysis |
| `/optimize` | EA optimization planning |

#### Validation Commands
| Command | Description |
|---------|-------------|
| `/backtest` | Statistical validation (Monte Carlo, WFA) |
| `/validate-ftmo` | FTMO compliance checker |

#### Research Commands
| Command | Description |
|---------|-------------|
| `/research` | Deep research with all MCPs |

#### Utility Commands
| Command | Description |
|---------|-------------|
| `/optimize-prompt` | Optimize any prompt using 23 principles |

### MCP Tools (Direct Access)

I have direct access to these MCP servers:

#### Search & Research
- `perplexity-search___search` - AI-synthesized answers with citations
- `brave-search___brave_web_search` - Broad web search, news, recent events
- `context7___get-library-docs` - Technical documentation for libraries

#### Code & Development
- `github___*` - Repository management, code search, PRs, issues
- `sequential-thinking___sequentialthinking` - Structured problem-solving
- `code-reasoning___code-reasoning` - Complex code analysis

#### Data
- `postgres___query` - Database queries (read-only)

---

## MQL5 Elite Ops Agents (Reference Personas)

When I need domain-specific expertise, I can adopt these personas from `.bmad/mql5-elite-ops/agents/`:

| Agent | Expertise | When to Invoke |
|-------|-----------|----------------|
| **Quantum Strategist** | PRD, risk analysis, FTMO compliance, R:R ratios | Strategy design, risk questions |
| **MQL5 Architect** | System design, async patterns, performance | Architecture decisions, module design |
| **Code Artisan** | Clean MQL5 code, optimization, implementation | Coding tasks, refactoring |
| **Deep Researcher** | Market analysis, fundamentals, sentiment | Market context, news impact |
| **Backtest Commander** | Validation, Monte Carlo, Walk-Forward | Testing protocols, robustness |

**To invoke**: Say "Act as [Agent Name]" or ask domain-specific questions and I'll naturally adopt the relevant persona.

---

## Key Documents

| Document | Location | Purpose |
|----------|----------|---------|
| PRD v2.1 | `DOCS/prd.md` | **THE BIBLE** - All implementation specs |
| Architecture | In PRD Section 5 | System design, layers, components |
| Risk Framework | In PRD Section 10 | FTMO compliance, position sizing |
| Phase Roadmap | In PRD Section 14 | Implementation phases |

---

## Repository Structure

```
EA_SCALPER_XAUUSD/
├── 🚀 MAIN_EAS/           # Expert Advisors
│   ├── PRODUCTION/        # Production-ready EAs
│   ├── DEVELOPMENT/       # Active development
│   └── TESTING/           # Testing phase
├── 📚 LIBRARY/            # Code library
│   ├── MQL5_Components/   # MQL5 modules
│   └── INCLUDES/          # Shared .mqh files
├── 🔧 WORKSPACE/          # Development environment
├── 🛠️ TOOLS/              # Python utilities
├── 📊 DATA/               # Trading data
├── Python_Agent_Hub/      # Python analysis agents
├── DOCS/                  # Documentation
│   └── prd.md             # PRD v2.1 (main reference)
├── .bmad/                 # BMAD method files
│   ├── bmm/               # BMad Method Module
│   ├── bmb/               # BMad Builder
│   └── mql5-elite-ops/    # Custom trading agents
└── .factory/              # Factory Droid config
    ├── commands/          # Slash commands
    ├── droids/            # Custom droids
    └── skills/            # Custom skills
```

---

## Builder Workflow

### The 3-Step Loop

```
┌─────────────────────────────────────────┐
│  STEP 1: PICK ONE THING (5 min)        │
│  - Open PRD, find smallest deliverable  │
│  - Example: "Create EliteOrderBlock.mqh"│
│  - NOT: "Design entire signal system"   │
├─────────────────────────────────────────┤
│  STEP 2: BUILD IT (1-4 hours)          │
│  - Write actual code                    │
│  - Use MCPs if stuck (/research)        │
│  - Reference Elite Ops agents as needed │
├─────────────────────────────────────────┤
│  STEP 3: VALIDATE (30 min)             │
│  - Compile and test                     │
│  - /validate-ftmo if risk-related       │
│  - Move to next small thing             │
└─────────────────────────────────────────┘
```

### Anti-Patterns to Avoid

| If doing this... | Do this instead |
|------------------|-----------------|
| Writing more docs | Write CODE |
| Refining PRD | PRD is DONE |
| Designing architecture | Architecture is IN THE PRD |
| Task > 4 hours | SPLIT IT SMALLER |
| Asking "how should we approach..." | Just START |

---

## Sequential Thinking Protocol

**MANDATORY** for all trading analysis. Use `sequential-thinking___sequentialthinking` MCP.

### When to Use
- Trading strategy analysis
- Risk management evaluation
- Code architecture decisions
- Bug diagnosis in trading logic
- Backtest result interpretation

### Quick Protocol
```
Thought 1: Define problem and constraints
Thought 2: Analyze current state
Thought 3: Evaluate options
Thought 4: Design approach with risk management
Thought 5: Implementation plan
```

### Triggers
- "ultrathink" - Maximum depth (5-7 thoughts)
- "think step by step" - Detailed analysis
- "analyze thoroughly" - Deep dive

---

## Development Commands

### MQL5 Compilation
```bash
# MetaEditor CLI (if configured)
"C:\Program Files\MetaTrader 5\metaeditor64.exe" /compile:"path\to\file.mq5"
```

### Python Environment
```bash
# Activate venv
cd Python_Agent_Hub
.venv\Scripts\activate

# Run tests
python -m pytest tests -q
```

### Quick Search
```bash
# Find MQL5 files (use Glob tool instead)
# Search for patterns (use Grep tool instead)
```

---

## FTMO Compliance Checklist

Every trade/module must respect:

| Constraint | Limit | Implementation |
|------------|-------|----------------|
| Max Daily DD | 5% | `FTMO_RiskManager` monitors |
| Max Total DD | 10% | Hard stop at 8% (buffer) |
| Risk per trade | 1% max | Position sizing formula |
| Daily risk limit | 2% max | Cumulative check |
| Emergency mode | Trigger at 4% daily | Stop new entries |

---

## Quick Reference Card

### I Need To... → Use This

| Task | Tool/Command |
|------|--------------|
| Research a topic | `/research` or say "deep research on X" |
| Validate strategy | `/validate-ftmo` |
| Review code | `/code-review` |
| Understand architecture | Reference PRD Section 5 |
| Check FTMO rules | Reference PRD Section 10 |
| Implement feature | Builder Workflow (pick → build → validate) |
| Analyze complex problem | "ultrathink" + sequential-thinking MCP |
| Find library docs | context7 MCP |
| Search GitHub | github MCP or `/research` |

---

## Session Checklist

At the start of each session:
1. ✅ What's the ONE thing to build today?
2. ✅ Is PRD open for reference?
3. ✅ Am I in Builder Mode (not Planning Mode)?

At the end of each session:
1. ✅ Did we produce CODE, not just docs?
2. ✅ Does it compile/run?
3. ✅ What's the next small task?

---

## Coding Guidelines

### MQL5 Standards
```mql5
#property copyright "EA_SCALPER_XAUUSD"
#property version   "1.00"
#property strict

// Use datetime, not int for time
// Check GetLastError() after trade ops
// Use SymbolInfoDouble/Integer (never hardcode)
// No blocking calls in OnTick() - use OnTimer()
```

### Python Standards
- Type hints for all functions
- Async/await for I/O operations
- Pydantic for data validation
- JSON schema versioning for EA↔Python messages

### Performance Constraints
- OnTick: < 50ms execution (no external calls)
- OnTimer: Python Hub calls (200ms interval, timeout 400ms)
- No WebRequest in OnTick - use OnTimer with bounded queue

---

## GitHub Reference Repositories

When implementing features, search these first:

| Category | Repos |
|----------|-------|
| Trading Frameworks | `freqtrade/freqtrade`, `polakowo/vectorbt` |
| ML for Trading | `AI4Finance-Foundation/FinRL`, `stefan-jansen/machine-learning-for-trading` |
| Backtesting | `mementum/backtrader`, `kernc/backtesting.py` |
| MQL5 | Search: `MQL5 FTMO`, `MQL5 scalper XAUUSD` |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Planning loop" | STOP. Pick smallest task. Build it. |
| "Need more info" | Check PRD first. It's comprehensive. |
| "Which tool?" | See Quick Reference Card above |
| "Complex problem" | Use ultrathink + sequential-thinking |
| Compilation error | Check includes, `#property strict` |
| Python timeout | Check Hub is running, reduce payload |

---

## Context Engineering (Self-Optimization)

Based on Anthropic's research: Context is FINITE. Every token competes for attention.

### Compaction Triggers

When to summarize and refocus:
- Session exceeds 30+ back-and-forth exchanges
- Multiple tangent topics explored
- "I feel lost" or "where were we?"
- Before starting a new major task

### Compaction Protocol

When triggered, I will:
1. **Summarize** key decisions made this session
2. **List** files created/modified
3. **Identify** current task and next step
4. **Discard** completed discussion threads from active focus

### Note-Taking for Long Sessions

For multi-hour sessions, I maintain mental notes:
- Key architectural decisions
- Unresolved issues/blockers
- Files that need attention
- User preferences discovered

### Self-Reflection Checkpoints

Every 10 exchanges, quick self-check:
- [ ] Am I still on the original task?
- [ ] Is my response at the right "altitude" (not too prescriptive, not too vague)?
- [ ] Am I being concise or adding unnecessary content?
- [ ] Should I suggest compaction?

### Context Efficiency Rules

1. **Minimal but sufficient** - Don't over-explain what the model can infer
2. **Examples > Rules** - Show, don't tell when possible
3. **Structure with delimiters** - XML tags, headers for clarity
4. **Prune completed threads** - Don't reference finished tasks

---

## Important Notes

- This is a professional trading system with real financial implications
- Always test in demo before live deployment
- FTMO limits are HARD constraints, not suggestions
- Quality of implementation directly impacts trading outcomes
- When in doubt, reference PRD v2.1 - it has all the answers

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v3.1 | 2025-11-27 | Context Engineering section, self-optimization protocols |
| v3.0 | 2025-11-27 | Factory Droid integration: Skills, Droids, Commands, Builder Workflow |
| v2.0 | 2025-11-20 | MCP servers, sequential thinking protocol |
| v1.0 | 2025-11-01 | Initial project setup |
