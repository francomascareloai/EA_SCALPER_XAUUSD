---
trigger: always_on
---

# Antigravity Agent Rules & Guidelines

## CRITICAL PRIORITY: Sequential Thinking Mandate

**FIRST AND FOREMOST**: ALL analysis, strategy development, and code review MUST use sequential thinking. This is not a suggestion - it's a mandatory requirement for this trading system.

- Activate sequential thinking for ANY trading-related analysis
- Use "ultrathink" or "think step by step" commands (or the `sequential-thinking` tool)
- Follow the detailed protocol outlined in the Sequential Thinking section below
- Never provide superficial analysis of trading strategies or risk management

## Project Overview

This is an EA (Expert Advisor) trading system repository focused on XAUUSD (Gold) scalping strategies for MetaTrader 5 (MT5). The project contains automated trading algorithms, AI-powered analysis tools, and a comprehensive codebase for algorithmic trading development.

## Repository Structure

The project is organized with a specialized structure for trading systems:

### Core Trading Components
- **🚀 MAIN_EAS/** - Primary Expert Advisors (production, development, testing)
  - `PRODUCTION/` - Production-ready EAs (v2.10+ with FTMO compliance)
  - `DEVELOPMENT/` - Active development EAs including ML-powered bots
  - `TESTING/` - EAs in testing phase
  - `BACKUP/` - Critical EA backups

- **📚 LIBRARY/** - Centralized code library
  - `MQL4_Components/` - MQL4 source code organized by strategy
  - `MQL5_Components/` - MQL5 advanced components and FTMO-ready strategies
  - `INCLUDES/` - Shared libraries (.mqh files)
  - `02_Strategies_Legacy/` - Legacy trading strategies

### Development Environment
- **🔧 WORKSPACE/** - Active development environment
  - `current_work/` - Current projects
  - `experiments/` - Testing and experiments
  - `Development/` - Development utilities and scripts

- **🛠️ TOOLS/** - Automation and analysis tools
  - `python_tools/` - Python utilities organized by function
  - `TOOLS_FINAL/` - Finalized tools and classification systems

### AI and Automation
- **🤖 AI_AGENTS/** - AI-powered agents and MCP servers
  - `MCP_Code_Checker/` - Model Context Protocol code validation server

- **📊 DATA/** - Trading data and results
  - `historical_data/` - Historical market data by timeframe
  - `backtest_results/` - Backtesting results
  - `live_results/` - Live trading results

## Common Development Commands

### Fast Development Tools
- **Use `ripgrep (rg)` instead of `grep`** for all project-wide searches
- **Use `fd` instead of `find`**
- **Use `jq` for JSON parsing**

## Sequential Thinking - CRITICAL PRIORITY

**MANDATORY**: All complex analysis, code review, and trading strategy development MUST use sequential thinking. This is not optional - it's required for quality results.

### Sequential Thinking Protocol

#### Step 1: Problem Decomposition
Break complex trading problems into sequential analysis steps:
```
Thought 1: Define the core trading problem and constraints
Thought 2: Analyze market conditions and data requirements
Thought 3: Evaluate existing solutions and their limitations
Thought 4: Design approach with risk management
Thought 5: Implementation strategy with testing phases
```

#### Step 2: Market Analysis Chain
For any trading decision:
```
Thought 1: Current market regime and volatility analysis
Thought 2: Technical indicator validation and reliability
Thought 3: Risk/reward ratio calculation
Thought 4: Entry/exit timing optimization
Thought 5: Position sizing and stop-loss logic
```

#### Step 3: Code Review Chain
For any EA analysis:
```
Thought 1: Strategy logic correctness and mathematical soundness
Thought 2: Risk management implementation and drawdown protection
Thought 3: Market condition adaptability evaluation
Thought 4: Performance bottlenecks and optimization opportunities
Thought 5: FTMO compliance and regulatory considerations
```

#### Step 4: Multi-Agent Coordination
When working with multiple trading agents:
```
Thought 1: Define agent responsibilities and communication protocols
Thought 2: Establish conflict resolution and priority hierarchies
Thought 3: Design fail-safes and error propagation controls
Thought 4: Implement monitoring and performance metrics
Thought 5: Create testing and validation frameworks
```

### Mandatory Sequential Thinking Triggers
**IMMEDIATELY activate sequential thinking when:**
- User mentions "analyze", "evaluate", "optimize", or "debug"
- Analyzing trading performance or backtest results
- Reviewing risk management or position sizing
- Designing new trading strategies or indicators
- Debugging complex multi-agent interactions
- Making architectural decisions affecting live trading

### Quality Standards for Sequential Thinking
Each thought MUST:
1. Build upon previous reasoning
2. Consider market context and trading implications
3. Address risk management explicitly
4. Include concrete action items
5. Reference specific files or data when applicable
