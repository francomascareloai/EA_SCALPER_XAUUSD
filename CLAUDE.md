# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## CRITICAL PRIORITY: Sequential Thinking Mandate

**FIRST AND FOREMOST**: ALL analysis, strategy development, and code review MUST use sequential thinking. This is not a suggestion - it's a mandatory requirement for this trading system.

- Activate sequential thinking for ANY trading-related analysis
- Use "ultrathink" or "think step by step" commands
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

### Python Environment
```bash
# Run trading agents with different configurations
python scripts/python/trading_agent_simple.py
python scripts/python/litellm_server.py
python scripts/python/simple_trading_proxy.py

# Classification and file management tools
python 🛠️/TOOLS/TOOLS_FINAL/Classification/classificador_otimizado.py

# MCP (Model Context Protocol) servers
python 🤖/AI_AGENTS/MCP_Code_Checker/src/main.py

# Run tests
python -m pytest tests -q
```

### MQL5/MQL4 Development
The project uses MetaTrader development environment. Expert Advisors are compiled and deployed to MT5 terminals.

### Configuration Files
- Use `.env` for API keys and environment variables
- Configuration files in `configs/` directory:
  - `litellm_config.yaml` - LLM model configuration
  - `litellm_openrouter_config.yaml` - OpenRouter integration

## Key Architecture Patterns

### Trading System Architecture
1. **Multi-Agent Trading System** - Multiple coordinated trading agents
2. **FTMO-Ready Strategies** - Prop firm compliant risk management
3. **ML Integration** - Machine learning models for market analysis
4. **Real-time Data Processing** - Live market data analysis

### Code Organization Philosophy
- **Production vs Development** - Clear separation between production EAs and development code
- **Scalable Structure** - Organized for thousands of trading files
- **Multi-Language Support** - MQL4, MQL5, and Python integration
- **Metadata-Driven** - Extensive metadata organization for trading strategies

### AI Integration Patterns
- **Model Context Protocol (MCP)** servers for code validation
- **LiteLLM integration** for multiple AI model providers
- **Automated classification** and analysis tools
- **Real-time trading agents** with AI decision making

## Specialized Components

### Expert Advisors (EAs)
- `EA_AUTONOMOUS_XAUUSD_ELITE_v2.0.mq5` - Main production EA
- `EA_FTMO_Scalper_Elite_v2.10_BaselineWithImprovements.mq5` - FTMO-compliant version
- `XAUUSD_ML_Trading_Bot.mq5` - Machine learning integrated trading bot

### Python Trading Infrastructure
- Trading agents with OpenRouter integration
- Automated classification and analysis systems
- Real-time monitoring and optimization tools
- Multi-agent coordination systems

### Risk Management
- FTMO-compliant risk parameters
- Automated position sizing
- Drawdown monitoring and controls
- Multi-timeframe risk analysis

## Working with This Codebase

### For Trading Development
1. Start with EAs in `🚀 MAIN_EAS/PRODUCTION/` for production strategies
2. Use `🔧 WORKSPACE/Development/` for new strategy development
3. Test strategies in `📊 DATA/backtest_results/` before deployment

### For AI/ML Development
1. Configure API keys in `.env` following `.env.example`
2. Use MCP servers in `🤖 AI_AGENTS/` for code validation
3. Integrate with trading agents in `scripts/python/`

### For File Management
1. Use classification tools in `🛠️ TOOLS/TOOLS_FINAL/Classification/`
2. Follow the established folder structure with emoji indicators
3. Maintain metadata organization in appropriate folders

## Environment Setup

1. Create `.env` file from `.env.example` with your API keys
2. Install Python dependencies (requirements scattered across subdirectories)
3. Configure MetaTrader 5 terminal for EA deployment
4. Set up MCP servers for AI integration

## MCP Servers Integration

This project has extensive MCP (Model Context Protocol) server integration for enhanced capabilities:

### Available MCP Servers

#### Search & Research
- **brave-search** - Web search using Brave Search API with `BRAVE_API_KEY`
- **tavily-search** - Advanced web search with Tavily AI (Smithery hosted)
- **perplexity-search** - Perplexity AI search integration
- **exa** - High-quality web search and content extraction

#### Development & Code Tools
- **github** - GitHub repository access and management (requires `GITHUB_PERSONAL_ACCESS_TOKEN`)
- **context7** - Documentation and code context retrieval for libraries
- **claude-context** - Advanced codebase indexing and semantic search
- **code-reasoning** - Sequential thinking for complex problem-solving
- **sequential-thinking** - Structured reasoning and analysis

#### Database & Data Management
- **postgres** - PostgreSQL database access (connected to Supabase)
- **filesystem** - File system access and management

#### Automation & Workflow
- **n8n** - Workflow automation via n8n platform with Docker
- **linear** - Project management integration (Linear API)
- **notion** - Notion database and page management

#### Web & Browser Automation
- **playwright** - Browser automation and testing
- **microsoft-playwright-mcp** - Microsoft's Playwright integration
- **shay-5555-gif-chrome-devtools-mcp-2** - Chrome DevTools automation
- **smithery-ai-fetch** - Web content fetching and extraction

#### Communication & Collaboration
- **docfork-mcp** - Documentation search and retrieval
- **smithery-toolbox** - General utility tools

### When to Use Each MCP Server

#### For Trading Research & Analysis
- Use **brave-search** or **tavily-search** for market news and analysis
- Use **perplexity-search** for in-depth trading strategy research
- Use **exa** for high-quality content extraction from financial websites

#### For Code Development
- Use **github** for repository management and code review
- Use **context7** for library documentation and examples
- Use **claude-context** for codebase semantic search and understanding
- Use **code-reasoning** for complex algorithmic problem-solving

#### For Data Analysis
- Use **postgres** for storing and analyzing trading data
- Use **filesystem** for managing backtest results and historical data

#### For Automation
- Use **n8n** for creating automated trading workflows
- Use **linear** for project management and task tracking
- Use **notion** for documentation and knowledge management

#### For Testing & Validation
- Use **playwright** for web-based trading interface testing
- Use **microsoft-playwright-mcp** for advanced browser automation
- Use **chrome-devtools-mcp** for debugging web-based trading platforms

### MCP Configuration
Configuration is stored in `.claude.json` with server-specific settings including API keys, timeouts, and connection parameters. Some servers require Docker containers or external API keys.

## Sequential Thinking - CRITICAL PRIORITY

**MANDATORY**: All complex analysis, code review, and trading strategy development MUST use sequential thinking. This is not optional - it's required for quality results.

### When to Use Sequential Thinking
**ALWAYS use for:**
- Trading strategy analysis and optimization
- Risk management logic evaluation
- Multi-agent system design
- Market data analysis and pattern recognition
- Code architecture decisions
- Bug diagnosis in complex trading logic
- Performance optimization analysis
- Backtest result interpretation

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
Thought 3: Market condition handling and edge cases
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

### Sequential Thinking Commands
Use these explicit triggers:
- `"ultrathink"` - Maximum depth analysis (5-7 thoughts minimum)
- `"think step by step"` - Detailed sequential analysis
- `"analyze this thoroughly"` - Deep dive with market context
- `"chain of reasoning"` - Connect multiple analysis points

### Quality Standards for Sequential Thinking
Each thought MUST:
1. Build upon previous reasoning
2. Consider market context and trading implications
3. Address risk management explicitly
4. Include concrete action items
5. Reference specific files or data when applicable

### Examples of Required Sequential Thinking

#### For EA Analysis:
```
User: "Analyze the performance of EA_AUTONOMOUS_XAUUSD_ELITE_v2.0"
Response should use sequential thinking covering:
1. Strategy mechanics and entry logic analysis
2. Risk management implementation review
3. Market condition适应性 evaluation
4. Performance metrics interpretation
5. Optimization recommendations
```

#### For Strategy Development:
```
User: "Create a new scalping strategy for gold"
Response must use sequential thinking:
1. Market characteristics and volatility patterns
2. Timeframe and indicator selection
3. Entry/exit signal generation logic
4. Risk parameters and position sizing
5. Backtesting and validation approach
```

## Fast Development Tools

This project emphasizes fast, efficient development workflows using modern command-line tools:

### Search & File Operations
- **Use `ripgrep (rg)` instead of `grep`** for all project-wide searches
  - `rg "pattern"` - Search content across the project
  - `rg --files | rg "name"` - Find files by name
  - `rg -t python "def"` - Language-specific searches
  - `rg -n -A 3 -B 3` - Get context around matches
- **Use `fd` (or `fdfind` on Ubuntu/Debian) instead of `find`**
  - Respects .gitignore automatically
  - Much faster than traditional find
- **Use `jq` for JSON parsing and transformations**

### Tool Installation
```bash
# macOS
brew install ripgrep fd jq

# Ubuntu/Debian
sudo apt update && sudo apt install -y ripgrep fd-find jq
# Create alias: alias fd=fdfind
```

### Performance Guidelines
- **File Reading**: Cap reads at 250 lines for large files
- **Code Search**: Prefer `rg -n -A 3 -B 3` for context over full file reads
- **JSON Processing**: Always use `jq` instead of regex for JSON manipulation
- **Batch Operations**: Use `rg --files-with-matches` to find files, then process in parallel

### Examples for This Project
```bash
# Find all MQL5 files
rg --files -t mq5 "\.mq5$"

# Search for trading strategies
rg -t mq4 -t mq5 "strategy|scalper|expert"

# Find configuration files
rg --files | rg "\.(yaml|json|toml)$"

# Search for risk management code
rg -n -A 5 -B 5 "risk|drawdown|stop_loss"

# Parse trading data from JSON
cat trading_results.json | jq '.trades | map(select(.profit > 0))'
```

### Quality Assurance & Sequential Thinking Verification

**Self-Check Requirements for All Responses:**
1. Did I use sequential thinking for this analysis?
2. Is there a clear chain of reasoning from problem to solution?
3. Are risk management implications thoroughly addressed?
4. Are market conditions and context properly considered?
5. Are there concrete, actionable recommendations?

**Red Flags That Indicate Inadequate Sequential Thinking:**
- Single-paragraph responses to complex trading questions
- Generic advice without specific implementation details
- Missing risk analysis in strategy recommendations
- No consideration of market regimes or conditions
- Superficial code review without deeper analysis

**If You Catch Yourself Not Using Sequential Thinking:**
- Stop and restart with proper sequential thinking
- Use explicit trigger: "Let me think through this step by step"
- Reference the Sequential Thinking Protocol above
- Ensure each thought builds upon the previous one

### Important Notes

- This is a professional trading system with real financial implications
- **Sequential thinking is MANDATORY for all analysis - no exceptions**
- Always test EAs thoroughly in demo environments before live deployment
- The project contains both legacy and modern trading strategies
- File organization uses emoji prefixes for visual categorization
- Multi-language support requires careful path management for includes
- MCP servers require proper API keys and configuration for full functionality
- Some MCP servers need Docker or external services to be running
- **Quality of analysis directly impacts trading performance and financial outcomes**