# Architecture: AI Agents (Python)

## Executive Summary
The AI Agents component (`🤖 AI_AGENTS`) is a Multi-Agent System (MAS) designed to provide intelligent decision-making support to the Trading Engine. It utilizes the Model Context Protocol (MCP) to integrate with external tools and the MetaTrader 5 terminal.

## Technology Stack
*   **Language:** Python
*   **Architecture:** Multi-Agent System (MAS)
*   **Integration:** Model Context Protocol (MCP)
*   **Communication:** Message Queue, Shared Memory, Event System

## Agent Roles

### 1. Master Coordinator (`agent_master_coordinator`)
*   **Role:** Orchestrates the entire agent ecosystem.
*   **Responsibilities:** Task management, resource allocation, agent coordination.
*   **MCP Servers:** `metatrader5`, `sequential-thinking`, `task_management`.

### 2. Analysis Agent (`agent_01_market_analysis`)
*   **Role:** Technical and fundamental analysis.
*   **Capabilities:** Chart analysis, pattern recognition, signal generation.

### 3. Strategy Agent (`agent_02_strategy_development`)
*   **Role:** Strategy formulation and refinement.
*   **Capabilities:** Strategy logic, parameter tuning.

### 4. Risk Agent (`agent_03_risk_management`)
*   **Role:** Risk oversight and compliance.
*   **Capabilities:** Risk monitoring, drawdown control, exposure management.

### 5. Optimization Agent (`agent_04_optimization`)
*   **Role:** Performance optimization.
*   **Capabilities:** Backtesting, parameter optimization.

### 6. Testing Agent (`agent_05_testing`)
*   **Role:** Quality assurance and validation.
*   **Capabilities:** System testing, validation checks.

## Communication Architecture
The agents communicate via three primary channels:
1.  **Message Queue:** Asynchronous communication for task distribution.
2.  **Shared Memory:** Low-latency access to real-time market data.
3.  **Event System:** Event-driven coordination for immediate actions.

## MCP Integration
The system uses MCP to standardize tool access:
*   **Protocol Version:** 1.0
*   **Supported Tools:** File operations, Market Data, Trade Execution.
*   **Configuration:** `MCP_Integration/MCP_Configs`
