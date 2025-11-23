# Architecture: Trading Engine (MQL5)

## Executive Summary
The Trading Engine is a high-performance Expert Advisor (EA) written in MQL5, designed for scalping XAUUSD. It integrates Machine Learning predictions, Smart Money Concepts (SMC), and strict Risk Management to execute trades with low latency.

## Technology Stack
*   **Language:** MQL5 (MetaQuotes Language 5)
*   **Platform:** MetaTrader 5
*   **Key Libraries:**
    *   `XAUUSD_ML_Core`: Core logic and ML integration
    *   `XAUUSD_ML_Strategies`: Strategy implementation
    *   `XAUUSD_ML_Risk`: Risk management and position sizing
    *   `XAUUSD_ML_Visual`: Dashboard and UI

## Architecture Pattern
**Event-Driven Modular Monolith**
The system runs on the MetaTrader 5 event loop (`OnTick`, `OnInit`, `OnDeinit`). It uses a modular object-oriented design where distinct responsibilities (Core, Strategy, Risk, UI) are encapsulated in separate classes, instantiated as global objects.

## Component Overview

### 1. Main Entry Point (`XAUUSD_ML_Trading_Bot.mq5`)
*   **Responsibility:** Orchestrates the system lifecycle and event handling.
*   **Key Functions:**
    *   `OnInit()`: Initializes components and loads ML models.
    *   `OnTick()`: Main execution loop. Triggers analysis, risk checks, and strategy execution.
    *   `OnDeinit()`: Cleans up resources.

### 2. ML Core (`CXAUUSD_MLCore`)
*   **Responsibility:** Market analysis, ML model management, and trade execution.
*   **Features:**
    *   Latency optimization (Pre-stops, Order buffering).
    *   ML Model updates.
    *   Market state analysis.

### 3. Strategies (`CXAUUSD_MLStrategies`)
*   **Responsibility:** Generates trade signals based on market analysis.
*   **Supported Strategies:**
    *   Smart Money Concepts (Order Blocks, FVGs).
    *   ML Scalping.
    *   Volatility Breakout.

### 4. Risk Manager (`CXAUUSD_MLRisk`)
*   **Responsibility:** Validates signals and calculates position sizes.
*   **Features:**
    *   FTMO Compliance checks (Max Daily Loss, Max Drawdown).
    *   Dynamic lot sizing based on risk %.

### 5. Visual Interface (`CXAUUSD_MLVisual`)
*   **Responsibility:** Real-time dashboard and status updates.
*   **Features:**
    *   Displays PnL, Drawdown, and Active Strategy.
    *   Visualizes analysis results on the chart.

## Data Flow
1.  **Tick Event**: `OnTick()` is triggered by a new price quote.
2.  **Analysis**: `g_mlCore.AnalyzeMarket()` processes current data.
3.  **Risk Check**: `g_riskManager.ValidateRiskLimits()` ensures account safety.
4.  **Strategy**: `g_strategies.SelectOptimalStrategy()` picks the best approach and generates a signal.
5.  **Validation**: Signal is validated against risk and latency constraints.
6.  **Execution**: `g_mlCore.ExecuteTradeWithLatencyOptimization()` sends the order.

## Configuration
The EA is configured via Input Parameters in the following groups:
*   `=== ML CONFIGURATION ===`: Model confidence, update frequency.
*   `=== RISK MANAGEMENT ===`: Risk %, Drawdown limits, FTMO mode.
*   `=== STRATEGY SELECTION ===`: Toggles for individual strategies.
*   `=== LATENCY OPTIMIZATION ===`: Max latency, pre-stops.
*   `=== VISUAL INTERFACE ===`: UI toggles and colors.

## Modularization Strategy
The codebase has been refactored from a monolithic structure into a component-based architecture to improve maintainability and testability.

### Core Components (`Include/EA_Elite_Components/`)
*   **`Definitions.mqh`**: Centralized type definitions, enums, and shared data structures (`SAdvancedOrderBlock`, `SFTMOCompliance`, etc.). Acts as the contract between components.
*   **`EliteOrderBlock.mqh`**: Encapsulates Order Block detection logic in `CEliteOrderBlockDetector`.
*   **`EliteFVG.mqh`**: Encapsulates Fair Value Gap analysis in `CEliteFVGDetector`.
*   **`InstitutionalLiquidity.mqh`**: Handles liquidity pool identification via `CInstitutionalLiquidityDetector`.
*   **`FTMO_RiskManager.mqh`**: Dedicated risk management class `CFTMORiskManager` that enforces prop firm rules.

### Design Principles
*   **Encapsulation**: Each detector class manages its own state (e.g., `m_order_blocks` array) and exposes data via accessors (`GetOrderBlock`, `GetCount`).
*   **Dependency Injection**: Components are initialized with necessary dependencies rather than relying on global state where possible.
*   **Separation of Concerns**: Trading logic (Entry/Exit) is separated from Analysis logic (OB/FVG detection) and Risk logic.
