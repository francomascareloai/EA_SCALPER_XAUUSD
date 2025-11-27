# Technical Specification v2.1 (Elite Ops Edition)

**Author:** MQL5 Architect
**Date:** 2025-11-22
**Based on:** PRD v2.1

---

## 1. System Architecture: Hybrid Asynchronous

The system is divided into two autonomous entities that communicate asynchronously.

### 1.1 The Body (MQL5)
- **Role:** Execution, Immediate Risk Management, High-Frequency Logic.
- **Constraint:** `OnTick` must complete in < 50ms.
- **State:** Maintains the "Truth" of the account (Equity, Positions).

### 1.2 The Brain (Python)
- **Role:** Heavy Analysis, News Parsing, ML Inference.
- **Constraint:** Asynchronous only. Never blocks the Body.
- **State:** Stateless (mostly). Processes snapshots and returns context.

### 1.3 The Bridge (File Exchange / JSON)
- **Mechanism:** Shared Files in `Common/Files/EA_SCALPER_XAUUSD/`.
- **Why Files?** Robust, simple, persistent, and easy to debug. ZeroMQ is Phase 3.
- **Channels:**
    - `mql_to_py.json` (Request/Snapshot)
    - `py_to_mql.json` (Response/Context)
    - `heartbeat.timestamp` (Watchdog)

---

## 2. Directory Structure (MQL5)

We will adopt a strict modular structure to facilitate testing and replacement.

```text
MQL5/
  Experts/
    EA_SCALPER_XAUUSD.mq5       // Entry Point (Minimal Logic)
  Include/
    EA_SCALPER/
      Core/
        CEngine.mqh             // Main Orchestrator
        CState.mqh              // State Machine (IDLE, BUSY, SURVIVAL)
      Modules/
        Signal/
          ISignal.mqh           // Interface
          CTechScore.mqh        // MQL5 Native Scoring
        Risk/
          IRisk.mqh             // Interface
          CFTMORisk.mqh         // FTMO Rules Implementation
        Hub/
          CHubConnector.mqh     // Async File I/O
          CHeartbeat.mqh        // Watchdog Logic
        Persistence/
          CLocalCache.mqh       // JSON Parser/Saver
      Utils/
        CLogger.mqh
        CJson.mqh               // Third-party Lib
```

---

## 3. Class Design & Responsibilities

### 3.1 `CEngine` (The Orchestrator)
- **Members:** `CRiskManager`, `CSignalManager`, `CHubConnector`.
- **OnTick():**
    1.  Check `CHeartbeat`. If dead -> `SetState(EMERGENCY)`.
    2.  Update `CRiskManager` (Daily Loss).
    3.  If `SURVIVAL_MODE`, skip signals.
    4.  Get `Signal` from `CSignalManager`.
    5.  If `Signal > Threshold` AND `Risk == OK` -> Execute.
- **OnTimer():**
    1.  `CHubConnector.CheckInbox()` -> Read `py_to_mql.json`.
    2.  `CHubConnector.SendHeartbeat()`.

### 3.2 `CHeartbeat` (The Watchdog)
- **Logic:**
    - Writes `mql_heartbeat.txt` (Timestamp).
    - Reads `py_heartbeat.txt`.
    - `bool IsAlive()`: Returns `true` if `TimeCurrent() - LastPyTime < 15s`.

### 3.3 `CLocalCache` (The Bunker)
- **Responsibility:** Load/Save critical data.
- **Files:**
    - `news_calendar.json`:
        ```json
        {
          "events": [
            {"time": "1698754000", "impact": "HIGH", "currency": "USD"}
          ]
        }
        ```
    - `risk_params.json`:
        ```json
        {
          "risk_per_trade": 0.5,
          "max_daily_loss": 5.0,
          "mode": "NORMAL"
        }
        ```

---

## 4. Data Flow & Sequence

### 4.1 Normal Operation
1.  **MQL5 (OnTimer):** Writes `mql_to_py.json` (Price, Indicators).
2.  **Python (Loop):** Detects file change -> Analyzes -> Writes `py_to_mql.json`.
3.  **MQL5 (OnTimer):** Detects file change -> Updates `CState` (e.g., `TechSubScore`, `MarketRegime`).
4.  **MQL5 (OnTick):** Uses updated state to filter/execute trades.

### 4.2 Survival Mode (High Volatility)
1.  **Python:** Detects VIX spike.
2.  **Python:** Writes `py_to_mql.json` with `{"mode": "SURVIVAL", "risk_mult": 0.25}`.
3.  **MQL5:** Reads update. `CRiskManager` applies multiplier 0.25x to all new lots.

### 4.3 Emergency Mode (Python Death)
1.  **MQL5:** `CHeartbeat` checks time. `LastPyTime` is 20s ago.
2.  **MQL5:** `CEngine` sets `State = EMERGENCY`.
3.  **MQL5:** `CSignalManager` ignores all new signals.
4.  **MQL5:** `CRiskManager` continues to trail stops for open positions.

---

## 5. Implementation Plan (Phase 1)

1.  **Setup:** Create folder structure.
2.  **Core:** Implement `CHeartbeat` and `CHubConnector` (The communication layer).
3.  **Risk:** Port `FTMO_RiskManager` to `CFTMORisk` class.
4.  **Signal:** Port `EliteOrderBlock` to `CTechScore`.
5.  **Integration:** Wire them up in `CEngine`.

## 6. Libraries Required
- `JAson.mqh` (or similar) for robust JSON parsing in MQL5.
