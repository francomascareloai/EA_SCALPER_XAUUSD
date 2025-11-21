MCP Backtest & Validation (Spec Mode)

Overview
- Deterministic backtesting and validation for MT5 EAs using a spec file.
- No LLM in the trading loop. The EA stays fully deterministic.
- The MCP runner compiles the EA, runs MT5 Strategy Tester for defined periods, parses the report, and validates against pass/fail thresholds.

Requirements
- Windows with MetaTrader 5 installed
- Paths to `metaeditor64.exe` and `terminal64.exe`

Quick Start
1) Edit the paths and run (PowerShell):
   - `MCP\mcp_run.ps1 -MetaEditor "C:\\Program Files\\MetaTrader 5\\metaeditor64.exe" -Terminal "C:\\Program Files\\MetaTrader 5\\terminal64.exe" -Spec MCP\specs\FTMO_Conservative.json`

2) Or run via Python directly:
   - `python MCP\mcp_backtest_runner.py --metaeditor "C:\\...\\metaeditor64.exe" --terminal "C:\\...\\terminal64.exe" --spec MCP\specs\FTMO_Conservative.json`

Spec Structure
- See `MCP\specs\FTMO_Conservative.json` and `MCP\specs\Aggressive_50_3.json`.
- Includes: symbol, period, deposit, datasets (multiple From/To ranges), thresholds (return, max_dd, pf, wr), and EA parameter preset `.set` file.

Outputs
- `MCP\out\<run_id>\` with:
  - compiled logs, ini files
  - MT5 reports per dataset
  - `summary.json` with consolidated metrics and PASS/FAIL

Notes
- “Aggressive_50_3” is intentionally unrealistic to demonstrate failing a spec. Use conservative spec for FTMO-aligned validation.
- If MT5 Report HTML changes between builds, the parser falls back to robust keyword scanning.

