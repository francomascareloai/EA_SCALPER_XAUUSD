# Backtest Code Audit - Summary

## One-Liner
Found 4 P0 blockers (no Apex daily reset/overnight flat, risk engine bypass, sizing unit bug, threshold drift) plus unused safety modules; backtest realism is NO-GO.

## Version
v1 - Initial audit (2025-12-07)

## Key Findings
- Daily loss never resets in PropFirmManager (no `on_new_day`), and strategy lacks 4:59 PM ET/no-overnight enforcement → Apex breach risk.
- Risk engine bypassed with zero slippage/commissions; config realism knobs are unused, so results materially overstate performance.
- Execution thresholds inconsistent (strategy 70 vs runners 65/50); PositionSizer receives price units instead of pips, skewing lot sizes.
- SpreadMonitor, CircuitBreaker, StrategySelector, EntryOptimizer, TradeManager, ML stack all unused; tests lack end-to-end backtest coverage (“não testa 100%”).

## Decisions Needed
- Approve immediate P0 fixes (reset hook, slippage/commission wiring, sizing unit conversion, threshold alignment) before any further backtests.
- Confirm target execution threshold (70 vs 65) and desired slippage/commission assumptions for Apex realism.

## Blockers
- Apex compliance (daily reset, cutoff/overnight) not implemented.
- Realism costs (slippage/commission) omitted; risk engine bypassed.
- Lot sizing unit mismatch can over/under-size positions.

## Next Step
Implement P0 fixes and rerun tick backtest with validated data; add end-to-end pytest covering Apex rules and realism costs.
