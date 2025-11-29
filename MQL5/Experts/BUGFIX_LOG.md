Bugfix Index (EA_SCALPER_XAUUSD)
================================

- Emergency mode latch cleared and management kept alive during halts; risk refresh runs each tick and recovers on new day.
- Soft-stop now pauses only new entries (not full halt) and auto-resets on recovery; lot sizing uses equity.
- Order block validation fixed (liquidity flag populated; volume averaging normalized) so OB detection no longer rejects all signals.
- MTF manager uses persistent ATR handle and accepts real OB/FVG flags; FVG/OB structure now feeds MTF gating.
- Confluence scorer consumes real OB/FVG proximity with ATR-normalized scoring; detectors attached at init.
- FVG detection throttled to new M15 bars to reduce per-tick load while keeping entry context fresh.
- Entry flow now supplies actual OB/FVG/sweep levels to optimizer and blocks trades without structure context.
- Trade management partial-close PnL now uses per-partial profit (avoids overstating realized P/L).
- OrderFlowAnalyzer_v2: resets last price each bar to avoid cross-bar bias, sorts price levels before VA/POC to ensure correct ranges, and keeps metrics consistent.
- OrderFlowAnalyzer_v2: added sorted-level caching and bar/tick-count aware result cache to avoid redundant work; marks levels dirty on updates for real-time efficiency.
