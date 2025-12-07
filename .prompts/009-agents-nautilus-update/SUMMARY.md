# 🔧 AGENTS.md Nautilus Update Summary

**One-liner:** Updated AGENTS.md v3.4 → v3.4.1 to prioritize NautilusTrader (Python) over MQL5, marking MQL5 as LEGACY

## Version
v3.4.1 (from v3.4.0)

## Objective
Adjust AGENTS.md to reflect project's migration from MQL5 to NautilusTrader (Python) as primary development platform. Update bugfix_protocol, forge_rule, error_recovery, and examples to reference Nautilus instead of MQL5.

## Expected Changes
• Primary bugfix log: `MQL5/Experts/BUGFIX_LOG.md` → `nautilus_gold_scalper/BUGFIX_LOG.md`
• FORGE validation: metaeditor64 compilation → mypy + pytest
• New error recovery protocols: Python Type/Import Errors + Event-Driven Pattern Violation
• MQL5 compilation moved to `<mql5_legacy>` section (kept for reference)
• FORGE MCPs: context7★ + e2b★ primary (was metaeditor64 + mql5-docs)
• All examples updated to Nautilus context (Actors, Strategies, event-driven patterns)

## Files Modified
- `AGENTS.md` (in-place edits)
- `AGENTS_v3.4_BACKUP_PRE_NAUTILUS_UPDATE.md` (backup created)

## Decisions Needed
None - changes align with project's NautilusTrader migration

## Blockers
None

## Next Step
Execute 010-droid-refactoring-master.md (FASE 2-4: refactor TOP 5 droids with inheritance)

## Estimated Time
15-20 minutes (precision edits to specific sections)

## Intelligence Required
- **Sequential-thinking** (7+ thoughts) to ensure MQL5 knowledge preserved as legacy (not deleted)
- **Proactive problem detection** to verify no breaking changes for existing droid invocations
