# Data Directory Migration - 2025-12-08

## Summary

Migrated from unorganized data structure to centralized, config-driven approach.

## Problem

**Before:**
- Parquet files scattered in `data/ticks/`
- Hardcoded paths in scripts (`data/ticks/xauusd_2020_2024_stride20.parquet`)
- Confusion: "Which file should I use?"
- Filtered data before analysis (removed Asian session prematurely)
- No single source of truth

**Pain Point:** Previous session filtered data to 07:00-17:00 GMT **before** discovering if other sessions were profitable. Can't discover best sessions if you remove them first!

## Solution

**After:**
- `data/config.yaml` = Single source of truth
- All scripts read from config (no hardcoded paths)
- `data/raw/` = Original data (24h, all sessions)
- `data/processed/` = Derived datasets (by_session, by_period, experimental)
- `data/versions/` = Backups
- Clear documentation (README.md, QUICK_START.md)

## Migration Steps Completed

### 1. Created Directory Structure ✅

```
data/
├── raw/                  # Original data
├── processed/
│   ├── by_session/       # For empirical analysis
│   ├── by_period/        # Monthly/yearly subsets
│   └── experimental/     # Temporary tests
└── versions/             # Backups
```

### 2. Created config.yaml ✅

Single source of truth with:
- Active dataset path and stats
- Session coverage breakdown
- Quality metrics
- Generation settings
- Session definitions (for future filtering)

### 3. Created Metadata ✅

`data/raw/*.parquet.metadata.json` for each dataset:
- Generation info
- Statistics
- Session coverage
- Top trading days
- Data quality notes

### 4. Updated Scripts ✅

- `check_data_quality.py` → Reads from config.yaml
- `nautilus_gold_scalper/scripts/run_backtest.py` → Reads from config.yaml
- Both scripts validate config exists before running

### 5. Created Tools ✅

- `scripts/validate_data_structure.py` → Validates entire setup
- `scripts/generate_session_datasets.py` → Creates by_session/ files
- `data/.gitignore` → Ignores large files, keeps config

### 6. Documentation ✅

- `data/README.md` → Complete guide (philosophy, structure, workflows)
- `data/QUICK_START.md` → Quick reference for common tasks
- `data/MIGRATION.md` → This file

## File Mapping

### Old → New

| Old Location | New Location | Status |
|-------------|-------------|--------|
| `data/ticks/xauusd_2020_2024_stride20.parquet` | `data/raw/xauusd_2020_2024_stride20_full.parquet` | ✅ Copied |
| `data/ticks/filtered/xauusd_*.parquet` | ❌ Deleted | Experimental filter (not needed) |
| Hardcoded in scripts | `config.yaml` active_dataset.path | ✅ Centralized |

### Old Scripts → Updated

| Script | Change |
|--------|--------|
| `check_data_quality.py` | Now reads `config["active_dataset"]["path"]` |
| `run_backtest.py` | Now reads `config["active_dataset"]["path"]` |

## Validation

```bash
python scripts/validate_data_structure.py
```

**Result:** `[OK] ALL VALIDATIONS PASSED!`

## Rollback Plan (If Needed)

If issues arise:

1. Restore old structure:
   ```bash
   cp data/raw/xauusd_2020_2024_stride20_full.parquet data/ticks/xauusd_2020_2024_stride20.parquet
   ```

2. Revert script changes:
   ```bash
   git checkout HEAD -- check_data_quality.py
   git checkout HEAD -- nautilus_gold_scalper/scripts/run_backtest.py
   ```

3. System will work as before (but without improvements)

## Benefits

### For Users

- ✅ No confusion about which file to use
- ✅ One command to check active dataset: `cat data/config.yaml`
- ✅ Scripts "just work" (read config automatically)
- ✅ Can switch datasets by editing 1 line in config

### For Developers

- ✅ No hardcoded paths in code
- ✅ Clear structure (raw vs processed vs experimental)
- ✅ Metadata available for every dataset
- ✅ Validation tool catches inconsistencies
- ✅ Easy to add new datasets (follow pattern)

### For Analysis

- ✅ Keep ALL sessions (24h) in raw data
- ✅ Generate session-specific datasets on demand
- ✅ Discover empirically which sessions work (not assumptions)
- ✅ Compare results, THEN filter based on evidence

## Next Steps

### Immediate (Cleanup)

- [ ] Review `data/ticks/` for files to archive
- [ ] Move old CSV files to `data/versions/` if needed
- [ ] Update any remaining scripts that might use old paths

### Short Term (Session Analysis)

1. Generate session datasets:
   ```bash
   python scripts/generate_session_datasets.py
   ```

2. Backtest each session (Nov 2024):
   ```bash
   # Asian
   # (edit config.yaml: path = processed/by_session/..._asian.parquet)
   python scripts/run_backtest.py --start 2024-11-01 --end 2024-11-30
   
   # London
   # (edit config.yaml: path = processed/by_session/..._london.parquet)
   python scripts/run_backtest.py --start 2024-11-01 --end 2024-11-30
   
   # ... repeat for all sessions
   ```

3. Compare results:
   - Win rate per session
   - Sharpe ratio
   - Profit factor
   - Number of trades
   - Drawdown

4. Update config.yaml:
   ```yaml
   session_performance:
     analysis_status: "COMPLETED"
     best_sessions: ["overlap", "ny"]  # Based on data
     worst_sessions: ["asian"]         # Based on data
   ```

5. Apply session filter in strategy (evidence-based, not assumption)

### Long Term

- [ ] Generate stride5, stride10 versions for higher density
- [ ] Create by_period datasets for specific months
- [ ] Document session analysis results in DOCS/
- [ ] Consider automated data quality checks (CI/CD)

## Lessons Learned

### ❌ Mistake: Filter Before Analysis

**What happened:** Filtered data to 07:00-17:00 GMT assuming Asian + Late NY sessions were bad.

**Why bad:** Can't discover if those sessions are profitable if you remove them first!

**Fix:** Keep full 24h data in `raw/`, analyze each session empirically, THEN filter based on evidence.

### ✅ Improvement: Empirical Session Selection

**Old:** "Everyone says Asian session is bad" → Filter it out → Never test it

**New:** Keep all data → Test Asian session → Measure results → Make data-driven decision

**Result:** Might discover Asian session actually works well (less HFT competition? Different patterns?)

## Questions & Answers

**Q: Why not just use `data/ticks/` as before?**

A: Lacked organization. Mixing raw data, processed data, experiments in one directory caused confusion. New structure is clear: raw = original, processed = derived, experimental = temporary.

**Q: Why config.yaml instead of environment variables?**

A: Config.yaml is:
- Version controlled (git)
- Human readable
- Contains rich metadata (not just path)
- Easy to validate
- Supports complex structures (session definitions, etc.)

**Q: Can I still use old structure?**

A: Old parquet still exists in `data/ticks/` (not deleted), but scripts now use config.yaml. Recommend migrating fully to avoid confusion.

**Q: What if I want different dataset?**

A: Edit `data/config.yaml`:
```yaml
active_dataset:
  path: "data/processed/by_session/xauusd_2020_2024_stride20_london.parquet"
```

All scripts will automatically use new dataset.

## Contact

For issues with new structure:
1. Run `python scripts/validate_data_structure.py`
2. Check `data/README.md` for documentation
3. Review this MIGRATION.md for changes

## Changelog

- **2025-12-08**: Initial migration completed
  - Created directory structure
  - Created config.yaml
  - Updated scripts (check_data_quality.py, run_backtest.py)
  - Created validation + generation tools
  - Created documentation (README.md, QUICK_START.md, MIGRATION.md)
  - Validation: ALL PASSED ✅
