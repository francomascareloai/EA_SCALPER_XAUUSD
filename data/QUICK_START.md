# Data Directory - Quick Start Guide

## 🎯 Purpose

Eliminate confusion about which dataset to use. **ONE config file controls everything.**

## ⚡ Quick Reference

### For Users (Running Analysis)

```bash
# 1. Check current dataset
cat data/config.yaml | grep "name:"

# 2. Run validation
python check_data_quality.py

# 3. Run backtest (auto-uses config.yaml)
python scripts/run_backtest.py --start 2024-11-01 --end 2024-11-30
```

**That's it!** Scripts automatically read from `data/config.yaml`.

### For Developers (Writing Scripts)

```python
import yaml
from pathlib import Path

# Load config (ALWAYS start with this)
config = yaml.safe_load(open("data/config.yaml"))

# Get active dataset
data_path = Path(config["active_dataset"]["path"])
df = pd.read_parquet(data_path)

# That's it!
```

## 📂 Directory Structure

```
data/
├── config.yaml           ⭐ Single source of truth
├── README.md             📘 Full documentation
├── QUICK_START.md        ⚡ This file
│
├── raw/                  🔒 Original data (never modify)
│   ├── xauusd_2020_2024_stride20_full.parquet
│   └── *.metadata.json
│
├── processed/            🔧 Processed data
│   ├── by_session/       (for session analysis)
│   ├── by_period/        (monthly/yearly subsets)
│   └── experimental/     (safe to delete)
│
└── versions/             📦 Backups
```

## 🔄 Common Tasks

### Change Active Dataset

```yaml
# Edit data/config.yaml
active_dataset:
  path: "data/processed/by_session/xauusd_2020_2024_stride20_london.parquet"
  # Update other fields...
```

### Validate Everything is OK

```bash
python scripts/validate_data_structure.py
```

**Expected output:** `[OK] ALL VALIDATIONS PASSED!`

### Generate Session Datasets

```bash
# Split full data into sessions (Asian, London, Overlap, NY, Late NY)
python scripts/generate_session_datasets.py
```

**Output:** `data/processed/by_session/*.parquet`

### Analyze Data Quality

```bash
python check_data_quality.py
```

**Shows:** Tick distribution, session coverage, density, recommendations.

## 🚨 Common Mistakes (DON'T)

❌ **Hardcode paths in scripts:**
```python
# BAD
df = pd.read_parquet("data/ticks/xauusd_2020_2024_stride20.parquet")
```

✅ **Read from config:**
```python
# GOOD
config = yaml.safe_load(open("data/config.yaml"))
df = pd.read_parquet(config["active_dataset"]["path"])
```

---

❌ **Filter data before analysis:**
```python
# BAD - removes data BEFORE discovering if it's useful
df = df[df['hour'].between(7, 17)]  # Assumes Asian session is bad
```

✅ **Keep full data, analyze empirically:**
```python
# GOOD - discover which sessions work THEN filter
# 1. Generate session datasets
# 2. Backtest each session
# 3. Compare results
# 4. THEN filter based on evidence
```

---

❌ **Modify files in `data/raw/`:**
```bash
# BAD - corrupts original data
# (editing parquet file directly)
```

✅ **Regenerate instead:**
```bash
# GOOD - create new version
python scripts/generate_parquet.py --output data/raw/new_version.parquet
# Update config.yaml to point to new file
# Move old to data/versions/
```

## 🎓 Philosophy

**Old Way (Assumption-Based):**
- "Asian session is bad" → Filter it out → Never know if assumption was wrong

**New Way (Empirical):**
- Keep ALL data → Test each session → Measure results → Filter based on evidence

**Result:** Data-driven decisions, not gut feelings.

## 📚 Full Documentation

- **Complete guide:** `data/README.md`
- **Config reference:** `data/config.yaml` (has inline comments)
- **Validation:** `python scripts/validate_data_structure.py`

## 🛠️ Troubleshooting

### "Script can't find data file"

```bash
# Check config
cat data/config.yaml | grep "path:"

# Verify file exists
ls data/raw/*.parquet

# Re-run validation
python scripts/validate_data_structure.py
```

### "Data stats don't match config"

```bash
# Regenerate or fix config.yaml
# Run validation to see exact mismatch:
python scripts/validate_data_structure.py
```

### "Need different stride/period"

```bash
# Generate new dataset
python scripts/generate_parquet.py --stride 10

# Update config.yaml to point to new file

# Validate
python scripts/validate_data_structure.py
```

## 📊 Workflow Example: Session Analysis

```bash
# 1. Generate session datasets
python scripts/generate_session_datasets.py
# Output: data/processed/by_session/*.parquet

# 2. Test Asian session
# Edit data/config.yaml:
#   path: "data/processed/by_session/xauusd_2020_2024_stride20_asian.parquet"
python scripts/run_backtest.py --start 2024-11-01 --end 2024-11-30
# Note results: Sharpe, win rate, trades

# 3. Test London session
# Edit data/config.yaml:
#   path: "data/processed/by_session/xauusd_2020_2024_stride20_london.parquet"
python scripts/run_backtest.py --start 2024-11-01 --end 2024-11-30
# Note results

# 4. Repeat for all sessions

# 5. Compare results → Update config.yaml with best sessions

# 6. Restore full dataset, apply session filter in strategy based on evidence
# Edit data/config.yaml:
#   path: "data/raw/xauusd_2020_2024_stride20_full.parquet"
#   session_performance:
#     best_sessions: ["overlap", "ny"]  # Based on empirical results
```

## ✅ Checklist: New Developer Onboarding

- [ ] Read `data/README.md`
- [ ] Run `python scripts/validate_data_structure.py`
- [ ] Check `data/config.yaml` to see active dataset
- [ ] Run `python check_data_quality.py` to understand data
- [ ] Update any scripts to use `config.yaml` (not hardcoded paths)
- [ ] Test backtest with: `python scripts/run_backtest.py --start 2024-11-01 --end 2024-11-30`

**Time:** ~10 minutes

**Result:** Full understanding of data structure + verified working setup.

## 🎯 Remember

**Golden Rule:** All scripts read `data/config.yaml`. No hardcoded paths.

**Why?** Changing datasets = edit 1 line in config.yaml, not 20 scripts.

**When in doubt:** Run `python scripts/validate_data_structure.py`
