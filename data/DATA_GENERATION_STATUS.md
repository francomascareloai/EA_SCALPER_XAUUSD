# Data Generation Status - 2025-12-08

## ✅ Completed

### Structure
- ✅ `data/config.yaml` - Single source of truth
- ✅ `data/raw/` - Original datasets
- ✅ `data/processed/` - Session/period datasets  
- ✅ `data/versions/` - Backups
- ✅ All scripts updated to read from config.yaml

### Scripts
- ✅ **`scripts/generate_parquet.py`** - MASTER script (unified, tested)
- ✅ Deleted 4 old script versions
- ✅ Factory CLI compatible (no input() prompts)
- ✅ Temp chunks strategy (safe, resumable)

### Generated Datasets

#### 2003-2025 Data (22 years)

**Stride 20** ✅ **COMPLETE**
```
File:   xauusd_2003_2025_stride20_full.parquet
Size:   353.2 MB
Ticks:  32,729,302
Period: 2003-05-05 to 2025-11-28
Days:   5,850
Avg:    5,594 ticks/day
Time:   52 minutes
Status: VALIDATED ✅
```

**Stride 10** ⏳ **PENDING**
```
Expected size:   ~700 MB
Expected ticks:  ~65M
Expected time:   50-60 min
Status: Ready to generate
```

**Stride 5** ⏳ **PENDING**
```
Expected size:   ~1.4 GB
Expected ticks:  ~130M
Expected time:   50-60 min
Status: Ready to generate
```

#### 2020-2024 Data (5 years) - BACKUP

**Stride 20** ✅ **ARCHIVED**
```
File:   xauusd_2020_2024_stride20_full.parquet (BACKUP in versions/)
Size:   281 MB
Ticks:  25,522,123
Status: Backup mantido para comparação
```

## 🎯 Current Active Dataset

```yaml
active_dataset:
  name: xauusd_2003_2025_stride20_full
  path: data/raw/xauusd_2003_2025_stride20_full.parquet
  ticks: 32,729,302
  period: 2003-2025 (22 years)
  stride: 20
```

## 🚀 Next Steps

### 1. Generate Stride 10 (High Priority)
```bash
python scripts/generate_parquet.py --strides 10 --force
```
**Estimated time:** 50-60 minutes  
**Output:** 65M ticks, 700MB  
**Use case:** Higher density backtesting

### 2. Generate Stride 5 (Optional)
```bash
python scripts/generate_parquet.py --strides 5 --force
```
**Estimated time:** 50-60 minutes  
**Output:** 130M ticks, 1.4GB  
**Use case:** Ultra-high density (stress testing)

### 3. Generate Session Datasets
```bash
python scripts/generate_session_datasets.py
```
**Output:** 5 session-specific parquets (Asian, London, Overlap, NY, Late NY)  
**Use case:** Empirical session analysis

### 4. Test Backtests
```bash
# Test with stride 20 (fast)
python scripts/run_backtest.py --start 2024-11-01 --end 2024-11-30

# Compare with stride 10 (higher density)
# (update config.yaml to point to stride 10)
python scripts/run_backtest.py --start 2024-11-01 --end 2024-11-30
```

## 📊 Why Multiple Strides?

**Stride 20** (353MB, 32M ticks)
- ✅ Fast backtesting (5 years in minutes)
- ✅ Good for strategy development
- ✅ Ideal for WFA (multiple folds)

**Stride 10** (700MB, 65M ticks)
- ✅ 2x more density
- ✅ Better for final validation
- ✅ More realistic tick-by-tick

**Stride 5** (1.4GB, 130M ticks)
- ✅ Ultra-high density
- ✅ Stress testing
- ✅ Production validation

## 🔧 Command Reference

### Generate Data
```bash
# Single stride
python scripts/generate_parquet.py --strides 10 --force

# Multiple strides (sequential)
python scripts/generate_parquet.py --strides 20 10 5 --force

# Resume after crash
python scripts/generate_parquet.py --strides 10 --resume

# Test mode (1M rows only, ~2 min)
python scripts/generate_parquet.py --strides 20 --test --force
```

### Validate
```bash
# Check active dataset quality
python check_data_quality.py

# Validate structure
python scripts/validate_data_structure.py
```

### Switch Datasets
```bash
# Edit data/config.yaml:
active_dataset:
  path: "data/raw/xauusd_2003_2025_stride10_full.parquet"  # Change here
```

## 📝 Notes

### CSV Source
- **File:** `Python_Agent_Hub/ml_pipeline/data/CSV_2003-2025XAUUSD_ftmo_all-TICK-No Session.csv`
- **Size:** 30.6 GB
- **Rows:** ~655 million
- **Period:** 2003-2025 (22 years)
- **Status:** VALIDATED ✅

### Performance
- **Stride 20:** 52 min (proven)
- **Stride 10:** ~50-60 min (estimated)
- **Stride 5:** ~50-60 min (estimated)
- **Strategy:** Temp chunks (safe, resumable)

### Critical Fix
**Problem:** Scripts travavam em 0% (input() não funciona no Factory CLI)  
**Solution:** Removido todos input() prompts, agora 100% automatizado com flags CLI  
**Result:** Script MASTER testado e validado ✅

## 🎓 Lessons Learned

1. **input() não funciona no Factory CLI** → Use flags CLI (--force, --resume)
2. **Append parquet é lento** → Use temp chunks + concatenação final
3. **CSV de 30GB é OK** → Pandas lida bem com chunking
4. **Teste antes de rodar 50 min** → Use --test mode
5. **Um script master > múltiplos scripts** → Mais fácil manter

## 🔗 Related Files

- **Config:** `data/config.yaml`
- **README:** `data/README.md`
- **Quick Start:** `data/QUICK_START.md`
- **Migration:** `data/MIGRATION.md`
- **Generator:** `scripts/generate_parquet.py` (MASTER)
- **Validator:** `scripts/validate_data_structure.py`

## ⏭️ Immediate Next Action

**Rodar stride 10 completo:**
```bash
python scripts/generate_parquet.py --strides 10 --force
```

Deixar rodar ~50-60 min, depois validar com `check_data_quality.py`.
