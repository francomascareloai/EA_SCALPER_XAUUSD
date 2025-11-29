# MLFinLab - Machine Learning Financial Laboratory
# Hudson & Thames - Implementation of "Advances in Financial ML" by López de Prado

Repository: hudson-and-thames/mlfinlab
Focus: Quantitative Finance ML, Triple Barrier, Fractional Differentiation

## Overview

MlFinlab python library is a perfect toolbox that every financial machine learning researcher needs.
It covers every step of the ML strategy creation, starting from data structures generation and finishing with backtest statistics.

## Key Modules (López de Prado Implementations)

### 1. Data Structures
- Dollar bars, Volume bars, Tick bars
- Information-driven bars (imbalance bars, run bars)

### 2. Labeling (CRITICAL for Trading ML)
- **Triple Barrier Labeling** - The correct way to label financial data
- Fixed-time horizon labeling
- Meta-labeling (betting sizing)

### 3. Sampling
- Sequential bootstrapping (handles overlapping labels)
- Concurrent labels handling
- Uniqueness and average uniqueness

### 4. Feature Engineering
- **Fractional Differentiation** - Preserves memory while achieving stationarity
- Structural breaks (CUSUM, SADF)

### 5. Cross-Validation
- **Purged K-Fold CV** - Prevents information leakage
- Embargo periods
- Combinatorial purged CV

### 6. Bet Sizing
- Kelly criterion implementation
- Sizing based on classifier confidence

### 7. Feature Importance
- Mean Decrease Impurity (MDI)
- Mean Decrease Accuracy (MDA)
- Single Feature Importance (SFI)

### 8. Backtest Overfitting Tools
- Deflated Sharpe Ratio
- Probabilistic Sharpe Ratio
- Combinatorial Symmetric Cross Validation (CSCV)

### 9. Synthetic Data Generation
- Correlated random walks
- Microstructural features

### 10. Networks
- Minimum Spanning Trees
- Hierarchical clustering

## Why This Matters for Trading

1. **Triple Barrier Labeling**: Eliminates lookahead bias (90% of retail backtests are invalid without this)
2. **Purged K-Fold**: Prevents overfitting from temporal leakage
3. **Fractional Differentiation**: Makes prices stationary while keeping predictive power
4. **Meta-Labeling**: Separate signal generation from bet sizing

## Code Example: Triple Barrier Method

```python
from mlfinlab.labeling import get_events, add_vertical_barrier

# Define barriers
vertical_barriers = add_vertical_barrier(close, t_events, num_days=5)
events = get_events(
    close=close,
    t_events=t_events,
    pt_sl=[1, 1],  # profit taking and stop loss multipliers
    target=daily_vol,
    min_ret=0.01,
    num_threads=1,
    vertical_barrier_times=vertical_barriers
)
```

## Code Example: Fractional Differentiation

```python
from mlfinlab.features import frac_diff_ffd

# Apply fractional differentiation
frac_diff_prices = frac_diff_ffd(close_prices, d=0.4, thres=1e-5)
```

## License
Commercial license required from Hudson & Thames
