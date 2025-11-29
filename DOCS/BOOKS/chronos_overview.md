# Chronos - Amazon's Time Series Foundation Models

Repository: amazon-science/chronos-forecasting
Focus: Pretrained Foundation Models for Zero-Shot Time Series Forecasting

## Overview

Chronos is a family of **pretrained time series forecasting models** from Amazon Science.
It transforms time series into sequences of tokens and uses language model architectures.

## Model Types

### 1. Chronos-2 (Latest - Oct 2025)
- **Zero-shot** support for univariate, multivariate, and covariate-informed forecasting
- State-of-the-art on fev-bench and GIFT-Eval benchmarks
- 90%+ win rate against Chronos-Bolt in head-to-head

### 2. Chronos-Bolt (Fast)
- Patch-based variant with direct multi-step forecasting
- **250x faster** than original Chronos
- **20x more memory efficient**
- Uses patches of multiple observations

### 3. Original Chronos
- Language model architecture (T5-based)
- Scaling + quantization to tokens
- Cross-entropy loss training
- Probabilistic forecasts via sampling

## Available Models

| Model | Parameters |
|-------|------------|
| chronos-2 | 120M |
| chronos-bolt-tiny | 9M |
| chronos-bolt-mini | 21M |
| chronos-bolt-small | 48M |
| chronos-bolt-base | 205M |
| chronos-t5-tiny | 8M |
| chronos-t5-mini | 20M |
| chronos-t5-small | 46M |
| chronos-t5-base | 200M |
| chronos-t5-large | 710M |

## Installation

```bash
pip install chronos-forecasting
```

## Usage Example (Chronos-2)

```python
import pandas as pd
from chronos import Chronos2Pipeline

pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")

# Load historical data
context_df = pd.read_parquet("train.parquet")
future_df = pd.read_parquet("test.parquet").drop(columns="target")

# Generate forecasts with covariates
pred_df = pipeline.predict_df(
    context_df,
    future_df=future_df,
    prediction_length=24,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column="id",
    timestamp_column="timestamp",
    target="target",
)
```

## Key Features

1. **Zero-Shot Forecasting**: No fine-tuning needed
2. **Probabilistic Outputs**: Returns quantile forecasts
3. **Multivariate Support**: Handle multiple time series
4. **Covariate Support**: Include exogenous variables
5. **Pre-trained**: Trained on diverse time series datasets

## Why This Matters for Trading

- **No domain adaptation needed**: Works out-of-box on financial data
- **Uncertainty quantification**: Probabilistic forecasts for risk management
- **Fast inference**: Chronos-Bolt suitable for real-time trading
- **State-of-the-art**: Beats traditional statistical methods

## Citation

```bibtex
@article{ansari2024chronos,
  title={Chronos: Learning the Language of Time Series},
  author={Ansari et al.},
  journal={Transactions on Machine Learning Research},
  year={2024}
}
```
