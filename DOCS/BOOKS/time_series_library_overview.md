# Time Series Library (TSLib) - Tsinghua University

Repository: thuml/Time-Series-Library
Focus: State-of-the-art Deep Learning Models for Time Series Analysis

## Overview

TSLib is an open-source library from Tsinghua University for deep time series analysis.
Covers five mainstream tasks: long-term forecasting, short-term forecasting, imputation, anomaly detection, classification.

## Leaderboard (March 2024)

### Long-term Forecasting (Look-Back-96)
1. **TimeXer** - Best for exogenous variables
2. **iTransformer** - Inverted attention mechanism
3. **TimeMixer** - Multiscale mixing

### Short-term Forecasting
1. **TimesNet** - Temporal 2D-Variation
2. **Non-stationary Transformer**
3. **FEDformer** - Frequency domain

### Anomaly Detection
1. **TimesNet**
2. **FEDformer**
3. **Autoformer**

## Implemented Models (All Available)

### State-of-the-Art Transformers
- **TimeXer** (NeurIPS 2024) - Forecasting with exogenous variables
- **TimeMixer** (ICLR 2024) - Decomposable multiscale mixing
- **iTransformer** (ICLR 2024) - Inverted transformers
- **PatchTST** (ICLR 2023) - Patch-based, 64 words approach
- **TimesNet** (ICLR 2023) - Temporal 2D-Variation

### Classic Transformers
- **FEDformer** (ICML 2022) - Frequency enhanced decomposed
- **Autoformer** (NeurIPS 2021) - Decomposition with auto-correlation
- **Informer** (AAAI 2021) - Efficient long sequence
- **Non-stationary Transformer** (NeurIPS 2022)

### Linear Models
- **DLinear** (AAAI 2023) - Simple but effective
- **TiDE** - Time-series Dense Encoder
- **TSMixer** - All-MLP architecture

### New Additions (2024-2025)
- **TimeFilter** (ICML 2025) - Patch-specific graph filtration
- **KAN-AD** (ICML 2025) - Kolmogorov-Arnold Networks
- **WPMixer** (AAAI 2025) - Multi-resolution mixing
- **Mamba** - State space model
- **SegRNN** - Segment recurrent

### Large Time Series Models (Zero-Shot)
- **Chronos2** (arXiv 2025)
- **TiRex** (NeurIPS 2025)
- **Sundial** (ICML 2025)
- **Time-MoE** (ICLR 2025)
- **Moirai** (ICML 2024)
- **TimesFM** (ICML 2024)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Long-term forecasting
bash ./scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh

# Short-term forecasting
bash ./scripts/short_term_forecast/TimesNet_M4.sh

# Anomaly detection
bash ./scripts/anomaly_detection/PSM/TimesNet.sh
```

## Developing Custom Models

1. Add model to `./models/`
2. Include in `Exp_Basic.model_dict`
3. Create scripts in `./scripts/`

## Key Insights for Trading

### Best Models by Use Case

| Use Case | Recommended Model |
|----------|-------------------|
| Long-term forecasting | TimeXer, iTransformer |
| With external features | TimeXer (designed for exogenous) |
| Real-time (fast) | DLinear, PatchTST |
| Anomaly detection | TimesNet, FEDformer |
| Zero-shot | Chronos2, TimesFM |

### Important Findings

1. **Linear models often outperform transformers** on pure forecasting
2. **PatchTST**: Treating time series as 64 "words" works surprisingly well
3. **Look-back length matters**: 96 vs searching yields different rankings
4. **Exogenous variables**: TimeXer specifically designed for this

## Citation

```bibtex
@inproceedings{wu2023timesnet,
  title={TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis},
  author={Haixu Wu et al.},
  booktitle={ICLR},
  year={2023}
}
```
