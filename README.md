# EA_SCALPER_XAUUSD v2.2

**Automated Gold Trading System for Apex Trader Funding**

> After many requests and messages, I've made this repository public again. This is a personal project that I've been developing to automate gold (XAUUSD) trading with a focus on prop firm challenges.

---

## Overview

EA_SCALPER_XAUUSD is an advanced Expert Advisor (trading robot) designed specifically for **XAUUSD (Gold)** scalping on MetaTrader 5. The system is optimized for **Apex Trader Funding** challenges, with strict risk management and compliance with prop firm rules.

### Key Features

- **Smart Money Concepts (SMC)** - Order blocks, liquidity sweeps, fair value gaps
- **Multi-Timeframe Analysis** - M1 execution with H1/H4 bias confirmation
- **Session-Based Trading** - Optimized for London and New York sessions
- **Regime Detection** - ML-powered market regime classification (Trend/Range/Volatile)
- **Advanced Risk Management** - Trailing drawdown protection, position sizing, circuit breakers
- **ONNX Integration** - Machine learning models for direction prediction

---

## Trading Strategies

### 1. SMC Scalping (Primary)
- Identifies institutional order blocks and liquidity zones
- Trades retracements to order blocks with confluence
- Targets 1:2 to 1:3 risk-reward ratios

### 2. Session Breakout
- Captures London and New York session volatility
- Breakout entries with momentum confirmation
- Time-based position management

### 3. Regime-Adaptive
- Uses ML to classify current market regime
- Adjusts strategy parameters based on regime
- Avoids trading in unfavorable conditions (random walk)

---

## Architecture

```
EA_SCALPER_XAUUSD/
├── MQL5/                    # MetaTrader 5 source code
│   ├── Experts/             # Main EA files
│   ├── Include/EA_SCALPER/  # Modular components
│   └── Scripts/             # Utility scripts
├── models/                  # ONNX ML models
├── scripts/                 # Python analysis tools
│   ├── oracle/              # Backtest validation (WFA, Monte Carlo)
│   └── backtest/            # Strategy testing
├── nautilus_gold_scalper/   # NautilusTrader migration (Python)
└── DOCS/                    # Documentation
```

### Core Modules

| Module | Description |
|--------|-------------|
| `CRegimeDetector` | ML-based market regime classification |
| `CSessionFilter` | Trading session management |
| `COrderBlockDetector` | SMC order block identification |
| `CRiskManager` | Position sizing and drawdown protection |
| `CTradeManager` | Order execution and management |
| `CMTFManager` | Multi-timeframe data aggregation |

---

## Apex Trader Funding Compliance

This EA is specifically designed for Apex Trader Funding rules:

| Rule | Implementation |
|------|----------------|
| **Trailing Drawdown (10%)** | Real-time high-water mark tracking |
| **No Overnight Positions** | Auto-close before 4:59 PM ET |
| **Consistency Rule (30%)** | Daily profit cap monitoring |
| **Risk per Trade** | 0.5-1% maximum |

---

## Requirements

- **Platform**: MetaTrader 5 (FTMO/Apex terminal recommended)
- **Broker**: Any with XAUUSD and low spreads
- **Account**: $50,000+ recommended for proper position sizing
- **VPS**: Recommended for 24/5 operation

---

## Disclaimer

This is a **personal project** shared for educational purposes. Trading involves substantial risk of loss and is not suitable for all investors.

- Past performance does not guarantee future results
- Use at your own risk
- Always test on demo accounts first
- I am not responsible for any financial losses

---

## Status

**Current Version**: v2.2  
**Status**: Active Development  
**Target**: Apex Trader Funding Challenges

---

## Contact

This repository is maintained by **Franco** as a personal trading automation project.

If you find this useful, feel free to star the repo!

---

*"The market is never wrong. Opinions are."* - Jesse Livermore
