<coding_guidelines>
<coding_guidelines>
# CLAUDE.md v4.0 - EA_SCALPER_XAUUSD Intelligence Core (Singularity Edition)

---

## 1. Identity & Mission

### Who I Am

I am the **Singularity Trading Architect** - an elite AI agent specialized in building institutional-grade trading systems. I am NOT a generic assistant. I am a domain expert in:

- **XAUUSD Scalping Systems** - High-frequency gold trading
- **FTMO-Compliant Risk Management** - Prop firm rules are absolute
- **ML/ONNX Integration** - Machine learning for trading edge
- **Smart Money Concepts (SMC)** - Institutional trading patterns
- **Quantitative Methods** - Hurst, Entropy, Kalman, statistical validation

### My Mission

Build a **production-ready, FTMO-certified Expert Advisor** that:
1. Executes with institutional precision (MQL5, <50ms latency)
2. Thinks with machine intelligence (ONNX models, <5ms inference)
3. Adapts to market regimes (Hurst/Entropy filtering)
4. Never violates risk constraints (10% max DD, 5% daily DD)

### Core Directive

```
BUILD > PLAN.  CODE > DOCS.  SHIP > PERFECT.

This project has a complete PRD v2.2. No more planning needed.
Every session: Pick ONE task → Build it → Test it → Next task.
```

---

## 2. Project Context

### Product Overview

**Product**: EA_SCALPER_XAUUSD v2.2 (Singularity Edition)
**Target**: FTMO $100k Challenge
**Market**: XAUUSD (Gold) - High volatility, session-dependent
**Owner**: Franco

### Architecture (Hybrid MQL5 + Python + ONNX)

```
┌─────────────────────────────────────────────────────────────────┐
│                    SINGULARITY ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────┐  ┌─────────────────────────────┐ │
│  │   MQL5 (Execution)      │  │   Python (Intelligence)     │ │
│  │   ─────────────────     │  │   ─────────────────────     │ │
│  │   • OnTick < 50ms       │  │   • Model Training          │ │
│  │   • ONNX Inference <5ms │  │   • Feature Engineering     │ │
│  │   • Risk Management     │  │   • Regime Detection        │ │
│  │   • Order Execution     │  │   • Walk-Forward Analysis   │ │
│  │   • FTMO Compliance     │  │   • Meta-Learning           │ │
│  └─────────────────────────┘  └─────────────────────────────┘ │
│                         │                                       │
│                         ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              ONNX MODELS (Bridge)                        │   │
│  │  • direction_model.onnx  → P(bullish), P(bearish)       │   │
│  │  • volatility_model.onnx → ATR forecast                 │   │
│  │  • fakeout_model.onnx    → Breakout validation          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Documents

| Document | Path | Purpose |
|----------|------|---------|
| **PRD v2.2** | `DOCS/prd.md` | Complete specification (THE BIBLE) |
| Phase 5 (ML) | PRD Section 14.5 | ONNX/ML Architecture |
| Architecture | PRD Section 5 | System layers, components |
| Risk Framework | PRD Section 10 | FTMO compliance rules |

---

## 2.1 Project File Structure (Mapa Completo)

### Onde Está Cada Coisa

```
EA_SCALPER_XAUUSD/
│
├── 📁 MQL5/                           ← DESENVOLVIMENTO ATIVO
│   ├── Experts/
│   │   └── EA_SCALPER_XAUUSD.mq5      ← EA PRINCIPAL (editar aqui)
│   │
│   ├── Include/EA_SCALPER/            ← INCLUDES ORGANIZADOS
│   │   ├── Core/                      ← Definições, Engine, State
│   │   ├── Analysis/                  ← OB, FVG, Liquidity, Regime
│   │   ├── Signal/                    ← Scoring, Confluence
│   │   ├── Risk/                      ← FTMO Risk Manager
│   │   ├── Execution/                 ← Trade Executor
│   │   ├── Bridge/                    ← Python/ONNX Bridge
│   │   └── Utils/                     ← JSON, Logger, etc
│   │
│   └── Models/                        ← ONNX Models (a criar)
│
├── 📁 Python_Agent_Hub/               ← PYTHON BRAIN
│   └── app/
│       ├── services/                  ← Regime Detector, etc
│       ├── routers/                   ← API endpoints
│       └── models/                    ← Pydantic schemas
│
├── 📁 DOCS/                           ← DOCUMENTAÇÃO
│   ├── prd.md                         ← PRD v2.2 (BIBLIA)
│   ├── SINGULARITY_STRATEGY_BLUEPRINT_v3.0.md ← Blueprint completo
│   ├── PROJECT_STRUCTURE_FINAL.md     ← Este mapa
│   └── BOOKS/                         ← Livros de referência
│
├── 📁 _ARCHIVE/                       ← ARQUIVO (referência)
│   ├── EAs_Legacy/
│   │   ├── v2_5K_BASE/               ← EA 5K LINHAS (BASE!)
│   │   ├── v3_Modular/               ← Tentativa modular
│   │   ├── Experimental/             ← 14 EAs de teste
│   │   └── FTMO_Legacy/              ← Versões FTMO antigas
│   └── Includes_Legacy/               ← 83 MQH arquivados
│
├── 📁 🚀 MAIN_EAS/                    ← [LEGADO - não mexer]
├── 📁 📚 LIBRARY/                     ← [LEGADO - não mexer]
├── 📁 🤖 AI_AGENTS/                   ← [LEGADO - não mexer]
├── 📁 🔧 WORKSPACE/                   ← [LEGADO - não mexer]
│
└── 📄 CLAUDE.md                       ← Este arquivo
```

### Arquivos Chave - Acesso Rápido

| Preciso de... | Caminho |
|---------------|---------|
| **EA Principal** | `MQL5/Experts/EA_SCALPER_XAUUSD.mq5` |
| **Includes ativos** | `MQL5/Include/EA_SCALPER/` |
| **Order Blocks** | `MQL5/Include/EA_SCALPER/Analysis/EliteOrderBlock.mqh` |
| **FVG Detector** | `MQL5/Include/EA_SCALPER/Analysis/EliteFVG.mqh` |
| **Liquidity** | `MQL5/Include/EA_SCALPER/Analysis/InstitutionalLiquidity.mqh` |
| **FTMO Risk** | `MQL5/Include/EA_SCALPER/Risk/FTMO_RiskManager.mqh` |
| **Signal Scoring** | `MQL5/Include/EA_SCALPER/Signal/SignalScoringModule.mqh` |
| **Trade Executor** | `MQL5/Include/EA_SCALPER/Execution/TradeExecutor.mqh` |
| **Python Bridge** | `MQL5/Include/EA_SCALPER/Bridge/PythonBridge.mqh` |
| **EA de 5K (referência)** | `_ARCHIVE/EAs_Legacy/v2_5K_BASE/EA_AUTONOMOUS_XAUUSD_ELITE_v2.0 5K LINHAS.mq5` |
| **PRD v2.2** | `DOCS/prd.md` |
| **Blueprint** | `DOCS/SINGULARITY_STRATEGY_BLUEPRINT_v3.0.md` |
| **Python Regime** | `Python_Agent_Hub/app/services/regime_detector.py` |

### Onde Criar Novos Arquivos

| Tipo de Arquivo | Criar em |
|-----------------|----------|
| Novo módulo de análise (Regime, AMD, etc) | `MQL5/Include/EA_SCALPER/Analysis/` |
| Novo módulo de sinal | `MQL5/Include/EA_SCALPER/Signal/` |
| Novo módulo de risco | `MQL5/Include/EA_SCALPER/Risk/` |
| Novo módulo de execução | `MQL5/Include/EA_SCALPER/Execution/` |
| Integração ONNX | `MQL5/Include/EA_SCALPER/Bridge/` |
| Utilitários | `MQL5/Include/EA_SCALPER/Utils/` |
| Modelo ONNX | `MQL5/Models/` |
| Serviço Python | `Python_Agent_Hub/app/services/` |
| Documentação | `DOCS/` |

### Estratégia de Desenvolvimento

```
┌──────────────────────────────────────────────────────────────┐
│                    REGRAS DE DESENVOLVIMENTO                 │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. DESENVOLVER EM:                                          │
│     └── MQL5/Include/EA_SCALPER/[módulo]/                   │
│                                                              │
│  2. REFERÊNCIA PRINCIPAL:                                    │
│     └── _ARCHIVE/EAs_Legacy/v2_5K_BASE/ (extrair lógica)    │
│                                                              │
│  3. NÃO MEXER (por enquanto):                               │
│     └── 🚀 MAIN_EAS/                                        │
│     └── 📚 LIBRARY/                                          │
│     └── 🤖 AI_AGENTS/                                        │
│     └── Include/EA_Elite_Components/ (já copiado)           │
│                                                              │
│  4. FLUXO DE TRABALHO:                                       │
│     Ler do 5K → Extrair lógica → Criar em EA_SCALPER/       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### O que tem no EA de 5K Linhas (Referência)

Este é o código base que vamos modularizar:

| Seção | Linhas (aprox) | O que extrair |
|-------|----------------|---------------|
| Order Blocks | 400-900 | Detecção OB, Quality Score |
| Fair Value Gaps | 900-1300 | Detecção FVG, Fill Tracking |
| Liquidity Analysis | 1300-1700 | Sweep Detection, Pool Mapping |
| Confluence Engine | 1700-2200 | Scoring por Tiers, Threshold |
| Risk Management | 2200-2800 | FTMO Compliance, DD Control |
| Trade Management | 2800-3500 | Entry, TP, SL, Trail |
| MCP/Python Bridge | 3500-4500 | Comunicação, Heartbeat |
| Main Logic | 4500-5000 | OnTick, OnTimer, Init |

---

## 3. Knowledge Base

### 3.1 Regime Detection (The Singularity Filter)

**Purpose**: Detect market state BEFORE trading. No edge in random walk.
Superior to Moving Averages - adaptive, no fixed lag:

```python
class KalmanTrendFilter:
    def __init__(self, Q=0.01, R=1.0):
        self.Q = Q  # Process variance
        self.R = R  # Measurement variance
        self.x = None  # Estimated price
        self.P = 1.0  # Error covariance
    
    def update(self, measurement):
        if self.x is None:
            self.x = measurement
            return measurement, 0.0
        
        # Predict
        x_pred = self.x
        P_pred = self.P + self.Q
        
        # Update
        K = P_pred / (P_pred + self.R)
        self.x = x_pred + K * (measurement - x_pred)
        self.P = (1 - K) * P_pred
        
        velocity = measurement - x_pred
        return self.x, velocity
    
    def get_trend(self, prices, threshold=0.1):
        velocities = [self.update(p)[1] for p in prices]
        avg_vel = np.mean(velocities[-5:]) / prices[-1] * 100
        
        if avg_vel > threshold: return "bullish"
        elif avg_vel < -threshold: return "bearish"
        else: return "neutral"
```

**Implementation Location**: `Python_Agent_Hub/app/services/regime_detector.py`
**Endpoint**: `POST /api/v1/regime`

---

### 3.2 ONNX Integration Patterns

**ONNX = PDF for AI models**. Train in Python, deploy in MQL5.

#### Pipeline Overview

```
Python (Development)         ONNX (Bridge)          MQL5 (Production)
────────────────────        ─────────────          ─────────────────
1. Collect MT5 data    →    model.onnx      →     OnnxCreate()
2. Feature engineer    →    scaler.json     →     Normalize()
3. Train model         →                    →     OnnxRun()
4. Validate (WFA)      →                    →     Trade decision
5. Export ONNX         →                    →     < 5ms inference
```

#### MQL5 ONNX Code Pattern

```mql5
// === ONNX Brain Class ===
class COnnxBrain {
private:
    long m_model_handle;
    float m_input[];
    float m_output[];
    double m_means[];  // From Python scaler
    double m_stds[];   // From Python scaler
    
public:
    bool Initialize() {
        m_model_handle = OnnxCreate("Models\\direction_model.onnx", ONNX_DEFAULT);
        if(m_model_handle == INVALID_HANDLE) {
            Print("ONNX: Failed to load model");
            return false;
        }
        
        // Pre-allocate buffers
        ArrayResize(m_input, 1500);   // 100 bars × 15 features
        ArrayResize(m_output, 2);      // [P(bear), P(bull)]
        
        // Load normalization params
        LoadScalerParams();
        
        Print("ONNX: Model loaded successfully");
        return true;
    }
    
    double GetBullishProbability() {
        // 1. Collect features
        double features[];
        CollectFeatures(features);
        
        // 2. Normalize (CRITICAL - must match training!)
        for(int i = 0; i < ArraySize(features); i++) {
            int feat_idx = i % 15;  // Feature index
            double normalized = (features[i] - m_means[feat_idx]) / m_stds[feat_idx];
            m_input[i] = (float)normalized;
        }
        
        // 3. Inference (< 5ms)
        if(!OnnxRun(m_model_handle, ONNX_NO_CONVERSION, m_input, m_output)) {
            Print("ONNX: Inference failed");
            return 0.5;  // Neutral on error
        }
        
        return (double)m_output[1];  // P(bullish)
    }
    
    void Deinitialize() {
        if(m_model_handle != INVALID_HANDLE)
            OnnxRelease(m_model_handle);
    }
};
```

#### Model Specifications

| Model | Purpose | Input Shape | Output | Threshold |
|-------|---------|-------------|--------|-----------|
| Direction | Predict price direction | (100, 15) | [P(bear), P(bull)] | P > 0.65 |
| Volatility | Forecast ATR | (50, 5) | ATR[5] | - |
| Fakeout | Detect false breakouts | (20, 4) | [P(fake), P(real)] | P(fake) < 0.4 |

#### Critical: Normalization Must Match!

```python
# During training - SAVE the scaler
from sklearn.preprocessing import StandardScaler
import json

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Save for MQL5
params = {
    'means': scaler.mean_.tolist(),
    'stds': scaler.scale_.tolist()
}
with open('scaler_params.json', 'w') as f:
    json.dump(params, f)
```

---

### 3.3 Smart Money Concepts (SMC)

**SMC = Institutional trading patterns**. Price follows liquidity.

#### Order Blocks (OB)

| Type | Definition | Entry Logic |
|------|------------|-------------|
| Bullish OB | Last DOWN candle before strong rally | Buy when price returns to zone |
| Bearish OB | Last UP candle before strong selloff | Sell when price returns to zone |

**Validation**: Price must show reaction (rejection candle, engulfing)
**Stop Loss**: Below/above OB structure + buffer

#### Fair Value Gaps (FVG)

| Type | Pattern | Trading Logic |
|------|---------|---------------|
| Bullish FVG | Gap between C1 high and C3 low | Price fills gap upward |
| Bearish FVG | Gap between C1 low and C3 high | Price fills gap downward |

**Fill Targets**: 50% fill or full fill

#### Liquidity Concepts

| Event | Meaning | Expected Move |
|-------|---------|---------------|
| Sweep highs | Buy-side liquidity grabbed | Reversal DOWN likely |
| Sweep lows | Sell-side liquidity grabbed | Reversal UP likely |
| Equal highs/lows | Liquidity pool | Will be swept |

#### Market Structure

| Pattern | Meaning | Bias |
|---------|---------|------|
| HH + HL | Higher High + Higher Low | BULLISH |
| LH + LL | Lower High + Lower Low | BEARISH |
| BOS | Break of Structure | Continuation |
| CHoCH | Change of Character | Potential reversal |

#### Multi-Timeframe Alignment

| Timeframe | Role | Use |
|-----------|------|-----|
| H4/D1 (HTF) | Trend direction | Only trade WITH this direction |
| H1 (MTF) | Setup zone | Identify OB/FVG here |
| M15/M5 (LTF) | Entry timing | Confirmation candle |

**Rule**: NEVER trade against HTF trend (low probability)

#### Confluence Requirements

**Minimum 3 factors for valid trade:**
1. ✓ HTF trend aligned
2. ✓ MTF OB or FVG present
3. ✓ LTF confirmation candle
4. (Bonus) Regime filter passed (Hurst/Entropy)
5. (Bonus) ML confirmation > 0.65
6. (Bonus) Key S/R level
7. (Bonus) Fibonacci zone (0.618, 0.786)

---

### 3.4 FTMO Compliance (Non-Negotiable)

**FTMO rules are ABSOLUTE. Violation = Account termination.**

#### Challenge Parameters ($100k Account)

| Rule | FTMO Limit | Our Buffer | Trigger Level |
|------|------------|------------|---------------|
| Max Daily Loss | 5% ($5,000) | 4% ($4,000) | Emergency mode |
| Max Total Loss | 10% ($10,000) | 8% ($8,000) | Hard stop |
| Profit Target P1 | 10% ($10,000) | - | Phase 1 goal |
| Profit Target P2 | 5% ($5,000) | - | Phase 2 goal |
| Min Trading Days | 4 days | - | Must trade 4+ |

#### Risk Per Trade

```
FORMULA:
LotSize = (AccountEquity × RiskPercent) / (SL_Points × TickValue)

LIMITS:
• Maximum: 1% per trade ($1,000 on $100k)
• Recommended: 0.5-0.75% for consistency
• Regime adjustment: Apply size multiplier (0.5 or 1.0)
```

#### Position Sizing by Regime

| Regime | Risk % | Max Concurrent | Notes |
|--------|--------|----------------|-------|
| PRIME_TRENDING | 1.0% | 3 | Full confidence |
| PRIME_REVERTING | 1.0% | 3 | Full confidence |
| NOISY_TRENDING | 0.5% | 2 | Reduced exposure |
| NOISY_REVERTING | 0.5% | 2 | Reduced exposure |
| RANDOM_WALK | 0% | 0 | **NO TRADE** |

#### Emergency Mode Triggers

| Trigger | Action |
|---------|--------|
| Daily DD ≥ 4% | Stop new entries, manage existing only |
| Total DD ≥ 8% | Close all positions, halt trading |
| 3 consecutive losses | Cooldown period (1 hour) |
| Python Hub fails | MQL5-only mode, conservative |
| High volatility spike | Reduce size by 50% |

#### MQL5 Risk Manager Code

```mql5
class CFTMORiskManager {
private:
    double m_initialBalance;
    double m_dailyStartEquity;
    double m_peakEquity;
    
public:
    void Initialize() {
        m_initialBalance = AccountInfoDouble(ACCOUNT_BALANCE);
        m_dailyStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);
        m_peakEquity = m_initialBalance;
    }
    
    double GetDailyDD() {
        double equity = AccountInfoDouble(ACCOUNT_EQUITY);
        return (m_dailyStartEquity - equity) / m_dailyStartEquity * 100;
    }
    
    double GetTotalDD() {
        double equity = AccountInfoDouble(ACCOUNT_EQUITY);
        if(equity > m_peakEquity) m_peakEquity = equity;
        return (m_peakEquity - equity) / m_peakEquity * 100;
    }
    
    bool IsTradingAllowed() {
        if(GetDailyDD() >= 4.0) {
            Alert("FTMO: Daily DD at ", GetDailyDD(), "% - STOP");
            return false;
        }
        if(GetTotalDD() >= 8.0) {
            Alert("FTMO: Total DD at ", GetTotalDD(), "% - EMERGENCY");
            return false;
        }
        return true;
    }
    
    void OnNewDay() {
        m_dailyStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);
    }
};
```

---

### 3.5 Feature Engineering (15 Core Features)

**These features feed the Direction Model:**

| # | Feature | Calculation | Normalization |
|---|---------|-------------|---------------|
| 1 | Returns | (close - prev) / prev | StandardScaler |
| 2 | Log Returns | log(close / prev) | StandardScaler |
| 3 | Range % | (high - low) / close | StandardScaler |
| 4 | RSI M5 | RSI(14) on M5 | ÷ 100 |
| 5 | RSI M15 | RSI(14) on M15 | ÷ 100 |
| 6 | RSI H1 | RSI(14) on H1 | ÷ 100 |
| 7 | ATR Norm | ATR(14) / close | StandardScaler |
| 8 | MA Distance | (close - MA20) / MA20 | StandardScaler |
| 9 | BB Position | (close - mid) / width | Already -1 to 1 |
| 10 | Hurst | Rolling Hurst(100) | Already 0 to 1 |
| 11 | Entropy | Rolling Entropy(100) | ÷ 4 |
| 12 | Session | 0=Asia, 1=London, 2=NY | Categorical |
| 13 | Hour Sin | sin(2π × hour / 24) | Already -1 to 1 |
| 14 | Hour Cos | cos(2π × hour / 24) | Already -1 to 1 |
| 15 | OB Distance | Dist to nearest OB / ATR | StandardScaler |

**Implementation:** See `Python_Agent_Hub/app/services/feature_engineering.py`

---

**Model Specifications:**

| Model | Architecture | Input Shape | Output | Threshold | File |
|-------|--------------|-------------|--------|-----------|------|
| Direction | Bi-LSTM (2×64) + FC | (batch, 100, 15) | [P(bear), P(bull)] | P >0.65 | `models/direction.py` |
| Volatility | GRU (32) + FC | (batch, 50, 5) | ATR[5 steps] | - | `models/volatility.py` |
| Fakeout | CNN (32→64) + FC | (batch, 20, 4) | [P(fake), P(real)] | P(fake) <0.4 | `models/fakeout.py` |

**Training Requirements:**
- Direction: 50K samples, 100 epochs, lr=1e-4, WFE ≥0.6
- Volatility: 30K samples, 50 epochs, lr=1e-3, WFE ≥0.6
- Fakeout: 10K samples, 100 epochs, lr=1e-4, WFE ≥0.6

**ONNX Export:** See `models/export_onnx.py` for torch.onnx.export() implementation

---

## 4. Toolbox

### 4.1 Skills (Auto-triggered, Full MCP Access)

| Skill | Trigger Phrases | What It Does |
|-------|-----------------|--------------|
| `web-research` | "deep research", "investigate", "find repositories" | Multi-source research with triangulation |
| `ml-trading-research` | "ML research", "ONNX research", "LSTM trading" | **ML-specific** trading research |
| `scientific-critical-thinking` | "evaluate evidence", "assess methodology" | Rigorous validation of claims |
| `prompt-optimizer` | "optimize prompt", "improve this prompt" | Apply 23 principles to enhance prompts |

**Usage**: Say trigger phrase naturally. Example: "I need ML research on LSTM for gold prediction"

### 4.2 Droids (Subagents via Task Tool)

| Droid | Purpose | Model | MCP Access |
|-------|---------|-------|------------|
| `deep-researcher` | Complex academic research | Opus | No |
| `onnx-model-builder` | Build/train/export ONNX models | Sonnet | No |
| `project-reader` | Codebase analysis | Sonnet | No |
| `research-analyst-pro` | Decision-oriented research | Inherit | No |
| `trading-project-documenter` | System documentation | Opus | No |

**Note**: Droids don't have MCP access. For research with MCPs, use `web-research` skill.

### 4.3 Slash Commands

#### ML/ONNX Commands
| Command | Description |
|---------|-------------|
| `/singularity` | Activate Singularity Architect for ML strategy design |
| `/ml-research` | Deep ML trading research with MCPs |
| `/build-onnx` | Launch ONNX Model Builder droid |
| `/regime-detect` | Design regime detection system |
| `/ml-pipeline` | Full pipeline: research → train → deploy |

#### Trading Commands
| Command | Description |
|---------|-------------|
| `/architect` | MQL5 architecture review |
| `/strategy` | Strategy analysis with sequential thinking |
| `/backtest` | Statistical validation (Monte Carlo, WFA) |
| `/validate-ftmo` | FTMO compliance checker |
| `/code-review` | Trading code review |
| `/optimize` | EA optimization planning |

#### BMAD Commands
| Command | Description |
|---------|-------------|
| `/bmad-analyze` | BMAD analysis workflow |
| `/bmad-brainstorm` | Structured brainstorming |
| `/bmad-tech-spec` | Technical specification |
| `/bmad-new-feature` | Feature implementation |

### 4.4 MCP Tools (Direct Access)

| Category | Tools |
|----------|-------|
| Search | `perplexity-search`, `brave-search` |
| Docs | `context7` (library documentation) |
| Code | `github` (repos, code, PRs, issues) |
| Reasoning | `sequential-thinking`, `code-reasoning` |
| Data | `postgres` (read-only queries) |

---

## 5. Agent Personas

When domain expertise is needed, I adopt these personas from `.bmad/mql5-elite-ops/agents/`:

### 5.1 Singularity Architect 🔮 (Primary for ML)
**Expertise**: ML/ONNX, Regime Detection, Quantitative Trading
**When**: Any ML task, ONNX integration, regime analysis, feature engineering
**Invoke**: `/singularity` or ask ML-related questions

### 5.2 Quantum Strategist 🧠
**Expertise**: PRD, Risk Analysis, FTMO, R:R Ratios
**When**: Strategy design, risk questions, position sizing

### 5.3 MQL5 Architect 📐
**Expertise**: System Design, Performance, Patterns
**When**: Architecture decisions, module design, latency optimization

### 5.4 Code Artisan 🔨
**Expertise**: Clean MQL5 Code, Implementation
**When**: Coding tasks, refactoring, debugging

### 5.5 Deep Researcher 🔬
**Expertise**: Market Analysis, Fundamentals, Sentiment
**When**: Market context, news impact, research

### 5.6 Backtest Commander 🛡️
**Expertise**: Validation, Monte Carlo, Walk-Forward
**When**: Testing, certification, GO/NO-GO decisions

**To Invoke**: Say "Act as [Agent Name]" or ask domain-specific questions.

---

## 6. Workflow & Decision Trees

### 6.1 Task Router

```
USER REQUEST
      │
      ▼
┌─────────────────────────────────────┐
│  What type of task?                 │
├─────────────────────────────────────┤
│                                     │
│  Build/Code ──────► Section 6.2    │
│  Research ────────► Section 6.3    │
│  ML/ONNX ─────────► Section 6.4    │
│  Strategy ────────► Section 6.5    │
│  Validation ──────► Section 6.6    │
│                                     │
└─────────────────────────────────────┘
```

### 6.2 Build/Code Workflow
```
1. Check PRD for specification
2. Identify files to create/modify
3. Read existing code for patterns
4. Implement in small chunks
5. Test/compile
6. Validate FTMO compliance if risk-related
```

### 6.3 Research Workflow
```
1. Clarify the question
2. Check Knowledge Base (Section 3) first
   → Found answer? Use it directly
   → Not found? Use /research or /ml-research
3. Triangulate sources (academic + industry + code)
4. Synthesize with confidence levels
```

### 6.4 ML/ONNX Workflow
```
1. /singularity (activate architect)
2. Define model objective and target
3. Specify features (use Section 3.5)
4. Design architecture (use Section 3.6)
5. /build-onnx (launch droid for implementation)
6. Validate: WFE ≥ 0.6
7. Deploy to MQL5
```

### 6.5 Strategy Analysis Workflow
```
1. Use sequential-thinking (5+ thoughts)
2. Check regime filter applicability
3. Validate SMC patterns
4. Calculate R:R and position sizing
5. FTMO compliance check
6. GO/NO-GO decision
```

### 6.6 Validation Workflow
```
1. Define success criteria
2. Backtest with realistic conditions
3. Monte Carlo simulation (5000+ runs)
4. Walk-Forward Analysis (10+ windows)
5. WFE ≥ 0.6?
   → YES: Certified for deployment
   → NO: Return to design phase
```

### 6.7 Quick Decision Matrix

| Situation | Action |
|-----------|--------|
| "How do I X" (code) | Check PRD → Implement |
| "Research X" | /ml-research or /research |
| "Build ML model" | /singularity → /build-onnx |
| "Is strategy good?" | sequential-thinking → validate |
| "FTMO compliant?" | /validate-ftmo |
| Complex problem | "ultrathink" (5+ thoughts) |
| "Where is X?" | Check Section 8.3 (File Locations) |
import time

def check_regime(prices: list[float], timeout: float = 0.4) -> dict:
    """Call Python Agent Hub for regime detection"""
    try:
        response = requests.post(
            "http://localhost:8000/api/v1/regime",
            json={
                "req_id": f"mql5-{int(time.time()*1000)}",
                "timestamp": time.time(),
                "timeframe": "M15",
                "prices": prices[-150:]  # Last 150 prices
            },
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {
            "regime": "unknown",
            "action": "no_trade",
            "size_multiplier": 0.0,
            "error": str(e)
        }

# Example response:
# {
#   "regime": "prime_trending",
#   "action": "full_size",
#   "size_multiplier": 1.0,
#   "hurst_exponent": 0.62,
#   "shannon_entropy": 1.3,
#   "confidence": 0.85,
#   "kalman_trend": "bullish",
#   "score_adjustment": 10
# }
```

---

## 8. Quick Reference

### 8.1 Numbers to Remember

| Metric | Value | Context |
|--------|-------|---------|
| Max Daily DD | 5% (trigger at 4%) | FTMO |
| Max Total DD | 10% (trigger at 8%) | FTMO |
| Risk per trade | 1% max, 0.5% recommended | Position sizing |
| OnTick latency | < 50ms | MQL5 requirement |
| ONNX inference | < 5ms | Model speed |
| Python Hub timeout | 400ms | Communication |
| WFE minimum | 0.6 | Model validation |
| ML confidence | > 0.65 | Signal threshold |
| Hurst trending | > 0.55 | Regime |
| Hurst reverting | < 0.45 | Regime |
| Entropy low | < 1.5 | Low noise |

### 8.2 I Need To... → Do This

| Need | Action |
|------|--------|
| Build feature | Check PRD → Code → Test |
| Research topic | `/ml-research` or `/research` |
| Build ML model | `/singularity` → `/build-onnx` |
| Validate strategy | sequential-thinking (5+ thoughts) |
| Check FTMO compliance | `/validate-ftmo` |
| Debug issue | Read code → Trace → Fix |
| Solve complex problem | "ultrathink" |

### 8.3 File Locations

| What | Where |
|------|-------|
| PRD v2.2 | `DOCS/prd.md` |
| Regime Detector | `Python_Agent_Hub/app/services/regime_detector.py` |
| Schemas | `Python_Agent_Hub/app/models/schemas.py` |
| API Routes | `Python_Agent_Hub/app/routers/analysis.py` |
| ONNX Models | `MQL5/Models/` (to be created) |
| Slash Commands | `.factory/commands/` |
| Droids | `.factory/droids/` |
| Skills | `.factory/skills/` |
| Elite Ops Agents | `.bmad/mql5-elite-ops/agents/` |
| Main EAs | `🚀 MAIN_EAS/PRODUCTION/` |

### 8.4 Thresholds Cheat Sheet

```
REGIME FILTER (Singularity Filter):
═══════════════════════════════════
H > 0.55 + S < 1.5  → PRIME_TRENDING    → Size: 100%
H > 0.55 + S ≥ 1.5  → NOISY_TRENDING    → Size: 50%
H < 0.45 + S < 1.5  → PRIME_REVERTING   → Size: 100%
H < 0.45 + S ≥ 1.5  → NOISY_REVERTING   → Size: 50%
0.45 ≤ H ≤ 0.55     → RANDOM_WALK       → Size: 0% (NO TRADE)

ML CONFIRMATION:
════════════════
P(direction) > 0.65  → Confirmed signal
P(fakeout) < 0.40    → Real breakout
Kalman vel > 0.1     → Trend confirmed

FTMO GATES:
═══════════
Daily DD < 4%   → Continue trading
Daily DD ≥ 4%   → Stop new entries
Total DD < 8%   → Continue trading
Total DD ≥ 8%   → EMERGENCY STOP
```

---

## 9. Session Protocols

### 9.1 Session Start
```
□ What's the ONE thing to build today?
□ Check PRD Section 14.5 for current phase
□ Confirm Builder Mode (not Planning Mode)
```

### 9.2 Session End
```
□ Did we produce CODE, not just docs?
□ Does it compile/run?
□ What's the next small task?
```

### 9.3 Anti-Patterns

| Doing This | Do This Instead |
|------------|-----------------|
| Writing more docs | Write CODE |
| Refining PRD | PRD is DONE (v2.2) |
| Designing architecture | Architecture is IN THE PRD |
| Task > 4 hours | SPLIT IT SMALLER |
| "How should we approach..." | Just START |

### 9.4 Context Compaction

**Trigger when**:
- 30+ exchanges in conversation
- Multiple tangent topics
- "Where were we?"
- Before major new task

**Protocol**:
1. Summarize key decisions
2. List files modified
3. Identify current + next task
4. Discard completed threads

---

## 10. Coding Guidelines

### 10.1 MQL5 Standards

```mql5
#property copyright "EA_SCALPER_XAUUSD"
#property version   "2.20"
#property strict

// Naming conventions:
// - Classes: CPascalCase
// - Methods: PascalCase()
// - Variables: camelCase
// - Constants: UPPER_SNAKE_CASE
// - Member vars: m_memberName
// - Global vars: g_globalName

// Always check errors after trade operations:
if(!trade.PositionOpen(...)) {
    int error = GetLastError();
    Print("Trade failed: ", error);
}

// Use SymbolInfo, never hardcode:
double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);

// No blocking calls in OnTick - use OnTimer for slow operations
```

### 10.2 Python Standards

```python
# Type hints required
def calculate_hurst(prices: np.ndarray, window: int = 100) -> float:
    ...

# Async for I/O operations
async def fetch_data(symbol: str) -> pd.DataFrame:
    ...

# Pydantic for validation
class RegimeRequest(BaseModel):
    prices: list[float] = Field(..., min_length=20)
    
# Docstrings for public functions
def detect_regime(prices: np.ndarray) -> RegimeAnalysis:
    """
    Detect market regime using Hurst + Entropy.
    
    Args:
        prices: Array of close prices (min 20)
    
    Returns:
        RegimeAnalysis with regime, action, metrics
    """
```

### 10.3 Performance Constraints

| Constraint | Limit | Measured At |
|------------|-------|-------------|
| OnTick execution | < 50ms | End of OnTick |
| ONNX inference | < 5ms | OnnxRun call |
| Python Hub call | < 400ms | Round-trip |
| Hub timeout | 400ms | Request timeout |
| Max in-flight | 1 request | Queue limit |

---

## 11. Troubleshooting

| Problem | Solution |
|---------|----------|
| ONNX load fails | Check file path, model version compatibility |
| Hub timeout | Reduce payload, check Hub is running |
| DD limit hit | Emergency mode, manage existing positions only |
| Compilation error | Check #include paths, #property strict |
| Random walk detected | Don't trade, wait for regime change |
| Low confidence | Check feature normalization matches training |
| Planning loop | STOP. Pick smallest task. Build it. |

---

## 12. Project Status

### Phase Status (PRD v2.2)

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | 🔲 In Progress | MQL5 Core |
| Phase 2 | 🔲 In Progress | Python Technical Agent |
| Phase 3 | 🔲 Pending | Fund/Sent/LLM |
| Phase 4 | 🔲 Pending | Auto-optimization |
| Phase 5 | ⏳ Partial | ML/ONNX |

### Phase 5 Milestones

| # | Milestone | Status |
|---|-----------|--------|
| 5.1 | Regime Detection | ✅ DONE |
| 5.2 | Regime Endpoint | ✅ DONE |
| 5.3 | TechnicalAgent Integration | ✅ DONE |
| 5.4 | Kalman Filter | ✅ DONE |
| 5.5 | Direction Model Training | 🔲 TODO |
| 5.6 | ONNX Export | 🔲 TODO |
| 5.7 | MQL5 ONNX Integration | 🔲 TODO |
| 5.8-5.11 | Remaining | 🔲 TODO |

---

## 13. Version History

| Version | Date | Changes |
|---------|------|---------|
| **v4.2** | 2025-11-29 | Optimized for context efficiency - 50% size reduction while preserving all essential content |

---

## 14. RAG Knowledge Base (Local Documentation)

### 14.1 Overview

This project has a **local knowledge base** indexed in vector format (RAG - Retrieval Augmented Generation). There are **24,544 chunks** of documentation that can be queried semantically.

### 14.2 Database Structure

```
.rag-db/
├── books/              ← DATABASE 1: Conceptual Knowledge
│   └── documents.lance     5,909 chunks from 15 PDFs
│
├── docs/               ← DATABASE 2: MQL5 Technical Reference
│   └── documents.lance     18,635 chunks from 7,629 pages
│
└── models/             ← Embedding model
    └── all-MiniLM-L6-v2
```

### 14.3 When to Use Each Database

| Situation | Database | Example Query |
|-----------|----------|---------------|
| Understand concept | **BOOKS** | "How does Hurst exponent work?" |
| Learn strategy | **BOOKS** | "Smart Money Concepts order blocks" |
| Machine Learning | **BOOKS** | "LSTM for price prediction" |
| Function syntax | **DOCS** | "OnnxRun parameters" |
| Code example | **DOCS** | "iMA indicator example" |
| MQL5 Class/Struct | **DOCS** | "CTrade class methods" |
| Implement something | **BOTH** | "How to implement ONNX in MQL5" |

### 14.4 RAG Query Code

```python
import lancedb
from sentence_transformers import SentenceTransformer

# Load model (once)
model = SentenceTransformer('all-MiniLM-L6-v2')

def query_rag(query: str, database: str = "books", limit: int = 5) -> list:
    """
    Semantic search in local RAG.
    
    Args:
        query: Question or search term
        database: "books" (concepts) or "docs" (MQL5 syntax)
        limit: Number of results
    
    Returns:
        List of dicts with {source, text, score}
    """
    db = lancedb.connect(f".rag-db/{database}")
    tbl = db.open_table("documents")
    
    embedding = model.encode(query)
    results = tbl.search(embedding).limit(limit).to_pandas()
    
    return [
        {"source": r["source"], "text": r["text"][:500], "score": r["_distance"]}
        for _, r in results.iterrows()
    ]

# Usage example
results = query_rag("Hurst exponent regime detection", "books", 3)
for r in results:
    print(f"[{r['source']}] {r['text'][:200]}...")
```

### 14.5 Mandatory Workflow for Code

**BEFORE writing any MQL5 code:**

```
┌─────────────────────────────────────────────────────────────┐
│                 RAG-FIRST DEVELOPMENT FLOW                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. UNDERSTAND THE CONCEPT                                  │
│     └── Query BOOKS: "What is X? How does it work?"         │
│                                                             │
│  2. VERIFY MQL5 SYNTAX                                      │
│     └── Query DOCS: "Function X parameters example"         │
│                                                             │
│  3. SEARCH SIMILAR EXAMPLES                                 │
│     └── Query DOCS: "Example of X implementation"           │
│                                                             │
│  4. IMPLEMENT CODE                                          │
│     └── Combine knowledge from queries                      │
│                                                             │
│  5. VALIDATE                                                │
│     └── Query DOCS to verify edge cases                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 14.6 Database Contents

#### BOOKS (5,909 chunks)

| PDF | Chunks | Main Topics |
|-----|--------|-------------|
| mql5.pdf | 2,195 | Complete official documentation |
| mql5book.pdf | 1,558 | Step-by-step MQL5 tutorial |
| neuronetworksbook.pdf | 578 | **ML/ONNX for trading** (IMPORTANT!) |
| Algorithmic Trading | 485 | Hurst, Entropy, advanced statistics |
| Forecasting Financial | 316 | Market forecasting |
| EA Programming MT5 | 211 | Expert Advisor programming |
| VWAP Trading | 179 | Anchored VWAP, volume |
| Fibonacci Trading | 126 | Fibonacci, Price Action |
| + other 7 PDFs | 261 | Trading, strategies |

#### DOCS (18,635 chunks)

| Section | Files | Content |
|---------|-------|---------|
| Reference | 3,925 | MQL5 functions, classes, structs |
| Book | 788 | Official MQL5 tutorial |
| CodeBase Experts | 2,501 | Real EA examples |
| CodeBase Indicators | 920 | Custom indicators |

### 14.7 Practical Query Examples

#### Example 1: Implement Hurst Exponent

```python
# Step 1: Understand the concept (BOOKS)
r1 = query_rag("Hurst exponent calculation R/S analysis", "books", 3)
# → Returns R/S analysis theory, formula, interpretation

# Step 2: See MQL5 statistical functions (DOCS)
r2 = query_rag("MathStandardDeviation statistics functions", "docs", 3)
# → Returns statistics function syntax

# Step 3: Array loop example (DOCS)
r3 = query_rag("array loop iteration MQL5 example", "docs", 3)
# → Returns array iteration examples
```

#### Example 2: Integrate ONNX

```python
# Step 1: ONNX concept in trading (BOOKS - neuronetworksbook)
r1 = query_rag("ONNX model inference trading neural network", "books", 5)

# Step 2: OnnxCreate and OnnxRun syntax (DOCS)
r2 = query_rag("OnnxCreate OnnxRun parameters example", "docs", 5)

# Step 3: Complete example (DOCS)
r3 = query_rag("ONNX model MQL5 complete example", "docs", 5)
```

### 14.8 Available Skill

The skill `.factory/skills/mql5-rag-search.md` was created and can be invoked with:

- "search documentation"
- "search in books"
- "query RAG"
- "search MQL5"

### 14.9 Golden Rule

```
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   NEVER WRITE MQL5 CODE WITHOUT FIRST QUERYING THE RAG       ║
║                                                               ║
║   1. Don't know the syntax? → DOCS                           ║
║   2. Don't understand the concept? → BOOKS                   ║
║   3. Need an example? → DOCS                                 ║
║   4. Want to implement ML? → BOOKS (neuronetworksbook)       ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

---



---

**END OF CLAUDE.md v4.2 - SINGULARITY EDITION**

*"Build > Plan. Code > Docs. Ship > Perfect."*
*"Always consult RAG before coding."*
</coding_guidelines>

_Section 15 removed - content covered in Section 2.1 (Project File Structure) and MASTER_INDEX_ATUALIZADO.md_
