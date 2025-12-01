# PROJECT_CONTEXT.md - Contexto Compartilhado para Todos os Agentes

> **TODOS OS SKILLS DEVEM LER ESTE ARQUIVO ANTES DE QUALQUER ACAO**
> Atualizado: 2025-11-30

---

## 1. IDENTIFICACAO DO PROJETO

| Campo | Valor |
|-------|-------|
| **Nome** | EA_SCALPER_XAUUSD |
| **Versao** | v3.30 - Singularity Order Flow Edition |
| **Mercado** | XAUUSD (Gold) |
| **Timeframes** | H1 (direcao) → M15 (zonas) → M5 (execucao) |
| **Objetivo** | FTMO $100k Challenge |
| **Owner** | Franco |

---

## 2. ESTRUTURA DE ARQUIVOS PRINCIPAL

```
EA_SCALPER_XAUUSD/
│
├── MQL5/                          # ⭐ PASTA PRINCIPAL DO CODIGO
│   ├── Experts/
│   │   └── EA_SCALPER_XAUUSD.mq5  # ⭐⭐⭐ EA PRINCIPAL (v3.30)
│   │
│   ├── Include/EA_SCALPER/        # Modulos do EA
│   │   ├── INDEX.md               # 📖 Documentacao completa (1997 linhas)
│   │   ├── Analysis/              # CMTFManager, CRegimeDetector, etc.
│   │   ├── Backtest/              # CBacktestRealism.mqh
│   │   ├── Bridge/                # PythonBridge, COnnxBrain
│   │   ├── Core/                  # Definitions, CState, CEngine
│   │   ├── Execution/             # CTradeManager, TradeExecutor
│   │   ├── Risk/                  # FTMO_RiskManager
│   │   ├── Signal/                # CConfluenceScorer
│   │   └── Strategy/
│   │
│   ├── Models/                    # Modelos ONNX
│   │   ├── direction_model.onnx   # ⭐ Modelo de direcao atual
│   │   └── scaler_params_final.json
│   │
│   └── Scripts/                   # Scripts de teste
│
├── Python_Agent_Hub/              # Backend Python (FastAPI)
│   ├── requirements.txt           # Dependencias Python
│   └── ml_pipeline/               # Pipeline ML para ONNX
│
├── .factory/                      # Configuracoes Factory
│   ├── skills/                    # Skills dos agentes
│   └── PROJECT_CONTEXT.md         # ⭐ ESTE ARQUIVO
│
├── DOCS/                          # Documentacao
│   ├── _INDEX.md                  # Indice geral
│   ├── 02_IMPLEMENTATION/         # Plano e progresso
│   └── 04_REPORTS/                # Backtests e decisoes
│
└── scripts/                       # Scripts Python auxiliares
    ├── baseline_backtest.py       # Backtest baseline
    └── validate_data.py           # Validacao de dados
```

---

## 3. ARQUIVOS CRITICOS POR AGENTE

### 🔮 ORACLE (Backtest/Validacao)
| Arquivo | Caminho | Uso |
|---------|---------|-----|
| **EA Principal** | `MQL5/Experts/EA_SCALPER_XAUUSD.mq5` | Robo a testar |
| **CBacktestRealism** | `MQL5/Include/EA_SCALPER/Backtest/CBacktestRealism.mqh` | Realism simulation |
| **Modelo ONNX** | `MQL5/Models/direction_model.onnx` | Validar modelo ML |
| **Scaler Params** | `MQL5/Models/scaler_params_final.json` | Params de normalizacao |
| **INDEX.md** | `MQL5/Include/EA_SCALPER/INDEX.md` | Arquitetura completa |

### 🔥 CRUCIBLE (Estrategia)
| Arquivo | Caminho | Uso |
|---------|---------|-----|
| **CMTFManager** | `MQL5/Include/EA_SCALPER/Analysis/CMTFManager.mqh` | Multi-timeframe |
| **CStructureAnalyzer** | `MQL5/Include/EA_SCALPER/Analysis/CStructureAnalyzer.mqh` | SMC BOS/CHoCH |
| **CFootprintAnalyzer** | `MQL5/Include/EA_SCALPER/Analysis/CFootprintAnalyzer.mqh` | Order Flow |
| **EliteOrderBlock** | `MQL5/Include/EA_SCALPER/Analysis/EliteOrderBlock.mqh` | Order Blocks |
| **EliteFVG** | `MQL5/Include/EA_SCALPER/Analysis/EliteFVG.mqh` | Fair Value Gaps |

### 🛡️ SENTINEL (Risco)
| Arquivo | Caminho | Uso |
|---------|---------|-----|
| **FTMO_RiskManager** | `MQL5/Include/EA_SCALPER/Risk/FTMO_RiskManager.mqh` | Risk compliance |
| **CDynamicRiskManager** | `MQL5/Include/EA_SCALPER/Risk/` | Risco dinamico |
| **EA Inputs** | `MQL5/Experts/EA_SCALPER_XAUUSD.mq5` (linhas 60-90) | Params de risco |

### ⚒️ FORGE (Codigo)
| Arquivo | Caminho | Uso |
|---------|---------|-----|
| **INDEX.md** | `MQL5/Include/EA_SCALPER/INDEX.md` | Arquitetura completa |
| **Definitions.mqh** | `MQL5/Include/EA_SCALPER/Core/Definitions.mqh` | Enums e structs |
| **Todos os .mqh** | `MQL5/Include/EA_SCALPER/**/*.mqh` | Modulos para editar |

### 🔍 ARGUS (Pesquisa)
| Arquivo | Caminho | Uso |
|---------|---------|-----|
| **ORDER_FLOW_README** | `MQL5/Include/EA_SCALPER/Analysis/ORDER_FLOW_README.md` | Refs Order Flow |
| **INDEX.md** | `MQL5/Include/EA_SCALPER/INDEX.md` | Estado atual |

---

## 4. BIBLIOTECAS DE BACKTEST (PYTHON)

### 4.1 Recomendadas para Este Projeto

| Biblioteca | Uso | Install |
|------------|-----|---------|
| **vectorbt** | Backtest vetorizado, rapido | `pip install vectorbt` |
| **backtesting.py** | Backtest simples, visual | `pip install backtesting` |
| **pandas** | Manipulacao de dados | Ja instalado |
| **numpy** | Calculos numericos | Ja instalado |
| **scipy** | Monte Carlo, stats | `pip install scipy` |
| **matplotlib** | Graficos | Ja instalado |

### 4.2 Para Walk-Forward Analysis

```python
# Estrutura basica de WFA com vectorbt
import vectorbt as vbt
import pandas as pd

def run_wfa(data, n_windows=10, is_ratio=0.7):
    """Walk-Forward Analysis"""
    results = []
    window_size = len(data) // n_windows
    
    for i in range(n_windows):
        is_start = i * int(window_size * (1 - 0.25))  # 25% overlap
        is_end = is_start + int(window_size * is_ratio)
        oos_end = is_start + window_size
        
        is_data = data.iloc[is_start:is_end]
        oos_data = data.iloc[is_end:oos_end]
        
        # Otimizar em IS, testar em OOS
        # ... implementacao
        
    return calculate_wfe(results)
```

### 4.3 Para Monte Carlo

```python
import numpy as np

def monte_carlo_block_bootstrap(trades, n_sim=5000, block_size=7):
    """Block Bootstrap Monte Carlo - preserva autocorrelacao"""
    profits = trades['profit'].values
    n_blocks = len(profits) // block_size
    
    max_dds = []
    for _ in range(n_sim):
        # Resample blocks
        indices = np.random.randint(0, n_blocks, size=n_blocks)
        sim_profits = []
        for idx in indices:
            sim_profits.extend(profits[idx*block_size:(idx+1)*block_size])
        
        # Calculate equity curve and max DD
        equity = np.cumsum(sim_profits) + 100000
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / peak * 100
        max_dds.append(dd.max())
    
    return {
        'dd_5th': np.percentile(max_dds, 5),
        'dd_50th': np.percentile(max_dds, 50),
        'dd_95th': np.percentile(max_dds, 95),
        'dd_99th': np.percentile(max_dds, 99)
    }
```

---

## 5. COMO RODAR BACKTEST

### 5.1 Via MetaTrader 5 Strategy Tester

1. Abrir MT5
2. Ctrl+R (Strategy Tester)
3. Selecionar: `EA_SCALPER_XAUUSD.mq5`
4. Simbolo: XAUUSD
5. Periodo: M5 (execucao) - dados H1/M15 sao carregados internamente
6. Modo: "Every tick based on real ticks"
7. **IMPORTANTE**: Usar dados de pelo menos 2 anos

### 5.2 Via Python (Exportar trades do MT5)

```python
# Exportar trades do MT5 → arquivo CSV
# Depois analisar com Python

import pandas as pd
from pathlib import Path

# Carregar trades exportados
trades = pd.read_csv('backtest_trades.csv')

# Metricas basicas
print(f"Total trades: {len(trades)}")
print(f"Win rate: {(trades['profit'] > 0).mean():.1%}")
print(f"Profit factor: {trades[trades['profit']>0]['profit'].sum() / abs(trades[trades['profit']<0]['profit'].sum()):.2f}")
```

### 5.3 Dados para Backtest

| Fonte | Tipo | Localizacao |
|-------|------|-------------|
| MT5 History | Ticks reais | Terminal MT5 |
| CSV Export | Trades | `data/backtest_results/` |
| OHLCV | Candles | `data/historical/` |

---

## 6. PARAMETROS FTMO (LIMITES)

```
┌─────────────────────────────────────────────────────────┐
│              FTMO $100k CHALLENGE                       │
├─────────────────────────────────────────────────────────┤
│  Daily Drawdown Limit:    5% ($5,000)  → Trigger: 4%   │
│  Total Drawdown Limit:   10% ($10,000) → Trigger: 8%   │
│  Profit Target Phase 1:  10% ($10,000)                 │
│  Profit Target Phase 2:   5% ($5,000)                  │
│  Min Trading Days:        4 dias                       │
│  Risk per Trade:          0.5-1% max                   │
└─────────────────────────────────────────────────────────┘

REGRAS INVIOLAVEIS:
1. NUNCA ultrapassar 5% DD diario
2. NUNCA ultrapassar 10% DD total
3. SEMPRE usar stop loss
4. DD calculado com EQUITY (nao balance)
```

---

## 7. MODELOS ONNX

### 7.1 Modelo Atual

| Campo | Valor |
|-------|-------|
| **Arquivo** | `MQL5/Models/direction_model.onnx` |
| **Tipo** | Classificacao binaria (UP/DOWN) |
| **Input** | Features normalizadas (ver scaler_params) |
| **Output** | Probabilidade direcao |
| **Threshold** | P > 0.65 para trade |

### 7.2 Validar Modelo ONNX

```python
import onnxruntime as ort
import numpy as np
import json

# Carregar modelo
session = ort.InferenceSession('MQL5/Models/direction_model.onnx')

# Carregar scaler params
with open('MQL5/Models/scaler_params_final.json') as f:
    scaler = json.load(f)

# Inferencia
input_name = session.get_inputs()[0].name
features = np.array([[...]], dtype=np.float32)  # Normalizar primeiro!
prediction = session.run(None, {input_name: features})
```

---

## 8. METRICAS TARGET

| Metrica | Target | Minimo GO |
|---------|--------|-----------|
| WFE (Walk-Forward Efficiency) | >= 0.6 | >= 0.5 |
| Max Drawdown | < 6% | < 8% |
| Monte Carlo 95th DD | < 8% | < 10% |
| Profit Factor | > 2.0 | > 1.5 |
| Win Rate | > 70% | > 55% |
| SQN | > 2.5 | >= 2.0 |
| Sharpe Ratio | > 2.0 | > 1.5 |
| Trades (amostra) | 200+ | >= 100 |
| Periodo testado | 3+ anos | >= 2 anos |

---

## 9. FLUXO DE VALIDACAO

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   FORGE     │────▶│   ORACLE    │────▶│  SENTINEL   │────▶│   GO/NO-GO  │
│  Implementa │     │  Valida     │     │  Risk Check │     │   Decision  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
      │                   │                   │                    │
      ▼                   ▼                   ▼                    ▼
   Codigo MQL5      WFA + Monte Carlo    Lot sizing         Pode ir live?
   Modelo ONNX      Bias detection       DD limits          Reducao size?
   Testes unit      GO-NOGO metrics      FTMO compliance    Proximo passo
```

---

## 10. COMANDOS RAPIDOS

### Ver estrutura do EA:
```bash
type MQL5\Include\EA_SCALPER\INDEX.md | more
```

### Listar modulos:
```bash
dir /b MQL5\Include\EA_SCALPER\Analysis\
```

### Ver parametros do EA:
```bash
type MQL5\Experts\EA_SCALPER_XAUUSD.mq5 | findstr "input"
```

### Rodar backtest Python:
```bash
cd scripts
python baseline_backtest.py
```

---

## 11. REFERENCIAS IMPORTANTES

| Documento | Caminho | Conteudo |
|-----------|---------|----------|
| **INDEX.md** | `MQL5/Include/EA_SCALPER/INDEX.md` | Arquitetura completa, 1997 linhas |
| **AGENTS.md** | `AGENTS.md` | Routing de agentes |
| **PLAN_v1.md** | `DOCS/02_IMPLEMENTATION/PLAN_v1.md` | Plano de implementacao |
| **PROGRESS.md** | `DOCS/02_IMPLEMENTATION/PROGRESS.md` | Status atual |

---

*Gerado por BMad Builder 🧙 - 2025-11-30*
*TODOS os skills devem ler este arquivo antes de qualquer acao!*
