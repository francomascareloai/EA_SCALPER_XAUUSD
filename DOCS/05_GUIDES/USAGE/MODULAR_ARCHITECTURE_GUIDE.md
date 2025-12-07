# 🏗️ Arquitetura Modular - Guia Completo
**Sistema**: EA_SCALPER_XAUUSD v2.2  
**Framework**: NautilusTrader  
**Atualizado**: 2025-12-07

---

## 🎯 Resposta Rápida

**Pergunta**: "Quero testar todas as estratégias. Preciso mudar vários arquivos?"

**Resposta**: ❌ **NÃO!** 

✅ **Mude APENAS 1 arquivo**: `configs/strategy_config.yaml`

```yaml
execution:
  execution_threshold: 60  # ← Baixar para gerar mais trades
  use_selector: true       # ← true = StrategySelector decide
                           #   false = sempre usa GoldScalperStrategy
  use_mtf: true            # ← true = usa HTF/MTF
  use_footprint: true      # ← true = usa FootprintAnalyzer
```

**Depois roda**:
```bash
python nautilus_gold_scalper/scripts/run_backtest.py --start 2024-01-01 --end 2024-12-31
```

**Pronto!** Tudo funciona.

---

## 📐 Arquitetura em Camadas

### Visão Geral (Pyramid Model)

```
┌─────────────────────────────────────────────────────────────┐
│                    LAYER 1: RUNNER                          │
│                run_backtest.py (648 linhas)                 │
│   • Load config YAML                                        │
│   • Setup Nautilus engine                                   │
│   • Instantiate strategy                                    │
│   • Run simulation                                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  LAYER 2: STRATEGY                          │
│            GoldScalperStrategy (1,064 linhas)               │
│   • Receives ticks/bars                                     │
│   • Delegates to sub-modules                                │
│   • Makes final trade decision                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                 LAYER 3: SUB-MODULES                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ ConfluenceScorer │ FootprintAnalyzer │ RegimeDetector │  │
│  │   (991 linhas) │   (990 linhas)   │  (442 linhas)  │    │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ SessionFilter  │ StrategySelector │  EntryOptimizer│      │
│  │   (175 linhas) │   (551 linhas)   │  (305 linhas)  │    │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  LAYER 4: RISK/EXECUTION                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ PropFirmMgr   │  CircuitBreaker │ TimeConstraint │      │
│  │   (170 linhas) │   (541 linhas)   │  (95 linhas)   │    │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐                         │
│  │ PositionSizer │ ExecutionModel │                         │
│  │   (132 linhas) │  (1.3KB file)  │                        │
│  └──────────────┘  └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   LAYER 5: CONFIG                           │
│              strategy_config.yaml (115 linhas)              │
│   • Todos os knobs tunáveis                                 │
│   • Confluence weights, footprint params                    │
│   • Risk limits, execution settings                         │
│   • SINGLE SOURCE OF TRUTH                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Como Funciona (Fluxo de Execução)

### 1️⃣ Startup (run_backtest.py)

```python
# Linha 96-140: Carrega config YAML
config_dict = load_yaml_config(Path("configs/strategy_config.yaml"))

# Linha 141-220: Build strategy config object
strategy_config = build_strategy_config(
    config_dict,
    bar_type,
    instrument_id
)
# ↓ Todas as configs vão para GoldScalperConfig dataclass

# Linha 373: Instantiate strategy
strategy = GoldScalperStrategy(config=strategy_config)

# Linha 397: Run!
engine.run()
```

**Resultado**: **1 arquivo YAML** controla TUDO.

---

### 2️⃣ Strategy Initialization (GoldScalperStrategy)

```python
# gold_scalper_strategy.py:159-240
def on_start(self):
    # Inicializa sub-módulos baseado no config
    
    # 1. Session Filter (sempre ativo se use_session_filter=True)
    if self.config.use_session_filter:
        self._session_filter = SessionFilter(...)
    
    # 2. Regime Detector (sempre ativo se use_regime_filter=True)
    if self.config.use_regime_filter:
        self._regime_detector = RegimeDetector(...)
    
    # 3. Footprint Analyzer (controlado por use_footprint)
    if self.config.use_footprint:
        self._footprint = FootprintAnalyzer(...)
    
    # 4. Strategy Selector (controlado por use_selector)
    if self.config.use_selector:
        self._strategy_selector = StrategySelector(...)
    
    # 5. Risk modules (sempre ativos se prop_firm_enabled=True)
    if self.config.prop_firm_enabled:
        self._prop_firm = PropFirmManager(...)
        self._circuit_breaker = CircuitBreaker(...)
        self._time_constraint = TimeConstraintManager(...)
```

**Resultado**: Config YAML **ativa/desativa módulos** automaticamente.

---

### 3️⃣ Signal Generation (Per Tick/Bar)

```python
# gold_scalper_strategy.py:439-488
def _check_for_signal(self):
    # GATE 1: Session filter
    if self._session_filter and not self._session_filter.can_trade():
        return  # ← Block: fora de London/NY
    
    # GATE 2: Regime filter
    if self._regime_detector:
        regime = self._regime_detector.get_regime()
        if regime == Regime.RANDOM:
            return  # ← Block: mercado random
    
    # GATE 3: Time constraint (Apex 4:59 PM)
    if self._time_constraint and not self._time_constraint.can_trade():
        return  # ← Block: após 4:59 PM ET
    
    # GATE 4: Circuit breaker
    if self._circuit_breaker and not self._circuit_breaker.can_trade():
        return  # ← Block: loss streak
    
    # GATE 5: Confluence score
    signal = self._confluence_scorer.check_entry(...)
    if signal.total_score < self.config.execution_threshold:
        return  # ← Block: score < 70
    
    # GATE 6: Strategy Selector (opcional)
    if self._strategy_selector:
        strategy = self._strategy_selector.select_strategy(context)
        if strategy == StrategyType.STRATEGY_NONE:
            return  # ← Block: nenhuma strategy adequada
    
    # ✅ PASSED ALL GATES → Generate order
    self._submit_order(signal)
```

**Resultado**: Filtros em **cascata** (AND logic). Todos devem passar.

---

## 🎛️ Pontos de Configuração Centralizados

### ÚNICO Arquivo de Config: `strategy_config.yaml`

```yaml
# ============================================================
# SEÇÃO 1: CONFLUENCE (Score System)
# ============================================================
confluence:
  min_score_to_trade: 70  # ← THRESHOLD PRINCIPAL
                           # 60 = mais trades, menos qualidade
                           # 70 = balanceado (atual)
                           # 80 = poucos trades, alta qualidade
  
  footprint_weight: 10     # Peso do footprint no score
  fib_weight: 10           # Peso dos fib levels
  
  session_weights:         # Pesos por sessão (override defaults)
    asian: null            # null = usa default do código
    london: null
    ny_overlap: null
    ny: null

# ============================================================
# SEÇÃO 2: FOOTPRINT (Order Flow)
# ============================================================
footprint:
  cluster_size: 0.5        # Tamanho do cluster (pips)
  imbalance_ratio: 3.0     # Buy/Sell imbalance threshold
  stacked_min: 3           # Mínimo de imbalances stacked
  absorption_threshold: 15.0
  lookback_bars: 20        # Quantos bars analisar

# ============================================================
# SEÇÃO 3: FIBONACCI
# ============================================================
fibonacci:
  enabled: true            # ← Ligar/desligar Fib
  use_levels: [0.382, 0.5, 0.618]  # Quais níveis usar
  tp_ext: [1.272, 1.618, 2.0]      # Take profit extensions

# ============================================================
# SEÇÃO 4: RISK MANAGEMENT
# ============================================================
risk:
  max_risk_per_trade: 0.01  # 1% por trade
  dd_soft: 0.03             # 3% daily DD warning
  dd_hard: 0.05             # 5% total DD limit
  kelly_fraction: 0.25      # Kelly criterion fraction

# ============================================================
# SEÇÃO 5: EXECUTION (Controla Módulos)
# ============================================================
execution:
  execution_threshold: 70   # ← Repetido (sync com confluence)
  
  # ✅ FLAGS DE MÓDULOS (true/false)
  use_selector: true        # StrategySelector on/off
  use_mtf: true             # HTF/MTF confluence on/off
  use_footprint: true       # FootprintAnalyzer on/off
  
  # Realism knobs
  slippage_ticks: 2         # Slippage realista
  commission_per_contract: 2.5  # Comissão por lote
  fill_model: realistic     # immediate | realistic | worst_case
  
  # Apex rules
  allow_overnight: false    # NUNCA true para Apex
  max_spread_points: 80     # Bloqueia se spread >80 cents

# ============================================================
# SEÇÃO 6: CIRCUIT BREAKER
# ============================================================
circuit_breaker:
  level_1_losses: 3         # 3 losses → cooldown 5 min
  level_2_losses: 5         # 5 losses → cooldown 15 min + size -25%
  level_3_dd: 3.0           # 3% DD → cooldown 30 min + size -50%
  level_4_dd: 4.0           # 4% DD → pausar até próximo dia
  level_5_dd: 4.5           # 4.5% DD → lockdown (manual reset)
  
  size_multipliers:
    level_2: 0.75           # 75% do tamanho normal
    level_3: 0.5            # 50% do tamanho normal

# ============================================================
# SEÇÃO 7: TIME CONSTRAINTS (Apex Cutoff)
# ============================================================
time:
  cutoff_et: "16:59"        # 4:59 PM ET deadline
  warning_et: "16:00"       # Warning 1h antes
  urgent_et: "16:30"        # Urgent 30min antes
  emergency_et: "16:55"     # Emergency 5min antes

# ============================================================
# SEÇÃO 8: CONSISTENCY (Apex 30% Rule)
# ============================================================
consistency:
  daily_profit_cap_pct: 30.0  # Máximo 30% profit diário
```

---

## 🔀 Como Trocar Estratégias

### Cenário 1: "Quero MAIS TRADES"

**Config Atual** (conservador):
```yaml
confluence:
  min_score_to_trade: 70

execution:
  execution_threshold: 70
  use_selector: true
  use_mtf: true
  use_footprint: true
```

**Resultado**: 0 trades em Nov 2024 (muito restritivo)

---

**Solução A** (abaixar threshold):
```yaml
confluence:
  min_score_to_trade: 60  # ← De 70 → 60

execution:
  execution_threshold: 60  # ← De 70 → 60
```

**Resultado esperado**: +50-100% mais trades

---

**Solução B** (desligar filtros):
```yaml
execution:
  use_selector: false      # ← Desliga StrategySelector
  use_mtf: false           # ← Desliga HTF/MTF check
  use_footprint: true      # ← Mantém footprint (core)
```

**Resultado esperado**: +200% mais trades (menos filtros)

---

### Cenário 2: "Quero Testar SEM Footprint"

```yaml
execution:
  use_footprint: false  # ← Desliga FootprintAnalyzer
```

**O que acontece**:
1. FootprintAnalyzer **não é inicializado**
2. Confluence scorer **não chama** footprint
3. Score vem apenas de: Session + Regime + Fib + MTF

**Resultado**: Mais simples, potencialmente mais trades.

---

### Cenário 3: "Quero Testar StrategySelector Sozinho"

**Goal**: Ver qual strategy o selector escolhe para cada contexto.

```yaml
execution:
  use_selector: true       # ← ON
  execution_threshold: 50  # ← Baixar para gerar signals

# E depois verificar logs:
# strategy_selector vai escolher:
# - STRATEGY_TREND_FOLLOW (se Hurst > 0.55)
# - STRATEGY_MEAN_REVERT (se Hurst < 0.45)
# - STRATEGY_SMC_SCALPER (default)
# - STRATEGY_NONE (se unsafe)
```

---

### Cenário 4: "Quero Testar TODAS Combinações"

**Approach**: Grid search via script.

```python
# scripts/grid_search_strategies.py (a criar)
import itertools

configs = {
    'execution_threshold': [50, 60, 70, 80],
    'use_selector': [True, False],
    'use_mtf': [True, False],
    'use_footprint': [True, False],
}

for combo in itertools.product(*configs.values()):
    # Gera YAML temporário
    # Roda backtest
    # Salva resultados
```

**Total**: 4 × 2 × 2 × 2 = **32 combinações**

---

## 📦 Módulos Independentes

### Isolation Principle

Cada módulo é **self-contained**:

```
FootprintAnalyzer:
  ├── Inputs: bars, ticks
  ├── Logic: delta, imbalance, absorption
  ├── Outputs: score (0-100)
  └── Zero dependency on outros módulos ✅

RegimeDetector:
  ├── Inputs: price series
  ├── Logic: Hurst exponent, entropy
  ├── Outputs: Regime (TRENDING/REVERTING/RANDOM)
  └── Zero dependency on outros módulos ✅

StrategySelector:
  ├── Inputs: MarketContext (regime, session, news, circuit)
  ├── Logic: Decision tree (6 gates)
  ├── Outputs: StrategyType enum
  └── Depende apenas de MarketContext struct ✅
```

**Vantagem**: Pode **adicionar/remover módulos** sem quebrar outros.

---

## 🧪 Como Testar Módulos Isoladamente

### Teste 1: Footprint Isolado

```python
# tests/test_footprint_isolated.py
from src.signals.footprint_analyzer import FootprintAnalyzer

# Setup
footprint = FootprintAnalyzer(config)

# Feed bars
for bar in historical_bars:
    footprint.update(bar, tick_data)

# Check output
score = footprint.get_score()
assert 0 <= score <= 100
```

**Sem precisar**:
- Strategy completa
- Nautilus engine
- Outros módulos

---

### Teste 2: Regime Detector Isolado

```python
# tests/test_regime_detector.py
from src.indicators.regime_detector import RegimeDetector

detector = RegimeDetector(lookback=50)

for bar in bars:
    detector.on_bar(bar)

regime = detector.get_regime()
assert regime in [Regime.TRENDING, Regime.REVERTING, Regime.RANDOM]
```

---

## 🔌 Dependency Injection Pattern

### Como Strategy Recebe Módulos

```python
# gold_scalper_strategy.py:159-240
class GoldScalperStrategy:
    def __init__(self, config: GoldScalperConfig):
        self.config = config
        
        # Módulos inicializados em on_start()
        self._session_filter = None
        self._regime_detector = None
        self._footprint = None
        self._strategy_selector = None
    
    def on_start(self):
        # Dependency Injection baseado em config flags
        if self.config.use_session_filter:
            self._session_filter = SessionFilter(...)  # ← Injeta
        
        if self.config.use_regime_filter:
            self._regime_detector = RegimeDetector(...)  # ← Injeta
        
        # Se flag=False, módulo fica None
        # E checks no código são:
        if self._footprint:  # ← Safe check
            score += self._footprint.get_score()
```

**Vantagem**: Módulos **opcionales** sem código duplicado.

---

## 📊 Command Line Overrides

### CLI > YAML

Você pode **override** config YAML via CLI:

```bash
# Override threshold
python run_backtest.py --threshold 60

# Código em run_backtest.py:591-595
threshold = args.threshold if args.threshold is not None \
            else exec_cfg.get("execution_threshold", 70)
```

**Hierarchy**:
```
CLI args > YAML config > Código defaults
```

**Suportado atualmente**:
- `--threshold`: Execution threshold
- `--start/--end`: Período
- `--sample`: Sample rate
- `--no-news`: Desliga news filter

**Fácil adicionar mais**:
```python
# run_backtest.py:578-585
parser.add_argument('--no-footprint', action='store_true')
parser.add_argument('--no-selector', action='store_true')

# E depois
use_footprint = not args.no_footprint
use_selector = not args.no_selector
```

---

## 🏆 Best Practices

### 1. **Single Source of Truth**

✅ **BOM**:
```yaml
# strategy_config.yaml
execution:
  execution_threshold: 70
```

❌ **RUIM**:
```python
# Hardcoded em 5 lugares diferentes
THRESHOLD = 70
```

---

### 2. **Feature Flags**

✅ **BOM**:
```yaml
execution:
  use_footprint: false  # ← Desliga via config
```

❌ **RUIM**:
```python
# Comentar código manualmente
# footprint_score = self._footprint.get_score()
```

---

### 3. **Fail-Safe Checks**

✅ **BOM**:
```python
if self._footprint:  # ← Check if initialized
    score += self._footprint.get_score()
```

❌ **RUIM**:
```python
score += self._footprint.get_score()  # ← Crash se None
```

---

### 4. **Config Validation**

✅ **BOM**:
```python
if config.execution_threshold < 0 or config.execution_threshold > 100:
    raise ValueError("Threshold must be 0-100")
```

❌ **RUIM**:
```python
# Assume config está correto
```

---

## 🚀 Workflow Prático

### Cenário: "Testar 3 Configurações Diferentes"

**Setup 1: Conservative** (threshold 80)
```bash
cp configs/strategy_config.yaml configs/conservative.yaml
# Edit: execution_threshold: 80
python run_backtest.py --config configs/conservative.yaml --start 2024-01-01 --end 2024-12-31
```

**Setup 2: Balanced** (threshold 70, atual)
```bash
python run_backtest.py --start 2024-01-01 --end 2024-12-31
```

**Setup 3: Aggressive** (threshold 60, sem footprint)
```bash
cp configs/strategy_config.yaml configs/aggressive.yaml
# Edit: execution_threshold: 60, use_footprint: false
python run_backtest.py --config configs/aggressive.yaml --start 2024-01-01 --end 2024-12-31
```

**Comparar resultados**:
```bash
# Cada run gera logs/backtest_latest/metrics.json
# Comparar: Sharpe, DD, win rate, # trades
```

---

## 📚 Resumo (TL;DR)

| Pergunta | Resposta |
|----------|----------|
| **Quantos arquivos mudar?** | ✅ **1 arquivo**: `strategy_config.yaml` |
| **Como adicionar nova strategy?** | 1. Herdar `BaseStrategy` 2. Registrar em `StrategyType` 3. Add case em `StrategySelector` |
| **Como testar sem módulo X?** | Set `use_X: false` no YAML |
| **Como trocar threshold?** | CLI: `--threshold 60` OU YAML: `execution_threshold: 60` |
| **Precisa recompilar?** | ❌ Não, Python = interpreted |
| **Módulos são acoplados?** | ❌ Não, cada um é isolado |
| **Config é centralizada?** | ✅ Sim, `strategy_config.yaml` |

---

## 🔗 Arquivos Relevantes

```
nautilus_gold_scalper/
├── scripts/
│   └── run_backtest.py              # ← RUNNER (entry point)
├── configs/
│   └── strategy_config.yaml         # ← CONFIG (single source of truth)
├── src/
│   ├── strategies/
│   │   ├── gold_scalper_strategy.py # ← MAIN STRATEGY
│   │   ├── base_strategy.py         # ← Base class
│   │   └── strategy_selector.py     # ← Strategy selector
│   ├── signals/
│   │   ├── confluence_scorer.py     # ← Score system
│   │   ├── footprint_analyzer.py    # ← Footprint module
│   │   └── entry_optimizer.py       # ← Fib optimizer
│   ├── indicators/
│   │   ├── regime_detector.py       # ← Regime detection
│   │   └── session_filter.py        # ← Session detection
│   └── risk/
│       ├── prop_firm_manager.py     # ← Apex manager
│       ├── circuit_breaker.py       # ← Circuit breaker
│       └── time_constraint_manager.py  # ← Time cutoff
└── tests/
    └── test_*.py                    # ← Unit tests por módulo
```

---

**Próximo**: Ver `DOCS/05_GUIDES/USAGE/BACKTEST_WORKFLOW.md` para workflow completo de backtesting.

