# ORACLE v2.2 REFINEMENT PLAN
## "Institutional-Grade Statistical Validator"

**Data**: 2024-11-30
**Autor**: Droid (Factory CLI)
**Base**: DEEP_DIVE_BACKTESTING_MASTER.md (3799 linhas, 7 subtemas)
**Status**: AGUARDANDO APROVAÇÃO

---

## 1. EXECUTIVE SUMMARY

### Objetivo
Refinar o Oracle Backtest Commander de v2.1 para v2.2, integrando todo o conhecimento do deep dive research (MATRIX-level) para transformá-lo em um validador de nível institucional.

### Escopo
- **Manter**: Estrutura modular existente (SKILL.md + references.md + checklists.md)
- **Aprimorar**: Conteúdo de cada arquivo com conhecimento do deep dive
- **Criar**: Scripts Python production-ready baseados no código do relatório
- **Preservar**: Backup dos arquivos atuais antes de modificar

### Resultado Esperado
Oracle capaz de realizar validação GO/NO-GO com o mesmo rigor de quant funds institucionais, incluindo:
- Walk-Forward Analysis com Purged CV
- Monte Carlo Block Bootstrap
- Deflated Sharpe Ratio (PSR/DSR/PBO)
- Simulação de execução realista
- Validação específica para Prop Firms (FTMO)
- Confidence Score automatizado (0-100)

---

## 2. GAP ANALYSIS DETALHADO

### 2.1 Comparativo de Capabilities

| Capability | Oracle v2.1 | Deep Dive Report | Gap Level | Ação |
|------------|-------------|------------------|-----------|------|
| **WFA Implementation** | Workflow básico em texto | Classe WalkForwardAnalyzer completa | 🔴 HIGH | Implementar classe |
| **WFA Rolling vs Anchored** | Mencionado | Algoritmos detalhados + recomendação | 🟡 MEDIUM | Documentar + código |
| **Purged K-Fold CV** | Não existe | Lopez de Prado implementation | 🔴 HIGH | Adicionar |
| **CPCV (Combinatorial)** | Não existe | Full implementation | 🔴 HIGH | Adicionar |
| **Monte Carlo** | Menção "5000 runs" | MonteCarloBlockBootstrap completo | 🔴 HIGH | Implementar classe |
| **Block Size Optimization** | Não existe | Politis & White formula | 🔴 HIGH | Adicionar |
| **PSR Calculation** | Fórmula básica | SharpeAnalyzer com skew/kurtosis | 🟡 MEDIUM | Aprimorar |
| **DSR Calculation** | Fórmula básica | Com E[max(SR)] para N trials | 🟡 MEDIUM | Aprimorar |
| **PBO (Probability of Overfit)** | Não existe | Full implementation | 🔴 HIGH | Adicionar |
| **MinTRL** | Não existe | Minimum Track Record Length | 🟡 MEDIUM | Adicionar |
| **Execution Simulation** | Não existe em Oracle | ExecutionSimulator completo | 🔴 HIGH | Criar |
| **Slippage Model** | Não existe | Dinâmico por condição de mercado | 🔴 HIGH | Criar |
| **Spread Model XAUUSD** | Não existe | Session-aware com volatility | 🔴 HIGH | Criar |
| **Latency Simulation** | Não existe | Log-normal + spikes | 🟡 MEDIUM | Criar |
| **MT5 Trade Export** | Não existe | MT5TradeExporter class | 🟡 MEDIUM | Criar |
| **GO/NO-GO Pipeline** | Checklist básico | GoNoGoValidator automatizado | 🔴 HIGH | Criar |
| **Prop Firm Validation** | FTMO básico | Framework completo + daily DD | 🔴 HIGH | Expandir |
| **Confidence Score** | Não existe | Algoritmo 0-100 automatizado | 🟡 MEDIUM | Adicionar |
| **4-Level Robustness** | Não existe | Build Alpha framework | 🔴 HIGH | Adicionar |
| **VaR / CVaR** | Não existe | Value at Risk calculations | 🟡 MEDIUM | Adicionar |

### 2.2 Contagem de Gaps

- 🔴 **HIGH**: 13 gaps críticos
- 🟡 **MEDIUM**: 8 gaps importantes
- 🟢 **LOW**: 0 gaps menores

**Total**: 21 melhorias a implementar

---

## 3. ARQUITETURA DA SOLUÇÃO

### 3.1 Estrutura de Arquivos (Antes vs Depois)

```
ANTES (Oracle v2.1):                    DEPOIS (Oracle v2.2):
.factory/skills/oracle/                 .factory/skills/oracle/
├── SKILL.md (5.0 KB)                   ├── SKILL.md (~12 KB) [ENHANCED]
├── references.md (3.5 KB)              ├── references.md (~8 KB) [ENHANCED]
└── checklists.md (5.2 KB)              └── checklists.md (~10 KB) [ENHANCED]
                                        
scripts/oracle/                         scripts/oracle/
├── deflated_sharpe.py (básico)         ├── deflated_sharpe.py [ENHANCED]
├── monte_carlo.py (básico)             ├── monte_carlo.py [ENHANCED]
├── walk_forward.py (básico)            ├── walk_forward.py [ENHANCED]
└── (3 arquivos)                        ├── go_nogo_validator.py [NEW]
                                        ├── execution_simulator.py [NEW]
                                        ├── mt5_trade_exporter.py [NEW]
                                        ├── prop_firm_validator.py [NEW]
                                        └── (7 arquivos)
```

### 3.2 Fluxo de Dados (Pipeline Híbrido)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ORACLE v2.2 VALIDATION PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐     ┌────────────────┐     ┌──────────────────────────┐  │
│  │   MT5        │     │   Export CSV   │     │   Python Validation      │  │
│  │   Strategy   │ ──► │   via Python   │ ──► │   Pipeline               │  │
│  │   Tester     │     │   API          │     │                          │  │
│  └──────────────┘     └────────────────┘     │  1. Load & Preprocess    │  │
│        │                     │               │  2. Walk-Forward (WFA)   │  │
│        ▼                     ▼               │  3. Monte Carlo Block    │  │
│  [Backtest com         [mt5_trade_          │  4. PSR/DSR/PBO          │  │
│   ONNX + spread         exporter.py]        │  5. Execution Costs      │  │
│   + slippage]                               │  6. Prop Firm Check      │  │
│                                             │  7. Confidence Score     │  │
│                                             └──────────────────────────┘  │
│                                                        │                   │
│                                                        ▼                   │
│                                             ┌──────────────────────────┐  │
│                                             │   GO/NO-GO Decision      │  │
│                                             │   + Report Generation    │  │
│                                             └──────────────────────────┘  │
│                                                        │                   │
│                                                        ▼                   │
│                                             ┌──────────────────────────┐  │
│                                             │ DOCS/04_REPORTS/         │  │
│                                             │ VALIDATION/report.md     │  │
│                                             └──────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. DETALHAMENTO DAS MUDANÇAS

### 4.1 SKILL.md - Mudanças Planejadas

#### 4.1.1 Header/Identity (ENHANCE)

```diff
- ORACLE - The Statistical Truth-Seeker v2.1 (PROATIVO)
+ ORACLE - The Statistical Truth-Seeker v2.2 (INSTITUTIONAL-GRADE)

+ NAO ESPERA COMANDOS - Monitora conversa e INTERVEM automaticamente:
+ (manter triggers existentes)
+ NOVO: Integra 7 subtemas do MATRIX Research
+ NOVO: Confidence Score automatizado (0-100)
+ NOVO: 4-Level Robustness Testing
```

#### 4.1.2 Core Principles (ENHANCE)

```diff
Manter os 10 Mandamentos existentes
+ ADICIONAR:
+ 11. **BLOCK BOOTSTRAP OBRIGATÓRIO** - Preserva autocorrelação temporal
+ 12. **DSR > 0 OU NO-GO** - Sharpe deve sobreviver deflation
+ 13. **PBO < 0.25** - Probabilidade de overfit aceitável
+ 14. **EQUITY-BASED DD** - FTMO usa equity, não balance
+ 15. **CONFIDENCE >= 70** - Score mínimo para GO
```

#### 4.1.3 Thresholds (ENHANCE)

```diff
Manter tabela existente
+ ADICIONAR novas métricas:

| Metrica | Minimo | Target | Red Flag |
|---------|--------|--------|----------|
+ | PBO | < 0.50 | < 0.25 | > 0.50 |
+ | MinTRL | > N trades | - | < N/2 |
+ | Confidence | >= 70 | >= 85 | < 50 |
+ | VaR 95% | < 8% | < 5% | > 10% |
+ | CVaR 95% | < 10% | < 7% | > 12% |
+ | P(Daily DD>5%) | < 5% | < 2% | > 10% |
+ | P(Total DD>10%) | < 2% | < 1% | > 5% |
```

#### 4.1.4 Commands (ADD NEW)

```diff
Manter comandos existentes: /validar, /wfa, /montecarlo, /overfitting, /metricas, /go-nogo, /ftmo, /bias, /comparar, /robustez

+ ADICIONAR:
+ | /propfirm | Validação específica FTMO (daily DD, equity-based) |
+ | /confidence | Calcular Confidence Score detalhado |
+ | /export | Exportar trades do MT5 para CSV |
+ | /pbo | Calcular Probability of Backtest Overfitting |
+ | /execution | Simular custos de execução realistas |
+ | /pipeline | Executar pipeline completo GO/NO-GO |
```

#### 4.1.5 Workflows (MAJOR ENHANCE)

**Workflow /validar (REWRITE)**

```
NOVO PIPELINE 7-STEP:

STEP 1: LOAD & PREPROCESS
├── Carregar trades (CSV ou MT5 export)
├── Validar formato e colunas
├── Calcular métricas básicas
└── Verificar amostra mínima (>= 100 trades)

STEP 2: WALK-FORWARD ANALYSIS
├── Configurar: Rolling, 15 windows, 70/30, purge 5 bars
├── Executar WalkForwardAnalyzer
├── Calcular WFE por janela
├── Threshold: WFE >= 0.6

STEP 3: MONTE CARLO BLOCK BOOTSTRAP
├── Configurar: 5000 runs, block_size = n^(1/3)
├── Executar MonteCarloBlockBootstrap
├── Calcular distribuição de DD
├── Threshold: 95th DD < 8%

STEP 4: OVERFITTING DETECTION
├── Calcular PSR (>= 0.90)
├── Calcular DSR (> 0)
├── Calcular PBO (< 0.25)
├── Calcular MinTRL

STEP 5: EXECUTION COST ANALYSIS
├── Aplicar ExecutionSimulator (PESSIMISTIC)
├── Recalcular métricas com custos
├── Verificar se ainda passa thresholds

STEP 6: PROP FIRM VALIDATION
├── Calcular P(Daily DD > 5%)
├── Calcular P(Total DD > 10%)
├── Simular 10 losing streak
├── Threshold: P(breach) < 5%

STEP 7: CONFIDENCE SCORE & DECISION
├── Calcular score 0-100
├── Compilar todos resultados
├── Emitir GO / CAUTION / NO-GO
├── Gerar relatório completo
```

#### 4.1.6 4-Level Robustness Framework (ADD NEW SECTION)

```
## 4-Level Robustness Testing

### LEVEL 1 - BASELINE (Obrigatório)
□ Out-of-Sample Testing (30% holdout)
□ Walk-Forward Analysis (15+ windows, WFE >= 0.6)
□ 200+ trades, 2+ anos de dados

### LEVEL 2 - ADVANCED (Recomendado)
□ PSR > 0.90
□ DSR > 0 (ajustado por N trials)
□ PBO < 0.25
□ Noise Test: 80%+ mantém performance

### LEVEL 3 - PROP FIRMS (Para FTMO)
□ P(Daily DD > 5%) < 5%
□ P(Total DD > 10%) < 2%
□ Spread widening test (+50%)
□ 10 losing streak não viola DD

### LEVEL 4 - INSTITUTIONAL (Opcional)
□ CPCV (Combinatorial Purged CV)
□ Multiple regime testing
□ Stress scenarios (flash crash, news)
□ Market impact simulation
```

#### 4.1.7 Confidence Score System (ADD NEW SECTION)

```
## Confidence Score (0-100)

| Componente | Pontos | Critério |
|------------|--------|----------|
| WFA Pass | 25 | WFE >= 0.6 |
| Monte Carlo Pass | 25 | 95th DD < 8% |
| Sharpe Pass | 20 | PSR >= 0.90 AND DSR > 0 |
| Prop Firm Pass | 20 | P(breach) < 5% |
| Warnings | -5 each | Por warning detectado |

**Interpretação:**
- 85-100: STRONG GO
- 70-84: GO
- 50-69: INVESTIGATE
- < 50: NO-GO
```

---

### 4.2 references.md - Mudanças Planejadas

#### 4.2.1 Scripts Python (ENHANCE)

```diff
Manter seção existente

+ ADICIONAR detalhes de cada script:

## Scripts Python Detalhados

### walk_forward.py
- Classe: WalkForwardAnalyzer
- Config: WFAType (ROLLING/ANCHORED), n_windows, is_ratio, purge_gap, embargo_pct
- Output: WFAResult com WFE, windows details, is_robust flag
- Uso: python -m scripts.oracle.walk_forward --input trades.csv --windows 15

### monte_carlo.py
- Classe: MonteCarloBlockBootstrap
- Config: n_simulations, block_size (auto ou manual), confidence_levels
- Output: MonteCarloResult com distribuições, VaR, CVaR, P(ruin)
- Uso: python -m scripts.oracle.monte_carlo --input trades.csv --runs 5000

### deflated_sharpe.py
- Classe: SharpeAnalyzer
- Calcula: PSR, DSR, MinTRL, Expected Max Sharpe
- Output: SharpeAnalysisResult com interpretação
- Uso: python -m scripts.oracle.deflated_sharpe --input returns.csv --trials 10

### go_nogo_validator.py [NEW]
- Classe: GoNoGoValidator
- Integra: WFA + MC + Sharpe + Execution + PropFirm
- Output: ValidationResult com decision, confidence, reasons
- Uso: python -m scripts.oracle.go_nogo_validator --input trades.csv

### execution_simulator.py [NEW]
- Classe: ExecutionSimulator
- Config: ExecutionConfig (slippage, spread, latency, rejection)
- Modes: DEV, VALIDATION, STRESS
- Output: ExecutionResult com custos detalhados

### mt5_trade_exporter.py [NEW]
- Classe: MT5TradeExporter
- Funções: connect, export_deals, export_paired_trades, save_to_csv
- Requer: MetaTrader5 Python package

### prop_firm_validator.py [NEW]
- Classe: PropFirmValidator
- Específico: FTMO rules (daily DD equity-based)
- Calcula: P(daily breach), P(total breach), position sizing
```

#### 4.2.2 Fórmulas Completas (ADD NEW SECTION)

```
## Fórmulas Matemáticas

### Sharpe Ratio (Anualizado)
SR = sqrt(252) * mean(returns) / std(returns)

### Probabilistic Sharpe Ratio (PSR)
PSR = Φ[(SR_obs - SR*) * sqrt(n-1) / sqrt(1 + 0.5*SR² - γ₃*SR + (γ₄-3)/4 * SR²)]

Onde:
- Φ = CDF da normal padrão
- γ₃ = skewness
- γ₄ = kurtosis
- n = número de observações

### Expected Max Sharpe (sob H0)
E[max(SR)] ≈ sqrt(2 * ln(N)) - (γ + ln(ln(N))) / (2 * sqrt(2 * ln(N)))

Onde:
- N = número de trials/estratégias testadas
- γ = 0.5772... (constante de Euler-Mascheroni)

### Deflated Sharpe Ratio (DSR)
DSR = (SR_obs - E[max(SR)]) / SE(SR)

DSR > 0 significa que o Sharpe sobrevive ao ajuste por múltiplos testes

### Walk-Forward Efficiency (WFE)
WFE = mean(OOS_performance) / mean(IS_performance)

### Probability of Backtest Overfitting (PBO)
PBO = (1 - rank_correlation(IS, OOS)) / 2

### Minimum Track Record Length (MinTRL)
MinTRL = z² * (1 + 0.5*SR² - γ₃*SR + (γ₄-3)/4 * SR²) / (SR - SR*)² + 1

### Value at Risk (VaR)
VaR_α = percentile(DD_distribution, α * 100)

### Conditional VaR (CVaR / Expected Shortfall)
CVaR_α = mean(DD | DD >= VaR_α)

### Optimal Block Size (Politis & White)
block_size = n^(1/3)

### FTMO Daily DD (Equity-Based)
Daily_DD = (Start_of_Day_Balance - Current_Equity) / Initial_Balance * 100
```

#### 4.2.3 Configurações Template (ADD NEW SECTION)

```
## Configuration Templates

### WFA Config (Recommended)
```python
WFA_CONFIG = {
    "type": "rolling",
    "n_windows": 15,
    "is_ratio": 0.75,
    "overlap": 0.20,
    "purge_gap": 0.02,
    "embargo_pct": 0.01,
    "min_trades_per_window": 30,
    "min_wfe": 0.6
}
```

### Monte Carlo Config (Recommended)
```python
MC_CONFIG = {
    "n_simulations": 5000,
    "block_size": "auto",  # n^(1/3)
    "confidence_levels": [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
    "initial_balance": 100000
}
```

### Execution Config (PESSIMISTIC - For FTMO)
```python
EXEC_CONFIG = {
    "base_slippage": 5.0,
    "slippage_news_mult": 10.0,
    "adverse_only": True,
    "base_spread": 25.0,
    "spread_asian_mult": 3.0,
    "base_latency": 100,
    "spike_probability": 0.15,
    "base_rejection_prob": 0.10
}
```
```

---

### 4.3 checklists.md - Mudanças Planejadas

#### 4.3.1 GO/NO-GO Checklist (MAJOR ENHANCE)

```diff
Manter estrutura existente

+ ADICIONAR novos itens:

### 5. OVERFITTING DETECTION (NEW)
□ PSR calculado?
□ PSR >= 0.90?
□ DSR calculado?
□ DSR > 0?
□ PBO calculado?
□ PBO < 0.25?
□ MinTRL verificado?
□ N trades > MinTRL?

### 6. PROP FIRM SPECIFIC (ENHANCE)
□ P(Daily DD > 5%) calculado?
□ P(Daily DD > 5%) < 5%?
□ P(Total DD > 10%) calculado?
□ P(Total DD > 10%) < 2%?
□ Simulou 10 losing streak?
□ 10 losses não viola DD?
□ Spread widening (+50%) testado?

### 7. CONFIDENCE SCORE (NEW)
□ WFA Component: ___/25
□ Monte Carlo Component: ___/25
□ Sharpe Component: ___/20
□ Prop Firm Component: ___/20
□ Warnings Penalty: ___
□ TOTAL SCORE: ___/100
□ Score >= 70?
```

#### 4.3.2 4-Level Robustness Checklist (ADD NEW)

```
## 4-Level Robustness Testing Checklist

### LEVEL 1 - BASELINE (Obrigatório para qualquer GO)
□ Out-of-Sample Testing (30% holdout)?
□ Walk-Forward Analysis feito?
□ WFE >= 0.6?
□ 200+ trades na amostra?
□ 2+ anos de dados?
□ Diferentes regimes incluídos?

LEVEL 1 PASS: □ YES □ NO

### LEVEL 2 - ADVANCED (Recomendado para FTMO)
□ PSR > 0.90?
□ DSR > 0?
□ PBO < 0.25?
□ Noise Test executado?
□ 80%+ performance mantida com ruído?
□ Múltiplas janelas temporais testadas?

LEVEL 2 PASS: □ YES □ NO

### LEVEL 3 - PROP FIRMS (Obrigatório para FTMO)
□ P(Daily DD > 5%) < 5%?
□ P(Total DD > 10%) < 2%?
□ Spread widening +50% testado?
□ 10 losing streak simulado?
□ Position sizing = max 1% risk?
□ Praticou em demo/free trial?

LEVEL 3 PASS: □ YES □ NO

### LEVEL 4 - INSTITUTIONAL (Opcional - Para Scaling)
□ CPCV (Combinatorial Purged CV)?
□ Multiple regime testing?
□ Stress scenarios testados?
□ Market impact simulado?
□ Execution costs pessimistas?
□ Slippage adverso modelado?

LEVEL 4 PASS: □ YES □ NO

### RESULTADO
□ LEVEL 1 PASS → Pode considerar paper trading
□ LEVEL 1+2 PASS → Pode considerar demo
□ LEVEL 1+2+3 PASS → Pode iniciar FTMO Challenge
□ LEVEL 1+2+3+4 PASS → Institutional-grade ready
```

#### 4.3.3 Anti-Overfitting Checklist (ADD NEW)

```
## Anti-Overfitting Checklist (10 Pontos)

ANTES DE CONFIAR EM UM BACKTEST:

□ 1. Dados OOS genuínos (nunca vistos durante desenvolvimento)?
□ 2. WFA com WFE >= 0.6?
□ 3. Monte Carlo 95th DD < 8%?
□ 4. PSR > 0.90?
□ 5. DSR > 0 (ajustado por N testes)?
□ 6. PBO < 0.25?
□ 7. Número de parâmetros <= 4?
□ 8. Mais de 200 trades na amostra?
□ 9. Mais de 2 anos de dados?
□ 10. Lógica econômica faz sentido?

CONTAGEM: ___/10

SE < 8 "SIM" → SUSPEITAR DE OVERFIT
SE < 5 "SIM" → OVERFIT MUITO PROVÁVEL
```

#### 4.3.4 Backtest Quality Checklist (ADD NEW)

```
## Backtest Quality Checklist

### QUALIDADE DOS DADOS
□ Tick data ou OHLC de qualidade?
□ Spread realista incluído?
□ Slippage modelado?
□ Comissão incluída?
□ Swap/rollover considerado?

### EXECUÇÃO
□ Every tick ou Open prices?
□ Execução em close of bar?
□ Requote/rejection modelado?
□ Latência considerada?

### PERÍODO
□ >= 2 anos de dados?
□ Inclui volatilidade alta (2020, 2022)?
□ Inclui volatilidade baixa?
□ Diferentes regimes macro?

### CONFIGURAÇÃO MT5
□ Modeling: Every tick based on real ticks?
□ Spread: Current ou Custom realista?
□ Commission: Igual ao broker real?
□ Initial deposit: $100,000?
□ Leverage: 1:30 (FTMO)?
```

#### 4.3.5 Pre-Challenge Checklist (ADD NEW)

```
## Pre-Challenge Checklist (FTMO)

ANTES DE INICIAR QUALQUER PROP FIRM CHALLENGE:

### VALIDAÇÃO ESTATÍSTICA
□ GO/NO-GO checklist completo e PASS?
□ Todas métricas dentro dos thresholds?
□ WFA aprovado (WFE >= 0.6)?
□ Monte Carlo aprovado (95th DD < 8%)?
□ Overfitting descartado (PSR >= 0.90, DSR > 0)?
□ Confidence Score >= 70?

### PREPARAÇÃO TÉCNICA
□ EA compilado sem erros?
□ VPS estável configurado?
□ Broker correto selecionado?
□ Symbol = XAUUSD verificado?
□ Parâmetros = backtest aprovado (EXATOS)?

### RISK MANAGEMENT
□ Risk per trade definido (0.5-1%)?
□ Max daily DD interno = 4% (buffer)?
□ Circuit breakers ativos no EA?
□ Emergency mode testado?

### TIMING
□ Começar segunda-feira (não sexta)?
□ Evitar semana de FOMC/NFP inicial?
□ Primeiros dias: observar apenas?

### MENTAL
□ Preparado para drawdown?
□ Não vai interferir manualmente?
□ Confiança no sistema validado?

TOTAL: ___/20
SE < 18 → NÃO INICIAR CHALLENGE
```

---

## 5. SCRIPTS PYTHON - ESPECIFICAÇÕES

### 5.1 walk_forward.py (ENHANCE)

**Tamanho Estimado**: ~300 linhas
**Classes**: WalkForwardAnalyzer, WFAWindow, WFAResult, WFAType
**Fonte**: Deep Dive Subtema 1 (linhas 1-500)

**Funcionalidades**:
- Rolling e Anchored WFA
- Purge gap e embargo support
- WFE calculation por janela e agregado
- Report generation em Markdown
- Critérios de robustez automáticos

### 5.2 monte_carlo.py (ENHANCE)

**Tamanho Estimado**: ~250 linhas
**Classes**: MonteCarloBlockBootstrap, MonteCarloConfig, MonteCarloResult
**Fonte**: Deep Dive Subtema 2 (linhas 500-1000)

**Funcionalidades**:
- Block bootstrap (preserva autocorrelação)
- Optimal block size (Politis & White)
- Distribuições de DD e Equity
- VaR e CVaR calculation
- Probability of ruin
- Confidence score component
- Report generation

### 5.3 deflated_sharpe.py (ENHANCE)

**Tamanho Estimado**: ~200 linhas
**Classes**: SharpeAnalyzer, SharpeAnalysisResult
**Fonte**: Deep Dive Subtema 3 (linhas 1000-1400)

**Funcionalidades**:
- PSR with skewness/kurtosis adjustment
- DSR with E[max(SR)] for N trials
- MinTRL calculation
- PBO calculation
- Interpretation strings
- Report generation

### 5.4 go_nogo_validator.py (NEW)

**Tamanho Estimado**: ~400 linhas
**Classes**: GoNoGoValidator, ValidationCriteria, ValidationResult, Decision
**Fonte**: Deep Dive Subtema 5 (linhas 1700-2200)

**Funcionalidades**:
- Integra todos os validadores
- 7-step pipeline
- Confidence score calculation
- Decision: GO / CAUTION / NO-GO
- Full report generation
- CLI interface

### 5.5 execution_simulator.py (NEW)

**Tamanho Estimado**: ~350 linhas
**Classes**: ExecutionSimulator, ExecutionConfig, ExecutionResult, MarketCondition
**Fonte**: Deep Dive Subtema 4 (linhas 1400-1700)

**Funcionalidades**:
- Dynamic slippage model
- Session-aware spread (XAUUSD specific)
- Latency simulation (log-normal + spikes)
- Order rejection simulation
- Statistics tracking
- Apply to trades DataFrame
- Report generation

### 5.6 mt5_trade_exporter.py (NEW)

**Tamanho Estimado**: ~200 linhas
**Classes**: MT5TradeExporter
**Fonte**: Deep Dive Subtema 5 (linhas 2200-2500)

**Funcionalidades**:
- Connect to MT5 terminal
- Export deals from history
- Pair entries with exits
- Save to CSV with metadata
- CLI interface

### 5.7 prop_firm_validator.py (NEW)

**Tamanho Estimado**: ~250 linhas
**Classes**: PropFirmValidator, FTMORules
**Fonte**: Deep Dive Subtema 6 (linhas 2800-3200)

**Funcionalidades**:
- FTMO-specific rules
- Daily DD calculation (equity-based)
- Total DD tracking
- Probability of breach calculation
- Position sizing recommendations
- Pre-challenge checklist validation

---

## 6. ORDEM DE IMPLEMENTAÇÃO

### Fase 1: Documentação (Skill Files)
1. ✅ Criar este plano de implementação
2. [ ] Aguardar aprovação do usuário
3. [ ] Backup dos arquivos atuais
4. [ ] Atualizar SKILL.md
5. [ ] Atualizar references.md
6. [ ] Atualizar checklists.md

### Fase 2: Scripts Core
7. [ ] Implementar walk_forward.py (enhance)
8. [ ] Implementar monte_carlo.py (enhance)
9. [ ] Implementar deflated_sharpe.py (enhance)

### Fase 3: Scripts Novos
10. [ ] Implementar execution_simulator.py (new)
11. [ ] Implementar prop_firm_validator.py (new)
12. [ ] Implementar mt5_trade_exporter.py (new)

### Fase 4: Integração
13. [ ] Implementar go_nogo_validator.py (new)
14. [ ] Testar pipeline completo
15. [ ] Documentar usage examples

### Fase 5: Validação
16. [ ] Verificar todos os scripts executam sem erro
17. [ ] Verificar skill carrega corretamente no Factory
18. [ ] Criar exemplo de uso end-to-end

---

## 7. MÉTRICAS DE SUCESSO

### Antes (Oracle v2.1)
- Total Size: 13.7 KB
- Scripts: 3 básicos
- Checklists: 5
- Fórmulas documentadas: 6
- Automation: Manual

### Depois (Oracle v2.2)
- Total Size: ~30 KB
- Scripts: 7 production-ready
- Checklists: 12+
- Fórmulas documentadas: 15+
- Automation: Confidence Score + GO/NO-GO Pipeline

### Improvement
- Coverage: Basic → Institutional
- Robustness Testing: 1 level → 4 levels
- Prop Firm: Basic → Complete Framework
- Confidence: Manual → Automated 0-100 Score

---

## 8. RISCOS E MITIGAÇÕES

| Risco | Impacto | Mitigação |
|-------|---------|-----------|
| Quebrar skill existente | HIGH | Backup completo antes de modificar |
| Scripts não executam | MEDIUM | Testar cada um individualmente |
| Tamanho excessivo | LOW | Manter modular, usar references |
| Complexidade alta | MEDIUM | Documentação clara, exemplos |

---

## 9. APROVAÇÃO

**Status**: AGUARDANDO APROVAÇÃO DO USUÁRIO

### Para aprovar, confirme:
1. [ ] Escopo está correto
2. [ ] Ordem de implementação OK
3. [ ] Pode prosseguir com a implementação

---

*Documento criado por Droid (Factory CLI)*
*Baseado em DEEP_DIVE_BACKTESTING_MASTER.md*
*Data: 2024-11-30*
