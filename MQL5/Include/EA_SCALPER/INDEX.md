# EA_SCALPER_XAUUSD v3.20 - Singularity MTF Edition
## Indice Completo da Arquitetura

---

## 1. Visao Geral

### 1.1 O Que E Este Robo?

O **EA_SCALPER_XAUUSD** e um Expert Advisor (robo de trading) desenvolvido especificamente para scalping em XAUUSD (Ouro) na plataforma MetaTrader 5. Ele combina:

- **Smart Money Concepts (SMC)** - Metodologia institucional de trading
- **Machine Learning (ONNX)** - Modelos de direcao treinados em Python
- **Multi-Timeframe Analysis (MTF)** - H1/M15/M5 para precisao maxima
- **FTMO Compliance** - Regras rigorosas para prop firms

### 1.2 Arquitetura Multi-Timeframe (v3.20)

```
┌─────────────────────────────────────────────────────────────────┐
│                    ARQUITETURA MTF v3.20                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   H1 (HTF)  ════════════════════════════════════════════════   │
│   │ Funcao: Filtro de Direcao                                  │
│   │ REGRA: NUNCA operar contra a tendencia do H1               │
│   │ Analise: Trend (bullish/bearish/ranging)                   │
│   └──────────────────────────────────────────────────────────   │
│                          │                                      │
│                          ▼                                      │
│   M15 (MTF) ════════════════════════════════════════════════   │
│   │ Funcao: Identificacao de Zonas Estruturais                 │
│   │ Analise: Order Blocks, FVGs, Liquidity Pools               │
│   │ REGRA: Trade apenas quando preco toca zona M15             │
│   └──────────────────────────────────────────────────────────   │
│                          │                                      │
│                          ▼                                      │
│   M5 (LTF)  ════════════════════════════════════════════════   │
│   │ Funcao: Execucao Precisa                                   │
│   │ Analise: Confirmacao de entrada, ATR para SL               │
│   │ REGRA: Entry candle deve confirmar direcao                 │
│   └──────────────────────────────────────────────────────────   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Vantagens do MTF

| Aspecto | Single TF (M15) | MTF (H1+M15+M5) |
|---------|-----------------|-----------------|
| Oportunidades/dia | 5-12 | 15-30 |
| Tamanho do SL | 40-60 pts | 25-40 pts |
| Custo de Spread | 2.5-5% | 5-8% |
| Win Rate Esperado | 60-65% | 70-75% |
| R:R Medio | 1.5:1 | 2.0-2.5:1 |

---

## 2. Estrutura de Pastas

```
MQL5/
├── Experts/
│   └── EA_SCALPER_XAUUSD.mq5          # EA Principal (479 linhas)
│
├── Include/EA_SCALPER/
│   ├── Core/                           # Nucleo do Sistema
│   │   ├── Definitions.mqh             # Enums e estruturas globais
│   │   ├── CState.mqh                  # Maquina de estados
│   │   └── CEngine.mqh                 # Motor principal
│   │
│   ├── Analysis/                       # Modulos de Analise
│   │   ├── CMTFManager.mqh             # [NEW v3.20] Gerenciador MTF
│   │   ├── CStructureAnalyzer.mqh      # BOS/CHoCH/Swing Points
│   │   ├── EliteOrderBlock.mqh         # Detector de Order Blocks
│   │   ├── EliteFVG.mqh                # Detector de Fair Value Gaps
│   │   ├── CLiquiditySweepDetector.mqh # Detector de Sweeps
│   │   ├── CRegimeDetector.mqh         # Hurst + Entropy
│   │   ├── CAMDCycleTracker.mqh        # Ciclo AMD
│   │   ├── CSessionFilter.mqh          # Filtro de Sessoes
│   │   ├── CNewsFilter.mqh             # Filtro de Noticias
│   │   ├── CEntryOptimizer.mqh         # Otimizador de Entrada
│   │   └── InstitutionalLiquidity.mqh  # Liquidez Institucional
│   │
│   ├── Signal/                         # Geracao de Sinais
│   │   ├── CConfluenceScorer.mqh       # Scorer de Confluencia
│   │   └── SignalScoringModule.mqh     # Modulo de Scoring
│   │
│   ├── Risk/                           # Gerenciamento de Risco
│   │   ├── FTMO_RiskManager.mqh        # Compliance FTMO
│   │   └── CDynamicRiskManager.mqh     # Risco Dinamico
│   │
│   ├── Execution/                      # Execucao de Ordens
│   │   ├── CTradeManager.mqh           # Gerenciador de Trades
│   │   └── TradeExecutor.mqh           # Executor Legacy
│   │
│   ├── Bridge/                         # Integracao Externa
│   │   ├── COnnxBrain.mqh              # Cerebro ONNX (ML)
│   │   ├── PythonBridge.mqh            # Ponte Python
│   │   └── CFundamentalsBridge.mqh     # [NEW] Ponte Fundamentals
│   │
│   ├── Signal/                         # Geracao de Sinais
│   │   ├── CConfluenceScorer.mqh       # Scorer de Confluencia
│   │   ├── SignalScoringModule.mqh     # Modulo de Scoring
│   │   └── CFundamentalsIntegrator.mqh # [NEW] Integracao Tech+Fund
│   │
│   ├── Modules/                        # Modulos Auxiliares
│   │   ├── Hub/
│   │   │   ├── CHubConnector.mqh       # Conector do Hub
│   │   │   └── CHeartbeat.mqh          # Heartbeat
│   │   └── Persistence/
│   │       └── CLocalCache.mqh         # Cache Local
│   │
│   └── Utils/                          # Utilitarios
│       └── CJson.mqh                   # Parser JSON
│
└── Models/                             # Modelos ONNX
    ├── direction_model_final.onnx      # Modelo de Direcao
    └── scaler_params_final.json        # Parametros do Scaler
```

---

## 3. Modulos de Analise (Analysis/)

### 3.1 CMTFManager.mqh - Gerenciador Multi-Timeframe [NEW v3.20]

**Proposito**: Coordenar analise em 3 timeframes (H1, M15, M5)

**Estruturas Principais**:
```cpp
struct SHTFAnalysis {     // Analise H1
   ENUM_MTF_TREND trend;  // Bullish/Bearish/Neutral/Ranging
   double strength;       // Forca da tendencia (0-100)
   double atr;           // ATR para sizing
};

struct SMTFAnalysis {     // Analise M15
   bool has_ob_zone;      // Tem Order Block?
   bool has_fvg_zone;     // Tem Fair Value Gap?
   bool has_liquidity;    // Tem Liquidity Pool?
   double zone_high;      // Topo da zona
   double zone_low;       // Fundo da zona
};

struct SLTFAnalysis {     // Analise M5
   bool bullish_confirmation;  // Candle bullish?
   bool bearish_confirmation;  // Candle bearish?
   double entry_price;         // Preco de entrada
   double optimal_sl;          // SL baseado em M5 ATR
};

struct SMTFConfluence {   // Resultado Final
   ENUM_MTF_ALIGNMENT alignment;  // Perfect/Good/Weak/None
   int aligned_count;             // Quantos TFs alinhados
   double confidence;             // Confianca (0-100)
   double position_multiplier;    // Multiplicador de posicao
};
```

**Metodos Principais**:
| Metodo | Descricao |
|--------|-----------|
| `Init(symbol)` | Inicializa handles de indicadores |
| `Update()` | Atualiza analise de todos os TFs |
| `GetConfluence()` | Retorna score de confluencia MTF |
| `CanTradeLong()` | Verifica se pode comprar (H1 bullish) |
| `CanTradeShort()` | Verifica se pode vender (H1 bearish) |
| `GetPositionMultiplier()` | 100%, 75%, 50%, 0% baseado em alinhamento |

**Logica de Alinhamento**:
```
3 TFs alinhados → PERFECT  → 100% position size
2 TFs alinhados → GOOD     → 75% position size
1 TF alinhado   → WEAK     → 50% position size
0 TFs alinhados → NONE     → 0% (NAO OPERA)
```

---

### 3.2 CStructureAnalyzer.mqh - Analisador de Estrutura

**Proposito**: Detectar Break of Structure (BOS), Change of Character (CHoCH), e Swing Points

**Conceitos SMC**:
```
ESTRUTURA BULLISH:          ESTRUTURA BEARISH:
                            
    HH ←── Higher High          LH ←── Lower High
   /  \                        /  \
  /    \                      /    \
 /      \                    /      \
HL ←── Higher Low           /        \
                           /          \
                          LL ←── Lower Low

BOS = Break of Structure (continuacao)
CHoCH = Change of Character (reversao)
```

**Estruturas Principais**:
```cpp
struct SSwingPoint {
   double price;
   datetime time;
   ENUM_SWING_TYPE type;    // SWING_HIGH ou SWING_LOW
   bool is_broken;          // Foi quebrado?
   int strength;            // Quantas velas confirmam
};

struct SStructureState {
   ENUM_STRUCTURE_TYPE trend;  // BULLISH/BEARISH/RANGING
   SSwingPoint last_high;
   SSwingPoint last_low;
   bool bos_detected;          // BOS aconteceu?
   bool choch_detected;        // CHoCH aconteceu?
};
```

**Metodos Principais**:
| Metodo | Descricao |
|--------|-----------|
| `AnalyzeStructure()` | Analisa estrutura atual |
| `AnalyzeMTFStructure()` | Analisa H1, M15, M5 separadamente |
| `IsMTFAligned()` | Verifica se todos TFs estao alinhados |
| `GetHTFState()` | Estado do H1 |
| `GetMTFState()` | Estado do M15 |
| `GetLTFState()` | Estado do M5 |
| `DetectBOS()` | Detecta Break of Structure |
| `DetectCHoCH()` | Detecta Change of Character |

---

### 3.3 EliteOrderBlock.mqh - Detector de Order Blocks

**Proposito**: Identificar Order Blocks institucionais

**O Que E Um Order Block?**
```
BULLISH ORDER BLOCK:                BEARISH ORDER BLOCK:

        │ Rally                            │
        │   ↑                              │ Drop
        │   │                              │   ↓
     ┌──┴───┴──┐                        ┌──┴───┴──┐
     │ ULTIMA  │ ← Entry Zone           │ ULTIMA  │ ← Entry Zone
     │ VELA    │                        │ VELA    │
     │ DOWN    │                        │  UP     │
     └─────────┘                        └─────────┘
         │                                  │
    Acumulacao                         Distribuicao
    antes do                           antes da
    movimento                          queda
```

**Estrutura Principal**:
```cpp
struct SEliteOrderBlock {
   double zone_high;           // Topo do OB
   double zone_low;            // Fundo do OB
   ENUM_OB_TYPE type;          // BULLISH ou BEARISH
   datetime formation_time;    // Quando formou
   double quality_score;       // 0-100 (quality)
   int touch_count;            // Quantas vezes testado
   bool is_fresh;              // Primeiro toque?
   bool is_mitigated;          // Ja foi mitigado?
   double displacement;        // Tamanho do movimento
   bool has_liquidity_grab;    // Pegou liquidez antes?
};
```

**Criterios de Qualidade**:
| Criterio | Pontos | Descricao |
|----------|--------|-----------|
| Displacement forte | +20 | Movimento > 2x ATR apos OB |
| Volume acima da media | +15 | Volume > 1.5x media |
| Quebra de estrutura | +20 | BOS apos formar OB |
| Fresh (primeiro toque) | +15 | Nunca foi tocado |
| Confluencia com FVG | +10 | OB dentro de FVG |
| Liquidez antes | +20 | Sweep antes de formar |

---

### 3.4 EliteFVG.mqh - Detector de Fair Value Gaps

**Proposito**: Identificar gaps de preco criados por movimento institucional

**O Que E Um FVG?**
```
BULLISH FVG:                    BEARISH FVG:

   Candle 3 ──►  ┌───┐             ┌───┐  ◄── Candle 1
                 │   │             │   │
                 │   │             │   │
   GAP ────────► │░░░│ ◄── Fill    │░░░│ ◄── GAP (Fill Zone)
                 │░░░│   Zone      │░░░│
                 └───┘             └───┘
   Candle 1 ──►  ┌───┐             ┌───┐  ◄── Candle 3
                 │   │             │   │
                 └───┘             └───┘

   FVG = Espaco entre High de C1 e Low de C3
   Preco tende a "preencher" esse gap
```

**Estrutura Principal**:
```cpp
struct SEliteFairValueGap {
   double upper_level;         // Topo do gap
   double lower_level;         // Fundo do gap
   ENUM_FVG_TYPE type;         // BULLISH ou BEARISH
   datetime formation_time;    // Quando formou
   ENUM_FVG_STATE state;       // OPEN, PARTIALLY_FILLED, FILLED
   double fill_percentage;     // % preenchido
   int touch_count;            // Quantas vezes tocou
   bool is_fresh;              // Primeiro toque?
   double quality_score;       // 0-100
};
```

**Estados do FVG**:
```
OPEN (100%)          PARTIALLY (50%)       FILLED (0%)
┌─────────┐          ┌─────────┐          ┌─────────┐
│░░░░░░░░░│          │         │          │         │
│░░░░░░░░░│          │░░░░░░░░░│          │         │
│░░░░░░░░░│          │░░░░░░░░░│          │         │
└─────────┘          └─────────┘          └─────────┘
   100%                 50%                   0%
   
Entry: 50% fill e o ponto otimo de entrada
```

---

### 3.5 CLiquiditySweepDetector.mqh - Detector de Sweeps

**Proposito**: Identificar quando "smart money" pega stop losses

**O Que E Liquidez?**
```
BSL (Buy-Side Liquidity)     SSL (Sell-Side Liquidity)
= Stop Losses de Shorts      = Stop Losses de Longs

    ┌───────────────┐
    │  ═══ BSL ═══  │ ← Stops acima dos highs
    │               │
    │    RANGE      │
    │               │
    │  ═══ SSL ═══  │ ← Stops abaixo dos lows
    └───────────────┘

SWEEP = Preco quebra o nivel, pega os stops, e VOLTA
```

**Estrutura Principal**:
```cpp
struct SLiquidityPool {
   double level;              // Nivel de preco
   ENUM_LIQUIDITY_TYPE type;  // BSL ou SSL
   int touch_count;           // Quantas vezes testado
   double strength;           // Estimativa de stops
   bool is_equal_level;       // Equal highs/lows?
   bool is_swept;             // Ja foi swept?
   datetime sweep_time;       // Quando foi swept
};

struct SSweepEvent {
   SLiquidityPool pool;       // Pool que foi swept
   double sweep_price;        // Preco maximo do sweep
   double sweep_depth;        // Quao alem foi
   bool has_rejection;        // Teve rejeicao?
   bool returned_inside;      // Voltou pra range?
   bool is_valid_sweep;       // Sweep valido?
};
```

**Equal Highs/Lows - Ima de Liquidez**:
```
EQUAL HIGHS (EQH):           EQUAL LOWS (EQL):

   ════════ EQH ════════        
      │   │   │                    
      │   │   │              ════════ EQL ════════
    ┌─┴─┬─┴─┬─┴─┐                  │   │   │
    │   │   │   │            ┌─────┴───┴───┴─────┐
    │   │   │   │            │                   │
    └───┴───┴───┘            └───────────────────┘

   Cada toque = mais stops acumulados
   Smart Money VAI BUSCAR essa liquidez
```

---

### 3.6 CRegimeDetector.mqh - Detector de Regime

**Proposito**: Determinar se mercado esta em condicao favoravel para trading

**Metricas**:

#### Hurst Exponent (H)
```
H > 0.55  →  TRENDING (momentum)      ✓ TRADE
H < 0.45  →  MEAN-REVERTING (fade)    ✓ TRADE
H ≈ 0.50  →  RANDOM WALK              ✗ NO TRADE

O Hurst mede persistencia:
- H alto = preco continua na mesma direcao
- H baixo = preco tende a reverter
- H = 0.5 = aleatorio (nao previsivel)
```

#### Shannon Entropy (S)
```
S < 1.5   →  LOW NOISE (sinal claro)    ✓ Full size
S 1.5-2.5 →  MEDIUM NOISE               ◐ Half size
S > 2.5   →  HIGH NOISE (muito ruido)   ✗ No trade

Entropy mede "desordem" no mercado
```

**Matriz de Decisao**:
```
┌─────────────────┬───────────────┬───────────────┐
│                 │  Entropy < 1.5 │ Entropy >= 1.5│
├─────────────────┼───────────────┼───────────────┤
│  Hurst > 0.55   │ PRIME_TREND   │ NOISY_TREND   │
│  (Trending)     │ Size: 100%    │ Size: 50%     │
├─────────────────┼───────────────┼───────────────┤
│  Hurst < 0.45   │ PRIME_REVERT  │ NOISY_REVERT  │
│  (Reverting)    │ Size: 100%    │ Size: 50%     │
├─────────────────┼───────────────┼───────────────┤
│  Hurst ~ 0.50   │ RANDOM_WALK   │ RANDOM_WALK   │
│  (Random)       │ Size: 0%      │ Size: 0%      │
└─────────────────┴───────────────┴───────────────┘
```

---

### 3.7 CAMDCycleTracker.mqh - Rastreador de Ciclo AMD

**Proposito**: Identificar fase do ciclo de manipulacao institucional

**O Que E AMD?**
```
AMD = Accumulation → Manipulation → Distribution

FASE 1: ACCUMULATION (Acumulacao)
┌─────────────────────────────────┐
│  ═══════════════════════════   │  ← Range tight
│     Smart Money acumulando     │  ← Volume baixo
└─────────────────────────────────┘

FASE 2: MANIPULATION (Manipulacao)
┌─────────────────────────────────┐
│         ↑ Fake Breakout         │  ← Sweep de liquidez
│    ═════╱                       │  ← Stops acionados
│        ↓                        │  ← Reversao rapida
└─────────────────────────────────┘

FASE 3: DISTRIBUTION (Distribuicao)
┌─────────────────────────────────┐
│                    ↗            │  ← Movimento real
│                  ↗              │  ← Alto volume
│                ↗                │  ← Direcao verdadeira
└─────────────────────────────────┘
```

**Estrutura Principal**:
```cpp
struct SAMDCycle {
   ENUM_AMD_PHASE phase;      // ACCUMULATION/MANIPULATION/DISTRIBUTION
   datetime phase_start;      // Inicio da fase
   double accumulation_high;  // Range da acumulacao
   double accumulation_low;
   double manipulation_extreme; // Ponto extremo da manipulacao
   double distribution_target;  // Alvo da distribuicao
   int phase_duration;        // Duracao em barras
};
```

**Quando Entrar**:
```
ACCUMULATION → Aguardar
MANIPULATION → Preparar (sweep acontecendo)
DISTRIBUTION → ENTRAR! (movimento real comecou)
```

---

### 3.8 CSessionFilter.mqh - Filtro de Sessoes

**Proposito**: Operar apenas nas sessoes mais lucrativas

**Sessoes de Trading (GMT)**:
```
                    ASIA        LONDON        NEW YORK
                  (00-08)      (08-16)        (13-21)
                    │            │              │
00:00 ─────────────┼────────────┼──────────────┼─────── 00:00
                   │            │              │
     ░░░░░░░░░░░░░░│            │              │
     ░ ASIA LOW ░░░│────────────┤              │
     ░ VOLUME  ░░░░│            │──────────────┤
     ░░░░░░░░░░░░░░│            │ LONDON/NY    │
                   │            │ OVERLAP      │
                   │            │ (BEST!)      │
                   │            │──────────────┤
                   │            │              │
```

**Configuracao Padrao**:
| Sessao | Padrao | Motivo |
|--------|--------|--------|
| Asia | OFF | Baixo volume, spreads altos |
| London | ON | Alta liquidez, tendencias claras |
| NY | ON | Continuacao de London |
| London/NY Overlap | ON | MELHOR periodo |
| Late NY (17:00+) | OFF | Volume caindo |
| Friday tarde | OFF | Risco de gap no weekend |

---

### 3.9 CNewsFilter.mqh - Filtro de Noticias

**Proposito**: Evitar volatilidade de noticias de alto impacto

**Impacto das Noticias**:
```
HIGH IMPACT (Vermelho):
- NFP (Non-Farm Payrolls)
- FOMC Interest Rate
- CPI (Inflation)
- GDP
→ BLOQUEAR 30min antes e 30min depois

MEDIUM IMPACT (Laranja):
- Retail Sales
- PMI
- Unemployment Claims
→ BLOQUEAR 15min antes (opcional)

LOW IMPACT (Amarelo):
- Consumer Confidence
- Housing Data
→ Ignorar
```

**Fontes de Dados**:
1. Calendario economico MQL5
2. ForexFactory (via scraping)
3. Investing.com (via API)

---

### 3.10 CEntryOptimizer.mqh - Otimizador de Entrada

**Proposito**: Melhorar o preco de entrada para maximizar R:R

**Por Que Otimizar?**
```
MARKET ENTRY:              OPTIMIZED ENTRY:
                           
Entry ────► ●              Entry ───────────► ●
            │                                 │
            │ Risk                            │ Risk (menor!)
            │                                 │
SL ─────────┼───                  SL ─────────┼───
            │                                 │
            │ Reward                          │ Reward (maior!)
            │                                 │
TP ─────────┼───                  TP ─────────┼───

Market: 1.5:1 R:R              Optimized: 3.0:1 R:R
55% WR = 0.275R expectancy     55% WR = 0.85R expectancy
```

**Metodos de Otimizacao**:
```cpp
// Prioridade 1: FVG 50% Fill (melhor R:R)
optimal_entry = fvg_low + (fvg_high - fvg_low) * 0.5;

// Prioridade 2: OB 70% Retest
optimal_entry = ob_low + (ob_high - ob_low) * 0.7;

// Prioridade 3: Market Entry (somente se sinal muito forte)
optimal_entry = current_price;
```

**Limites de SL para Scalping XAUUSD**:
| Parametro | Valor | Razao |
|-----------|-------|-------|
| Max SL | 50 pts ($50) | Limite maximo absoluto |
| Min SL | 15 pts ($15) | Minimo para evitar ruido |
| Default SL | 30 pts ($30) | Quando sem estrutura |

---

## 4. Modulos de Sinal (Signal/)

### 4.1 CConfluenceScorer.mqh - Scorer de Confluencia

**Proposito**: Combinar todos os sinais em um score final

**Sistema de Pontuacao**:
```cpp
struct SConfluenceScore {
   int total_score;           // 0-100
   int confluences_found;     // Quantos fatores
   ENUM_SIGNAL_TIER tier;     // A/B/C/D
   ENUM_SIGNAL_TYPE direction;// BUY/SELL/NONE
};
```

**Fatores de Confluencia**:
| Fator | Pontos | Descricao |
|-------|--------|-----------|
| HTF Trend Aligned | +20 | H1 na mesma direcao |
| Fresh OB | +15 | Order Block fresco |
| FVG Zone | +10 | Preco em FVG |
| Liquidity Swept | +20 | Sweep confirmado |
| BOS/CHoCH | +15 | Estrutura quebrada |
| Prime Regime | +10 | Hurst/Entropy favoravel |
| AMD Distribution | +10 | Fase de distribuicao |

**Tiers de Sinal**:
```
TIER A (90-100): Confluencia perfeita → Full size
TIER B (75-89):  Boa confluencia     → 75% size
TIER C (60-74):  Confluencia minima  → 50% size
TIER D (0-59):   Insuficiente        → NO TRADE
```

---

### 4.2 SignalScoringModule.mqh - Modulo de Scoring

**Proposito**: Score principal combinando Tech + Fundamental + Sentiment

**Formula**:
```
FinalScore = (TechScore × WeightTech) + 
             (FundScore × WeightFund) + 
             (SentScore × WeightSent)

Pesos Padrao:
- TechScore: 60% (analise tecnica)
- FundScore: 25% (fundamentals)
- SentScore: 15% (sentimento)
```

**Threshold de Execucao**:
```
Score >= 85 → EXECUTE
Score 70-84 → WAIT for better entry
Score < 70  → NO TRADE
```

---

## 5. Modulos de Risco (Risk/)

### 5.1 FTMO_RiskManager.mqh - Compliance FTMO

**Proposito**: Garantir que o EA nunca viole regras da FTMO

**Regras FTMO ($100k)**:
| Regra | Limite FTMO | Nosso Buffer | Acao |
|-------|-------------|--------------|------|
| Daily Loss Max | 5% ($5,000) | 4% ($4,000) | PARA NOVOS TRADES |
| Total Loss Max | 10% ($10,000) | 8% ($8,000) | FECHA TUDO |
| Profit P1 | 10% | - | Meta Phase 1 |
| Profit P2 | 5% | - | Meta Phase 2 |
| Min Days | 4 dias | - | Operar 4+ dias |

**Maquina de Estados**:
```
NORMAL ──────────────────────────────────────────────┐
   │                                                 │
   │ Daily DD >= 4%                                  │
   ▼                                                 │
SOFT_STOP ───────────────────────────────────────────┤
   │ (sem novos trades, gerencia existentes)         │
   │                                                 │
   │ Total DD >= 8%                                  │
   ▼                                                 │
EMERGENCY ───────────────────────────────────────────┤
   │ (fecha tudo, para completamente)                │
   │                                                 │
   │ Novo dia + DD recuperado                        │
   ▼                                                 │
NORMAL ◄─────────────────────────────────────────────┘
```

**Calculo de Lote**:
```cpp
double CalculateLot(double sl_points) {
    double risk_amount = AccountEquity * RiskPercent / 100;
    double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double lot = risk_amount / (sl_points * tick_value);
    
    // Apply regime multiplier
    lot *= regime_multiplier;  // 0.5 or 1.0
    
    // Apply MTF multiplier
    lot *= mtf_multiplier;     // 0.5, 0.75, or 1.0
    
    // Clamp to limits
    lot = MathMin(lot, max_lot);
    lot = MathMax(lot, min_lot);
    
    return NormalizeDouble(lot, 2);
}
```

---

### 5.2 CDynamicRiskManager.mqh - Risco Dinamico

**Proposito**: Ajustar risco baseado em performance

**Ajustes Dinamicos**:
```
WINNING STREAK (3+ wins):
- Manter risco atual
- Considerar aumentar leve (+0.1%)

LOSING STREAK (3+ losses):
- Reduzir risco pela metade
- Cooldown de 1 hora
- Revisar criterios de entrada

RECOVERY MODE (apos DD):
- Risco minimo (0.25%)
- Apenas Tier A signals
- Incrementar gradualmente
```

---

## 6. Modulos de Execucao (Execution/)

### 6.1 CTradeManager.mqh - Gerenciador de Trades

**Proposito**: Gerenciar posicoes abertas com TPs parciais

**Modos de Gerenciamento**:
```cpp
enum ENUM_MANAGEMENT_MODE {
    MGMT_SIMPLE,        // SL/TP fixos
    MGMT_BREAKEVEN,     // Move SL para entrada apos X pips
    MGMT_TRAILING,      // Trailing stop
    MGMT_PARTIAL_TP     // TPs parciais (PADRAO)
};
```

**Configuracao Partial TP**:
```
Configuracao Padrao:
- TP1: 1.5R → Fecha 40% da posicao
- TP2: 2.5R → Fecha 30% da posicao
- TP3: 4.0R → Deixa 30% correr com trailing

Exemplo com $100 de risco:
- TP1 hit: +$60 (40% × 1.5R)
- TP2 hit: +$75 (30% × 2.5R)
- TP3 hit: +$120 (30% × 4.0R)
- Total: +$255 = 2.55R
```

---

### 6.2 TradeExecutor.mqh - Executor Legacy

**Proposito**: Executar ordens de forma confiavel

**Funcionalidades**:
- Retry automatico em caso de requote
- Slippage control
- Magic number management
- Trade comments para tracking

---

## 7. Bridge (Integracao)

### 7.1 COnnxBrain.mqh - Cerebro ONNX

**Proposito**: Executar modelo de ML dentro do MT5

**Fluxo de Inferencia**:
```
1. Coletar Features (15)
   │
   ▼
2. Normalizar (StandardScaler)
   │
   ▼
3. Reshape (100 bars × 15 features)
   │
   ▼
4. OnnxRun()  ← < 5ms
   │
   ▼
5. Output: [P(bear), P(bull)]
   │
   ▼
6. Se P(bull) > 0.65 → Confirma BUY
   Se P(bear) > 0.65 → Confirma SELL
```

**15 Features do Modelo**:
| # | Feature | Calculo |
|---|---------|---------|
| 1 | Returns | (close - prev) / prev |
| 2 | Log Returns | log(close / prev) |
| 3 | Range % | (high - low) / close |
| 4 | RSI M5 | RSI(14) / 100 |
| 5 | RSI M15 | RSI(14) / 100 |
| 6 | RSI H1 | RSI(14) / 100 |
| 7 | ATR Norm | ATR(14) / close |
| 8 | MA Distance | (close - MA20) / MA20 |
| 9 | BB Position | (close - mid) / width |
| 10 | Hurst | Rolling Hurst(100) |
| 11 | Entropy | Rolling Entropy(100) / 4 |
| 12 | Session | 0=Asia, 1=London, 2=NY |
| 13 | Hour Sin | sin(2π × hour / 24) |
| 14 | Hour Cos | cos(2π × hour / 24) |
| 15 | OB Distance | Dist to OB / ATR |

---

### 7.2 PythonBridge.mqh - Ponte Python

**Proposito**: Comunicar com Python Agent Hub

**Endpoints**:
| Endpoint | Metodo | Funcao |
|----------|--------|--------|
| `/api/v1/regime` | POST | Obter analise de regime |
| `/api/v1/signal` | POST | Obter confirmacao de sinal |
| `/api/v1/health` | GET | Verificar status do hub |

**Timeout**: 400ms (fail fast para nao travar OnTick)

---

### 7.3 CFundamentalsBridge.mqh - Ponte Fundamentals [NEW v3.21]

**Proposito**: Integrar analise fundamental de XAUUSD

**Descricao**: Bridge HTTP para consumir endpoints de fundamentals do Python Agent Hub. Busca dados macro (FRED), oil (yfinance), e sentiment (FinBERT).

**Endpoints Consumidos**:
| Endpoint | Metodo | Funcao |
|----------|--------|--------|
| `/api/v1/signal` | GET | Sinal agregado de fundamentals |
| `/api/v1/macro` | GET | Real Yields, DXY, VIX (FRED) |
| `/api/v1/oil` | GET | Gold-Oil ratio (42% importance!) |
| `/api/v1/sentiment` | GET | News sentiment (FinBERT) |

**Estrutura SFundamentalsSignal**:
```cpp
struct SFundamentalsSignal {
   string signal;           // STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL
   double score;            // -10 to +10
   int    score_adjustment; // Pontos para ajustar score tecnico
   double size_multiplier;  // 0.5 a 1.0
   double confidence;       // 0.0 a 1.0
   string bias;             // BULLISH, NEUTRAL, BEARISH
   double macro_score;      // Score macro
   double oil_score;        // Score oil (importante!)
   double etf_score;        // Score ETF flows
   double sentiment_score;  // Score sentiment
};
```

**Cache**: 5 minutos (fundamentals nao mudam rapido)

**Insights Criticos** (da pesquisa RAG):
- Oil tem 42% de feature importance para gold!
- Real Yields quebrou correlacao em 2022 (-0.82 -> -0.55)
- Central banks comprando 1100+ tonnes/ano mudou dinamica
- Gold-Oil ratio normal: 15-25 (atual ~72 = gold muito caro)

---

### 7.4 CFundamentalsIntegrator.mqh - Fusao Tech+Fund [NEW v3.21]

**Proposito**: Combinar analise tecnica (CConfluenceScorer) com fundamentals

**Logica de Integracao**:
```
PESO PADRAO:
- Tecnico: 70%
- Fundamental: 30%

ALINHAMENTO:
- Tech BULLISH + Fund BULLISH = +10 bonus
- Tech BULLISH + Fund BEARISH = -15 penalidade

RESULTADO:
- Final Score = (Tech * 0.7) + (Fund * 0.3) + Alignment Bonus
- Size Mult = min(Tech_Mult, Fund_Mult) * 1.2 se alinhado
```

**Estrutura SIntegratedSignal**:
```cpp
struct SIntegratedSignal {
   // Tecnico
   ENUM_SIGNAL_TYPE tech_direction;
   double tech_score;
   
   // Fundamental
   string fund_signal;
   double fund_score;
   
   // Integrado
   double final_score;
   bool is_aligned;
   bool should_trade;
   double size_multiplier;
   string reason;
};
```

**Metodos Principais**:
- `GetIntegratedSignal()` - Calcula sinal completo
- `ShouldTrade()` - Retorna se deve operar
- `GetSizeMultiplier()` - Multiplier ajustado

---

## 8. Fluxo de Execucao do EA

### 8.1 OnInit() - Inicializacao

```
1. Inicializa Risk Manager (FTMO)
2. Inicializa Scoring Engine
3. Inicializa Trade Executor
4. Inicializa Trade Manager (parciais)
5. Inicializa MTF Manager (H1/M15/M5)
6. Inicializa Modulos de Analise:
   - Regime Detector
   - Structure Analyzer
   - Sweep Detector
   - AMD Tracker
   - Session Filter
   - News Filter
   - Entry Optimizer
7. Conecta Confluence Scorer
8. Inicia Timer
```

### 8.2 OnTick() - Cada Tick

```
┌────────────────────────────────────────────────────────────┐
│                    SISTEMA DE GATES                        │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  GATE 1: Emergency Mode?                                   │
│  └── Se SIM → RETURN (nao faz nada)                       │
│                                                            │
│  GATE 2: Risk Manager OK?                                  │
│  └── Verifica DD diario/total                             │
│  └── Se violado → EMERGENCY MODE                          │
│                                                            │
│  GATE 3: Session Filter OK?                                │
│  └── Verifica se esta em sessao permitida                 │
│  └── Se NAO → RETURN                                       │
│                                                            │
│  GATE 4: News Filter OK?                                   │
│  └── Verifica noticias proximas                           │
│  └── Se HIGH IMPACT → RETURN                              │
│                                                            │
│  GATE 5: Regime Filter OK?                                 │
│  └── Verifica Hurst/Entropy                               │
│  └── Se RANDOM_WALK → RETURN                              │
│                                                            │
│  GATE 6: MTF Direction OK? (NEW v3.20)                    │
│  └── Verifica tendencia H1                                │
│  └── Se contra H1 → RETURN                                │
│                                                            │
│  GATE 7: Structure/Signal OK?                              │
│  └── Verifica BOS, OB, FVG, Sweep                         │
│  └── Se sem sinal → RETURN                                │
│                                                            │
│  GATE 8: MTF Confirmation OK? (NEW v3.20)                 │
│  └── Verifica alinhamento M15+M5                          │
│  └── Se weak alignment → Reduce size                      │
│                                                            │
│  GATE 9: Confluence Score OK?                              │
│  └── Score >= 70 (Tier C minimo)                          │
│  └── Se < 70 → RETURN                                     │
│                                                            │
│  GATE 10: Entry Optimization                               │
│  └── Calcula entry, SL, TPs otimizados                    │
│  └── Se R:R < 1.5 → RETURN                                │
│                                                            │
│  PASSED ALL GATES → EXECUTE TRADE!                         │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 8.3 OnTimer() - Cada Segundo

```
1. Atualiza Regime (Hurst/Entropy)
2. Atualiza Estrutura (BOS/CHoCH)
3. Atualiza Liquidity Pools
4. Atualiza AMD Phase
5. Gerencia Posicoes Abertas (TPs parciais)
6. Heartbeat com Python Hub (se conectado)
```

---

## 9. Parametros de Entrada

### 9.1 Risk Management
| Parametro | Default | Descricao |
|-----------|---------|-----------|
| InpRiskPerTrade | 0.5% | Risco por trade |
| InpMaxDailyLoss | 5.0% | Max DD diario (FTMO) |
| InpSoftStop | 3.5% | Nivel de soft stop |
| InpMaxTotalLoss | 10.0% | Max DD total (FTMO) |
| InpMaxTradesPerDay | 20 | Limite de trades/dia |

### 9.2 Scoring Engine
| Parametro | Default | Descricao |
|-----------|---------|-----------|
| InpExecutionThreshold | 85 | Score minimo para executar |
| InpWeightTech | 0.6 | Peso: Tecnico |
| InpWeightFund | 0.25 | Peso: Fundamental |
| InpWeightSent | 0.15 | Peso: Sentimento |

### 9.3 Execution
| Parametro | Default | Descricao |
|-----------|---------|-----------|
| InpSlippage | 50 | Max slippage (points) |
| InpMagicNumber | 123456 | Identificador unico |
| InpTradeComment | "SINGULARITY" | Comentario nos trades |

### 9.4 Session Filter
| Parametro | Default | Descricao |
|-----------|---------|-----------|
| InpAllowAsian | false | Permitir sessao Asia |
| InpAllowLateNY | false | Permitir NY tardio |
| InpGMTOffset | 0 | Offset GMT do broker |
| InpFridayCloseHour | 14 | Hora de fechar na sexta |

### 9.5 News Filter
| Parametro | Default | Descricao |
|-----------|---------|-----------|
| InpNewsFilterEnabled | true | Ativar filtro |
| InpBlockHighImpact | true | Bloquear alto impacto |
| InpBlockMediumImpact | false | Bloquear medio impacto |

### 9.6 Entry Optimization
| Parametro | Default | Descricao |
|-----------|---------|-----------|
| InpMinRR | 1.5 | R:R minimo |
| InpTargetRR | 2.5 | R:R alvo |
| InpMaxWaitBars | 10 | Max barras esperando |

### 9.7 Multi-Timeframe (v3.20)
| Parametro | Default | Descricao |
|-----------|---------|-----------|
| InpUseMTF | true | Ativar arquitetura MTF |
| InpMinMTFConfluence | 60.0 | Confluencia MTF minima |
| InpRequireHTFAlign | true | Exigir alinhamento H1 |
| InpRequireMTFZone | true | Exigir zona M15 |
| InpRequireLTFConfirm | true | Exigir confirmacao M5 |

---

## 10. Performance Esperada

### 10.1 Metricas Alvo

| Metrica | Alvo | Razao |
|---------|------|-------|
| Win Rate | 65-75% | MTF + SMC + ML confluence |
| Average R:R | 2.0-2.5 | Entry optimization |
| Profit Factor | 2.0+ | High WR × High R:R |
| Max Drawdown | <8% | FTMO buffer |
| Trades/Day | 3-8 | Quality over quantity |
| Expectancy | 0.5-0.8R | Consistent edge |

### 10.2 Formula de Expectancy

```
E = (WinRate × AvgWin) - (LossRate × AvgLoss)

Exemplo com nossos alvos:
E = (0.70 × 2.0R) - (0.30 × 1.0R)
E = 1.4R - 0.3R
E = 1.1R por trade

Com 5 trades/dia × 20 dias = 100 trades/mes
Profit mensal esperado = 100 × 1.1R × 0.5% = 55%
(ajustado para realidade: ~10-15%/mes)
```

---

## 11. Checklist de Validacao

### Antes de Operar Live:

- [ ] Backtest em 1+ ano de dados M5
- [ ] Walk-Forward Analysis (10+ janelas)
- [ ] Monte Carlo Simulation (5000+ runs)
- [ ] Demo trading por 2+ semanas
- [ ] Verificar slippage real vs backtest
- [ ] Testar em diferentes condicoes de mercado
- [ ] Validar compliance FTMO
- [ ] Configurar alertas de emergencia

### Checklist Diario:

- [ ] Verificar DD atual
- [ ] Verificar calendario economico
- [ ] Verificar spread atual
- [ ] Verificar conexao do broker
- [ ] Revisar trades do dia anterior

---

## 12. Troubleshooting

| Problema | Causa Provavel | Solucao |
|----------|----------------|---------|
| EA nao abre trades | Score baixo | Verificar confluencias |
| SL muito grande | ATR alto | Esperar volatilidade reduzir |
| Muitos trades filtrados | Filtros agressivos | Ajustar thresholds |
| DD alto | Regime ruim | Verificar Hurst/Entropy |
| Slippage alto | Liquidez baixa | Evitar Asia, noticias |
| ONNX nao carrega | Path errado | Verificar Models/ folder |

---

## 13. Historico de Versoes

| Versao | Data | Mudancas |
|--------|------|----------|
| 3.20 | 2024-11 | MTF Architecture (H1+M15+M5) |
| 3.10 | 2024-11 | Entry Optimizer SL limits |
| 3.00 | 2024-11 | Singularity Edition (ML/ONNX) |
| 2.00 | 2024-10 | SMC Core Modules |
| 1.00 | 2024-09 | Initial Release |

---

## 14. Creditos e Referencias

**Metodologia**:
- Smart Money Concepts (SMC)
- ICT (Inner Circle Trader) Methodology
- Institutional Order Flow

**Literatura**:
- "Algorithmic Trading" - Hurst Exponent, Entropy
- "MQL5 Programming" - MetaQuotes
- "Neural Networks for Trading" - ONNX Integration

**Ferramentas**:
- MetaTrader 5
- Python (PyTorch, ONNX)
- Walk-Forward Analysis Framework

---

*EA_SCALPER_XAUUSD v3.20 - Singularity MTF Edition*
*"Trade with the institutions, not against them"*
