# FUTURE IMPROVEMENTS - EA_SCALPER_XAUUSD (CONSOLIDADO)

> **Author:** FORGE v3.1  
> **Date:** 2025-12-01  
> **Status:** Living Document - CONSOLIDADO de 3 arquivos  

---

## STATUS GERAL

### ✅ JA IMPLEMENTADO

| Feature | Arquivo | Data |
|---------|---------|------|
| **Volume Profile (POC/VAH/VAL)** | `Python_Agent_Hub/ml_pipeline/indicators/volume_profile.py` | 2024-11-28 |
| **Volume Delta (Tick Rule)** | `Python_Agent_Hub/ml_pipeline/indicators/volume_delta.py` | 2024-11-28 |
| **R-Multiple Tracker** | `Python_Agent_Hub/ml_pipeline/risk/r_multiple_tracker.py` | 2024-11-28 |
| **Risk of Ruin Calculator** | `Python_Agent_Hub/ml_pipeline/risk/risk_of_ruin.py` | 2024-11-28 |
| **Hurst Exponent (R/S)** | `MQL5/Include/EA_SCALPER/Analysis/CRegimeDetector.mqh` | Implementado |
| **Shannon Entropy** | `MQL5/Include/EA_SCALPER/Analysis/CRegimeDetector.mqh` | Implementado |
| **Multi-scale Hurst (v4.0)** | `CRegimeDetector.mqh` (50/100/200 bars) | Implementado |
| **Variance Ratio Test** | `CRegimeDetector.mqh` | Implementado |
| **Regime Transition Detection** | `CRegimeDetector.mqh` (velocity, proximity) | Implementado |
| **Hurst em ONNX Features** | `COnnxBrain.mqh` | Implementado |
| **Session OK field** | `CMTFManager.mqh` (campo existe) | Implementado |

### ❌ NAO IMPLEMENTADO (BACKLOG)

| Feature | Prioridade | Esforco | Status |
|---------|------------|---------|--------|
| **Fibonacci Golden Pocket** | P1 | 2-3h | TODO |
| **Fibonacci Extensions TP** | P1 | 2-3h | TODO |
| **Fib Score Confluence** | P1 | 2h | TODO |
| **Bayesian Confluence** | P1 | 2-3h | TODO |
| **Adaptive Kelly** | P1 | 2-3h | TODO |
| **Spread Awareness** | P1 | 1-2h | TODO |
| **Fibonacci Clusters** | P2 | 4-6h | PLANNED |
| **M15 Trend Independente** | P2 | 3-4h | PLANNED |
| **BOS/CHoCH M15 Melhorado** | P2 | 3-4h | PLANNED |
| **Session Gating MTF** | P2 | 2-3h | PLANNED |
| **Regime Ensemble** | P2 | 1-2d | PLANNED |
| **HMM Regime** | P2 | 1d | PLANNED |
| **Adaptive Timeframes** | P2 | 3-4h | PLANNED |
| **Dynamic Circuit Breaker** | P2 | 3-4h | PLANNED |
| **Tuning Thresholds MTF** | P2 | 2-3h | PLANNED |
| **Telemetria MTF Debug** | P3 | 2-3h | PLANNED |
| **Imbalance Detection** | P3 | 4-6h | PLANNED |
| **Transformer-Lite ONNX** | P3 | 3-5d | PLANNED |
| **Spread Predictor LSTM** | P4 | 2-3d | IDEA |
| **Meta-Learning Selector** | P4 | 1w+ | IDEA |
| **Portfolio Heat Mgmt** | P4 | 1d | IDEA |

---

## PHASE 0: XAUUSD EA ARCHETYPE (ARGUS + BENCHMARKS) 🧠

### 0.1 Objetivo

Consolidar um “arquétipo” de EA XAUUSD competitivo em prop firms (FTMO $100k) a partir de:
- Pesquisa em MQL5 / Forex Factory sobre os EAs de ouro com melhor feedback real (clientes, sinais, reviews).
- Engenharia reversa conceitual de códigos open-source focados em XAUUSD.
- Integração disso tudo com a arquitetura atual do EA_SCALPER_XAUUSD (regimes, SMC, ONNX, Python Hub).

Meta: transformar o EA_SCALPER_XAUUSD em um “Gold EA de classe MQL5/FTMO”, com:
- Confluência robusta (ML + SMC + regime + contexto).
- Gestão de risco FTMO-first (limites de DD + circuit breaker).
- Execução limpa (sem grid/marti, SL/TP reais, trailing saudável).

---

### 0.2 Padrões observados em EAs de referência (internet)

**Produtos comerciais (MQL5 / lojas EA)**
- Gold Trade Pro MT5 (Profalgo Limited, XAUUSD-only, “real strategy”, sem grid/marti explícito + sinais live).
- Diversos scalpers XAUUSD (ADX Scalper Xauusd, Scalper Sniper Xauusd, Killer Scalper Xauusd, Ai Xauusd Scalper, ScalpPrime EA) com:
  - Especialização em ouro, TF M1–M15.
  - Filtro forte de horário (sessões) e volatilidade.
  - Marketing pesado em cima de “low DD” e “prop firm friendly”.
- EAs de ouro com feedback polarizado (ex.: Golden Ai EA / Gold scalpers com reviews de conta explodida em 1 dia), tipicamente:
  - Uso de grid/martingale ou R:R desbalanceado (muitos pequenos ganhos, poucas perdas gigantes).

**Threads Forex Factory (XAUUSD EAs / scalpers)**
- [FREE EA] FxMath Apex Aurum-XAUUSD Scalper for MT5: scalper de ouro em beta com feedback da comunidade sobre spreads, latência e DD.
- Free MT4 Expert – ScalpVolumeX_Gold EA: EA gratuito otimizado para XAUUSD, discussão intensa sobre parâmetros, horário e risco.
- Outras threads de EAs de ouro mostram o padrão: resultados bons até mudança de regime → crash quando grid/marti está escondido.

**Códigos open-source analisados (GitHub)**
- `IntradayAurumPulsePro` (repo `gold-trade-advanced-bot`):
  - Intraday EA específico para XAUUSDm.
  - Sinais via RSI + CCI + MA, com lógica de múltiplas entradas (ladder) sem martingale.
  - SL baseado em 2x ATR, lotes discretos por faixa de balance/equity.
  - Filtro de horário (6h–20h), filtro de “nova barra” e BE+/trailing em dois estágios.
- `Golden7` (EA de confluência, multi-mercado, excelente como framework):
  - Motor de score com 7 indicadores (MA, ADX, DI, Bollinger, RVI, SAR, RSI) + bias fundamental manual.
  - Pesos ajustáveis por indicador e `InpMinScore` mínimo para sinal.
  - Filtros de tendência (MA HTF), volatilidade (faixa de ATR) e tempo (sessões).
  - SL/TP dinâmico via ATR, BE+ e trailing stop sofisticados.

**Conclusões-chave de padrão:**
- Especialização total em XAUUSD (símbolo e TF bem definidos).
- Estratégias “reais” (trend, mean reversion, S/R, SMC) com **SL/TP reais** (sem grid/marti oculto).
- Uso consistente de **ATR** / volatilidade para dimensionar SL/TP e filtrar mercados extremos.
- Filtros fortes de horário/sessão (Londres/NY), evitando horários mortos ou extremamente voláteis.
- Gestão de risco clara e, em vários casos, presets dedicados para **prop firms**.
- BE+ + trailing em múltiplos estágios como padrão para “lockar” lucro sem matar as caudas positivas.

---

### 0.3 Blueprint de arquitetura para EA_SCALPER_XAUUSD (inspirado em benchmarks)

#### 0.3.1 Motor de Confluência v2 (Score Engine)

**Motivação:** os melhores EAs de ouro não dependem de um único sinal; usam confluência (vários indicadores + contexto). Nosso EA já tem ONNX + SMC + regime → precisamos consolidar isso em um **score unificado**, à la Golden7.

**Arquivos alvo:**
- `MQL5/Include/EA_SCALPER/Logic/CConfluenceScorer.mqh`
- `MQL5/Include/EA_SCALPER/ML/COnnxBrain.mqh`
- `MQL5/Include/EA_SCALPER/Analysis/CRegimeDetector.mqh`
- `MQL5/Include/EA_SCALPER/Structure/*` (BOS/CHoCH, OBs, FVGs)

**Proposta (conceito):**
```mql5
double CConfluenceScorer::CalculateScore(const SSignalContext &ctx)
{
   // Componentes normalizados em [0,100]
   double scoreMl       = MlScore(ctx.pDirection);          // ONNX P(direction)
   double scoreTrend    = TrendScore(ctx.trendHtf, ctx.trendLtf);
   double scoreRegime   = RegimeScore(ctx.regimeType, ctx.regimeConfidence);
   double scoreSmc      = SmcScore(ctx.orderBlock, ctx.fvg, ctx.liquiditySweep);
   double scoreSession  = SessionScore(ctx.session, ctx.sessionQuality);
   double scoreVol      = VolatilityScore(ctx.atr, ctx.range, ctx.spread);

   double score =
      w_ml      * scoreMl      +
      w_trend   * scoreTrend   +
      w_regime  * scoreRegime  +
      w_smc     * scoreSmc     +
      w_session * scoreSession +
      w_vol     * scoreVol;

   return score;
}
```

**Regras de decisão:**
- Só abrir long se `score >= MinScoreLong` e `scoreBull > scoreBear`.
- Só abrir short se `score >= MinScoreShort` e `scoreBear > scoreBull`.
- Possível camada bayesiana (já planejada em 1.2) para mapear score → probabilidade de win.

**Esforço:** 1–2 dias (inclui calibração inicial)  
**Prioridade:** **P1** (core do EA)  
**Status:** PLANNED (depende de: Bayesian Confluence)

---

#### 0.3.2 SL/TP dinâmicos via ATR + regime

**Motivação:** IntradayAurumPulsePro e Golden7 usam ATR como base para SL/TP, evitando SL fixo em pontos que ignora a volatilidade do ouro. Para FTMO, precisamos de R em dólares estável independentemente do regime.

**Arquivos alvo:**
- `MQL5/Include/EA_SCALPER/Risk/CRiskManager.mqh`
- `MQL5/Include/EA_SCALPER/Trade/CTradeManager.mqh`
- `MQL5/Include/EA_SCALPER/Analysis/CRegimeDetector.mqh`

**Proposta (conceito):**
```mql5
struct SVolatilityProfile
{
   double atrPoints;
   double rangePoints;
   double regimeMultiplier;   // p.ex. >1.0 em regime explosivo
};

double CRiskManager::CalculateSlPoints(const SVolatilityProfile &v, double rDollars)
{
   double slAtr = v.atrPoints * atrMultBase * v.regimeMultiplier;
   slAtr = Clamp(slAtr, slMinPoints, slMaxPoints);
   return slAtr;
}
```

**Regras:**
- SL em pontos derivado de ATR + regime; TP em múltiplos de R (ex.: 1.5R, 2R).
- Lote derivado de `risk_per_trade_%` + SL em dinheiro (coerente com FTMO).

**Esforço:** 1 dia  
**Prioridade:** **P1**  
**Status:** PLANNED (sinergia com Adaptive Kelly + Spread Awareness)

---

#### 0.3.3 Filtros de sessão + volatilidade (Intraday Gating)

**Motivação:** EAs bem avaliados de ouro operam em janelas específicas (Londres/NY) e evitam extremos de volatilidade; isso reduz whipsaws e spikes de DD.

**Arquivos alvo:**
- `MQL5/Include/EA_SCALPER/Session/CSessionManager.mqh`
- `MQL5/Include/EA_SCALPER/Analysis/CRegimeDetector.mqh`
- `MQL5/Include/EA_SCALPER/Logic/CSetupValidator.mqh`

**Proposta:**
- Definir “gold sessions”:
  - Sessão A: Londres (por ex. 08:00–11:00 server).
  - Sessão B: NY e overlap (13:30–17:00).
- Filtro de volatilidade:
  - Operar apenas se `ATR` entre `[atrMin, atrMax]`.
  - Pausar se volatilidade ultra-alta + spreads fora do normal (integra com Spread Awareness).

**Esforço:** 4–6h  
**Prioridade:** **P1/P2**  
**Status:** PLANNED (relacionado a Session Gating MTF)

---

#### 0.3.4 Ladder de entradas SEM martingale

**Motivação:** IntradayAurumPulsePro usa múltiplas entradas (até 5) com thresholds graduais, mas sem multiplicar lotes. Isso permite “scale in” sensato sem grid/marti.

**Arquivos alvo:**
- `MQL5/Include/EA_SCALPER/Trade/CTradeManager.mqh`
- `MQL5/Include/EA_SCALPER/Logic/CSetupExecutor.mqh`

**Proposta:**
- Definir `entry_index` (0..N-1) por contexto:
  - Cada nova entrada só permitida se:
    - DD local não excede limite.
    - Preço se moveu X pontos a favor/contra dentro de estrutura válida (SMC).
  - Lotes constantes ou **decrescentes**, nunca exponenciais.
- Garantir que a soma de risco de todas as entradas de um ciclo fique ≤ R_max_trade ou ≤ % diária permitida.

**Esforço:** 1 dia  
**Prioridade:** P2  
**Status:** PLANNED (depende de Risk Engine FTMO)

---

#### 0.3.5 Risk Engine FTMO (Daily/Total + Circuit Breaker)

**Motivação:** EAs comerciais “prop-friendly” expõem setfiles específicos para challenges. Nosso objetivo é ter isso nativo: o EA deve proteger a conta antes de violar regras FTMO.

**Arquivos alvo:**
- `MQL5/Include/EA_SCALPER/Risk/FTMO_RiskManager.mqh`
- `MQL5/Include/EA_SCALPER/Risk/CRiskManager.mqh`

**Proposta (complementa Dynamic Circuit Breaker já listado em 2.10):**
- Parâmetros:
  - `MaxDailyLossPct` (ex.: 4%).
  - `MaxTotalLossPct` (ex.: 8%).
  - `MaxTradesPerDay`, `MaxTradesPerHour`.
  - Flag `PropModeEnabled`.
- Lógica:
  - Tracking de PnL diário (realizado + não realizado).
  - Quando gatilho diário atingir, bloquear novas entradas até reset do dia.
  - Quando DD total atingir gatilho, entrar em “modo defesa” (lotes mínimos / desligar).

**Esforço:** 1–2 dias  
**Prioridade:** **P1** (FTMO)  
**Status:** PLANNED (refina Dynamic Circuit Breaker)

---

#### 0.3.6 Gestão de trade multi-estágio (BE+ + Trailing)

**Motivação:** tanto IntradayAurumPulsePro quanto Golden7 usam dois estágios:
- BE+ ao atingir certo lucro.
- Trailing dinamicamente depois disso.

**Arquivos alvo:**
- `MQL5/Include/EA_SCALPER/Trade/CPositionManager.mqh`
- `MQL5/Include/EA_SCALPER/Risk/CRiskManager.mqh`

**Proposta:**
- Estágios típicos:
  - Stage 1: lucro atinge 1R → mover SL para +0.2–0.3R.
  - Stage 2: lucro atinge 1.5–2R → ativar trailing (fixa ou ATR-based).
  - (Opcional) Stage 3: em lucros > 3R, apertar trailing para capturar caudas.

**Esforço:** 4–6h  
**Prioridade:** P2  
**Status:** PLANNED

---

#### 0.3.7 Telemetria e logging de sinais

**Motivação:** para validar que o EA está operando como planejado e para permitir análise Oracle/ARGUS posterior, precisamos logar contexto no momento da decisão (similar à transparência de sinas MQL5/Myfxbook).

**Arquivos alvo:**
- `MQL5/Include/EA_SCALPER/Telemetry/CTelemetryLogger.mqh` (a criar ou estender)
- `Python_Agent_Hub/*` (integração futura)

**Proposta:**
- Logar por trade:
  - Score total + componentes (ml, trend, regime, smc, session, vol).
  - Regime, sessão, ATR, spread, SL/TP em R.
  - Informações de ladder (índice de entrada, risco agregado).
- Output:
  - CSV em `data/logs/EA_SCALPER_XAUUSD/`.
  - Opcional: envio para Python Hub para análise pós-sessão.

**Esforço:** 1 dia  
**Prioridade:** P2/P3  
**Status:** PLANNED

---

### 0.4 Impacto esperado (FTMO & robustez)

- Reduzir probabilidade de violar limites FTMO antes de atingir alvo de lucro.
- Tornar o comportamento do EA mais previsível e consistente com backtests (menos “surpresas” em live).
- Facilitar WFA / Monte Carlo / stress tests, pois:
  - Risco fica parametrizado em R múltiplos e % de conta.
  - Filtros de sessão/volatilidade reduzem regimes “não vistos” no treino.
- Melhor narrativa comercial (se um dia for publicado), alinhada aos EAs de referência de XAUUSD no mercado.

---

### 0.5 Referências externas (ARGUS scan)

**MQL5 / EAs de Ouro**
- Gold Trade Pro MT5 – Profalgo Limited – XAUUSD-only, estratégia declaradamente “real” (sem grid/marti), com sinais live e setfile para prop firm.
- ADX Scalper Xauusd, Scalper Sniper Xauusd, Killer Scalper Xauusd, Ai Xauusd Scalper MT5 – família de scalpers de ouro com foco em M1–M15, forte uso de filtros de sessão/volatilidade.
- ScalpPrime EA MT5 – scalper XAUUSD com lógica baseada em Fibonacci + volume, performance verificada em contas reais (marketing).
- Golden Ai EA MT5 e outros “Gold Scalper PRO” – exemplos de EAs com backtests excelentes mas reviews relatando grandes perdas em pouco tempo (indicando risco de grid/marti / R:R desbalanceado).

**Forex Factory (threads XAUUSD EA)**
- [FREE EA] FxMath Apex Aurum-XAUUSD Scalper for MT5 (Beta Tester) – thread com EA gratuito de ouro e feedback detalhado da comunidade sobre desempenho real.
- Free MT4 Expert – ScalpVolumeX_Gold EA – thread de EA de ouro gratuito, focada em otimização de parâmetros, horário e risco.

**GitHub (código open-source analisado)**
- `gold-trade-advanced-bot / IntradayAurumPulsePro` – EA intraday para XAUUSDm com:
  - Sinais via RSI/CCI/MA + múltiplas entradas sem martingale.
  - SL baseado em ATR, filtro de horário, motor de BE+/trailing.
- `golden7 / golden7-withfundamentals-makesound.mq5` – EA de confluência multi-indicador com:
  - Score ponderado de 7 indicadores + bias fundamental.
  - Filtros de tendência, volatilidade (ATR) e tempo.
  - SL/TP dinâmicos, BE+ e trailing avançado.

> Observação: código-fonte desses repositórios foi usado apenas para identificar padrões de arquitetura, não para copiar lógica proprietária 1:1. O EA_SCALPER_XAUUSD seguirá sua própria implementação alinhada ao PRD v2.2.

---

## PHASE 1: QUICK WINS (P1) ⚡

### 1.1 Fibonacci Integration [PRIORITY: HIGH] 🆕

**Source:** ARGUS Research Report (2025-12-01)
**Research:** [`DOCS/03_RESEARCH/FINDINGS/FIBONACCI_XAUUSD_RESEARCH.md`](../03_RESEARCH/FINDINGS/FIBONACCI_XAUUSD_RESEARCH.md)
**Evidence:** SSRN Paper (Shanaev & Gibson, 2022) - Niveis 38.2%, 50%, 61.8% estatisticamente significativos

#### 1.1.1 Golden Pocket Entry Zone

**Arquivos:** `CConfluenceScorer.mqh`, `EliteOrderBlock.mqh`, `EliteFVG.mqh`

```mql5
bool IsInGoldenPocket(double price, double swingHigh, double swingLow)
{
   double range = swingHigh - swingLow;
   double fib618 = swingHigh - range * 0.618;
   double fib65 = swingHigh - range * 0.65;
   return (price >= fib65 && price <= fib618);
}

// Adicionar ao score de confluencia
if(IsInGoldenPocket(current_price, swing_high, swing_low))
{
   confluence_score += 15;
   if(IsNearOrderBlock(current_price)) confluence_score += 10;
   if(IsInFVG(current_price)) confluence_score += 10;
}
```

**Esforco:** 2-3 horas | **Status:** ❌ TODO

---

#### 1.1.2 Fibonacci Extensions para TP

**Arquivo:** `CTradeManager.mqh`

```mql5
struct SFibTargets
{
   double tp1_1272;    // 127.2% extension - TP1
   double tp2_1618;    // 161.8% extension - TP2
   double tp3_200;     // 200% extension - opcional
};

SFibTargets CalculateFibTargets(double swingLow, double swingHigh, bool isBullish)
{
   SFibTargets targets;
   double range = swingHigh - swingLow;
   
   if(isBullish)
   {
      targets.tp1_1272 = swingHigh + range * 0.272;
      targets.tp2_1618 = swingHigh + range * 0.618;
      targets.tp3_200 = swingHigh + range * 1.0;
   }
   else
   {
      targets.tp1_1272 = swingLow - range * 0.272;
      targets.tp2_1618 = swingLow - range * 0.618;
      targets.tp3_200 = swingLow - range * 1.0;
   }
   return targets;
}
```

**Esforco:** 2-3 horas | **Status:** ❌ TODO

---

#### 1.1.3 Niveis Fib - USAR vs EVITAR

| Nivel | Status | Acao |
|-------|--------|------|
| 38.2% | ✅ USAR | Shallow pullback em trend forte |
| 50.0% | ✅ USAR | Nivel psicologico |
| 61.8% | ✅ USAR | Golden Ratio - principal |
| 23.6% | ❌ EVITAR | REDUZ poder preditivo |
| 76.4% | ❌ EVITAR | REDUZ poder preditivo |

---

### 1.2 Bayesian Confluence [PRIORITY: HIGH]

**Arquivo:** `CConfluenceScorer.mqh`

**Atual:**
```mql5
// Score aditivo fixo
conf = 0;
if(htf_aligned) conf += 30;
if(mtf_structure) conf += 35;
if(ltf_confirmed) conf += 35;
```

**Proposto:**
```mql5
// Bayesian probability
// P(Win|Evidence) = P(HTF|Win) * P(MTF|Win) * P(LTF|Win) * P(Win) / P(Evidence)
double p_win_given_htf = 0.62;      // Calibrado via backtest
double p_win_given_mtf = 0.58;
double p_win_given_ltf = 0.55;
double prior_win = 0.52;
```

**Esforco:** 2-3 horas | **Status:** ❌ TODO

---

### 1.3 Adaptive Kelly Position Sizing [PRIORITY: HIGH]

**Arquivo:** `FTMO_RiskManager.mqh`

**Proposto:**
```mql5
double kelly = CalculateAdaptiveKelly(win_rate, win_rate_std, avg_win, avg_loss);
double risk_percent = kelly * 0.5;  // Half-Kelly for safety

if(current_dd > 0.03) risk_percent *= 0.5;   // 50% em DD > 3%
if(current_dd > 0.05) risk_percent *= 0.25;  // 25% em DD > 5%
```

**Esforco:** 2-3 horas | **Status:** ❌ TODO

---

### 1.4 Spread Awareness [PRIORITY: MEDIUM]

**Arquivo:** `TradeExecutor.mqh`

```mql5
double current_spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * _Point;
double avg_spread = GetAverageSpread(100);
double spread_ratio = current_spread / avg_spread;

if(spread_ratio > 1.5)
{
   if(signal_urgency < 0.8)
      return WAIT_FOR_BETTER_SPREAD;
   else
      AdjustEntryForSpread(spread_ratio);
}
```

**Esforco:** 1-2 horas | **Status:** ❌ TODO

---

## PHASE 2: MTF IMPROVEMENTS (P2) 📊

### 2.1 M15 Trend Independente

**Arquivo:** `CMTFManager.mqh`

**Problema:** Hoje `m_mtf.trend` herda direcao do H1.

**Solucao:**
- Calcular trend proprio em M15
- Rebaixar alinhamento quando H1 e M15 divergem
- Bloquear sinais quando H1 e M15 opostos

**Esforco:** 3-4 horas | **Status:** ❌ PLANNED

---

### 2.2 BOS/CHoCH M15 Melhorado

**Arquivo:** `CMTFManager.mqh`

**Melhorias:**
- Analisar cadeia de 3-5 swings (nao so ultimo)
- Exigir rompimento minimo em ATR (> 0.5-1.0 ATR)
- Marcar "trend leg" vs "pullback leg"

**Esforco:** 3-4 horas | **Status:** ❌ PLANNED

---

### 2.3 Session Gating no MTF

**Arquivo:** `CMTFManager.mqh`

**Atual:** Campo `session_ok` existe mas nao e usado para gating.

**Proposto:**
- Integrar filtro de sessao em `GetConfluence()`
- Fora de sessao core: reduzir confianca ou exigir mais alinhamento

**Esforco:** 2-3 horas | **Status:** ❌ PLANNED

---

### 2.4 Tuning de Thresholds

**Melhorias:**
- Expor `m_min_trend_strength` e `m_min_confluence` nos inputs
- Ajustar por regime (mais confluencia em noisy)
- Calibrar via WFA

**Esforco:** 2-3 horas | **Status:** ❌ PLANNED

---

### 2.5 Telemetria MTF Debug

**Adicionar:**
- Dump de `SMTFConfluence` no sinal
- Flags de qual pilar faltou quando sinal rejeitado

**Esforco:** 2-3 horas | **Status:** ❌ PLANNED

---

## PHASE 2: INTELLIGENCE LAYER (P2) 🧠

### 2.6 Fibonacci Clusters

**Novo Arquivo:** `CFibClusterDetector.mqh`

```mql5
struct SFibCluster
{
   double price_level;
   int    fib_count;        // Quantos Fibs convergem
   int    strength;         // 1-5 baseado em fib_count
};

class CFibClusterDetector
{
   double m_tolerance;      // 20 pontos
   SFibCluster[] FindClusters(int num_swings = 3);
};
```

**Esforco:** 4-6 horas | **Status:** ❌ PLANNED

---

### 2.7 Regime Ensemble

```
┌─────────────────────────────────────────────────────────────────┐
│                    REGIME ENSEMBLE                              │
├─────────────────────────────────────────────────────────────────┤
│  Hurst R/S ──────┐                                              │
│  Shannon Entropy ─┼──► Voting + Weighted Average ──► REGIME    │
│  HMM (2-state) ───┤         com confianca                      │
│  Fractal Dim ─────┘                                             │
└─────────────────────────────────────────────────────────────────┘
```

**Status:** Hurst e Entropy JA implementados. Falta HMM e Fractal Dim.

**Esforco:** 1-2 dias | **Status:** ❌ PLANNED (parcial)

---

### 2.8 HMM Regime

**Novo Arquivo:** `CHMMRegime.mqh`

```mql5
class CHMMRegime
{
   double m_transition[2][2];     // Matriz de transicao
   double m_state_prob[2];        // P(trending), P(ranging)
   
   void Update(double returns[]);
   int GetMostLikelyState();
};
```

**Esforco:** 1 dia | **Status:** ❌ PLANNED

---

### 2.9 Adaptive Timeframes

**Arquivo:** `CMTFManager.mqh`

```mql5
void AdaptTimeframes()
{
   double vol_ratio = m_current_atr / m_avg_atr_20d;
   
   if(vol_ratio > 1.5)      // Alta vol
   {
      m_htf = PERIOD_H4; m_mtf = PERIOD_H1; m_ltf = PERIOD_M15;
   }
   else if(vol_ratio < 0.7) // Baixa vol
   {
      m_htf = PERIOD_M30; m_mtf = PERIOD_M15; m_ltf = PERIOD_M5;
   }
}
```

**Esforco:** 3-4 horas | **Status:** ❌ PLANNED

---

### 2.10 Dynamic Circuit Breaker

**Arquivo:** `FTMO_RiskManager.mqh`

```mql5
struct SCircuitBreaker
{
   int    consecutive_losses;      // Trigger: 3
   double daily_loss_percent;      // Trigger: 2%
   double hourly_loss_percent;     // Trigger: 1%
   int    trades_per_hour;         // Trigger: 5
   
   bool IsTriggered();
   int GetCooldownMinutes();
};
```

**Esforco:** 3-4 horas | **Status:** ❌ PLANNED

---

## PHASE 3: MACHINE LEARNING (P3) 🤖

### 3.1 Transformer-Lite para Direcao

```
Input: [OHLCV + Regime + Indicators] x 100 bars

┌─────────────────┐
│ Multi-Head      │  4 heads, 32 dim
│ Self-Attention  │
└────────┬────────┘
         ▼
┌─────────────────────────────────────────┐
│ Output:                                  │
│  • Direction (softmax 3-class)           │
│  • Magnitude (regression)                │
│  • Confidence (sigmoid)                  │
└─────────────────────────────────────────┘

Total params: ~10K | Latency: ~3ms
```

**Esforco:** 3-5 dias | **Status:** ❌ PLANNED

---

### 3.2 Imbalance Detection (Python)

**Arquivo a criar:** `Python_Agent_Hub/ml_pipeline/indicators/imbalance_detector.py`

```python
def detect_imbalance_patterns(df):
    # Volume Spike (> 2x media)
    # Rejection Candle (corpo pequeno, sombra grande)
    # Absorption (preco parado, volume alto)
    # Breakout with Volume
```

**Esforco:** 4-6 horas | **Status:** ❌ PLANNED

---

## PHASE 4: IDEAS (P4) 💡

### 4.1 Spread Predictor LSTM
Prever spread nos proximos 5 minutos para timing de entrada.

### 4.2 Meta-Learning Strategy Selector
Market Context → Meta-Model → Strategy Selection

### 4.3 Portfolio Heat Management
Gerenciar risco total com multiplas posicoes abertas.

---

## ARQUIVOS JA IMPLEMENTADOS (REFERENCIA)

### Python - Indicators
```
Python_Agent_Hub/ml_pipeline/indicators/
├── volume_profile.py    ✅ POC/VAH/VAL
├── volume_delta.py      ✅ Delta from ticks/OHLCV
└── __init__.py
```

### Python - Risk
```
Python_Agent_Hub/ml_pipeline/risk/
├── r_multiple_tracker.py  ✅ Van Tharp R-Multiple
├── risk_of_ruin.py        ✅ Ralph Vince Monte Carlo
└── __init__.py
```

### MQL5 - Regime Detection
```
MQL5/Include/EA_SCALPER/Analysis/CRegimeDetector.mqh
├── ✅ Hurst Exponent (R/S method)
├── ✅ Multi-scale Hurst (50/100/200)
├── ✅ Shannon Entropy
├── ✅ Variance Ratio Test
├── ✅ Regime Classification (4 tipos)
├── ✅ Transition Detection (velocity, proximity)
└── ✅ Confidence Scoring
```

---

## PRIORITY MATRIX

| Feature | Impact | Effort | Priority | Status |
|---------|--------|--------|----------|--------|
| Fibonacci Golden Pocket | HIGH | LOW | **P1** | ❌ TODO |
| Fibonacci Extensions TP | HIGH | LOW | **P1** | ❌ TODO |
| Fib Score Confluence | HIGH | LOW | **P1** | ❌ TODO |
| Bayesian Confluence | HIGH | LOW | **P1** | ❌ TODO |
| Adaptive Kelly | HIGH | LOW | **P1** | ❌ TODO |
| Spread Awareness | HIGH | LOW | **P1** | ❌ TODO |
| M15 Trend Independente | MEDIUM | MEDIUM | P2 | ❌ PLANNED |
| BOS/CHoCH M15 Melhorado | MEDIUM | MEDIUM | P2 | ❌ PLANNED |
| Session Gating MTF | MEDIUM | LOW | P2 | ❌ PLANNED |
| Fibonacci Clusters | MEDIUM | MEDIUM | P2 | ❌ PLANNED |
| Regime Ensemble | HIGH | MEDIUM | P2 | ⚠️ PARCIAL |
| HMM Regime | MEDIUM | MEDIUM | P2 | ❌ PLANNED |
| Adaptive TFs | MEDIUM | LOW | P2 | ❌ PLANNED |
| Circuit Breaker | MEDIUM | LOW | P2 | ❌ PLANNED |
| Tuning Thresholds | MEDIUM | LOW | P2 | ❌ PLANNED |
| Telemetria MTF | LOW | LOW | P3 | ❌ PLANNED |
| Imbalance Detection | LOW | MEDIUM | P3 | ❌ PLANNED |
| Transformer ONNX | HIGH | HIGH | P3 | ❌ PLANNED |
| Spread Predictor | LOW | MEDIUM | P4 | 💡 IDEA |
| Meta-Learning | LOW | HIGH | P4 | 💡 IDEA |
| Portfolio Heat | LOW | MEDIUM | P4 | 💡 IDEA |

---

## CHANGELOG

| Date | Change |
|------|--------|
| 2025-12-01 | **CONSOLIDADO** - Merge de 3 arquivos (FUTURE_IMPROVEMENTS, MTF_FUTURE_IMPROVEMENTS, FUTURE_IMPLEMENTATIONS) |
| 2025-12-01 | Added status de implementacao (o que ja foi feito vs pendente) |
| 2025-12-01 | Added Fibonacci Integration (ARGUS research) |
| 2024-11-29 | FUTURE_IMPLEMENTATIONS.md criado (Volume Profile, R-Multiple, Risk of Ruin) |

---

## ARQUIVOS DEPRECADOS (REMOVER)

Apos confirmar este consolidado, remover:
- `DOCS/02_IMPLEMENTATION/MTF_FUTURE_IMPROVEMENTS.md`
- `DOCS/02_IMPLEMENTATION/FUTURE_IMPLEMENTATIONS.md`

---

*"O codigo de hoje e o legado de amanha. Construa para durar."* - FORGE
