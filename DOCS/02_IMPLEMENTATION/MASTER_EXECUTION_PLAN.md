# MASTER EXECUTION PLAN v2.0 - EA_SCALPER_XAUUSD
## Do Código ao Challenge FTMO - Validação Institucional

**Criado**: 2025-12-01
**Atualizado**: 2025-12-01 (Party Mode Evolution)
**Versão**: 2.0 - Institutional Grade
**Tempo Total Estimado**: 4-6 semanas

---

## ÍNDICE

1. [Visão Geral](#visão-geral)
2. [Fase 0: Audit do Código](#fase-0-audit-do-código) ✅
3. [Fase 1: Validação de Dados](#fase-1-validação-de-dados)
4. [Fase 2: Backtest Baseline Multi-Regime](#fase-2-backtest-baseline-multi-regime)
5. [Fase 3: Treinamento ML](#fase-3-treinamento-ml)
6. [Fase 4: Shadow Exchange Validation](#fase-4-shadow-exchange-validation) 🆕
7. [Fase 5: Validação Estatística Institucional](#fase-5-validação-estatística-institucional) 🔄
8. [Fase 6: Stress Testing Extremo](#fase-6-stress-testing-extremo) 🆕
9. [Fase 7: Demo Trading](#fase-7-demo-trading)
10. [Fase 8: Challenge FTMO](#fase-8-challenge-ftmo)
11. [Checklist Geral](#checklist-geral)
12. [Apêndices](#apêndices)

---

## VISÃO GERAL

### Filosofia v2.0: "Simule a FÍSICA, não apenas a LÓGICA"

> A maioria dos backtests falha porque simula a **lógica** da estratégia mas ignora a **física** da infraestrutura. Para um scalper híbrido (Python + MQL5) operando em XAUUSD numa Prop Firm, a latência não é uma constante - é uma variável estocástica brutal.
> 
> — ARGUS Research, 2025-11-30

### Princípios Fundamentais

| Princípio | Descrição | Implementação |
|-----------|-----------|---------------|
| **Tick Data First** | Sempre usar dados de tick para máxima precisão | 24GB tick data disponível |
| **Event-Driven** | Simular execução realista, não vetorizada | Shadow Exchange em Python |
| **Multi-Regime** | Validar em TODOS os regimes de mercado | Trending, Ranging, Reverting |
| **Multi-Sessão** | XAUUSD se comporta diferente por sessão | Asia, London, NY, Overlap |
| **Latência Estocástica** | Latência como variável aleatória, não constante | Gamma + Poisson model |
| **Custos Realistas** | Spread, slippage, latência, rejeições, packet loss | CBacktestRealism.mqh + Python |
| **Statistical Rigor** | WFA, CPCV, Monte Carlo, PSR/DSR/PBO, MinTRL | Oracle Pipeline v2.0 |
| **Stress Testing** | Simular cenários extremos antes de arriscar capital | 6 cenários de stress |

### Status Atual (2025-12-01)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PROGRESSO DO PROJETO v2.0                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  FASE 0: AUDIT DO CÓDIGO      ████████████████████ 100% ✅ COMPLETA         │
│  ├── Compilação               ✅ 0 erros, 0 warnings                        │
│  ├── FTMO_RiskManager         ✅ 20/20                                      │
│  ├── CTradeManager            ✅ 18/20                                      │
│  ├── CRegimeDetector          ✅ 19/20                                      │
│  ├── CMTFManager              ✅ 20/20 (upgrade)                            │
│  ├── CFootprintAnalyzer       ✅ 20/20 v3.1 (upgrade)                       │
│  └── CConfluenceScorer        ✅ 20/20 (de stub para full)                  │
│                                                                             │
│  FASE 1: VALIDAÇÃO DADOS      ░░░░░░░░░░░░░░░░░░░░   0% ⬜ PENDENTE         │
│  FASE 2: BACKTEST MULTI-REG   ░░░░░░░░░░░░░░░░░░░░   0% ⬜ PENDENTE         │
│  FASE 3: TREINAMENTO ML       ░░░░░░░░░░░░░░░░░░░░   0% ⬜ PENDENTE         │
│  FASE 4: SHADOW EXCHANGE      ░░░░░░░░░░░░░░░░░░░░   0% ⬜ PENDENTE  🆕     │
│  FASE 5: VALIDAÇÃO ORACLE     ░░░░░░░░░░░░░░░░░░░░   0% ⬜ PENDENTE  🔄     │
│  FASE 6: STRESS TESTING       ░░░░░░░░░░░░░░░░░░░░   0% ⬜ PENDENTE  🆕     │
│  FASE 7: DEMO TRADING         ░░░░░░░░░░░░░░░░░░░░   0% ⬜ PENDENTE         │
│  FASE 8: CHALLENGE FTMO       ░░░░░░░░░░░░░░░░░░░░   0% ⬜ PENDENTE         │
│                                                                             │
│  PROGRESSO GERAL: 12.5% (1/8 fases)                                         │
│  PRÓXIMO PASSO: Fase 1 - Validação de Dados                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Fluxo de Fases v2.0

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FLUXO DE VALIDAÇÃO v2.0                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  FASE 0 ──▶ FASE 1 ──▶ FASE 2 ──▶ FASE 3 ──▶ FASE 4 ──▶ FASE 5             │
│   AUDIT      DATA    BASELINE     ML      SHADOW    ORACLE                  │
│   1-2d       1d       3-4d       3-5d      3-4d      3-4d                   │
│    ✅                                                                       │
│                                                                             │
│  FASE 5 ──▶ FASE 6 ──▶ FASE 7 ──▶ FASE 8                                   │
│   ORACLE    STRESS     DEMO      FTMO                                       │
│   3-4d      2-3d      2 sem     4+ sem                                      │
│                                                                             │
│  ⚠️ GATES DE DECISÃO:                                                       │
│  ├── Após FASE 2: Se PF < 1.3 → PARAR e revisar estratégia                 │
│  ├── Após FASE 4: Se divergência MT5 vs Shadow > 15% → Investigar          │
│  ├── Após FASE 5: Se Confidence < 70 → NO-GO                               │
│  └── Após FASE 6: Se falhar stress test crítico → NO-GO                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Critérios de Aprovação (GO/NO-GO) v2.0

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THRESHOLDS FTMO $100k - INSTITUTIONAL                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  VALIDAÇÃO ESTATÍSTICA (Fase 5):                                           │
│  ├── WFE Global                          >= 0.60                            │
│  ├── WFE Trending (Hurst > 0.55)         >= 0.65                            │
│  ├── WFE Ranging (0.45-0.55)             >= 0.50                            │
│  ├── WFE Reverting (Hurst < 0.45)        >= 0.45 (ou não operar)           │
│  ├── OOS Windows Positivos               >= 70%                             │
│  ├── Monte Carlo 95th DD (baseline)      < 8%                               │
│  ├── Monte Carlo 95th DD (pessimistic)   < 10%                              │
│  ├── Monte Carlo 95th DD (stress)        < 12%                              │
│  ├── PSR (Probabilistic Sharpe)          >= 0.90                            │
│  ├── DSR (Deflated Sharpe)               > 0                                │
│  ├── PBO (Probability Backtest Overfit)  < 0.50                             │
│  ├── MinTRL vs Trades Disponíveis        trades >= MinTRL                   │
│  └── Confidence Score                    >= 75 (não 70!)                    │
│                                                                             │
│  VALIDAÇÃO FTMO ESPECÍFICA:                                                 │
│  ├── P(Daily DD > 5%)                    < 5%                               │
│  ├── P(Total DD > 10%)                   < 2%                               │
│  ├── P(Daily DD > 4% buffer)             < 10%                              │
│  ├── P(Total DD > 8% buffer)             < 5%                               │
│  └── Profit Target Viável (10%/30d)      P > 50%                            │
│                                                                             │
│  STRESS TESTING (Fase 6):                                                   │
│  ├── News Storm Test                     PASS                               │
│  ├── Flash Crash Test                    PASS                               │
│  ├── Connection Loss Test                PASS                               │
│  ├── Regime Transition Test              PASS                               │
│  ├── Liquidity Dry-up Test               PASS                               │
│  └── Circuit Breaker Failure Test        PASS                               │
│                                                                             │
│  DECISÃO FINAL:                                                             │
│  ├── STRONG_GO: Todos passam + Confidence >= 85 + zero falhas stress       │
│  ├── GO:        Todos passam + Confidence >= 75                             │
│  ├── CAUTIOUS:  1-2 falhas marginais + Confidence 65-74                     │
│  └── NO_GO:     Qualquer falha crítica ou Confidence < 65                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Recursos de Dados Disponíveis

| Dataset | Path | Tamanho | Período | Uso |
|---------|------|---------|---------|-----|
| **Tick Data Principal** | `Python_Agent_Hub/ml_pipeline/data/XAUUSD_ftmo_all_desde_2003.csv` | 24.3 GB | 2003-2025 | Shadow Exchange |
| Tick 2020 (backup) | `XAUUSD_ftmo_2020_ticks_dukascopy.csv` | ~2 GB | 2020 | Testes rápidos |
| M5 Bars | `Bars_2020-2025XAUUSD_ftmo-M5-No Session.csv` | 22 MB | 2020-2025 | Referência |
| RAG MQL5 Docs | `.rag-db/docs/` | - | - | Sintaxe |
| RAG Books | `.rag-db/books/` | - | - | Conceitos ML |

---

## FASE 0: AUDIT DO CÓDIGO

**Duração**: 1-2 dias ✅ COMPLETA
**Status**: 100% - Score Médio 19.5/20

### Resumo de Resultados

| Módulo | Score | Status | Melhorias |
|--------|-------|--------|-----------|
| EA_SCALPER_XAUUSD.mq5 | ✅ | Compila 0 erros | - |
| FTMO_RiskManager | 20/20 | ✅ Aprovado | DD buffers, GV persistence |
| CTradeManager | 18/20 | ✅ Aprovado | Retry 3x, partials, trailing |
| CRegimeDetector | 19/20 | ✅ Aprovado | Hurst R/S, entropy, array fix |
| CMTFManager | 20/20 | ✅ Upgrade | Hurst impl, trend strength 2-comp |
| CFootprintAnalyzer | 20/20 | ✅ Upgrade v3.1 | Absorption confidence scoring |
| CConfluenceScorer | 20/20 | ✅ Upgrade | De stub 10/20 para full impl |

**Documentação**: Todos os bugs registrados em `MQL5/Experts/BUGFIX_LOG.md`

---

## FASE 1: VALIDAÇÃO DE DADOS

**Duração**: 1-2 dias
**Sessões**: 2 simultâneas
**Objetivo**: Garantir que dados históricos são confiáveis para validação institucional

### 1.1 Sessão A: Scripts de Validação

#### Tarefa 1.A.1: Criar Script de Validação de Tick Data

```
PROMPT PARA FORGE:

"Forge, crie um script Python robusto para validar tick data XAUUSD:

Salvar em: scripts/validate_data.py

REQUISITOS:

1. CARREGAMENTO EFICIENTE (24GB de dados):
   - Usar file seeking, não carregar tudo em memória
   - Processar em chunks de 1M linhas
   - Progress bar para feedback

2. VERIFICAÇÕES DE INTEGRIDADE:
   - Formato correto (datetime, bid, ask)
   - Timestamps em ordem cronológica
   - Sem duplicatas de timestamp
   - Spread bid-ask sempre positivo
   - Spread não excede 500 pontos (anomalia)
   - Preços dentro de range histórico ($1000-$3500)
   - Sem preços zero ou negativos

3. ANÁLISE DE GAPS:
   - Identificar gaps > 5 minutos (exceto weekends)
   - Classificar: Gap normal (< 1h), Gap longo (1-24h), Gap crítico (> 24h)
   - Excluir weekends (sexta 22:00 UTC - domingo 22:00 UTC)

4. ANÁLISE DE SPREAD POR SESSÃO:
   - Calcular spread médio/max por sessão (Asia, London, NY)
   - Identificar anomalias (spread > 3x média)
   - Verificar se spread durante news é realista

5. ANÁLISE DE REGIME:
   - Calcular Hurst por período (rolling 1000 ticks)
   - Identificar % do tempo em cada regime
   - Verificar diversidade de regimes nos dados

6. RELATÓRIO:
   - Total de ticks
   - Período coberto (data início/fim)
   - % de dados limpos
   - Distribuição por sessão
   - Distribuição por regime
   - Lista de problemas críticos
   - SCORE de qualidade (0-100)

7. CRITÉRIOS DE APROVAÇÃO:
   - Mínimo 3 anos de dados (para WFA robusto)
   - >= 98% de dados limpos
   - Gaps críticos < 0.1% do tempo
   - Todas as sessões representadas
   - Todos os regimes representados (trending, ranging, reverting)
   - Score de qualidade >= 90

Usar pandas com chunks, tqdm para progress, gerar relatório markdown."
```

**Critério de Sucesso**: Score de qualidade >= 90

---

#### Tarefa 1.A.2: Executar Validação e Gerar Relatório

```
PROMPT PARA FORGE:

"Forge, execute a validação completa:

1. Execute: python scripts/validate_data.py --input [tick_data_path] --output DOCS/04_REPORTS/VALIDATION/DATA_QUALITY_REPORT.md

2. Se score < 90:
   - Identificar problemas específicos
   - Propor soluções (interpolação, remoção, download adicional)
   - Re-executar após correções

3. Gerar resumo executivo com:
   - Score final
   - Períodos recomendados para IS/OOS
   - Alertas sobre períodos problemáticos
   - Recomendação de sessões para evitar/focar
"
```

---

#### Tarefa 1.A.3: Criar Script de Conversão para Formatos de Backtester

```
PROMPT PARA FORGE:

"Forge, crie script para converter tick data para formatos usados pelos backtesters:

Salvar em: scripts/convert_tick_data.py

FORMATOS DE SAÍDA:

1. NPZ (para HftBacktest):
   - Arrays numpy comprimidos
   - Colunas: timestamp_ns, bid, ask, bid_size, ask_size
   - Tamanho de posição default se não disponível

2. Parquet (para análise rápida):
   - Particionado por ano/mês
   - Compressão snappy
   - Schema otimizado

3. CSV chunks (para processamento incremental):
   - Arquivos de 1M linhas cada
   - Nomeação: XAUUSD_ticks_YYYYMM_NNN.csv

4. MT5 format (para backtest MT5):
   - Se precisar importar de volta

Incluir:
- Validação durante conversão
- Resume de conversão interrompida
- Verificação de integridade pós-conversão
"
```

---

### 1.2 Sessão B: Pesquisa de Data Quality (PARALELO)

#### Tarefa 1.B.1: Pesquisar Melhores Práticas

```
PROMPT PARA ARGUS:

"Argus, pesquise melhores práticas para validação de dados de backtest institucional:

TRIANGULAÇÃO NECESSÁRIA:

1. ACADÊMICO:
   - Papers sobre data quality em backtesting
   - Métodos de detecção de survivorship bias
   - Técnicas de limpeza de dados tick

2. PRÁTICO:
   - Como fundos quant validam dados?
   - Ferramentas comerciais (TickData, Dukascopy)
   - Problemas comuns com dados retail vs institutional

3. EMPÍRICO:
   - Fóruns (QuantConnect, Quantopian archives)
   - GitHub repos de data validation
   - Experiências documentadas de traders

PERGUNTAS A RESPONDER:
1. Qual o mínimo de dados para backtest estatisticamente válido?
2. Como detectar look-ahead bias em dados?
3. Como validar que spread histórico é realista?
4. Diferença entre tick data Dukascopy vs broker real?
5. Como tratar gaps de dados corretamente?

SALVAR EM: DOCS/03_RESEARCH/FINDINGS/DATA_QUALITY_BEST_PRACTICES.md
"
```

---

### 1.3 Checkpoint Fase 1

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CHECKPOINT FASE 1                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  □ Script validate_data.py criado e testado                                │
│  □ Tick data validado (24GB)                                               │
│  □ Score de qualidade >= 90                                                │
│  □ Mínimo 3 anos de dados limpos                                           │
│  □ Todas as sessões representadas                                          │
│  □ Todos os regimes representados                                          │
│  □ Gaps críticos < 0.1%                                                    │
│  □ Script de conversão criado                                              │
│  □ Dados convertidos para NPZ/Parquet                                      │
│  □ Relatório DATA_QUALITY_REPORT.md gerado                                 │
│  □ Pesquisa de best practices concluída                                    │
│                                                                             │
│  SE TODOS ✅ → Prosseguir para FASE 2                                      │
│  SE ALGUM ❌ → Corrigir dados ou obter dados adicionais                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## FASE 2: BACKTEST BASELINE MULTI-REGIME

**Duração**: 3-4 dias
**Sessões**: 3 simultâneas
**Objetivo**: Verificar se estratégia funciona em TODOS os regimes e sessões, SEM ML

### 2.1 Conceito: Por que Multi-Regime?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PROBLEMA DO BACKTEST TRADICIONAL                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  BACKTEST TRADICIONAL:                                                      │
│  ├── Roda em TODO o período                                                │
│  ├── Obtém métricas AGREGADAS                                              │
│  ├── Resultado: "PF = 1.5, Win Rate = 55%"                                 │
│  └── PROBLEMA: Pode ser 90% trending (fácil) + 10% ranging (perdedor)      │
│                                                                             │
│  BACKTEST MULTI-REGIME:                                                     │
│  ├── Segmenta dados por regime (Hurst)                                     │
│  ├── Roda backtest SEPARADO em cada regime                                 │
│  ├── Resultado: "PF_trending=1.8, PF_ranging=0.9, PF_reverting=1.2"       │
│  └── INSIGHT: Saber ONDE a estratégia funciona e onde falha               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Definição de Regimes e Sessões

```
REGIMES (baseado em Hurst Exponent):
├── TRENDING:   Hurst > 0.55 (momentum funciona)
├── RANDOM:     0.45 <= Hurst <= 0.55 (NÃO OPERAR)
└── REVERTING:  Hurst < 0.45 (mean reversion funciona)

SESSÕES (horário UTC):
├── ASIA:       00:00 - 07:00 (baixa liquidez, spread alto)
├── LONDON:     07:00 - 12:00 (alta liquidez, melhor spread)
├── OVERLAP:    12:00 - 16:00 (máxima liquidez)
├── NEW_YORK:   16:00 - 21:00 (boa liquidez)
└── CLOSE:      21:00 - 00:00 (baixa liquidez)

MATRIZ DE TESTES (15 combinações):
┌─────────────┬──────────┬──────────┬──────────┐
│             │ TRENDING │ RANDOM   │REVERTING │
├─────────────┼──────────┼──────────┼──────────┤
│ ASIA        │ Test 1   │ Test 2   │ Test 3   │
│ LONDON      │ Test 4   │ Test 5   │ Test 6   │
│ OVERLAP     │ Test 7   │ Test 8   │ Test 9   │
│ NEW_YORK    │ Test 10  │ Test 11  │ Test 12  │
│ CLOSE       │ Test 13  │ Test 14  │ Test 15  │
└─────────────┴──────────┴──────────┴──────────┘

EXPECTATIVAS:
├── RANDOM: NÃO DEVE OPERAR (filter bloqueia)
├── ASIA + qualquer: Menor PF (spread alto)
├── OVERLAP + TRENDING: Melhor PF esperado
└── Se PF < 1.0 em qualquer combinação válida: Investigar
```

### 2.3 Sessão A: Configurar e Segmentar Dados

#### Tarefa 2.A.1: Criar Script de Segmentação por Regime

```
PROMPT PARA FORGE:

"Forge, crie script para segmentar dados por regime e sessão:

Salvar em: scripts/backtest/segment_data.py

FUNCIONALIDADES:

1. DETECÇÃO DE REGIME:
   - Calcular Hurst Exponent rolling (window=1000 ticks)
   - Calcular Shannon Entropy rolling
   - Classificar cada período como TRENDING/RANDOM/REVERTING
   - Detectar transições de regime

2. DETECÇÃO DE SESSÃO:
   - Mapear timestamp para sessão (ASIA/LONDON/OVERLAP/NY/CLOSE)
   - Ajustar para horário de verão (DST)

3. SEGMENTAÇÃO:
   - Criar datasets separados por regime
   - Criar datasets separados por sessão
   - Criar datasets combinados (regime + sessão)
   - Manter timestamps originais para referência

4. ESTATÍSTICAS:
   - % de tempo em cada regime
   - % de tempo em cada sessão
   - Volatilidade média por segmento
   - Spread médio por segmento

5. OUTPUT:
   - data/segments/regime_trending.parquet
   - data/segments/regime_random.parquet
   - data/segments/regime_reverting.parquet
   - data/segments/session_*.parquet
   - data/segments/combined_*.parquet
   - data/segments/SEGMENT_STATS.json

USO:
python scripts/backtest/segment_data.py --input [tick_data] --output data/segments/
"
```

---

#### Tarefa 2.A.2: Criar Backtester Event-Driven Base

```
PROMPT PARA FORGE:

"Forge, crie o backtester event-driven para baseline tests:

Salvar em: scripts/backtest/event_backtester.py

ARQUITETURA:

class EventBacktester:
    '''
    Backtester event-driven que processa tick por tick.
    NÃO usa vetorização para evitar look-ahead bias.
    '''
    
    def __init__(self, config: BacktestConfig):
        self.data_feed = TickDataFeed(config.data_path)
        self.strategy = config.strategy
        self.execution = ExecutionSimulator(config.execution_params)
        self.portfolio = Portfolio(config.initial_capital)
        self.risk_manager = RiskManager(config.risk_params)
        self.logger = TradeLogger()
    
    def run(self):
        for tick in self.data_feed:
            # 1. Atualizar estado do mercado
            self.strategy.on_tick(tick)
            
            # 2. Verificar SL/TP de posições abertas
            self.portfolio.check_exits(tick, self.execution)
            
            # 3. Gerar sinais
            signal = self.strategy.generate_signal(tick)
            
            # 4. Verificar risco
            if signal and self.risk_manager.can_trade(signal):
                # 5. Simular execução com latência
                fill = self.execution.execute(signal, tick)
                
                if fill:
                    self.portfolio.add_position(fill)
                    self.logger.log_entry(fill)
            
            # 6. Atualizar métricas de risco
            self.risk_manager.update(self.portfolio)
        
        return self.logger.get_trades()

COMPONENTES A IMPLEMENTAR:

1. TickDataFeed: Iterator eficiente sobre tick data
2. ExecutionSimulator: Simula slippage, spread, latência, rejeições
3. Portfolio: Gerencia posições, calcula equity, DD
4. RiskManager: Daily/Total DD, circuit breaker
5. TradeLogger: Registra trades no formato Oracle

ESTRATÉGIAS BASE (para testes):

1. strategies/ma_cross.py - MA Cross simples (baseline burro)
2. strategies/regime_filtered.py - MA Cross + filtro de regime
3. strategies/session_filtered.py - MA Cross + filtro de sessão
4. strategies/confluence_lite.py - Versão simplificada do EA
5. strategies/full_ea.py - Lógica completa do EA (portada)

OUTPUT: trades.csv no formato Oracle-compatible
"
```

---

### 2.4 Sessão B: Executar Backtests por Segmento (PARALELO)

#### Tarefa 2.B.1: Rodar Baseline em Cada Regime

```
PROMPT PARA FORGE:

"Forge, execute backtests separados por regime:

CONFIGURAÇÃO COMUM:
├── Capital: $100,000
├── Risk per trade: 0.5%
├── SL/TP: Conforme estratégia
├── Custos: NORMAL (spread 25pts, slippage 5pts, latência 50ms)
├── Período: 2020-2024 (IS para WFA)

EXECUÇÕES:

1. TRENDING (Hurst > 0.55):
   python scripts/backtest/run_backtest.py \
     --data data/segments/regime_trending.parquet \
     --strategy confluence_lite \
     --output data/results/baseline_trending.csv

2. RANDOM (0.45-0.55):
   python scripts/backtest/run_backtest.py \
     --data data/segments/regime_random.parquet \
     --strategy confluence_lite \
     --output data/results/baseline_random.csv
   
   EXPECTATIVA: Zero trades (filtro deve bloquear)

3. REVERTING (Hurst < 0.45):
   python scripts/backtest/run_backtest.py \
     --data data/segments/regime_reverting.parquet \
     --strategy confluence_lite \
     --output data/results/baseline_reverting.csv

PARA CADA RESULTADO, CALCULAR:
├── Total de trades
├── Win Rate
├── Profit Factor
├── Max Drawdown (%)
├── Sharpe Ratio
├── Average Trade ($)
└── Average R:R

GERAR TABELA COMPARATIVA:
| Regime | Trades | WR | PF | Max DD | Sharpe |
|--------|--------|----|----|--------|--------|
"
```

---

#### Tarefa 2.B.2: Rodar Baseline em Cada Sessão

```
PROMPT PARA FORGE:

"Forge, execute backtests separados por sessão:

EXECUÇÕES:

1. ASIA (00:00-07:00 UTC):
   python scripts/backtest/run_backtest.py \
     --data data/segments/session_asia.parquet \
     --strategy confluence_lite \
     --output data/results/baseline_asia.csv

2. LONDON (07:00-12:00 UTC):
   python scripts/backtest/run_backtest.py \
     --data data/segments/session_london.parquet \
     --strategy confluence_lite \
     --output data/results/baseline_london.csv

3. OVERLAP (12:00-16:00 UTC):
   python scripts/backtest/run_backtest.py \
     --data data/segments/session_overlap.parquet \
     --strategy confluence_lite \
     --output data/results/baseline_overlap.csv

4. NEW_YORK (16:00-21:00 UTC):
   python scripts/backtest/run_backtest.py \
     --data data/segments/session_ny.parquet \
     --strategy confluence_lite \
     --output data/results/baseline_ny.csv

5. CLOSE (21:00-00:00 UTC):
   python scripts/backtest/run_backtest.py \
     --data data/segments/session_close.parquet \
     --strategy confluence_lite \
     --output data/results/baseline_close.csv

GERAR TABELA COMPARATIVA:
| Sessão | Trades | WR | PF | Max DD | Spread Médio |
|--------|--------|----|----|--------|--------------|

IDENTIFICAR:
├── Melhor sessão (maior PF)
├── Pior sessão (menor PF)
├── Sessões a evitar (PF < 1.0)
└── Correlação spread vs performance
"
```

---

### 2.5 Sessão C: Análise de Resultados (PARALELO)

#### Tarefa 2.C.1: Análise Comparativa Multi-Regime

```
PROMPT PARA ORACLE:

"Oracle, analise os resultados dos backtests multi-regime:

DADOS:
├── data/results/baseline_trending.csv
├── data/results/baseline_random.csv
├── data/results/baseline_reverting.csv
├── data/results/baseline_*.csv (sessões)

ANÁLISE REQUERIDA:

1. TABELA COMPARATIVA COMPLETA:
   Regime x Sessão com todas as métricas

2. IDENTIFICAÇÃO DE PONTOS FORTES:
   - Onde a estratégia performa melhor?
   - Existe edge consistente?
   - Qual combinação regime+sessão é ótima?

3. IDENTIFICAÇÃO DE PONTOS FRACOS:
   - Onde a estratégia perde?
   - O filtro de regime está funcionando?
   - Alguma sessão deve ser bloqueada?

4. ANÁLISE DE SEQUÊNCIAS:
   - Max losing streak por segmento
   - Recovery time médio
   - Correlação entre segmentos (diversificação?)

5. RECOMENDAÇÕES:
   - Ajustar parâmetros por sessão?
   - Desabilitar trading em algum segmento?
   - Ajustar risk por regime?

SALVAR EM: DOCS/04_REPORTS/BACKTESTS/MULTI_REGIME_ANALYSIS.md

THRESHOLDS MÍNIMOS PARA CONTINUAR:
├── PF Global >= 1.3
├── PF Trending >= 1.5
├── PF Reverting >= 1.0 (ou não operar)
├── Zero trades em RANDOM
├── Max DD <= 15%
├── >= 100 trades total
"
```

---

### 2.6 Checkpoint Fase 2

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CHECKPOINT FASE 2                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  □ Script segment_data.py criado e executado                               │
│  □ Dados segmentados por regime (3 arquivos)                               │
│  □ Dados segmentados por sessão (5 arquivos)                               │
│  □ EventBacktester implementado e testado                                  │
│  □ Estratégias base portadas (ma_cross, confluence_lite)                   │
│                                                                             │
│  □ Backtest por regime executado (3 runs)                                  │
│  □ Backtest por sessão executado (5 runs)                                  │
│  □ Zero trades em regime RANDOM (filtro funciona)                          │
│                                                                             │
│  □ PF Global >= 1.3                                                        │
│  □ PF Trending >= 1.5                                                      │
│  □ Max DD <= 15%                                                           │
│  □ >= 100 trades total                                                     │
│                                                                             │
│  □ Análise MULTI_REGIME_ANALYSIS.md gerada                                 │
│  □ Pontos fortes identificados                                             │
│  □ Pontos fracos identificados                                             │
│  □ Recomendações de ajuste documentadas                                    │
│                                                                             │
│  SE TODOS ✅ → Prosseguir para FASE 3                                      │
│  SE PF < 1.3 → ⚠️ PARAR! Revisar estratégia antes de continuar            │
│  SE Max DD > 15% → ⚠️ Reduzir risk per trade e re-testar                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## FASE 3: TREINAMENTO ML

**Duração**: 3-5 dias
**Sessões**: 3 simultâneas
**Objetivo**: Treinar modelo ONNX que ADICIONA edge (não substitui)

### 3.1 Princípio: ML como Filtro, não como Estratégia

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FILOSOFIA ML v2.0                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ❌ ERRADO: ML gera sinais de entrada                                      │
│  ├── Problema: Se baseline falha, ML não salva                             │
│  ├── Problema: Overfitting em features                                     │
│  └── Problema: Caixa preta sem explicabilidade                             │
│                                                                             │
│  ✅ CORRETO: ML filtra sinais do baseline                                  │
│  ├── Baseline gera candidatos de trade                                     │
│  ├── ML diz: "Este candidato tem P=0.72 de sucesso"                       │
│  ├── Se P > 0.65: Executar                                                 │
│  └── Se P < 0.65: Skip                                                     │
│                                                                             │
│  BENEFÍCIO: Se ML falhar, baseline ainda funciona (PF 1.3)                 │
│  BENEFÍCIO: ML só precisa ser "melhor que random" para adicionar edge     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Sessão A: Feature Engineering

#### Tarefa 3.A.1: Criar Pipeline de Features

```
PROMPT PARA onnx-model-builder:

"Preciso criar features para o modelo de direção XAUUSD.

CONTEXTO:
├── EA: EA_SCALPER_XAUUSD v3.30
├── Timeframe: M5
├── Integração: COnnxBrain.mqh

FEATURES (15 conforme INDEX.md do EA):

GRUPO 1: PRICE ACTION (5 features)
├── 1. Returns: (close - prev_close) / prev_close
├── 2. Log Returns: log(close / prev_close)
├── 3. Range %: (high - low) / close
├── 4. Body %: abs(close - open) / (high - low + 1e-8)
└── 5. Upper Shadow %: (high - max(open, close)) / (high - low + 1e-8)

GRUPO 2: MULTI-TIMEFRAME RSI (3 features)
├── 6. RSI M5 (14): RSI normalizado / 100
├── 7. RSI M15 (14): RSI normalizado / 100
└── 8. RSI H1 (14): RSI normalizado / 100

GRUPO 3: VOLATILITY (3 features)
├── 9. ATR Norm: ATR(14) / close
├── 10. MA Distance: (close - MA20) / MA20
└── 11. BB Position: (close - BB_mid) / BB_width

GRUPO 4: REGIME (2 features)
├── 12. Hurst: Rolling(100) - raw value
└── 13. Entropy: Rolling(100) / 4 - normalizado

GRUPO 5: TEMPORAL (2 features)
├── 14. Hour Sin: sin(2π × hour / 24)
└── 15. Hour Cos: cos(2π × hour / 24)

NORMALIZAÇÃO:
├── StandardScaler para features contínuas
├── Salvar parâmetros em scaler_params.json
└── Mesma normalização em training e inference

DADOS:
├── Usar tick data resamplado para M5
├── Período: 2020-2024 (IS)
├── Split TEMPORAL: Train 60%, Val 20%, Test 20%
├── NUNCA shuffle em time series!

OUTPUT:
├── scripts/ml/feature_engineering.py
├── data/ml/features_train.parquet
├── data/ml/features_val.parquet
├── data/ml/features_test.parquet
├── data/ml/scaler_params.json
"
```

---

#### Tarefa 3.A.2: Definir Target

```
PROMPT PARA onnx-model-builder:

"Defina o target para o modelo de direção:

TARGET: Direção nas próximas N barras

CONFIGURAÇÃO:
├── N = 6 barras (30 minutos em M5)
├── Threshold: Movimento > 0.1% para ser classificado
│   ├── Se close[t+6] > close[t] * 1.001 → UP (1)
│   ├── Se close[t+6] < close[t] * 0.999 → DOWN (0)
│   └── Se dentro do threshold → NEUTRAL (excluir do training)

BALANCEAMENTO:
├── Verificar distribuição UP/DOWN/NEUTRAL
├── Se desbalanceado (>60/40): Usar class weights
├── Não usar oversampling (causa data leakage em time series)

ALTERNATIVA (se muitos neutrals):
├── Target binário com threshold menor (0.05%)
├── Ou usar regressão (prever retorno) + binarizar depois

OUTPUT:
├── Coluna 'target' adicionada aos features parquet
├── Estatísticas de distribuição no log
"
```

---

### 3.3 Sessão B: Treinamento do Modelo

#### Tarefa 3.B.1: Treinar com Walk-Forward

```
PROMPT PARA onnx-model-builder:

"Treine o modelo usando Walk-Forward Training (NÃO k-fold!):

ARQUITETURA RECOMENDADA:
├── Input: (batch, 100, 15) - 100 barras históricas, 15 features
├── LSTM ou GRU: 64-128 units (não muito grande = overfit)
├── Dropout: 0.3 (regularização)
├── Dense: 32 units
├── Output: 2 classes (softmax)

WALK-FORWARD TRAINING:
├── Janela 1: Train [0:60%], Val [60:80%]
├── Janela 2: Train [10:70%], Val [70:90%]
├── Janela 3: Train [20:80%], Val [80:100%]
├── Para cada janela: Early stopping em val_loss

HIPERPARÂMETROS:
├── Learning rate: 0.001 com decay
├── Batch size: 64-128
├── Epochs: Max 100 com early stopping (patience=10)
├── Optimizer: Adam

MÉTRICAS A RASTREAR:
├── Accuracy (train vs val por janela)
├── AUC-ROC
├── Precision/Recall por classe
├── Calibration: P=0.7 deve significar 70% de acerto real

CRITÉRIOS DE SUCESSO:
├── Accuracy OOS média > 55% (melhor que random)
├── Calibration: Brier score < 0.25
├── Sem overfit severo: IS/OOS accuracy ratio < 1.3

OUTPUT:
├── Modelo salvo em MQL5/Models/direction_model.onnx
├── Scaler params em MQL5/Models/scaler_params.json
├── Training report em DOCS/04_REPORTS/ML/TRAINING_REPORT.md
"
```

---

#### Tarefa 3.B.2: Exportar para ONNX

```
PROMPT PARA onnx-model-builder:

"Exporte o modelo treinado para ONNX:

REQUISITOS ONNX:
├── Input shape: (1, 100, 15) - batch 1 para inference
├── Output shape: (1, 2) - [P(down), P(up)]
├── Opset version: 12+
├── Otimizar para inference (fold constants, etc.)

VERIFICAÇÕES:
├── Testar inference em Python com ONNX Runtime
├── Comparar output PyTorch vs ONNX (devem ser iguais)
├── Verificar que soma das probabilidades = 1.0
├── Medir latência de inference (deve ser < 5ms)

ARQUIVOS:
├── MQL5/Models/direction_model.onnx
├── MQL5/Models/scaler_params.json
├── MQL5/Models/model_metadata.json (arquitetura, versão, etc.)
"
```

---

### 3.4 Sessão C: Integração e Validação (PARALELO)

#### Tarefa 3.C.1: Atualizar COnnxBrain.mqh

```
PROMPT PARA FORGE:

"Forge, atualize COnnxBrain.mqh para usar o novo modelo:

ARQUIVO: MQL5/Include/EA_SCALPER/Bridge/COnnxBrain.mqh

VERIFICAÇÕES:
├── Path do modelo correto
├── Input shape corresponde ao esperado
├── Normalização usando scaler_params.json
├── Output parsing correto

FLUXO DE INFERENCE:
1. Coletar últimas 100 barras M5
2. Calcular 15 features
3. Normalizar com StandardScaler params
4. Reshape para (1, 100, 15)
5. OnnxRun()
6. Extrair P(up) e P(down)
7. Se P(up) > 0.65 → Confirma BUY
8. Se P(down) > 0.65 → Confirma SELL
9. Senão → Sem confirmação ML

PERFORMANCE:
├── Medir latência de inference
├── Target: < 5ms
├── Se > 5ms: Otimizar modelo ou cache

TESTES:
├── Criar Scripts/Test_COnnxBrain.mq5
├── Testar carregamento
├── Testar inference
├── Verificar que output faz sentido
"
```

---

#### Tarefa 3.C.2: Validar Modelo com Oracle

```
PROMPT PARA ORACLE:

"Oracle, valide o modelo ML:

DADOS:
├── Predictions do modelo em dados OOS (test set)
├── Labels reais

CALCULAR:

1. ACCURACY METRICS:
   ├── Accuracy global
   ├── Accuracy por classe (UP/DOWN)
   ├── Precision/Recall por classe
   └── F1 Score

2. CALIBRATION:
   ├── Calibration plot (predicted prob vs actual freq)
   ├── Brier score
   ├── Expected Calibration Error (ECE)
   └── Se P=0.7 previsto, ~70% devem ser corretos

3. TEMPORAL STABILITY:
   ├── Accuracy por mês
   ├── Detectar drift de performance
   └── Identificar períodos problemáticos

4. OVERFITTING CHECK:
   ├── Accuracy IS vs OOS por janela WF
   ├── Ratio IS/OOS < 1.3?
   └── WFE do modelo >= 0.5?

CRITÉRIOS:
├── Accuracy OOS > 55%
├── Brier score < 0.25
├── WFE modelo >= 0.5
├── IS/OOS ratio < 1.3

SALVAR EM: DOCS/04_REPORTS/ML/MODEL_VALIDATION.md

SE FALHAR: Retreinar com menos complexidade ou mais regularização
"
```

---

### 3.5 Checkpoint Fase 3

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CHECKPOINT FASE 3                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  □ Features engineered (15 features)                                       │
│  □ Target definido e balanceado                                            │
│  □ Modelo treinado com Walk-Forward                                        │
│  □ Accuracy OOS > 55%                                                      │
│  □ Brier score < 0.25 (bem calibrado)                                      │
│  □ WFE modelo >= 0.5                                                       │
│  □ IS/OOS ratio < 1.3 (sem overfit severo)                                 │
│                                                                             │
│  □ Modelo exportado para ONNX                                              │
│  □ Inference latência < 5ms                                                │
│  □ COnnxBrain.mqh atualizado                                               │
│  □ Teste de integração passou                                              │
│                                                                             │
│  □ Backtest COM ML executado                                               │
│  □ ML MELHORA métricas vs baseline                                         │
│  │   ├── PF com ML >= PF baseline                                          │
│  │   ├── Win Rate com ML >= Win Rate baseline                              │
│  │   └── DD com ML <= DD baseline                                          │
│                                                                             │
│  SE TODOS ✅ → Prosseguir para FASE 4                                      │
│  SE ML PIORA métricas → Desabilitar ML ou retreinar                        │
│  SE accuracy < 55% → Retreinar com features diferentes                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## FASE 4: SHADOW EXCHANGE VALIDATION 🆕

**Duração**: 3-4 dias
**Sessões**: 2 simultâneas
**Objetivo**: Validar sistema em simulador que emula a FÍSICA da infraestrutura

### 4.1 Conceito: Shadow Exchange

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SHADOW EXCHANGE CONCEPT                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  POR QUE MT5 TESTER NÃO É SUFICIENTE:                                      │
│  ├── Não simula latência de rede variável                                  │
│  ├── Não simula packet loss (TCP retransmission)                           │
│  ├── Não simula GC pauses do Python                                        │
│  ├── Não simula state desync entre Python e MQL5                           │
│  └── Spread/slippage são aproximações, não realistas                       │
│                                                                             │
│  SHADOW EXCHANGE:                                                           │
│  ├── Simulador 100% Python que EMULA a exchange                            │
│  ├── Processa tick por tick com latência realista                          │
│  ├── Injeta falhas de rede, GC pauses, requotes                            │
│  ├── Usa a MESMA lógica do EA (portada para Python)                        │
│  └── Se sobreviver aqui, sobrevive na FTMO                                 │
│                                                                             │
│  ARQUITETURA:                                                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │
│  │ TICK FEED   │───>│   STRATEGY  │───>│  EXCHANGE   │                     │
│  │ (L1 Data)   │    │ (EA Logic)  │    │  EMULATOR   │                     │
│  └─────────────┘    └─────────────┘    └─────────────┘                     │
│        │                  │                  │                              │
│        v                  v                  v                              │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │              LATENCY INJECTION ENGINE                        │           │
│  │  ├── Network: Gamma(2.0, 0.005) + base_ping                 │           │
│  │  ├── Packet Loss: Poisson(0.001) → +200ms                   │           │
│  │  ├── GC Pause: Random 10-50ms a cada 100 ticks              │           │
│  │  ├── News Multiplier: 3x durante eventos                    │           │
│  │  └── Volatility Drag: +10ms se vol > threshold              │           │
│  └─────────────────────────────────────────────────────────────┘           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Modelo de Latência Estocástica

```python
# MODELO DE LATÊNCIA (de ARGUS research)

def latency_model(
    base_ping_ms: float = 20,
    is_news: bool = False,
    volatility_percentile: float = 50
) -> float:
    '''
    Modelo de latência com 4 componentes:
    L_total = L_net + L_inf + L_proc + L_queue
    '''
    
    # 1. Network Jitter (Gamma distribution)
    # Assimétrica positiva: maioria ~10ms, mas picos de 100ms+
    L_net = base_ping_ms + np.random.gamma(2.0, 5.0)
    
    # 2. Inference Time (incluindo possível GC pause)
    L_inf = 5  # Base inference
    if np.random.random() < 0.05:  # 5% chance de GC pause
        L_inf += np.random.uniform(20, 80)
    
    # 3. Broker Processing (aumenta em vol alta e news)
    L_proc = 10
    if is_news:
        L_proc *= 3
    if volatility_percentile > 75:
        L_proc *= 1.5
    
    # 4. Packet Loss / Retransmission (Poisson process)
    # 0.1% dos pacotes se perdem → TCP retransmit → 200ms+ delay
    L_queue = 0
    if np.random.random() < 0.001:
        L_queue = np.random.uniform(200, 400)
    
    return L_net + L_inf + L_proc + L_queue
```

### 4.3 Sessão A: Implementar Shadow Exchange

#### Tarefa 4.A.1: Criar Exchange Emulator

```
PROMPT PARA FORGE:

"Forge, implemente o Shadow Exchange:

Salvar em: scripts/backtest/shadow_exchange.py

COMPONENTES:

1. CLASS ExchangeEmulator:
   '''
   Emula uma exchange que aceita/rejeita/executa ordens
   com comportamento realista.
   '''
   
   def __init__(self, config):
       self.spread_model = DynamicSpreadModel(config)
       self.slippage_model = DynamicSlippageModel(config)
       self.latency_model = LatencyModel(config)
       self.rejection_model = RejectionModel(config)
   
   def submit_order(self, order, market_state) -> ExecutionResult:
       # 1. Simular latência
       latency = self.latency_model.sample(market_state)
       
       # 2. Preço após latência (mercado se moveu)
       price_at_execution = self._price_after_latency(
           order.price, latency, market_state.velocity
       )
       
       # 3. Check rejeição
       if self.rejection_model.should_reject(order, market_state):
           return ExecutionResult(rejected=True, reason='requote')
       
       # 4. Calcular spread e slippage
       spread = self.spread_model.get_spread(market_state)
       slippage = self.slippage_model.get_slippage(order, market_state)
       
       # 5. Preço final de execução
       final_price = price_at_execution + spread/2 + slippage
       
       return ExecutionResult(
           filled=True,
           fill_price=final_price,
           latency_ms=latency,
           spread_paid=spread,
           slippage=slippage
       )

2. CLASS DynamicSpreadModel:
   - Spread base por sessão (Asia 1.5x, London 1.0x, etc.)
   - Multiplicador por volatilidade
   - Spike durante news (5x)
   - Random variance ±20%

3. CLASS DynamicSlippageModel:
   - Base slippage proporcional ao spread
   - Volatility drag
   - Size impact (sqrt law para orders grandes)
   - Sempre adverso (contra nós)

4. CLASS LatencyModel:
   - Gamma distribution para network
   - Poisson para packet loss
   - GC pause injection
   - News multiplier

5. CLASS RejectionModel:
   - Base rejection rate 2%
   - News: 15-30%
   - High volatility: 10%
   - Requote se preço moveu > threshold

CONFIGURAÇÕES PRÉ-DEFINIDAS:
├── OPTIMISTIC: Spread 0.8x, slippage 0.5x, latência 0.5x, rejection 1%
├── NORMAL: Spread 1.0x, slippage 1.0x, latência 1.0x, rejection 2%
├── PESSIMISTIC: Spread 1.5x, slippage 2.0x, latência 1.5x, rejection 5%
└── STRESS: Spread 3.0x, slippage 5.0x, latência 3.0x, rejection 15%
"
```

---

#### Tarefa 4.A.2: Portar Lógica do EA para Python

```
PROMPT PARA FORGE:

"Forge, porte a lógica essencial do EA para Python:

Salvar em: scripts/backtest/strategies/ea_logic_python.py

NÃO É NECESSÁRIO portar TUDO. Apenas:

1. CONFLUENCE SCORING:
   - Portar lógica de CConfluenceScorer
   - Pesos dos fatores
   - Threshold de execução

2. REGIME DETECTION:
   - Hurst Exponent calculation
   - Shannon Entropy
   - Classificação de regime

3. SESSION DETECTION:
   - Mapear hora para sessão
   - Filtros de sessão

4. SIGNAL GENERATION:
   - Condições de entrada
   - Direção do trade

5. RISK MANAGEMENT:
   - Position sizing
   - SL/TP calculation
   - Daily/Total DD check

A LÓGICA PYTHON DEVE PRODUZIR OS MESMOS SINAIS QUE O MQL5!

TESTE DE PARIDADE:
├── Rodar MQL5 em período X → gerar trades
├── Rodar Python no mesmo período → gerar trades
├── Comparar: devem ser ~95% iguais
├── Diferenças aceitáveis: timing de 1-2 ticks por latência
"
```

---

### 4.4 Sessão B: Executar Shadow Backtest (PARALELO)

#### Tarefa 4.B.1: Backtest Shadow vs MT5

```
PROMPT PARA FORGE:

"Forge, execute backtests comparativos:

PERÍODO: 2024-01 a 2024-06 (6 meses, dados OOS)

EXECUÇÕES:

1. MT5 Strategy Tester (baseline de referência):
   - Configuração padrão
   - Exportar trades para mt5_trades_2024h1.csv

2. Shadow Exchange NORMAL:
   python scripts/backtest/run_shadow.py \
     --data data/ticks/2024_h1.npz \
     --strategy ea_logic_python \
     --mode normal \
     --output shadow_normal_2024h1.csv

3. Shadow Exchange PESSIMISTIC:
   python scripts/backtest/run_shadow.py \
     --data data/ticks/2024_h1.npz \
     --strategy ea_logic_python \
     --mode pessimistic \
     --output shadow_pessimistic_2024h1.csv

4. Shadow Exchange STRESS:
   python scripts/backtest/run_shadow.py \
     --data data/ticks/2024_h1.npz \
     --strategy ea_logic_python \
     --mode stress \
     --output shadow_stress_2024h1.csv

COMPARAÇÃO:
| Métrica | MT5 | Shadow Normal | Shadow Pess | Shadow Stress |
|---------|-----|---------------|-------------|---------------|
| Trades  |     |               |             |               |
| PF      |     |               |             |               |
| Max DD  |     |               |             |               |
| Sharpe  |     |               |             |               |
"
```

---

#### Tarefa 4.B.2: Análise de Divergência

```
PROMPT PARA ORACLE:

"Oracle, analise a divergência entre MT5 e Shadow Exchange:

DADOS:
├── mt5_trades_2024h1.csv
├── shadow_normal_2024h1.csv
├── shadow_pessimistic_2024h1.csv
├── shadow_stress_2024h1.csv

ANÁLISE:

1. TRADE MATCHING:
   - Quantos trades coincidem (mesmo horário ±1min)?
   - Quantos trades apenas no MT5?
   - Quantos trades apenas no Shadow?

2. DIVERGÊNCIA DE MÉTRICAS:
   - Δ PF = PF_mt5 - PF_shadow
   - Δ DD = DD_shadow - DD_mt5
   - Δ Sharpe = Sharpe_mt5 - Sharpe_shadow

3. ANÁLISE DE CUSTOS:
   - Custos médios no Shadow
   - Quanto os custos impactaram o PF?
   - Latência média vs worst case

4. STRESS DEGRADATION:
   - Quanto o sistema degrada sob stress?
   - Ainda é lucrativo em STRESS mode?

THRESHOLDS:
├── Divergência MT5 vs Shadow Normal: < 15%
├── Shadow Normal ainda lucrativo: PF >= 1.2
├── Shadow Pessimistic ainda lucrativo: PF >= 1.0
├── Shadow Stress: Pode ser negativo, mas DD controlado

SE divergência > 15%:
├── Investigar causa
├── Pode ser: lógica diferente, timing, ou custos
├── Ajustar até convergir

SALVAR EM: DOCS/04_REPORTS/VALIDATION/SHADOW_DIVERGENCE_ANALYSIS.md
"
```

---

### 4.5 Checkpoint Fase 4

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CHECKPOINT FASE 4                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  □ Shadow Exchange implementado                                            │
│  │   ├── ExchangeEmulator                                                  │
│  │   ├── DynamicSpreadModel                                                │
│  │   ├── DynamicSlippageModel                                              │
│  │   ├── LatencyModel (Gamma + Poisson)                                    │
│  │   └── RejectionModel                                                    │
│                                                                             │
│  □ Lógica do EA portada para Python                                        │
│  □ Teste de paridade MQL5 vs Python: >= 95% trades iguais                  │
│                                                                             │
│  □ Backtest MT5 executado (referência)                                     │
│  □ Backtest Shadow NORMAL executado                                        │
│  □ Backtest Shadow PESSIMISTIC executado                                   │
│  □ Backtest Shadow STRESS executado                                        │
│                                                                             │
│  □ Divergência MT5 vs Shadow Normal < 15%                                  │
│  □ Shadow Normal: PF >= 1.2                                                │
│  □ Shadow Pessimistic: PF >= 1.0                                           │
│  □ Shadow Stress: DD <= 15%                                                │
│                                                                             │
│  □ SHADOW_DIVERGENCE_ANALYSIS.md gerado                                    │
│                                                                             │
│  SE TODOS ✅ → Prosseguir para FASE 5                                      │
│  SE divergência > 15% → Investigar e corrigir                              │
│  SE Shadow Pessimistic PF < 1.0 → Estratégia muito sensível a custos      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## FASE 5: VALIDAÇÃO ESTATÍSTICA INSTITUCIONAL 🔄

**Duração**: 3-4 dias
**Sessões**: 3 simultâneas
**Objetivo**: Validação com rigor estatístico de nível institucional

### 5.1 Oracle Pipeline v2.0

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   ORACLE VALIDATION PIPELINE v2.0                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ETAPA 1: SEGMENTAÇÃO (preparação)                                         │
│  ├── Segmentar trades por regime (Hurst)                                   │
│  ├── Segmentar trades por sessão                                           │
│  └── Criar datasets para cada análise                                      │
│                                                                             │
│  ETAPA 2: WFA MULTI-DIMENSIONAL                                            │
│  ├── WFA Global (todos os trades)                                          │
│  ├── WFA por Regime (trending, ranging, reverting)                         │
│  ├── WFA por Sessão (asia, london, ny, etc.)                               │
│  └── Critério: WFE >= 0.60 global, >= 0.50 por segmento                    │
│                                                                             │
│  ETAPA 3: MONTE CARLO MULTI-CENÁRIO                                        │
│  ├── MC Baseline (custos normais, 10000 runs)                              │
│  ├── MC Pessimistic (custos 2x, 10000 runs)                                │
│  ├── MC Stress (custos 5x, news storms, 5000 runs)                         │
│  └── Critério: 95th DD < 8% baseline, < 12% stress                         │
│                                                                             │
│  ETAPA 4: DETECÇÃO DE OVERFITTING                                          │
│  ├── PSR (Probabilistic Sharpe Ratio) >= 0.90                              │
│  ├── DSR (Deflated Sharpe) > 0                                             │
│  ├── PBO (Probability Backtest Overfit) < 0.50 via CPCV                    │
│  ├── MinTRL (Minimum Track Record) vs trades disponíveis                   │
│  └── Critério: Todos devem passar                                          │
│                                                                             │
│  ETAPA 5: VALIDAÇÃO FTMO ESPECÍFICA                                        │
│  ├── P(Daily DD > 5%) calculado                                            │
│  ├── P(Total DD > 10%) calculado                                           │
│  ├── Trailing DD simulation                                                │
│  ├── Profit target viability                                               │
│  └── Critério: P(daily) < 5%, P(total) < 2%                               │
│                                                                             │
│  ETAPA 6: CONFIDENCE SCORE                                                 │
│  ├── Agregar todas as métricas                                             │
│  ├── Calcular score 0-100                                                  │
│  ├── Identificar weak points                                               │
│  └── Emitir decisão GO/NO-GO                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Sessão A: WFA Multi-Dimensional

#### Tarefa 5.A.1: Implementar WFA por Segmento

```
PROMPT PARA FORGE:

"Forge, estenda o walk_forward.py para análise por segmento:

Salvar em: scripts/oracle/walk_forward_segmented.py

FUNCIONALIDADES:

1. WFA GLOBAL:
   - Como já implementado
   - 12-15 janelas rolling
   - IS 70%, OOS 30%

2. WFA POR REGIME:
   - Filtrar trades por regime antes de WFA
   - Calcular WFE separado para:
     ├── TRENDING (Hurst > 0.55)
     ├── RANDOM (0.45-0.55) - deve ter ~0 trades
     └── REVERTING (Hurst < 0.45)

3. WFA POR SESSÃO:
   - Filtrar trades por sessão
   - Calcular WFE para:
     ├── ASIA
     ├── LONDON
     ├── OVERLAP
     └── NEW_YORK

4. WFA CRUZADO:
   - Treinar em regime X, testar em regime Y
   - Detectar se estratégia generaliza

5. OUTPUT:
   - Tabela completa de WFE por segmento
   - Identificação de segmentos fracos
   - Recomendação de ajuste

USO:
python scripts/oracle/walk_forward_segmented.py \
  --input shadow_normal_trades.csv \
  --segments regime,session \
  --output DOCS/04_REPORTS/VALIDATION/WFA_SEGMENTED.md
"
```

---

#### Tarefa 5.A.2: Executar WFA Completo

```
PROMPT PARA ORACLE:

"Oracle, execute WFA completo com análise por segmento:

DADOS: shadow_normal_trades.csv (output da Fase 4)

CONFIGURAÇÃO:
├── Janelas: 12 rolling
├── IS/OOS: 70/30
├── Purge gap: 5 trades
├── Min trades por janela: 10

EXECUTAR:

1. WFA Global:
   python -m scripts.oracle.walk_forward_segmented \
     --input shadow_normal_trades.csv \
     --mode global

2. WFA por Regime:
   python -m scripts.oracle.walk_forward_segmented \
     --input shadow_normal_trades.csv \
     --mode regime

3. WFA por Sessão:
   python -m scripts.oracle.walk_forward_segmented \
     --input shadow_normal_trades.csv \
     --mode session

TABELA DE RESULTADOS:
| Segmento | Windows | WFE | OOS+ % | Status |
|----------|---------|-----|--------|--------|
| GLOBAL   |         |     |        |        |
| TRENDING |         |     |        |        |
| REVERTING|         |     |        |        |
| ASIA     |         |     |        |        |
| LONDON   |         |     |        |        |
| OVERLAP  |         |     |        |        |
| NY       |         |     |        |        |

CRITÉRIOS:
├── Global WFE >= 0.60: PASS
├── Trending WFE >= 0.65: PASS
├── Reverting WFE >= 0.45: PASS (ou não operar)
├── Nenhuma sessão com WFE < 0.40: PASS
"
```

---

### 5.3 Sessão B: Monte Carlo Multi-Cenário (PARALELO)

#### Tarefa 5.B.1: Implementar MC Multi-Cenário

```
PROMPT PARA FORGE:

"Forge, estenda monte_carlo.py para multi-cenário:

Salvar em: scripts/oracle/monte_carlo_scenarios.py

CENÁRIOS:

1. BASELINE:
   - Custos normais (como no backtest)
   - 10,000 simulações
   - Block size automático

2. PESSIMISTIC:
   - Spread 2x
   - Slippage 2x
   - Aplicar penalidade de custo a cada trade
   - 10,000 simulações

3. STRESS:
   - Spread 5x
   - Slippage 5x
   - Injetar 5 "news events" com DD spike
   - 5,000 simulações

4. NEWS_STORM:
   - Simular 5 events de alto impacto consecutivos
   - Cada event: +1-3% DD instantâneo
   - Verificar se circuit breaker aguenta

5. FLASH_CRASH:
   - Injetar um evento de 5%+ DD instantâneo
   - Verificar recovery

PARA CADA CENÁRIO, CALCULAR:
├── DD distribution (5th, 50th, 95th, 99th)
├── VaR 95%
├── CVaR 95% (Expected Shortfall)
├── P(DD > 5%)
├── P(DD > 10%)
├── Confidence Score parcial

OUTPUT:
├── Tabela comparativa de cenários
├── Distribuições de DD por cenário
├── Recomendação de position size se stress falhar
"
```

---

#### Tarefa 5.B.2: Executar MC Multi-Cenário

```
PROMPT PARA ORACLE:

"Oracle, execute Monte Carlo multi-cenário:

DADOS: shadow_normal_trades.csv

CONFIGURAÇÃO:
├── Capital: $100,000
├── Block size: Auto (preservar autocorrelação)

EXECUTAR:

1. Baseline (10k runs):
   python -m scripts.oracle.monte_carlo_scenarios \
     --input shadow_normal_trades.csv \
     --scenario baseline \
     --simulations 10000

2. Pessimistic (10k runs):
   python -m scripts.oracle.monte_carlo_scenarios \
     --input shadow_normal_trades.csv \
     --scenario pessimistic \
     --simulations 10000

3. Stress (5k runs):
   python -m scripts.oracle.monte_carlo_scenarios \
     --input shadow_normal_trades.csv \
     --scenario stress \
     --simulations 5000

TABELA DE RESULTADOS:
| Cenário | 95th DD | VaR 95% | CVaR 95% | P(>5%) | P(>10%) |
|---------|---------|---------|----------|--------|---------|
| Baseline|         |         |          |        |         |
| Pessim  |         |         |          |        |         |
| Stress  |         |         |          |        |         |

CRITÉRIOS:
├── Baseline 95th DD < 8%: PASS
├── Pessimistic 95th DD < 10%: PASS
├── Stress 95th DD < 12%: PASS (alguma margem)
├── Baseline P(>10%) < 2%: PASS
├── Pessimistic P(>10%) < 5%: PASS

SALVAR EM: DOCS/04_REPORTS/VALIDATION/MC_SCENARIOS.md
"
```

---

### 5.4 Sessão C: Detecção de Overfitting (PARALELO)

#### Tarefa 5.C.1: Implementar CPCV para PBO

```
PROMPT PARA FORGE:

"Forge, implemente Combinatorial Purged CV para calcular PBO:

Salvar em: scripts/oracle/cpcv.py

CONCEITO:

CPCV gera TODOS os caminhos possíveis de IS/OOS, não apenas sequenciais.
Permite calcular PBO (Probability of Backtest Overfitting) de forma robusta.

IMPLEMENTAÇÃO:

class CPCV:
    '''
    Combinatorial Purged Cross-Validation
    Bailey et al. (2014)
    '''
    
    def __init__(self, n_splits=6, n_test_splits=2, purge_gap=0):
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_gap = purge_gap
    
    def split(self, X):
        '''
        Gera todas as combinações C(n_splits, n_test_splits)
        '''
        from itertools import combinations
        
        n = len(X)
        fold_size = n // self.n_splits
        
        for test_folds in combinations(range(self.n_splits), self.n_test_splits):
            test_idx = []
            for fold in test_folds:
                start = fold * fold_size
                end = start + fold_size
                test_idx.extend(range(start, end))
            
            # Train = tudo menos test (com purge)
            train_idx = [i for i in range(n) if i not in test_idx]
            
            # Aplicar purge
            train_idx = self._apply_purge(train_idx, test_idx)
            
            yield train_idx, test_idx
    
    def calculate_pbo(self, is_performance, oos_performance):
        '''
        PBO = proporção de combinações onde melhor IS != melhor OOS
        '''
        # Rank correlation
        correlation = spearmanr(is_performance, oos_performance)[0]
        
        # PBO baseado em correlação
        pbo = (1 - correlation) / 2
        
        return pbo

USO:
from scripts.oracle.cpcv import CPCV

cpcv = CPCV(n_splits=6, n_test_splits=2)
is_perfs, oos_perfs = [], []

for train_idx, test_idx in cpcv.split(trades):
    is_perf = calculate_sharpe(trades.iloc[train_idx])
    oos_perf = calculate_sharpe(trades.iloc[test_idx])
    is_perfs.append(is_perf)
    oos_perfs.append(oos_perf)

pbo = cpcv.calculate_pbo(is_perfs, oos_perfs)
"
```

---

#### Tarefa 5.C.2: Executar Análise Completa de Overfitting

```
PROMPT PARA ORACLE:

"Oracle, execute análise completa de overfitting:

DADOS: shadow_normal_trades.csv
N_TRIALS: 10 (número de combinações de parâmetros testadas)

CALCULAR:

1. PSR (Probabilistic Sharpe Ratio):
   python -m scripts.oracle.deflated_sharpe \
     --input shadow_normal_trades.csv \
     --metric psr

2. DSR (Deflated Sharpe Ratio):
   python -m scripts.oracle.deflated_sharpe \
     --input shadow_normal_trades.csv \
     --metric dsr \
     --trials 10

3. PBO (via CPCV):
   python -m scripts.oracle.cpcv \
     --input shadow_normal_trades.csv

4. MinTRL (Minimum Track Record Length):
   - Quantos trades precisamos para 95% de confiança?
   - Temos trades suficientes?

TABELA DE RESULTADOS:
| Métrica | Valor | Threshold | Status |
|---------|-------|-----------|--------|
| PSR     |       | >= 0.90   |        |
| DSR     |       | > 0       |        |
| PBO     |       | < 0.50    |        |
| MinTRL  |       | <= trades |        |

INTERPRETAÇÃO:
├── PSR >= 0.90: Sharpe provavelmente real
├── DSR > 0: Sharpe sobrevive deflation por N trials
├── PBO < 0.50: Baixo risco de overfit
├── trades >= MinTRL: Track record suficiente

SALVAR EM: DOCS/04_REPORTS/VALIDATION/OVERFITTING_ANALYSIS.md
"
```

---

### 5.5 Sessão D: GO/NO-GO Aggregation

#### Tarefa 5.D.1: Criar Pipeline Agregador

```
PROMPT PARA FORGE:

"Forge, crie o pipeline agregador GO/NO-GO v2.0:

Salvar em: scripts/oracle/go_nogo_v2.py

FUNCIONALIDADES:

1. CARREGAR TODOS OS RESULTADOS:
   - WFA (global e segmentado)
   - Monte Carlo (todos os cenários)
   - Overfitting (PSR, DSR, PBO)
   - FTMO specific

2. CALCULAR CONFIDENCE SCORE:

   def calculate_confidence_score(results):
       score = 0
       breakdown = {}
       
       # WFA (25 pontos)
       wfe = results['wfa']['global_wfe']
       if wfe >= 0.70: score += 25
       elif wfe >= 0.60: score += 20
       elif wfe >= 0.50: score += 10
       breakdown['wfa'] = ...
       
       # Monte Carlo Baseline (25 pontos)
       dd_95 = results['mc_baseline']['dd_95th']
       if dd_95 < 6: score += 25
       elif dd_95 < 8: score += 20
       elif dd_95 < 10: score += 10
       breakdown['mc_baseline'] = ...
       
       # Monte Carlo Stress (15 pontos)
       dd_95_stress = results['mc_stress']['dd_95th']
       if dd_95_stress < 10: score += 15
       elif dd_95_stress < 12: score += 10
       breakdown['mc_stress'] = ...
       
       # Overfitting (20 pontos)
       if results['psr'] >= 0.90: score += 7
       if results['dsr'] > 0: score += 7
       if results['pbo'] < 0.50: score += 6
       breakdown['overfitting'] = ...
       
       # FTMO Specific (15 pontos)
       if results['ftmo']['p_daily_breach'] < 5: score += 8
       if results['ftmo']['p_total_breach'] < 2: score += 7
       breakdown['ftmo'] = ...
       
       return score, breakdown

3. EMITIR DECISÃO:
   - STRONG_GO: score >= 85 + zero falhas críticas
   - GO: score >= 75
   - CAUTIOUS: score 65-74
   - NO_GO: score < 65 ou qualquer falha crítica

4. IDENTIFICAR WEAK POINTS:
   - Listar métricas que não passaram
   - Sugerir ações de correção

5. GERAR RELATÓRIO COMPLETO:
   - Markdown formatado
   - Todas as tabelas
   - Gráficos de distribuição
   - Decisão final com justificativa
"
```

---

#### Tarefa 5.D.2: Executar GO/NO-GO Final

```
PROMPT PARA ORACLE:

"Oracle, execute o pipeline GO/NO-GO v2.0 completo:

python -m scripts.oracle.go_nogo_v2 \
  --trades shadow_normal_trades.csv \
  --output DOCS/04_REPORTS/DECISIONS/GO_NOGO_REPORT_v2.md

O RELATÓRIO DEVE INCLUIR:

1. EXECUTIVE SUMMARY:
   - Decisão: STRONG_GO / GO / CAUTIOUS / NO_GO
   - Confidence Score: X/100
   - Data da análise

2. BREAKDOWN DE SCORE:
   | Categoria | Pontos | Máximo | Status |
   |-----------|--------|--------|--------|
   | WFA       |        | 25     |        |
   | MC Base   |        | 25     |        |
   | MC Stress |        | 15     |        |
   | Overfit   |        | 20     |        |
   | FTMO      |        | 15     |        |
   | TOTAL     |        | 100    |        |

3. DETALHES POR CATEGORIA:
   - Todas as métricas calculadas
   - Comparação com thresholds
   - Gráficos relevantes

4. WEAK POINTS:
   - Lista de métricas que não passaram
   - Impacto de cada weak point
   - Ações sugeridas

5. POSITION SIZE RECOMMENDATION:
   - Se confidence < 85: Reduzir risk
   - Fórmula: risk_adjusted = risk_base * (confidence / 100)

6. NEXT STEPS:
   - Se GO: Prosseguir para Stress Testing
   - Se CAUTIOUS: Quais ajustes fazer
   - Se NO_GO: O que precisa mudar
"
```

---

### 5.6 Checkpoint Fase 5

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CHECKPOINT FASE 5                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  WFA MULTI-DIMENSIONAL:                                                    │
│  □ WFA Global executado (WFE >= 0.60)                                      │
│  □ WFA Trending executado (WFE >= 0.65)                                    │
│  □ WFA Reverting executado (WFE >= 0.45)                                   │
│  □ WFA por sessão executado (nenhuma < 0.40)                               │
│  □ WFA_SEGMENTED.md gerado                                                 │
│                                                                             │
│  MONTE CARLO MULTI-CENÁRIO:                                                │
│  □ MC Baseline: 95th DD < 8%                                               │
│  □ MC Pessimistic: 95th DD < 10%                                           │
│  □ MC Stress: 95th DD < 12%                                                │
│  □ MC Baseline: P(>10%) < 2%                                               │
│  □ MC_SCENARIOS.md gerado                                                  │
│                                                                             │
│  DETECÇÃO DE OVERFITTING:                                                  │
│  □ PSR >= 0.90                                                             │
│  □ DSR > 0                                                                 │
│  □ PBO < 0.50                                                              │
│  □ trades >= MinTRL                                                        │
│  □ OVERFITTING_ANALYSIS.md gerado                                          │
│                                                                             │
│  GO/NO-GO:                                                                 │
│  □ Confidence Score calculado                                              │
│  □ Decisão emitida                                                         │
│  □ GO_NOGO_REPORT_v2.md gerado                                             │
│                                                                             │
│  SE Confidence >= 75 (GO) → Prosseguir para FASE 6                         │
│  SE Confidence 65-74 (CAUTIOUS) → Revisar weak points                      │
│  SE Confidence < 65 (NO_GO) → Voltar para Fase 2/3 e ajustar               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## FASE 6: STRESS TESTING EXTREMO 🆕

**Duração**: 2-3 dias
**Sessões**: 2 simultâneas
**Objetivo**: Validar que o sistema sobrevive a cenários extremos

### 6.1 Catálogo de Stress Tests

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CATÁLOGO DE STRESS TESTS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TEST 1: NEWS STORM                                                        │
│  ├── Descrição: 5 eventos de alto impacto consecutivos                     │
│  ├── Simulação: NFP + CPI + FOMC + ECB + BOE em 2 semanas                 │
│  ├── Impacto: Spread 5x, latência 3x, rejection 30%                        │
│  └── Critério: Sistema ainda lucrativo OU DD < 8%                          │
│                                                                             │
│  TEST 2: FLASH CRASH                                                       │
│  ├── Descrição: Movimento de 3%+ em < 5 minutos                            │
│  ├── Simulação: Injetar gap de 3% contra posição aberta                    │
│  ├── Impacto: SL pode ser saltado, slippage 100+ pips                      │
│  └── Critério: DD do evento < 5%, recovery em < 2 semanas                  │
│                                                                             │
│  TEST 3: CONNECTION LOSS                                                   │
│  ├── Descrição: Perda de conexão por 30s a 5 minutos                       │
│  ├── Simulação: Ordem enviada mas não confirmada                           │
│  ├── Impacto: Estado desync entre local e broker                           │
│  └── Critério: Sistema detecta e reconcilia, DD adicional < 1%             │
│                                                                             │
│  TEST 4: REGIME TRANSITION RAPID                                           │
│  ├── Descrição: 3+ mudanças de regime em 1 dia                             │
│  ├── Simulação: Trending → Random → Reverting → Trending                   │
│  ├── Impacto: Sinais conflitantes, whipsaws                                │
│  └── Critério: Sistema reduz exposição, DD < 3% no dia                     │
│                                                                             │
│  TEST 5: LIQUIDITY DRY-UP                                                  │
│  ├── Descrição: Spread 10x por 1 hora                                      │
│  ├── Simulação: Sessão asiática quieta + feriado                           │
│  ├── Impacto: Custos de execução proibitivos                               │
│  └── Critério: Sistema não opera OU aceita spread alto                     │
│                                                                             │
│  TEST 6: CIRCUIT BREAKER STRESS                                            │
│  ├── Descrição: Testar limites do circuit breaker                          │
│  ├── Simulação: Sequência de 5 losses que aproxima do limite               │
│  ├── Impacto: CB deve ativar e pausar trading                              │
│  └── Critério: CB ativa ANTES de violar, nunca depois                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Sessão A: Implementar Stress Tests

#### Tarefa 6.A.1: Criar Framework de Stress Testing

```
PROMPT PARA FORGE:

"Forge, crie framework de stress testing:

Salvar em: scripts/stress/stress_framework.py

ARQUITETURA:

class StressTest:
    '''Base class para stress tests'''
    
    def __init__(self, name, description):
        self.name = name
        self.description = description
    
    def inject(self, backtest_engine):
        '''Injeta o cenário de stress no backtester'''
        raise NotImplementedError
    
    def evaluate(self, results) -> StressResult:
        '''Avalia se passou no teste'''
        raise NotImplementedError

class NewsStormTest(StressTest):
    '''5 eventos de alto impacto consecutivos'''
    
    def __init__(self):
        super().__init__('NEWS_STORM', '5 high-impact events in 2 weeks')
        self.events = [
            {'name': 'NFP', 'spread_mult': 5, 'latency_mult': 3, 'reject_rate': 0.30},
            {'name': 'CPI', 'spread_mult': 4, 'latency_mult': 2, 'reject_rate': 0.25},
            {'name': 'FOMC', 'spread_mult': 6, 'latency_mult': 4, 'reject_rate': 0.35},
            {'name': 'ECB', 'spread_mult': 3, 'latency_mult': 2, 'reject_rate': 0.20},
            {'name': 'BOE', 'spread_mult': 3, 'latency_mult': 2, 'reject_rate': 0.20},
        ]
    
    def inject(self, engine):
        for event in self.events:
            engine.schedule_news_event(event)
    
    def evaluate(self, results) -> StressResult:
        max_dd = results.max_drawdown
        profitable = results.net_profit > 0
        
        passed = profitable or max_dd < 8.0
        return StressResult(
            test_name=self.name,
            passed=passed,
            max_dd=max_dd,
            details=f'Profitable: {profitable}, Max DD: {max_dd:.2f}%'
        )

class FlashCrashTest(StressTest):
    '''Movimento de 3%+ em < 5 minutos'''
    # ... implementar

class ConnectionLossTest(StressTest):
    '''Perda de conexão por 30s-5min'''
    # ... implementar

# Etc para outros tests

STRESS TEST RUNNER:

class StressTestRunner:
    def __init__(self, backtest_config):
        self.config = backtest_config
        self.tests = [
            NewsStormTest(),
            FlashCrashTest(),
            ConnectionLossTest(),
            RegimeTransitionTest(),
            LiquidityDryupTest(),
            CircuitBreakerTest(),
        ]
    
    def run_all(self, trades_df) -> StressReport:
        results = []
        for test in self.tests:
            result = self._run_single(test, trades_df)
            results.append(result)
        
        return StressReport(results)
"
```

---

#### Tarefa 6.A.2: Implementar Cada Stress Test

```
PROMPT PARA FORGE:

"Forge, implemente cada stress test individualmente:

1. NEWS_STORM (scripts/stress/news_storm.py):
   - Injetar 5 eventos em 2 semanas simuladas
   - Cada evento: 4 horas de condições adversas
   - Spread 3-6x, latência 2-4x, rejection 20-35%

2. FLASH_CRASH (scripts/stress/flash_crash.py):
   - Injetar gap de 3% contra posição aberta
   - SL deve ser saltado (gap > SL distance)
   - Calcular slippage real
   - Verificar DD máximo

3. CONNECTION_LOSS (scripts/stress/connection_loss.py):
   - Simular ordem enviada mas não confirmada
   - Mercado se move 50 pips durante desconexão
   - Sistema deve detectar e reconciliar

4. REGIME_TRANSITION (scripts/stress/regime_transition.py):
   - 3 mudanças de regime em 1 dia
   - Injetar sinais conflitantes
   - Verificar se sistema reduz exposição

5. LIQUIDITY_DRYUP (scripts/stress/liquidity_dryup.py):
   - Spread 10x por 1 hora
   - Sistema deve parar de operar ou aceitar custos

6. CIRCUIT_BREAKER (scripts/stress/circuit_breaker.py):
   - Sequência de 5 losses crescentes
   - CB deve ativar em 4% DD (não 5%)
   - Verificar que não viola limite

CADA TESTE DEVE:
├── Ter critérios claros de PASS/FAIL
├── Gerar log detalhado do que aconteceu
├── Calcular métricas relevantes
└── Produzir recomendação se falhar
"
```

---

### 6.3 Sessão B: Executar Stress Tests (PARALELO)

#### Tarefa 6.B.1: Rodar Todos os Stress Tests

```
PROMPT PARA FORGE:

"Forge, execute todos os stress tests:

python scripts/stress/run_all_stress.py \
  --trades shadow_normal_trades.csv \
  --config stress_config.json \
  --output DOCS/04_REPORTS/VALIDATION/STRESS_TEST_REPORT.md

O RELATÓRIO DEVE INCLUIR:

1. SUMMARY TABLE:
   | Test | Status | Max DD | Details |
   |------|--------|--------|---------|
   | NEWS_STORM | PASS/FAIL | X.X% | ... |
   | FLASH_CRASH | PASS/FAIL | X.X% | ... |
   | CONNECTION_LOSS | PASS/FAIL | X.X% | ... |
   | REGIME_TRANSITION | PASS/FAIL | X.X% | ... |
   | LIQUIDITY_DRYUP | PASS/FAIL | X.X% | ... |
   | CIRCUIT_BREAKER | PASS/FAIL | X.X% | ... |

2. DETAILED ANALYSIS PER TEST:
   - O que foi simulado
   - Como o sistema reagiu
   - Métricas calculadas
   - Por que passou/falhou

3. OVERALL VERDICT:
   - Todos PASS: Sistema robusto
   - 1-2 FAIL não-críticos: Investigar
   - Qualquer FAIL crítico: NO-GO

4. RECOMMENDATIONS:
   - Ajustes necessários
   - Parâmetros a modificar
   - Cenários a evitar
"
```

---

#### Tarefa 6.B.2: Análise de Stress por SENTINEL

```
PROMPT PARA SENTINEL:

"Sentinel, analise os resultados dos stress tests do ponto de vista de risco:

DADOS: STRESS_TEST_REPORT.md

ANÁLISE REQUERIDA:

1. CIRCUIT BREAKER EFFECTIVENESS:
   - CB ativou no momento certo?
   - Margem de segurança adequada?
   - Recovery time após CB?

2. WORST CASE DD:
   - Qual o pior DD em todos os testes?
   - Este DD viola limites FTMO?
   - Probabilidade de ocorrer em live?

3. POSITION SIZING ADEQUACY:
   - Com DD de stress, position size está ok?
   - Recomendação de ajuste se necessário

4. RECOVERY ANALYSIS:
   - Tempo médio de recovery por stress
   - Alguns stresses são irrecuperáveis?

5. FTMO SURVIVAL PROBABILITY:
   - Dado os stress tests, qual P(passar FTMO)?
   - Quais cenários matam o challenge?

SALVAR EM: DOCS/04_REPORTS/DECISIONS/STRESS_RISK_ANALYSIS.md
"
```

---

### 6.4 Checkpoint Fase 6

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CHECKPOINT FASE 6                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  □ Framework de stress testing implementado                                │
│  □ 6 stress tests implementados                                            │
│                                                                             │
│  RESULTADOS DOS TESTES:                                                    │
│  □ NEWS_STORM: PASS (DD < 8% ou lucrativo)                                 │
│  □ FLASH_CRASH: PASS (DD < 5%, recovery < 2 sem)                           │
│  □ CONNECTION_LOSS: PASS (reconcilia, DD adicional < 1%)                   │
│  □ REGIME_TRANSITION: PASS (DD < 3% no dia)                                │
│  □ LIQUIDITY_DRYUP: PASS (não opera ou aceita)                             │
│  □ CIRCUIT_BREAKER: PASS (ativa antes de violar)                           │
│                                                                             │
│  □ STRESS_TEST_REPORT.md gerado                                            │
│  □ STRESS_RISK_ANALYSIS.md gerado                                          │
│                                                                             │
│  SE TODOS PASS → Prosseguir para FASE 7                                    │
│  SE 1-2 FAIL não-críticos → Investigar e decidir                           │
│  SE qualquer FAIL crítico → Corrigir antes de continuar                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## FASE 7: DEMO TRADING

**Duração**: 2+ semanas
**Sessões**: 3 simultâneas (monitoramento)
**Objetivo**: Validar em tempo real antes de arriscar dinheiro

### 7.1 Setup

```
PROMPT PARA FORGE:

"Forge, configure o EA para demo trading:

1. Abrir conta demo FTMO ou broker
   - Capital: $100,000
   - Alavancagem: 1:100
   - Servidor: Demo

2. Instalar EA no MT5:
   - Copiar EA_SCALPER_XAUUSD.ex5 para Experts/
   - Copiar Models/ para pasta correta
   - Configurar inputs conforme backtest

3. Parâmetros de produção:
   - InpRiskPerTrade = 0.5% (ou ajustado por confidence)
   - InpUseONNX = true
   - InpUseMTF = true
   - Todos os filtros ativos

4. Ativar AutoTrading
5. Verificar que EA carregou modelo ONNX
6. Verificar conexão com Python Hub (se aplicável)

Me confirme quando estiver rodando."
```

### 7.2 Monitoramento Diário

```
ROTINA DIÁRIA:

MANHÃ (antes de London):
├── CRUCIBLE: Análise de mercado, news check
├── SENTINEL: DD status, posições abertas
└── Verificar: EA rodando? Erros no log?

DURANTE SESSÃO:
├── Monitorar EA via MT5 mobile
├── Alertas de DD configurados
└── Não interferir (deixar sistema operar)

FIM DO DIA:
├── SENTINEL: Relatório de DD final
├── ORACLE: Análise dos trades do dia
└── Log: Documentar observações

SEMANAL:
├── ORACLE: Performance vs backtest
├── CRUCIBLE: Mercado correspondeu ao esperado?
└── Decisão: Continuar normal ou ajustar
```

### 7.3 Critérios de Validação Demo

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CRITÉRIOS DE VALIDAÇÃO DEMO                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  APÓS 2 SEMANAS, VERIFICAR:                                                │
│                                                                             │
│  TÉCNICO:                                                                  │
│  □ EA rodou sem crashes                                                    │
│  □ Sem erros críticos no log                                               │
│  □ ONNX inference funcionando                                              │
│  □ Python Hub estável (se usado)                                           │
│                                                                             │
│  EXECUÇÃO:                                                                 │
│  □ Trades executados corretamente                                          │
│  □ SL/TP funcionando                                                       │
│  □ Slippage real <= backtest + 5 pips                                      │
│  □ Spread real dentro do esperado                                          │
│                                                                             │
│  PERFORMANCE:                                                              │
│  □ DD nunca excedeu 4% (buffer FTMO)                                       │
│  □ Performance dentro de ±30% do backtest                                  │
│  □ Win rate dentro de ±10% do backtest                                     │
│  │                                                                          │
│  │  SE DIVERGÊNCIA > 30%:                                                  │
│  │  ├── Investigar causa                                                   │
│  │  ├── Pode ser: regime diferente, execução, ou bug                       │
│  │  └── NÃO prosseguir até entender                                        │
│                                                                             │
│  RISCO:                                                                    │
│  □ Circuit breaker não ativou (mercado normal)                             │
│  □ Position sizing correto                                                 │
│  □ Filtros de sessão/regime funcionando                                    │
│                                                                             │
│  SE TODOS ✅ → Prosseguir para FASE 8 (FTMO)                               │
│  SE ALGUM ❌ → Investigar e corrigir antes                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## FASE 8: CHALLENGE FTMO

**Duração**: 4+ semanas (Phase 1 + Phase 2)
**Objetivo**: PASSAR O CHALLENGE!

### 8.1 Regras FTMO ($100k)

```
MEMORIZAR:

PHASE 1 (30 dias):
├── Profit Target: 10% ($10,000)
├── Max Daily Loss: 5% ($5,000)
├── Max Total Loss: 10% ($10,000)
├── Min Trading Days: 4
└── VIOLAÇÃO DE DD = CONTA TERMINADA IMEDIATAMENTE

PHASE 2 (60 dias):
├── Profit Target: 5% ($5,000)
├── Max Daily Loss: 5% ($5,000)
├── Max Total Loss: 10% ($10,000)
├── Min Trading Days: 4
└── MAIS TEMPO, MESMO RISCO

BUFFERS DE SEGURANÇA (nosso sistema):
├── Soft stop Daily: 4% (não 5%)
├── Soft stop Total: 8% (não 10%)
├── Hard stop: CB fecha tudo automaticamente
└── NUNCA operar após CB ativar no mesmo dia
```

### 8.2 Contingências

```
SE DD DIÁRIO > 3%:
├── Reduzir InpRiskPerTrade para 0.25%
├── Monitorar mais de perto
└── Considerar pausar por 1 dia

SE DD DIÁRIO > 4%:
├── PAUSAR EA imediatamente
├── Analisar o que aconteceu
└── Só retomar no dia seguinte

SE DD TOTAL > 6%:
├── Modo ultra-conservador
├── Risk 0.25% max
└── Apenas Tier A setups

SE DD TOTAL > 8%:
├── PARAR completamente
├── Aceitar a perda do challenge
├── Analisar, aprender, tentar novamente depois
└── NUNCA arriscar os últimos 2%
```

---

## CHECKLIST GERAL

### Fase 0: Audit ✅ COMPLETA
- [x] EA compila sem erros
- [x] Todos os módulos auditados (score médio 19.5/20)
- [x] Bugs documentados no BUGFIX_LOG.md

### Fase 1: Validação de Dados ⬜
- [ ] Script validate_data.py criado
- [ ] Tick data validado (score >= 90)
- [ ] Dados convertidos para NPZ/Parquet
- [ ] Best practices pesquisadas

### Fase 2: Backtest Multi-Regime ⬜
- [ ] Dados segmentados por regime/sessão
- [ ] EventBacktester implementado
- [ ] Backtest por regime executado
- [ ] Backtest por sessão executado
- [ ] PF Global >= 1.3
- [ ] MULTI_REGIME_ANALYSIS.md gerado

### Fase 3: Treinamento ML ⬜
- [ ] Features engineered
- [ ] Modelo treinado (WF training)
- [ ] Accuracy OOS > 55%
- [ ] ONNX exportado e testado
- [ ] ML melhora métricas

### Fase 4: Shadow Exchange 🆕 ⬜
- [ ] Shadow Exchange implementado
- [ ] Lógica do EA portada para Python
- [ ] Divergência MT5 vs Shadow < 15%
- [ ] Shadow Pessimistic PF >= 1.0

### Fase 5: Validação Estatística 🔄 ⬜
- [ ] WFA global WFE >= 0.60
- [ ] WFA por regime/sessão completo
- [ ] MC Baseline 95th DD < 8%
- [ ] MC Stress 95th DD < 12%
- [ ] PSR >= 0.90, DSR > 0, PBO < 0.50
- [ ] Confidence Score >= 75
- [ ] GO/NO-GO = GO

### Fase 6: Stress Testing 🆕 ⬜
- [ ] 6 stress tests implementados
- [ ] Todos os stress tests PASS
- [ ] STRESS_TEST_REPORT.md gerado

### Fase 7: Demo Trading ⬜
- [ ] 2+ semanas de demo
- [ ] Performance similar backtest (±30%)
- [ ] Sem bugs de execução
- [ ] DD nunca > 4%

### Fase 8: FTMO ⬜
- [ ] Phase 1 iniciada
- [ ] Monitoramento diário
- [ ] DD sempre < 4%
- [ ] Profit target atingido
- [ ] Phase 2 passada
- [ ] FUNDED! 🎉

---

## APÊNDICES

### A. Scripts Oracle Disponíveis

| Script | Função | Status |
|--------|--------|--------|
| `walk_forward.py` | WFA Rolling/Anchored | ✅ Implementado |
| `walk_forward_segmented.py` | WFA por regime/sessão | 🆕 A implementar |
| `monte_carlo.py` | Block Bootstrap MC | ✅ Implementado |
| `monte_carlo_scenarios.py` | MC multi-cenário | 🆕 A implementar |
| `deflated_sharpe.py` | PSR/DSR | ✅ Implementado |
| `cpcv.py` | CPCV para PBO | 🆕 A implementar |
| `execution_simulator.py` | Custos realistas | ✅ Implementado |
| `prop_firm_validator.py` | Validação FTMO | ✅ Implementado |
| `go_nogo_validator.py` | Pipeline básico | ✅ Implementado |
| `go_nogo_v2.py` | Pipeline v2.0 | 🆕 A implementar |

### B. Modelo de Latência

```python
def latency_model(base_ping=20, is_news=False, vol_pct=50):
    # Network jitter (Gamma)
    L_net = base_ping + np.random.gamma(2.0, 5.0)
    
    # GC pause (5% chance)
    L_inf = 5 + (np.random.uniform(20, 80) if np.random.random() < 0.05 else 0)
    
    # Broker processing
    L_proc = 10 * (3 if is_news else 1) * (1.5 if vol_pct > 75 else 1)
    
    # Packet loss (0.1% chance)
    L_queue = np.random.uniform(200, 400) if np.random.random() < 0.001 else 0
    
    return L_net + L_inf + L_proc + L_queue
```

### C. Configurações de Stress

| Cenário | Spread | Slippage | Latência | Rejection |
|---------|--------|----------|----------|-----------|
| Normal | 1.0x | 1.0x | 1.0x | 2% |
| Pessimistic | 1.5x | 2.0x | 1.5x | 5% |
| Stress | 3.0x | 5.0x | 3.0x | 15% |
| News Event | 5.0x | 10.0x | 4.0x | 30% |

### D. Thresholds por Regime

| Métrica | Trending | Random | Reverting |
|---------|----------|--------|-----------|
| WFE mínimo | 0.65 | N/A | 0.45 |
| PF esperado | >= 1.5 | 0 trades | >= 1.0 |
| Operar? | SIM | NÃO | CAUTELA |

---

*"A diferença entre um trader amador e um profissional é a preparação."*

**Este plano representa validação de nível institucional. Siga-o rigorosamente.**

**GOOD LUCK! 🚀**
