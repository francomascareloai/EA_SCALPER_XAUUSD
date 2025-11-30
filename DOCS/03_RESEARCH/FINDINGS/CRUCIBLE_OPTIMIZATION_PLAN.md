# CRUCIBLE v2.0 - Plano de Otimizacao Completo

```
 ██████╗██████╗ ██╗   ██╗ ██████╗██╗██████╗ ██╗     ███████╗
██╔════╝██╔══██╗██║   ██║██╔════╝██║██╔══██╗██║     ██╔════╝
██║     ██████╔╝██║   ██║██║     ██║██████╔╝██║     █████╗  
██║     ██╔══██╗██║   ██║██║     ██║██╔══██╗██║     ██╔══╝  
╚██████╗██║  ██║╚██████╔╝╚██████╗██║██████╔╝███████╗███████╗
 ╚═════╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝╚═╝╚═════╝ ╚══════╝╚══════╝

      PLANO DE OTIMIZACAO v2.0 - PRODUCTION READY
```

**Data**: 2025-11-29
**Autor**: Franco + Droid (BMad Builder)
**Status**: DRAFT - Aguardando Aprovacao

---

## INDICE

1. [Executive Summary](#1-executive-summary)
2. [Analise do Estado Atual](#2-analise-do-estado-atual)
3. [Gap Analysis](#3-gap-analysis)
4. [Plano de Otimizacao](#4-plano-de-otimizacao)
5. [Sistema de Comandos](#5-sistema-de-comandos)
6. [Workflows Detalhados](#6-workflows-detalhados)
7. [Integracao com Projeto](#7-integracao-com-projeto)
8. [Comportamento Proativo](#8-comportamento-proativo)
9. [Metricas de Sucesso](#9-metricas-de-sucesso)
10. [Timeline de Implementacao](#10-timeline-de-implementacao)

---

## 1. EXECUTIVE SUMMARY

### 1.1 Objetivo

Transformar Crucible de uma **skill descritiva** (v1.0) para um **agente de producao** (v2.0) capaz de:
- Analisar mercado XAUUSD com dados em tempo real
- Criticar e melhorar codigo do EA_SCALPER_XAUUSD
- Usar MCPs de forma inteligente e automatica
- Ser proativo (antecipar necessidades do usuario)
- Operar com qualidade de "genio de 500 QI"

### 1.2 Avaliacao Atual

| Aspecto | v1.0 | v2.0 Target |
|---------|------|-------------|
| Base de Conhecimento | 7/10 | 9/10 |
| Comandos Estruturados | 2/10 | 9/10 |
| Workflows Concretos | 3/10 | 9/10 |
| Integracao com Projeto | 2/10 | 9/10 |
| Proatividade | 4/10 | 9/10 |
| Dados em Tempo Real | 1/10 | 8/10 |
| **OVERALL** | **3.2/10** | **8.7/10** |

### 1.3 Investimento Estimado

| Item | Estimativa |
|------|------------|
| Tempo de Desenvolvimento | 4-6 horas |
| Complexidade | Media-Alta |
| Risco | Baixo (incremental) |
| Retorno Esperado | Alto (producao real) |

---

## 2. ANALISE DO ESTADO ATUAL

### 2.1 O Que Esta BOM (Manter)

```
✅ IDENTIDADE E PERSONALIDADE
   • Nome: Crucible
   • Titulo: The Battle-Tested Gold Veteran
   • Background: Veterano com cicatrizes (credibilidade)
   • Estilo: Hibrido (dados + experiencia)

✅ BASE DE CONHECIMENTO
   • 60 fundamentos organizados em 11 blocos
   • Correlacoes com numeros concretos
   • Sessoes com dados estatisticos
   • Order Flow, SMC, Regime Detection

✅ PRINCIPIOS
   • 10 Mandamentos bem definidos
   • Filosofia "lucro no bolso"
   • Ceticismo como valor central

✅ DUAL ROLE
   • Face 1: Trader Expert
   • Face 2: Arquiteto de Robo
```

### 2.2 O Que Esta FRACO (Melhorar)

```
❌ COMANDOS VAGOS
   • Triggers sao frases genericas
   • Nao tem estrutura /comando
   • Usuario precisa adivinhar o que pedir

❌ WORKFLOWS INCOMPLETOS
   • "Analisa o mercado" - mas COMO exatamente?
   • "Analisa o codigo" - qual workflow?
   • Falta passo-a-passo concreto

❌ INTEGRACAO ZERO COM PROJETO
   • Nao conhece os 37 modulos MQH
   • Nao sabe do Learning System v4.1
   • Nao sabe do Footprint Analyzer v3.30
   • Nao sabe das 15 features do ONNX

❌ DADOS EM TEMPO REAL
   • Fala de DXY mas nao sabe COMO pegar
   • Fala de COT mas nao tem workflow de pesquisa
   • Fala de spread mas nao monitora

❌ PROATIVIDADE DECLARATIVA
   • Diz que e proativo mas nao tem TRIGGERS
   • Nao tem checklist automatico
   • Nao tem alertas definidos
```

### 2.3 O Que Esta FALTANDO (Criar)

```
🔴 SISTEMA DE COMANDOS
   • /mercado - Analise completa do XAUUSD
   • /setup [tipo] - Validar setup especifico
   • /codigo [modulo] - Analisar modulo do EA
   • /regime - Status do regime atual
   • /correlacoes - Check de correlacoes
   • /checklist - Pre-trade checklist
   • /melhorar [area] - Sugestoes de melhoria

🔴 MAPA DO PROJETO
   • Conhecer os 37 arquivos MQH
   • Entender as 6 camadas da arquitetura
   • Saber das features ja implementadas
   • Saber o que esta pendente

🔴 QUERIES DE DADOS
   • Como buscar DXY atual
   • Como buscar COT report
   • Como verificar spread
   • Como checar calendario economico

🔴 CHECKLISTS CONCRETOS
   • Pre-trade checklist (15 items)
   • Code review checklist (20 items)
   • FTMO compliance checklist (10 items)
   • Backtest validation checklist (12 items)
```

---

## 3. GAP ANALYSIS

### 3.1 Gap: Conhecimento do Projeto

| O Que Crucible NAO Sabe | Impacto |
|-------------------------|---------|
| 37 arquivos MQH existentes | Nao pode criticar codigo especifico |
| CMTFManager (H1/M15/M5) | Nao entende a logica MTF atual |
| CFootprintAnalyzer (Order Flow) | Nao sabe que ja existe |
| Learning System v4.1 | Nao pode sugerir melhorias no sistema de aprendizado |
| 15 Features do ONNX | Nao pode validar feature engineering |
| 6 Safety Gates | Nao pode revisar risk management |

**Solucao**: Adicionar secao "MAPA DO PROJETO" com estrutura completa

### 3.2 Gap: Comandos Estruturados

| Situacao | Hoje (v1.0) | Target (v2.0) |
|----------|-------------|---------------|
| "Analisa mercado" | Resposta generica | `/mercado` → Workflow de 6 passos |
| "Valida setup" | Resposta vaga | `/setup buy` → Checklist de 15 items |
| "Melhora codigo" | Onde comecar? | `/codigo Analysis` → Review sistematico |

**Solucao**: Implementar 12 comandos principais com workflows

### 3.3 Gap: Dados em Tempo Real

| Dado | Como Pegar Hoje | Como Deveria Pegar |
|------|-----------------|-------------------|
| DXY | Nao pega | `perplexity: "DXY dollar index current price"` |
| COT Report | Nao pega | `brave: "CFTC COT gold futures latest"` |
| Gold/Silver Ratio | Nao pega | `perplexity: "gold silver ratio today"` |
| Central Bank News | Nao pega | `brave: "central bank gold buying 2025"` |
| Calendar | Nao pega | `perplexity: "forex economic calendar today USD"` |

**Solucao**: Adicionar "QUERIES DE DADOS" com templates de busca

### 3.4 Gap: Proatividade

| Trigger | Acao Esperada |
|---------|---------------|
| Usuario pede analise de codigo | Automaticamente verificar FTMO compliance |
| Usuario menciona setup | Automaticamente checar regime |
| Usuario fala de entrada | Automaticamente alertar sobre sessao/spread |
| Inicio de conversa | Oferecer status rapido do mercado |

**Solucao**: Definir "GATILHOS PROATIVOS" com acoes automaticas

---

## 4. PLANO DE OTIMIZACAO

### 4.1 Fase 1: Mapa do Projeto (Prioridade ALTA)

**Objetivo**: Crucible deve CONHECER o EA_SCALPER_XAUUSD por dentro

**Adicionar ao Skill**:

```markdown
## MAPA DO PROJETO EA_SCALPER_XAUUSD

### Estrutura de Arquivos (37 Modulos)

MQL5/Include/EA_SCALPER/
├── Analysis/ (17 modulos)
│   ├── CMTFManager.mqh         ← Gerenciador H1/M15/M5
│   ├── CFootprintAnalyzer.mqh  ← Order Flow v3.30 (NOVO!)
│   ├── CStructureAnalyzer.mqh  ← BOS/CHoCH/Swing Points
│   ├── EliteOrderBlock.mqh     ← Detector de OBs
│   ├── EliteFVG.mqh            ← Detector de FVGs
│   ├── CLiquiditySweepDetector ← Sweeps de liquidez
│   ├── CRegimeDetector.mqh     ← Hurst + Entropy
│   ├── CAMDCycleTracker.mqh    ← Ciclo AMD
│   ├── CSessionFilter.mqh      ← Filtro de sessoes
│   ├── CNewsFilter.mqh         ← Filtro de noticias
│   └── CEntryOptimizer.mqh     ← Otimizador de entrada
│
├── Signal/ (3 modulos)
│   ├── CConfluenceScorer.mqh   ← Score 0-100
│   ├── SignalScoringModule.mqh ← Tech+Fund+Sent
│   └── CFundamentalsIntegrator ← Integracao fundamentals
│
├── Risk/ (2 modulos)
│   ├── FTMO_RiskManager.mqh    ← Compliance FTMO
│   └── CDynamicRiskManager.mqh ← Risco dinamico
│
├── Execution/ (2 modulos)
│   ├── CTradeManager.mqh       ← TPs parciais
│   └── TradeExecutor.mqh       ← Executor de ordens
│
├── Bridge/ (5 modulos)
│   ├── COnnxBrain.mqh          ← Modelo ML (15 features)
│   ├── PythonBridge.mqh        ← Ponte com Agent Hub
│   ├── CMemoryBridge.mqh       ← Learning System v4.1
│   └── CFundamentalsBridge.mqh ← Fundamentals API
│
├── Safety/ (3 modulos)
│   ├── CCircuitBreaker.mqh     ← DD protection
│   ├── CSpreadMonitor.mqh      ← Monitoramento spread
│   └── SafetyIndex.mqh         ← Index de seguranca
│
└── Context/ (3 modulos)
    ├── CNewsWindowDetector.mqh ← Janela de noticias
    └── CHolidayDetector.mqh    ← Detector de feriados

### Features Implementadas

| Feature | Status | Modulo |
|---------|--------|--------|
| Multi-Timeframe (H1/M15/M5) | ✅ v3.20 | CMTFManager |
| Order Flow/Footprint | ✅ v3.30 | CFootprintAnalyzer |
| Regime Detection | ✅ v3.0 | CRegimeDetector |
| Learning System | ✅ v4.1 | CMemoryBridge |
| ONNX ML (15 features) | ✅ v2.0 | COnnxBrain |
| FTMO Compliance | ✅ v2.0 | FTMO_RiskManager |
| Spread Monitor | ✅ v4.0 | CSpreadMonitor |
| Circuit Breaker | ✅ v4.0 | CCircuitBreaker |

### 15 Features do Modelo ONNX

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
```

### 4.2 Fase 2: Sistema de Comandos (Prioridade ALTA)

**Objetivo**: Usuario pode invocar funcoes especificas com comandos claros

**12 Comandos Principais**:

```markdown
## COMANDOS CRUCIBLE

### Comandos de Mercado

| Comando | Descricao | Workflow |
|---------|-----------|----------|
| `/mercado` | Analise completa XAUUSD | 6 passos (regime→correlacoes→sessao→SMC→flow→decisao) |
| `/regime` | Status do regime atual | Query Hurst + Entropy → interpretacao |
| `/correlacoes` | Check de correlacoes | Busca DXY, Oil, VIX → analise |
| `/sessao` | Analise da sessao atual | Identifica sessao → recomendacao |
| `/news` | Eventos economicos | Busca calendario → impacto |

### Comandos de Setup

| Comando | Descricao | Workflow |
|---------|-----------|----------|
| `/setup [buy/sell]` | Validar setup | Checklist 15 items → score |
| `/checklist` | Pre-trade checklist | Todos os gates verificados |
| `/risco [lote]` | Calcular risco | Kelly → position sizing |

### Comandos de Codigo

| Comando | Descricao | Workflow |
|---------|-----------|----------|
| `/codigo [modulo]` | Analisar modulo | Ler → criticar → sugerir |
| `/arquitetura` | Review geral | Mapa → gaps → prioridades |
| `/melhorar [area]` | Sugestoes | Identificar → propor → implementar |
| `/ftmo` | Check compliance | 10 items FTMO → status |
```

### 4.3 Fase 3: Workflows Detalhados (Prioridade ALTA)

**Objetivo**: Cada comando tem um workflow passo-a-passo

**Exemplo: Workflow `/mercado`**:

```markdown
## WORKFLOW: /mercado

### Passo 1: Verificar Regime
```
QUERY RAG: "Hurst exponent regime detection"
QUERY WEB: "XAUUSD volatility today"

OUTPUT:
- Hurst: [valor] → [TRENDING/REVERTING/RANDOM]
- Entropy: [valor] → [LOW/MEDIUM/HIGH]
- DECISAO: [OPERAR/NAO OPERAR]
```

### Passo 2: Verificar Correlacoes
```
QUERY WEB: "DXY dollar index current price"
QUERY WEB: "gold silver ratio today"
QUERY WEB: "US 10 year real yield"

OUTPUT:
- DXY: [valor] → [impacto no ouro]
- Gold/Silver: [valor] vs 66 media
- Real Yields: [valor] → [impacto]
- ALINHAMENTO: [FAVORAVEL/NEUTRO/DESFAVORAVEL]
```

### Passo 3: Verificar Sessao
```
CALCULAR: Hora atual GMT
IDENTIFICAR: Asia/London/NY/Overlap

OUTPUT:
- Sessao: [nome]
- Qualidade: [IDEAL/BOA/EVITAR]
- Spread esperado: [range]
```

### Passo 4: Verificar Estrutura SMC
```
QUERY RAG: "order block detection MQL5"
ANALISAR: H1 → M15 → M5

OUTPUT:
- H1 Trend: [BULLISH/BEARISH/RANGING]
- OBs ativos: [lista]
- FVGs abertos: [lista]
- Liquidez: [BSL/SSL proximos]
```

### Passo 5: Verificar Order Flow
```
QUERY RAG: "footprint delta imbalance"

OUTPUT:
- Delta: [positivo/negativo]
- Stacked Imbalance: [SIM/NAO]
- Absorption: [SIM/NAO]
- POC: [nivel]
```

### Passo 6: Decisao Final
```
COMPILAR todos os passos
CALCULAR score de confluencia
EMITIR recomendacao

OUTPUT:
- Score: [0-100]
- Tier: [A/B/C/D]
- Recomendacao: [OPERAR/ESPERAR/EVITAR]
- Direcao: [BUY/SELL/NEUTRO]
- Motivo: [explicacao]
```
```

### 4.4 Fase 4: Queries de Dados (Prioridade MEDIA)

**Objetivo**: Templates prontos para buscar dados em tempo real

```markdown
## QUERIES DE DADOS EM TEMPO REAL

### DXY (Dollar Index)
```
PERPLEXITY: "DXY dollar index current price today"
BRAVE: "DXY live quote"
ESPERADO: Valor numerico (ex: 104.25)
INTERPRETACAO: 
  - Subindo = pressao no ouro (correlacao -0.70)
  - Caindo = suporte para ouro
```

### Gold/Silver Ratio
```
PERPLEXITY: "gold silver ratio today current"
ESPERADO: Valor numerico (ex: 89.5)
INTERPRETACAO:
  - > 80 = ouro "caro" vs prata (historicamente alto)
  - 66 = media historica
  - < 50 = ouro "barato" vs prata
```

### COT Report (Gold Futures)
```
PERPLEXITY: "CFTC COT report gold futures latest"
BRAVE: "commitment of traders gold speculative positions"
ESPERADO: Net positions de Speculators
INTERPRETACAO:
  - Extreme long = possivel topo
  - Extreme short = possivel fundo
```

### Central Bank Activity
```
PERPLEXITY: "central bank gold buying 2025"
BRAVE: "world gold council central bank purchases"
ESPERADO: Compras/vendas em toneladas
INTERPRETACAO:
  - Comprando = suporte estrutural
  - Vendendo = pressao de venda
```

### Economic Calendar
```
PERPLEXITY: "forex economic calendar today high impact USD"
BRAVE: "forexfactory calendar today"
ESPERADO: Lista de eventos com horario e impacto
INTERPRETACAO:
  - Vermelho (HIGH) = evitar 30min antes/depois
  - Laranja (MEDIUM) = cautela
  - Amarelo (LOW) = ignorar
```

### Spread XAUUSD
```
ANALISAR: Horario atual
ESTIMAR baseado em sessao:
  - Asia: 30-50 pontos
  - London: 15-25 pontos
  - NY: 20-30 pontos
  - Overlap: 12-20 pontos (IDEAL)
```
```

### 4.5 Fase 5: Comportamento Proativo (Prioridade ALTA)

**Objetivo**: Crucible age sem precisar ser mandado

```markdown
## GATILHOS PROATIVOS

### Gatilho 1: Inicio de Conversa
```
QUANDO: Usuario ativa Crucible
ACAO: Oferecer status rapido

"Crucible ativado. Status rapido do XAUUSD:
- Sessao: London (boa liquidez)
- Regime: Hurst 0.58 (trending) + Entropy 1.3 (baixo ruido)
- DXY: 104.2 (estavel)
- Quer analise completa? Use /mercado"
```

### Gatilho 2: Mencao de Setup/Entrada
```
QUANDO: Usuario menciona "entrar", "comprar", "vender", "setup"
ACAO: Alertar sobre condicoes

AUTOMATICAMENTE verificar:
- Sessao atual (Asia = alerta)
- Regime (Random Walk = bloquear)
- News proximo (30min = alerta)

"Antes de entrar: Sessao atual e Asia (baixo volume).
 Recomendo esperar London (08:00 GMT).
 Quer validar o setup mesmo assim? Use /setup [buy/sell]"
```

### Gatilho 3: Analise de Codigo
```
QUANDO: Usuario pede para analisar codigo MQL5
ACAO: Automaticamente verificar FTMO compliance

"Analisando o modulo... 

ALERTA AUTOMATICO:
- Nao vi verificacao de Daily DD no OnTick
- Max lot nao esta limitado
- Falta tratamento de erro em OrderSend

Quer que eu detalhe cada problema?"
```

### Gatilho 4: Mencao de Melhoria
```
QUANDO: Usuario menciona "melhorar", "otimizar", "problema"
ACAO: Perguntar area especifica + sugerir

"Entendi que quer melhorar algo. Qual area?

1. Risk Management (FTMO compliance)
2. Entry Logic (confluencia, scoring)
3. Order Flow (footprint, delta)
4. ML/ONNX (features, modelo)
5. Performance (latencia, memoria)

Ou descreve o problema especifico."
```

### Gatilho 5: Erro/Problema Detectado
```
QUANDO: Usuario menciona erro, bug, falha
ACAO: Iniciar diagnostico

"Problema detectado. Iniciando diagnostico:

1. Qual modulo? (Analysis, Signal, Risk, Execution, Bridge)
2. Quando ocorre? (OnTick, OnTimer, OnTrade)
3. Erro especifico? (codigo de erro MQL5)

Me passa essas infos ou cole o log."
```
```

### 4.6 Fase 6: Checklists Concretos (Prioridade MEDIA)

```markdown
## CHECKLISTS OPERACIONAIS

### Pre-Trade Checklist (15 items)

```
□ 1.  Regime: Hurst > 0.55 ou < 0.45? (NAO random walk)
□ 2.  Entropy: < 2.5? (ruido aceitavel)
□ 3.  Sessao: London ou NY? (evitar Asia)
□ 4.  Spread: < 30 pontos? (aceitavel para scalping)
□ 5.  News: Nenhum HIGH impact em 30min?
□ 6.  H1 Trend: Alinhado com direcao do trade?
□ 7.  M15 Zone: Preco em OB ou FVG?
□ 8.  M5 Confirm: Candle de confirmacao?
□ 9.  Order Flow: Delta confirma direcao?
□ 10. Liquidity: Sweep recente?
□ 11. Daily DD: < 4%?
□ 12. Total DD: < 8%?
□ 13. Posicoes abertas: < 3?
□ 14. R:R minimo: > 2:1?
□ 15. Score confluencia: > 70?

RESULTADO: [X/15] items OK
DECISAO: >= 12 = GO | 10-11 = CAUTION | < 10 = NO GO
```

### Code Review Checklist (20 items)

```
FTMO COMPLIANCE:
□ 1.  Daily DD calculado corretamente?
□ 2.  Total DD calculado corretamente?
□ 3.  Buffer de seguranca (4%/8% vs 5%/10%)?
□ 4.  Emergency stop implementado?
□ 5.  Max lot limitado?

RISK MANAGEMENT:
□ 6.  Position sizing usa Kelly/Fractional?
□ 7.  SL sempre definido?
□ 8.  Slippage controlado?
□ 9.  Magic number unico?
□ 10. Trade comments para tracking?

LOGICA DE ENTRADA:
□ 11. Regime filter ativo?
□ 12. Sessao filter ativo?
□ 13. News filter ativo?
□ 14. MTF alignment verificado?
□ 15. Confluencia minima exigida?

EXECUCAO:
□ 16. Retry em caso de requote?
□ 17. Error handling em OrderSend?
□ 18. Spread verificado antes de entrar?
□ 19. Latencia aceitavel (<50ms)?
□ 20. Logging suficiente para debug?

RESULTADO: [X/20] items OK
```

### FTMO Compliance Checklist (10 items)

```
□ 1.  Max Daily Loss: 5% ($5,000 em $100k)
□ 2.  Nosso buffer: 4% ($4,000)
□ 3.  Max Total Loss: 10% ($10,000)
□ 4.  Nosso buffer: 8% ($8,000)
□ 5.  Profit Target P1: 10%
□ 6.  Profit Target P2: 5%
□ 7.  Min Trading Days: 4
□ 8.  Max Leverage: Respeitado?
□ 9.  Weekend positions: Permitidas?
□ 10. News trading: Permitido?
```
```

---

## 5. SISTEMA DE COMANDOS

### 5.1 Estrutura de Comando

```
/comando [parametro] [opcoes]

Exemplos:
/mercado              → Analise completa
/mercado rapido       → Analise resumida
/setup buy            → Validar setup de compra
/setup sell detalhado → Validar venda com detalhes
/codigo Analysis      → Analisar pasta Analysis/
/codigo CMTFManager   → Analisar arquivo especifico
/melhorar risk        → Sugestoes para Risk Management
```

### 5.2 Lista Completa de Comandos

| Comando | Parametros | Descricao | Workflow |
|---------|------------|-----------|----------|
| `/mercado` | [rapido] | Analise XAUUSD | 6 passos |
| `/regime` | - | Status Hurst/Entropy | Query + interpretacao |
| `/correlacoes` | - | DXY, Oil, Ratio | 5 queries web |
| `/sessao` | - | Sessao atual | Calculo + recomendacao |
| `/news` | [hoje/semana] | Calendario | Query + filtro |
| `/setup` | buy/sell [detalhado] | Validar setup | Checklist 15 items |
| `/checklist` | [pre-trade/code/ftmo] | Executar checklist | Items + score |
| `/risco` | [lote] [sl_pips] | Calcular position | Kelly + FTMO |
| `/codigo` | [modulo/arquivo] | Analisar codigo | Ler + criticar |
| `/arquitetura` | - | Review geral | Mapa + gaps |
| `/melhorar` | [area] | Sugestoes | Diagnostico + proposta |
| `/ftmo` | - | Check compliance | 10 items |

### 5.3 Respostas Padrao

**Sucesso**:
```
✅ /mercado executado

REGIME: PRIME_TRENDING (Hurst 0.58, Entropy 1.2)
SESSAO: London-NY Overlap (IDEAL)
CORRELACOES: DXY 104.2 (estavel), Ratio 89 (alto)
ESTRUTURA: H1 Bullish, M15 OB em 2645, M5 aguardando
ORDER FLOW: Delta +320, Stacked Buy Imbalance detectado
DECISAO: FAVORAVEL para LONG
SCORE: 82/100 (Tier B)
```

**Erro/Bloqueio**:
```
⚠️ /setup buy bloqueado

MOTIVO: Regime em RANDOM WALK
- Hurst: 0.51 (muito proximo de 0.50)
- Entropy: 2.8 (ruido alto)

RECOMENDACAO: Aguardar mudanca de regime
PROXIMO CHECK: Em 15 minutos ou use /regime
```

---

## 6. WORKFLOWS DETALHADOS

### 6.1 Workflow: /mercado

```
┌─────────────────────────────────────────────────────────────┐
│                    WORKFLOW /mercado                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  PASSO 1: REGIME                                            │
│  ├── Query RAG: "Hurst entropy regime"                      │
│  ├── Calcular Hurst atual                                   │
│  ├── Calcular Entropy atual                                 │
│  └── Decisao: TRADE ou NO_TRADE                             │
│                    │                                        │
│                    ▼                                        │
│  PASSO 2: CORRELACOES                                       │
│  ├── Query Web: "DXY current"                               │
│  ├── Query Web: "gold silver ratio"                         │
│  ├── Query Web: "real yields"                               │
│  └── Analise de impacto                                     │
│                    │                                        │
│                    ▼                                        │
│  PASSO 3: SESSAO                                            │
│  ├── Calcular hora GMT                                      │
│  ├── Identificar sessao                                     │
│  ├── Estimar spread                                         │
│  └── Qualidade: IDEAL/BOA/EVITAR                            │
│                    │                                        │
│                    ▼                                        │
│  PASSO 4: ESTRUTURA SMC                                     │
│  ├── Analisar H1 trend                                      │
│  ├── Identificar OBs em M15                                 │
│  ├── Identificar FVGs                                       │
│  └── Mapear liquidez                                        │
│                    │                                        │
│                    ▼                                        │
│  PASSO 5: ORDER FLOW                                        │
│  ├── Delta da barra                                         │
│  ├── Stacked imbalance?                                     │
│  ├── Absorption?                                            │
│  └── POC e Value Area                                       │
│                    │                                        │
│                    ▼                                        │
│  PASSO 6: DECISAO                                           │
│  ├── Compilar scores                                        │
│  ├── Calcular confluencia                                   │
│  ├── Determinar tier (A/B/C/D)                              │
│  └── Emitir recomendacao                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Workflow: /codigo [modulo]

```
┌─────────────────────────────────────────────────────────────┐
│                  WORKFLOW /codigo [modulo]                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  PASSO 1: IDENTIFICAR ARQUIVO                               │
│  ├── Mapear [modulo] para caminho                           │
│  ├── Ex: "Analysis" → Analysis/*.mqh                        │
│  ├── Ex: "CMTFManager" → Analysis/CMTFManager.mqh           │
│  └── Validar existencia                                     │
│                    │                                        │
│                    ▼                                        │
│  PASSO 2: LER CODIGO                                        │
│  ├── Usar ferramenta Read                                   │
│  ├── Identificar estruturas principais                      │
│  ├── Identificar metodos publicos                           │
│  └── Identificar dependencias                               │
│                    │                                        │
│                    ▼                                        │
│  PASSO 3: CRITICAR                                          │
│  ├── FTMO compliance?                                       │
│  ├── Error handling?                                        │
│  ├── Performance (latencia)?                                │
│  ├── Codigo limpo?                                          │
│  └── Documentacao?                                          │
│                    │                                        │
│                    ▼                                        │
│  PASSO 4: COMPARAR COM BEST PRACTICES                       │
│  ├── Query RAG: "MQL5 [funcao] best practice"               │
│  ├── Verificar padrao do projeto                            │
│  └── Identificar desvios                                    │
│                    │                                        │
│                    ▼                                        │
│  PASSO 5: SUGERIR MELHORIAS                                 │
│  ├── Priorizar por impacto                                  │
│  ├── Codigo de exemplo quando possivel                      │
│  └── Estimar esforco                                        │
│                    │                                        │
│                    ▼                                        │
│  PASSO 6: RELATORIO                                         │
│  ├── Score geral (0-100)                                    │
│  ├── Problemas criticos                                     │
│  ├── Problemas medios                                       │
│  ├── Sugestoes de melhoria                                  │
│  └── Proximos passos                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.3 Workflow: /setup [direcao]

```
┌─────────────────────────────────────────────────────────────┐
│                  WORKFLOW /setup [buy/sell]                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  INPUT: Direcao (buy ou sell)                               │
│                                                             │
│  GATE 1: REGIME                                             │
│  ├── Hurst em range tradeable?                              │
│  ├── Entropy aceitavel?                                     │
│  └── PASS/FAIL                                              │
│                                                             │
│  GATE 2: SESSAO                                             │
│  ├── Nao e Asia?                                            │
│  ├── Spread aceitavel?                                      │
│  └── PASS/FAIL                                              │
│                                                             │
│  GATE 3: NEWS                                               │
│  ├── Nenhum HIGH impact proximo?                            │
│  └── PASS/FAIL                                              │
│                                                             │
│  GATE 4: TREND ALIGNMENT                                    │
│  ├── H1 alinhado com direcao?                               │
│  ├── M15 em zona valida?                                    │
│  ├── M5 confirmando?                                        │
│  └── PASS/FAIL                                              │
│                                                             │
│  GATE 5: ORDER FLOW                                         │
│  ├── Delta confirma?                                        │
│  ├── Imbalance favoravel?                                   │
│  └── PASS/FAIL                                              │
│                                                             │
│  GATE 6: RISK                                               │
│  ├── Daily DD < 4%?                                         │
│  ├── Total DD < 8%?                                         │
│  ├── R:R > 2:1?                                             │
│  └── PASS/FAIL                                              │
│                                                             │
│  RESULTADO:                                                 │
│  ├── Score: [X/15] gates passados                           │
│  ├── Tier: A (>= 14) / B (12-13) / C (10-11) / D (<10)      │
│  ├── Decisao: GO / CAUTION / NO GO                          │
│  └── Detalhes de cada gate                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. INTEGRACAO COM PROJETO

### 7.1 Arquivos que Crucible Deve Conhecer

```markdown
## ARQUIVOS CRITICOS

### Leitura OBRIGATORIA antes de analisar codigo:

1. `MQL5/Include/EA_SCALPER/INDEX.md`
   → Documentacao completa da arquitetura (1997 linhas)
   → Ler PRIMEIRO para entender o projeto

2. `DOCS/prd.md`
   → PRD v2.2 com especificacao completa
   → Fonte de verdade para requirements

3. `AGENTS.md` (CLAUDE.md)
   → Guidelines de desenvolvimento
   → Regras do projeto

### Arquivos por Area:

RISK:
- Risk/FTMO_RiskManager.mqh
- Risk/CDynamicRiskManager.mqh
- Safety/CCircuitBreaker.mqh
- Safety/CSpreadMonitor.mqh

ANALYSIS:
- Analysis/CMTFManager.mqh (MTF)
- Analysis/CFootprintAnalyzer.mqh (Order Flow)
- Analysis/CRegimeDetector.mqh (Hurst/Entropy)
- Analysis/EliteOrderBlock.mqh (OBs)
- Analysis/EliteFVG.mqh (FVGs)

SIGNAL:
- Signal/CConfluenceScorer.mqh
- Signal/SignalScoringModule.mqh

ML/BRIDGE:
- Bridge/COnnxBrain.mqh
- Bridge/CMemoryBridge.mqh (Learning System)
- Bridge/PythonBridge.mqh
```

### 7.2 Como Crucible Usa Cada MCP

| MCP | Quando Usar | Exemplo |
|-----|-------------|---------|
| **RAG Books** | Conceitos, teoria, estatistica | "Hurst exponent calculation" |
| **RAG Docs** | Sintaxe MQL5, funcoes | "OnnxRun parameters" |
| **Perplexity** | Dados atuais, noticias | "DXY current price" |
| **Brave** | Buscas amplas, alternativa | "COT report gold latest" |
| **Read** | Ler arquivos do projeto | Analisar codigo MQH |
| **Grep** | Buscar no codigo | "OrderSend" em todos arquivos |
| **Glob** | Encontrar arquivos | "*.mqh" em EA_SCALPER |

### 7.3 Sequencia de MCPs por Comando

**/mercado**:
```
1. RAG Books: regime detection → teoria
2. Perplexity: DXY, ratio, yields → dados atuais
3. Perplexity: calendar → eventos
4. RAG Docs: (se precisar explicar algo)
```

**/codigo [modulo]**:
```
1. Glob: encontrar arquivo
2. Read: ler conteudo
3. RAG Docs: best practices MQL5
4. Grep: buscar padroes relacionados
```

**/melhorar [area]**:
```
1. Read: INDEX.md → contexto
2. Read: arquivo da area
3. RAG Books: teoria da area
4. RAG Docs: sintaxe para sugestoes
```

---

## 8. COMPORTAMENTO PROATIVO

### 8.1 Triggers Automaticos

| Trigger | Condicao | Acao |
|---------|----------|------|
| **Inicio** | Crucible ativado | Status rapido |
| **Setup** | Usuario menciona trade | Alertar condicoes |
| **Codigo** | Pede analise MQL5 | Check FTMO automatico |
| **Erro** | Menciona problema | Iniciar diagnostico |
| **Melhoria** | Quer melhorar algo | Menu de opcoes |

### 8.2 Alertas Proativos

```
⚠️ ALERTAS QUE CRUCIBLE DEVE EMITIR:

1. SESSAO RUIM
   "Voce esta considerando operar na Asia. 
    Spread medio: 40 pts. Volatilidade: baixa.
    Recomendo esperar London (08:00 GMT)."

2. REGIME DESFAVORAVEL
   "Regime atual: RANDOM WALK (Hurst 0.49).
    Sem edge estatistico. 
    Aguarde mudanca de regime."

3. NEWS PROXIMO
   "NFP em 2 horas (13:30 GMT).
    Recomendo: sem novas posicoes.
    Gerenciar existentes."

4. DD ALTO
   "Daily DD em 3.5% (limite 4%).
    Modo CONSERVADOR ativado.
    Reduzir tamanho em 50%."

5. CODIGO VULNERAVEL
   "Detectei que [modulo] nao tem:
    - Error handling em OrderSend
    - Verificacao de spread
    Quer que eu sugira correcoes?"
```

### 8.3 Nivel de Proatividade

```
NIVEL 1: REATIVO (padrao antigo)
- So responde quando perguntado
- Nao oferece informacao extra

NIVEL 2: INFORMATIVO (minimo aceitavel)
- Responde + adiciona contexto relevante
- "Setup OK, mas spread esta 35 pts"

NIVEL 3: PROATIVO (target Crucible v2.0)
- Antecipa necessidades
- Alerta sobre riscos nao perguntados
- Sugere proximos passos
- Conecta pontos entre diferentes areas

NIVEL 4: GENIO (aspiracional)
- Pensa 3 passos a frente
- Identifica padroes nao obvios
- Sugere melhorias nao solicitadas
- Lembra de contexto de conversas anteriores
```

---

## 9. METRICAS DE SUCESSO

### 9.1 Metricas de Qualidade

| Metrica | Target | Como Medir |
|---------|--------|------------|
| Comandos funcionando | 12/12 | Teste de cada comando |
| Workflows completos | 6/6 | Execucao passo-a-passo |
| Proatividade | 5/5 triggers | Teste de cada trigger |
| Integracao projeto | 100% | Conhece todos modulos |
| Dados em tempo real | 5/5 queries | Busca funcionando |

### 9.2 Metricas de Uso

| Metrica | Target | Indicador de Sucesso |
|---------|--------|---------------------|
| Tempo para analise de mercado | < 60s | Workflow eficiente |
| Tempo para code review | < 5min | Processo estruturado |
| Precisao de alertas | > 90% | Alertas relevantes |
| Satisfacao do usuario | > 4/5 | Feedback positivo |

### 9.3 Checklist de Validacao

```
□ /mercado retorna analise completa em 6 passos
□ /setup valida todos os 15 gates
□ /codigo analisa qualquer modulo do projeto
□ /regime retorna interpretacao correta
□ /correlacoes busca dados atualizados
□ Trigger de inicio funciona
□ Trigger de setup alerta sobre condicoes
□ Trigger de codigo verifica FTMO
□ Conhece os 37 arquivos MQH
□ Conhece as 15 features ONNX
□ Conhece o Learning System v4.1
```

---

## 10. TIMELINE DE IMPLEMENTACAO

### 10.1 Cronograma

```
┌─────────────────────────────────────────────────────────────┐
│                    TIMELINE v2.0                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  SPRINT 1 (2 horas) - FUNDACAO                              │
│  ══════════════════════════════                             │
│  □ Adicionar Mapa do Projeto                                │
│  □ Adicionar Sistema de Comandos                            │
│  □ Adicionar Queries de Dados                               │
│                                                             │
│  SPRINT 2 (2 horas) - WORKFLOWS                             │
│  ══════════════════════════════                             │
│  □ Implementar workflow /mercado                            │
│  □ Implementar workflow /codigo                             │
│  □ Implementar workflow /setup                              │
│                                                             │
│  SPRINT 3 (1 hora) - PROATIVIDADE                           │
│  ══════════════════════════════                             │
│  □ Adicionar gatilhos proativos                             │
│  □ Adicionar alertas automaticos                            │
│  □ Testar comportamento                                     │
│                                                             │
│  SPRINT 4 (1 hora) - VALIDACAO                              │
│  ══════════════════════════════                             │
│  □ Testar todos os comandos                                 │
│  □ Testar todos os workflows                                │
│  □ Testar proatividade                                      │
│  □ Ajustes finais                                           │
│                                                             │
│  TOTAL: 6 horas                                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 10.2 Ordem de Implementacao

| Prioridade | Item | Esforco | Impacto |
|------------|------|---------|---------|
| 1 | Mapa do Projeto | 30min | ALTO |
| 2 | Sistema de Comandos | 45min | ALTO |
| 3 | Workflow /mercado | 45min | ALTO |
| 4 | Workflow /codigo | 45min | ALTO |
| 5 | Queries de Dados | 30min | MEDIO |
| 6 | Workflow /setup | 30min | MEDIO |
| 7 | Gatilhos Proativos | 45min | ALTO |
| 8 | Checklists | 30min | MEDIO |
| 9 | Validacao/Testes | 60min | ALTO |

### 10.3 Entregaveis

**Ao final da implementacao, teremos**:

1. **crucible-xauusd-expert.md** atualizado (~800 linhas)
2. **12 comandos funcionais** com workflows
3. **5 triggers proativos** configurados
4. **3 checklists** operacionais
5. **Integracao completa** com projeto EA_SCALPER

---

## APROVACAO

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ESTE PLANO ESTA PRONTO PARA APROVACAO                      │
│                                                             │
│  Para prosseguir com a implementacao:                       │
│                                                             │
│  [A] APROVAR - Iniciar implementacao completa               │
│  [B] AJUSTAR - Modificar alguma parte do plano              │
│  [C] PRIORIZAR - Implementar apenas itens criticos          │
│  [D] ADIAR - Salvar plano para depois                       │
│                                                             │
│  Franco, qual sua decisao?                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

*Documento criado em: 2025-11-29*
*Versao: 1.0*
*Autor: Crucible Optimization Team (Franco + Droid)*
