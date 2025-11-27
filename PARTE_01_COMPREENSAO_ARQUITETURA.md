# EA_SCALPER_XAUUSD – Multi-Agent Hybrid System
## PARTE 1: Compreensão e Arquitetura

---

# 📌 SEÇÃO 1 – COMPREENSÃO DO PROBLEMA

- **Objetivo estratégico**: Desenvolver um EA de scalping para XAUUSD que opere com alta precisão em prop firms (FTMO), combinando análise técnica avançada (SMC/ICT) com scoring multi-fator para maximizar win rate e respeitar limites de risco rigorosos.

- **Restrição FTMO - Max Daily Loss (5%)**: O EA não pode perder mais que 5% do saldo inicial em um único dia; ultrapassar = desqualificação imediata.

- **Restrição FTMO - Max Total Loss (10%)**: Perda acumulada máxima de 10% do saldo inicial; ultrapassar = conta eliminada.

- **Restrição FTMO - Profit Target**: Necessário atingir 10% de lucro para passar o challenge (sem prazo limite na versão atual).

- **Multi-agente (MQL5+Python)**: MQL5 garante execução em tempo real com latência mínima; Python permite análise complexa (NLP de notícias, LLM reasoning, APIs externas) sem bloquear o EA.

- **Risco de slippage em XAUUSD**: Alta volatilidade causa execuções distantes do preço desejado; necessário considerar slippage no cálculo de SL/TP.

- **Risco de overtrading**: Scalping gera muitos sinais; sem filtro rigoroso (score threshold), o EA pode abrir trades demais e acumular perdas.

- **Risco de eventos macro**: NFP, FOMC, CPI podem causar gaps e spreads de 50+ pips em XAUUSD; operar nesses momentos é roleta.

- **Risco de violação por sequência de losses**: 3-4 stops seguidos podem consumir 2-3% do capital rapidamente, aproximando do Max Daily Loss.

- **Performance crítica**: OnTick precisa rodar em <50ms para não perder oportunidades em scalping; chamadas externas devem ser assíncronas ou em OnTimer.

---

# 🏗️ SEÇÃO 2 – ARQUITETURA DE ALTO NÍVEL (MQL5 + PYTHON)

## 2.1 Camadas MQL5

```
┌─────────────────────────────────────────────────────────────────┐
│                         EA_SCALPER_XAUUSD                        │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 1: Data & Events                                          │
│  ├── OnTick() - Captura de preço em tempo real                   │
│  ├── OnTimer() - Chamadas assíncronas ao Python Hub              │
│  └── OnTradeTransaction() - Monitoramento de execuções           │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 2: Strategy / Signal Layer                                │
│  ├── COrderBlockModule - Detecta OBs em MTF                      │
│  ├── CFVGModule - Identifica Fair Value Gaps                     │
│  ├── CLiquidityModule - Mapeia pools de liquidez                 │
│  ├── CMarketStructureModule - Analisa HH/HL/LH/LL                │
│  └── CVolatilityModule - ATR e filtros de volatilidade           │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 3: Scoring Engine                                         │
│  └── CSignalScoringModule                                        │
│      ├── ComputeTechScore() - Score técnico 0-100                │
│      ├── MergePythonScores() - Integra Fund/Sent do Hub          │
│      └── ComputeFinalScore() - Pontuação final ponderada         │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 4: Risk & Execution                                       │
│  ├── CFTMORiskManager - Controle de DD, lot sizing, veto         │
│  └── CTradeExecutor - Envio de ordens com retry logic            │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 5: Logging & Notifications                                │
│  └── CLogger - Arquivos, push notifications, reasoning strings   │
└─────────────────────────────────────────────────────────────────┘
```

## 2.2 Python Agent Hub

```
┌─────────────────────────────────────────────────────────────────┐
│                       PYTHON AGENT HUB                           │
│                    (FastAPI REST Server)                         │
├─────────────────────────────────────────────────────────────────┤
│  AGENT 1: TechnicalAgent                                         │
│  └── Análise MTF avançada, confluência de indicadores            │
├─────────────────────────────────────────────────────────────────┤
│  AGENT 2: FundamentalAgent                                       │
│  └── Calendário econômico, impacto de notícias, DXY correlation  │
├─────────────────────────────────────────────────────────────────┤
│  AGENT 3: SentimentAgent                                         │
│  └── Twitter/X sentiment, COT data, retail positioning           │
├─────────────────────────────────────────────────────────────────┤
│  AGENT 4: LLMReasoningAgent                                      │
│  └── Gera reasoning string explicando contexto do trade          │
└─────────────────────────────────────────────────────────────────┘
```

**Escolha de comunicação: HTTP/REST (FastAPI)**

Justificativa:
- Simplicidade de implementação e debugging
- MQL5 possui WebRequest() nativo
- Stateless, fácil de escalar
- Timeout configurável (crucial para não travar OnTick)
- JSON parsing disponível em MQL5

## 2.3 Fluxo de um Tick "Perfeito"

```
[TICK CHEGA]
     │
     ▼
[1] OnTick() captura Bid/Ask
     │
     ▼
[2] Verificar se há posição aberta
     │── SIM ──► Gerenciar trailing/BE ──► FIM
     │
     NO
     ▼
[3] Chamar módulos técnicos (OB, FVG, Liquidity, Structure, ATR)
     │
     ▼
[4] CSignalScoringModule.ComputeTechScore() ──► TechScore (0-100)
     │
     ▼
[5] TechScore >= 60? (pre-filter)
     │── NO ──► FIM (sinal fraco demais)
     │
     YES
     ▼
[6] Consultar cache do Python Hub (atualizado via OnTimer)
     │
     ▼
[7] CSignalScoringModule.ComputeFinalScore(Tech, Fund, Sent)
     │
     ▼
[8] FinalScore >= ExecutionThreshold (85)?
     │── NO ──► Log "Signal rejected: score X < 85" ──► FIM
     │
     YES
     ▼
[9] CFTMORiskManager.CanOpenTrade(risk%, SL_points)?
     │── NO ──► Log "Trade vetoed by RiskManager" ──► FIM
     │
     YES
     ▼
[10] CTradeExecutor.OpenPosition(direction, lot, SL, TP)
     │
     ▼
[11] CLogger.LogTrade(reasoning_string) + Push Notification
     │
     ▼
[FIM]
```

## 2.4 Diagrama de Comunicação MQL5 ↔ Python

```
MQL5_EA (OnTimer a cada 30s)
    │
    ├──► HTTP POST ──► http://127.0.0.1:8000/analyze
    │                       │
    │                       ▼
    │               [Python Agent Hub]
    │                       │
    │    ┌──────────────────┼──────────────────┐
    │    ▼                  ▼                  ▼
    │ [TechAgent]    [FundAgent]        [SentAgent]
    │    │                  │                  │
    │    └──────────────────┼──────────────────┘
    │                       ▼
    │               [LLM Reasoning]
    │                       │
    │                       ▼
    ◄── HTTP 200 + JSON ────┘
    │
    ▼
[Cache Local no EA]
```
