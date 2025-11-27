# EA_SCALPER_XAUUSD – Multi-Agent Hybrid System
## PARTE 1: Compreensão do Problema e Arquitetura

---

# 📌 SEÇÃO 1 – COMPREENSÃO DO PROBLEMA

- **Objetivo estratégico**: Desenvolver um EA de scalping para XAUUSD que opere com alta precisão em prop firms (FTMO), combinando análise técnica SMC/ICT com scoring multi-fator para maximizar win rate e respeitar limites de risco.

- **Restrição FTMO - Max Daily Loss (5%)**: Perder mais que 5% do saldo em um dia = desqualificação imediata.

- **Restrição FTMO - Max Total Loss (10%)**: Perda acumulada máxima de 10% = conta eliminada.

- **Restrição FTMO - Profit Target**: 10% de lucro para passar o challenge.

- **Multi-agente (MQL5+Python)**: MQL5 executa em tempo real (<50ms); Python analisa dados complexos (NLP, LLM) sem bloquear o EA.

- **Risco de slippage em XAUUSD**: Alta volatilidade causa execuções distantes do preço; considerar no cálculo de SL/TP.

- **Risco de overtrading**: Scalping gera muitos sinais; sem filtro (score >= 85), acumula perdas.

- **Risco de eventos macro**: NFP, FOMC, CPI causam gaps e spreads de 50+ pips; evitar operar.

- **Risco de sequência de losses**: 3-4 stops = 2-3% de perda rápida, aproximando do Max Daily Loss.

- **Performance crítica**: OnTick < 50ms; chamadas externas em OnTimer (assíncrono).

---

# 🏗️ SEÇÃO 2 – ARQUITETURA DE ALTO NÍVEL

## 2.1 Camadas MQL5

```
┌─────────────────────────────────────────────────────────────┐
│                     EA_SCALPER_XAUUSD                        │
├─────────────────────────────────────────────────────────────┤
│  LAYER 1: Data & Events                                      │
│  ├── OnTick() - Preço real-time                              │
│  ├── OnTimer() - Chamadas Python (30s)                       │
│  └── OnTradeTransaction() - Monitor execuções                │
├─────────────────────────────────────────────────────────────┤
│  LAYER 2: Strategy / Signal Layer                            │
│  ├── COrderBlockModule                                       │
│  ├── CFVGModule                                              │
│  ├── CLiquidityModule                                        │
│  ├── CMarketStructureModule                                  │
│  └── CVolatilityModule                                       │
├─────────────────────────────────────────────────────────────┤
│  LAYER 3: Scoring Engine                                     │
│  └── CSignalScoringModule (Tech + Fund + Sent = Final)       │
├─────────────────────────────────────────────────────────────┤
│  LAYER 4: Risk & Execution                                   │
│  ├── CFTMORiskManager (VETO POWER)                           │
│  └── CTradeExecutor                                          │
├─────────────────────────────────────────────────────────────┤
│  LAYER 5: Logging                                            │
│  └── CLogger (Push notifications, reasoning strings)         │
└─────────────────────────────────────────────────────────────┘
```

## 2.2 Python Agent Hub (FastAPI REST)

```
┌─────────────────────────────────────────────────────────────┐
│                    PYTHON AGENT HUB                          │
│                 http://127.0.0.1:8000                        │
├─────────────────────────────────────────────────────────────┤
│  TechnicalAgent   → Análise MTF avançada                     │
│  FundamentalAgent → Calendário, notícias, DXY                │
│  SentimentAgent   → Twitter, COT, retail positioning         │
│  LLMReasoningAgent → Gera reasoning string                   │
└─────────────────────────────────────────────────────────────┘
```

**Por que HTTP/REST?** MQL5 tem WebRequest() nativo, stateless, timeout configurável, fácil debug.

## 2.3 Fluxo do Tick Perfeito

```
[TICK] → Spread OK? → Posição aberta? → Análise técnica
                                              ↓
                                     TechScore >= 60?
                                              ↓
                                     Cache Python (Fund/Sent)
                                              ↓
                                     FinalScore >= 85?
                                              ↓
                                     RiskManager.CanOpenTrade()?
                                              ↓
                                     EXECUTA + LOG + PUSH
```
