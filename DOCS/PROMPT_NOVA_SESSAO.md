# Prompt para Nova Sessao - EA_SCALPER_XAUUSD

Use este prompt para iniciar uma nova sessao de desenvolvimento com contexto completo.

---

## Prompt Completo (Copiar e Colar)

```
## Contexto do Projeto

Projeto: EA_SCALPER_XAUUSD
Objetivo: Expert Advisor de alta assertividade para XAUUSD, FTMO-compliant
Modelo: Claude Opus 4.5 com Sequential Thinking

### Documentos Criticos

1. **PRD v2.1**: `DOCS/prd.md` - Especificacao completa do sistema
2. **Master Plan**: `DOCS/MASTER_PLAN_EA_SCALPER_XAUUSD.md` - Analise profunda e roadmap
3. **CLAUDE.md**: Instrucoes de desenvolvimento (carrega automatico)

### Arquitetura Definida (4 Camadas)

```
CAMADA ESTRATEGICA (LLM): Daily briefing, Trade analysis, Optimization
CAMADA INTELIGENCIA (Python): LSTM-CNN direction, Regime classifier, Fund/Sent
CAMADA SINAL (MQL5): Order Block, FVG, Liquidity, Scoring
CAMADA EXECUCAO (MQL5): FTMO Risk Manager, Trade Executor, Position Manager
```

### Metricas Target

- Win Rate: 65-75%
- Risk:Reward: 1.5:1 a 2:1
- Monthly: 12-20%
- Max DD: 5-8%
- FTMO Pass Rate: 75-85%

### Insights Chave da Analise

1. **90% WR e armadilha** - O correto e expectancy positiva
2. **ML nao cria edge** - ML AMPLIFICA edge existente do SMC
3. **Ensemble confidence** - Trade apenas quando TODOS sinais alinham
4. **Regime matters** - Mesma estrategia nao funciona em todos mercados
5. **Validation first** - Walk-forward + Monte Carlo antes de live

### Timeline

- Semana 1-2: MQL5 Core (SMC + Risk)
- Semana 3-4: ML Integration (LSTM + ONNX)
- Semana 5-6: Regime + Validation
- Semana 7-8: Paper Trading
- Semana 9-10: Live Small → FTMO

### Estrutura de Arquivos a Criar

```
MQL5/
├── Include/
│   ├── Modules/
│   │   ├── EliteOrderBlock.mqh    ← PRIMEIRO
│   │   ├── EliteFVG.mqh
│   │   ├── InstitutionalLiquidity.mqh
│   │   └── MarketStructure.mqh
│   ├── Core/
│   │   ├── SignalScoring.mqh
│   │   └── TradeExecutor.mqh
│   └── Risk/
│       └── FTMO_RiskManager.mqh
└── Experts/
    └── EA_SCALPER_XAUUSD.mq5
```

## Modo de Operacao

**BUILDER MODE**: Codigo, nao documentacao.

- Consultar PRD para specs
- Consultar Master Plan para decisoes de arquitetura
- NAO perguntar, FAZER
- Validar cada modulo antes de proximo

## Tarefa Atual

[ESPECIFICAR AQUI A TAREFA]

Exemplos:
- "Implementar EliteOrderBlock.mqh seguindo o PRD"
- "Criar pipeline de features para LSTM"
- "Integrar ONNX model no MQL5"

## Comando

Comecar imediatamente. Ler PRD se necessario. Codigo primeiro.
```

---

## Prompt Resumido (Para Tarefas Rapidas)

```
Projeto: EA_SCALPER_XAUUSD
Docs: PRD em DOCS/prd.md, Master Plan em DOCS/MASTER_PLAN_EA_SCALPER_XAUUSD.md

Target: 65-75% WR, 1.8:1 R:R, FTMO-compliant
Arquitetura: MQL5 (SMC) + Python (LSTM-CNN via ONNX) + LLM (strategic)

Modo: BUILDER - codigo, nao docs.

Tarefa: [ESPECIFICAR]

Comecar agora.
```

---

## Prompts por Fase

### Fase 1: MQL5 Core

```
Contexto: EA_SCALPER_XAUUSD, FTMO-compliant scalper
Docs: DOCS/prd.md (Secao 5.2 - Componentes), DOCS/MASTER_PLAN_EA_SCALPER_XAUUSD.md

Tarefa: Implementar [MODULO] em MQL5

Requisitos:
- Detectar [PADRAO] conforme PRD
- Retornar score 0-100
- Interface: Initialize(), Calculate(), GetScore()
- Seguir padroes do projeto

Criar arquivo: MQL5/Include/Modules/[MODULO].mqh

Comecar agora. Codigo completo.
```

### Fase 2: ML Integration

```
Contexto: EA_SCALPER_XAUUSD, adicionando ML layer
Docs: DOCS/MASTER_PLAN_EA_SCALPER_XAUUSD.md (Secao 3.2)

Tarefa: [ESCOLHER]
- Criar data pipeline para features
- Treinar LSTM-CNN model
- Exportar para ONNX
- Integrar ONNX em MQL5

Arquitetura ML:
- Input: 60 timesteps, 20 features
- Model: Conv1D → BiLSTM → Attention → Dense
- Output: Probabilidade bullish (0-1)

Comecar agora.
```

### Fase 3: Validation

```
Contexto: EA_SCALPER_XAUUSD, validando sistema
Docs: DOCS/MASTER_PLAN_EA_SCALPER_XAUUSD.md (Secao 4)

Tarefa: [ESCOLHER]
- Implementar Walk-Forward Analysis
- Implementar Monte Carlo Simulation
- Rodar Stress Tests
- Criar checklist pre-live

Criterios:
- WFE > 0.5
- Monte Carlo 95% DD < 12%
- Sharpe > 1.2

Comecar agora.
```

---

## Comandos Uteis na Sessao

| Comando | Acao |
|---------|------|
| `/research [topic]` | Pesquisa profunda com MCPs |
| `/validate-ftmo` | Verificar compliance FTMO |
| `/code-review` | Review de codigo trading |
| `/optimize-prompt` | Otimizar qualquer prompt |
| `ultrathink` | Analise profunda com sequential thinking |

---

## Arquivos de Referencia

| Arquivo | Conteudo |
|---------|----------|
| `DOCS/prd.md` | PRD v2.1 completo |
| `DOCS/MASTER_PLAN_EA_SCALPER_XAUUSD.md` | Analise profunda + roadmap |
| `CLAUDE.md` | Instrucoes de desenvolvimento |
| `.bmad/mql5-elite-ops/agents/` | Personas especializadas |

---

## Notas Importantes

1. **Nunca mais planejamento** - PRD e Master Plan estao completos
2. **Builder Mode always** - Codigo > documentacao
3. **Validar cada passo** - Testar antes de proximo
4. **Consultar docs** - Nao perguntar o que ja esta documentado
5. **Sequential thinking** - Para analises complexas

---

*Ultima atualizacao: 2025-11-28*
