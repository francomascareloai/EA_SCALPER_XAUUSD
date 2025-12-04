# ACTION PLAN SIMPLIFICADO v1.0

**Data**: 2025-12-01  
**Objetivo**: Direção clara com workstreams paralelos  
**Filosofia**: Menos planejamento, mais execução

---

## STATUS ATUAL (O QUE JÁ TEMOS)

```
✅ COMPLETO:
├── Código MQL5 compilando (0 erros)
├── Tick data convertido (318M ticks → 5.5GB Parquet)
├── Scripts Oracle existentes (WFA, Monte Carlo, GO/NO-GO)
├── Modelos ONNX treinados (direction_model.onnx)
└── FTMO Risk Manager implementado

⏳ PENDENTE:
├── Validação GENIUS dos dados
├── Backtest com dados tick
├── Walk-Forward Analysis
└── GO/NO-GO decision
```

---

## PRÓXIMO PASSO IMEDIATO

```
┌─────────────────────────────────────────────────────────────────┐
│                    FAZER AGORA (1-2 horas)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Rodar validação completa dos dados Parquet:                  │
│                                                                 │
│   python scripts/oracle/validate_data.py \                     │
│       --input data/processed/ticks_2024.parquet \              │
│       --output DOCS/04_REPORTS/VALIDATION/                     │
│                                                                 │
│   Isso vai gerar: DATA_QUALITY_GENIUS.md com score             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## WORKSTREAMS PARALELOS (5 AGENTES)

### Como usar os agentes em paralelo:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        5 SESSÕES PARALELAS                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SESSÃO 1: 🔮 ORACLE                                                        │
│  └── "Oracle, roda validate_data.py nos Parquet e gera relatório"          │
│                                                                             │
│  SESSÃO 2: ⚒️ FORGE                                                         │
│  └── "Forge, cria segment_data.py para separar por regime/sessão"          │
│                                                                             │
│  SESSÃO 3: 🔍 ARGUS                                                         │
│  └── "Argus, pesquisa papers sobre regime detection para XAUUSD"           │
│                                                                             │
│  SESSÃO 4: 🛡️ SENTINEL                                                      │
│  └── "Sentinel, calcula Kelly optimal para os dados de 2024"               │
│                                                                             │
│  SESSÃO 5: 🔥 CRUCIBLE                                                      │
│  └── "Crucible, analisa correlações XAUUSD vs DXY/Yields atuais"           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## ROADMAP SIMPLIFICADO (4 SEMANAS)

```
SEMANA 1: VALIDAÇÃO
├── Dia 1-2: Validar dados Parquet (ORACLE)
├── Dia 3-4: Segmentar por regime/sessão (FORGE)
└── Dia 5-7: Backtest baseline no tick data (ORACLE)

SEMANA 2: OTIMIZAÇÃO
├── Dia 1-3: Walk-Forward Analysis (ORACLE)
├── Dia 4-5: Monte Carlo simulation (ORACLE)
└── Dia 6-7: Kelly calibration por regime (SENTINEL)

SEMANA 3: INTEGRAÇÃO
├── Dia 1-3: Testar EA com modelo ONNX (FORGE)
├── Dia 4-5: Stress testing (ORACLE + SENTINEL)
└── Dia 6-7: GO/NO-GO decision (ORACLE)

SEMANA 4: DEMO
├── Dia 1-7: Demo trading FTMO
└── Se OK: Challenge
```

---

## PROMPTS PRONTOS PARA CADA AGENTE

### 🔮 ORACLE - Validação e Backtest
```
Oracle, preciso que você:
1. Rode validate_data.py nos arquivos data/processed/ticks_*.parquet
2. Gere relatório de qualidade em DOCS/04_REPORTS/VALIDATION/
3. Identifique gaps, anomalias e score de qualidade
```

### ⚒️ FORGE - Código
```
Forge, preciso que você:
1. Crie scripts/backtest/segment_data.py
2. Função: Separar ticks por regime (trending/ranging/reverting)
3. Função: Separar por sessão (Asian/London/NY)
4. Output: data/segments/*.parquet
```

### 🔍 ARGUS - Research
```
Argus, pesquise:
1. Papers recentes sobre regime detection em XAUUSD
2. Melhores práticas de WFA para prop trading
3. EVT (Extreme Value Theory) para tail risk em scalping
```

### 🛡️ SENTINEL - Risco
```
Sentinel, calcule:
1. Kelly optimal para cada regime usando dados 2024
2. Position sizing máximo para FTMO $100k
3. Circuit breaker thresholds (DD 4%, 5%, 8%)
```

### 🔥 CRUCIBLE - Estratégia
```
Crucible, analise:
1. Correlação atual XAUUSD vs DXY
2. Comportamento por sessão (qual sessão tem melhor edge?)
3. Setup atual: está em regime favorável para scalping?
```

---

## DECISÕES SIMPLES

| Se... | Então... |
|-------|----------|
| Dados têm gaps > 24h | Parar, re-exportar dados |
| WFE < 0.5 | Estratégia não funciona, voltar à pesquisa |
| Monte Carlo DD > 15% | Reduzir risk/trade |
| Confidence < 70 | NO-GO, não fazer challenge |
| Tudo OK | GO para demo, depois challenge |

---

## ARQUIVOS QUE IMPORTAM (IGNORE O RESTO)

```
LEIA ESTES:
├── DOCS/02_IMPLEMENTATION/PROGRESS.md          # Status atual
├── DOCS/02_IMPLEMENTATION/ACTION_PLAN_SIMPLE.md # Este arquivo
├── data/processed/CONVERSION_STATS.json         # Stats dos dados
└── scripts/oracle/go_nogo_validator.py          # Pipeline final

IGNORE (muito longo, já consolidado):
├── MASTER_EXECUTION_PLAN.md
├── MASTER_EXECUTION_PLAN_FINAL.md
└── Qualquer arquivo > 1000 linhas de planejamento
```

---

## CHECKLIST DIÁRIO

```
[ ] Qual fase estou? (Validação/Otimização/Integração/Demo)
[ ] Qual agente usar hoje?
[ ] O que fazer em 2-4 horas?
[ ] Commit no final do dia
```

---

## REGRA DE OURO

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   SE ESTÁ CONFUSO → RODE UM SCRIPT                             │
│                                                                 │
│   Melhor rodar validate_data.py e ver resultado                │
│   do que ler mais 1000 linhas de plano                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

*Criado em 2025-12-01 para simplificar a vida*
