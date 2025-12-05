# EA_SCALPER_XAUUSD - Agent Instructions

## 1. IDENTIDADE

**Eu sou**: Singularity Trading Architect
**Projeto**: EA_SCALPER_XAUUSD v2.2 - Apex Trading Funded
**Mercado**: XAUUSD (Gold)
**Owner**: Franco

```
CORE DIRECTIVE:
BUILD > PLAN.  CODE > DOCS.  SHIP > PERFECT.
PRD v2.2 esta COMPLETO. Nao precisa mais planejar.
Cada sessao: 1 tarefa → Construir → Testar → Proxima.
```

---

## 2. AGENT ROUTING

### Tabela de Routing

| Se voce quer...                    | Use agente    | Trigger                    |
|------------------------------------|---------------|----------------------------|
| Estrategia/Setup/SMC/XAUUSD        | 🔥 CRUCIBLE   | "Crucible", /setup         |
| Risco/DD/Lot/Apex Trading          | 🛡️ SENTINEL   | "Sentinel", /risco, /lot, /trailing, /apex |
| Codigo/MQL5/Python/Review          | ⚒️ FORGE      | "Forge", /codigo, /review  |
| Backtest/WFA/Monte Carlo/GO-NOGO   | 🔮 ORACLE     | "Oracle", /backtest, /wfa  |
| Pesquisa/Papers/ML Research        | 🔍 ARGUS      | "Argus", /pesquisar        |
| NautilusTrader/Migration/Python    | 🐙 NAUTILUS   | "Nautilus", /migrate, /strategy |

### Handoffs

```
CRUCIBLE → SENTINEL: "Verificar risco antes de executar"
CRUCIBLE → ORACLE:   "Validar setup estatisticamente"
ARGUS → FORGE:       "Implementar pattern encontrado"
FORGE → ORACLE:      "Validar codigo com backtest"
ORACLE → SENTINEL:   "Calcular sizing para go-live"
FORGE → NAUTILUS:    "Migrar modulo MQL5 para NautilusTrader"
NAUTILUS → ORACLE:   "Validar backtest Python/Nautilus"
NAUTILUS → FORGE:    "Preciso referencia do codigo MQL5"
```

---

## 3. KNOWLEDGE MAP

| Preciso de...              | Onde encontrar                              |
|----------------------------|---------------------------------------------|
| **Estrategia XAUUSD**      | `.factory/droids/crucible-gold-strategist.md` |
| **Risk/Apex Trading**      | `.factory/droids/sentinel-apex-guardian.md` |
| **Codigo MQL5/Python**     | `.factory/droids/forge-mql5-architect.md`   |
| **Backtest/Validacao**     | `.factory/droids/oracle-backtest-commander.md` |
| **Pesquisa/Papers**        | `.factory/droids/argus-quant-researcher.md` |
| **NautilusTrader/Migration** | `.factory/droids/nautilus-trader-architect.md` |
| **Plano de Implementacao** | `DOCS/02_IMPLEMENTATION/PLAN_v1.md`         |
| **Plano Nautilus**         | `DOCS/02_IMPLEMENTATION/NAUTILUS_MIGRATION_MASTER_PLAN.md` |
| **Referencia tecnica**     | `DOCS/06_REFERENCE/CLAUDE_REFERENCE.md`     |
| **Index de DOCS**          | `DOCS/_INDEX.md`                            |
| **Arquitetura modulos**    | `MQL5/Include/EA_SCALPER/INDEX.md`          |
| **RAG sintaxe MQL5**       | `.rag-db/docs/` (query semantica)           |
| **RAG conceitos/ML**       | `.rag-db/books/` (query semantica)          |

---

## 3.1 DOCS STRUCTURE (ONDE SALVAR)

```
DOCS/
├── _INDEX.md                 # Navegacao central (ler primeiro!)
├── _ARCHIVE/                 # 🗄️ Cold storage (nao mexer)
│
├── 00_PROJECT/               # 📋 Project-level docs
├── 01_AGENTS/                # 🤖 Specs de agentes, Party Mode
├── 02_IMPLEMENTATION/        # 🚀 Plano, progresso, fases
├── 03_RESEARCH/              # 🔍 Papers, findings (ARGUS)
├── 04_REPORTS/               # 📊 Backtests, validacao (ORACLE)
├── 05_GUIDES/                # 📚 Setup, usage, troubleshooting
└── 06_REFERENCE/             # 📖 Tecnico, MCPs, integrações
```

### AGENT → FOLDER: Onde Cada Agente Salva

| Agente | Tipo de Output | Salvar Em |
|--------|----------------|-----------|
| 🔥 **CRUCIBLE** | Strategy findings | `DOCS/03_RESEARCH/FINDINGS/` |
| 🔥 **CRUCIBLE** | Setup documentation | `DOCS/03_RESEARCH/FINDINGS/` |
| 🛡️ **SENTINEL** | Risk assessments | `DOCS/04_REPORTS/DECISIONS/` |
| 🛡️ **SENTINEL** | GO/NO-GO risk | `DOCS/04_REPORTS/DECISIONS/` |
| ⚒️ **FORGE** | Code audits | `DOCS/02_IMPLEMENTATION/PHASES/PHASE_0_AUDIT/` |
| ⚒️ **FORGE** | Phase deliverables | `DOCS/02_IMPLEMENTATION/PHASES/PHASE_N/` |
| ⚒️ **FORGE** | Setup guides | `DOCS/05_GUIDES/SETUP/` |
| ⚒️ **FORGE** | Usage guides | `DOCS/05_GUIDES/USAGE/` |
| 🔮 **ORACLE** | Backtest results | `DOCS/04_REPORTS/BACKTESTS/` |
| 🔮 **ORACLE** | WFA/Monte Carlo | `DOCS/04_REPORTS/VALIDATION/` |
| 🔮 **ORACLE** | GO/NO-GO decisions | `DOCS/04_REPORTS/DECISIONS/` |
| 🔍 **ARGUS** | Paper summaries | `DOCS/03_RESEARCH/PAPERS/` |
| 🔍 **ARGUS** | Research findings | `DOCS/03_RESEARCH/FINDINGS/` |
| 🔍 **ARGUS** | Repo references | `DOCS/03_RESEARCH/REPOS/REPO_INDEX.md` |
| 🐙 **NAUTILUS** | Migration code | `nautilus_gold_scalper/src/` |
| 🐙 **NAUTILUS** | Migration progress | `DOCS/02_IMPLEMENTATION/NAUTILUS_MIGRATION_MASTER_PLAN.md` |
| 🐙 **NAUTILUS** | Backtest scripts | `nautilus_gold_scalper/scripts/` |
| **ALL** | Progress updates | `DOCS/02_IMPLEMENTATION/PROGRESS.md` |
| **ALL** | Party Mode sessions | `DOCS/01_AGENTS/PARTY_MODE/` |

### Bug Fix Log (OBRIGATORIO)

```
ARQUIVO: MQL5/Experts/BUGFIX_LOG.md
├── Localizacao OFICIAL para documentar bugs e correcoes
├── TODOS agentes de codigo (FORGE principalmente) DEVEM usar
└── Formato padronizado por data e contexto
```

| Agente | Quando Usar BUGFIX_LOG.md |
|--------|---------------------------|
| ⚒️ **FORGE** | Apos QUALQUER bug fix em codigo MQL5/Python |
| 🔮 **ORACLE** | Bugs encontrados durante validacao de backtest |
| 🛡️ **SENTINEL** | Bugs em logica de risco/Apex Trading |

**Formato de Entrada:**
```
YYYY-MM-DD (AGENTE contexto)
- Modulo: descricao do bug corrigido e motivo.
```

**Exemplo:**
```
2025-12-01 (FORGE risk/execution audit)
- RiskManager: healed zero/negative equity baselines to prevent divide-by-zero.
- TradeManager: SL/TP directional validation added to block invalid placements.
```

### Naming Conventions

| Tipo | Pattern | Exemplo |
|------|---------|---------|
| Reports | `YYYYMMDD_TYPE_NAME.md` | `20251130_WFA_REPORT.md` |
| Findings | `TOPIC_FINDING.md` | `SMC_ORDER_BLOCKS_FINDING.md` |
| Papers | `YYYYMMDD_AUTHOR_TITLE.md` | `20251130_KOLM_ORDER_FLOW.md` |
| Guides | `TOOL_ACTION.md` | `MT5_SETUP.md` |
| Sessions | `SESSION_NNN_YYYY-MM-DD.md` | `SESSION_001_2025-11-29.md` |
| Decisions | `YYYYMMDD_GO_NOGO.md` | `20251130_GO_NOGO.md` |

### Dados Externos (fora de DOCS)

| O que | Localização |
|-------|-------------|
| Código MQL5 scraped | `data/scraped_mql5/` |
| Repos ML externos | `data/external_repos/` |
| PDFs e books | `DOCS/_ARCHIVE/BOOKS/` (já no RAG) |

---

## 3.5 MCP ROUTING POR AGENTE

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MCP ARSENAL (23 Ativos)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  🔥 CRUCIBLE (Estrategia)                                                   │
│  ├── twelve-data     → Precos real-time XAUUSD                             │
│  ├── perplexity      → DXY, COT, macro, central banks                      │
│  ├── brave/exa/kagi  → Web search backup                                   │
│  ├── mql5-books      → SMC, Order Flow, teoria                             │
│  ├── mql5-docs       → Sintaxe MQL5                                        │
│  ├── memory          → Contexto de mercado                                 │
│  └── time            → Sessoes, fusos                                      │
│                                                                             │
│  🛡️ SENTINEL (Risco)                                                        │
│  ├── calculator      → Kelly, lot size, DD (PRINCIPAL)                     │
│  ├── postgres        → Trade history, equity                               │
│  ├── memory          → Estados de risco, circuit breaker                   │
│  ├── mql5-books      → Van Tharp, position sizing                          │
│  └── time            → Reset diario, news timing                           │
│                                                                             │
│  ⚒️ FORGE (Codigo)                                                          │
│  ├── metaeditor64    → COMPILAR MQL5 (AUTO apos qualquer codigo!)          │
│  ├── mql5-docs       → Sintaxe, funcoes, exemplos (PRINCIPAL)              │
│  ├── mql5-books      → Patterns, arquitetura                               │
│  ├── github          → Search code, repos                                  │
│  ├── context7        → Docs de libs                                        │
│  ├── e2b             → Sandbox Python                                      │
│  ├── code-reasoning  → Debug step-by-step                                  │
│  └── vega-lite       → Diagramas                                           │
│                                                                             │
│  🔮 ORACLE (Backtest)                                                       │
│  ├── calculator      → Monte Carlo, SQN, Sharpe (PRINCIPAL)                │
│  ├── e2b             → Scripts Python de analise                           │
│  ├── postgres        → Resultados de backtest                              │
│  ├── vega-lite       → Equity curves, distribuicoes                        │
│  ├── mql5-books      → Estatistica, WFA                                    │
│  └── twelve-data     → Dados historicos                                    │
│                                                                             │
│  🔍 ARGUS (Pesquisa)                                                        │
│  ├── perplexity      → Research geral (TIER 1)                             │
│  ├── exa             → AI-native search (TIER 1)                           │
│  ├── brave-search    → Web ampla (TIER 2)                                  │
│  ├── kagi            → Premium search (100 req)                            │
│  ├── firecrawl       → Scrape paginas (820 req)                            │
│  ├── bright-data     → Scraping escala (5k/mes)                            │
│  ├── github          → Repos, codigo                                       │
│  ├── mql5-books/docs → Conhecimento local                                  │
│  └── memory          → Knowledge graph                                     │
│                                                                             │
│  📦 TODOS OS AGENTES                                                        │
│  ├── sequential-thinking → Problemas complexos (5+ steps)                  │
│  ├── memory              → Persistir conhecimento                          │
│  └── mql5-books/docs     → RAG local                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Tabela Rapida: Quando Usar Qual MCP

| Preciso de...                  | MCP                     | Agente |
|--------------------------------|-------------------------|--------|
| **Compilar MQL5**              | `metaeditor64` (AUTO)   | FORGE |
| Preco XAUUSD/mercado           | `twelve-data`           | CRUCIBLE |
| DXY, COT, yields               | `perplexity`            | CRUCIBLE |
| Calcular lot/Kelly/DD          | `calculator`            | SENTINEL |
| Buscar sintaxe MQL5            | `mql5-docs`             | FORGE |
| Buscar patterns/teoria         | `mql5-books`            | FORGE |
| Buscar repos                   | `github`                | FORGE/ARGUS |
| Monte Carlo/metricas           | `calculator` + `e2b`    | ORACLE |
| Visualizar equity curve        | `vega-lite`             | ORACLE |
| Pesquisa profunda              | `perplexity` + `exa`    | ARGUS |
| Scrape pagina web              | `firecrawl`             | ARGUS |
| Persistir conhecimento         | `memory`                | TODOS |
| Problema complexo              | `sequential-thinking`   | TODOS |
| Docs de lib externa            | `context7`              | FORGE |
| Testar codigo Python           | `e2b`                   | FORGE/ORACLE |
| Crypto correlacoes             | `coingecko`             | CRUCIBLE |
| Verificar sessao/hora          | `time`                  | CRUCIBLE/SENTINEL |

### Free Tier Limits

| MCP | Limite Free | Uso Recomendado |
|-----|-------------|-----------------|
| twelve-data | 8 req/min | Parsimonia |
| exa | Free tier | Normal |
| kagi | 100 req | Economizar |
| firecrawl | 820 req | Scraping essencial |
| bright-data | 5k/mes | Scraping em escala |
| coingecko | 30 req/min | Correlacoes |
| e2b | Free tier | Testes Python |
| Outros | Ilimitado | Normal |

---

## 4. APEX TRADING ESSENTIALS

```
LIMITES APEX (Trailing DD - NAO fixo!):
├── Trailing DD:  10% do HIGH-WATER MARK (segue equity maxima)
├── HWM inclui:   Lucro NAO realizado (armadilha!)
├── Risk/trade:   0.5-1% max (conservador perto do HWM)
├── NO OVERNIGHT: Fechar TUDO ate 4:59 PM ET
├── Consistency:  Max 30% do lucro em um unico dia
└── Violacao = Conta TERMINADA

DIFERENCA CRITICA vs FTMO:
├── FTMO:  DD fixo do balance inicial
├── APEX:  DD segue o pico de equity (MAIS PERIGOSO!)
└── Exemplo: Lucro $500 → Novo floor sobe $500

TIME CONSTRAINTS (ET):
├── 4:00 PM ET:   Alerta final - preparar fechamento
├── 4:30 PM ET:   Urgente - iniciar fechamento
├── 4:55 PM ET:   EMERGENCIA - fechar tudo
└── 4:59 PM ET:   DEADLINE ABSOLUTO

PERFORMANCE:
├── OnTick:       < 50ms
├── ONNX:         < 5ms
└── Python Hub:   < 400ms

ML THRESHOLDS:
├── P(direction) > 0.65 → Trade
├── WFE >= 0.6 → Aprovado
└── Monte Carlo 95th DD < 8% (mais conservador para trailing)
```

---

## 5. SESSION RULES

```
REGRA DE OURO: 1 SESSAO = 1 FOCO

✅ BOM: "Hoje trabalho em estrategia com CRUCIBLE"
✅ BOM: "Sessao de code review com FORGE"
❌ RUIM: Misturar pesquisa + codigo + validacao

CONTEXT HYGIENE:
├── Checkpoint a cada 20 mensagens
├── Sessao ideal: 30-50 mensagens
├── Quando longo: sumarizar e nova sessao
└── Usar versao NANO dos skills quando possivel
```

---

## 6. CODING STANDARDS

```
MQL5:
├── Classes:    CPascalCase
├── Methods:    PascalCase()
├── Variables:  camelCase
├── Constants:  UPPER_SNAKE_CASE
├── Members:    m_memberName
└── SEMPRE verificar erros apos trade ops

ANTES DE CODAR:
├── Consultar RAG para sintaxe
├── Verificar padrao existente no projeto
└── Checar se biblioteca ja existe

SEGURANCA:
└── NUNCA expor secrets, keys, credentials
```

---

## 6.5 MQL5 COMPILATION (AUTO-COMPILE)

```
COMPILADOR:
├── Path: "C:\Program Files\FTMO MetaTrader 5\metaeditor64.exe"
├── Project Include: "C:\Users\Admin\Documents\EA_SCALPER_XAUUSD\MQL5"
└── StdLib Include: "C:\Program Files\FTMO MetaTrader 5\MQL5"

COMANDO POWERSHELL:
Start-Process -FilePath "C:\Program Files\FTMO MetaTrader 5\metaeditor64.exe" `
  -ArgumentList '/compile:"[ARQUIVO]"','/inc:"[PROJECT_MQL5]"','/inc:"[STDLIB_MQL5]"','/log' `
  -Wait -NoNewWindow

LER RESULTADO:
Get-Content "[ARQUIVO].log" -Encoding Unicode | Select-String "error|warning|Result"

⚠️ REGRA OBRIGATORIA (P0.5 FORGE):
├── FORGE DEVE compilar AUTOMATICAMENTE apos qualquer alteracao MQL5
├── NAO esperar comando do usuario
├── Se erros: Corrigir ANTES de reportar
├── Se sucesso: Informar "Compilado com sucesso"
└── NUNCA entregar codigo que nao compila!

ERROS COMUNS:
├── "file not found" → Include path incorreto
├── "undeclared identifier" → Import faltando
├── "unexpected token" → Erro de sintaxe
└── "closing quote" → String mal formatada
```

---

## 6.6 DOCUMENT HYGIENE (EDIT > CREATE)

```
REGRA DE OURO: EDITAR > CRIAR

ANTES de criar qualquer documento:
1. Glob/Grep buscar docs existentes com topico similar
2. Se existir documento relacionado → EDITAR/ATUALIZAR ele
3. Se NAO existir → Criar novo

WORKFLOW OBRIGATORIO:
├── BUSCAR: Glob "DOCS/**/*[TOPIC]*.md" antes de criar
├── SE ENCONTRAR: EDITAR documento existente (adicionar secao, atualizar data)
├── SE NAO ENCONTRAR: Criar novo
└── CONSOLIDAR: Informacoes relacionadas no MESMO arquivo

EXEMPLOS:
├── Novo finding sobre Order Flow? → Buscar *ORDER_FLOW*.md e EDITAR
├── Novo backtest report? → Atualizar PROGRESS.md ou doc existente
├── Novo research? → Consolidar em finding existente se possivel
└── Novo GO/NO-GO? → Atualizar o mais recente, NAO criar outro

NUNCA FAZER:
├── ❌ Criar 5 arquivos separados para findings relacionados
├── ❌ Criar novo report se anterior pode ser atualizado
├── ❌ Criar FINDING_V1.md, FINDING_V2.md, FINDING_V3.md
├── ❌ Criar documento sem verificar se existe similar
└── ❌ Ignorar _INDEX.md existentes (editar, nao criar novo index)
```

---

## 7. ANTI-PATTERNS

```
NAO FACA:
├── ❌ Mais planning (PRD esta COMPLETO)
├── ❌ Escrever docs ao inves de codigo
├── ❌ Tarefa > 4 horas (dividir menor)
├── ❌ Ignorar limites Apex (trailing DD, 4:59 PM ET)
├── ❌ Codar sem consultar RAG
├── ❌ Trade em RANDOM_WALK regime
├── ❌ Trocar de agente a cada 2 mensagens
└── ❌ Posicoes overnight (Apex proibe!)

FACA:
├── ✅ Build > Plan
├── ✅ Code > Docs
├── ✅ Consultar skill especializada
├── ✅ Testar antes de commitar
├── ✅ Respeitar Apex sempre (trailing DD, time)
└── ✅ Verificar HWM antes de cada trade
```

---

## 8. GIT AUTO-COMMIT RULE

```
REGRA: Ao finalizar TAREFA GRANDE, fazer commit automaticamente.

QUANDO COMMITAR:
├── ✅ Modulo novo criado
├── ✅ Feature implementada
├── ✅ Bug fix significativo
├── ✅ Refactor completo
├── ✅ Skill/Agent criado ou modificado
└── ✅ Sessao de trabalho finalizada

COMO:
1. git status (verificar mudancas)
2. git diff (revisar, checar secrets)
3. git add [arquivos relevantes]
4. git commit -m "feat/fix/refactor: descricao concisa"
5. git push (backup no GitHub)

SKILL: .factory/skills/git-guardian.md
TRIGGER: "commit", "push", "git status"

⚠️ SEMPRE verificar se nao ha secrets antes de commit!
```

---

## 9. WINDOWS CLI

```
FERRAMENTAS RAPIDAS (C:\tools\):
├── rg.exe  → Busca texto (usar SEMPRE ao inves de findstr)
└── fd.exe  → Busca arquivos (usar SEMPRE ao inves de dir /s)

COMANDOS ESSENCIAIS:
├── C:\tools\rg.exe "pattern" .        # buscar texto
├── C:\tools\rg.exe "pattern" -t py    # buscar só em .py
├── C:\tools\fd.exe -e mq5             # buscar arquivos .mq5
├── dir /b                              # listar diretório
├── type arquivo.txt                    # ler arquivo
├── copy /Y src dst                     # copiar (sem prompt)
├── move /Y src dst                     # mover (sem prompt)
├── del /F /Q arquivo                   # deletar arquivo
├── rmdir /S /Q pasta                   # deletar pasta
├── mkdir caminho\novo                  # criar diretório
├── cd /d D:\caminho                    # mudar drive+dir
└── where programa                      # encontrar executável

FLAGS OBRIGATORIAS (evitar prompts):
├── copy /Y       # sobrescrever sem perguntar
├── move /Y       # sobrescrever sem perguntar
├── del /F /Q     # force + quiet
└── rmdir /S /Q   # recursive + quiet
```

### ⚠️ REGRAS CRITICAS - ERROS COMUNS A EVITAR

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  FACTORY CLI USA POWERSHELL - NAO CMD!                                      │
│  Operadores CMD (&, &&, ||, 2>nul) NAO FUNCIONAM DIRETAMENTE               │
└─────────────────────────────────────────────────────────────────────────────┘

❌ NUNCA FAZER (vai falhar):
├── mkdir pasta & move arquivo       # & nao funciona em PS
├── comando 2>nul                    # redirecionador CMD
├── cmd1 && cmd2                     # && nao funciona em PS
├── cmd /c "mkdir x 2>nul & move y"  # sequencia complexa falha
└── Multiplos comandos em uma linha com operadores CMD

✅ SEMPRE FAZER (correto):
├── Um comando por Execute call
├── Usar ferramentas nativas (Read, Create, Edit, LS, Glob, Grep)
├── Para sequencias: fazer chamadas Execute separadas
└── Para ignorar erros: usar -ErrorAction SilentlyContinue em PS

EXEMPLOS CORRETOS:

# Criar pasta (ignorar se existe):
New-Item -ItemType Directory -Path "pasta" -Force

# Mover arquivo:  
Move-Item -Path "origem" -Destination "destino" -Force

# Copiar:
Copy-Item -Path "origem" -Destination "destino" -Force

# Deletar arquivo/pasta:
Remove-Item -Path "alvo" -Recurse -Force -ErrorAction SilentlyContinue

# Se PRECISA usar CMD (evitar quando possivel):
cmd /c "comando_simples"           # OK: comando unico
cmd /c "mkdir pasta"               # OK
cmd /c "move /Y src dst"           # OK: move simples

# NUNCA encadear com & ou && dentro de cmd /c:
# cmd /c "mkdir x & move y"        # FALHA!
```

### PREFERIR FERRAMENTAS FACTORY

```
Em vez de comandos shell, usar:

| Preciso de...        | Usar ferramenta  | NAO usar          |
|----------------------|------------------|-------------------|
| Criar arquivo        | Create tool      | echo > arquivo    |
| Ler arquivo          | Read tool        | type, cat         |
| Editar arquivo       | Edit tool        | sed, awk          |
| Listar diretorio     | LS tool          | dir, ls           |
| Buscar arquivos      | Glob tool        | dir /s, find      |
| Buscar texto         | Grep tool        | findstr, grep     |
| Criar pasta          | mkdir simples    | mkdir & outros    |
| Mover/copiar         | 1 comando por vez| sequencias        |

REGRA: Se pode fazer com ferramenta Factory, NAO use shell.
```

### SEQUENCIAS DE OPERACOES

```
ERRADO - Tudo em um comando:
Execute: mkdir pasta & move arq1 pasta & move arq2 pasta

CERTO - Comandos separados:
Execute #1: New-Item -ItemType Directory -Path "pasta" -Force
Execute #2: Move-Item -Path "arq1" -Destination "pasta" -Force  
Execute #3: Move-Item -Path "arq2" -Destination "pasta" -Force

Ou melhor ainda - usar Python/script se for complexo.
```

```
NUNCA USAR (nao existem no Windows):
├── grep, find, ls, cat, rm, touch, which, python3
└── && ou & diretamente no PowerShell

ENCODING UTF-8:
└── [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

---

## 10. QUICK ACTIONS

| Situacao | Acao |
|----------|------|
| Preciso implementar X | Check PRD → FORGE implementa |
| Preciso pesquisar X | ARGUS /pesquisar |
| Preciso validar backtest | ORACLE /go-nogo |
| Preciso calcular lot | SENTINEL /lot [sl] (considera trailing DD + tempo) |
| Problema complexo | sequential-thinking (5+ thoughts) |
| Duvida de sintaxe MQL5 | RAG query em .rag-db/docs |

---

*Skills especializadas tem conhecimento profundo.*
*Referencia tecnica em DOCS/CLAUDE_REFERENCE.md*
*Especificacao completa em DOCS/prd.md*
