# DOCS REORGANIZATION PLAN

**Data**: 2025-11-30  
**Status**: APROVADO PARA EXECUÇÃO  
**Objetivo**: Organização definitiva que todos os agentes conhecem

---

## PROBLEMA ATUAL

```
CAOS IDENTIFICADO:
├── 43 arquivos na raiz (sem categorização)
├── ~5000+ arquivos em SCRAPED/ e RESEARCH/repos/
├── Múltiplos "planos" superseded
├── Múltiplos "prompts" obsoletos
├── Repos de código clonados em DOCS (lugar errado)
├── Nenhuma convenção de nomenclatura
├── Agentes não sabem onde salvar
└── Fases do plano não têm pastas dedicadas
```

---

## NOVA ESTRUTURA

```
DOCS/
│
├── _INDEX.md                         # MASTER INDEX - explica tudo
│
├── _ARCHIVE/                         # 🗄️ COLD STORAGE
│   ├── LEGACY/                       # Docs superseded
│   ├── BOOKS/                        # PDFs, material de referência
│   └── OLD_PROMPTS/                  # Prompts antigos
│
├── 00_PROJECT/                       # 📋 Project-level
│   ├── prd.md                        # Product Requirements Document
│   ├── CHANGELOG.md                  # Histórico de versões
│   └── ARCHITECTURE.md               # Arquitetura do sistema
│
├── 01_AGENTS/                        # 🤖 Sistema de Agentes
│   ├── TEAM_SPECIFICATION.md         # Especificação do time
│   ├── PARTY_MODE/                   # Sessões Party Mode
│   │   └── SESSION_NNN_DATE.md
│   └── BACKUPS/                      # Backups de skills
│
├── 02_IMPLEMENTATION/                # 🚀 Implementação
│   ├── PLAN_v1.md                    # Plano atual
│   ├── PROGRESS.md                   # Tracker de progresso
│   └── PHASES/                       # Deliverables por fase
│       ├── PHASE_0_AUDIT/
│       │   └── AUDIT_MQL5.md
│       ├── PHASE_1_DATA/
│       │   └── DATA_QUALITY_REPORT.md
│       ├── PHASE_2_VALIDATION/
│       │   ├── WFA_REPORT.md
│       │   ├── MONTECARLO_REPORT.md
│       │   └── GO_NOGO_REPORT.md
│       ├── PHASE_3_ML/
│       │   ├── MODEL_ARCHITECTURE.md
│       │   └── ML_VALIDATION_REPORT.md
│       ├── PHASE_4_INTEGRATION/
│       │   └── INTEGRATION_TEST_REPORT.md
│       ├── PHASE_5_HARDENING/
│       │   └── CONTINGENCY_PLAN.md
│       └── PHASE_6_PAPER/
│           └── PAPER_TRADING_FINAL.md
│
├── 03_RESEARCH/                      # 🔍 Pesquisa (ARGUS)
│   ├── PAPERS/                       # Resumos de papers
│   │   └── YYYYMMDD_PAPER_TITLE.md
│   ├── FINDINGS/                     # Descobertas de pesquisa
│   │   └── TOPIC_FINDING.md
│   ├── CITATIONS.md                  # Tracker de citações
│   └── REPOS/                        # Links para repos externos
│       └── REPO_INDEX.md             # (não clonar, só referenciar)
│
├── 04_REPORTS/                       # 📊 Relatórios (ORACLE)
│   ├── BACKTESTS/                    # Resultados de backtest
│   │   └── YYYYMMDD_BACKTEST_NAME.md
│   ├── VALIDATION/                   # WFA, Monte Carlo
│   │   └── YYYYMMDD_VALIDATION_TYPE.md
│   └── DECISIONS/                    # GO/NO-GO decisions
│       └── YYYYMMDD_DECISION.md
│
├── 05_GUIDES/                        # 📚 Guias
│   ├── SETUP/                        # Configuração
│   │   ├── MT5_SETUP.md
│   │   └── RAG_SETUP.md
│   ├── USAGE/                        # Uso
│   │   └── BACKTEST_GUIDE.md
│   └── TROUBLESHOOTING/              # Resolução de problemas
│
└── 06_REFERENCE/                     # 📖 Referência Técnica
    ├── CLAUDE_REFERENCE.md           # Referência para Claude
    ├── MQL5/                         # MQL5 específico
    ├── PYTHON/                       # Python específico
    └── INTEGRATIONS/                 # MCPs, APIs
        ├── MCP_INDEX.md
        └── MCP_RECOMMENDATIONS.md
```

---

## AGENT → FOLDER MAPPING

### Onde Cada Agente Salva

| Agente | Pasta Principal | Subpastas |
|--------|-----------------|-----------|
| 🔥 CRUCIBLE | `03_RESEARCH/FINDINGS/` | Strategy findings |
| 🛡️ SENTINEL | `04_REPORTS/DECISIONS/` | Risk assessments |
| ⚒️ FORGE | `02_IMPLEMENTATION/PHASES/` | Audit, code docs |
| 🔮 ORACLE | `04_REPORTS/` | Backtests, Validation, Decisions |
| 🔍 ARGUS | `03_RESEARCH/` | Papers, Findings, Citations |
| ALL | `02_IMPLEMENTATION/PROGRESS.md` | Update progress |

### Tabela Rápida por Tipo de Arquivo

| Tipo de Arquivo | Pasta | Naming Convention |
|-----------------|-------|-------------------|
| Backtest result | `04_REPORTS/BACKTESTS/` | `YYYYMMDD_BACKTEST_NAME.md` |
| WFA report | `04_REPORTS/VALIDATION/` | `YYYYMMDD_WFA_REPORT.md` |
| Monte Carlo | `04_REPORTS/VALIDATION/` | `YYYYMMDD_MC_REPORT.md` |
| GO/NO-GO | `04_REPORTS/DECISIONS/` | `YYYYMMDD_GO_NOGO.md` |
| Research finding | `03_RESEARCH/FINDINGS/` | `TOPIC_FINDING.md` |
| Paper summary | `03_RESEARCH/PAPERS/` | `YYYYMMDD_AUTHOR_TITLE.md` |
| Phase deliverable | `02_IMPLEMENTATION/PHASES/PHASE_N/` | Match plan name |
| Setup guide | `05_GUIDES/SETUP/` | `TOOL_SETUP.md` |
| Party Mode session | `01_AGENTS/PARTY_MODE/` | `SESSION_NNN_YYYY-MM-DD.md` |

---

## MIGRATION PLAN

### Step 1: Create New Structure
```bash
# Criar pastas
mkdir DOCS\_ARCHIVE
mkdir DOCS\_ARCHIVE\LEGACY
mkdir DOCS\_ARCHIVE\BOOKS
mkdir DOCS\_ARCHIVE\OLD_PROMPTS
mkdir DOCS\00_PROJECT
mkdir DOCS\01_AGENTS
mkdir DOCS\01_AGENTS\BACKUPS
mkdir DOCS\02_IMPLEMENTATION
mkdir DOCS\02_IMPLEMENTATION\PHASES
mkdir DOCS\02_IMPLEMENTATION\PHASES\PHASE_0_AUDIT
mkdir DOCS\02_IMPLEMENTATION\PHASES\PHASE_1_DATA
mkdir DOCS\02_IMPLEMENTATION\PHASES\PHASE_2_VALIDATION
mkdir DOCS\02_IMPLEMENTATION\PHASES\PHASE_3_ML
mkdir DOCS\02_IMPLEMENTATION\PHASES\PHASE_4_INTEGRATION
mkdir DOCS\02_IMPLEMENTATION\PHASES\PHASE_5_HARDENING
mkdir DOCS\02_IMPLEMENTATION\PHASES\PHASE_6_PAPER
mkdir DOCS\03_RESEARCH
mkdir DOCS\03_RESEARCH\PAPERS
mkdir DOCS\03_RESEARCH\FINDINGS
mkdir DOCS\03_RESEARCH\REPOS
mkdir DOCS\04_REPORTS
mkdir DOCS\04_REPORTS\BACKTESTS
mkdir DOCS\04_REPORTS\VALIDATION
mkdir DOCS\04_REPORTS\DECISIONS
mkdir DOCS\05_GUIDES
mkdir DOCS\05_GUIDES\SETUP
mkdir DOCS\05_GUIDES\USAGE
mkdir DOCS\05_GUIDES\TROUBLESHOOTING
mkdir DOCS\06_REFERENCE
mkdir DOCS\06_REFERENCE\MQL5
mkdir DOCS\06_REFERENCE\PYTHON
mkdir DOCS\06_REFERENCE\INTEGRATIONS
```

### Step 2: Move Files to New Locations

#### Keep & Move (Active Files)
| From | To |
|------|-----|
| `IMPLEMENTATION_PLAN_v1.md` | `02_IMPLEMENTATION/PLAN_v1.md` |
| `AUDIT_MQL5.md` | `02_IMPLEMENTATION/PHASES/PHASE_0_AUDIT/` |
| `AGENT_TEAM_SPECIFICATION.md` | `01_AGENTS/TEAM_SPECIFICATION.md` |
| `AGENTS_BACKUP_FULL.md` | `01_AGENTS/BACKUPS/` |
| `PARTY_MODE/` | `01_AGENTS/PARTY_MODE/` |
| `CLAUDE_REFERENCE.md` | `06_REFERENCE/` |
| `MCP_INDEX.md` | `06_REFERENCE/INTEGRATIONS/` |
| `MCP_RECOMMENDATIONS.md` | `06_REFERENCE/INTEGRATIONS/` |
| `RAG_SETUP_GUIDE.md` | `05_GUIDES/SETUP/` |
| `GUIA_BACKTEST_DEPLOY.md` | `05_GUIDES/USAGE/BACKTEST_GUIDE.md` |
| `ML_TRADING_KNOWLEDGE_BASE.md` | `03_RESEARCH/FINDINGS/` |
| `ORDERFLOW_FOOTPRINT_RESEARCH.md` | `03_RESEARCH/FINDINGS/` |

#### Archive (Superseded Files)
| File | Reason |
|------|--------|
| `MASTER_PLAN_EA_SCALPER_XAUUSD.md` | Superseded by PLAN_v1 |
| `IMPLEMENTATION_ROADMAP.md` | Superseded by PLAN_v1 |
| `PLANO_IMPLEMENTACAO_XAUUSD.md` | Superseded by PLAN_v1 |
| `FUTURE_IMPLEMENTATIONS.md` | Superseded by PLAN_v1 |
| `CHATGPT_SYSTEM_PROMPT*.md` | Não usado mais |
| `GPT5_PRO_*.md` | Legacy |
| `PROMPT_NOVA_SESSAO.md` | Obsoleto |
| `SYSTEM_PROMPT_ULTRA_COMPACT.txt` | Obsoleto |
| `ANALISE_PROFUNDA_PROJETO.md` | Superseded |
| `NETWORK_ANALYSIS_REPORT.md` | Legacy |
| `CRUCIBLE_OPTIMIZATION_PLAN.md` | Implemented in skill |
| `KNOWLEDGE_BASE.md` | Merged into agents |
| `ML_ARCHITECTURE.md` | Will be recreated in Phase 3 |
| `MULTI_STRATEGY_NEWS_TRADING_SPEC.md` | Legacy spec |
| `PROPOSTA_SUBAGENTES_ESPECIALIZADOS.md` | Implemented |
| `PROJECT_ORGANIZATION_ANALYSIS.md` | Superseded |
| `PROJECT_STRUCTURE_FINAL.md` | Superseded |
| `README_INDICES.md` | Legacy |
| `SINGULARITY_STRATEGY_BLUEPRINT_v3.0.md` | Legacy |
| `SUMMARY.md` | Legacy |
| `MCP_RESEARCH_*.md` | Merged into MCP_INDEX |

#### Special Handling (Large Folders)
| Folder | Action | Reason |
|--------|--------|--------|
| `SCRAPED/` | DELETE or move outside DOCS | ~4000 files, in RAG already |
| `RESEARCH/repos/` | DELETE or move outside DOCS | Full repos, wrong place |
| `BOOKS/` | Move to `_ARCHIVE/BOOKS/` | Reference only |
| `_COLD_STORAGE/` | Merge into `_ARCHIVE/` | Consolidate |
| `Legacy/` | Merge into `_ARCHIVE/LEGACY/` | Consolidate |
| `Docs_EA/` | Review and archive | Probably legacy |

### Step 3: Create _INDEX.md

Ver próxima seção.

### Step 4: Update AGENTS.md

Adicionar seção sobre estrutura de DOCS.

### Step 5: Verify & Cleanup

```bash
# Verificar estrutura
dir /s DOCS\*.md | find /c ".md"

# Remover pastas vazias
# (manual review first)
```

---

## _INDEX.md CONTENT

```markdown
# DOCS INDEX

Last updated: YYYY-MM-DD

## Quick Navigation

| Preciso de... | Vá para |
|---------------|---------|
| Plano de implementação | `02_IMPLEMENTATION/PLAN_v1.md` |
| Progresso atual | `02_IMPLEMENTATION/PROGRESS.md` |
| Deliverables de fase | `02_IMPLEMENTATION/PHASES/PHASE_N/` |
| Relatórios de backtest | `04_REPORTS/BACKTESTS/` |
| Validação (WFA/MC) | `04_REPORTS/VALIDATION/` |
| Pesquisa | `03_RESEARCH/` |
| Guias de setup | `05_GUIDES/SETUP/` |
| Referência técnica | `06_REFERENCE/` |
| Arquivos antigos | `_ARCHIVE/` |

## Agent Ownership

| Agent | Owns | Creates |
|-------|------|---------|
| CRUCIBLE | Strategy | `03_RESEARCH/FINDINGS/` |
| SENTINEL | Risk | `04_REPORTS/DECISIONS/` |
| FORGE | Code | `02_IMPLEMENTATION/PHASES/`, `05_GUIDES/` |
| ORACLE | Validation | `04_REPORTS/` |
| ARGUS | Research | `03_RESEARCH/` |

## Naming Conventions

- **Reports**: `YYYYMMDD_TYPE_NAME.md`
- **Findings**: `TOPIC_FINDING.md`
- **Guides**: `TOOL_ACTION.md`
- **Sessions**: `SESSION_NNN_YYYY-MM-DD.md`

## Folder Structure

[Ver diagrama na seção NOVA ESTRUTURA]
```

---

## DECISION NEEDED FROM USER

### Sobre SCRAPED/ e RESEARCH/repos/

Essas pastas têm milhares de arquivos:
- `SCRAPED/`: ~4000+ arquivos (já estão no RAG)
- `RESEARCH/repos/`: Repos completos clonados (tensortrade, ml_for_trading)

**Opções**:

| Opção | Ação | Prós | Contras |
|-------|------|------|---------|
| A | DELETAR | Limpa, já no RAG | Perde material |
| B | MOVER para fora de DOCS | Mantém, organizado | Ainda ocupa espaço |
| C | COMPRIMIR em .zip | Backup compacto | Menos acessível |

**Minha recomendação**: Opção A (DELETAR) para SCRAPED se já está no RAG. Opção B para RESEARCH/repos (mover para `/data/external_repos/`).

---

## EXECUTION CHECKLIST

- [ ] User aprova estrutura
- [ ] User decide sobre SCRAPED/RESEARCH
- [ ] Criar nova estrutura de pastas
- [ ] Mover arquivos ativos
- [ ] Arquivar arquivos superseded
- [ ] Criar _INDEX.md
- [ ] Atualizar AGENTS.md com seção DOCS
- [ ] Deletar/mover pastas grandes
- [ ] Commit final

---

## BENEFÍCIOS DA NOVA ESTRUTURA

1. **Self-documenting**: Nomes explicam conteúdo
2. **Agent-aware**: Cada agente sabe onde salvar
3. **Phase-aware**: Deliverables têm casa
4. **Scalable**: Fácil adicionar novas fases/agentes
5. **Clean**: Root só tem _INDEX.md
6. **Traceable**: Arquivos têm data no nome
7. **Archived**: Legacy separado do ativo

---

*"Um lugar para cada coisa, cada coisa em seu lugar."*
