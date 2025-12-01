# EA_SCALPER_XAUUSD - PROGRESS TRACKER

**Última Atualização**: 2025-11-30 14:45  
**Versão do Plano**: 1.0  
**Status Geral**: Phase 1 COMPLETO → Pronto para Phase 2

---

## AGENT ROUTING POR PHASE

| Phase | Foco | Agente Principal | Agentes Suporte | Trigger |
|-------|------|------------------|-----------------|---------|
| **0 - AUDIT** | Código/Estrutura | ⚒️ **FORGE** | - | `/review`, `/codigo` |
| **1 - DATA + BASELINE** | Dados/Backtest básico | 🔮 **ORACLE** | ⚒️ FORGE (scripts) | `/backtest` |
| **2 - VALIDATION** | WFA/Monte Carlo/GO-NOGO | 🔮 **ORACLE** | 🛡️ SENTINEL (risco) | `/wfa`, `/go-nogo` |
| **3 - ML/ONNX** | Treinar modelo | ⚒️ **FORGE** | 🔍 ARGUS (research) | `/codigo` |
| **4 - EA INTEGRATION** | Código/Testes | ⚒️ **FORGE** | 🔮 ORACLE (validar) | `/review` |
| **5 - HARDENING** | Risco/Monitoring | 🛡️ **SENTINEL** | ⚒️ FORGE (código) | `/risco` |
| **6 - PAPER TRADING** | Validação live | 🔮 **ORACLE** | 🛡️ SENTINEL (risco) | `/go-nogo` |

**Quando chamar cada agente:**
- ⚒️ **FORGE** → Codar, auditar, revisar código
- 🔮 **ORACLE** → Validar, backtest, estatísticas, WFA, Monte Carlo
- 🛡️ **SENTINEL** → Calcular risco, lot size, FTMO compliance
- 🔥 **CRUCIBLE** → Ajustar estratégia (se validação falhar)
- 🔍 **ARGUS** → Pesquisar papers, repos, ML research

---

## TIMELINE DE FASES

| Phase | Nome | Status | Data Início | Data Fim | Notas |
|-------|------|--------|-------------|----------|-------|
| 0 | AUDIT | ✅ COMPLETO | 2025-11-30 | 2025-11-30 | MQL5 85%, Python 90% |
| 1 | DATA + BASELINE | ✅ COMPLETO | 2025-11-30 | 2025-11-30 | Baseline: estratégia simples não funciona |
| 2 | VALIDATION | 🔄 PRÓXIMO | - | - | WFA + Monte Carlo |
| 3 | ML/ONNX MODEL | ⏳ PENDENTE | - | - | Modelos existem, precisam validação |
| 4 | EA INTEGRATION | ⏳ PENDENTE | - | - | - |
| 5 | HARDENING | ⏳ PENDENTE | - | - | - |
| 6 | PAPER TRADING | ⏳ PENDENTE | - | - | - |

---

## PHASE 0: AUDIT (✅ COMPLETO)

### Tasks

| Task | Descrição | Status | Deliverable |
|------|-----------|--------|-------------|
| 0.1 | Audit MQL5 Structure | ✅ COMPLETO | AUDIT_MQL5.md |
| 0.2 | Audit Python Agent Hub | ✅ COMPLETO | AUDIT_PYTHON.md |
| 0.3 | Create Gap Analysis | ✅ COMPLETO | GAP_ANALYSIS.md |

### Key Findings

**MQL5 (85% completo)**:
- ✅ 38 módulos mapeados
- ✅ EA v3.30 funcional
- ✅ FTMO compliance implementado
- ⚠️ ONNX models ausentes em MQL5/Models (precisa copiar)

**Python (90% completo)**:
- ✅ FastAPI v4.0 funcional
- ✅ 15 features ML implementadas
- ✅ Modelos ONNX treinados
- ✅ 40+ GB dados disponíveis
- ⚠️ End-to-end testing pendente

### Checkpoint Decision
**RESULTADO**: ✅ Pode prosseguir para Phase 1
- Nenhum gap crítico
- Dados já disponíveis
- Modelos existem (precisam validação)

---

## PHASE 1: DATA + BASELINE (🔄 EM PROGRESSO)

### Tasks

| Task | Descrição | Status | Deliverable |
|------|-----------|--------|-------------|
| 1.1 | Data Acquisition | ⏭️ SKIP | Dados Dukascopy já existem |
| 1.2 | Data Validation | ✅ COMPLETO | DATA_QUALITY_REPORT.md |
| 1.3 | Baseline Backtest | ✅ COMPLETO | BASELINE_METRICS.md |

### Task 1.2 - Data Validation (✅ COMPLETO)

**Ações realizadas:**
- Limpeza de arquivos duplicados/corrompidos (10 arquivos removidos)
- Export de dados Dukascopy via QuantDataManager (M5, M15, H1, H4)
- Validação automatizada via `scripts/validate_data.py`

**Resultados:**

| Arquivo | Rows | Período | Qualidade |
|---------|------|---------|-----------|
| Bars_2020-2025XAUUSD_ftmo-M5 | 419,195 | 2020-2025 | ✅ EXCELENTE |
| bars-2020-2025XAUUSD_ftmo-M15 | 139,738 | 2020-2025 | ✅ EXCELENTE |
| bars-2020-2025XAUUSD_ftmo-H1 | 34,951 | 2020-2025 | ✅ EXCELENTE |
| bars-2020-2025XAUUSD_ftmo-H4 | 9,138 | 2020-2025 | ✅ EXCELENTE |

**Checks passados:**
- ✅ Zero duplicatas
- ✅ Zero valores negativos
- ✅ OHLC sanity OK
- ✅ 100% volume coverage
- ✅ 94.5% completeness
- ✅ Gaps apenas em feriados (Natal, Páscoa, Ano Novo)

**Deliverable:** `DOCS/04_REPORTS/VALIDATION/DATA_QUALITY_REPORT.md`

### Data Inventory (Atualizado)

**OHLC Bars (para Python/ML):**

| Dataset | Tamanho | Período | Rows | Status |
|---------|---------|---------|------|--------|
| Bars M5 | 24 MB | 2020-2025 | 419k | ✅ Validado |
| Bars M15 | 8 MB | 2020-2025 | 140k | ✅ Validado |
| Bars H1 | 2 MB | 2020-2025 | 35k | ✅ Validado |
| Bars H4 | 533 KB | 2020-2025 | 9k | ✅ Validado |

**Tick Data (para MT5 Backtest):**

| Dataset | Tamanho | Período | Status |
|---------|---------|---------|--------|
| XAUUSD_ftmo_all_desde_2003.csv | 26 GB | 2003-2025 | ✅ Fonte primária |
| XAUUSD_ftmo_2020_ticks_dukascopy.csv | 12.7 GB | 2020-2025 | ✅ Fonte primária |
| xauusd-ticks-2024-2025_MT5.csv | 428 MB | 2024-2025 | ✅ Backup |

**Uso:**
- MT5 Backtest → Ticks Dukascopy (máxima precisão)
- Python ML/WFA → Barras OHLC (velocidade)

### Task 1.3 - Baseline Backtest (✅ COMPLETO)

**Estratégia testada:** MA Crossover (10/50) - estratégia simples sem ML

**Resultados:**

| Métrica | Valor | Target | Status |
|---------|-------|--------|--------|
| Total Trades | 9,915 | >= 100 | PASS |
| Win Rate | 31.4% | >= 45% | FAIL |
| Profit Factor | 0.97 | >= 1.3 | FAIL |
| Max Drawdown | 83.4% | <= 15% | FAIL |
| Sharpe Ratio | -0.48 | >= 0.5 | FAIL |
| P&L | -52.1% | > 0 | FAIL |

**Conclusão:** Baseline POBRE (1/6 critérios) - **ESPERADO!**

**Significado:**
- Estratégia simples NÃO funciona em XAUUSD M5
- Não há "alpha" óbvio em MA crossover
- ML/Regime detection são NECESSÁRIOS para edge
- Temos benchmark claro para medir valor do ML

**Deliverable:** `DOCS/04_REPORTS/BACKTESTS/BASELINE_METRICS.md`

---

## PHASE 1 CHECKPOINT: ✅ COMPLETO

**Decisão:** Prosseguir para Phase 2 (Validation Pipeline)

**Justificativa:**
- Dados validados e prontos
- Baseline estabelecido (referência para comparação)
- Confirmado que estratégia simples não funciona → ML necessário

---

## QUICK WINS

```
[x] Copiar direction_model_final.onnx → MQL5/Models/direction_model.onnx ✅
[x] Adicionar onnxruntime ao requirements.txt ✅
[ ] Testar Python Hub startup (python main.py) - próxima sessão
```

---

## MÉTRICAS DE PROGRESSO

```
Overall Progress: ██████░░░░░░░░░░░░░░ 25% (Phase 1.2 de 6)

Phase 0: ████████████████████ 100%
Phase 1: ██████████░░░░░░░░░░  50% (Task 1.2 done, 1.3 pending)
Phase 2: ░░░░░░░░░░░░░░░░░░░░   0%
Phase 3: ░░░░░░░░░░░░░░░░░░░░   0%
Phase 4: ░░░░░░░░░░░░░░░░░░░░   0%
Phase 5: ░░░░░░░░░░░░░░░░░░░░   0%
Phase 6: ░░░░░░░░░░░░░░░░░░░░   0%
```

---

## PRÓXIMA TAREFA

**Foco**: Task 1.3 - Baseline Backtest

**Objetivo**: Rodar backtest simples (sem ML) para estabelecer baseline de performance.

**Estratégia Baseline**:
- Entry: Cruzamento de MAs ou regra simples
- Exit: Fixed TP/SL
- Risk: 1% por trade
- Período: 2020-2024 (deixar 2025 para OOS)

---

*Atualizado em 2025-11-30 14:30 por ORACLE via Droid*
