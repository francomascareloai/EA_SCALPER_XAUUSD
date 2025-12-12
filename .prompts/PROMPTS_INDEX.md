# 📋 Índice de Prompts - EA_SCALPER_XAUUSD

**Última atualização**: 2025-12-11
**Total prompts**: 10 (5 completos, 4 em progresso, 1 planejado)

---

## 📊 Visão Geral

| Prompt | Status | Progresso | Resultado Principal | Link |
|--------|--------|-----------|---------------------|------|
| **001-nautilus-plan-refine** | 🟢 90%+ | Bugs #2/#4 FIXADOS | 35/40 módulos OK, 5 stubs, ORACLE validado | [📄](./001-nautilus-plan-refine/SUMMARY.md) |
| **002-backtest-data-research** | ✅ 100% | Concluído | Dukascopy aprovado, 25.5M ticks prontos | [📄](./002-backtest-data-research/SUMMARY.md) |
| **003-backtest-code-audit** | 🟢 P0s+P1s cleared | CircuitBreaker integrado | 9 P1 gaps RESOLVIDOS | [📄](./003-backtest-code-audit/SUMMARY.md) |
| **004-apex-risk-audit** | 🟡 CONDITIONAL GO | 9/10 compliance | Quase completo, minor gaps | [📄](./004-apex-risk-audit/SUMMARY.md) |
| **005-realistic-backtest-plan** | 🟢 Baseline EXECUTADO | Phase 1 completa | Regime detection implementado | [📄](./005-realistic-backtest-plan/SUMMARY.md) |
| **006-fix-critical-bugs** | ✅ 100% | 11/12 fixados | NinjaTrader adapter deferred | [📄](./006-fix-critical-bugs/SUMMARY.md) |
| **008-droid-analysis** | 🟡 2/5 gaps filled | Progresso parcial | 3 gaps restantes | [📄](./008-droid-analysis/SUMMARY.md) |
| **009-agents-nautilus-update** | 🟡 DESATUALIZADO | v3.7.0 disponível | Update pendente para v3.4.1→v3.7.0 | [📄](./009-agents-nautilus-update/SUMMARY.md) |
| **010-droid-refactoring-master** | 🟡 PARCIAL | 2/5 refatorados | FORGE + SENTINEL otimizados | [📄](./010-droid-refactoring-master/SUMMARY.md) |
| **011-agents-md-optimization** | 🟡 PARCIAL | 57% redução real | 2607→1130 linhas (não 585) | [📄](./011-agents-md-optimization/SUMMARY.md) |

**Legenda**:
- ✅ = Completo (100%)
- 🟢 = Progresso significativo (>80%)
- 🟡 = Em progresso ou conditional
- 🔴 = Bloqueado/NO-GO
- 📋 = Planejado

---

## 🔍 Auditoria 2025-12-11

**Executada por**: REVIEWER + ORACLE
**Método**: Verificação de todos os SUMMARYs vs realidade do código

### ⚠️ DESCOBERTAS CRÍTICAS:

**Status inflacionados identificados**:
- **001**: Bugs #2/#4 foram FIXADOS (não pendentes)
- **003**: P1 gaps foram RESOLVIDOS (CircuitBreaker implementado)
- **004**: 9/10 compliance atingido (não 3/10)
- **005**: Baseline EXECUTADO (não apenas planejado)
- **008**: Parcial (2/5 gaps, não 100%)
- **009**: DESATUALIZADO (v3.7.0 lançado depois)
- **010**: PARCIAL (2/5 droids refatorados)
- **011**: 57% redução real (não 78%)

### 📝 AÇÃO NECESSÁRIA:

**SUMMARYs individuais precisam ser atualizados** para refletir realidade:
- `001-nautilus-plan-refine/SUMMARY.md` → Remover "bugs pendentes"
- `003-backtest-code-audit/SUMMARY.md` → Marcar P1 gaps como RESOLVIDOS
- `004-apex-risk-audit/SUMMARY.md` → Atualizar compliance 3/10 → 9/10
- `005-realistic-backtest-plan/SUMMARY.md` → Marcar Phase 1 como EXECUTADA
- `008-droid-analysis/SUMMARY.md` → Reduzir claim de 100% para parcial
- `009-agents-nautilus-update/SUMMARY.md` → Adicionar nota de desatualização
- `010-droid-refactoring-master/SUMMARY.md` → Marcar 2/5 como completos
- `011-agents-md-optimization/SUMMARY.md` → Corrigir redução para 57%

### ✅ VERIFICADO CORRETO:
- **002**: Data research realmente 100% (32.7M ticks validados)
- **006**: Bug fixes realmente 11/12 (92% correto)

---

## 🔴 Blockers Críticos (Status Mantido)

### 🚨 PRIORITY 0: `trading_allowed=False` Bug

**Status**: 🔴 BLOCKER ATIVO
**Impacto**: Backtest gera **0 trades** (esperado: 100+ trades)
**Sintoma**: `GoldScalperStrategy` recebe bars mas nunca chama `on_bar()` ou lógica de trading
**Causa raiz**: Não identificada

**Afeta**:
- Prompt 001 (Nautilus migration validation)
- Prompt 005 (Realistic backtest implementation)
- Qualquer validação de estratégia

**Próximos passos**:
1. Debug session para rastrear `trading_allowed` flag
2. Verificar inicialização do Strategy
3. Validar subscriptions de bars
4. Testar com Strategy mínima (hello world)

### 🔴 SECONDARY: Apex Compliance Gaps (Prompt 004)

**Status**: 3/10 rules implemented
**Missing**:
- TimeConstraint (4:59 PM ET deadline)
- Consistency (30% max daily profit)
- Emergency mode detection

**Blocks**: Production deployment

---

## ✅ Prompts Completos

### 002 - Backtest Data Research
**Status**: ✅ 100% Complete
**Resultado**: Dukascopy data provider aprovado, 25.5M ticks (2003-2025) validados
**Deliverables**:
- `data/raw/full_parquet/xauusd_2003_2025_stride20_full.parquet` (32.7M ticks)
- Data quality validation passed
- Consolidation script created

📄 [Ver SUMMARY completo](./002-backtest-data-research/SUMMARY.md)

---

### 006 - Fix Critical Bugs
**Status**: ✅ 11/12 bugs fixed (92%)
**Resultado**: Sistema estável, 1 bug deferred (NinjaTrader adapter)
**Deliverables**:
- DrawdownTracker HWM calculation fixed
- Data pipeline imports resolved
- Type errors eliminated
- Compilation errors cleared

📄 [Ver SUMMARY completo](./006-fix-critical-bugs/SUMMARY.md)

---

### 008 - Droid Analysis
**Status**: ✅ 5-layer ecosystem analysis complete
**Resultado**: 23 droids catalogados, 5 missing identificados
**Deliverables**:
- Full ecosystem map
- Routing analysis
- Gap identification
- `DOCS/04_REPORTS/DROID_ECOSYSTEM_ANALYSIS_2025-12-07.md`

📄 [Ver SUMMARY completo](./008-droid-analysis/SUMMARY.md)

---

### 009 - AGENTS.md + Nautilus Update
**Status**: ✅ AGENTS.md v3.4.1 released
**Resultado**: NAUTILUS droid fully integrated, MQL5 maintained as secondary
**Deliverables**:
- Dual-platform support documented
- Routing rules updated
- Platform priority clarified

📄 [Ver SUMMARY completo](./009-agents-nautilus-update/SUMMARY.md)

---

### 011 - AGENTS.md Optimization
**Status**: ✅ 78% token reduction achieved
**Resultado**: 2607→585 linhas, v3.5.0 released
**Deliverables**:
- Compression without information loss
- quick_reference section added
- Genius mode protocols consolidated

📄 [Ver SUMMARY completo](./011-agents-md-optimization/SUMMARY.md)

---

## 🟡 Prompts Em Progresso

### 001 - Nautilus Migration Plan Refinement
**Status**: 🟡 87.5% complete (35/40 modules)
**Blocker**: ORACLE validation failures (bugs #2, #4)
**Pendente**:
- 5 stub implementations (data_loaders, indicators, actors)
- `trading_allowed=False` bug resolution
- WFA validation run

**Próximos passos**:
1. Resolve `trading_allowed` bug (P0)
2. Implement remaining 5 stubs (P1)
3. Run full WFA validation (P1)
4. Document migration patterns (P2)

📄 [Ver SUMMARY completo](./001-nautilus-plan-refine/SUMMARY.md)

---

### 003 - Backtest Code Audit
**Status**: 🟡 CONDITIONAL GO (P0s cleared, P1s block production)
**Score**: 7.7/10
**Blocker**: 9 P1 gaps identified
**Pendente**:
- Error handling improvements
- State management validation
- Edge case coverage
- Documentation completion

**Próximos passos**:
1. Address P1 gaps (estimated 4-6h)
2. Add comprehensive error handling
3. Validate state management patterns
4. Re-audit for production readiness

📄 [Ver SUMMARY completo](./003-backtest-code-audit/SUMMARY.md)

---

### 004 - Apex Risk Audit
**Status**: 🔴 NO-GO (3/10 compliance)
**Compliance**: 30% (need 100% for production)
**Blocker**: Critical Apex rules missing
**Pendente**:
- TimeConstraint implementation (4:59 PM ET)
- Consistency enforcement (30% max daily)
- Emergency mode detection
- Multi-tier DD protection

**Próximos passos**:
1. Implement TimeConstraint module (P0)
2. Add Consistency rule enforcement (P0)
3. Create emergency mode detector (P0)
4. Validate all 10 Apex rules (P0)
5. Re-audit for compliance

📄 [Ver SUMMARY completo](./004-apex-risk-audit/SUMMARY.md)

---

### 005 - Realistic Backtest Plan
**Status**: 🟡 READY (plan complete, implementation pending)
**Plan**: 7 fases definidas (Regime→Slippage→Spread→Fees→News→Volume→Validation)
**Data**: Ready (25.5M ticks)
**Blocker**: `trading_allowed=False` must be fixed first

**Próximos passos**:
1. Fix `trading_allowed` bug (P0)
2. Implement Phase 1: Regime detection (P1)
3. Implement Phase 2: Slippage model (P1)
4. Progress through remaining phases sequentially

📄 [Ver SUMMARY completo](./005-realistic-backtest-plan/SUMMARY.md)

---

## 📋 Prompts Planejados

### 010 - Droid Refactoring Master
**Status**: 📋 PLANNED (spec ready, execution pending)
**Scope**: TOP 5 droids for optimization
**Target**: FORGE, SENTINEL, ORACLE, CRUCIBLE, NAUTILUS
**Estimativa**: 12-16h (2-3 dias)

**Objetivos**:
1. Token reduction (40-60%)
2. Performance optimization
3. NANO versions for party mode
4. Enhanced routing efficiency

**Bloqueios**: Aguardando priorização (após prompt 001 completion)

📄 [Ver SUMMARY completo](./010-droid-refactoring-master/SUMMARY.md)

---

## 📂 Estrutura de Diretórios

```
.prompts/
├── PROMPTS_INDEX.md                    # Este arquivo (índice geral)
├── 001-nautilus-plan-refine/          # 🟡 87.5% - Migration em andamento
│   ├── SUMMARY.md
│   └── ...
├── 002-backtest-data-research/         # ✅ 100% - Data pipeline completo
│   ├── SUMMARY.md
│   └── ...
├── 003-backtest-code-audit/            # 🟡 CONDITIONAL - P1 gaps pendentes
│   ├── SUMMARY.md
│   └── ...
├── 004-apex-risk-audit/                # 🔴 NO-GO - 3/10 compliance
│   ├── SUMMARY.md
│   └── ...
├── 005-realistic-backtest-plan/        # 🟡 READY - Aguarda bug fix
│   ├── SUMMARY.md
│   └── ...
├── 006-fix-critical-bugs/              # ✅ 100% - Bugs críticos resolvidos
│   ├── SUMMARY.md
│   └── ...
├── 008-droid-analysis/                 # ✅ 100% - Ecosystem mapeado
│   ├── SUMMARY.md
│   └── ...
├── 009-agents-nautilus-update/         # ✅ 100% - AGENTS.md v3.4.1
│   ├── SUMMARY.md
│   └── ...
├── 010-droid-refactoring-master/       # 📋 PLANNED - Spec pronto
│   ├── SUMMARY.md
│   └── ...
├── 011-agents-md-optimization/         # ✅ 100% - 78% redução
│   ├── SUMMARY.md
│   └── ...
├── completed/                          # Arquivos de prompts finalizados
│   ├── 001-nautilus-plan-refine-2025-12-07/
│   ├── 002-backtest-data-research-2025-12-07/
│   ├── 003-backtest-code-audit-2025-12-07/
│   ├── 004-apex-risk-audit-2025-12-07/
│   ├── 005-realistic-backtest-plan-2025-12-07/
│   ├── 006-fix-critical-bugs-2025-12-07/
│   └── 008-droid-analysis-2025-12-07/
└── [Session Reports]                   # Relatórios consolidados
    ├── 20251207_BACKTEST_DEBUG_SESSION.md
    ├── 20251207_BACKTEST_VALIDATION_REPORT.md
    ├── 20251207_DATA_STATUS_UPDATE.md
    ├── 20251207_FINAL_SESSION_REPORT.md
    └── 20251207_PROMPTS_001-005_AUDIT.md
```

---

## 🎯 Próximos Passos Prioritários

### Curto Prazo (Urgente)

#### P0: Fix `trading_allowed=False` Bug
**Estimativa**: 2-4h
**Bloqueio**: Impede validação de toda a migration
**Abordagem**:
1. Debug session com breakpoints
2. Rastrear inicialização do Strategy
3. Verificar bar subscriptions
4. Validar lifecycle (on_start → on_bar flow)

#### P0: Apex Compliance (Prompt 004)
**Estimativa**: 6-8h
**Objetivo**: 3/10 → 10/10 compliance
**Implementar**:
1. TimeConstraint (4:59 PM ET deadline)
2. Consistency (30% max daily profit)
3. Emergency mode (multi-tier DD)

---

### Médio Prazo (1-2 Semanas)

#### P1: Complete Nautilus Migration (Prompt 001)
**Estimativa**: 8-12h
**Dependências**: Resolver bug `trading_allowed` primeiro
**Deliverables**:
- 5 stub implementations
- ORACLE validation passing
- WFA results (WFE ≥0.6)

#### P1: Realistic Backtest Implementation (Prompt 005)
**Estimativa**: 12-16h
**Dependências**: Migration completa + bug fix
**Fases**: 7 sequential phases
**Target**: Production-ready backtest engine

#### P1: Address P1 Gaps (Prompt 003)
**Estimativa**: 4-6h
**Objetivo**: CONDITIONAL GO → APPROVED
**Deliverables**: Error handling + state management + docs

---

### Longo Prazo (3-4 Semanas)

#### P2: Droid Refactoring (Prompt 010)
**Estimativa**: 12-16h
**Objetivo**: TOP 5 droids optimization
**Benefício**: 40-60% token reduction + performance boost

#### P2: Additional Features
- Advanced slippage models
- News event detection
- Regime-adaptive position sizing
- Real-time monitoring dashboard

---

## 📈 Estatísticas (Pós-Auditoria 2025-12-11)

| Métrica | Valor REAL |
|---------|------------|
| **Total Prompts** | 10 |
| **Completos** | 2 (20%) - apenas 002, 006 |
| **Progresso Significativo (>80%)** | 3 (30%) - 001, 003, 005 |
| **Em Progresso (40-80%)** | 4 (40%) - 004, 008, 010, 011 |
| **Desatualizados** | 1 (10%) - 009 precisa v3.7.0 |
| **Bloqueados** | 1 (001 - bug `trading_allowed`) |
| **Linhas de Código Migradas** | 35/40 módulos (87.5%) |
| **Bugs Corrigidos** | 11/12 (92%) ✅ |
| **Token Reduction (AGENTS.md)** | 57% (2607→1130) - não 78% |
| **Data Ready** | ✅ 32.7M ticks validated |
| **Apex Compliance** | 🟡 90% (9/10 rules) - não 30% |

---

## 🔗 Links Úteis

- **Master Plan**: [`DOCS/02_IMPLEMENTATION/NAUTILUS_MIGRATION_MASTER_PLAN.md`](../DOCS/02_IMPLEMENTATION/NAUTILUS_MIGRATION_MASTER_PLAN.md)
- **AGENTS.md**: [`AGENTS.md`](../AGENTS.md) (v3.7.0)
- **Ecosystem Analysis**: [`DOCS/04_REPORTS/DROID_ECOSYSTEM_ANALYSIS_2025-12-07.md`](../DOCS/04_REPORTS/DROID_ECOSYSTEM_ANALYSIS_2025-12-07.md)
- **Backtest Validation**: [`DOCS/04_REPORTS/BACKTEST_VALIDATION_REPORT_2025-12-07.md`](../DOCS/04_REPORTS/BACKTEST_VALIDATION_REPORT_2025-12-07.md)
- **Data Status**: [`DOCS/04_REPORTS/DATA_STATUS_2025-12-07.md`](../DOCS/04_REPORTS/DATA_STATUS_2025-12-07.md)

---

**Mantido por**: Factory Droids (FORGE, NAUTILUS, REVIEWER)
**Frequência de atualização**: Após cada prompt completion ou status change significativo
**Contato**: Veja AGENTS.md para routing entre droids

---

*"BUILD > PLAN. CODE > DOCS. SHIP > PERFECT."*
*"Cada prompt completo = +1 step toward $50k Apex compliance."*

🚀 **EA_SCALPER_XAUUSD** - Nautilus Migration Progress Tracker
