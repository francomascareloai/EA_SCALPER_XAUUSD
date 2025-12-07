# 🔍 PROMPT: Análise Detalhada dos Droids (FASE 1)

## 📋 Objective

Executar **FASE 1** do plano de otimização dos droids: análise completa e mapeamento de redundâncias nos TOP 5 droids (NAUTILUS, ORACLE, FORGE, SENTINEL, RESEARCH-ANALYST-PRO) para preparar refatoração com herança hierárquica do AGENTS.md v3.4.

**Why it matters:** Eliminar 69% de duplicação (33,750 tokens) e criar sistema de herança que propaga melhorias do AGENTS.md automaticamente para todos os droids.

---

## 📁 Context

**Master Plan:**
@.factory/specs/2025-12-07-plano-otimiza-o-completa-dos-droids-token-efficiency-consistency.md

**Base Framework:**
@AGENTS.md (v3.4.0 com strategic_intelligence, genius_mode_templates, complexity_assessment)

**Droids to Analyze:**
- @.factory/droids/nautilus-trader-architect.md (53KB - maior)
- @.factory/droids/oracle-backtest-commander.md (38KB)
- @.factory/droids/forge-mql5-architect.md (37KB)
- @.factory/droids/sentinel-apex-guardian.md (37KB)
- @.factory/droids/research-analyst-pro.md (31KB)

**Current State:**
- Total: 196KB = 49,000 tokens
- Party Mode overhead: 61,700 tokens
- Redundância estimada: 70-80%

---

## 🎯 Requirements

### 1. Mapear Redundâncias (Linha-por-Linha)

Para cada droid dos TOP 5:

#### 1.1 Identificar Seções Duplicadas do AGENTS.md
- **Protocols genéricos**: NEVER/ALWAYS/MUST rules que já existem em `<strategic_intelligence>`
- **Genius templates**: Output formats, checklists que já existem em `<genius_mode_templates>`
- **Complexity assessment**: Rules de SIMPLE/MEDIUM/COMPLEX já em `<complexity_assessment>`
- **Error recovery**: Patterns já em `<error_recovery>`
- **Enforcement validation**: Checks já em `<enforcement_validation>`

#### 1.2 Extrair Conhecimento Único de Domínio

Para CADA droid, isolar **apenas** o que é especialização:

**NAUTILUS:**
- MQL5 → Python migration mappings (OnInit → on_start, OnTick → on_quote_tick, etc)
- Event-driven architecture patterns (MessageBus, Cache, Actor vs Strategy)
- Performance targets específicos (<1ms on_bar, <100μs on_tick)
- ParquetDataCatalog / BacktestNode setup

**ORACLE:**
- Statistical thresholds (WFE ≥0.6, DSR >0, PSR ≥0.85, MC_95th_DD<5%)
- Walk-Forward formulas (IS/OOS split, purged CV)
- Monte Carlo specifications (Block bootstrap, 5000 runs)
- GO/NO-GO gates (7-step checklist)

**FORGE:**
- Python/Nautilus-specific anti-patterns (circular imports, mutable defaults, blocking in handlers)
- Deep Debug protocol
- Context7 integration for NautilusTrader docs
- Test scaffolding templates (pytest fixtures)

**SENTINEL:**
- Apex Trading formulas (10% trailing DD from HWM, 4:59 PM ET deadline)
- Position sizing formulas (Kelly, time multiplier, DD awareness)
- Circuit breaker levels (WARNING/CAUTION/DANGER/BLOCKED)
- Recovery protocols específicos de Apex

**RESEARCH-ANALYST-PRO:**
- Multi-source triangulation methodology
- Source credibility rating system
- Confidence level frameworks
- Research log structure

#### 1.3 Definir Perguntas Extras por Droid

Cada droid mantém **3 additional_reflection_questions** específicas do domínio:

**Exemplo (NAUTILUS):**
```xml
<question id="18" category="nautilus_specific">
  Does this follow NautilusTrader event-driven patterns?
  (MessageBus for signals, cache for data access, no globals)
</question>
```

Identificar quais perguntas extras cada droid deve ter.

---

### 2. Criar REDUNDANCY_MAP.md Detalhado

Output: `.prompts/008-droid-analysis/REDUNDANCY_MAP.md`

**Estrutura:**
```markdown
# Redundancy Map - Droid Optimization

## Executive Summary
- Total redundancy: {XX}KB ({YY}% of TOP 5)
- Token savings potential: {ZZ,ZZZ} tokens
- Droids analyzed: 5 (NAUTILUS, ORACLE, FORGE, SENTINEL, RESEARCH)

## NAUTILUS (53KB → {target}KB)

### Redundâncias Identificadas

#### 1. Protocols Genéricos (lines XXX-YYY, ~{size}KB)
**Conteúdo duplicado:**
```
[Trecho do droid]
```

**Já existe em AGENTS.md:**
```
<strategic_intelligence>
  <mandatory_reflection_protocol>
    ...
```

**Ação:** REMOVE (herdar de AGENTS.md)

#### 2. [Próxima redundância...]

### Conhecimento Único a Manter

#### 1. Migration Mappings (lines XXX-YYY, ~{size}KB)
**Conteúdo:**
```
<mql5_to_python>
  <map from="OnInit()" to="Strategy.__init__() + on_start()"/>
```

**Ação:** MANTER (especialização Nautilus)

#### 2. [Próximo conhecimento único...]

### Additional Reflection Questions (3 perguntas)
1. [Question 18: ...]
2. [Question 19: ...]
3. [Question 20: ...]

### Savings
- Remove: {XX}KB
- Keep: {YY}KB
- Target size: {YY}KB (economia {ZZ}%)

---

## ORACLE (38KB → {target}KB)

[Mesma estrutura]

---

## FORGE (37KB → {target}KB)

[Mesma estrutura]

---

## SENTINEL (37KB → {target}KB)

[Mesma estrutura]

---

## RESEARCH-ANALYST-PRO (31KB → {target}KB)

[Mesma estrutura]

---

## Aggregate Analysis

### Total Savings
- Before: 196KB (49,000 tokens)
- After: {XX}KB ({YY,YYY} tokens)
- Savings: {ZZ}KB ({WW,WWW} tokens, {PP}%)

### Party Mode Impact
- Before overhead: 61,700 tokens
- After overhead: {XX,XXX} tokens
- Savings: {YY,YYY} tokens ({PP}%)

### Inheritance Map

```
AGENTS.md v3.4 (base)
├── strategic_intelligence (7 mandatory questions)
├── genius_mode_templates (4 templates)
├── complexity_assessment (4 levels)
├── enforcement_validation
└── compressed_protocols

    ↓ HERANÇA

NAUTILUS (especialização)
├── domain_knowledge (migration, patterns, performance)
└── additional_reflection_questions (3 perguntas extras)

ORACLE (especialização)
├── domain_knowledge (thresholds, WFA, Monte Carlo, GO/NO-GO)
└── additional_reflection_questions (3 perguntas extras)

[etc]
```

### Next Steps
1. Review REDUNDANCY_MAP.md
2. Execute 009-agents-nautilus-update (ajustar AGENTS.md)
3. Execute 010-droid-refactoring-master (FASE 2-4)
```

---

### 3. Análise de Consistência

Verificar **inconsistências estruturais** entre droids:

**Tabela de Features:**
| Feature | AGENTS.md v3.4 | NAUTILUS | ORACLE | FORGE | SENTINEL | RESEARCH |
|---------|----------------|----------|--------|-------|----------|----------|
| Genius templates | ✅ 4 templates | ? | ? | ? | ? | ? |
| Enforcement validation | ✅ | ? | ? | ? | ? | ? |
| Complexity assessment | ✅ 4 levels | ? | ? | ? | ? | ? |
| Compressed protocols | ✅ fast+emergency | ? | ? | ? | ? | ? |
| Pattern learning | ✅ auto-learning | ? | ? | ? | ? | ? |

Preencher com ✅ (tem), ❌ (falta), ⚠️ (tem mas diferente).

---

### 4. Schema de Herança (XML Template)

Criar o template `<droid_specialization>` que todos os droids refatorados vão usar:

```xml
<droid_specialization>
  <metadata>
    <name>{droid_name}</name>
    <version>{version}</version>
    <inherits_from>AGENTS.md v3.4.0</inherits_from>
    <target_size>{XX}KB</target_size>
  </metadata>
  
  <inheritance>
    <protocols>
      <protocol name="strategic_intelligence" inherit="full"/>
      <protocol name="mandatory_reflection_protocol" inherit="full"/>
      <protocol name="proactive_problem_detection" inherit="full"/>
      <protocol name="genius_mode_templates" inherit="full"/>
      <protocol name="complexity_assessment" inherit="full"/>
      <protocol name="compressed_protocols" inherit="full"/>
      <protocol name="enforcement_validation" inherit="full"/>
      <protocol name="pattern_learning" inherit="full"/>
    </protocols>
  </inheritance>
  
  <domain_knowledge>
    <!-- SÓ conhecimento específico do domínio -->
  </domain_knowledge>
  
  <additional_reflection_questions>
    <!-- SÓ as 3 perguntas EXTRAS do domínio -->
    <question id="N" category="{domain}_specific">
      {question text}
    </question>
  </additional_reflection_questions>
  
  <domain_guardrails>
    <!-- Mantém APENAS guardrails específicos do domínio -->
    <guardrail>{specific rule}</guardrail>
  </domain_guardrails>
</droid_specialization>
```

---

## 📤 Output

### Primary Output
**File:** `.prompts/008-droid-analysis/droid-analysis.md`

**Structure:**
```xml
<droid_analysis_report>
  <metadata>
    <version>1.0</version>
    <date>{YYYY-MM-DD}</date>
    <phase>FASE 1 - Análise Detalhada</phase>
    <execution_time>{XX}min</execution_time>
  </metadata>
  
  <executive_summary>
    <total_redundancy_kb>{XX}</total_redundancy_kb>
    <total_redundancy_percent>{YY}</total_redundancy_percent>
    <token_savings>{ZZ,ZZZ}</token_savings>
    <party_mode_improvement>{WW}%</party_mode_improvement>
  </executive_summary>
  
  <redundancy_map>
    <!-- Link to REDUNDANCY_MAP.md -->
    See: .prompts/008-droid-analysis/REDUNDANCY_MAP.md
  </redundancy_map>
  
  <droid name="NAUTILUS">
    <current_size>53KB</current_size>
    <target_size>{XX}KB</target_size>
    <savings>{YY}KB ({ZZ}%)</savings>
    
    <redundancies>
      <section name="Protocols Genéricos" lines="XXX-YYY" size="{N}KB" action="REMOVE"/>
      <section name="Genius Templates" lines="XXX-YYY" size="{N}KB" action="REMOVE"/>
      <!-- etc -->
    </redundancies>
    
    <domain_knowledge>
      <section name="Migration Mappings" lines="XXX-YYY" size="{N}KB" action="KEEP"/>
      <section name="Event Architecture" lines="XXX-YYY" size="{N}KB" action="KEEP"/>
      <!-- etc -->
    </domain_knowledge>
    
    <additional_questions>
      <question id="18">{text}</question>
      <question id="19">{text}</question>
      <question id="20">{text}</question>
    </additional_questions>
  </droid>
  
  <droid name="ORACLE">
    <!-- Mesma estrutura -->
  </droid>
  
  <droid name="FORGE">
    <!-- Mesma estrutura -->
  </droid>
  
  <droid name="SENTINEL">
    <!-- Mesma estrutura -->
  </droid>
  
  <droid name="RESEARCH-ANALYST-PRO">
    <!-- Mesma estrutura -->
  </droid>
  
  <inheritance_schema>
    <template>{XML template completo}</template>
    <usage_instructions>
      Como usar este template para refatorar cada droid...
    </usage_instructions>
  </inheritance_schema>
  
  <consistency_analysis>
    <feature_matrix>
      {Tabela de features preenchida}
    </feature_matrix>
    <gaps>
      - NAUTILUS falta: {list}
      - ORACLE falta: {list}
      <!-- etc -->
    </gaps>
  </consistency_analysis>
  
  <next_steps>
    <step number="1">Review REDUNDANCY_MAP.md e droid-analysis.md</step>
    <step number="2">Execute 009-agents-nautilus-update.md (ajustar AGENTS.md para Nautilus focus)</step>
    <step number="3">Execute 010-droid-refactoring-master.md (FASE 2-4: implementar refatoração)</step>
  </next_steps>
  
  <confidence>HIGH</confidence>
  <dependencies>
    <dependency>AGENTS.md v3.4 must remain stable during refactoring</dependency>
    <dependency>Git backups before any droid edits</dependency>
  </dependencies>
  
  <open_questions>
    <question>Should NANO versions be created for all TOP 5 or only on-demand?</question>
    <question>How to handle Party Mode detection (automatic vs manual NANO switching)?</question>
  </open_questions>
  
  <assumptions>
    <assumption>AGENTS.md v3.4 protocols are complete and won't need major changes</assumption>
    <assumption>Droids can correctly inherit protocols via reference (no technical blocker)</assumption>
  </assumptions>
</droid_analysis_report>
```

---

### Secondary Output
**File:** `.prompts/008-droid-analysis/REDUNDANCY_MAP.md`

(Estrutura detalhada acima na seção 2)

---

### Tertiary Output
**File:** `.prompts/008-droid-analysis/SUMMARY.md`

**Required Sections:**
- **One-liner:** "Identified {XX}KB redundancy ({YY}% of TOP 5) with token savings of {ZZ,ZZZ} - ready for FASE 2 refactoring"
- **Version:** v1.0
- **Key Findings:**
  - Total redundancy: {XX}KB ({YY}%)
  - Token savings: {ZZ,ZZZ} ({WW}% of TOP 5)
  - Party Mode improvement: +{PP}% budget libre
  - All droids missing: {list common gaps}
- **Decisions Needed:**
  - Approve target sizes per droid? (NAUTILUS: {XX}KB, ORACLE: {YY}KB, etc)
  - Create NANO versions for all TOP 5 or selective?
- **Blockers:**
  - None (analysis phase has no external blockers)
- **Next Step:**
  - Execute 009-agents-nautilus-update.md

---

## ✅ Success Criteria

- [ ] REDUNDANCY_MAP.md created with line-by-line analysis for all TOP 5 droids
- [ ] Each droid has redundancies mapped (what to REMOVE) and domain knowledge extracted (what to KEEP)
- [ ] Each droid has 3 additional_reflection_questions identified
- [ ] Inheritance schema XML template created
- [ ] Consistency matrix filled (which droids have which features)
- [ ] Token savings calculated (before/after with %)
- [ ] Party Mode impact quantified
- [ ] SUMMARY.md created with substantive one-liner
- [ ] Confidence level assigned (should be HIGH after thorough analysis)
- [ ] Next steps clearly defined (009 → 010)

---

## ⚡ Intelligence Application

**Use sequential-thinking** for this complex analysis (15+ thoughts minimum):
1. What is REAL problem? → Massive duplication across droids, no inheritance mechanism
2. What am I NOT seeing? → Some "duplication" might be intentional specialization (verify!)
3. What breaks if I remove X? → Must verify knowledge is truly in AGENTS.md before marking REMOVE
4. Is there simpler approach? → Could merge droids instead of refactor (NO - lose specialization)
5. What happens 5 steps ahead? → Refactored droids → Party Mode works → Future AGENTS.md upgrades propagate
6. Edge cases? → What if droid has partial overlap (50% match AGENTS.md but customized)?
7. Optimization? → Can we automate inheritance checking in future?

**Proactive problem detection:**
- Dependencies: Will refactored droids break existing Task invocations?
- Performance: Will inheritance add latency (NO - static reference, not dynamic)
- Security: N/A (no external inputs)
- Maintainability: IMPROVED (single source of truth in AGENTS.md)

**Use ARGUS research:**
- Query memory MCP for past droid optimization attempts
- Search DOCS/ for previous reports on droid redundancy

---

## 🎯 Estimated Time: 30-45 minutes

**Breakdown:**
- Read all 5 droids: 15 min
- Map redundancies line-by-line: 15 min
- Extract domain knowledge: 10 min
- Create REDUNDANCY_MAP.md: 5 min
- Write droid-analysis.md: 10 min
- Create SUMMARY.md: 5 min

**Total:** 60 min (with buffer for deep analysis)

---

**EXECUTE THIS PROMPT WITH:** claude-sonnet-4 (high intelligence required for pattern recognition)
