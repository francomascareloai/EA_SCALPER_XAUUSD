# Brainstorming Session Results

**Session Date:** 2025-11-21
**Facilitator:** {{agent_role}} {{agent_name}}
**Participant:** Franco

## Session Start

{{session_start_plan}}

## Executive Summary

**Topic:** Avaliação crítica e melhoria do PRD do EA_SCALPER_XAUUSD para prop firms (FTMO e similares)

**Session Goals:**
- Verificar completude e clareza do PRD v2.0
- Validar aderência às regras de prop (FTMO): daily/total loss, news filter, latência
- Identificar lacunas em arquitetura MQL5 ↔ Python (Agent Hub), explainability e scoring
- Priorizar ajustes para implementação imediata e definir próximos passos

**Riscos críticos a evitar (pre-mortem):**
- Latência/IO estourando (OnTimer/bridge >400–500 ms), fila de IO saturada, ausência de degrade para “MQL5 only”
- Violações FTMO por cálculo inconsistente de `ProjectedDailyLoss%`/`MaxTotalLoss%` ou race conditions entre posições abertas e novas entradas
- Bridge Python sem tratamento de timeout/erro → decisões com dados atrasados/faltantes
- Scoring mal calibrado (thresholds inadequados, penalidades de spread/news ausentes)
- Falta de filtro de notícias ou janela errada (NFP/CPI)
- Observabilidade insuficiente: logs incompletos, sem alertas ao aproximar dos limites, parâmetros sem versionamento

**Contramedidas prioritárias:**
- Governança FTMO: fórmulas explícitas de `ProjectedDailyLoss%`, `SoftStop%`, bloqueio de novas entradas; log de todas transições de estado
- Latência: proibir WebRequest em OnTick; OnTimer com `MaxInFlight=1`, fila limitada, timeout curto, degrade automático para “MQL5 only” com flag `degraded_mode`
- Bridge resiliente: contrato de erro (`error`), retries 0/1 com backoff curto; log de `latency_ms`; quedas controladas
- Scoring robusto: thresholds por regime (spread, sessão, volatilidade) + penalidades de news/spread; validar com backtests 3–5 anos e spreads sintéticos
- News filter: janelas exatas (N minutos antes/depois) e comportamento definido (bloquear novas ordens, reduzir lote, fechar parciais)
- Observabilidade: logging CSV + push; versionar parâmetros; alertas em 50/70/90% do MaxDailyLoss; rotação de logs

**Risk Matrix (probabilidade / impacto / mitigação focada em lógica)**
- Cálculo de Daily/Total Loss divergente (Média / Alta): fórmula única de `ProjectedDailyLoss%` com snapshot 00:00 broker; bloquear só ao atingir o limite; log de transições e equidade usada.
- Latência/IO > 400–500 ms (Alta / Alta): OnTimer com fila limitada e `MaxInFlight=1`, timeout curto, degrade para “MQL5 only”; métricas de `latency_ms` e fila, sem travar o fluxo normal quando saudável.
- Bridge Python indisponível/timeout (Média / Alta): retries curtos (0–1) e fallback imediato; flag `degraded_mode`; decisões continuam via MQL5 puro se o hub falhar.
- Scoring desalinhado (Média / Média): thresholds por regime (spread, sessão, vol) e penalidades explícitas; validar com backtests 3–5 anos com spreads realistas; não reduzir frequência por medo, mas ajustar qualidade do sinal.
- News handling mal configurado (Baixa / Alta): janela configurável (p.ex. 15–30 min) antes/depois apenas para high-impact; opções: reduzir lote ou pular entrada — comportamento definido, não “bloquear tudo”.
- Observabilidade fraca (Baixa / Alta): log completo (CSV + push) com Score/Reason e estado de risco; alertas graduais (50/70/90% do MaxDailyLoss) para ação informada, não para paralisar o robô.
- Spread fora do limite (Média / Média): checar `NoTradeIfSpreadPoints` antes da entrada; log do motivo se bloquear; limites calibrados ao XAUUSD.

**Devil's Advocate (stress-test de suposições-chave)**
- Pesos fixos W_Tech/W_Fund/W_Sent bastam → regimes mudam; adotar perfis por spread/sessão/vol com chaveamento logado.
- Daily/Total Loss é “apagar incêndio” → risco de corrida entre cálculo e entrada; checar `ProjectedDailyLoss%` imediatamente antes da ordem com lock, log de equidade e motivo allow/deny.
- Latência controlada só com timeout → sem visibilidade de IO; métricas mínimas: `io_in_flight`, `last_io_latency_ms`, flag `degraded_mode`; alerta se média > limiar.
- News filter genérico resolve → eventos variam (NFP/CPI/FOMC/discursos); janelas distintas por evento, opção de apenas reduzir lote em vez de bloquear.
- Scoring suficiente sem “gates” → lixo entra; criar gatekeeper pré-score (spread, hora proibida, drawdown near limit). Se falhar, nem calcula score; log do motivo.
- Logs/push opcionais → sem rastreabilidade FTMO; campos mínimos por sinal/trade: req_id, regime, score breakdown, decisão do RiskManager, `ProjectedDailyLoss%`, `degraded_mode`, latência IO; rotação + hash/versão dos parâmetros carregados.
- Backtest 3–5 anos basta → regimes de spread/notícia/latência variam; incluir cenários sintéticos: spread x2, slippage alto, latência 300–600 ms, perda do Hub; medir queda de score/execução.

**Techniques Used:** {{techniques_list}}

**Total Ideas Generated:** {{total_ideas}}

### Key Themes Identified:

{{key_themes}}

## Technique Sessions

{{technique_sessions}}

## Idea Categorization

### Immediate Opportunities

_Ideas ready to implement now_

{{immediate_opportunities}}

### Future Innovations

_Ideas requiring development/research_

{{future_innovations}}

### Moonshots

_Ambitious, transformative concepts_

{{moonshots}}

### Insights and Learnings

_Key realizations from the session_

{{insights_learnings}}

## Action Planning

### Top 3 Priority Ideas

#### #1 Priority: {{priority_1_name}}

- Rationale: {{priority_1_rationale}}
- Next steps: {{priority_1_steps}}
- Resources needed: {{priority_1_resources}}
- Timeline: {{priority_1_timeline}}

#### #2 Priority: {{priority_2_name}}

- Rationale: {{priority_2_rationale}}
- Next steps: {{priority_2_steps}}
- Resources needed: {{priority_2_resources}}
- Timeline: {{priority_2_timeline}}

#### #3 Priority: {{priority_3_name}}

- Rationale: {{priority_3_rationale}}
- Next steps: {{priority_3_steps}}
- Resources needed: {{priority_3_resources}}
- Timeline: {{priority_3_timeline}}

## Reflection and Follow-up

### What Worked Well

{{what_worked}}

### Areas for Further Exploration

{{areas_exploration}}

### Recommended Follow-up Techniques

{{recommended_techniques}}

### Questions That Emerged

{{questions_emerged}}

### Next Session Planning

- **Suggested topics:** {{followup_topics}}
- **Recommended timeframe:** {{timeframe}}
- **Preparation needed:** {{preparation}}

---

_Session facilitated using the BMAD CIS brainstorming framework_
