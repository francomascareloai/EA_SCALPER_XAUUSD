# EA_SCALPER_XAUUSD – Product Requirements Document v2.2 (Singularity Edition)

**Autor:** Franco
**Data:** 2025‑11‑27
**Versão:** 2.2 (Singularity Edition - ONNX/ML Integration)
**Mercado alvo:** XAUUSD (Gold) – foco em contas de **Prop Firm (FTMO e similares)**

---

## 1. Visão Geral

O **EA_SCALPER_XAUUSD** será um sistema de trading institucional, multi‑agente, especializado em XAUUSD, com:

* **Execução & risco em MQL5** (alta velocidade, aderência às regras da prop).
* **Agentes de análise em Python** (técnico avançado, fundamental, sentimento, LLMs) acessados via um **Python Agent Hub** (sem CLIPROXY / sem interface via CLI).
* **Transparência total:** cada trade tem uma explicação em linguagem natural (*Reasoning String*).
* **Engine de Score (0–100)**: só executa setups classe **A+** (score ≥ limiar configurável).

---

## 2. Objetivos & Métricas de Sucesso

### 2.1 Performance de Trading

1. **Expectativa positiva** consistente em backtests e forward tests:

   * Win Rate ≥ 40%.
   * Risk:Reward médio ≥ 1:1.5.
2. **Drawdown controlado**:

   * Máx. DD absoluto ≤ 10% do saldo inicial (parametrizável por tipo de conta).

### 2.2 Conformidade com Prop Firms (ex.: FTMO)

Regras típicas FTMO (parametrizáveis):

* **Max Daily Loss:** 5% do saldo inicial (contas standard).
* **Max Total Loss:** 10% do saldo inicial.
* **Objetivo:** 0 violações de regras em:

  * Backtests multi‑ano (XAUUSD).
  * Simulações de Challenge/Verification.
  * Operação real em conta demo/prop.

### 2.3 Transparência & Explainability

* 100% dos trades executados devem ter:

  * **Score final** + decomposição por fator (Técnico, Fundamental, Sentimento).
  * **Reasoning String** legível para humanos.
  * Registro em log + envio via:

    * Push Notification (MetaTrader).
    * E opcionalmente arquivo CSV/JSON.

### 2.4 Estabilidade & Latência

* **OnTick (MQL5):** tempo de execução < 50 ms (sem chamadas externas).
* **Chamadas ao Python Agent Hub (quando usadas em tempo real):**

  * Round-trip médio < 500 ms.
  * Timeout rígido (ex.: 300–400 ms) + fallback seguro (não abrir novas posições).

---

## 3. Escopo de Produto

### 3.1 MVP – Fases 1 & 2

**Meta:** Robô scalper XAUUSD consistente, com arquitetura modular, já com Technical Score robusto.

#### Entregas Principais

1. **Arquitetura Modular MQL5**

   * Refatorar:

     * `EliteOrderBlock`
     * `EliteFVG`
     * `InstitutionalLiquidity`
     * `FTMO_RiskManager`
   * Cada componente como módulo/“serviço” interno com interface clara (ex.: classes ou structs dedicados).

2. **SignalScoringModule (MQL5)**

   * Cálculo de `TechScore` a partir de:

     * Order Blocks.
     * FVGs.
     * Liquidity Sweeps.
     * Market Structure (HH/HL/LH/LL).
     * Volatilidade (ATR, range horário).
   * Cálculo do `FinalScore` (inicialmente só Técnico) e tomada de decisão:

     * `if TechScore >= TechThreshold` → candidato a trade.
     * Lógica de side (BUY/SELL) pela direção de mercado + contexto de liquidez.

3. **Python Agent – Technical Analyst (Phase 2)**

   * **Python Agent Hub** (sem CLIPROXY): serviço Python local que:

     * Recebe contexto de mercado do EA (via HTTP/REST ou ZeroMQ/TCP).
     * Retorna:

       * `TechSubScore_Python` (0–100).
       * Lista de padrões detectados (ex.: wedges, triangles, volatility regimes).
   * MQL5 integra essa pontuação ao `TechScore` global com peso configurável. O campo retornado pelo Python é `tech_subscore`.

4. **Explainabilidade Básica (MVP)**

   * Log estruturado no terminal + arquivo (`.csv`/`.log`):

     * `Time, Symbol, Direction, Price, SL, TP, TechScore, OB_Flag, FVG_Flag, Trend, ATR, ...`
   * Mensagens simples do tipo:

     * `"Score: 88 | OB: Yes | FVG: Yes | Trend: Bullish | ATR: High"`.

### 3.2 Growth Features – Phase 3

1. **Fundamental & News Agent (Python)**

   * Coleta de:

     * Calendário econômico (High/Medium impact).
     * Eventos relevantes para USD/Gold.
   * Saída:

     * `FundBias` em [-1, 1] (bearish → bullish).
     * `FundScore` (0–100).

2. **Sentiment Agent (Python)**

   * Consome dados de sentimento retail (ex.: Myfxbook ou similares).
   * Saída:

     * `SentScore` (0–100).
     * `SentBias` (contrarian/follow‑crowd).

3. **LLM‑Based Reasoning**

   * Uso de LLM via Python (Gemini 3 Pro, GPT‑5.1, etc.) para gerar:

     * `Reasoning String` detalhada:

       * Contexto de mercado.
       * Por que entrar.
       * Onde estão os riscos.
       * Como o trade se encaixa na sessão (Londres/NY/Ásia).
   * LLM chamado **fora do OnTick** (assíncrono ou em background).

4. **Dynamic Drawdown Control**

   * `FTMO_RiskManager` ajusta tamanho de lote dinamicamente:

     * À medida que o drawdown diário se aproxima do limite (por ex. 3%, 4%, >4.5%).
   * Possível reduzir:

     * Freq. de trades.
     * Risco por trade.
     * Encerrar o dia de forma automática próximo ao limite.

### 3.3 Visão de Futuro (Phase 4+)

1. **Portfolio Manager Agent**

   * Generalização da arquitetura para múltiplos ativos (índices, FX majors).
   * Gestão de correlação e alocação de risco por ativo.

2. **Self‑Optimization / Meta‑Learning**

   * Coletar histórico de trades + contexto + reasoning e alimentar:

     * Agente de *Performance Analyst*.
     * Agente de *Self‑Optimizer* que propõe ajustes:

       * Pesos de score.
       * Horários operacionais.
       * Distâncias típicas de SL/TP.
   * Inspiração em frameworks multi‑agentes com auto‑reflexão e data synthesis para trading.

---

## 4. Filosofia de Trading & Restrições de Domínio

### 4.1 Mercado: XAUUSD (Gold)

* Alta volatilidade intraday.
* Comportamento forte em sessões Londres/NY.
* Sensível a:

  * Dados macro de USD.
  * Notícias de inflação / juros / Fed / risco geopolítico.

### 4.2 Prop Trading & FTMO

Diretrizes (parametrizáveis no EA):

* **Regra de Max Daily Loss** (ex.: 5% FTMO): o EA precisa:

  * Monitorar P/L acumulado do dia, incluindo trades abertos.
  * Ajustar lote ou parar de operar antes do limite.
* **Max Total Loss**: não permitir que equity caia abaixo de (saldo inicial − limite).
* **News Filter:** opção para:

  * Não abrir novas posições N minutos antes/depois de high‑impact news.
  * Encerrar ou reduzir exposição em eventos específicos.

### 4.3 Princípio “Risk First”

* **FTMO_RiskManager** tem poder de veto:

  * Trade só é executado se:

    * `RiskManager` aprovar.
    * Score final ≥ threshold.
* Se qualquer check de risco falhar:

  * Trade é cancelado e logado.

---

## 5. Arquitetura de Sistema

### 5.1 Visão por Camadas (Event‑Driven)

Inspirado em arquiteturas de sistemas algorítmicos modernos (event‑driven, camadas lógicas distintas).

**Camadas:**

1. **Data & Events Layer (MQL5)**

   * Eventos: `OnTick`, `OnTimer`, `OnTradeTransaction`, `OnInit`, `OnDeinit`, `OnChartEvent`.
   * Coleta:

     * Ticks, candles, spreads, swaps, ATR.
     * Estado de posições/ordens.

2. **Strategy & Agents Layer**

   * Módulos MQL5 (OrderBlock, FVG, Liquidity, MS, ATR).
   * `SignalScoringModule`.
   * Interface com **Python Agent Hub** (via HTTP/REST ou ZeroMQ).

3. **Execution & Risk Layer**

   * `TradeExecutor` (MQL5).
   * `FTMO_RiskManager`.
   * Gerenciamento de SL/TP, partials, trailing, break-even.
   * Ordem de ações por tick: (1) partial close, (2) move to breakeven, (3) trailing. Apenas uma ação por tick; demais ficam em fila.

4. **Telemetry & Explainability Layer**

   * Logger local (arquivo/CSV).
   * Push Notifications.
   * Exportador para JSON (integração futura com dashboards externos).

### 5.2 Componentes Principais

1. **Core EA (MQL5)**

   * Ponto de entrada: um único EA (`EA_SCALPER_XAUUSD`) carregado no gráfico de XAUUSD.
   * Gerencia ciclo de vida, estados internos e interações com agentes.

2. **Módulos de Análise Técnica (MQL5)**

   * `EliteOrderBlockModule`
   * `EliteFVGModule`
   * `InstitutionalLiquidityModule`
   * `MarketStructureModule`
   * `VolatilityModule` (ATR, ranges, sessões).

3. **Risk Engine**

   * `FTMO_RiskManager`:

     * Cálculo de tamanho de lote.
     * Cálculo de risco por trade (% da equity).
     * Monitoramento de drawdown diário e total.
   * Filtros:

     * Horários (sessões, blocos de horário proibidos).
     * News / eventos.

4. **Python Agent Hub (sem CLIPROXY)**

   * Serviço Python persistente:

     * Exposto como:

       * API REST local (via `WebRequest` em MQL5).
       * OU servidor ZeroMQ (EA como cliente).
   * Responsável por orquestrar:

     * `Technical Agent (Python)`
     * `Fundamental Agent`
     * `Sentiment Agent`
     * `LLM Reasoning Agent`
   * Formato de mensagem:

     * JSON request/response (definido na seção de Interfaces).

5. **Storage / Logs**

   * MQL5:

     * Arquivo `EA_SCALPER_XAUUSD_logs.csv`.
   * Python:

     * Logs próprios (para agentes e chamadas LLM).

310: 
311: 5. **Local Persistence Layer (JSON/CSV)**
312: 
313:    * **Critical Data Cache:**
314:      * `news_calendar.json`: Cópia local do calendário para filtro de notícias offline.
315:      * `daily_context.json`: "Daily Briefing" do Deep Researcher (Regime de Mercado, Viés).
316:      * `risk_params.json`: Parâmetros de risco ajustados dinamicamente.
317:    * **Objetivo:** Garantir que o EA (Body) continue operando com inteligência básica mesmo se o Python (Brain) morrer.
318: 
319: ---

## 6. Modelo de Estados do EA

Estados principais:

1. `IDLE`

   * Sem posição, aguardando setup.
2. `SIGNAL_PENDING`

   * Condições técnicas favoráveis, aguardando:

     * Score final.
     * Aprovação de risco.
3. `POSITION_OPEN`

   * Posição aberta; gerenciamento ativo (SL/TP, trailing, partials).
4. `COOLDOWN`

   * Após eventos específicos:

     * Stop loss consecutivos.
     * Aproximação de max daily loss.
   * Temporariamente suspende novas entradas.

6. `SURVIVAL_MODE` (Market Driven)

   * Ativado por:
     * Alta Volatilidade (VIX explosivo).
     * Alerta de "Guerra/Crise" do Deep Researcher.
   * Ações:
     * Reduzir lote drasticamente (ex: 50% ou 25%).
     * Apertar Trailing Stop.

7. `EMERGENCY_MODE` (System Driven)

   * Ativado quando:
     * Limite de max daily loss atingido.
     * **Heartbeat falha** (Python morto).
   * Ações:
     * Proibir novas posições.
     * Gerenciar apenas posições abertas (MQL5 Only).

---

## 7. Multi‑Agent Ecosystem

### 7.1 Visão Geral

Agentes e seus papéis:

1. **Technical Analyst (MQL5 + Python)**

   * MQL5: OB, FVG, Liquidez, Estrutura de Mercado, ATR.
   * Python: padrões mais complexos (clusters, regimes de volatilidade, microstructure features).

2. **Fundamental Analyst (Python) – Phase 3**

   * Calendário econômico, notícias macro.
   * Bias e score fundamental.

3. **Sentiment Analyst (Python) – Phase 3**

   * Dados de posicionamento retail.
   * `SentScore` e indicação contrarian.

4. **Risk Manager (MQL5)**

   * FTMO constraints.
   * Pos sizing, limites, filtros de horário/news.

5. **LLM Reasoning Agent (Python) – Phase 3**

   * Gera `Reasoning String` amigável, baseado:

     * Nos scores.
     * No contexto de mercado.
     * Em histórico recente.

### 7.2 Formatos de Entrada/Saída dos Agentes Python

Contrato de interface (EA ↔ Python):

- Campos obrigatórios em toda requisição: `schema_version`, `req_id`, `timeout_ms`.
- Campos obrigatórios em toda resposta: `schema_version`, `req_id`, `error` (objeto ou `null`).
- Convenção de nomes: `snake_case` em ambas direções.
- Pontuação técnica retornada pelo Python: `tech_subscore` (subscore específico do Python, 0–100).

**Request JSON (exemplo)**

```json
{
  "schema_version": "1.0",
  "req_id": "b2e0e3d6-0001-4d8f-9a5a-req",
  "timeout_ms": 400,
  "symbol": "XAUUSD",
  "timeframe": "M5",
  "timestamp": 1732185600,
  "price_context": {
    "bid": 1965.20,
    "ask": 1965.40,
    "atr": 2.5,
    "session": "London"
  },
  "technical_snapshot": {
    "order_block": "bullish",
    "fvg": "yes",
    "liquidity_sweep": "buy_side_taken",
    "market_structure": "bullish",
    "recent_rr": 1.8
  },
  "recent_trades_summary": {
    "last_5_trades_rr_mean": 1.6,
    "last_5_trades_winrate": 0.6
  }
}
```

**Response JSON (exemplo)**

```json
{
  "schema_version": "1.0",
  "req_id": "b2e0e3d6-0001-4d8f-9a5a-req",
  "tech_subscore": 82,
  "patterns": ["volatility_compression", "mini_range_breakout"],
  "confidence": 0.78,
  "fund_score": 65,
  "fund_bias": 0.4,
  "sent_score": 30,
  "sent_bias": -0.5,
  "error": null
}
```

---

## 8. Scoring Engine

### 8.1 Fórmulas

**Pontuações de base:**

* `TechScore` ∈ [0, 100]
* `FundScore` ∈ [0, 100]
* `SentScore` ∈ [0, 100]

**Peso configurável (inputs do EA):**

* `W_Tech` (ex.: 0.6)
* `W_Fund` (ex.: 0.25)
* `W_Sent` (ex.: 0.15)

**Score final:**

```text
FinalScore = (TechScore * W_Tech) +
             (FundScore * W_Fund) +
             (SentScore * W_Sent)
```

**Regra de execução:**

* Executar trade se:

  * `FinalScore >= ExecutionThreshold` (ex.: 85).
  * `FTMO_RiskManager` aprovar.
  * Filtros de sessão/news permitirem.

### 8.2 Normalização & Penalidades

* Penalizar score quando:

  * Spread alto.
  * Proximidade de news.
  * Sessão com baixa liquidez (Ásia para alguns setups).
* Bônus de score quando:

  * Confluência de múltiplos sinais (OB + FVG + sweep na mesma zona).
  * Direção alinhada com tendência de timeframe maior.

### 8.3 Perfis de Regime e Ajuste Dinâmico de Pesos

Definir perfis com critérios objetivos e chaveamento logado:

- **normal:** spread <= limite, fora de janela de news, vol moderada, Hub OK
- **spread_alto:** spread > limite_n1 (bloqueia se > limite hard), reduz pesos de sinais sensíveis a ruído
- **vol_alta:** ATR/timeframe ou range horário acima do pctl definido → elevar ExecutionThreshold, priorizar confluência
- **news_window:** dentro da janela configurada por evento (tabela em 10.5) → opcional reduzir lote ou pular entrada
- **hub_degraded:** Hub Python indisponível/timeout → operar apenas TechScore MQL5, W_Fund=W_Sent=0, flag `degraded_mode=true`

Para cada perfil registrar: critérios, pesos (W_Tech/W_Fund/W_Sent), delta de `ExecutionThreshold`, ação (executa/reduz lote/ignora sinal), e log do chaveamento com timestamp e motivo.

---

## 9. Explainability & Notificações

### 9.1 Log Estruturado

Para cada sinal/trade:

* Campos mínimos:

  * `Timestamp`
  * `Symbol`
  * `Direction` (BUY/SELL)
  * `EntryPrice`
  * `SL`, `TP`
  * `FinalScore`, `TechScore`, `FundScore`, `SentScore`
  * `RiskPerTrade_%`
  * `DailyDD_%`, `TotalDD_%`
  * Flags: `OB`, `FVG`, `LiquiditySweep`, `Trend`, `Session`, `Spread`, etc.
  * `ReasoningString` (ou link para arquivo separado).

Campos adicionais de telemetria e correlação:

* `log_level` (INFO|WARN|ERROR)
* `req_id` (correlação EA↔Python)
* `latency_ms` (por requisição e por decisão)
* `degraded_mode` (bool – se decisão foi tomada sem Python)

Política de segurança de logs:

* Nunca registrar chaves de API ou PII; aplicar redação automática quando necessário.

### 9.2 Push Notification

Template sugerido:

> `EA_SCALPER_XAUUSD – Trade Executado`
> `Tipo: BUY | Preço: 1965.40 | SL: 1962.90 | TP: 1970.40`
> `Score: 91 (Tech 88 / Fund 72 / Sent 40)`
> `Reason: [Reasoning String resumida em 1–2 frases]`

### 9.3 Reasoning String (gerada por LLM – Phase 3)

* Deve responder:

  1. **Contexto de mercado:** tendência, sessão, volatilidade.
  2. **Zona de valor:** OB/FVG, liquidez varrida, localização do SL e TP.
  3. **Risco:** proximidade de notícia, volatilidade do dia, situação de DD.
  4. **Justificativa de direcional:** por que comprar/vender aqui.

---

## 10. Gestão de Risco & Conformidade FTMO

### 10.1 Position Sizing

* Fórmula básica:

  * `LotSize = f(AccountEquity, RiskPerTrade%, SL_Points, TickValue)`
* `RiskPerTrade%` dinâmico:

  * Reduzir quando:

    * Daily DD > X% (ex.: 2%, 3%).

### 10.2 Controle de Drawdown Diário

* Tracking separado para:

  * P/L fechado do dia.
  * P/L flutuante de trades abertos.
* `ProjectedDailyLoss = ClosedPL_Today + OpenFloatingLoss`.
* Parâmetros:

  * `MaxDailyLoss%` (ex.: 5%).
  * `SoftStop%` (ex.: 3.5%).
* Ações:

  * Ao atingir `SoftStop%`:

    * Reduzir lotes.
    * Aumentar thresholds de score.
  * Ao atingir ou se aproximar criticamente de `MaxDailyLoss%`:

    * Bloquear novas entradas até o dia seguinte.

#### 10.2.1 Fórmulas e regras operacionais (broker time)

- Snapshot diário: `EquityAtDayStart` capturado em 00:00 do servidor (broker time).
- `CurrentEquity` inclui P/L flutuante de posições abertas.
- `ProjectedDailyLoss% = (EquityAtDayStart - CurrentEquity) / EquityAtDayStart * 100`.
- Bloquear novas entradas quando `ProjectedDailyLoss% >= MaxDailyLoss%`.
- Em `SoftStop%`:
  - Reduzir `RiskPerTrade%` e/ou aumentar `ExecutionThreshold` por multiplicadores configuráveis.
  - Pode reduzir frequência de operações (cooldown adicional).
- Registrar em log cada transição de estado com motivo (vide 9.1 Log Estruturado).

### 10.3 Controle de Drawdown Total

* Monitora `Equity >= InitialBalance * (1 - MaxTotalLoss%)`.
* Se violado:

  * EA entra permanentemente em `EMERGENCY_MODE` até reinicialização manual.

### 10.4 Gatekeeper de Entrada (pré-score)

Pipeline antes de calcular qualquer score:

- Checar `NoTradeIfSpreadPoints`; se exceder, abortar sinal e logar motivo.
- Checar sessão/horários proibidos; se fora, abortar e logar.
- Checar `ProjectedDailyLoss%` e `MaxTotalLoss%`; se >= limite, bloquear novas entradas; se >= SoftStop%, elevar thresholds ou reduzir lote conforme configuração.
- Checar janelas de news (10.5) e perfil de regime ativo (8.3) para decidir executar/reduzir/pular.
- Somente se todos gates passarem → calcular scores (Tech/Fund/Sent) e seguir fluxo normal.

Registrar em log o motivo de bloqueio ou continuação (req_id, spread, sessão, DD, perfil, ação tomada).

### 10.5 Tabela de News por Evento

Eventos de alto impacto (configuráveis) com janelas distintas e ação padrão:

- **NFP/CPI/FOMC:** janela sugerida 30 min antes / 30 min depois → ação: pular entrada ou reduzir lote 50%.
- **Discursos Fed/BoE/ECB:** 15 min antes / 15 min depois → ação: opcional reduzir lote; operar somente se spread normal.
- **Feriados ilíquidos:** flag manual; opcional: operar somente em sessões principais.

Regras:

- A janela gera perfil `news_window`; ExecutionThreshold pode subir e lote pode reduzir, mas evitar bloqueio genérico.
- Todas decisões relacionadas a news devem ser logadas com evento, janela, ação (executou/pulou/reduziu lote).

---

## 11. Requisitos Não Funcionais

### 11.1 Performance

* `OnTick`:

  * Execução < 50 ms (sem chamada externa).
  * PROIBIDO realizar `WebRequest`/IPC dentro de `OnTick`.
  * Toda integração externa deve ocorrer via `OnTimer` com fila limitada (ex.: `io_queue`) e backpressure:
    * Bounded queue (tamanho configurável), máximo de requisições simultâneas (ex.: `MaxInFlight = 1`).
    * `timeout_ms` curto e sem retries (fail fast). Em timeout/erro: degradar para modo “MQL5 only”.
    * Cada requisição possui `req_id` para telemetria e correlação.
  * Toda chamada a Python deve ser feita:

    * Em `OnTimer`.
    * Ou como processo assíncrono (fila de requests/respostas).

* Python Agent Hub:

  * Média de resposta < 500 ms.
  * Timeout curto + fallback conservador (ignorar contribuição de agente naquela decisão).

### 11.2 Resiliência & Falhas

* **Heartbeat Protocol:**
  * MQL5 envia "Ping" a cada 5s.
  * Se sem "Pong" por 15s -> **EMERGENCY_MODE (MQL5 Only)**.
  * Tenta reconectar silenciosamente em background.

* Se Python Agent Hub não responder:
  * Logar erro.
  * Usar dados cacheados (`news_calendar.json`) para segurança.
  * Operar em modo “MQL5 only” (apenas TechScore MQL5).

* Se falha crítica:
  * Parar novas entradas.
  * Manter gerenciamento de trades abertos.

### 11.3 Segurança

* Chaves de API (OpenAI, Gemini, news feeds, etc):

  * Guardadas apenas em `.env` no lado Python.
  * Nunca no código MQL5.
* No MetaTrader:

  * Usar `WebRequest` apenas para URLs confiáveis.
  * Configurar “Allow WebRequest for listed URL” (Tools → Options → Expert Advisors) com whitelist de domínios autorizados.

### 11.4 Métricas e Orçamento de Latência / IO

- Monitorar e logar: `io_in_flight`, `last_io_latency_ms`, `avg_io_latency_ms`, `degraded_mode` (bool), tamanho da fila.
- Orçamento: P95 de IO < timeout configurado; se P95 exceder por N ciclos, acionar alerta e avaliar redução de dependência do Hub.
- Degradação: ao detectar timeout/erro, fallback imediato para perfil `hub_degraded` (8.3) sem travar loop; retomar automático quando Hub stabiliza.

---

## 12. Telemetria & Observabilidade

1. **KPIs de Trading**
   * Expectancy, Win Rate, Risco médio por trade, Max consecutive wins/losses, Sharpe/Sortino.

2. **KPIs de Sistema**
   * `avg_io_latency_ms`, `p95_io_latency_ms`, % requisições com timeout, % decisões em `degraded_mode`.
   * Tempo médio de `OnTick`; tamanho médio/máximo da fila `io_queue`.
   * Nº de falhas por dia/semana.

3. **Campos mínimos de log (sinal/trade)**
   * `req_id`, regime ativo, `FinalScore` + breakdown (Tech/Fund/Sent), decisão do RiskManager, `ProjectedDailyLoss%` usado, spread, flag `degraded_mode`, `latency_ms`.
   * Motivo de bloqueio se gatekeeper negar (10.4) ou se news ajustar decisão (10.5).

4. **Alertas**
   * Thresholds graduais: 50/70/90% do `MaxDailyLoss%`; P95 de IO acima do orçamento; % timeouts acima de X% em N minutos.

5. **Exportação**
   * Logs diários em CSV + opcional JSON para dashboards.
   * Hash/versão dos parâmetros carregados a cada inicialização para rastreabilidade.

---

## 13. Testes & Validação

### 13.1 Backtests

* Período:

  * Mínimo 3–5 anos de XAUUSD com tick data.
* Condições:

  * Spread realista (incluindo spikes).
  * News filter ativado/desativado.

### 13.2 Stress Tests

* Cenários:

  * Volatilidade extrema (NFP, CPI, FOMC).
  * Slippage alto.
  * Perda de conexão temporária.
  * Erros do Agent Hub (sem resposta).

### 13.3 Simulação de Prop Firm

* Scripts de simulação:

  * Aplicar regras de Max Daily Loss e Max Total Loss no backtest.
  * Contar quantas vezes seriam violadas.

### 13.4 Matriz de Testes por Regime

Combinar cenários: sessão (Londres/NY) × spread (normal/alto) × news (on/off) × Hub (on/off). Sucesso = zero violação FTMO, acionamento de fallback em ≤1 tick quando Hub falha, e P95 IO dentro do orçamento.

### 13.5 Critérios de Aceitação por Fase

- **Phase 1 (MQL5 core):** OnTick < 50 ms; nenhuma violação FTMO em 30 dias simulação; gatekeeper bloqueia sinais inválidos e loga motivo.
- **Phase 2 (Hub técnico):** Fallback para `hub_degraded` em ≤1 tick quando timeout/erro; % decisões em degraded_mode reportada; latência Hub P95 < timeout.
- **Phase 3 (Fund/Sent/LLM):** Scores integrados sem aumentar violações FTMO; Reasoning Strings presentes em 95% dos trades; news table aplicada.
- **Phase 4 (Auto-otimização/portfólio):** Não aumentar risco FTMO; logs e hashes de parâmetros versionados.

### 13.6 Cenários Sintéticos de Stress

- Spread x2, slippage alto, latência simulada 300–600 ms.
- Hub offline intermitente → verificar continuidade MQL5-only e logs.
- Eventos high-impact em sequência (NFP + CPI) com janelas diferentes.
- Critério: queda de win rate ≤ limite definido; zero violação FTMO; logs completos incluindo motivos de bloqueio.

---

## 14. Roadmap de Implementação (alto nível)

Gates de Fase (go/no‑go): avançar para a próxima fase somente se:

- Seção 2 (Performance) atendida.
- Seção 10 (FTMO) sem violações em simulações/backtests.
- Seção 11 (Latência) dentro dos limites.
- Seção 13 (Testes) executada com resultados anexados (artefatos/backtests).

1. **Phase 1 – Core MQL5**

   * Refatoração modular.
   * TechScore totalmente MQL5.
   * RiskManager com FTMO básico.

2. **Phase 2 – Python Technical Agent**

   * Implementar Agent Hub Python.
   * Integração bidirecional EA ↔ Python.
   * Score combinado MQL5 + Python.

3. **Phase 3 – Fund/Sent + LLM Reasoning**

   * Fundamental Agent.
   * Sentiment Agent.
   * Reasoning Agent (LLM).
   * Reasoning Strings completas.

4. **Phase 4 – Auto‑Otimização & Portfólio**

   * Performance Analyst.
   * Self‑Optimizer.
   * Suporte multi‑ativos.

5. **Phase 5 – ONNX/ML Integration (Singularity Architecture)**

   * Regime Detection (Hurst Exponent + Shannon Entropy + Kalman Filter).
   * ONNX Direction Model (LSTM/xLSTM for price prediction).
   * ONNX Volatility Model (GRU for ATR forecasting).
   * ONNX Fakeout Detector (CNN for breakout validation).
   * Meta-Learning Pipeline (self-optimization from trade history).
   * MQL5 ONNX Runtime integration.

---

## 14.5 Phase 5: ONNX/ML Architecture (Singularity)

### 14.5.1 Overview

Phase 5 introduces Machine Learning as a **confirmation layer**, not replacement for SMC/Price Action. The "Singularity Architecture" combines:

1. **Regime Detection Layer** - Statistical filtering (Hurst + Entropy)
2. **ML Confirmation Layer** - ONNX models for probabilistic validation
3. **Meta-Learning Layer** - Self-optimization from historical performance

### 14.5.2 Regime Detection Module

**Already Implemented in Python Agent Hub** (`app/services/regime_detector.py`)

```python
# Singularity Filter Logic
if H > 0.55 and S < 1.5:
    regime = "PRIME_TRENDING"    # Full size, momentum strategies
elif H > 0.55 and S >= 1.5:
    regime = "NOISY_TRENDING"    # Half size, wider stops
elif H < 0.45 and S < 1.5:
    regime = "PRIME_REVERTING"   # Full size, fade extremes
elif H < 0.45 and S >= 1.5:
    regime = "NOISY_REVERTING"   # Half size, tight TP
else:  # 0.45 <= H <= 0.55
    regime = "RANDOM_WALK"       # NO TRADE - no statistical edge
```

**Endpoints**:
- `POST /api/v1/regime` - Get regime analysis with Hurst, Entropy, Kalman trend

**Integration**:
- TechnicalAgent automatically applies regime filter when prices provided
- Score adjustment: -30 for RANDOM_WALK, +10 for PRIME regimes

### 14.5.3 ONNX Model Ensemble

**Models to Implement**:

| Model | Architecture | Input | Output | Purpose |
|-------|--------------|-------|--------|---------|
| Direction | LSTM/xLSTM | 100 bars × 15 features | P(bullish), P(bearish) | Confirm trade direction |
| Volatility | GRU | 50 bars × 5 features | ATR forecast (5 bars) | Dynamic SL/TP |
| Fakeout | CNN | 20 bars × OHLC | P(fakeout), P(real) | Filter false breakouts |
| Regime | Random Forest | Hurst, Entropy, ADX, etc. | Regime classification | ML-based regime |

**Feature Engineering (15 Core Features)**:
1. Close returns (normalized)
2. Log returns
3. Range % (high-low / close)
4. RSI M5, M15, H1 (3 features)
5. ATR (normalized)
6. MA distance (close - MA20)
7. Hurst exponent (rolling 100)
8. Shannon entropy (rolling 100)
9. Session indicator (0=Asia, 1=London, 2=NY)
10. Hour of day (cyclical encoding)
11. OB proximity score
12. FVG proximity score

**ONNX Integration in MQL5**:
```mql5
// Load models
long direction_model = OnnxCreate("Models\\direction_model.onnx", ONNX_DEFAULT);
long volatility_model = OnnxCreate("Models\\volatility_model.onnx", ONNX_DEFAULT);

// Inference
float input[], output[];
OnnxRun(direction_model, ONNX_NO_CONVERSION, input, output);
double p_bullish = output[0];
```

### 14.5.4 Hybrid Strategy Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    SINGULARITY TRADE FLOW                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. REGIME FILTER (First Gate)                              │
│     ├─ Calculate Hurst Exponent                             │
│     ├─ Calculate Shannon Entropy                            │
│     └─ IF RANDOM_WALK → STOP (no trade)                     │
│                                                             │
│  2. SMC SETUP (Second Gate)                                 │
│     ├─ HTF trend alignment (Kalman filter)                  │
│     ├─ MTF Order Block / FVG identification                 │
│     └─ LTF confirmation candle                              │
│                                                             │
│  3. ML CONFIRMATION (Third Gate)                            │
│     ├─ Direction Model: P(direction) >= 0.65?               │
│     ├─ Fakeout Model: P(fakeout) < 0.4?                     │
│     └─ Adjust confidence based on ML output                 │
│                                                             │
│  4. POSITION SIZING                                         │
│     ├─ Base: 1% risk                                        │
│     ├─ × Regime multiplier (0.5 or 1.0)                     │
│     ├─ × ML confidence boost (1.0 to 1.25)                  │
│     └─ Volatility Model → SL/TP adjustment                  │
│                                                             │
│  5. RISK MANAGEMENT (Final Gate)                            │
│     ├─ FTMO compliance check                                │
│     ├─ Daily DD check                                       │
│     └─ Execute or reject                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 14.5.5 Meta-Learning Pipeline

**Data Collection**:
- Every trade logged with: entry, exit, scores, features, regime, outcome
- Stored in `trades_history.json` for analysis

**Performance Analysis** (Weekly):
```python
def analyze_performance(trades_df):
    # Find patterns in losing trades
    losers = trades_df[trades_df['pnl'] < 0]
    
    # Identify regime failures
    regime_perf = trades_df.groupby('regime')['pnl'].mean()
    
    # Detect feature anomalies
    for feature in features:
        if significant_difference(losers[feature], winners[feature]):
            flag_for_review(feature)
    
    return recommendations
```

**Self-Optimization**:
- Adjust weights (W_Tech, W_Fund, W_Sent) based on regime performance
- Retrain ONNX models quarterly with new data
- Walk-Forward Analysis (WFE >= 0.6) required before deployment

### 14.5.6 Validation Requirements

| Metric | Minimum | Target |
|--------|---------|--------|
| WFE (Walk-Forward Efficiency) | 0.6 | 0.75 |
| OOS Win Rate | 55% | 60% |
| OOS Sharpe | 1.0 | 1.5 |
| Monte Carlo 5th %ile DD | < 12% | < 9% |
| Inference Time | < 100ms | < 50ms |
| FTMO Compliance | 100% | 100% |

### 14.5.7 Files Structure

```
EA_SCALPER_XAUUSD/
├── MQL5/
│   └── Models/
│       ├── direction_model.onnx
│       ├── volatility_model.onnx
│       └── fakeout_model.onnx
├── Python_Agent_Hub/
│   └── app/
│       └── services/
│           ├── regime_detector.py      # ✅ IMPLEMENTED
│           ├── ml_inference.py         # ONNX inference service
│           └── meta_learner.py         # Self-optimization
└── .factory/
    ├── droids/
    │   └── onnx-model-builder.md       # ✅ CREATED
    └── commands/
        ├── singularity.md              # ✅ CREATED
        ├── build-onnx.md               # ✅ CREATED
        ├── regime-detect.md            # ✅ CREATED
        └── ml-pipeline.md              # ✅ CREATED
```

### 14.5.8 Phase 5 Milestones

| Milestone | Deliverable | Status |
|-----------|-------------|--------|
| 5.1 | Regime Detection in Python Hub | ✅ DONE |
| 5.2 | Regime endpoint `/api/v1/regime` | ✅ DONE |
| 5.3 | TechnicalAgent + Regime integration | ✅ DONE |
| 5.4 | Kalman Filter trend estimation | ✅ DONE |
| 5.5 | Direction Model training pipeline | 🔲 TODO |
| 5.6 | Direction Model ONNX export | 🔲 TODO |
| 5.7 | MQL5 ONNX integration | 🔲 TODO |
| 5.8 | Volatility Model | 🔲 TODO |
| 5.9 | Fakeout Detector | 🔲 TODO |
| 5.10 | Meta-Learning Pipeline | 🔲 TODO |
| 5.11 | Full validation (WFA, Monte Carlo) | 🔲 TODO |

---

## 15. Assumptions & Constraints

- Tempo de referência: horário do servidor (broker time). Início do dia às 00:00 do servidor para `EquityAtDayStart`.
- Especificações de contrato do XAUUSD (ponto, tick value, lot step, min lot) devem ser lidas em runtime via `SymbolInfo*`; nunca hardcode.
- Proibido `WebRequest`/IPC em `OnTick`. Integrações externas apenas via `OnTimer` com fila limitada, `timeout_ms` curto e `MaxInFlight` pequeno.
- Whitelist de `WebRequest`: configurar domínios permitidos no terminal MetaTrader.
- Sem martingale, grid ou averaging down.
- Sem manutenção de posições no fim de semana, a menos que `AllowWeekendHold=true` (padrão recomendado: `false`).

## 16. Parâmetros de Configuração (defaults sugeridos)

Risco:

- `RiskPerTrade% = 0.5`
- `MaxDailyLoss% = 5.0`
- `SoftStop% = 3.5`
- `MaxTotalLoss% = 10.0`

Higiene de execução:

- `NoTradeIfSpreadPoints = 80`
- `MaxSlippagePoints = 50`
- `MinATR_M5 = 10.0`
- `SessionWindows = ["08:00–11:30", "13:30–16:30"]` (broker time)
- `NewsFreezeMinutes = 30`

Scoring:

- `W_Tech = 0.6`, `W_Fund = 0.25`, `W_Sent = 0.15`
- `ExecutionThreshold = 85`

Regimes (exemplo inicial):

- `spread_alto`: `NoTradeIfSpreadPoints = 120`, `ExecutionThreshold` +5, `W_Tech=0.7`, `W_Fund=0.2`, `W_Sent=0.1`, ação: pular ou reduzir lote 50% se acima do hard limit.
- `vol_alta`: `ExecutionThreshold` +5, manter pesos, exigir confluência de sinais (OB + FVG + sweep) para validar setup.
- `news_window`: janela por evento (10.5), opção padrão reduzir lote 50% ou pular; logar decisão.
- `hub_degraded`: `W_Tech=1.0`, `W_Fund=0`, `W_Sent=0`, `degraded_mode=true`, `ExecutionThreshold` base; continuar apenas com módulos MQL5.

Ponte Python:

- `OnTimerIntervalMs = 200`
- `MaxInFlight = 1`
- `TimeoutMs = 400`
