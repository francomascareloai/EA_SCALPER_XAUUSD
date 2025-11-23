**SEÇÃO 1 – COMPREENSÃO DO PROBLEMA**

- Construir um EA especializado em XAUUSD para contas de prop firm (ex.: FTMO), com foco em scalping inteligente (OB/FVG/liquidez/estrutura/ATR) e risco extremamente controlado.  
- Respeitar regras de avaliação/conta FTMO: limite de perda diária (Max Daily Loss), limite de perda total (Max Total Loss), controle de exposição e evitar grandes rebaixamentos intradiários.  
- Garantir que nenhum trade seja aberto sem: (1) setup técnico forte, (2) score global (Tech+Fund+Sent) acima de um ExecutionThreshold, (3) aprovação explícita do FTMO_RiskManager.  
- Usar arquitetura MQL5 + Python para separar: execução rápida em MQL5 (OnTick < 50ms) e inteligência pesada (notícias, sentimento, LLM) em Python, evitando travar o terminal.  
- Multi-agente em Python permite enriquecer o TechScore (ex.: checar correlações, volatilidade de índices), trazer FundScore (macro, calendário, Fed) e SentScore (notícias, Twitter, options flow) sem “engordar” o código MQL5.  
- Riscos clássicos a mitigar: slippage alto em XAUUSD, spreads alargados em news, overtrading em dias bons, “revenge trading” em dias maus, violar Max Daily Loss/Total Loss por acumular muitas posições pequenas.  
- Transparência: cada trade deve ter uma Reasoning String clara (por quê entrou / não entrou, risco, contexto), para auditoria e aprendizado do trader.  
- O sistema precisa ser testável e validável (backtest/stress test) com simulação fiel das regras de FTMO antes de ir para conta real.  


---

**SEÇÃO 2 – ARQUITETURA DE ALTO NÍVEL (MQL5 + PYTHON)**

**Camadas MQL5**

- Data & Events  
  - `OnTick`: decisão rápida, leitura de preços, checagem mínima de condições, chamado dos módulos técnicos leves.  
  - `OnTimer`: responsável por tarefas mais pesadas/não-críticas (chamadas ao Python Hub, atualização de contexto macro/sentimento, house-keeping).  
  - `OnTradeTransaction`: atualizar métricas de risco (lucro/prejuízo realizado, DD diário, DD total) em tempo real.

- Strategy / Signal Layer  
  - Módulos: `COrderBlockModule`, `CFVGModule`, `CLiquidityModule`, `CMarketStructureModule`, `CVolatilityModule`.  
  - Cada módulo lê dados de preço (timeframes configurados), detecta padrões (OB, FVG, sweeps, estrutura HH/HL/LH/LL, ATR) e retorna sinais normalizados para o Scoring.

- Scoring Engine  
  - `CSignalScoringModule`: transforma os sinais técnicos em um `TechScore 0–100`.  
  - Combina com `FundScore` e `SentScore` vindos de Python para gerar `FinalScore 0–100`.  
  - Apenas se `FinalScore >= ExecutionThreshold` o trade pode prosseguir para etapa de risco.

- Execution & FTMO_RiskManager  
  - `CFTMORiskManager`: monitora equity, DD diário/total, ajusta risco dinâmico (tamanho de lote) e pode bloquear novas entradas.  
  - `CTradeExecutor`: recebe direção, SL/TP, lote, e envia ordens (respeitando filtros de spread, horário, número máximo de posições).  
  - O Risk Manager tem poder de veto final, mesmo com sinal técnico/perfeito.

- Logging & Notifications  
  - `CLogger`: registra eventos importantes (scores, riscos, bloqueios) e Reasoning Strings.  
  - Integração com push notifications/email para sinais importantes, wins/losses, quase-violação de limites de FTMO.

**Python Agent Hub**

- Agentes em Python (exemplos):  
  - `TechnicalAgent`: confirma sinais técnicos (ex.: checa correlações, volatilidade em outros ativos, clusterização de FVGs/OBs).  
  - `FundamentalAgent`: lê calendário econômico, notícias, macro (Fed, CPI, NFP) e retorna `fund_score` e `fund_bias`.  
  - `SentimentAgent`: monitora sentimento (notícias, redes sociais, COT, opções) e retorna `sent_score` e `sent_bias`.  
  - `LLMReasoningAgent`: sintetiza tudo em uma explicação curta (`llm_reasoning_short`) e pode ajustar sub-scores.

- Modo de chamada:  
  - Escolha: **HTTP/REST** via `WebRequest`.  
    - Justificativa: simples de debugar, amplamente suportado, flexível (pode rodar local ou remoto), facilita logs e observabilidade (Flask/FastAPI).  
    - ZeroMQ é excelente para baixa latência e alta taxa de mensagens, mas HTTP/REST já é suficiente para chamadas esporádicas (1–2/ minuto) e mais simples para inicial.

- Formato de resposta do Hub (JSON resumido):  
  - Exemplo:  
    - `tech_subscore_python`: contribuição técnica complementar (0–100).  
    - `fund_score`: score fundamental (0–100).  
    - `fund_bias`: string (`"bullish"`, `"bearish"`, `"neutral"`).  
    - `sent_score`: score de sentimento (0–100).  
    - `sent_bias`: string (`"risk_on"`, `"risk_off"`, etc.).  
    - `llm_reasoning_short`: explicação curta.

**Fluxo de um Tick “perfeito” (passo a passo)**

1. Tick chega em `OnTick` (XAUUSD).  
2. EA atualiza métricas de risco (via `CFTMORiskManager`, baseado em equity atual, dia corrente).  
3. EA chama módulos técnicos (`COrderBlockModule`, `CFVGModule`, `CLiquidityModule`, `CMarketStructureModule`, `CVolatilityModule`) em buffers já preparados.  
4. `CSignalScoringModule` calcula `TechScore` local com base nos sinais.  
5. Periodicamente (ou sob demanda), `OnTimer` envia um POST HTTP:  
   - `MQL5_EA -> HTTP POST -> Python_Hub` com contexto técnico resumido.  
6. `Python_Hub` responde com JSON:  
   - `Python_Hub -> JSON -> MQL5_EA` contendo `tech_subscore_python`, `fund_score`, `sent_score`, `llm_reasoning_short`.  
7. EA combina tudo em `FinalScore = f(TechScore, FundScore, SentScore)`.  
8. Se `FinalScore >= ExecutionThreshold`, a estratégia gera uma proposta de trade (direção, SL/TP, lote base).  
9. EA chama `FTMO_RiskManager.CanOpenTrade(...)`; risk manager aplica regras de Max Daily Loss/Total Loss e ajuste de lote dinâmico.  
10. Se aprovado, `CTradeExecutor` envia ordem; `CLogger` registra Reasoning String e métricas.  


---

**SEÇÃO 3 – DESIGN DETALHADO DO EA EM MQL5**

**Principais módulos/classes**

- `COrderBlockModule`  
  - Responsabilidades: detectar order blocks relevantes (bullish/bearish) em timeframes configurados, marcar zonas de interesse.  
  - Inputs: série de candles (H1/H4/M15), estrutura de mercado atual, parâmetros de profundidade/força de OB.  
  - Outputs: flags (`hasBullishOB`, `hasBearishOB`), preços de OB, confiabilidade (0–1). Contribuição indireta para TechScore.

- `CFVGModule`  
  - Responsabilidades: detectar Fair Value Gaps em múltiplos timeframes, classificar se são zonas de interesse para fill/entry.  
  - Inputs: candles M1–H1, parâmetros de largura mínima de FVG, proximidade do preço atual.  
  - Outputs: `hasFVG`, tipo (`bullish/bearish`), distância em pontos, score local para FVG.

- `CLiquidityModule`  
  - Responsabilidades: identificar pools de liquidez (tops/bottoms limpos, equal highs/lows), sweeps recentes, stop hunts.  
  - Inputs: estrutura de swing highs/lows, volatilidade recente, horário da sessão.  
  - Outputs: flags de `liquidity_sweep` (buy-side/sell-side), localização das pools, risk flags (ex.: “acima de HTF liquidity”).

- `CMarketStructureModule`  
  - Responsabilidades: determinar estrutura de mercado (HH/HL/LH/LL), tendência (bullish/bearish/range), pontos de quebra de estrutura.  
  - Inputs: ponto de swing, fractals, timeframe de tendência (ex.: H1, H4).  
  - Outputs: `bullishTrend` (bool), `market_structure_state` (enum: BOS up, BOS down, range), confiabilidade.

- `CVolatilityModule`  
  - Responsabilidades: medir volatilidade com ATR, spreads, slippage observado, sessões (Asia/London/NY).  
  - Inputs: ATR em diversos timeframes (M5, M15, H1), spread atual, histórico de slippage.  
  - Outputs: `atr_value`, classificação (`low/normal/high vol`), filtros (ex.: bloquear entradas com spread > X pips).

- `CSignalScoringModule`  
  - Responsabilidades: transformar sinais dos módulos em `TechScore 0–100`. Combinar com `FundScore`, `SentScore` para `FinalScore`.  
  - Inputs: flags de OB/FVG/liquidez/estrutura, `atr`, scores de Python.  
  - Outputs: `TechScore`, `FinalScore`, componentes intermediários (para Reasoning String).

- `CFTMORiskManager`  
  - Responsabilidades: monitorar equity, DD diário/total, calcular risco por trade (lote), aplicar política de redução de risco e bloquear entradas quando necessário.  
  - Inputs: equity atual, lucro/prejuízo realizado, parâmetros de Max Daily Loss/Total Loss, risco base por trade, SL em pontos.  
  - Outputs: aprovação de trade (`CanOpenTrade` true/false), lote recomendado, informação de DD atual.

- `CTradeExecutor`  
  - Responsabilidades: enviar ordens Buy/Sell com SL/TP, ajustar slippage, checar spread e filtros de horário.  
  - Inputs: direção, lote, SL/TP, Reasoning String (para log).  
  - Outputs: resultado da execução, tickets, erros tratados.

- `CLogger`  
  - Responsabilidades: logging estruturado e notificação; registro de Reasoning Strings para cada trade/decisão importante.  
  - Inputs: mensagens, scores, status de risco, eventos de trade.  
  - Outputs: logs no Journal/arquivo/push notifications.

**Pseudocódigo do `OnTick` ideal**

```text
void OnTick()
{
    if(_Symbol != "XAUUSD") return;
    if(!IsTradeAllowed())   return;
    if(SpreadMuitoAlto())   return;

    // 1. Atualizar risco (dia, equity, DD)
    riskManager.OnNewTickUpdate();   // recalcula DD diário/total se necessário

    // 2. Atualizar sinais técnicos rápidos (usando dados já pré-carregados)
    signals.hasOB        = obModule.HasValidOB();
    signals.hasFVG       = fvgModule.HasValidFVG();
    signals.liqSweep     = liqModule.GetLatestSweepDirection();
    signals.bullishTrend = msModule.IsBullishTrend();
    signals.atr          = volModule.GetATR();

    // 3. Calcular TechScore local
    double techScoreLocal = scoring.ComputeTechScore(
                                signals.hasOB,
                                signals.hasFVG,
                                signals.bullishTrend,
                                signals.atr
                            );

    // 4. Obter scores de Python (atualizados periodicamente em OnTimer)
    double techSubPy  = context.tech_subscore_python;   // cache atualizado por OnTimer
    double fundScore  = context.fund_score;
    double sentScore  = context.sent_score;

    // 5. Combinar scores
    double finalScore = scoring.ComputeFinalScore(CombineTech(techScoreLocal, techSubPy),
                                                  fundScore,
                                                  sentScore);

    if(finalScore < ExecutionThreshold)
        return;        // Sem sinal forte o suficiente

    // 6. Gerar proposta de trade (direção/SL/TP) com base nos sinais
    TradeProposal proposal = strategy.BuildTradeProposal(signals);
    if(!proposal.is_valid)
        return;

    // 7. Calcular se o risco é aceitável
    double stoplossPoints = proposal.stoploss_points;
    if(!riskManager.CanOpenTrade(RiskPerTradePerc, stoplossPoints))
    {
        logger.Warn("FTMO_RiskManager vetou nova entrada; DD elevado.");
        return;
    }

    // 8. Calcular lote ajustado ao DD
    double lot = riskManager.CalculateLotSize(RiskPerTradePerc, stoplossPoints);
    if(lot <= 0)
        return;

    // 9. Executar trade
    if(tradeExecutor.Execute(proposal.direction, lot, proposal.sl_price, proposal.tp_price))
    {
        logger.Info("Trade executado com sucesso; FinalScore=" + DoubleToString(finalScore, 2));
    }
}
```

Observação de desempenho:  
- Chamadas ao Python **não** são feitas dentro de `OnTick`.  
- `OnTick` usa apenas dados em memória/cache, mantendo o tempo de execução tipicamente < 10–20ms.  
- `OnTimer` roda a cada X segundos para atualizar o contexto vindo de Python.  


---

**SEÇÃO 4 – CÓDIGO MQL5 ESSENCIAL**

Código focado em: EA principal, `CFTMORiskManager`, `CSignalScoringModule`, stubs dos outros módulos e marcações de integração com Python.

```mq5
//+------------------------------------------------------------------+
//|  EA_SCALPER_XAUUSD – Multi-Agent Hybrid System                   |
//+------------------------------------------------------------------+
#property strict
#property copyright "TradeDev_Master"
#property link      ""
#property version   "1.00"
#property description "EA_SCALPER_XAUUSD – XAUUSD Prop-Firm Scalper (OB/FVG/SMC + Risk First)"

#include <Trade\Trade.mqh>

//--- Inputs principais
input double         InpInitialAccountBalance = 100000.0;  // Saldo inicial (para Max Total Loss)
input double         InpRiskPerTradePerc      = 0.40;      // Risco base por trade (% da equity)
input double         InpMaxDailyLossPerc      = 4.0;       // Max Daily Loss (% do saldo inicial do dia)
input double         InpMaxTotalLossPerc      = 10.0;      // Max Total Loss (% do saldo inicial da conta)
input double         InpExecutionThreshold    = 85.0;      // Score mínimo para executar trade
input ENUM_TIMEFRAMES InpTrendTF             = PERIOD_H1;  // Timeframe de tendência
input ENUM_TIMEFRAMES InpSignalTF            = PERIOD_M15; // Timeframe de construção de setup
input ENUM_TIMEFRAMES InpEntryTF             = PERIOD_M5;  // Timeframe de entrada fina

//--- Integração futura com Python
input bool           InpUsePythonHub          = true;
input string         InpPythonHubURL          = "http://127.0.0.1:8000/score";
input int            InpPythonHubTimeoutMs    = 150;       // Timeout curto para não travar

//+------------------------------------------------------------------+
//| Logger simples                                                   |
//+------------------------------------------------------------------+
class CLogger
  {
public:
   void Info(const string msg)  { Print("[INFO] ",  msg); }
   void Warn(const string msg)  { Print("[WARN] ",  msg); }
   void Error(const string msg) { Print("[ERROR] ", msg); }
  };

//+------------------------------------------------------------------+
//| Stubs dos módulos técnicos (a implementar depois)                |
//+------------------------------------------------------------------+
class COrderBlockModule
  {
public:
   bool HasValidOB()
     {
      // TODO: Implementar detecção de Order Blocks relevantes.
      return(false);
     }
  };

class CFVGModule
  {
public:
   bool HasValidFVG()
     {
      // TODO: Implementar detecção de Fair Value Gaps relevantes.
      return(false);
     }
  };

class CLiquidityModule
  {
public:
   int GetLatestSweepDirection()
     {
      // TODO: Retornar direção de liquidity sweep: -1 (sell-side), 1 (buy-side), 0 (nenhum).
      return(0);
     }
  };

class CMarketStructureModule
  {
public:
   bool IsBullishTrend()
     {
      // TODO: Implementar detecção de tendência bullish/bearish com base em HH/HL/LH/LL.
      return(true);
     }
  };

class CVolatilityModule
  {
public:
   double GetATR()
     {
      // TODO: Implementar cálculo de ATR com base no timeframe de entrada/sinal.
      return(1.0);
     }
  };

//+------------------------------------------------------------------+
//| Módulo de Scoring                                                |
//+------------------------------------------------------------------+
class CSignalScoringModule
  {
public:
   // Calcula score técnico simplificado 0–100
   double ComputeTechScore(const bool hasOB,
                           const bool hasFVG,
                           const bool bullishTrend,
                           const double atr)
     {
      double score = 0.0;

      // Exemplo simples de ponderação (ajustar depois):
      if(hasOB)        score += 30.0;
      if(hasFVG)       score += 20.0;
      if(bullishTrend) score += 30.0;

      // Volatilidade (ATR) – penalizar vol muito baixa ou muito alta
      if(atr > 0.0)
        {
         // Este é um placeholder; na prática, calibrar faixas de ATR para XAUUSD.
         score += 20.0;
        }

      // Limitar a 0–100
      if(score < 0.0)   score = 0.0;
      if(score > 100.0) score = 100.0;
      return(score);
     }

   // Combina Tech, Fund e Sentiment em um FinalScore 0–100
   double ComputeFinalScore(const double tech,
                            const double fund,
                            const double sent)
     {
      // Exemplo: 50% técnico, 30% fundamental, 20% sentimento
      double final_score = 0.5 * tech + 0.3 * fund + 0.2 * sent;

      if(final_score < 0.0)   final_score = 0.0;
      if(final_score > 100.0) final_score = 100.0;
      return(final_score);
     }
  };

//+------------------------------------------------------------------+
//| FTMO Risk Manager                                                |
//+------------------------------------------------------------------+
class CFTMORiskManager
  {
private:
   double  m_initial_balance;
   double  m_max_daily_loss_perc;
   double  m_max_total_loss_perc;

   // Níveis internos de "soft" DD diário para ajuste de risco:
   double  m_soft_dd_level1;   // início da redução leve
   double  m_soft_dd_level2;   // redução mais agressiva
   double  m_hard_dd_level;    // região de "quase bloqueio"

   // Controle de dia
   int     m_day;
   int     m_month;
   int     m_year;
   double  m_day_start_equity;

   CLogger *m_logger;

   void    EnsureDailyStart()
     {
      MqlDateTime now;
      TimeToStruct(TimeCurrent(), now);

      if(now.day != m_day || now.mon != m_month || now.year != m_year || m_day_start_equity <= 0.0)
        {
         m_day   = now.day;
         m_month = now.mon;
         m_year  = now.year;
         m_day_start_equity = AccountInfoDouble(ACCOUNT_EQUITY);
         if(m_logger != NULL)
            m_logger.Info("Novo dia detectado. Equity inicial do dia = " +
                          DoubleToString(m_day_start_equity, 2));
        }
     }

public:
                     CFTMORiskManager()
     {
      m_initial_balance     = 0.0;
      m_max_daily_loss_perc = 0.0;
      m_max_total_loss_perc = 0.0;
      m_soft_dd_level1      = 0.0;
      m_soft_dd_level2      = 0.0;
      m_hard_dd_level       = 0.0;
      m_day                 = 0;
      m_month               = 0;
      m_year                = 0;
      m_day_start_equity    = 0.0;
      m_logger              = NULL;
     }

   bool    Init(const double initial_balance,
                const double max_daily_loss_perc,
                const double max_total_loss_perc,
                CLogger *logger)
     {
      m_logger              = logger;
      m_initial_balance     = initial_balance;
      m_max_daily_loss_perc = max_daily_loss_perc;
      m_max_total_loss_perc = max_total_loss_perc;

      // Definir níveis suaves como frações do Max Daily Loss
      m_soft_dd_level1 = 0.25 * m_max_daily_loss_perc; // ex.: 1% se MaxDaily=4%
      m_soft_dd_level2 = 0.60 * m_max_daily_loss_perc; // ex.: 2.4% se MaxDaily=4%
      m_hard_dd_level  = 0.90 * m_max_daily_loss_perc; // ex.: 3.6% se MaxDaily=4%

      EnsureDailyStart();
      if(m_logger != NULL)
        {
         m_logger.Info("FTMO_RiskManager inicializado. MaxDailyLoss=" +
                       DoubleToString(m_max_daily_loss_perc, 2) + "%, MaxTotalLoss=" +
                       DoubleToString(m_max_total_loss_perc, 2) + "%");
        }
      return(true);
     }

   void    OnNewTickUpdate()
     {
      // Apenas garante que o dia está atualizado, equity inicial foi capturada
      EnsureDailyStart();
     }

   double  GetCurrentDailyDDPerc()
     {
      EnsureDailyStart();
      double equity = AccountInfoDouble(ACCOUNT_EQUITY);
      if(m_day_start_equity <= 0.0)
         return(0.0);
      double dd = (m_day_start_equity - equity) / m_day_start_equity * 100.0;
      if(dd < 0.0) dd = 0.0;   // DD diário só considera perdas
      return(dd);
     }

   double  GetCurrentTotalDDPerc()
     {
      double equity = AccountInfoDouble(ACCOUNT_EQUITY);
      if(m_initial_balance <= 0.0)
         return(0.0);
      double dd = (m_initial_balance - equity) / m_initial_balance * 100.0;
      if(dd < 0.0) dd = 0.0;
      return(dd);
     }

   // Ajusta risco base de acordo com DD diário
   double  AdjustRiskByDailyDD(const double base_risk_perc)
     {
      double dd = GetCurrentDailyDDPerc();
      double factor = 1.0;

      if(dd >= m_hard_dd_level)
        {
         // Próximo do limite de Max Daily Loss: praticamente bloquear risco
         factor = 0.0;
        }
      else if(dd >= m_soft_dd_level2)
        {
         // DD moderado-alto: reduzir risco drasticamente
         factor = 0.25;
        }
      else if(dd >= m_soft_dd_level1)
        {
         // DD leve: reduzir risco pela metade
         factor = 0.50;
        }
      else
        {
         factor = 1.0;
        }

      double adjusted = base_risk_perc * factor;
      return(adjusted);
     }

   // Cálculo de lote baseado em risco % da equity e stoploss em pontos
   double  CalculateLotSize(const double risk_perc,
                            const double stoploss_points)
     {
      if(stoploss_points <= 0.0)
         return(0.0);

      double equity = AccountInfoDouble(ACCOUNT_EQUITY);
      double adjusted_risk = AdjustRiskByDailyDD(risk_perc);

      if(adjusted_risk <= 0.0)
         return(0.0);

      double risk_amount = equity * adjusted_risk / 100.0;

      double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
      double tick_size  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
      double point      = _Point;

      if(tick_value <= 0.0 || tick_size <= 0.0 || point <= 0.0)
         return(0.0);

      // Valor por ponto por 1 lote
      double value_per_point_per_lot = tick_value * (point / tick_size);

      double lot = risk_amount / (stoploss_points * value_per_point_per_lot);

      // Ajustar para limites do símbolo
      double min_lot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
      double max_lot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
      double lot_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

      if(lot < min_lot) lot = min_lot;
      if(lot > max_lot) lot = max_lot;

      lot = MathFloor(lot / lot_step) * lot_step;
      lot = NormalizeDouble(lot, (int)SymbolInfoInteger(_Symbol, SYMBOL_VOLUME_DIGITS));

      return(lot);
     }

   // Verifica se pode abrir trade, dado risco base (%) e stoploss em pontos
   bool    CanOpenTrade(const double risk_perc,
                        const double stoploss_points)
     {
      EnsureDailyStart();

      double daily_dd  = GetCurrentDailyDDPerc();
      double total_dd  = GetCurrentTotalDDPerc();
      double equity    = AccountInfoDouble(ACCOUNT_EQUITY);

      // Risco potencial máximo adicional (aproximação: risco_perc ajustado)
      double adjusted_risk = AdjustRiskByDailyDD(risk_perc);

      // Se o ajuste resultou em 0, não deve abrir trade
      if(adjusted_risk <= 0.0)
        {
         if(m_logger != NULL)
            m_logger.Warn("Risco ajustado é 0. Bloqueando novas entradas (DD diário elevado).");
         return(false);
        }

      // Avaliar se o trade pode fazer o DD diário exceder o limite
      double potential_daily_dd = daily_dd + adjusted_risk;
      if(potential_daily_dd > m_max_daily_loss_perc)
        {
         if(m_logger != NULL)
            m_logger.Warn("Trade vetado: potencial DD diário (" +
                          DoubleToString(potential_daily_dd, 2) +
                          "%) excede MaxDailyLoss.");
         return(false);
        }

      // Avaliar se o trade pode fazer o DD total exceder o limite
      double potential_total_dd = total_dd + adjusted_risk;
      if(potential_total_dd > m_max_total_loss_perc)
        {
         if(m_logger != NULL)
            m_logger.Warn("Trade vetado: potencial DD total (" +
                          DoubleToString(potential_total_dd, 2) +
                          "%) excede MaxTotalLoss.");
         return(false);
        }

      // Opcional: pode-se adicionar checagem de número máximo de trades/dia, etc.

      return(true);
     }
  };

//+------------------------------------------------------------------+
//| Executor de trades (stub)                                       |
//+------------------------------------------------------------------+
class CTradeExecutor
  {
private:
   CTrade  m_trade;
   CLogger *m_logger;

public:
   void SetLogger(CLogger *logger)
     {
      m_logger = logger;
     }

   bool ExecuteBuy(const double lot,
                   const double sl_price,
                   const double tp_price,
                   const string reason)
     {
      // TODO: Implementar checagens de spread, horário, etc.
      if(m_logger != NULL)
         m_logger.Info("Executando BUY XAUUSD, lot=" +
                       DoubleToString(lot, 2) + ", SL=" +
                       DoubleToString(sl_price, _Digits) + ", TP=" +
                       DoubleToString(tp_price, _Digits) + ". Motivo: " + reason);
      // Placeholder: não envia ordem real ainda
      // m_trade.Buy(lot, _Symbol, 0.0, sl_price, tp_price, reason);
      return(true);
     }

   // Métodos para SELL, fechamentos, etc. seriam adicionados aqui.
  };

//+------------------------------------------------------------------+
//| Contexto compartilhado p/ integração Python (cache)              |
//+------------------------------------------------------------------+
struct SPythonContext
  {
   double tech_subscore_python;
   double fund_score;
   double sent_score;
   string llm_reasoning_short;
  };

//+------------------------------------------------------------------+
//| Variáveis globais do EA                                          |
//+------------------------------------------------------------------+
CLogger               g_logger;
COrderBlockModule     g_ob;
CFVGModule            g_fvg;
CLiquidityModule      g_liq;
CMarketStructureModule g_ms;
CVolatilityModule     g_vol;
CSignalScoringModule  g_scoring;
CFTMORiskManager      g_risk;
CTradeExecutor        g_executor;

SPythonContext        g_py_context;

//+------------------------------------------------------------------+
//| Stub de chamada ao Python (implementar depois com WebRequest)    |
//+------------------------------------------------------------------+
bool CallPythonHub(double &tech_subscore_py,
                   double &fund_score,
                   double &sent_score)
  {
   // TODO: Implementar uso de WebRequest com JSON para InpPythonHubURL.
   // IMPORTANTE: WebRequest NÃO deve ser chamada em todos os ticks;
   // o ideal é chamá-la em OnTimer ou em ticks espaçados (ex.: 1 vez por minuto).
   // Em caso de falha/timeout, a função deve retornar false e o EA operar em modo seguro.
   return(false);
  }

//+------------------------------------------------------------------+
//| Funções padrão do EA                                             |
//+------------------------------------------------------------------+
int OnInit()
  {
   g_logger.Info("Inicializando EA_SCALPER_XAUUSD...");

   g_executor.SetLogger(&g_logger);

   g_py_context.tech_subscore_python = 0.0;
   g_py_context.fund_score           = 50.0; // neutro
   g_py_context.sent_score           = 50.0; // neutro
   g_py_context.llm_reasoning_short  = "";

   g_risk.Init(InpInitialAccountBalance,
               InpMaxDailyLossPerc,
               InpMaxTotalLossPerc,
               &g_logger);

   // Exemplo: configurar timer para chamadas periódicas ao Python (a cada 60s)
   EventSetTimer(60);

   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason)
  {
   EventKillTimer();
   g_logger.Info("Desinicializando EA_SCALPER_XAUUSD. Motivo=" + IntegerToString(reason));
  }

//+------------------------------------------------------------------+
//| OnTimer – ideal para chamar Python Hub sem travar OnTick         |
//+------------------------------------------------------------------+
void OnTimer()
  {
   if(!InpUsePythonHub)
      return;

   double tech_py  = 0.0;
   double fund     = 0.0;
   double sent     = 0.0;

   bool ok = CallPythonHub(tech_py, fund, sent);
   if(ok)
     {
      g_py_context.tech_subscore_python = tech_py;
      g_py_context.fund_score           = fund;
      g_py_context.sent_score           = sent;
      // g_py_context.llm_reasoning_short seria preenchido no futuro a partir do JSON.
      g_logger.Info("PythonHub atualizado com sucesso.");
     }
   else
     {
      // Em caso de falha, manter contexto anterior e operar com dados locais
      g_logger.Warn("Falha ao chamar PythonHub. Operando com contextos anteriores.");
     }
  }

//+------------------------------------------------------------------+
//| OnTick – lógica principal rápida                                 |
//+------------------------------------------------------------------+
void OnTick()
  {
   if(_Symbol != "XAUUSD")
      return;

   // Checar se negociação está permitida e spread está aceitável
   if(!IsTradeAllowed())
      return;

   // Exemplo bem simples de filtro de spread
   double spread_points = (SymbolInfoDouble(_Symbol, SYMBOL_ASK) -
                           SymbolInfoDouble(_Symbol, SYMBOL_BID)) / _Point;
   if(spread_points > 50) // placeholder: 50 pontos (~5 pips em ouro com 1 dígito extra)
     {
      g_logger.Warn("Spread muito alto. Ignorando tick.");
      return;
     }

   // Atualizar dados de risco (novo dia, equity, etc.)
   g_risk.OnNewTickUpdate();

   // Ler sinais dos módulos técnicos (neste estágio ainda são stubs)
   bool   hasOB        = g_ob.HasValidOB();
   bool   hasFVG       = g_fvg.HasValidFVG();
   int    liqSweepDir  = g_liq.GetLatestSweepDirection();
   bool   bullishTrend = g_ms.IsBullishTrend();
   double atr          = g_vol.GetATR();

   // Calcular TechScore local
   double techScoreLocal = g_scoring.ComputeTechScore(hasOB, hasFVG, bullishTrend, atr);

   // Combinar com scores de Python (cache)
   double tech_combined = techScoreLocal;
   if(InpUsePythonHub)
     {
      // Exemplo: combinar tech local com tech_subscore_python de forma simples
      tech_combined = 0.7 * techScoreLocal + 0.3 * g_py_context.tech_subscore_python;
     }

   double finalScore = g_scoring.ComputeFinalScore(tech_combined,
                                                   g_py_context.fund_score,
                                                   g_py_context.sent_score);

   if(finalScore < InpExecutionThreshold)
      return; // Sem sinal suficientemente forte

   // Placeholder: construir proposta de trade simplificada (ex.: apenas BUY se tendência bullish)
   bool   directionBuy = bullishTrend; // extremamente simplificado
   double sl_points    = 200.0;        // placeholder – na prática, derivar do OB/FVG/estrutura
   double tp_points    = 400.0;        // placeholder – relação R:R 1:2, por exemplo

   // Consultar Risk Manager
   if(!g_risk.CanOpenTrade(InpRiskPerTradePerc, sl_points))
      return;

   double lot = g_risk.CalculateLotSize(InpRiskPerTradePerc, sl_points);
   if(lot <= 0.0)
      return;

   // Calcular preços de SL/TP (simples, apenas para exemplo)
   double price      = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double sl_price   = 0.0;
   double tp_price   = 0.0;

   if(directionBuy)
     {
      sl_price = price - sl_points * _Point;
      tp_price = price + tp_points * _Point;
     }
   else
     {
      price    = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      sl_price = price + sl_points * _Point;
      tp_price = price - tp_points * _Point;
     }

   string reason = "FinalScore=" + DoubleToString(finalScore, 2) +
                   ", TechLocal=" + DoubleToString(techScoreLocal, 2) +
                   ", TechPy=" + DoubleToString(g_py_context.tech_subscore_python, 2) +
                   ", Fund=" + DoubleToString(g_py_context.fund_score, 2) +
                   ", Sent=" + DoubleToString(g_py_context.sent_score, 2);

   if(directionBuy)
      g_executor.ExecuteBuy(lot, sl_price, tp_price, reason);
   else
     {
      // TODO: implementar ExecuteSell similar
      g_logger.Info("Sinal SELL detectado, mas ExecuteSell ainda não implementado. Reason: " + reason);
     }
  }
```

Comentários importantes:  
- A integração com Python está só marcada via `CallPythonHub()` e `OnTimer`; o corpo de `CallPythonHub()` será implementado depois com `WebRequest`.  
- O Risk Manager já aplica lógica de ajuste de risco dinâmico conforme DD diário e bloqueia trades se ultrapassaria Max Daily/Total Loss, alinhado com FTMO.  
- Os módulos técnicos são stubs, focando aqui na arquitetura e no gerenciador de risco.  


---

**SEÇÃO 5 – INTERFACE COM PYTHON AGENT HUB**

**Formato do request JSON enviado pelo EA (exemplo)**

Campos mínimos (podem ser expandidos depois):

```json
{
  "symbol": "XAUUSD",
  "timeframe_trend": "H1",
  "timeframe_signal": "M15",
  "timeframe_entry": "M5",
  "timestamp_utc": "2025-11-22T14:30:00Z",
  "bid": 2415.10,
  "ask": 2415.20,
  "has_ob": true,
  "has_fvg": true,
  "liquidity_sweep": "buy_side",
  "market_structure": "bullish",
  "atr": 3.5,
  "local_tech_score": 72.0
}
```

**Formato de response JSON esperado do Hub**

```json
{
  "tech_subscore_python": 12.0,
  "fund_score": 65.0,
  "fund_bias": "slightly_bearish",
  "sent_score": 58.0,
  "sent_bias": "mild_risk_on",
  "llm_reasoning_short": "NY session, gold reacting to dovish Fed remarks; bullish liquidity grab below London low, but NFP later so reduce size."
}
```

**Função em pseudocódigo MQL5 para `CallPythonHub`**

```text
bool CallPythonHub(double &tech_subscore_py,
                   double &fund_score,
                   double &sent_score)
{
    string url      = InpPythonHubURL;
    string headers  = "Content-Type: application/json\r\n";
    char   data[];
    char   result[];
    string response_headers;
    string body;

    // 1. Montar JSON de request com contexto atual
    string ts = TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS);
    body = StringFormat(
        "{\"symbol\":\"%s\",\"timeframe_trend\":\"%s\",\"timeframe_signal\":\"%s\","
        "\"timeframe_entry\":\"%s\",\"timestamp_utc\":\"%s\",\"bid\":%.2f,\"ask\":%.2f,"
        "\"has_ob\":%s,\"has_fvg\":%s,\"market_structure\":\"%s\",\"atr\":%.4f,"
        "\"local_tech_score\":%.2f}",
        _Symbol,
        EnumToString(InpTrendTF),
        EnumToString(InpSignalTF),
        EnumToString(InpEntryTF),
        ts,
        SymbolInfoDouble(_Symbol, SYMBOL_BID),
        SymbolInfoDouble(_Symbol, SYMBOL_ASK),
        hasOB ? "true" : "false",
        hasFVG ? "true" : "false",
        bullishTrend ? "bullish" : "bearish",
        atr,
        localTechScore
    );

    // 2. Converter body para array de char
    StringToCharArray(body, data, 0, WHOLE_ARRAY, CP_UTF8);

    // 3. Chamar WebRequest (POST)
    int status = WebRequest("POST", url, headers, InpPythonHubTimeoutMs,
                            data, result, response_headers);
    if(status != 200)
    {
        // Falha de rede / timeout / status != 200 -> modo seguro
        tech_subscore_py = 0.0;
        fund_score       = 50.0; // neutro
        sent_score       = 50.0;
        return false;
    }

    // 4. Converter resposta para string e parsear JSON
    string resp_str = CharArrayToString(result);
    // Aqui usaria-se um parser JSON (ex.: CJA e libs externas).
    // Pseudocódigo:
    //   json = ParseJSON(resp_str);
    //   tech_subscore_py = json["tech_subscore_python"];
    //   fund_score       = json["fund_score"];
    //   sent_score       = json["sent_score"];
    //   g_py_context.llm_reasoning_short = json["llm_reasoning_short"];

    return true;
}
```

Pontos-chave:  
- `WebRequest` exige que a URL esteja autorizada nas configurações do terminal.  
- Em caso de falha/timeout, o EA **não para**, apenas opera com dados locais (TechScore MQL5) e scores neutros de fund/sentimento.  
- `CallPythonHub` deve ser usada em `OnTimer`, não em todos os ticks.  


---

**SEÇÃO 6 – RACIOCÍNIO DE RISCO (FTMO) & DEEP THINKING**

**Configuração sugerida para conta FTMO 100k (XAUUSD scalping)**

Para uma conta de 100k (regras típicas FTMO: Max Daily Loss 5%, Max Total Loss 10%):

- Risk per trade %: **0.30–0.40%**  
  - Sugestão concreta: **0.40%** em condições normais.  
  - Em ouro, movimentos de 50–150 pips são comuns; com 0.4% você suporta sequências ruins sem chegar perto dos limites.

- Soft Daily Loss % (zona de redução de risco): **2%**  
  - A partir de 2% de perda no dia, reduzir risco.  
  - Evita que o dia “desande” e chegue rápido em -5%.

- Hard Max Daily Loss % (limite interno, mais conservador que FTMO): **4%**  
  - FTMO permitiria -5%, mas usar **-4%** como “hard” interno é prudente.  
  - Em -4%, o EA deve **parar de abrir novas entradas** (apenas gerenciar posições abertas, se houver).

- Max Total Loss % (limite interno): **8–9%**, menor que os 10% da FTMO  
  - Sugestão concreta: **8%**.  
  - Isso dá margem se um dia for muito ruim ou se houver gap/slippage extremo.

**Política de redução de risco dinâmica (DD diário)**

- 0–1% DD diário → risco normal  
  - Risk per trade = 0.40%.  
  - Mercado dentro da “zona saudável”; o sistema pode operar normalmente.

- 1–2.5% DD diário → risco reduzido  
  - Risk per trade ≈ 0.20% (metade).  
  - O EA entra em modo “defensivo”, tentando recuperar com cautela; a prioridade passa a ser **sobrevivência**, não agressão.

- 2.5–4% DD diário → risco mínimo  
  - Risk per trade ≈ 0.10% ou até 0.05%.  
  - Toda nova entrada deve ser extremamente filtrada; praticamente só setups A+ (score muito alto).  
  - Estratégia: é melhor sair do dia ligeiramente negativo do que tentar “voltar para o zero” e arriscar violar a regra.

- ≥ 4% DD diário → bloquear novas entradas  
  - EA não abre novas posições, apenas gerencia as que restarem.  
  - Garante que o trader **nunca** bata no Max Daily Loss da FTMO (5%) no backtest/real, salvo eventos extremos (gap descontrolado).

**Como evitar overtrading num dia bom**

- Definir alvo de lucro diário suave: ex., **+3–4%** de gain no dia.  
  - Ao atingir +3%, reduzir risco pela metade; ao atingir +4%, parar de abrir novas operações.  
  - Lógica: em prop firm, capital preservado é tão importante quanto lucro; entregar 3–4% num dia é excelente.

- Limitar número de trades por sessão/dia:  
  - Ex.: máximo 8–10 trades por dia; 3–4 trades na sessão de Londres, 3–4 na sessão de NY.  
  - Isso evita que, após um bom começo, o EA “devolva” tudo em overtrading.

- Exigir qualidade crescente dos setups:  
  - Depois de X trades ganhadores, aumentar temporariamente o ExecutionThreshold (ex.: de 85 para 90).  
  - O EA só continua operando se aparecer algo realmente extremo, filtrando setups medianos.

**Como lidar com sequência de 3 stops seguidos em XAUUSD**

- Regra de “3 strikes”:  
  - Se ocorrerem **3 stops consecutivos** no mesmo dia, pausar novas entradas por um período (ex.: 2–3 horas) ou até a próxima sessão.  
  - Isso ajuda a evitar o “spiral” de mercado difícil e a entrada na espiral emocional (mesmo num EA, o trader pode forçar parâmetros).

- Após 3 stops:  
  - Reduzir risco por trade para 50% do valor original pelo restante do dia.  
  - Aumentar ExecutionThreshold em 5–10 pontos (ex.: de 85 → 90–95) para o resto do dia.  
  - Interpretar isso como “condição de mercado ruim para a estratégia” – o EA precisa de filtros mais rígidos.

**Quando é melhor não operar, mesmo com setup técnico bom**

- Eventos macro de alto impacto:  
  - 30–60 minutos antes de NFP, FOMC, CPI, decisões de taxa de juros.  
  - O FundamentalAgent em Python deveria sinalizar `fund_bias = "high_risk_event"` e o EA:  
    - reduz risco para 0.1% ou 0%; ou  
    - simplesmente não abre novas operações nesse período.

- Spreads e liquidez:  
  - Se spread médio atual > threshold (ex.: > 50–70 pontos em XAUUSD) ou se houver saltos de spread muito rápidos, é sinal claro de liquidez pobre.  
  - Mesmo com OB/FVG perfeitos, a execução piora (slippage, fills ruins), destruindo R:R real.

- Estrutura de mercado confusa (chop):  
  - Se `CMarketStructureModule` detectar alternância rápida entre BOS up/down e ATR muito baixo, o mercado está “travado”.  
  - Melhor não operar: setups “lindos” em range micro frequentemente viram stop em ouro.

- Fator psicológico do trader (mercado real):  
  - Após um grande dia de lucro (ex.: > 5% no mês), faz sentido ficar mais conservador ou pausar para consolidar o psicológico.  
  - O EA pode ter um modo “capital protegido” quando o saldo da conta está acima de uma meta mensal.  


---

**SEÇÃO 7 – ESTRATÉGIA DE TESTES E VALIDAÇÃO**

**Backtests**

- Período e data range:  
  - Mínimo de **12–24 meses** de dados em XAUUSD, incluindo ciclos diferentes (alta, baixa, range), eventos macro fortes.  
  - Ideal: cobrir pelo menos 2 anos recentes (ex.: 2022–2024) para capturar diferentes regimes de volatilidade.

- Timeframes:  
  - Teste em M1 (modelo de tick) para simular bem scalping.  
  - Estratégia observa H1/M15/M5, mas backtest em M1 garante melhor precisão nas execuções/SL/TP.

- Qualidade de tick:  
  - Buscar **modelagem de 99%** (dados de tick reais, se possível).  
  - Spread variável, não fixo; XAUUSD sofre muito com spread/volatilidade em news.

**Stress tests**

- Spreads maiores:  
  - Rodar séries de backtests com spread multiplicado (ex.: 1.5x, 2x) para ver se a estratégia ainda é lucrativa ou ao menos não destrói a conta.  

- Slippage:  
  - Simular slippage de 10–30 pontos (1–3 pips) para entrada e saída, especialmente em horários de news.  
  - Avaliar se R:R nominal (1:2) se mantém com slippage ou cai para algo perigoso (1:1 real).

- News on/off:  
  - Uma bateria de testes com filtro de notícias ligado (não operar perto de high impact) e outra sem filtro.  
  - Comparar PF, DD, e quantidade de violação/quase-violação de Max Daily Loss.

**Testes específicos de FTMO**

- Simular Max Daily Loss e Max Total Loss:  
  - Usar o próprio `CFTMORiskManager` durante o backtest, logando:  
    - DD diário máximo por dia.  
    - DD total máximo durante o período.  
  - Criar logs/dumps com: data, DD diário, se o EA teria bloqueado novas entradas.

- Avaliar respeito às regras:  
  - Verificar se, em nenhum momento, o DD diário interno passa do limite interno (4%) no backtest (salvo exceções justificadas por eventos extremos).  
  - Em casos raros de spikes extremos (gaps, slippage muito além do modelado), manualmente inspecionar se a violação seria inevitável.

**Critérios de aprovação**

- Métricas de performance mínimas:  
  - Profit Factor (PF) ≥ 1.5 em 12–24 meses.  
  - Drawdown máximo (equity) ≤ 8% (inferior ao Max Total Loss de 10% da FTMO).  
  - Win rate adequado para scalper (ex.: 45–60%) com bom R:R (> 1:1.5).  
  - Número de dias com perda > 3% deve ser muito baixo.

- Limites de violação:  
  - Nenhum dia com DD diário ≥ limite interno (4%) no backtest (salvo exceções justificadas por eventos extremos).  
  - Pouquíssimos dias (por exemplo, < 2–3% dos dias) chegando a DD diário entre 3–4%.  
  - Log adicional: quantos dias teriam chegado perto de violar a FTMO (ex.: > 4.5% de DD num dia) em simulações com slippage extremo.

- Robustez:  
  - Re-testar com parâmetros ligeiramente perturbados (risco, thresholds, ATRs) para garantir que o sistema não é hiper-ajustado.  
  - Verificar consistência por ano, por trimestre e por sessão (Asia/London/NY).  


---

**SEÇÃO 8 – EXEMPLOS DE REASONING STRINGS DE TRADES**

Exemplo 1 – Trade WIN (BUY XAUUSD)  
"NY sessão, XAUUSD em tendência de alta em H1 com BOS recente, liquidez varrida abaixo da mínima de Londres e OB bullish respeitado em M15. TechScore+Python e fund_score (pós-FOMC dovish) indicaram alta probabilidade de continuação, com ATR elevado porém dentro da faixa aceitável e spread normal. Risco foi definido em 0.40% com SL abaixo da zona de liquidez varrida e TP em alvo de extensão de FVG, respeitando limites de Max Daily/Total Loss. A entrada foi consistente com a política de risco e o preço atingiu o TP rapidamente, consolidando lucro sem expor a conta a overtrading adicional no dia."

Exemplo 2 – Trade LOSS (SELL XAUUSD)  
"Londres tardia, estrutura em H1 começando a mostrar fraqueza após falha de continuação de alta, com sweep acima de máxima asiática e OB bearish recém-formado em M15. TechScore alto para venda, mas ATR moderado e fund_score neutro permitiram arriscar 0.30% com SL acima da máxima da sessão e TP em zona de liquidez anterior; spread e slippage previstos estavam dentro dos limites. O preço invalidou o OB e acionou o SL, aumentando o DD diário dentro da faixa de 1–2.5%, o que acionou redução automática de risco para trades seguintes. Apesar do loss, a decisão foi consistente com a política de risco e com o contexto técnico/fundamental disponível na hora."

Exemplo 3 – Sinal IGNORADO (score alto mas risco FTMO próximo do limite)  
"NY sessão, XAUUSD com forte tendência de alta, OB bullish alinhado em H1 e M15, FVG de continuação em M5 e TechScore+Python acima de 90, configurando setup A+. No entanto, o DD diário já estava em 3.8% após sequência difícil na manhã, próximo do limite interno de 4% e distante apenas ~1.2% do Max Daily Loss permitido pela prop firm. O FTMO_RiskManager reduziu o risco para 0.10% e, pela política de proteção, vetou novas entradas acima de 3.5–4% de DD diário, priorizando a preservação da conta. A decisão de NÃO entrar foi totalmente alinhada com o plano de risco: o setup era bom, mas o contexto de risco tornava a operação inaceitável em conta de prop."

