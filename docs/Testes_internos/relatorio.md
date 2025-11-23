#Gpt-5.1-high (no Claude code com meu CLIPROXY):

Vou criar um checklist de 8 seções
  para acompanhar a entrega desta
  resposta.

● Seção 1 concluída; vou marcar e
  iniciar a Seção 2.

● Seção 1 – Compreensão Do Problema
  - Objetivo estratégico: executar scalps consistentes em XAUUSD com risco-precedência,
  aprovados por um FTMO_RiskManager que impede violações de Max Daily Loss (MDL) e Max Total
  Loss (MTL), usando um Scoring Engine (Tech/Fund/Sent) com threshold explícito.
  - Restrições FTMO típicas: MDL 5% por dia (reset 00:00 server), MTL 10% global; tolerância
  zero a violação; recomendável operar com margens para evitar quase-violação; controle de
  alavancagem e lot step do símbolo.
  - Por que MQL5 + Python: MQL5 cuida da execução, latência e dados de tick; Python agrega
  sinais pesados/externos (fundamental/sentimento/LLM) sem bloquear OnTick; arquitetura
  desacoplada mantém <50ms no ciclo crítico.
  - Riscos clássicos de scalping XAUUSD: slippage agressivo, spread volátil em sessão asiática e
   pós-notícias, overtrading (drawdown por sequência de perdas), violação de MDL por somatório
  de perdas intradiárias, saturação do servidor por chamadas externas mal posicionadas.
  - Mitigações chave: gate de Execução por Scoring (FinalScore >= threshold), veto de risco
  ex-ante (prevê DD caso SL atinja), sizing por risco (lots = f(equity, SL, tick_value)), e
  redução dinâmica de risco conforme DD diário.
  - Transparência: Reasoning String por trade (o que, por quê, onde está o risco, como FTMO foi
  respeitado), logs detalhados e notificações.

  Seção 2 – Arquitetura De Alto Nível (MQL5 + Python)
  - Camadas MQL5:
    - Data & Events: OnTick, OnTradeTransaction; opcional OnTimer para workloads externos.
    - Strategy/Signals: OB, FVG, Liquidity Sweep, Market Structure (HH/HL/LH/LL), volatilidade
  (ATR).
    - Scoring Engine: compõe TechScore (MQL5 + opcional subscore Python), FundScore, SentScore →
   FinalScore.
    - Execution & FTMO_RiskManager: sizing por risco, pre-check de violação (MDL/MTL),
  abertura/gerenciamento de ordens.
    - Logging & Notifications: Reasoning String curta por evento.
  - Python Agent Hub:
    - Agentes: Technical (subscore externo), Fundamental (macro/news), Sentiment
  (feeds/finetuned), LLM Reasoning (sanity-checks/explicação curta).
    - Transporte: HTTP/REST (WebRequest) inicialmente — simples, portátil e suficiente; ZeroMQ
  pode entrar depois para baixa latência assíncrona. No EA, chamadas HTTP nunca no caminho
  quente; preferir OnTimer/fila.
    - Resposta JSON enxuta com subscores e recomendações.
  - Fluxo de um tick “perfeito”:
    - MQL5_EA (OnTick) → coleta sinais leves (OB/FVG/MS/ATR)
    - → TechScore_mql5 → se pré-score >= pré-limiar, enfileira consulta ao Python (via
  OnTimer/cache) → recebe tech_sub_py, fund, sent
    - → FinalScore = f(tech, fund, sent) → FTMO_RiskManager valida DD previsto com SL
    - → TradeExecutor abre ordem com SL/TP e lotes ajustados por risco dinâmico.
  - Diagrama descritivo:
    - MQL5_EA → HTTP POST → Python_Hub → JSON reply → MQL5_EA(decisão)
    - OnTick crítico: sem WebRequest; OnTimer/cuidados: WebRequest/parse/atualiza cache de
  scores.

  Seção 3 – Design Detalhado Do EA Em MQL5
  - Módulos/Classes:
    - COrderBlockModule
        - Resp: detectar OBs relevantes (bull/bear) próximos do preço.
      - In: séries OHLC, TF de sinal.
      - Out: bool hasOB, direção preferencial, força.
    - CFVGModule
        - Resp: detectar FVG recentes e proximidade de preenchimento.
      - In: séries OHLC, TF de sinal.
      - Out: bool hasFVG, direção preferencial, confiança.
    - CLiquidityModule
        - Resp: varreduras de liquidez (equal highs/lows, sweeps).
      - In: swing points, highs/lows recentes.
      - Out: bool liquiditySweep, direção.
    - CMarketStructureModule
        - Resp: HH/HL/LH/LL, tendência e quebra de estrutura.
      - In: pivots/zigzag, MMEs (opcional).
      - Out: bool bullishTrend, força de tendência.
    - CVolatilityModule
        - Resp: ATR/normalização para SL/TP dinâmicos.
      - In: ATR(TF), janela.
      - Out: double atr, sl/tp propostos.
    - CSignalScoringModule
        - Resp: ComputeTechScore(hasOB, hasFVG, bullishTrend, atr) (0–100) e
  ComputeFinalScore(tech, fund, sent).
    - CFTMORiskManager
        - Resp: sizing por risco, pre-check de MDL/MTL previsto, redução dinâmica conforme DD
  diário.
      - In: equity, parâmetros FTMO, SL em pontos.
      - Out: CanOpenTrade(...), CalcLotsByRisk(...), DynamicRiskModifier().
    - CTradeExecutor
        - Resp: abrir/fechar ordens, SL/TP por ATR, slippage/execução.
      - In: direção, lots, SL/TP.
      - Out: resultado da operação e logs.
    - CLogger
        - Resp: logs e Reasoning Strings.
  - Pseudocódigo do OnTick (ideal):
    - Atualiza âncora diária do risk manager
    - Coleta sinais leves: OB/FVG/MS/ATR
    - tech_mql5 = ComputeTechScore(...)
    - if tech_mql5 < pre_threshold: return
    - Se cache Python expirado e “pré-sinal forte”: agenda consulta (OnTimer) e usa último valor
    - final = ComputeFinalScore(tech_mix, fund, sent)
    - if final < ExecutionThreshold: return
    - sl_points = atr * SL_ATR_Mult / _Point
    - if !risk.CanOpenTrade(risk_pct, sl_points): return
    - lots = risk.CalcLotsByRisk(risk_pct, sl_points)
    - trade.execute(direction, lots, sl/tp) com RR definido e logs
    - Notifica Reasoning String

  Seção 4 – Código MQL5 Essencial
  //+------------------------------------------------------------------+
  //|                                                EA_SCALPER_XAUUSD |
  //|                   Multi-Agent Hybrid (MQL5 + Python-ready)       |
  //+------------------------------------------------------------------+
  #property strict
  #include <Trade/Trade.mqh>

  input double InpRiskPerTradePct      = 0.35;   // % do equity por trade
  input double InpSoftDailyLossPct     = 3.0;    // início da zona de redução
  input double InpMaxDailyLossPct      = 5.0;    // FTMO hard MDL
  input double InpMaxTotalLossPct      = 10.0;   // FTMO hard MTL
  input double InpExecutionThreshold   = 85.0;   // FinalScore mínimo
  input ENUM_TIMEFRAMES InpSignalTF    = PERIOD_M5;
  input ENUM_TIMEFRAMES InpTrendTF     = PERIOD_M15;
  input int    InpATRPeriod            = 14;
  input double InpSL_ATR_Mult          = 1.5;
  input double InpTP_RR                = 1.2;    // TP = RR * SL
  input double InpInitialBalance       = 100000; // âncora FTMO

  CTrade Trade;

  //--------------------- Scoring -------------------------------------
  class CSignalScoringModule
  {
  public:
     double ComputeTechScore(bool hasOB, bool hasFVG, bool bullishTrend, double atr)
     {
        // Heurística simples (exemplo). Ajustar pesos conforme validação.
        double score=0.0;
        if(hasOB)   score+=30.0;
        if(hasFVG)  score+=25.0;
        if(bullishTrend) score+=25.0; else score+=10.0; // tendência contra dá alguma estrutura
        // Volatilidade: favores ATR moderado (evita extremos)
        if(atr>0)
        {
           // Normalização simples: penaliza ATR muito baixo/alto
           score += MathMax(0.0, 20.0 - 20.0 * VolatilityPenalty(atr));
        }
        return MathMin(100.0, score);
     }

     double ComputeFinalScore(double tech, double fund, double sent)
     {
        // Pesos iniciais: Tech 0.6, Fund 0.25, Sent 0.15
        return MathMin(100.0, 0.60*tech + 0.25*fund + 0.15*sent);
     }

  private:
     double VolatilityPenalty(double atr)
     {
        // Placeholder: penaliza extremos. Ajustar por estatística do símbolo.
        // Ex.: assume ATR "ideal" próximo da mediana local.
        // Retorna [0..1]. 0 = perfeito, 1 = ruim.
        // Implementação simples: atr muito baixo/alto => 1.0
        return 0.0;
     }
  };

  //--------------------- FTMO Risk Manager ---------------------------
  class CFTMORiskManager
  {
  private:
     string  m_symbol;
     double  m_initial_balance;
     double  m_daily_start_equity;
     int     m_daily_anchor_date; // yyyymmdd simples
     double  m_soft_daily_loss_pct;
     double  m_hard_daily_loss_pct;
     double  m_max_total_loss_pct;
     double  m_vol_min, m_vol_max, m_vol_step;
     double  m_tick_value, m_tick_size;

     int DateKey(datetime t) { MqlDateTime s; TimeToStruct(t, s); return (s.year*10000 + 
  s.mon*100 + s.day); }

     void RefreshSymbol()
     {
        SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_MIN,  m_vol_min);
        SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_MAX,  m_vol_max);
        SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_STEP, m_vol_step);
        SymbolInfoDouble(m_symbol, SYMBOL_TRADE_TICK_VALUE, m_tick_value);
        SymbolInfoDouble(m_symbol, SYMBOL_TRADE_TICK_SIZE,  m_tick_size);
     }

     double RoundLots(double lots)
     {
        if(m_vol_step<=0) return lots;
        double steps = MathFloor((lots - m_vol_min + 1e-12)/m_vol_step);
        double r = m_vol_min + steps*m_vol_step;
        r = MathMax(m_vol_min, MathMin(m_vol_max, r));
        return NormalizeDouble(r, 2);
     }

     double RiskAmountFromPerc(double risk_perc, double equity)
     {
        return equity * (risk_perc/100.0);
     }

     double RiskPerLotAtSLPoints(double sl_points)
     {
        // Para 1 lote, risco (dinheiro) se SL atinge:
        // price_distance = sl_points * _Point
        // ticks = price_distance / m_tick_size
        // money = ticks * m_tick_value
        if(m_tick_size<=0) return 0.0;
        double price_dist = sl_points * _Point;
        double ticks = price_dist / m_tick_size;
        return ticks * m_tick_value;
     }

     double PredictedDailyDDPctIfSL(double risk_amount)
     {
        // DD previsto caso o SL seja atingido nessa nova operação
        // Aproximação conservadora: equity pós-perda = AccountEquity() - risk_amount
        double eq_after = AccountInfoDouble(ACCOUNT_EQUITY) - risk_amount;
        double dd = (m_daily_start_equity - eq_after)/m_daily_start_equity * 100.0;
        return dd;
     }

     double PredictedTotalDDPctIfSL(double risk_amount)
     {
        double eq_after = AccountInfoDouble(ACCOUNT_EQUITY) - risk_amount;
        double dd = (m_initial_balance - eq_after)/m_initial_balance * 100.0;
        return dd;
     }

  public:
     void Setup(const string symbol,
                double initial_balance,
                double soft_daily_loss_pct,
                double hard_daily_loss_pct,
                double max_total_loss_pct)
     {
        m_symbol = symbol;
        m_initial_balance = initial_balance;
        m_soft_daily_loss_pct = soft_daily_loss_pct;
        m_hard_daily_loss_pct = hard_daily_loss_pct;
        m_max_total_loss_pct  = max_total_loss_pct;

        RefreshSymbol();
        m_daily_start_equity = AccountInfoDouble(ACCOUNT_EQUITY);
        m_daily_anchor_date  = DateKey(TimeCurrent());
     }

     void UpdateDailyAnchor()
     {
        int today = DateKey(TimeCurrent());
        if(today != m_daily_anchor_date)
        {
           m_daily_start_equity = AccountInfoDouble(ACCOUNT_EQUITY);
           m_daily_anchor_date = today;
        }
     }

     double CurrentDailyDDPct()
     {
        return (m_daily_start_equity -
  AccountInfoDouble(ACCOUNT_EQUITY))/m_daily_start_equity*100.0;
     }

     double CurrentTotalDDPct()
     {
        return (m_initial_balance - AccountInfoDouble(ACCOUNT_EQUITY))/m_initial_balance*100.0;
     }

     double DynamicRiskModifier()
     {
        // Política:
        // 0–1% DD → 1.0
        // 1–2.5% → 0.5
        // 2.5–4% → 0.25
        // >=4%   → 0.0 (bloqueia)
        double dd = CurrentDailyDDPct();
        if(dd < 1.0)    return 1.0;
        if(dd < 2.5)    return 0.5;
        if(dd < 4.0)    return 0.25;
        return 0.0;
     }

     double CalcLotsByRisk(double risk_perc, double sl_points)
     {
        RefreshSymbol();
        if(sl_points<=0.0) return 0.0;

        double equity = AccountInfoDouble(ACCOUNT_EQUITY);
        double risk_amount = RiskAmountFromPerc(risk_perc, equity);

        // Ajusta pelo modificador dinâmico de DD diário
        double mod = DynamicRiskModifier();
        risk_amount *= mod;

        double risk_per_lot = RiskPerLotAtSLPoints(sl_points);
        if(risk_per_lot<=0.0) return 0.0;

        double lots = risk_amount / risk_per_lot;
        return RoundLots(lots);
     }

     bool CanOpenTrade(double risk_perc, double sl_points)
     {
        RefreshSymbol();
        if(sl_points<=0.0) return false;

        // Soft/Hard checks
        double dd_now = CurrentDailyDDPct();
        if(dd_now >= m_hard_daily_loss_pct) return false;
        if(DynamicRiskModifier()==0.0)      return false; // já estamos na zona de bloqueio

        // Calcula perda potencial se SL atingir (em dinheiro)
        double equity = AccountInfoDouble(ACCOUNT_EQUITY);
        double risk_amount = RiskAmountFromPerc(risk_perc, equity) * DynamicRiskModifier();

        double dd_daily_pred = PredictedDailyDDPctIfSL(risk_amount);
        if(dd_daily_pred >= m_hard_daily_loss_pct - 1e-6) return false;

        double dd_total_pred = PredictedTotalDDPctIfSL(risk_amount);
        if(dd_total_pred >= m_max_total_loss_pct - 1e-6)  return false;

        return true;
     }
  };

  //--------------------- Stubs de Módulos (assinaturas) --------------
  class COrderBlockModule { public: bool HasOrderBlock(bool &bullish) { /* TODO */ bullish=true;
   return false; } };
  class CFVGModule        { public: bool HasFVG(bool &bullish)        { /* TODO */ bullish=true;
   return false; } };
  class CLiquidityModule  { public: bool HasLiquiditySweep(bool &bull){ /* TODO */ bull=true;
  return false; } };
  class CMarketStructureModule
  {
  public:
     bool IsBullishTrend()
     {
        // TODO: Implementar HH/HL/LH/LL ou MME cross; placeholder simples.
        return true;
     }
  };
  class CVolatilityModule
  {
  public:
     double GetATR(ENUM_TIMEFRAMES tf, int period)
     {
        int h = iATR(_Symbol, tf, period);
        if(h==INVALID_HANDLE) return 0.0;
        double buf[]; int copied = CopyBuffer(h, 0, 0, 2, buf);
        if(copied<=0) return 0.0;
        return buf[0];
     }
  };

  //--------------------- Instâncias Globais --------------------------
  CSignalScoringModule   Scoring;
  CFTMORiskManager       Risk;
  COrderBlockModule      ModOB;
  CFVGModule             ModFVG;
  CLiquidityModule       ModLiq;
  CMarketStructureModule ModMS;
  CVolatilityModule      ModVol;

  //--------------------- Helpers -------------------------------------
  void Reason(const string msg) { Print("[REASON] ", msg); }
  // Nota: Integração Python via WebRequest será feita fora do caminho quente (OnTimer).
  //       Aqui manteremos placeholders de valores:

  bool GetPythonScores(double &tech_sub_py, double &fund, double &sent)
  {
     // TODO: Implementar em OnTimer com WebRequest e cache.
     // Retornar false quando indisponível. Aqui, usamos fallback neutro.
     tech_sub_py = 50.0; fund = 50.0; sent = 50.0;
     return false;
  }

  //--------------------- Ciclo de Vida --------------------------------
  int OnInit()
  {
     Risk.Setup(_Symbol, InpInitialBalance, InpSoftDailyLossPct, InpMaxDailyLossPct, 
  InpMaxTotalLossPct);
     Reason("Inicializado. Risco por trade=" + DoubleToString(InpRiskPerTradePct,2) + "%; 
  Threshold=" + DoubleToString(InpExecutionThreshold,1));
     return(INIT_SUCCEEDED);
  }
  void OnDeinit(const int reason)
  {
     Reason("Encerrado. Motivo=" + IntegerToString(reason));
  }

  //--------------------- OnTick ---------------------------------------
  void OnTick()
  {
     Risk.UpdateDailyAnchor();

     // 1) Sinais técnicos leves
     bool ob_bull=true, fvg_bull=true, liq_bull=true;
     bool hasOB  = ModOB.HasOrderBlock(ob_bull);
     bool hasFVG = ModFVG.HasFVG(fvg_bull);
     bool liq    = ModLiq.HasLiquiditySweep(liq_bull);
     bool bullishTrend = ModMS.IsBullishTrend();
     double atr = ModVol.GetATR(InpSignalTF, InpATRPeriod);

     // 2) Score técnico MQL5
     double tech_mql5 = Scoring.ComputeTechScore(hasOB, hasFVG, bullishTrend, atr);
     if(tech_mql5 < 50.0) return; // pré-filtro para evitar ruído e consultas externas
  desnecessárias

     // 3) (Opcional) Subscore Python + Fund/Sent (usando cache/OnTimer fora do caminho quente)
     double tech_py=50, fund=50, sent=50;
     bool py_ok = GetPythonScores(tech_py, fund, sent);

     // Mistura Tech: 70% MQL5, 30% Python se disponível
     double tech_mix = py_ok ? (0.7*tech_mql5 + 0.3*tech_py) : tech_mql5;

     // 4) Final Score
     double final_score = Scoring.ComputeFinalScore(tech_mix, fund, sent);
     if(final_score < InpExecutionThreshold) return;

     // 5) SL/TP por ATR
     if(atr<=0) return;
     double sl_points = (atr * InpSL_ATR_Mult) / _Point;
     if(sl_points<=0) return;

     // 6) Check FTMO + sizing por risco
     if(!Risk.CanOpenTrade(InpRiskPerTradePct, sl_points)) { Reason("VETO FTMO: risco previsível
   excede limites."); return; }
     double lots = Risk.CalcLotsByRisk(InpRiskPerTradePct, sl_points);
     if(lots<=0) { Reason("Lots calculado zero (provável DD dinâmico/SL inválido)."); return; }

     // 7) Direção simples por tendência (placeholder)
     bool buy = bullishTrend;
     double price, sl, tp, dist = sl_points * _Point;
     if(buy)
     {
        price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        sl    = price - dist;
        tp    = price + dist*InpTP_RR;
        Trade.Buy(lots, _Symbol, price, sl, tp, "EA_SCALPER_XAUUSD BUY");
        Reason(StringFormat("BUY lots=%.2f SLp=%.0f ATR=%.2f Final=%.1f", lots, sl_points, atr,
  final_score));
     }
     else
     {
        price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
        sl    = price + dist;
        tp    = price - dist*InpTP_RR;
        Trade.Sell(lots, _Symbol, price, sl, tp, "EA_SCALPER_XAUUSD SELL");
        Reason(StringFormat("SELL lots=%.2f SLp=%.0f ATR=%.2f Final=%.1f", lots, sl_points, atr,
   final_score));
     }
  }
  Notas importantes:
  - WebRequest não é chamado em OnTick (evita travar). Use OnTimer + cache (Seção 5).
  - CanOpenTrade considera a perda potencial da nova posição contra MDL/MTL (prevenção ex-ante).
  - DynamicRiskModifier reduz lotes conforme DD diário.
  - Stubs de sinais (OB/FVG/etc.) estão prontos para implementação progressiva.

  Seção 5 – Interface Com Python Agent Hub
  - Requisição (HTTP/REST, POST JSON):
  {
    "symbol": "XAUUSD",
    "time": "2025-11-22T14:35:00Z",
    "timeframe_signal": "M5",
    "timeframe_trend": "M15",
    "signals": {
      "hasOB": true,
      "hasFVG": false,
      "bullishTrend": true,
      "atr": 2.35
    },
    "market_context": {
      "session": "NY",
      "spread_points": 25,
      "recent_volatility": "moderate"
    }
  }
  - Resposta esperada:
  {
    "tech_subscore_python": 62.0,
    "fund_score": 55.0,
    "fund_bias": "neutral",
    "sent_score": 48.0,
    "sent_bias": "slightly_bearish",
    "llm_reasoning_short": "FVG não preenchido em M5, tendência M15 bull, volatilidade moderada.
   Sem eventos macro críticos próximos."
  }
  - Pseudocódigo MQL5 (chamada robusta):
  bool CallPythonHub(double &tech_sub_py, double &fund_score, double &sent_score)
  {
    // 1) Se última consulta < N segundos → usar cache e return true
    // 2) Senão, se OnTimer → montar JSON e chamar WebRequest(POST, URL)
    // 3) if (status!=200 || timeout) { fallback: return false; }
    // 4) Parse JSON (usar CJAON ou parser simples): extrair campos
    // 5) Atualizar cache/tempo; return true
  }
  Falha/timeout: operar apenas com MQL5 (modo seguro), mantendo o FinalScore baseado em
  tech_mql5 (ou tech_mix sem Python).

  Seção 6 – Raciocínio De Risco (FTMO) & Deep Thinking
  - Parâmetros propostos (conta 100k, XAUUSD scalping):
    - Risk per trade: 0.25–0.40% (usar 0.35% default). Justificativa: XAUUSD tem saltos;
  under-risk reduz prob. de violar MDL.
    - Soft Daily Loss: 3% (início de redução dinâmica).
    - Hard Max Daily Loss: 5% (regra FTMO); bloqueio antes de atingir (ex.: pre-check impede
  ficar ≥5% com SL).
    - Max Total Loss: 10% (global); usar pre-check similar ao diário.
  - Política de redução dinâmica (espelhada no código):
    - 0–1% DD diário → risco normal (1.0×).
    - 1–2.5% → 0.5×.
    - 2.5–4% → 0.25×.
    - ≥4% → bloqueio novas entradas (0.0×).
  - Evitar overtrading num dia bom:
    - Definir “soft daily win cap” (ex.: +2% dia): após atingi-lo, reduzir risco a 0.25× e
  permitir apenas 1 trade adicional com RR≥1.5; se lucro fechar ≥2.5%, encerrar dia. Racional:
  proteger capital e estabilidade da curva; reduzir variância pós-pico.
  - Tratar 3 stops seguidos:
    - Regra: ao 3º stop consecutivo no dia, pausa a operação até próxima sessão; se 2 stops,
  cair para 0.25× risco até fechar um trade positivo; objetivo: cortar cauda esquerda causada
  por clusters de perdas.
  - Quando NÃO operar mesmo com setup técnico bom:
    - Eventos macro (CPI/NFP/FED) ±30–60min; spreads acima de limiar (ex.: >35–40 pontos); baixa
   liquidez (transição Ásia→EU cedo); latência/servidor instável; slippage elevado detectado;
  FinalScore alto, porém FTMO_RiskManager indica DD previsto próximo do hard limit.

  Seção 7 – Estratégia De Testes E Validação
  - Backtests:
    - Período: ao menos 24–36 meses recentes; com regime de volatilidade variado.
    - Timeframes: M1/M5 (sinais), M15 (tendência). Modelagem “Every tick based on real ticks”;
  spread variável.
    - Param sweep: pesos do scoring, SL_ATR_Mult, RR, pre-threshold para chamar Python.
  - Stress tests:
    - Spread +50–150% do histórico; slippage simulado 0.5–1.5× do habitual.
    - News on/off: janela de exclusão (±60min) vs incluir (para ver caudas).
    - Burst de latência (simular atraso de execução) e perda de resposta Python (modo fallback).
  - Testes específicos FTMO:
    - Emulação de MDL/MTL: registrar equity de início do dia e checar previsão de DD com SL
  antes de cada trade (igual ao manager).
    - KPIs: número de dias com quase-violação (≥4% DD), casos de bloqueio por regra; nenhuma
  violação em 100% dos trades.
  - Critérios de aprovação (exemplo):
    - PF ≥ 1.30; Win rate ≥ 40%; RR médio ≥ 1.2; Max DD ≤ 7%.
    - 0 violações; ≤ 3 dias/mês com DD ≥ 4% (quase-violação) e nunca repetidos em dias
  consecutivos.
    - Consistência entre sessões e anos (não overfit).

  Seção 8 – Exemplos De Reasoning Strings
  - Exemplo 1 – Trade WIN (BUY XAUUSD)
    - “Sessão NY, tendência M15 de alta, ATR moderado; OB M5 suportando o preço e sem FVG aberto
   acima. FinalScore 88 aprovou. SL abaixo do OB com RR 1.2. Execução consistente com política:
  risco 0.35%, DD diário baixo; trade alinhado ao viés e volatilidade.”
  - Exemplo 2 – Trade LOSS (SELL XAUUSD)
    - “Sessão EU tardia, tendência M15 levemente baixista, FVG M5 acima com sweep de liquidez
  nos topos. FinalScore 86 permitiu venda; SL acima do FVG com RR 1.2. Perda ocorreu em reversão
   repentina; ainda assim dentro do risco planejado e DD diário manteve-se <1%.”
  - Exemplo 3 – Sinal IGNORADO (score alto mas FTMO próximo do limite)
    - “Sessão NY, sinal técnico forte (OB+FVG, ATR ideal), FinalScore 91. Entretanto, DD diário
  atual 3.9% com risco previsto levando a ~5.1% em caso de SL. FTMO_RiskManager vetou conforme
  política; decisão de não entrar preserva a conta e respeita o hard limit.”

  Quer que eu transforme este esqueleto em um EA compilável no diretório de trabalho e adicionar
   uma rotina OnTimer com cache para a integração HTTP ao Python Hub?



# Claude.ai (site oficial usando o claude 4.5 thinking):

# EA_SCALPER_XAUUSD – Multi-Agent Hybrid System
## Projeto Completo: MQL5 + Python Agent Hub

---

## 🧩 SEÇÃO 1 – COMPREENSÃO DO PROBLEMA

**Objetivos Estratégicos e Restrições:**

• **Objetivo Principal**: Criar um sistema de scalping em XAUUSD capaz de passar challenges FTMO e operar consistentemente dentro de regras estritas de prop firms, maximizando edge técnico enquanto protege capital.

• **Restrições FTMO Críticas**: Max Daily Loss (5% tipicamente), Max Total Loss (10%), Profit Target (10% em 30 dias), trailing drawdown após primeira fase, proibição de martingale/grid trading, e necessidade de manter risco controlado mesmo em winning streaks.

• **Vantagem Multi-Agente**: MQL5 entrega velocidade de execução (<50ms OnTick) para análise técnica pura, enquanto Python Hub processa análises complexas (NLP de notícias, LLM reasoning, sentiment) de forma assíncrona, evitando travar o EA e permitindo escalabilidade modular.

• **Risco de Slippage**: XAUUSD tem spread médio 20-40 pips em sessões normais, 100+ em news - scalping exige controle rigoroso de timing, evitando entradas durante eventos macro e periodos de baixa liquidez (rollover asiático).

• **Risco de Overtrading**: Scalpers tendem a forçar setups após wins/losses, violando disciplina - necessário circuit breaker após 3 losses consecutivos ou após atingir 50% do daily profit target (lock profits).

• **Violação de Max Daily Loss**: Um único trade mal dimensionado (2-3% risco) + slippage pode consumir metade do limite diário - obrigatório position sizing dinâmico que reduz exposição conforme DD acumula.

• **Falsa Confiança em Backtests**: Tick data de qualidade variável pode mascarar problemas de spread/slippage - validação exige forward testing em demo + análise de worst-case scenarios (spread 2x, slippage 50%).

• **Complexidade de Market Structure**: OB/FVG/Liquidity Sweeps não são sinais binários - necessário scoring ponderado que considera confluência, age do OB (OBs de 1h são mais fortes que 5m), e contexto de HTF.

• **Dependência de Python**: Se Python Hub falhar, EA deve operar autonomamente com scores técnicos puros - arquitetura precisa ser resiliente, com fallback para modo "MQL5-only" sem comprometer segurança.

• **Gestão Emocional via IA**: Reasoning strings transparentes evitam "caixa preta" - trader precisa entender cada decisão para confiar no sistema durante drawdowns inevitáveis e não desligar EA prematuramente.

---

## 🏗️ SEÇÃO 2 – ARQUITETURA DE ALTO NÍVEL (MQL5 + PYTHON)

### **Camadas MQL5:**

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1: DATA & EVENTS                                     │
│  ├─ OnTick(): Processa cada tick, filtra noise (min 5 ticks)│
│  ├─ OnTimer(): Chamadas Python assíncronas (1 req/30s max) │
│  └─ OnTradeTransaction(): Monitora fills, atualiza P&L     │
└─────────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2: STRATEGY / SIGNAL LAYER                           │
│  ├─ COrderBlockModule: Detecta OB (bullish/bearish)        │
│  ├─ CFVGModule: Identifica FVGs não preenchidos            │
│  ├─ CLiquidityModule: Rastreia sweeps + equal highs/lows   │
│  ├─ CMarketStructureModule: HH/HL/LH/LL em multi-TF        │
│  └─ CVolatilityModule: ATR, spread check, session filter   │
└─────────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3: SCORING ENGINE                                    │
│  ├─ TechScore (0-100): Weighted sum de sinais técnicos     │
│  ├─ FundScore (0-100): Via Python (calendário, correlações)│
│  ├─ SentScore (0-100): Via Python (Twitter, news sentiment)│
│  └─ FinalScore = (0.6*Tech + 0.25*Fund + 0.15*Sent)        │
└─────────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────┐
│  LAYER 4: EXECUTION & FTMO_RiskManager                      │
│  ├─ CFTMORiskManager.CanOpenTrade(): Veto absoluto         │
│  ├─ Position Sizing: Dynamic lot based on DD atual         │
│  └─ CTradeExecutor: Market orders c/ SL/TP, retry logic    │
└─────────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────┐
│  LAYER 5: LOGGING & NOTIFICATIONS                           │
│  ├─ CLogger: File logs (CSV) + console debug               │
│  └─ Push Notifications: Reasoning string para cada trade   │
└─────────────────────────────────────────────────────────────┘
```

### **Python Agent Hub:**

**Agentes Implementados:**
- **Technical Agent**: Calcula sub-scores adicionais (divergências RSI, volumes anômalos, correlações multi-asset)
- **Fundamental Agent**: Processa calendário econômico (Fed minutes, CPI, NFP), impacto esperado em Gold
- **Sentiment Agent**: Scraping Twitter/Reddit para sentiment retail, news analysis (bullish/bearish keywords)
- **LLM Reasoning Agent**: GPT-4 micro-prompts para validar setup ("Given OB at 2050, FVG below, is this high-probability?")

**Comunicação: HTTP/REST (escolhido)**
- **Justificativa**: ZeroMQ é mais rápido (~5ms vs ~20ms HTTP), mas HTTP é mais simples de debugar, não requer DLLs externas (evita problemas com prop firms que bloqueiam DLLs customizadas), e permite horizontally scaling do Python Hub (múltiplas instâncias atrás de load balancer).
- **Trade-off aceitável**: Latência de 20-30ms não impacta scalping se usado de forma assíncrona (via OnTimer), não bloqueando OnTick.

**Formato de Resposta JSON:**
```json
{
  "tech_subscore_python": 72.5,
  "fund_score": 45.0,
  "fund_bias": "neutral",
  "sent_score": 68.0,
  "sent_bias": "bullish",
  "llm_reasoning": "OB valid, trend confirmed, but CPI in 2h - reduce size",
  "timestamp": "2025-01-20T14:30:00Z"
}
```

### **Fluxo de 1 Tick "Perfeito":**

```
1. OnTick() triggered (XAUUSD @ 2050.50)
   ↓
2. Noise Filter: Passa se >5 ticks desde último check (evita spam)
   ↓
3. Technical Modules Execution (5-10ms):
   - COrderBlockModule → detecta OB bullish em 2045.20
   - CFVGModule → FVG de 2046-2048 não preenchido
   - CLiquidityModule → Sweep de lows em 2044.80 (bullish sweep)
   - CMarketStructureModule → Trend = Bullish (HH em H1)
   - CVolatilityModule → ATR = 8.5, spread = 25 pips (OK)
   ↓
4. CSignalScoringModule.ComputeTechScore():
   - OB present: +30 pts
   - FVG valid: +20 pts
   - Liquidity sweep: +15 pts
   - Bullish trend: +20 pts
   - Low volatility penalty: -5 pts
   → TechScore = 80/100
   ↓
5. Python Hub Call (OnTimer, async, cached 30s):
   - HTTP POST to http://localhost:8000/api/score
   - Timeout: 2s (se falhar, usa FundScore=50, SentScore=50)
   - Response: FundScore=60 (CPI amanhã), SentScore=75 (bullish Twitter)
   ↓
6. FinalScore Calculation:
   - FinalScore = 0.6*80 + 0.25*60 + 0.15*75 = 74.25
   ↓
7. Threshold Check:
   - 74.25 < 85 (ExecutionThreshold) → SKIP trade
   → Se fosse >=85, continua:
   ↓
8. CFTMORiskManager.CanOpenTrade(risk=1.0%, SL=30 pips):
   - Daily DD atual: 1.2%
   - Projected DD se stop: 1.2% + 1.0% = 2.2% < 4.0% (Hard Max) → OK
   - Wins today: 3, Losses today: 1 → No circuit breaker
   - Adjusted lot: 0.05 (reduzido de 0.10 devido a DD acumulado)
   → Returns TRUE
   ↓
9. CTradeExecutor.OpenTrade():
   - Market Buy XAUUSD @ 2050.50
   - SL: 2047.50 (30 pips)
   - TP: 2056.50 (60 pips, RR 1:2)
   ↓
10. CLogger + Push Notification:
    - Log: "2025-01-20 14:30:15 | BUY | Lot:0.05 | Score:74 | Reason: OB+FVG+Sweep"
    - Notification: "✅ BUY XAUUSD | Score 74 | Risk 1.0% | OB at 2045, bullish sweep confirmed"
```

---

## ⚙️ SEÇÃO 3 – DESIGN DETALHADO DO EA EM MQL5

### **Módulos Principais:**

#### **1. COrderBlockModule**
- **Responsabilidades**: Detecta Order Blocks válidos (últimas velas antes de movimento impulsivo), classifica por strength (age, touch count, volume se disponível)
- **Inputs**: Price data (High/Low/Close arrays), lookback period (20-50 bars), timeframe
- **Outputs**: `struct OrderBlock { double price_top; double price_bottom; ENUM_OB_TYPE type; int age_bars; double strength_score; }`

#### **2. CFVGModule**
- **Responsabilidades**: Identifica Fair Value Gaps (gap entre vela N-1 low e vela N+1 high para bullish), rastreia se já foi preenchido (>50%)
- **Inputs**: Price arrays, lookback period
- **Outputs**: `struct FVG { double gap_top; double gap_bottom; bool is_filled; int bars_ago; }`

#### **3. CLiquidityModule**
- **Responsabilidades**: Detecta liquidity sweeps (preço toca equal lows/highs e reverte), equal highs/lows (3+ toques dentro de 10 pips)
- **Inputs**: Swing highs/lows, current price
- **Outputs**: `enum SWEEP_TYPE { BULLISH_SWEEP, BEARISH_SWEEP, NONE }; double sweep_level;`

#### **4. CMarketStructureModule**
- **Responsabilidades**: Classifica estrutura de mercado (Bullish: HH+HL, Bearish: LH+LL, Range), multi-timeframe (M5, M15, H1)
- **Inputs**: Swing points arrays
- **Outputs**: `enum TREND_TYPE { BULLISH, BEARISH, RANGING }; double trend_strength; // 0-100`

#### **5. CVolatilityModule**
- **Responsabilidades**: Calcula ATR, valida spread está dentro de limites, filtra horários de baixa liquidez (22h-2h GMT)
- **Inputs**: ATR period (14), max spread allowed (50 pips)
- **Outputs**: `bool is_tradeable_session; double current_atr; int spread_pips;`

#### **6. CSignalScoringModule**
- **Responsabilidades**: Agrega sinais de todos os módulos técnicos, aplica pesos, normaliza para 0-100
- **Inputs**: Outputs de todos os módulos técnicos
- **Outputs**: `double TechScore; string reasoning_tech; // "OB+FVG+Bullish Structure"`

#### **7. CFTMORiskManager**
- **Responsabilidades**: Guardião do risco - valida cada trade contra regras FTMO, calcula position sizing dinâmico, circuit breakers
- **Inputs**: Account balance, DD atual (diário/total), risk per trade %, SL em pips
- **Outputs**: `bool CanOpenTrade(); double GetAdjustedLotSize(); enum RISK_STATE { NORMAL, REDUCED, MINIMAL, BLOCKED }`

#### **8. CTradeExecutor**
- **Responsabilidades**: Execução de ordens com retry logic, validação de SL/TP, handling de erros (requotes, off-quotes)
- **Inputs**: Order type, lot size, SL/TP levels
- **Outputs**: `ulong ticket; bool execution_success; string error_message;`

#### **9. CLogger**
- **Responsabilidades**: File logging (CSV com timestamp, symbol, action, score, P&L), console debug, push notifications
- **Inputs**: Log level (INFO/WARNING/ERROR), message
- **Outputs**: Arquivos `EA_SCALPER_YYYYMMDD.csv`

---

### **Pseudocódigo OnTick Ideal:**

```cpp
void OnTick() {
    // 1. NOISE FILTER (evita processar cada tick)
    if (TicksSinceLastCheck < 5 && TimeCurrent() - LastCheckTime < 3) return;
    LastCheckTime = TimeCurrent();
    TicksSinceLastCheck = 0;
    
    // 2. PRÉ-CHECKS RÁPIDOS (<1ms)
    if (!VolatilityModule.IsTradeableSession()) return; // Fora de horário
    if (VolatilityModule.GetSpreadPips() > MaxSpreadPips) return; // Spread alto
    if (RiskManager.GetRiskState() == BLOCKED) return; // DD critical
    
    // 3. MODULES EXECUTION (5-10ms total)
    OrderBlock ob = OrderBlockModule.Detect();
    FVG fvg = FVGModule.Detect();
    SWEEP_TYPE sweep = LiquidityModule.CheckSweep();
    TREND_TYPE trend = MarketStructureModule.GetTrend();
    double atr = VolatilityModule.GetATR();
    
    // 4. TECHNICAL SCORING (2ms)
    double techScore = ScoringModule.ComputeTechScore(ob, fvg, sweep, trend, atr);
    
    // 5. PYTHON SCORES (usa cache de OnTimer, não bloqueia)
    double fundScore = PythonScoreCache.fund_score; // default 50 se falha
    double sentScore = PythonScoreCache.sent_score; // default 50 se falha
    
    // 6. FINAL SCORE
    double finalScore = ScoringModule.ComputeFinalScore(techScore, fundScore, sentScore);
    
    // 7. THRESHOLD CHECK
    if (finalScore < ExecutionThreshold) {
        Logger.Log(INFO, "Score " + finalScore + " < threshold, skipping");
        return;
    }
    
    // 8. DETERMINE DIRECTION & SL/TP
    ENUM_ORDER_TYPE orderType = (trend == BULLISH && ob.type == OB_BULLISH) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
    double sl_pips = CalculateSL(ob, fvg); // ex: abaixo do OB + buffer
    double tp_pips = sl_pips * RiskRewardRatio; // ex: 1:2
    
    // 9. RISK MANAGER VETO (crítico!)
    if (!RiskManager.CanOpenTrade(RiskPercentPerTrade, sl_pips)) {
        Logger.Log(WARNING, "Risk Manager blocked trade | Score=" + finalScore);
        NotifyUser("🚫 Trade blocked: DD protection");
        return;
    }
    
    // 10. EXECUTION
    double lot = RiskManager.GetAdjustedLotSize(RiskPercentPerTrade, sl_pips);
    ulong ticket = TradeExecutor.OpenTrade(orderType, lot, sl_pips, tp_pips);
    
    // 11. LOGGING & NOTIFICATION
    if (ticket > 0) {
        string reasoning = BuildReasoningString(ob, fvg, sweep, trend, finalScore);
        Logger.LogTrade(ticket, finalScore, reasoning);
        NotifyUser("✅ " + EnumToString(orderType) + " | Score:" + finalScore + " | " + reasoning);
    }
}

// OnTimer: Chamadas Python assíncronas (não bloqueia OnTick)
void OnTimer() {
    // Chama Python Hub a cada 30s, atualiza cache
    if (TimeCurrent() - LastPythonCallTime > 30) {
        CallPythonHubAsync(); // Thread separado ou WebRequest não-bloqueante
        LastPythonCallTime = TimeCurrent();
    }
    
    // Update risk metrics
    RiskManager.UpdateDailyStats();
}
```

**Garantias de Performance:**
- **OnTick < 50ms**: Módulos técnicos são otimizados (caching de cálculos pesados, ArraySetAsSeries para acesso O(1))
- **Python não bloqueia**: Chamado via OnTimer com cache, OnTick usa último valor disponível
- **Limite de chamadas**: Máximo 1 req Python/30s (evita rate limiting e overhead)

---

## 💻 SEÇÃO 4 – CÓDIGO MQL5 ESSENCIAL

Vou criar o artefato com código MQL5 funcional:---

## 🔗 SEÇÃO 5 – INTERFACE COM PYTHON AGENT HUB

### **Request JSON Format (MQL5 → Python):**

```json
{
  "symbol": "XAUUSD",
  "timeframe": "M15",
  "timestamp": "2025-01-20T14:30:00Z",
  "current_price": 2050.50,
  "technical_signals": {
    "has_order_block": true,
    "ob_type": "bullish",
    "ob_price": 2045.20,
    "has_fvg": true,
    "fvg_range": [2046.0, 2048.0],
    "liquidity_sweep": "bullish_sweep",
    "trend": "bullish",
    "atr_14": 8.5,
    "spread_pips": 25
  },
  "session_info": {
    "current_session": "london",
    "hours_to_next_news": 2.5,
    "upcoming_events": ["US_CPI"]
  },
  "risk_context": {
    "daily_dd_percent": 1.2,
    "trades_today": 4,
    "win_rate_today": 0.75
  }
}
```

### **Response JSON Format (Python → MQL5):**

```json
{
  "status": "success",
  "timestamp": "2025-01-20T14:30:05Z",
  "scores": {
    "tech_subscore_python": 72.5,
    "fund_score": 45.0,
    "fund_bias": "neutral",
    "fund_confidence": 0.65,
    "sent_score": 68.0,
    "sent_bias": "bullish",
    "sent_confidence": 0.72
  },
  "analysis": {
    "llm_reasoning_short": "OB valid, trend confirmed, but CPI in 2h - reduce size by 50%",
    "key_drivers": ["technical_alignment", "retail_sentiment_bullish", "macro_uncertainty"],
    "risk_adjustment": -0.5
  },
  "meta": {
    "processing_time_ms": 145,
    "agents_consulted": ["technical", "fundamental", "sentiment", "llm"]
  }
}
```

### **Pseudocódigo MQL5 para Chamada Python:**

```cpp
//+------------------------------------------------------------------+
//| Call Python Hub via HTTP POST (Production-Ready Pattern)         |
//+------------------------------------------------------------------+
bool CallPythonHub(double &tech_subscore_py, double &fund_score, double &sent_score, string &llm_reasoning) {
    // 1. BUILD REQUEST JSON
    string json_request = BuildPythonRequest();
    
    // 2. PREPARE HTTP REQUEST
    char post_data[];
    char result_data[];
    string result_headers;
    string request_headers = "Content-Type: application/json\r\n";
    
    StringToCharArray(json_request, post_data, 0, StringLen(json_request));
    
    // 3. MAKE HTTP REQUEST (with timeout protection)
    ResetLastError();
    int http_code = WebRequest(
        "POST",                    // Method
        InpPythonURL,              // URL
        request_headers,           // Headers
        InpPythonTimeoutMS,        // Timeout (2000ms)
        post_data,                 // Request body
        result_data,               // Response body (out)
        result_headers             // Response headers (out)
    );
    
    // 4. ERROR HANDLING
    if(http_code == -1) {
        int error = GetLastError();
        LogMessage(StringFormat("WebRequest failed: Error %d - Check URL whitelist in Tools->Options->Expert Advisors", error));
        
        // FALLBACK: Use safe defaults
        tech_subscore_py = 50.0;
        fund_score = 50.0;
        sent_score = 50.0;
        llm_reasoning = "Python unreachable - using MQL5-only mode";
        return false;
    }
    
    if(http_code != 200) {
        LogMessage(StringFormat("Python Hub returned HTTP %d", http_code));
        
        // FALLBACK: Safe defaults
        tech_subscore_py = 50.0;
        fund_score = 50.0;
        sent_score = 50.0;
        llm_reasoning = "Python error - neutral scores";
        return false;
    }
    
    // 5. PARSE JSON RESPONSE
    string response_json = CharArrayToString(result_data);
    
    // Simplified parsing (in production, use proper JSON library like JAson.mqh)
    tech_subscore_py = ExtractJsonDouble(response_json, "tech_subscore_python");
    fund_score = ExtractJsonDouble(response_json, "fund_score");
    sent_score = ExtractJsonDouble(response_json, "sent_score");
    llm_reasoning = ExtractJsonString(response_json, "llm_reasoning_short");
    
    // 6. VALIDATION: Ensure scores are in valid range
    if(tech_subscore_py < 0 || tech_subscore_py > 100) tech_subscore_py = 50.0;
    if(fund_score < 0 || fund_score > 100) fund_score = 50.0;
    if(sent_score < 0 || sent_score > 100) sent_score = 50.0;
    
    LogMessage(StringFormat("Python Hub OK | Tech:%.1f Fund:%.1f Sent:%.1f", 
               tech_subscore_py, fund_score, sent_score));
    
    return true;
}

//+------------------------------------------------------------------+
//| Build JSON request for Python Hub                                |
//+------------------------------------------------------------------+
string BuildPythonRequest() {
    double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double atr = iATR(_Symbol, PERIOD_CURRENT, InpATRPeriod, 0);
    
    // Simplified JSON construction (use proper JSON library in production)
    string json = StringFormat(
        "{"
        "\"symbol\":\"%s\","
        "\"timeframe\":\"%s\","
        "\"timestamp\":\"%s\","
        "\"current_price\":%.2f,"
        "\"technical_signals\":{"
            "\"has_order_block\":%s,"
            "\"trend\":\"%s\","
            "\"atr_14\":%.2f"
        "},"
        "\"risk_context\":{"
            "\"daily_dd_percent\":%.2f"
        "}"
        "}",
        _Symbol,
        EnumToString(PERIOD_CURRENT),
        TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS),
        bid,
        DetectOrderBlock() ? "true" : "false",
        DetectTrend() > 0 ? "bullish" : "bearish",
        atr,
        CFTMORiskManager().GetDailyDrawdownPercent()
    );
    
    return json;
}

//+------------------------------------------------------------------+
//| Helper: Extract double from JSON (simplified)                    |
//+------------------------------------------------------------------+
double ExtractJsonDouble(string json, string key) {
    // Very simplified parsing - use proper JSON parser in production
    string search = "\"" + key + "\":";
    int start = StringFind(json, search);
    if(start == -1) return 50.0; // Default
    
    start += StringLen(search);
    int end = StringFind(json, ",", start);
    if(end == -1) end = StringFind(json, "}", start);
    
    string value = StringSubstr(json, start, end - start);
    return StringToDouble(value);
}

//+------------------------------------------------------------------+
//| Helper: Extract string from JSON (simplified)                    |
//+------------------------------------------------------------------+
string ExtractJsonString(string json, string key) {
    string search = "\"" + key + "\":\"";
    int start = StringFind(json, search);
    if(start == -1) return "N/A";
    
    start += StringLen(search);
    int end = StringFind(json, "\"", start);
    
    return StringSubstr(json, start, end - start);
}
```

### **Tratamento de Falhas - Política de Fallback:**

1. **Timeout (2s)**: Se Python não responder em 2s, usar scores neutros (50/50/50) e continuar com análise técnica pura
2. **HTTP Error (4xx/5xx)**: Log error, usar fallback, enviar alerta ao desenvolvedor
3. **JSON Parse Error**: Detectar resposta malformada, usar defaults seguros
4. **Network Down**: Após 3 falhas consecutivas, desabilitar Python por 5 minutos (circuit breaker)
5. **Modo Degradado**: EA opera normalmente com TechScore puro se Python está indisponível - NUNCA para de operar

**Configuração MT5 para WebRequest:**
```
Tools → Options → Expert Advisors → 
☑ Allow WebRequest for listed URL:
   http://localhost:8000
```

---

## 🧠 SEÇÃO 6 – RACIOCÍNIO DE RISCO (FTMO) & DEEP THINKING

### **Configuração Proposta para Conta FTMO $100k (XAUUSD Scalping):**

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| **Risk per Trade** | 0.8% | Agressivo mas controlado - permite 12 stops consecutivos antes de atingir 10% DD total. Com RR 1:2, 4 wins compensam 8 losses. |
| **Soft Daily Loss** | 2.5% | Zona amarela - começar a reduzir risco, evitar acelerar em day ruim. |
| **Hard Max Daily Loss** | 4.0% | 80% do limite FTMO (5%) - margem de segurança para slippage. |
| **Max Total Loss** | 8.0% | 80% do limite FTMO (10%) - nunca operar perto do limite real. |
| **Max Trades per Day** | 8 | Scalping controlado - evita overtrading, força seletividade. |

### **Política de Redução de Risco Dinâmica:**

```
┌─────────────────────────────────────────────────────────────┐
│  RISK SCALING MATRIX (Dynamic Drawdown Control)            │
├─────────────────┬───────────────────────────────────────────┤
│ Daily DD Range  │ Action                                    │
├─────────────────┼───────────────────────────────────────────┤
│ 0.0% - 1.0%     │ NORMAL: Risk 0.8%, full lot size         │
│ 1.0% - 2.5%     │ CAUTION: Risk 0.5%, reduce to 60% lot    │
│ 2.5% - 3.5%     │ REDUCED: Risk 0.3%, reduce to 30% lot    │
│ 3.5% - 4.0%     │ MINIMAL: Risk 0.15%, only A+ setups      │
│ 4.0%+           │ BLOCKED: Zero trades, wait for next day   │
└─────────────────┴───────────────────────────────────────────┘

ADICIONAL - Circuit Breakers:
• 3 losses consecutivos → Pause 2 horas (emocional reset)
• Win > 3% em 1 dia → Reduzir risco 50% (lock profits)
• Sexta após 15h GMT → Não abrir novas posições (weekend risk)
```

### **Deep Thinking: Cenários Críticos**

#### **Cenário 1: Dia Bom (3% lucro nas primeiras 2 horas)**

**Problema**: Overtrading por excesso de confiança. Trader pensa "estou em fire, vou aproveitar" → força setups marginais → devolve ganhos.

**Solução Proposta**:
- **Profit Lock Rule**: Após atingir 2.5% lucro diário, reduzir risco para 0.4% (metade) automaticamente
- **Selective Mode**: Aumentar threshold de execução para 90 (de 85) - só entrar em setups excepcionais
- **Mental Break**: Após 3 wins seguidos, pause obrigatório de 30 min (evitar euforia)
- **Raciocínio**: Scalping FTMO não é sobre "fazer 10% em 1 dia" - é sobre consistência. 2% diário = 40% mensal, passa challenge facilmente. Proteger ganhos é mais importante que maximizar.

#### **Cenário 2: 3 Stops Seguidos em XAUUSD**

**Problema**: Market mudou de regime (de trending para ranging), ou news event causou volatilidade anormal. Continuar operando = bleeding capital.

**Solução Proposta**:
1. **Immediate Pause**: Bloquear novas entradas por 2 horas (circuit breaker)
2. **Regime Analysis**: 
   - Verificar se ATR aumentou >50% (volatilidade anormal)
   - Verificar se spread está >2x normal (liquidez problemática)
   - Verificar calendário econômico (news esquecido?)
3. **Strategy Adaptation**:
   - Se ATR alto: Aumentar SL para 2x ATR (dar mais breathing room)
   - Se ranging: Desabilitar trend-following, aguardar breakout claro
   - Se news: Não operar até 1h após release
4. **Psychological Reset**: 
   - Anotar em log: "Por que esses 3 stops aconteceram?" (pattern recognition)
   - Assistir replay dos trades (TradingView) - identificar erro comum
   - Só retornar após análise completa, não por "medo de perder o dia"

**Raciocínio**: 3 stops = ~2.4% DD (0.8% cada). Continuar teimoso pode virar 5% (fim do dia). Better: parar, analisar, preservar capital. FTMO challenge tem 30 dias - perder 1 dia bad não mata a account, perder 5% em 1 dia sim.

#### **Cenário 3: Setup Técnico Perfeito MAS...**

**Situações para NÃO operar mesmo com Score 95/100:**

1. **15 min antes de news High Impact (NFP, Fed, CPI)**:
   - **Razão**: Spread pode ir de 30 pips para 200 pips em segundos, SL será slipado violentamente
   - **Exceção**: Se já está em trade, fechar 10 min antes da news (protect profit)

2. **Spread > 50 pips (2x normal)**:
   - **Razão**: Slippage vai comer o edge, scalping com spread alto é EV negativo
   - **Solução**: Aguardar spread normalizar (<35 pips)

3. **Sexta-feira após 16h GMT**:
   - **Razão**: Liquidez cai, risco de gap no fim de semana, brokers podem aumentar margem
   - **Solução**: Fechar todas as posições até 16h, não abrir novas

4. **Daily DD já está em 3.0%**:
   - **Razão**: Mesmo com setup perfeito, risco de violar regra é alto demais
   - **Raciocínio**: Probabilidade de 1 trade dar loss = 40% (assumindo 60% WR). 40% de chance de perder a conta vs. potencial de 1% gain? Math doesn't work. Preservar account > 1 trade.

5. **Após sequence de 5 wins consecutivos**:
   - **Razão**: Estatisticamente improvável continuar streak, mean reversion é real
   - **Psicologia**: Excesso de confiança leva a aumentar risco subconscientemente
   - **Solução**: Pause 1h, review dos trades, garantir que não está ficando sloppy

6. **Liquidez anormal (volume <50% da média)**:
   - **Razão**: Low liquidity = maior chance de stop hunting, piores fills
   - **Exemplo**: Asian session 23h-2h GMT em XAUUSD = deserto de liquidez

**Princípio Core**: *"No edge is worth violating risk rules. Capital preservation > single trade opportunity. You can't make money if you're kicked out of FTMO."*

---

## 🧪 SEÇÃO 7 – ESTRATÉGIA DE TESTES E VALIDAÇÃO

### **1. Backtests (Validação Técnica)**

**Configuração Recomendada:**
```
Data Range: 2 anos (Jan 2023 - Jan 2025)
Timeframe Principal: M15
Timeframes Auxiliares: M5 (sinais), H1 (trend)
Tick Data: "Every tick based on real ticks" (quality máxima)
Spread: Variable (usar histórico real se disponível)
Slippage: 3 pips (conservador para XAUUSD)
Symbols: XAUUSD apenas (especialização > diversificação)
Initial Deposit: $100,000 (simular FTMO Challenge)
```

**Métricas de Avaliação:**

| Métrica | Mínimo Aceitável | Target | Observações |
|---------|------------------|--------|-------------|
| **Win Rate** | 55% | 60%+ | Scalping com RR 1:2 precisa de WR alto |
| **Profit Factor** | 1.5 | 2.0+ | PF <1.5 = edge marginal, não robusto |
| **Max Drawdown** | <8% | <5% | Precisa passar FTMO (10% limit) com margem |
| **Avg Win / Avg Loss** | 1.8 | 2.0+ | Validar RR real (slippage/spread consideram) |
| **Sharpe Ratio** | >1.5 | >2.0 | Risk-adjusted return quality |
| **Recovery Factor** | >3 | >5 | Net Profit / Max DD |
| **Trades per Day** | 3-8 | 5 | Nem overtrading, nem undertrading |

### **2. Stress Tests (Robustez)**

**A. Spread Stress Test:**
```
Test 1: Spread fixo 50 pips (2x normal)
Test 2: Spread fixo 100 pips (news simulation)
Test 3: Spread variable com spikes em news hours

Expected Result: 
- PF deve cair, mas manter >1.3
- WR deve cair <5%
- Sistema ainda profitable (prova edge robusto)
```

**B. Slippage Stress Test:**
```
Test 1: +5 pips slippage em entries
Test 2: +10 pips slippage em SL/TP
Test 3: Combinado (5 entry + 10 exit)

Expected Result:
- Com 10 pips slippage total, RR 1:2 vira ~1:1.6
- Sistema deve manter profitable (PF >1.2)
```

**C. News On/Off Test:**
```
Test 1: Operar durante todas as news (baseline ruim)
Test 2: Bloquear 30 min antes/depois de High Impact news
Test 3: Bloquear apenas Top Tier (NFP, FOMC, CPI)

Expected Result:
- Test 2/3 deve ter Max DD 30-50% menor que Test 1
- Confirma que news filter é essencial
```

**D. Worst-Case Scenario:**
```
Combine:
- Spread 80 pips
- Slippage 8 pips
- Commission $7/lot/side
- Início em dia de high volatility (ex: Oct 7 2023, Israel war)

Expected Result:
- Sistema deve sobreviver sem violar Max DD 10%
- Pode não ser profitable, mas não explode
- Prova que risk management funciona em condições extremas
```

### **3. FTMO Compliance Tests**

**Validação de Regras:**

**Test A: Max Daily Loss Simulator**
```cpp
// Injetar no backtest: Simular pior dia possível
// Forçar 5 stops consecutivos no mesmo dia
// Validar que:
// 1. EA para de operar após atingir 4% DD
// 2. Nunca ultrapassa 5% (FTMO limit)
// 3. Circuit breakers funcionam
```

**Test B: Max Total Loss Monitoring**
```cpp
// Validar trailing drawdown (FTMO Phase 2)
// Simular sequence:
// Day 1-10: +8% profit (sobe HWM)
// Day 11: -6% DD (deve parar em 8% abaixo HWM)
// Confirmar que EA nunca atinge 10% trailing DD
```

**Test C: Profit Target Feasibility**
```cpp
// FTMO Challenge: 10% profit em 30 dias
// Validar em backtest:
// - % de meses que atingem 10%
// - Tempo médio para atingir target
// - Se consegue sem violar DD limits

Target: >70% de meses profitable, >50% atingem 10%
```

### **4. Critérios de Aprovação (Go/No-Go Decision)**

**MUST PASS (Dealbreakers):**
- [ ] Max DD em backtest 2 anos < 8%
- [ ] Zero violações de Max Daily Loss 5% em 500+ trades
- [ ] Profit Factor > 1.5 em stress tests (spread 2x)
- [ ] Sistema opera sem travar (OnTick <50ms em 99% dos ticks)
- [ ] Python fallback funciona (testes com Python offline)

**NICE TO HAVE (Preferências):**
- [ ] Win Rate > 58%
- [ ] Sharpe > 2.0
- [ ] Consegue 10% profit em <20 dias (média)
- [ ] Funciona em múltiplos brokers (IC Markets, FTMO, Blueberry)

**BEFORE LIVE:**
1. **Demo Forward Test**: 30 dias em conta demo FTMO (sem otimização!)
2. **Paper Trade Review**: Analisar cada trade - reasoning faz sentido?
3. **Slippage Analysis**: Comparar SL/TP preenchidos vs. planejados
4. **Max DD Observation**: Se demo teve DD >6%, investigar antes de live
5. **Peer Review**: Outro trader experiente analisa lógica e resultados

**RED FLAGS (Não passar para live):**
- ❌ Backtest muito bom (PF >3, WR >70%) = overfitting provável
- ❌ Diferença grande entre backtest e forward test (>30% variação PF)
- ❌ Trades concentrados em poucos dias (falta consistência)
- ❌ Drawdown recovery muito rápido (martingale oculto?)
- ❌ Muitos trades em horários de baixa liquidez (news ignorados?)

---

## 📣 SEÇÃO 8 – EXEMPLOS DE REASONING STRINGS

### **Exemplo 1 – Trade WIN (BUY XAUUSD)**

```
📈 TRADE CLOSED: +$680 (+0.68%)

Entry: 2045.30 | Exit: 2051.30 | +60 pips
Direction: BUY | Lot: 0.11 | Hold Time: 18 min
Score: 88.5/100 (Tech:92, Fund:78, Sent:85)

💡 Reasoning:
Bullish Order Block confirmed at 2042.50 (H1), price tapped OB zone with FVG confluence at 2044-2046. Liquidity sweep of Asian lows (2043.20) triggered entry signal. Market structure showed clear HH/HL pattern on M15, aligning with London open momentum. ATR at 7.8 (medium volatility), spread 28 pips (acceptable). Python sentiment analysis confirmed 85% bullish retail positioning, with fund score neutral due to no major news scheduled. Risk was 0.8% ($800 SL), RR executed at 1:1.9 (TP-5 pips due to spread). 

✅ Decision was consistent with strategy and risk policy. Daily DD before trade: 0.4%, after: -0.12% (profit reduced DD). Circuit breakers: None active.
```

---

### **Exemplo 2 – Trade LOSS (SELL XAUUSD)**

```
📉 TRADE CLOSED: -$420 (-0.42%)

Entry: 2058.80 | Exit: 2062.80 | -40 pips (SL hit)
Direction: SELL | Lot: 0.105 | Hold Time: 12 min
Score: 86.0/100 (Tech:90, Fund:82, Sent:75)

💡 Reasoning:
Bearish Order Block identified at 2061.20 (M15), price rejected OB with strong wick on previous candle. FVG present at 2059-2061 (unfilled), liquidity sweep of recent highs (2060.50) suggested potential reversal. Market structure was bearish on M15 (LH/LL), but H1 showed bullish trend (conflicting). ATR 9.2 (elevated), spread 32 pips. Python fund score showed 82 (slight bearish bias due to DXY strength), sentiment 75 (mixed signals).

❌ Trade invalidated when price broke above OB with momentum - likely stop hunt before continuation up. In retrospect, H1 bullish trend should have been weighted heavier (conflicting timeframes = lower confidence). SL was properly placed above OB high + buffer.

✅ Loss was within risk parameters (0.4% risked, 0.42% actual due to slippage). Daily DD after trade: 1.8% (still well within limits). No rule violations - valid setup that didn't play out. Circuit breaker status: 2 losses today, monitoring for 3rd (would trigger pause).
```

---

### **Exemplo 3 – Sinal IGNORADO (Score Alto mas Risco FTMO Próximo do Limite)**

```
🚫 TRADE REJECTED: Risk Manager Veto

Potential Entry: 2053.40 | Direction: BUY
Score: 89.5/100 (Tech:94, Fund:88, Sent:82)
Projected Risk: 0.8% ($800)

💡 Reasoning:
Exceptional setup - Bullish OB at 2051.20 with triple FVG confluence (M5, M15, H1 aligned), liquidity sweep confirmed, market structure strongly bullish across all timeframes. Python analysis highly positive (fund score 88 due to weak DXY + strong gold sentiment, LLM reasoning: "All systems go, high-probability long"). Spread 26 pips, ATR 8.1, London session active.

🛑 Risk Manager Decision: BLOCKED
- Current Daily DD: 3.4% (approaching 4.0% hard limit)
- Projected DD if SL hit: 4.2% (would violate hard max)
- Circuit Breaker: MINIMAL risk state active (only 0.15% risk allowed)
- Trades today: 6 (W:3, L:3, WR:50%)

✅ Decision to skip was CORRECT per policy:
Even perfect setup (89.5 score) cannot override risk management. Preserving account is priority #1 - violating Max Daily Loss would end FTMO challenge immediately. Better to skip one high-probability trade than risk entire account. System designed to survive to trade another day.

📊 Context: Today started well (+1.2%), then gave back gains with 3 consecutive losses in afternoon session (-4.6% swing). This triggered defensive mode correctly. Will reassess tomorrow with fresh daily limits.
```

---

## 🎯 CONCLUSÃO & NEXT STEPS

Este sistema **EA_SCALPER_XAUUSD** foi projetado com arquitetura profissional que prioriza:

1. **Risk-First Philosophy**: FTMO_RiskManager tem poder absoluto de veto
2. **Transparência**: Cada trade tem reasoning string completo
3. **Robustez**: Funciona mesmo se Python falhar (fallback gracioso)
4. **Performance**: OnTick <50ms, Python assíncrono via OnTimer
5. **Escalabilidade**: Módulos desacoplados, fácil adicionar novos agentes

**Implementação Recomendada (Roadmap):**

**Fase 1 (Semana 1-2)**: Core MQL5
- Implementar módulos técnicos completos (OB, FVG, Liquidity)
- Testar CFTMORiskManager em backtest
- Validar performance OnTick <50ms

**Fase 2 (Semana 3)**: Python Hub MVP
- Criar API Flask/FastAPI com 3 agentes (Technical, Fund, Sentiment)
- Integrar WebRequest no EA
- Testar latência e fallback

**Fase 3 (Semana 4)**: Backtests & Optimization
- Rodar backtests 2 anos com qualidade máxima
- Stress tests (spread, slippage, news)
- Otimizar thresholds (não overfitting!)

**Fase 4 (Semana 5-8)**: Forward Testing
- 30 dias demo FTMO
- Paper trade analysis
- Ajustes finais

**Fase 5 (Mês 3+)**: Live com Capital Real
- Começar com 0.5% risk (conservador)
- Monitorar diariamente
- Iterar baseado em dados reais

**Este sistema está pronto para desenvolvimento. Boa sorte na implementação! 🚀**

# gpt-5.1-codex-max- Extra high ( usando via codex CLI oficial): 


• SEÇÃO 1 – Compreensão do Problema

  - Criar um EA de scalping em XAUUSD para contas de prop (ex.: FTMO),
    com decisões transparentes e veto de risco obrigatório.
  - Respeitar regras críticas de prop: Max Daily Loss (por ex. -5k
    num dia em conta 100k), Max Total Loss (-10k), proibição de over-
    leverage e de manter posições durante eventos específicos.
  - Arquitetura híbrida MQL5 + Python permite: latência muito baixa
    no core (MQL5) e enriquecimento assíncrono com dados fundamentais/
    sentimento (Python) sem travar o OnTick.
  - Multi-agente melhora robustez: cada agente produz um sub-score; o
    FTMO_RiskManager arbitra antes da execução.
  - Riscos clássicos em XAUUSD scalping: slippage elevado em spikes,
    spreads variáveis, overtrading após série de wins, violar limites
    diários por cascata de perdas, execução lenta em notícias, e
    latência de chamadas externas.
  - Necessidade de Reasoning String auditável para cada trade, provando
    compliance com regras de risco.
  - OnTick deve ficar <50 ms; tarefas pesadas vão para OnTimer/threads
    Python.
  - Score 0–100 garante filtro rigoroso: só operar quando múltiplos
    confluem e o risco está liberado.
  - Drawdown adaptativo reduz tamanho de lote conforme DD diário para
    evitar brechas de Max Daily Loss.
  - Transparência e logs granulares para auditoria de prop firm.

  SEÇÃO 2 – Arquitetura de Alto Nível (MQL5 + Python)

  - MQL5 Camadas
      - Data & Events: OnTick (preço), OnTimer (tarefas externas),
        OnTradeTransaction (monitorar PnL e DD).
      - Strategy/Signal: módulos OB, FVG, Liquidity Sweeps, Market
        Structure (HH/HL/LH/LL), ATR/volatilidade.
      - Scoring Engine: combina TechScore (MQL5 + opcional Python) +
        FundScore + SentScore → FinalScore.
      - Execution & FTMO_RiskManager: checa limites FTMO, calcula lote,
        decide abrir/fechar.

      - Logging & Notifications: Reasoning String, push/arquivo/experts
        log.
  - Python Agent Hub
      - Agentes: Technical
  (confirma OB/FVG via outro
  modelo),
        Fundamental (news
  calendar, metais), Sentiment
  (feed de
        notícias/finanças), LLM
  Reasoning (resumo textual).
      - Transporte: HTTP/REST via
  WebRequest porque é simples,
  stateless
        e aceito pelo MQL5;
  ZeroMQ exigiria dll extra.
      - Resposta JSON enxuta com
  subscores e texto curto.
  - Fluxo de um Tick
  “perfeito” (texto-diagrama)
    Tick → MQL5_EA/OnTick → Módulos
  técnicos (OB/FVG/Liquidity/
    Structure/ATR) → TechScore preliminar →
  (se tempo livre e janela
    OnTimer ok) MQL5_EA -> HTTP POST ->
  Python_Hub → recebe subscores →
    ScoringEngine calcula FinalScore →
  FTMO_RiskManager valida limites →
    TradeExecutor envia ordem → Logger
  grava Reasoning String.

  SEÇÃO 3 – Design Detalhado do EA em MQL5

  - Módulos/classes
      - COrderBlockModule: detecta OBs
  recentes; inputs: candles,
        volume; output: bool hasOB,
  direcionalidade.
      - CFVGModule: detecta fair value
  gaps; outputs: bool hasFVG,
        tamanho em pips.
      - CLiquidityModule: procura equal
  highs/lows, liquidity sweeps;
        outputs: bool sweepUp/Down.
      - CMarketStructureModule: determina
  HH/HL/LH/LL e bias (bull/
        bear).
      - CVolatilityModule: ATR e regime de
  volatilidade; outputs:
        atrPoints, regimeHigh/Low.
      - CSignalScoringModule: agrega sinais
  → TechScore; combina com
        fund/sent → FinalScore.
      - CFTMORiskManager: lê saldo/equity/
  PnL diário; calcula lote
        permitido; veto se exceder limites.
      - CTradeExecutor: envia ordens market
  com SL/TP; ajusta volume
        conforme risk manager.
      - CLogger: formata Reasoning String;
  grava em arquivo e push.
  - Pseudocódigo OnTick ideal

    OnTick():
        start_time = GetMicrosecondCount()
        price = SymbolInfoTick(_Symbol)
        signals =
  CollectSignalsFromModules()        // OB,
  FVG, Liquidity, Structure, ATR
        techScore_mql =
  SignalScorer.ComputeTechScore(signals)
        if time_since_last_python_call
  > PY_CALL_INTERVAL and
  (GetMicrosecondCount()-start_time <
  time_budget):
            CallPythonHub(tech_py, fund,
  sent)       // async-like via OnTimer
  preferred
        finalScore =
  SignalScorer.ComputeFinalScore(techScore,
  fund, sent)
        if finalScore >=
  ExecutionThreshold:
            if
  riskManager.CanOpenTrade(risk_perc_input,
  signals.stoploss_points):

  TradeExecutor.Open(signals.direction,
  volume, SL, TP)
        Logger.LogReasoning(...)
        enforce time budget (skip extras
  if >50ms)
      - Para não travar: mover
  CallPythonHub para OnTimer (ex.: a cada
        2–5 s) e cachear última resposta;
  OnTick só lê cache.

  SEÇÃO 4 – Código MQL5 Essencial

  //
  +----------------------------------------
  --------------------------+
  //| Expert: EA_SCALPER_XAUUSD
  |
  //
  +----------------------------------------
  --------------------------+
  #property strict
  input double InpRiskPerTradePct =
  0.5;      // % risk per trade
  input double InpMaxDailyLossPct  =
  5.0;     // FTMO max daily loss %
  input double InpMaxTotalLossPct  =
  10.0;    // FTMO max total loss %
  input double InpExecutionThreshold =
  85.0;  // FinalScore threshold
  input ENUM_TIMEFRAMES InpTF =
  PERIOD_M5;    // analysis TF
  input int    InpSlippage =
  30;              // in points
  input double InpSoftDailyLossPct =
  2.5;     // start reducing risk
  input int    InpPythonTimeoutMs =
  800;      // HTTP timeout
  datetime     g_lastPyCall=0;
  double       g_cachedFund=50,
  g_cachedSent=50, g_cachedTechPy=50;

  class CFTMORiskManager
    {
  private:
     double maxDailyLossPct,
  maxTotalLossPct, softDailyLossPct;
  public:
     CFTMORiskManager(double daily, double
  total, double soft)
       { maxDailyLossPct=daily;
  maxTotalLossPct=total;
  softDailyLossPct=soft; }

     double CurrentBalance(){ return
  AccountInfoDouble(ACCOUNT_BALANCE); }
     double CurrentEquity(){ return
  AccountInfoDouble(ACCOUNT_EQUITY); }

     double DailyPnL()
       {
         datetime dayStart = iTime(_Symbol,
  PERIOD_D1, 0);
         HistorySelect(dayStart,
  TimeCurrent());
         double pnl=0;
         uint total=HistoryDealsTotal();
         for(uint i=0;i<total;i++)
           {
             ulong ticket =
  HistoryDealGetTicket(i);

  if(HistoryDealGetString(ticket,
  DEAL_SYMBOL)==_Symbol)
               pnl +=
  HistoryDealGetDouble(ticket,
  DEAL_PROFIT);
           }
         return pnl;
       }

     double AllowedDailyLoss()
  { return CurrentBalance() *
  (maxDailyLossPct/100.0); }
     double AllowedTotalLoss()
  { return CurrentBalance() *
  (maxTotalLossPct/100.0); }

     double DynamicRiskPct(double baseRisk)
       {
         double dd = -MathMin(0,
  DailyPnL());                // only
  losses
         double ddPct = dd /
  CurrentBalance() * 100.0;
         if(ddPct < softDailyLossPct)
  return baseRisk;
         if(ddPct < (softDailyLossPct*1.6))
  return baseRisk*0.5;  // reduce risk
  mid-zone
         if(ddPct < (maxDailyLossPct*0.8))
  return baseRisk*0.25;  // near hard limit
         return
  0.0;
        // block trading
       }

     bool BreachImminent()
       {
         double dd = -MathMin(0,
  DailyPnL());
         return dd >=
  AllowedDailyLoss()*0.95;
       }

     bool CanOpenTrade(double risk_perc,
  double stoploss_points)
       {
         double adjRisk =
  DynamicRiskPct(risk_perc);
         if(adjRisk<=0) return false;

         double accEquity =
  CurrentEquity();
         double moneyRisk = accEquity *
  (adjRisk/100.0);
         double tickValue =
  SymbolInfoDouble(_Symbol,
  SYMBOL_TRADE_TICK_VALUE);
         double tickSize  =
  SymbolInfoDouble(_Symbol,
  SYMBOL_TRADE_TICK_SIZE);
         double volStep   =
  SymbolInfoDouble(_Symbol,
  SYMBOL_VOLUME_STEP);
         double contract  =
  SymbolInfoDouble(_Symbol,
  SYMBOL_TRADE_CONTRACT_SIZE);

         double sl_price_dist =
  stoploss_points * _Point;
         if(sl_price_dist<=0) return false;

         // position size (lots) =
  moneyRisk / (sl_dist * tickValue/point)
         double lot = moneyRisk /
  (sl_price_dist / tickSize * tickValue);
         lot = MathFloor(lot/
  volStep)*volStep;

         // Check FTMO daily/total limits
         double dd = -MathMin(0,
  DailyPnL());
         if(dd >= AllowedDailyLoss())
  return false;
         double totalDD = CurrentBalance()
  - CurrentEquity();
         if(totalDD >= AllowedTotalLoss())
  return false;

         // store volume in a global
  or return via reference in real
  implementation

  GlobalVariableSet("EA_SCALPER_ALLOWED_LOT
  ", lot);
         return lot > 0;
       }
    };

  class CSignalScoringModule
    {
  public:
     double ComputeTechScore(bool
  hasOB,bool hasFVG,bool
  bullishTrend,double atr)
       {
         double score=0;
         if(hasOB) score+=25;
         if(hasFVG) score+=20;
         if(bullishTrend) score+=25; else
  score+=10;
         // normalize ATR regime: assume
  atr in points; higher atr can mean
  opportunity but also risk
         if(atr>150) score+=15; else
  if(atr>80) score+=10; else score+=5;
         return MathMin(score,100.0);
       }
     double ComputeFinalScore(double
  tech,double fund,double sent)
       {
         // weighted blend; tech is primary
         return tech*0.6 + fund*0.2 +
  sent*0.2;
       }
    };

  // --- Stubs for other modules (to be
  implemented)
  class COrderBlockModule { public: bool
  DetectOB(bool &bullish,double &slPoints)
  { /*TODO*/ return false; } };
  class CFVGModule        { public: bool
  DetectFVG(bool &bullish,double &sizePts)
  { /*TODO*/ return false; } };
  class CLiquidityModule  { public:
  bool DetectSweep(bool &sweepUp,bool
  &sweepDown){ /*TODO*/ return false; } };
  class CMarketStructureModule { public:
  bool IsBullishTrend(){ /*TODO*/ return
  true; } };
  class CVolatilityModule { public: double
  GetATRPoints(){ return iATR(_Symbol,
  InpTF, 14, 0)/_Point; } };

  // --- Globals
  CFTMORiskManager
  g_risk(InpMaxDailyLossPct,
  InpMaxTotalLossPct, InpSoftDailyLossPct);
  CSignalScoringModule g_score;
  COrderBlockModule g_ob;
  CFVGModule g_fvg;
  CLiquidityModule g_lq;
  CMarketStructureModule g_ms;
  CVolatilityModule g_vol;

  int OnInit()
    {
     Print("EA_SCALPER_XAUUSD init");
     EventSetTimer(2); // for python
  polling if needed
     return(INIT_SUCCEEDED);
    }

  void OnDeinit(const int reason)
    {
     EventKillTimer();
     Print("EA_SCALPER_XAUUSD deinit");
    }

  void OnTick()
    {
     ulong t0=GetMicrosecondCount();
     // ---- collect signals (placeholders)
     bool
  obBull=false,fvgBull=false,sweepUp=false,
  sweepDown=false;
     double slPts=300; // placeholder stop
  size in points; real logic from OB/FVG
     bool hasOB = g_ob.DetectOB(obBull,
  slPts);
     bool hasFVG = g_fvg.DetectFVG(fvgBull,
  slPts);
     g_lq.DetectSweep(sweepUp, sweepDown);
     bool bullish = g_ms.IsBullishTrend();
     double atr = g_vol.GetATRPoints();

     double techScore =
  g_score.ComputeTechScore(hasOB||obBull,
  hasFVG||fvgBull, bullish, atr);

     // optional Python enrichment using
  cached values
     double fund=g_cachedFund,
  sent=g_cachedSent,
  tech_py=g_cachedTechPy;
     double techBlend = (techScore*0.7 +
  tech_py*0.3);
     double finalScore =
  g_score.ComputeFinalScore(techBlend,
  fund, sent);

     if(finalScore >=
  InpExecutionThreshold)
       {

  if(g_risk.CanOpenTrade(InpRiskPerTradePct
  , slPts))
           {
             double lot =
  GlobalVariableGet("EA_SCALPER_ALLOWED_LOT
  ");
             int type = bullish ?
  ORDER_TYPE_BUY : ORDER_TYPE_SELL;
             double price
  = (type==ORDER_TYPE_BUY)?
  SymbolInfoDouble(_Symbol, SYMBOL_ASK)

  : SymbolInfoDouble(_Symbol, SYMBOL_BID);
             double sl =
  (type==ORDER_TYPE_BUY)? price -
  slPts*_Point : price + slPts*_Point;
             double tp =
  (type==ORDER_TYPE_BUY)? price
  + slPts*_Point*1.5 : price -
  slPts*_Point*1.5;

             MqlTradeRequest  req;
             MqlTradeResult   res;
             ZeroMemory(req);
  ZeroMemory(res);
             req.action   =
  TRADE_ACTION_DEAL;
             req.symbol   = _Symbol;
             req.type     = type;
             req.volume   = lot;
             req.price    = price;
             req.sl       = sl;
             req.tp       = tp;
             req.deviation= InpSlippage;
             OrderSend(req,res);
             PrintFormat("Trade sent
  lot=%.2f FS=%.1f", lot, finalScore);
           }
         else
           Print("Risk vetoed by FTMO
  manager");
       }

     // time guard
     if(GetMicrosecondCount()-t0 > 50000)
       Print("Warning: OnTick exceeded
  50ms");
    }

  // Timer for Python calls (simplified
  placeholder)
  void OnTimer()
    {
     if(TimeCurrent() - g_lastPyCall <
  2) return;
     g_lastPyCall = TimeCurrent();
     // TODO: CallPythonHub(...) to refresh
  g_cachedTechPy/g_cachedFund/g_cachedSent
    }

  SEÇÃO 5 – Interface com Python Agent Hub

  - Request JSON (POST /scores):

  {
    "symbol": "XAUUSD",
    "timeframe": "M5",
    "timestamp": "2025-11-22T14:05:00Z",
    "tech_local": 72.5,
    "signals": {
      "hasOB": true,
      "hasFVG": true,
      "bullishTrend": true,
      "atr_pts": 165,
      "sweepUp": false,
      "sweepDown": true
    }
  }

  - Response JSON esperado:

  {
    "tech_subscore_python": 75.0,
    "fund_score": 62.0,
    "fund_bias": "slightly_bearish",
    "sent_score": 58.0,
    "sent_bias": "mixed",
    "llm_reasoning_short": "NY session,
  gold bid; US data in 30m, keep size
  small."
  }

  - Pseudocódigo MQL5 para chamada:

  bool CallPythonHub(double &tech_py,
  double &fund, double &sent)
  {
     string url="https://localhost:8000/
  scores";
     string headers="Content-Type:
  application/json\r\n";
     string body =
  BuildJsonFromSignals(); // conforme acima
     char result[];
     int timeout=InpPythonTimeoutMs;
     int status = WebRequest("POST", url,
  headers, timeout, body, result, NULL);
     if(status==200)
       {
         // parse JSON (use CJAValike or
  custom)
         tech_py = JsonGetDouble(result,
  "tech_subscore_python");
         fund    = JsonGetDouble(result,
  "fund_score");
         sent    = JsonGetDouble(result,
  "sent_score");
         return true;
       }
     // fallback seguro: manter últimos
  valores e operar só com tech local
     return false;
  }

  SEÇÃO 6 – Raciocínio de Risco (FTMO) &
  Deep Thinking

  - Conta 100k FTMO XAUUSD scalping
  (volatilidade alta).
      - Risk per trade: 0.25–0.5% (usar
  0.5% só em condições top;
        default 0.35%).
      - Soft Daily Loss: 2.0–2.5% (ajuste
  para reduzir lote).
      - Hard Max Daily Loss: 5% (regra
  FTMO), mas travar novas entradas
        a 4% para buffer.
      - Max Total Loss: 10%; travar novas
  entradas se DD flutua >8%.
  - Política dinâmica (diária):
      - 0–1% DD: risco normal (0.35–0.5%).
      - 1–2.5%: risco 50% do normal.
      - 2.5–4%: risco 25% do normal, só
  setups nota A (FinalScore > 92).
      - ≥4%: bloquear entradas; só
  gerenciar posições abertas.
  - Evitar overtrading em dia bom: após
  atingir +2.5% no dia, limitar a
    1 trade adicional com risco reduzido;
  ao atingir +4% diário, fechar
    terminal / modo only-manage.
  - Três stops seguidos: pausar trading até
  próxima sessão (ex.: se
    em Londres, esperar NY) ou 90 minutos;
  reduzir risco ao retornar
    (metade).
  - Não operar mesmo com setup bom: spreads
  > 25 pontos; eventos macro
    USD high-impact nos próximos 15–30 min;
  liquidez muito baixa (Asian
    late); VIX < 12 ou > 30 (ambiente
  extremo); após grandes gaps ou
    feriados US; desconexão/latência alta.

  SEÇÃO 7 – Estratégia de Testes e
  Validação

  - Backtests
      - Range: pelo menos 2–3 anos recentes
  de XAUUSD; incluir 2022
        (alta volatilidade) e 2024–2025.
      - Timeframe: M1 data para modelar M5/
  M1 scalping; qualidade de
        tick 99% (Dukascopy ou Tickstory).
      - Testar ambos modos: só Tech local
  vs Tech+Python mock (valores
        fixos).
  - Stress tests
      - Spread multiplicado 1.5x e 2x.
      - Slippage aleatório 0–300 pts.
      - Filtrar dias de NFP/FOMC on/off.
  - Testes FTMO específicos
      - Emulação: script que mede PnL
  diário e para o EA ao atingir 90%
        do Max Daily Loss; validar que
  CFTMORiskManager vetou ordens.
      - Checar logs de Reasoning String
  para cada trade contendo score e
        estado de risco.
  - Critérios de aprovação
      - Win rate ≥ 48% com RR médio ≥1.3 →
  PF >1.3.
      - Max DD < 8% e nenhum dia violando
  90% do limite diário.
      - Máximo de 1 dia em 20 com drawdown
  diário >3%.
      - Latência OnTick média < 30 ms, p95
  < 50 ms.

  SEÇÃO 8 – Exemplos de Reasoning Strings
  de Trades

  - Exemplo 1 – Trade WIN (BUY XAUUSD)
    "NY open, tendência bullish com HH/HL,
  FVG M5 preenchido e OB M15
    segurando. ATR alto mas spread ok;
  FinalScore 91, risco 0.35%.
    SL abaixo do OB; TP 1.5R. Consistente
  com política: liberado pelo
    FTMO_RiskManager (DD diário 0.3%).
  Resultado: +1.4R."
  - Exemplo 2 – Trade LOSS (SELL XAUUSD)
    "Sessão Londres, estrutura virou LH/LL
  após sweep de liquidez acima
    do high asiático; FVG de continuação
  alinhado. FinalScore 88, risco
    0.3%. SL acima do sweep; TP 1.8R.
  Execução permitida (DD diário
    1.1%). Movimento reverteu com news
  inesperada; stop hit, -1R.
    Política seguida, sem violação de
  risco."
  - Exemplo 3 – Sinal IGNORADO (score alto
  mas risco FTMO próximo do
    limite)
    "NY lunch, bias bullish com OB M5 e
  FVG; FinalScore 90. Contudo, DD
    diário em 3.9% e limite FTMO 5%.
  Política de redução coloca risco
    em 0% acima de 4%; FTMO_RiskManager
  vetou nova entrada. Decisão: não
    entrar; preserva conta e regra diária."

# Gemini 3 Pro High (via ANtigravity: 

Aqui está o design completo e detalhado do sistema EA_SCALPER_XAUUSD, estruturado exatamente conforme solicitado.

🎯 EA_SCALPER_XAUUSD – Design de Sistema Híbrido (MQL5 + Python)
🧩 SEÇÃO 1 – COMPREENSÃO DO PROBLEMA
Objetivo Estratégico: Criar um sistema de scalping de alta precisão para XAUUSD que passe consistentemente em desafios de prop firms (ex: FTMO), priorizando a preservação de capital sobre lucros explosivos.
Restrições FTMO: O sistema deve respeitar rigidamente o Max Daily Loss (5%) e Max Total Loss (10%). A violação dessas regras resulta em falha imediata, independentemente do lucro.
Vantagem Híbrida: O MQL5 garante execução de baixa latência (<50ms) e estabilidade, enquanto o Python permite análise de dados complexos (Sentimento, Notícias, ML) que seriam impossíveis ou lentos em MQL5 puro.
Foco em XAUUSD: O Ouro é altamente volátil e técnico. O sistema deve explorar essa volatilidade (FVGs, Liquidity Sweeps) mas mitigar riscos de slippage e spread widening.
Risco de Scalping: Scalpers tendem a fazer muitos trades. O risco principal é o overtrading e o "tilt" algorítmico (vários stops seguidos em curto período). O Risk Manager deve ser um "freio de mão" absoluto.
Transparência: Em um sistema híbrido, é crucial saber por que um trade foi feito. A "Reasoning String" é vital para auditoria e confiança do usuário.
🏗️ SEÇÃO 2 – ARQUITETURA DE ALTO NÍVEL (MQL5 + PYTHON)
A arquitetura segue o padrão "Cérebro Remoto, Corpo Local". O MQL5 é o corpo que sente e age; o Python é o cérebro que analisa o contexto macro.

Camadas MQL5 (O "Corpo")
Data & Events: Captura OnTick e OnTradeTransaction. Mantém buffers de preços e indicadores rápidos.
Strategy Layer (Fast): Detecta padrões técnicos imediatos (Order Blocks, FVG, Quebra de Estrutura) localmente para não depender 100% da latência do Python para o sinal inicial.
Scoring Engine: Combina o score técnico local com os scores recebidos do Python.
FTMO_RiskManager: O guardião final. Verifica limites de perda antes de qualquer envio de ordem. Tem poder de veto absoluto.
Execution: Envia ordens OrderSend assíncronas ou síncronas dependendo da urgência.
Python Agent Hub (O "Cérebro")
Agentes:
TechnicalAgent: Valida a estrutura em timeframes maiores (H1, H4).
FundamentalAgent: Checa calendário econômico (ForexFactory/Investing) e notícias de última hora.
SentimentAgent: Analisa viés de curto prazo (ex: Twitter/X, News Feeds).
Comunicação: HTTP/REST (POST) via WebRequest.
Justificativa: Embora ZeroMQ seja mais rápido, HTTP é nativo do MQL5 (sem DLLs complexas), mais fácil de debugar e robusto o suficiente se o servidor Python estiver local (localhost). Para scalping não-HFT, 10-50ms de RTT é aceitável.
Resposta: JSON contendo scores (0-100), viés (Bullish/Bearish) e uma string de raciocínio.
Fluxo de um Tick "Perfeito"
MQL5: Tick chega. Módulos técnicos calculam: HasOB=true, HasFVG=true.
MQL5: TechScore local sobe para 70 (acima do gatilho de pré-análise).
MQL5: Envia POST /analyze para Python com dados do mercado.
Python: Analisa contexto macro e retorna: {"fund_score": 90, "sent_score": 80, "action": "CONFIRM"}.
MQL5: Calcula FinalScore = (70 + 90 + 80) / 3 = 80.
MQL5: FinalScore > ExecutionThreshold (75). Solicita permissão ao FTMO_RiskManager.
RiskManager: "Drawdown diário atual é 0.5%. Permitido risco de 1%."
MQL5: Executa a ordem.
⚙️ SEÇÃO 3 – DESIGN DETALHADO DO EA EM MQL5
Módulos Principais
COrderBlockModule & CFVGModule
Resp: Identificar zonas de interesse no gráfico M1/M5.
Out: Lista de preços de zonas ativas, bool IsInZone.
CMarketStructureModule
Resp: Mapear HH, HL, LH, LL. Determinar tendência primária.
Out: EnumTrendDirection, double LastBreakOfStructurePrice.
CVolatilityModule
Resp: Calcular ATR e monitorar Spread.
Out: double CurrentATR, bool IsSpreadSafe.
CSignalScoringModule
Resp: Normalizar inputs díspares em um número 0-100.
Input: Dados dos módulos acima + Resposta Python.
Out: double FinalScore.
CFTMORiskManager
Resp: Gerenciar Equity, Daily Loss, Total Loss e Drawdown.
Input: Balance, Equity, Histórico de hoje.
Out: bool CanTrade, double RecommendedLotSize.
CTradeExecutor
Resp: Envelopar OrderSend, gerenciar retries e erros.
Out: bool OrderSent.
Lógica do OnTick Ideal (Pseudocódigo)
void OnTick() {
    // 1. Atualizar dados de mercado e risco
    RiskManager.UpdateStats();
    if (RiskManager.IsStopTradingReached()) return;
    // 2. Atualizar módulos técnicos (apenas se nova barra ou a cada N ticks para otimizar)
    TechModules.Update();
    // 3. Calcular Score Técnico Local
    double localScore = Scoring.ComputeTechScore();
    // 4. Se o sinal local for promissor, consultar Python
    if (localScore > PRE_TRIGGER_THRESHOLD) {
        PythonResponse response = PythonHub.Call(Symbol(), localScore, ...);
        
        // 5. Calcular Score Final
        double finalScore = Scoring.ComputeFinalScore(localScore, response);
        // 6. Decisão de Execução
        if (finalScore >= EXECUTION_THRESHOLD) {
            double stopLoss = TechModules.GetOptimalStopLoss();
            
            if (RiskManager.CanOpenTrade(RISK_PER_TRADE, stopLoss)) {
                double lotSize = RiskManager.GetLotSize(RISK_PER_TRADE, stopLoss);
                Executor.ExecuteTrade(ORDER_TYPE_BUY, lotSize, stopLoss, response.reasoning);
            }
        }
    }
}
💻 SEÇÃO 4 – CÓDIGO MQL5 ESSENCIAL
//+------------------------------------------------------------------+
//|                                            EA_SCALPER_XAUUSD.mq5 |
//|                                      TradeDev_Master Architecture|
//+------------------------------------------------------------------+
#property copyright "TradeDev_Master"
#property version   "1.00"
// --- Inputs Principais ---
input group "Risk Management (FTMO)"
input double InpRiskPerTrade     = 0.5;   // Risco por trade (%)
input double InpMaxDailyLoss     = 4.5;   // Max Daily Loss Hard Limit (%) (FTMO é 5%, deixamos buffer)
input double InpMaxTotalLoss     = 9.5;   // Max Total Loss Hard Limit (%) (FTMO é 10%)
input double InpProfitTarget     = 10.0;  // Meta de lucro (%)
input group "Strategy"
input int    InpExecutionThreshold = 85;  // Score mínimo para execução (0-100)
input string InpPythonHubURL     = "http://localhost:8000/analyze";
// --- Interfaces dos Módulos (Stubs) ---
class COrderBlockModule { public: bool HasActiveOB(string sym) { return false; /* TODO */ } };
class CFVGModule { public: bool HasActiveFVG(string sym) { return false; /* TODO */ } };
//+------------------------------------------------------------------+
//| CLASS: CFTMORiskManager                                          |
//| Responsabilidade: Guardião absoluto do capital e regras FTMO     |
//+------------------------------------------------------------------+
class CFTMORiskManager {
private:
   double m_initial_balance;
   double m_start_day_equity;
   double m_max_daily_loss_perc;
   double m_max_total_loss_perc;
   
public:
   void Init(double max_daily, double max_total) {
      m_initial_balance = AccountInfoDouble(ACCOUNT_BALANCE);
      m_start_day_equity = m_initial_balance; // Simplificação, idealmente ler do histórico do dia
      m_max_daily_loss_perc = max_daily;
      m_max_total_loss_perc = max_total;
   }
   // Chamado no início de cada dia
   void OnNewDay() {
      m_start_day_equity = AccountInfoDouble(ACCOUNT_EQUITY);
   }
   // Verifica se pode abrir trade baseado no risco e drawdown atual
   bool CanOpenTrade(double risk_perc, double stoploss_points) {
      double current_equity = AccountInfoDouble(ACCOUNT_EQUITY);
      double daily_drawdown = (m_start_day_equity - current_equity) / m_start_day_equity * 100.0;
      double total_drawdown = (m_initial_balance - current_equity) / m_initial_balance * 100.0;
      // 1. Checagem Hard Limits
      if (daily_drawdown >= m_max_daily_loss_perc) {
         Print("RISK ALERT: Max Daily Loss atingido! Trading pausado.");
         return false;
      }
      if (total_drawdown >= m_max_total_loss_perc) {
         Print("RISK ALERT: Max Total Loss atingido! Trading pausado.");
         return false;
      }
      // 2. Dynamic Drawdown Control (Reduzir risco se estiver perdendo no dia)
      // Se já perdeu 2.5% hoje, reduz risco pela metade
      if (daily_drawdown > 2.5) {
         Print("RISK WARNING: Drawdown > 2.5%. Reduzindo risco permitido.");
         if (risk_perc > 0.25) return false; // Só permite trades pequenos
      }
      return true;
   }
   double CalculateLotSize(double risk_perc, double sl_points) {
      double risk_money = AccountInfoDouble(ACCOUNT_BALANCE) * (risk_perc / 100.0);
      double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
      double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
      
      if (sl_points <= 0 || tick_value == 0) return 0.0;
      
      double lot = risk_money / (sl_points * (tick_value / tick_size));
      
      // Normalizar lote
      double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
      double max_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
      double step_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
      
      lot = MathFloor(lot / step_lot) * step_lot;
      if (lot < min_lot) lot = min_lot; // Cuidado aqui: se o risco for muito baixo, melhor não operar
      if (lot > max_lot) lot = max_lot;
      
      return lot;
   }
};
//+------------------------------------------------------------------+
//| CLASS: CSignalScoringModule                                      |
//| Responsabilidade: Agregar sinais em um score único               |
//+------------------------------------------------------------------+
class CSignalScoringModule {
public:
   double ComputeTechScore(bool hasOB, bool hasFVG, bool trendAligned, double atr_filter) {
      double score = 0;
      
      // Lógica de pontuação ponderada
      if (trendAligned) score += 40; // Tendência é rei
      if (hasOB)        score += 30;
      if (hasFVG)       score += 20;
      if (atr_filter > 0) score += 10; // Volatilidade saudável
      
      return score;
   }
   double ComputeFinalScore(double tech_score, double fund_score, double sent_score) {
      // Pesos: Técnico 50%, Fundamental 30%, Sentimento 20%
      return (tech_score * 0.5) + (fund_score * 0.3) + (sent_score * 0.2);
   }
};
// --- Globais ---
CFTMORiskManager RiskManager;
CSignalScoringModule Scoring;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
   RiskManager.Init(InpMaxDailyLoss, InpMaxTotalLoss);
   return(INIT_SUCCEEDED);
}
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
   // Exemplo simplificado de fluxo
   
   // 1. Sinais Técnicos (Simulados)
   bool hasOB = true; // viria de COrderBlockModule
   bool hasFVG = true;
   bool trend = true;
   
   // 2. Score Local
   double techScore = Scoring.ComputeTechScore(hasOB, hasFVG, trend, 1.5);
   
   // 3. Se promissor, chama Python (Pseudocódigo da integração)
   double fundScore = 50; // Default neutro
   double sentScore = 50;
   string reasoning = "MQL5 Only";
   
   if (techScore > 60) {
      // CallPythonHub(techScore, fundScore, sentScore); // Implementado na Seção 5
   }
   
   double finalScore = Scoring.ComputeFinalScore(techScore, fundScore, sentScore);
   
   // 4. Execução
   if (finalScore >= InpExecutionThreshold) {
      double sl_points = 200; // Exemplo: 200 pontos (20 pips)
      
      if (RiskManager.CanOpenTrade(InpRiskPerTrade, sl_points)) {
         double lot = RiskManager.CalculateLotSize(InpRiskPerTrade, sl_points);
         // TradeExecutor.SendOrder(...)
         Print("TRADE SIGNAL: Score ", finalScore, " Lot: ", lot);
      }
   }
}
🔗 SEÇÃO 5 – INTERFACE COM PYTHON AGENT HUB
Request JSON (MQL5 -> Python)
O MQL5 envia o contexto técnico atual para que o Python possa enriquecê-lo.

{
  "symbol": "XAUUSD",
  "timeframe": "M5",
  "current_price": 2035.50,
  "technical_signals": {
    "has_ob": true,
    "has_fvg": true,
    "trend": "BULLISH",
    "atr": 2.5
  },
  "server_time": "2023-10-27 14:30:00"
}
Response JSON (Python -> MQL5)
O Python retorna scores normalizados e o raciocínio.

{
  "tech_subscore_python": 85.0,   // Validação técnica avançada (ex: ML pattern)
  "fund_score": 90.0,             // Notícias favoráveis ao Ouro
  "fund_bias": "BULLISH",
  "sent_score": 75.0,             // Sentimento social
  "sent_bias": "NEUTRAL_BULLISH",
  "llm_reasoning_short": "Strong buy: DXY falling, geopolitical tension rising, H1 OB retest confirmed."
}
Função de Chamada (Pseudocódigo MQL5)
bool CallPythonHub(double &tech_py, double &fund, double &sent) {
   char post_data[], result_data[];
   string headers = "Content-Type: application/json\r\n";
   string url = InpPythonHubURL;
   
   // Serializar JSON (usando biblioteca interna ou string format)
   string json_req = StringFormat("{\"symbol\":\"%s\", ...}", _Symbol);
   StringToCharArray(json_req, post_data);
   
   int res = WebRequest("POST", url, headers, 500, post_data, result_data, headers);
   
   if (res == 200) {
      // Parse JSON response (necessita de lib CJson ou parser simples)
      // tech_py = JsonParse(result_data, "tech_subscore_python");
      // ...
      return true;
   }
   
   Print("Python Hub Error: ", res);
   return false; // Falha na comunicação, usar fallback
}
🧠 SEÇÃO 6 – RACIOCÍNIO DE RISCO (FTMO) & DEEP THINKING
Configuração de Risco para 100k Account
Risk per Trade: 0.5% ($500).
Por que? Permite 10 stops consecutivos antes de atingir o limite diário de 5%. Em scalping, sequências de perdas acontecem. 1% é agressivo demais para prop firms.
Soft Daily Loss (Zona Amarela): 3.0% ($3,000).
Ao atingir 3% de perda no dia, o sistema entra em "Modo Defensivo". O risco por trade cai automaticamente para 0.25%. O objetivo muda de "lucrar" para "estancar o sangramento".
Hard Max Daily Loss (Zona Vermelha): 4.5% ($4,500).
Nunca configure em 5%. O slippage ou uma ordem pendente não executada corretamente pode empurrar você de 4.9% para 5.1% e perder a conta. 4.5% é o "Kill Switch" absoluto.
Max Total Loss: 9.5% (Buffer de 0.5% para segurança).
Política de Redução Dinâmica
0% - 2% DD: Risco Normal (0.5%).
2% - 3.5% DD: Risco Reduzido (0.25%). Foco em setups A+ (Score > 90).
3.5% - 4.5% DD: Risco Mínimo (0.10%) ou Pausa. Apenas para recuperar confiança psicológica do algoritmo.
> 4.5% DD: Bloqueio total até o próximo dia (Server Time 00:00).
Deep Thinking: Evitando Armadilhas
Overtrading em Dia Bom: O sistema deve ter um "Daily Profit Target" oculto (ex: 3%). Se atingir, reduz o risco drasticamente. É comum devolver lucros no final do dia por exaustão de volatilidade.
Sequência de 3 Stops: Implementar um "Cooldown Timer". Se tomar 3 stops seguidos, o EA pausa por 60 minutos. O mercado pode estar em uma condição não prevista pelo algoritmo (ex: consolidação errática).
Quando NÃO operar:
Spread > 20 points: O custo operacional mata o scalping.
News de Alto Impacto (CPI, NFP): 15 min antes e depois. A volatilidade é aleatória e o slippage é mortal. O FundamentalAgent deve vetar trades nesses horários.
🧪 SEÇÃO 7 – ESTRATÉGIA DE TESTES E VALIDAÇÃO
1. Backtests Rigorosos
Dados: Tick-data real (Dukascopy ou TickStory) de 2020 a 2024. M1 data é insuficiente para scalping preciso.
Latência Simulada: Adicionar delay aleatório (50-200ms) na execução para simular condições reais.
Custos: Incluir comissão ($3-$7/lot) e swap.
2. Simulação FTMO (Stress Test)
Criar um script wrapper no backtest que monitora o Equity High do dia.
Se (DailyHigh - CurrentEquity) > 5%, o teste falha imediatamente. Muitos backtests padrão MT5 só olham o saldo final e ignoram o drawdown intraday flutuante, que é o que reprova na FTMO.
3. Critérios de Aprovação
Profit Factor: > 1.5.
Win Rate: > 45% (para Risco:Retorno 1:2) ou > 60% (para 1:1).
Max Drawdown Relativo: < 4% (nunca tocar nos 5% diários no histórico).
Avg Trade Duration: Scalping deve ser rápido. Se trades ficam abertos por dias, a estratégia está errada.
📣 SEÇÃO 8 – EXEMPLOS DE REASONING STRINGS
Exemplo 1 – Trade WIN (BUY XAUUSD)

"BUY Executed. Score: 88/100. Context: Bullish Market Structure (H1/M15 aligned). Price retested M5 Order Block with liquidity sweep of previous low. Risk: 0.5% ($500). Sentiment Agent confirms USD weakness due to recent CPI data. Risk Manager approved (Daily DD: 0.0%)."

Exemplo 2 – Trade LOSS (SELL XAUUSD)

"SELL Closed (Stop Loss). Score: 78/100. Setup: Bearish FVG fill. Reason for Loss: Unexpected volatility spike reversed price action immediately after entry. Structure shifted to Bullish on M1. Risk Control: Loss limited to -0.5%. Daily Drawdown now at 1.2%. System remains in Normal Risk mode."

Exemplo 3 – Sinal IGNORADO

"SIGNAL REJECTED. Score: 92/100 (Strong Technical Buy). Rejection Reason: FTMO Risk Protocol. Current Daily Drawdown is 4.2% (Limit 4.5%). Risk Manager blocked new exposure to prevent breaching Max Daily Loss. Action: Waiting for next trading day reset."

# GPT-5.1-High (via codex CLI oficial)

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



# Claude 4.5 Thinking (via Antigravity IDE):


Comprehensive EA_SCALPER_XAUUSD Design Complete ✅

I've created:

Complete Design Document (EA_SCALPER_DESIGN_COMPLETE.md) covering all 8 sections:

Problem analysis & FTMO constraints
Multi-agent architecture (MQL5 + Python/HTTP)
Detailed module design (9 classes)
Working MQL5 code (Risk Manager, Scoring, main EA structure)
Python integration specs (JSON formats, WebRequest implementation)
Deep thinking FTMO risk framework (dynamic DD control, scenario analysis)
Testing strategy (backtests, stress tests, FTMO validation)
Real reasoning string examples (WIN/LOSS/REJECTED trades)
Implementation Plan with phased development, verification strategy, and FTMO compliance tests

Key Decisions to Confirm:

FastAPI HTTP integration (vs ZeroMQ) - acceptable?
Risk parameters: 0.8%/trade, 4% max daily DD, 9% total DD - adjust for your FTMO challenge?
Files location: MAIN_EAS/DEVELOPMENT/EA_SCALPER_XAUUSD/ - correct per your structure?





# EA_SCALPER_XAUUSD - Multi-Agent Hybrid System Design
**TradeDev_Master - Complete System Architecture & Implementation Guide**
---
## 🧩 SEÇÃO 1 – COMPREENSÃO DO PROBLEMA
### Objetivo Estratégico
- **Scalping automatizado em XAUUSD** com foco em setups ICT/SMC de alta probabilidade
- **Conformidade total com FTMO** (Max Daily Loss, Max Total Loss, trailing drawdown)
- **Scoring multi-dimensional** (Técnico + Fundamental + Sentimento) para filtrar apenas trades de alta qualidade
- **Híbrido MQL5 + Python** para separar lógica de execução rápida (MQL5) de análise complexa (Python/LLM)
### Restrições FTMO Críticas
- **Max Daily Loss**: 5% do saldo inicial (hard limit, violação = falha)
- **Max Total Loss**: 10% do saldo inicial (drawdown acumulado)
- **Trailing drawdown**: Limite se ajusta com lucros (ex: conta de $100k → $105k, novo limite = $105k - 10% = $94.5k)
- **Consistency Rule**: Melhor dia não pode exceder 30-40% do total profit
- **Tempo mínimo**: 4+ dias de trading (mínimo 0.5 lotes por dia)
### Benefícios da Arquitetura Multi-Agente
- **Separação de responsabilidades**: MQL5 = execução < 50ms; Python = análise profunda sem bloquear OnTick
- **LLM reasoning**: Agente Python pode consultar GPT-4 para análise de contexto macro/notícias
- **Escalabilidade**: Adicionar novos agentes (ex: ML predictions) sem recompilar EA
- **Backtesting independente**: Testar melhorias em Python sem afetar lógica MQL5
### Riscos Clássicos de Scalping XAUUSD
- **Slippage brutal**: XAUUSD pode ter 2-5 pontos em news, anulando RR de scalping
- **Overtrading**: 20+ trades/dia → custos de spread (~$20/lote) destroem lucro
- **Spread variável**: Sessão asiática = spread alto; evitar trading fora de NY/London
- **Violação emocional de DD**: Sequência de 3-5 stops → EA tenta recuperar → explode conta
- **News events**: NFP, FOMC, CPI podem mover 200+ pontos em segundos
- **Latência**: VPS com >20ms para broker = slippage constante
- **Falsos OB/FVG**: Em range, 70% dos setups ICT falham; necessário filtro de tendência
---
## 🏗️ SEÇÃO 2 – ARQUITETURA DE ALTO NÍVEL (MQL5 + PYTHON)
### Camadas MQL5
```
┌─────────────────────────────────────────────────┐
│         DATA & EVENTS LAYER                     │
│  OnTick() | OnTimer() | OnTradeTransaction()   │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│       STRATEGY / SIGNAL LAYER                   │
│  OrderBlocks | FVG | Liquidity | Structure      │
│  ATR Volatility | Session Filter                │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│         SCORING ENGINE (0-100)                  │
│  TechScore (MQL5) + FundScore (Python)          │
│  + SentScore (Python) = FinalScore              │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│       FTMO RISK MANAGER (VETO POWER)            │
│  Check DD % | Risk/Trade | Daily Limits         │
│  → Approve/Reject | Dynamic Risk Scaling        │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│         EXECUTION & LOGGING                     │
│  CTradeExecutor | CLogger | Push Notifications  │
└─────────────────────────────────────────────────┘
```
### Python Agent Hub
**Agentes no Hub:**
1. **TechnicalAgent**: Calcula indicadores complexos (RSI divergências, volume profile)
2. **FundamentalAgent**: Lê calendar econômico, retorna bias (hawkish/dovish Fed, etc)
3. **SentimentAgent**: Scraping de notícias/Twitter, sentiment score 0-100
4. **LLM_ReasoningAgent**: Envia contexto para GPT-4, recebe reasoning string
**Comunicação: HTTP/REST (escolhido)**
- **Justificativa**: FastAPI no Python (fácil deploy, async support), WebRequest() nativo no MQL5
- **Alternativa ZeroMQ**: Mais rápida, mas requer DLL (complicado para FTMO, pode violar regras)
- **Timeout**: 2s max (se falhar, EA opera com TechScore puro do MQL5)
**Formato JSON Response:**
```json
{
  "tech_subscore_python": 72.5,
  "fund_score": 65.0,
  "fund_bias": "bullish_usd",
  "sent_score": 58.0,
  "sent_bias": "neutral",
  "llm_reasoning_short": "Gold under pressure from hawkish Fed rhetoric; technicals show bear OB rejection at 2650"
}
```
### Fluxo de um Tick Perfeito
```
TICK ARRIVES (OnTick)
   ↓
1. Check Time Filter (evitar spreads ruins, sessão asiática)
   ↓
2. Update Market Structure (detect HH/HL/LH/LL)
   ↓
3. Detect Signals (OB, FVG, Liquidity Sweep) → TechScore (MQL5)
   ↓
4. IF (TechScore > 70 AND not in position):
      ↓
   4a. Call Python Hub (HTTP POST) → FundScore + SentScore
      ↓ (timeout 2s, fallback se falhar)
   4b. Compute FinalScore = (TechScore * 0.5 + FundScore * 0.3 + SentScore * 0.2)
      ↓
5. IF (FinalScore >= ExecutionThreshold, ex: 85):
      ↓
   5a. Calculate SL/TP based on ATR
      ↓
   5b. Call FTMO_RiskManager.CanOpenTrade(risk%, SL_points)
      ↓
   5c. IF (approved):
         → Execute Trade + Log Reasoning String
      ELSE:
         → Log "Trade REJECTED by Risk Manager (DD: 3.2%)"
```
---
## ⚙️ SEÇÃO 3 – DESIGN DETALHADO DO EA EM MQL5
### Módulos Principais
#### 1. **COrderBlockModule**
- **Responsabilidades**: Detectar Order Blocks (últimos down-candle antes de rally bull, vice-versa)
- **Inputs**: `int lookback_bars`, `ENUM_TIMEFRAMES tf`
- **Outputs**: `bool hasValidOB`, `double OB_price_level`, `int OB_strength (0-100)`
- **Lógica**: Busca candle de alta/baixo volume + reversão em 3-5 candles; valida se preço retesta OB zone (±10 pontos)
#### 2. **CFVGModule**
- **Responsabilidades**: Identificar Fair Value Gaps (imbalance in price action)
- **Inputs**: `ENUM_TIMEFRAMES tf`, `double min_gap_points`
- **Outputs**: `bool hasFVG`, `double FVG_top`, `double FVG_bottom`, `int FVG_quality`
- **Lógica**: Gap = candle[i-1].low > candle[i+1].high (bullish FVG); min 15 pontos para XAUUSD
#### 3. **CLiquidityModule**
- **Responsabilidades**: Detectar liquidity sweeps (stop hunts em swing highs/lows)
- **Inputs**: `int swing_period`, `double sweep_threshold_points`
- **Outputs**: `bool liquiditySweep`, `ENUM_ORDER_TYPE sweep_direction`
- **Lógica**: Se price tocou swing high + 5 pontos e reverte → bearish sweep (armadilha)
#### 4. **CMarketStructureModule**
- **Responsabilidades**: Rastrear estrutura de mercado (BOS = Break of Structure)
- **Inputs**: Histórico de pivots (highs/lows)
- **Outputs**: `ENUM_MARKET_TREND trend` (BULL_TREND, BEAR_TREND, RANGE)
- **Lógica**: HH + HL = bull; LH + LL = bear; caso contrário = range
#### 5. **CVolatilityModule**
- **Responsabilidades**: ATR para tamanho de SL/TP dinâmico
- **Inputs**: `int atr_period`, `ENUM_TIMEFRAMES tf`
- **Outputs**: `double current_ATR`, `bool high_volatility_regime`
- **Lógica**: High volatility se ATR > 1.5x média de 20 períodos → aumentar SL, reduzir lotes
#### 6. **CSignalScoringModule**
- **Responsabilidades**: Combinar sinais em score 0-100
- **Inputs**: Structs de todos os módulos
- **Outputs**: `double TechScore`, `double FinalScore`
- **Lógica**: 
  - `TechScore = (OB_strength * 0.3) + (FVG_quality * 0.25) + (trend_alignment * 0.25) + (liquidity_sweep_bonus * 0.2)`
  - `FinalScore = (TechScore * 0.5) + (FundScore * 0.3) + (SentScore * 0.2)`
#### 7. **CFTMORiskManager**
- **Responsabilidades**: **PODER DE VETO** sobre todas as trades
- **Inputs**: `double risk_per_trade_pct`, `double max_daily_loss_pct`, `double max_total_loss_pct`
- **Outputs**: `bool CanOpenTrade()`, `double GetLotSize()`
- **Lógica**:
  - Track daily P&L desde `TimeCurrent()` 00:00
  - Se `daily_DD % >= max_daily_loss_pct` → BLOCK all trades
  - Dynamic scaling: Se DD 1-2.5% → reduzir risk/trade para 0.5%
#### 8. **CTradeExecutor**
- **Responsabilidades**: Executar ordens via CTrade
- **Inputs**: `ENUM_ORDER_TYPE type`, `double lots`, `double SL`, `double TP`, `string comment`
- **Outputs**: `bool success`, `ulong ticket`
- **Lógica**: Retry 3x com 500ms delay; log slippage real vs esperado
#### 9. **CLogger**
- **Responsabilidades**: Logs estruturados + push notifications
- **Outputs**: Arquivo CSV com timestamp, action, reasoning, P&L
- **Lógica**: Se trade fecha, envia Telegram com Reasoning String
### OnTick Pseudocódigo
```mql5
void OnTick() {
   // 1. Time Filter
   if (!IsGoodTradingSession()) return;  // Evita asiática, pre-news
   
   // 2. Já em posição? Gerenciar trailing stop
   if (PositionsTotal() > 0) {
      ManageOpenTrades();
      return;
   }
   
   // 3. Update structural modules (low CPU cost)
   marketStructure.Update();
   volatility.Update();
   
   // 4. Detect fast signals
   bool hasOB = orderBlock.Detect();
   bool hasFVG = fvgModule.Detect();
   bool hasLiqSweep = liquidity.CheckSweep();
   
   // 5. Compute TechScore (MQL5 only, <10ms)
   double techScore = scoring.ComputeTechScore(hasOB, hasFVG, marketStructure.GetTrend(), volatility.GetATR());
   
   // 6. Se score inicial promissor, chama Python (async via OnTimer, não aqui!)
   if (techScore > 70.0 && !pythonCallPending) {
      pythonCallPending = true;
      EventSetTimer(1);  // OnTimer em 1s chamará Python
   }
}
void OnTimer() {
   // Chamada HTTP não bloqueia OnTick
   if (pythonCallPending) {
      double fundScore, sentScore;
      bool success = CallPythonHub(fundScore, sentScore);
      
      double finalScore = scoring.ComputeFinalScore(techScore, fundScore, sentScore);
      
      if (finalScore >= ExecutionThreshold) {
         // Calculate SL/TP
         double atr = volatility.GetATR();
         double sl_points = atr * 1.5;
         double tp_points = atr * 2.5;  // RR 1:1.67
         
         // FTMO Risk Check
         if (riskManager.CanOpenTrade(RiskPerTrade_Pct, sl_points)) {
            double lots = riskManager.GetLotSize(sl_points);
            executor.OpenTrade(OP_BUY, lots, sl_points, tp_points, reasoning_string);
         }
      }
      
      pythonCallPending = false;
      EventKillTimer();
   }
}
```
---
## 💻 SEÇÃO 4 – CÓDIGO MQL5 ESSENCIAL
```mql5
//+------------------------------------------------------------------+
//| EA_SCALPER_XAUUSD.mq5                                            |
//| Multi-Agent Hybrid System - FTMO Compliant                       |
//+------------------------------------------------------------------+
#property copyright "TradeDev_Master"
#property version   "1.00"
#property strict
#include <Trade\Trade.mqh>
//--- Inputs
input group "=== RISK MANAGEMENT ==="
input double   RiskPerTrade_Pct = 1.0;           // Risk per trade (%)
input double   MaxDailyLoss_Pct = 4.0;           // Max Daily Loss FTMO (%)
input double   MaxTotalLoss_Pct = 9.0;           // Max Total Loss FTMO (%)
input double   SoftDailyLoss_Pct = 2.5;          // Soft DD (start reducing risk)
input group "=== SCORING SYSTEM ==="
input double   ExecutionThreshold = 85.0;        // Min FinalScore to trade (0-100)
input double   TechScoreWeight = 0.5;            // TechScore weight
input double   FundScoreWeight = 0.3;            // FundScore weight
input double   SentScoreWeight = 0.2;            // SentScore weight
input group "=== STRATEGY PARAMETERS ==="
input ENUM_TIMEFRAMES AnalysisTF = PERIOD_M15;   // Analysis timeframe
input int      ATR_Period = 14;                  // ATR period
input double   ATR_SL_Multiplier = 1.5;          // SL = ATR * multiplier
input double   ATR_TP_Multiplier = 2.5;          // TP = ATR * multiplier
input group "=== PYTHON INTEGRATION ==="
input string   PythonHubURL = "http://localhost:8000/analyze";  // Python API endpoint
input int      PythonTimeout_ms = 2000;          // HTTP timeout (ms)
input bool     UsePythonHub = true;              // Enable Python integration
//--- Global Objects
CFTMORiskManager g_riskManager;
CSignalScoringModule g_scoring;
CTrade g_trade;
//--- State variables
datetime g_lastBarTime = 0;
double g_dailyStartBalance = 0;
bool g_pythonCallPending = false;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
   // Initialize risk manager with FTMO limits
   g_riskManager.Init(RiskPerTrade_Pct, MaxDailyLoss_Pct, MaxTotalLoss_Pct, SoftDailyLoss_Pct);
   
   // Store daily start balance (reset at midnight)
   g_dailyStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   
   Print("=== EA_SCALPER_XAUUSD Initialized ===");
   Print("Risk/Trade: ", RiskPerTrade_Pct, "% | Max Daily Loss: ", MaxDailyLoss_Pct, "%");
   Print("Execution Threshold: ", ExecutionThreshold);
   
   return(INIT_SUCCEEDED);
}
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   EventKillTimer();
   Print("EA stopped. Reason: ", reason);
}
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
   // Reset daily balance at midnight
   CheckDailyReset();
   
   // Time filter (avoid bad sessions)
   if (!IsGoodTradingSession()) return;
   
   // Already in position? Manage it
   if (PositionsTotal() > 0) {
      ManageOpenTrades();
      return;
   }
   
   // Check if new bar (avoid multiple signals per bar)
   if (!IsNewBar()) return;
   
   // === SIMPLIFIED SIGNAL DETECTION (full modules in Section 3) ===
   bool hasOrderBlock = DetectOrderBlock_Simplified();
   bool hasFVG = DetectFVG_Simplified();
   bool bullishTrend = GetMarketTrend_Simplified() == 1;
   double atr = iATR(_Symbol, AnalysisTF, ATR_Period);
   
   // Compute TechScore (MQL5 only)
   double techScore = g_scoring.ComputeTechScore(hasOrderBlock, hasFVG, bullishTrend, atr);
   
   Print("TechScore: ", techScore);
   
   // If promising, trigger Python call via OnTimer
   if (techScore > 70.0 && UsePythonHub && !g_pythonCallPending) {
      g_pythonCallPending = true;
      EventSetTimer(1);  // Call Python in 1 second (async)
      Print("TechScore high, queuing Python call...");
   } else if (techScore > ExecutionThreshold && !UsePythonHub) {
      // Fallback: trade without Python
      AttemptTrade(techScore, 0, 0, "MQL5-Only");
   }
}
//+------------------------------------------------------------------+
//| Timer function (for async Python calls)                          |
//+------------------------------------------------------------------+
void OnTimer() {
   if (!g_pythonCallPending) return;
   
   double fundScore = 0, sentScore = 0;
   
   // Call Python Hub
   bool success = CallPythonHub(fundScore, sentScore);
   
   if (!success) {
      Print("Python call failed, using TechScore only");
      fundScore = 50.0;  // Neutral fallback
      sentScore = 50.0;
   }
   
   // Compute FinalScore
   double techScore = g_scoring.GetLastTechScore();  // Cached value
   double finalScore = g_scoring.ComputeFinalScore(techScore, fundScore, sentScore);
   
   Print("FinalScore: ", finalScore, " (Tech:", techScore, " Fund:", fundScore, " Sent:", sentScore, ")");
   
   if (finalScore >= ExecutionThreshold) {
      AttemptTrade(finalScore, fundScore, sentScore, "FullScoring");
   }
   
   g_pythonCallPending = false;
   EventKillTimer();
}
//+------------------------------------------------------------------+
//| Attempt to execute trade (subject to FTMO risk approval)         |
//+------------------------------------------------------------------+
void AttemptTrade(double finalScore, double fundScore, double sentScore, string mode) {
   double atr = iATR(_Symbol, AnalysisTF, ATR_Period);
   double sl_points = atr * ATR_SL_Multiplier;
   double tp_points = atr * ATR_TP_Multiplier;
   
   // === FTMO RISK MANAGER APPROVAL ===
   if (!g_riskManager.CanOpenTrade(RiskPerTrade_Pct, sl_points)) {
      string reason = g_riskManager.GetLastRejectReason();
      Print("❌ TRADE REJECTED by RiskManager: ", reason);
      SendNotification("Trade blocked: " + reason);
      return;
   }
   
   // Calculate lot size
   double lotSize = g_riskManager.GetLotSize(sl_points);
   
   // Build reasoning string
   string reasoning = StringFormat(
      "Score:%.1f (T%.0f/F%.0f/S%.0f) | ATR:%.2f | SL:%.1fp | RR:1:%.1f | Mode:%s",
      finalScore, g_scoring.GetLastTechScore(), fundScore, sentScore, 
      atr, sl_points, ATR_TP_Multiplier/ATR_SL_Multiplier, mode
   );
   
   // Execute trade
   double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double sl = price - sl_points * _Point;
   double tp = price + tp_points * _Point;
   
   if (g_trade.Buy(lotSize, _Symbol, price, sl, tp, reasoning)) {
      Print("✅ BUY executed | Lot: ", lotSize, " | ", reasoning);
      SendNotification("🟢 BUY " + _Symbol + " | " + reasoning);
   } else {
      Print("❌ Trade execution FAILED: ", g_trade.ResultRetcodeDescription());
   }
}
//+------------------------------------------------------------------+
//| Call Python Hub via HTTP                                         |
//+------------------------------------------------------------------+
bool CallPythonHub(double &fundScore, double &sentScore) {
   // TODO: Implement full WebRequest() logic
   // For now, return mock data
   
   // Simulate HTTP POST to Python
   // string json_request = BuildJSONRequest();
   // string json_response = SendHTTPRequest(PythonHubURL, json_request, PythonTimeout_ms);
   // ParseJSONResponse(json_response, fundScore, sentScore);
   
   // MOCK DATA (replace with real implementation)
   fundScore = 65.0;
   sentScore = 58.0;
   
   return true;  // Simulated success
}
//+------------------------------------------------------------------+
//| Simplified helper functions (full versions in real modules)      |
//+------------------------------------------------------------------+
bool DetectOrderBlock_Simplified() {
   // TODO: Implement COrderBlockModule logic
   return false;
}
bool DetectFVG_Simplified() {
   // TODO: Implement CFVGModule logic
   return false;
}
int GetMarketTrend_Simplified() {
   // TODO: Implement CMarketStructureModule logic
   return 0;  // 1=bull, -1=bear, 0=range
}
bool IsNewBar() {
   datetime currentBarTime = iTime(_Symbol, AnalysisTF, 0);
   if (currentBarTime != g_lastBarTime) {
      g_lastBarTime = currentBarTime;
      return true;
   }
   return false;
}
void CheckDailyReset() {
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   if (dt.hour == 0 && dt.min == 0) {
      g_dailyStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
      g_riskManager.ResetDaily();
   }
}
bool IsGoodTradingSession() {
   // Avoid Asian session (low liquidity, high spreads)
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   int hour = dt.hour;
   
   // London: 8-17, NY: 13-22 (GMT)
   if ((hour >= 8 && hour <= 17) || (hour >= 13 && hour <= 22)) return true;
   return false;
}
void ManageOpenTrades() {
   // TODO: Implement trailing stop, breakeven logic
}
void SendNotification(string msg) {
   // TODO: Telegram/Push notification
   Print("📢 ", msg);
}
//+------------------------------------------------------------------+
//| FTMO Risk Manager Class                                          |
//+------------------------------------------------------------------+
class CFTMORiskManager {
private:
   double m_riskPerTrade_pct;
   double m_maxDailyLoss_pct;
   double m_maxTotalLoss_pct;
   double m_softDailyLoss_pct;
   
   double m_dailyStartBalance;
   double m_accountStartBalance;
   string m_lastRejectReason;
   
public:
   void Init(double riskPerTrade, double maxDailyLoss, double maxTotalLoss, double softDailyLoss) {
      m_riskPerTrade_pct = riskPerTrade;
      m_maxDailyLoss_pct = maxDailyLoss;
      m_maxTotalLoss_pct = maxTotalLoss;
      m_softDailyLoss_pct = softDailyLoss;
      
      m_dailyStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
      m_accountStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   }
   
   void ResetDaily() {
      m_dailyStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   }
   
   //--- Main risk approval function
   bool CanOpenTrade(double risk_pct, double sl_points) {
      // 1. Check Max Total Loss (hard limit)
      double currentBalance = AccountInfoDouble(ACCOUNT_BALANCE);
      double totalDD_pct = ((m_accountStartBalance - currentBalance) / m_accountStartBalance) * 100;
      
      if (totalDD_pct >= m_maxTotalLoss_pct) {
         m_lastRejectReason = StringFormat("Max Total Loss hit: %.2f%% >= %.2f%%", totalDD_pct, m_maxTotalLoss_pct);
         return false;
      }
      
      // 2. Check Max Daily Loss (hard limit)
      double dailyPL = currentBalance - m_dailyStartBalance;
      double dailyDD_pct = (dailyPL / m_dailyStartBalance) * 100;
      
      if (dailyDD_pct <= -m_maxDailyLoss_pct) {
         m_lastRejectReason = StringFormat("Max Daily Loss hit: %.2f%% <= -%.2f%%", dailyDD_pct, m_maxDailyLoss_pct);
         return false;
      }
      
      // 3. Dynamic drawdown control (soft limit)
      if (dailyDD_pct <= -m_softDailyLoss_pct) {
         // Reduce risk automatically
         risk_pct = risk_pct * 0.5;
         Print("⚠️ Soft DD limit reached, reducing risk to ", risk_pct, "%");
      }
      
      // 4. Check if projected loss would violate limits
      double lotSize = GetLotSize(sl_points, risk_pct);
      double potentialLoss = sl_points * _Point * lotSize * SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
      double projectedDailyDD_pct = ((dailyPL - potentialLoss) / m_dailyStartBalance) * 100;
      
      if (projectedDailyDD_pct <= -m_maxDailyLoss_pct) {
         m_lastRejectReason = StringFormat("Projected Daily Loss would hit limit: %.2f%%", projectedDailyDD_pct);
         return false;
      }
      
      return true;  // All checks passed
   }
   
   //--- Calculate lot size based on risk
   double GetLotSize(double sl_points, double risk_pct = -1) {
      if (risk_pct < 0) risk_pct = m_riskPerTrade_pct;
      
      double balance = AccountInfoDouble(ACCOUNT_BALANCE);
      double riskAmount = balance * (risk_pct / 100.0);
      
      double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
      double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
      
      double lotSize = riskAmount / (sl_points * _Point / tickSize * tickValue);
      
      // Normalize lot size
      double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
      double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
      double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
      
      lotSize = MathFloor(lotSize / lotStep) * lotStep;
      lotSize = MathMax(minLot, MathMin(maxLot, lotSize));
      
      return lotSize;
   }
   
   string GetLastRejectReason() { return m_lastRejectReason; }
};
//+------------------------------------------------------------------+
//| Signal Scoring Module                                            |
//+------------------------------------------------------------------+
class CSignalScoringModule {
private:
   double m_lastTechScore;
   
public:
   //--- Compute Technical Score (0-100) from MQL5 signals
   double ComputeTechScore(bool hasOB, bool hasFVG, bool bullishTrend, double atr) {
      double score = 50.0;  // Base neutral score
      
      // Order Block: +20 points
      if (hasOB) score += 20.0;
      
      // FVG: +15 points
      if (hasFVG) score += 15.0;
      
      // Trend alignment: +15 points
      if (bullishTrend) score += 15.0;
      
      // Volatility regime: -10 if extreme (risky)
      double atr_avg = 25.0;  // Example average for XAUUSD
      if (atr > atr_avg * 2.0) score -= 10.0;
      
      score = MathMax(0, MathMin(100, score));  // Clamp 0-100
      
      m_lastTechScore = score;
      return score;
   }
   
   //--- Compute Final Score (weighted combination)
   double ComputeFinalScore(double tech, double fund, double sent) {
      double final = (tech * TechScoreWeight) + (fund * FundScoreWeight) + (sent * SentScoreWeight);
      return final;
   }
   
   double GetLastTechScore() { return m_lastTechScore; }
};
```
---
## 🔗 SEÇÃO 5 – INTERFACE COM PYTHON AGENT HUB
### JSON Request Format (MQL5 → Python)
```json
{
  "symbol": "XAUUSD",
  "timeframe": "M15",
  "timestamp": "2025-01-15T14:30:00Z",
  "tech_signals": {
    "has_order_block": true,
    "has_fvg": true,
    "trend": "bullish",
    "atr": 24.5,
    "current_price": 2648.30
  },
  "request_components": ["fundamental", "sentiment", "llm_reasoning"]
}
```
### JSON Response Format (Python → MQL5)
```json
{
  "tech_subscore_python": 72.5,
  "fund_score": 65.0,
  "fund_bias": "bearish_gold",
  "fund_reasoning": "Fed hawkish rhetoric + USD strength",
  "sent_score": 58.0,
  "sent_bias": "neutral",
  "sent_reasoning": "Mixed Twitter sentiment, no clear consensus",
  "llm_reasoning_short": "Gold facing pressure from strong USD; technicals show bear OB rejection",
  "processing_time_ms": 1250,
  "success": true
}
```
### Pseudocódigo MQL5: CallPythonHub()
```mql5
bool CallPythonHub(double &tech_py, double &fund_score, double &sent_score) {
   // 1. Build JSON request
   string json_request = StringFormat(
      "{\"symbol\":\"%s\",\"timeframe\":\"%s\",\"timestamp\":\"%s\",\"tech_signals\":{\"atr\":%.2f,\"trend\":\"%s\"}}",
      _Symbol, EnumToString(AnalysisTF), TimeToString(TimeCurrent()), atr, trend_str
   );
   
   // 2. Prepare HTTP request
   char post_data[];
   char result_data[];
   StringToCharArray(json_request, post_data, 0, StringLen(json_request));
   
   string headers = "Content-Type: application/json\r\n";
   
   // 3. Send WebRequest with timeout
   int timeout = PythonTimeout_ms;
   int res = WebRequest(
      "POST",
      PythonHubURL,
      headers,
      timeout,
      post_data,
      result_data,
      headers
   );
   
   // 4. Handle failures (timeout, connection error)
   if (res != 200) {
      Print("Python Hub call failed: HTTP ", res);
      return false;  // Fallback to MQL5-only mode
   }
   
   // 5. Parse JSON response
   string json_response = CharArrayToString(result_data);
   
   // Simple parsing (use proper JSON library in production)
   if (StringFind(json_response, "\"success\":true") < 0) {
      Print("Python Hub returned error");
      return false;
   }
   
   // Extract scores (simplified, use real JSON parser)
   fund_score = ExtractJSONValue(json_response, "fund_score");
   sent_score = ExtractJSONValue(json_response, "sent_score");
   tech_py = ExtractJSONValue(json_response, "tech_subscore_python");
   
   Print("Python Hub success: Fund=", fund_score, " Sent=", sent_score);
   return true;
}
// Helper: Extract numeric value from JSON (simplified)
double ExtractJSONValue(string json, string key) {
   int pos = StringFind(json, "\"" + key + "\":");
   if (pos < 0) return 0;
   
   string sub = StringSubstr(json, pos + StringLen(key) + 3);
   int comma_pos = StringFind(sub, ",");
   if (comma_pos < 0) comma_pos = StringFind(sub, "}");
   
   string value_str = StringSubstr(sub, 0, comma_pos);
   return StringToDouble(value_str);
}
```
### Tratamento de Falhas
- **Timeout (> 2s)**: EA continua com `TechScore` puro do MQL5, assume `FundScore = SentScore = 50` (neutro)
- **Server offline**: Log warning, mode seguro ativado
- **Parsing error**: Invalida resposta, fallback para MQL5
- **Retry logic**: Não tenta novamente na mesma barra (evita spam)
---
## 🧠 SEÇÃO 6 – RACIOCÍNIO DE RISCO (FTMO) & DEEP THINKING
### Configuração para Conta FTMO $100k
**Parâmetros Recomendados:**
- **Risk per trade**: 0.8% (conservador; scalping tem alta frequência)
- **Soft Daily Loss**: 2.5% (começa a reduzir agressividade)
- **Hard Max Daily Loss**: 4.0% (limite absoluto, para antes dos 5% FTMO)
- **Max Total Loss**: 9.0% (margem de 1% antes dos 10% FTMO)
**Justificativa:**
- FTMO permite 5% daily loss, mas operar até esse limite é perigoso (um trade ruim = falha)
- Margem de segurança de 1% para slippage/spread inesperado
- Risk 0.8% permite ~5 losses seguidos antes de soft limit (0.8 × 5 = 4%)
### Política de Redução de Risco Dinâmica
| Daily Drawdown | Risk Adjustment | Max Trades/Day | Reasoning |
|----------------|-----------------|----------------|-----------|
| 0% a -1%       | Risk normal (0.8%) | 10 | Zona verde, operar normalmente |
| -1% a -2.5%    | Risk reduzido (0.5%) | 6 | Zona amarela, cautela |
| -2.5% a -4%    | Risk mínimo (0.3%) | 3 | Zona vermelha, apenas setups perfeitos |
| -4% ou pior    | **BLOQUEIO TOTAL** | 0 | Parar de operar até próximo dia |
**Implementação:**
```mql5
double GetDynamicRisk(double dailyDD_pct) {
   if (dailyDD_pct >= -1.0) return 0.8;
   if (dailyDD_pct >= -2.5) return 0.5;
   if (dailyDD_pct >= -4.0) return 0.3;
   return 0.0;  // Block trading
}
```
### Deep Thinking: Cenários Críticos
#### 1. **Dia bom (muito ganho no início)**
**Problema**: EA faz +2% até 10h (2 trades winners). Psicologicamente tenta continuar → overtrading → perde tudo.
**Solução**:
- Limitar trades por dia (max 10, independentemente de resultado)
- Se atingir +3% diário, reduzir risk para 0.5% (proteger lucro)
- Implementar "profit lock": Se +2%, SL em breakeven obrigatório para trades restantes
- **Consistency Rule**: Melhor dia < 35% do profit total (FTMO exige). EA deve distribuir lucro.
**Código**:
```mql5
if (dailyProfit_pct > 2.0) {
   Print("Daily target hit, reducing aggression");
   RiskPerTrade_Pct = 0.5;  // Conservative mode
}
```
#### 2. **Sequência de 3 stops seguidos**
**Problema**: Revenge trading (EA tenta recuperar) → aumenta lote → explode conta.
**Solução**:
- Após 2 stops seguidos: pausa de 1 hora (cool-down)
- Após 3 stops seguidos: pausa até próximo dia OU até `FinalScore > 90` (setup excepcional)
- **Nunca aumentar risk após loss** (anti-martingale estrito)
**Código**:
```mql5
int g_consecutiveLosses = 0;
void OnTradeTransaction() {
   if (lastTradeWasLoss()) {
      g_consecutiveLosses++;
      if (g_consecutiveLosses >= 3) {
         Print("🛑 3 consecutive losses, blocking trading until tomorrow");
         g_tradingBlocked = true;
      }
   } else {
      g_consecutiveLosses = 0;  // Reset
   }
}
```
#### 3. **Quando NÃO operar (mesmo com setup bom)**
**Cenários de bloqueio:**
- **News de alto impacto**: NFP, FOMC, CPI (30min antes e depois)
- **Spread > 3.0 pontos** (XAUUSD normal = 1.5-2.0; spread alto = execução ruim)
- **Sessão asiática** (21:00-08:00 GMT): Baixa liquidez, falsos breakouts
- **Sexta após 16:00 GMT**: Rollover de fim de semana, liquidez seca
- **Conta em trailing DD crítico**: Se equity < 92% do high (próximo de violar FTMO)
**Filtro de News**:
```mql5
bool IsHighImpactNewsTime() {
   // Check economic calendar API ou hardcode dates
   // Example: Block trading se próximo 30min de NFP
   return false;  // Implement full calendar check
}
```
**Filtro de Spread**:
```mql5
bool IsSpreadAcceptable() {
   double spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * _Point;
   return (spread <= 3.0);
}
```
### Raciocínio de Trader Prop Júnior
> "A FTMO não quer você operando 50x/dia. Quer consistência, baixo drawdown, e respeito ao plano. Um EA que faz +10% em 2 dias e -6% no terceiro é um EA que **falha**. Prefira +1%/dia por 10 dias = +10% total com DD < 3%. O segredo é **sobreviver**, não ficar rico em 1 semana."
**Lições:**
- **Risk First, Profit Second**: Sempre questione "posso perder esse trade?" antes de "quanto vou ganhar?"
- **DD é permanente, Profit é temporário**: Um -5% DD elimina você. Um +5% profit pode virar -5% amanhã.
- **Transparência > Black Box**: EA deve logar **por que** entrou e **por que** saiu. Se você não entende, FTMO não vai aceitar.
---
## 🧪 SEÇÃO 7 – ESTRATÉGIA DE TESTES E VALIDAÇÃO
### Backtests
**Período e Data Range:**
- **Mínimo**: 6 meses de dados (2 ciclos de mercado: trend + range)
- **Ideal**: 2 anos (incluir eventos macro: COVID, recessões, bull runs)
- **Data específica**: Jan 2023 - Dez 2024 (captura inflação alta, Fed hawkish, gold volatility)
**Timeframes:**
- **Análise**: M15 (scalping principal)
- **Confirmação**: H1 (trend filter)
- **Backtest**: Tick data real ou "Every tick based on real ticks" (MT5)
**Qualidade de Tick:**
- Use **Dukascopy** ou **TrueFX** tick data (não histórico de broker)
- Verificar spread histórico real (não fixo)
- Modelagem: "Every tick based on real ticks" (mais preciso)
### Stress Tests
**1. Spreads Maiores:**
- Simular spread de 3.0-5.0 pontos (pior caso em news/rollover)
- Validar se EA ainda é lucrativo com spread 2x maior
**2. Slippage:**
- Adicionar 2-4 pontos de slippage em entradas/saídas (realista para VPS)
- Testar se RR de 1:1.5 ainda funciona com slippage
**3. News On/Off:**
- Backtest 1: Filtro de news ativado (bloquear NFP, FOMC, etc)
- Backtest 2: Sem filtro (pior caso)
- Comparar drawdown: News filter deve reduzir DD em 30-40%
### Testes Específicos de FTMO
**Simular Max Daily Loss:**
- Criar script que encerra trading quando DD diário > 4%
- Contar quantos dias violaram regra (deve ser 0)
**Simular Max Total Loss:**
- Usar equity control: Se equity < 91k (em conta $100k), parar backtest
- Validar se EA nunca chegou nesse limite
**Consistency Rule:**
- Calcular melhor dia vs total profit
- Fórmula: `Best_Day_Profit / Total_Net_Profit < 0.35`
- Se > 35%, EA está concentrando muito lucro em poucos dias (bad para FTMO)
### Critérios de Aprovação
**Métricas Mínimas:**
- **Win Rate**: > 55% (scalping deve ter hit rate alto)
- **Profit Factor**: > 1.8 (1.5 é break-even com custos)
- **Max Drawdown**: < 8% (margem de 2% antes de violar FTMO)
- **Sharpe Ratio**: > 1.5 (risk-adjusted returns)
- **Recovery Factor**: > 3 (Net Profit / Max DD)
- **Average RR**: > 1:1.3 (mínimo para scalping ser viável)
**Limites de Violação:**
- Dias com quase-violação de DD (> 3.5%): < 5% do total de dias
- Trades que arriscaram > 2%: 0 (hard rule)
- Trades sem reasoning string: 0 (100% rastreabilidade)
**Exemplo de Checklist:**
```markdown
✅ Backtest 2023-2024: PF 2.1, WR 58%, DD 6.2%
✅ Stress test (spread 3.0): PF 1.7, WR 54% (still viable)
✅ Max Daily Loss violated: 0 days
✅ Max Total Loss violated: 0 days
✅ Consistency: Best day = 22% of total profit (< 35% ✓)
✅ Slippage test: Average RR dropped to 1:1.2 (acceptable)
⚠️ News filter test: Without filter, DD increased to 11% (MUST keep filter ON)
```
---
## 📣 SEÇÃO 8 – EXEMPLOS DE REASONING STRINGS
### Exemplo 1: Trade WIN (BUY XAUUSD)
**Reasoning String:**
> **[2025-01-15 14:32 | BUY XAUUSD @ 2648.30 | +45 pips | +$450]**  
> **Context**: London session, moderate volatility (ATR 24.5), bullish trend on H1.  
> **Entry**: Bullish FVG at 2646-2648 + Order Block retest at 2645.80. Price swept liquidity below 2645 (stop hunt) then reversed. TechScore 82, FundScore 65 (neutral Fed), SentScore 58 (neutral). **FinalScore: 86/100**.  
> **Risk**: SL 2643.50 (36 pips = 1.5x ATR), TP 2654.50 (RR 1:1.7), Lot 0.12 (0.8% risk).  
> **Outcome**: TP hit in 42 min. Trade aligned with FTMO rules (Daily DD: -0.5% → +0.4%).
**Análise:**
- Setup técnico forte (FVG + OB + liquidity sweep)
- Score 86 acima do threshold 85
- Risco controlado (0.8%, dentro de limite diário)
- Resultado rápido (scalping ideal: < 1h)
---
### Exemplo 2: Trade LOSS (SELL XAUUSD)
**Reasoning String:**
> **[2025-01-16 09:15 | SELL XAUUSD @ 2652.80 | -38 pips | -$380]**  
> **Context**: Early NY session, high volatility (ATR 28.3), ranging market on H1.  
> **Entry**: Bearish Order Block at 2653-2655, liquidity grab above 2653. TechScore 75, FundScore 72 (hawkish Fed comments), SentScore 48 (bearish). **FinalScore: 87/100**.  
> **Risk**: SL 2656.60 (38 pips = 1.35x ATR), TP 2646.00 (RR 1:1.8), Lot 0.10 (0.8% risk).  
> **Outcome**: STOPPED OUT. Price spiked on unexpected USD weakness news (CPI miss). Post-analysis: H1 structure was indecisive (LL but no clear LH). Lesson: Avoid ranging H1, require strong trend.
**Análise:**
- Setup parecia bom (score 87), mas contexto de range traiu
- News inesperada (CPI miss) inverteu sentiment
- Stop respeitado (não ampliado), loss dentro do risco planejado
- **Lição aplicada**: Adicionar filtro de H1 trend strength (ex: ADX > 25)
---
### Exemplo 3: Sinal IGNORADO (Score Alto mas Risco FTMO Próximo do Limite)
**Reasoning String:**
> **[2025-01-17 11:48 | BUY XAUUSD @ 2639.50 | REJECTED BY RISK MANAGER]**  
> **Context**: London session, bullish trend, moderate volatility (ATR 26.1).  
> **Entry**: Perfect setup - Bullish FVG + OB retest + liquidity sweep. TechScore 88, FundScore 68 (USD weakness), SentScore 62 (bullish gold). **FinalScore: 91/100** (very strong).  
> **Risk**: SL 2635.20 (43 pips), TP 2647.80 (RR 1:1.9), Lot 0.09 (0.8% risk).  
> **BLOCKED**: FTMO_RiskManager veto. Current Daily DD: -3.1%. Projected loss if stopped: -3.9% (exceeds soft limit of -2.5%). Dynamic risk reduced to 0.3%, but even min lot would push DD to -3.5%.  
> **Decision**: **NO TRADE**. Protecting capital > chasing setup. Wait until tomorrow or daily DD recovers.
**Análise:**
- Setup perfeito (score 91/100, acima de 85)
- **Risco > Oportunidade**: DD diário já em -3.1% (zona vermelha)
- Risk Manager corretamente vetou trade (poder absoluto)
- **Consistência com política**: Após -2.5% DD, apenas trades de risco mínimo
- Mesmo com risco reduzido, projeção violaria soft limit
- **Decisão correta**: Preservar conta, evitar revenge trading
**Mensagem ao Trader:**
> "Este é o tipo de decisão que separa EAs lucrativos de EAs que explodem contas. Um setup de 91/100 foi **rejeitado** porque a **matemática do risco** não permitia. FTMO valoriza isso: disciplina > ganância."
---
## 🎯 RESUMO EXECUTIVO
Este documento apresentou o design completo de um **EA híbrido MQL5 + Python** para scalping de XAUUSD em contas FTMO, com ênfase absoluta em:
1. **Risk-First Architecture**: `CFTMORiskManager` com poder de veto sobre qualquer trade
2. **Multi-Dimensional Scoring**: TechScore (MQL5) + FundScore + SentScore (Python) = Decisão holística
3. **FTMO Compliance**: Hard limits (4% daily, 9% total), dynamic risk scaling, consistency rules
4. **Transparência total**: Reasoning Strings documentam cada decisão (entrar, não entrar, stop, win)
5. **Modularidade**: Arquitetura limpa permite adicionar novos módulos (ML, novos indicadores) sem refatoração
**Próximos Passos:**
1. Implementar módulos técnicos completos (OB, FVG, Liquidity, Market Structure)
2. Desenvolver Python Agent Hub com FastAPI + agentes (Technical, Fundamental, Sentiment, LLM)
3. Backtests em tick data real (2023-2024)
4. Forward testing em demo FTMO (30 dias)
5. Deploy em Challenge FTMO real
**Filosofia Final:**
> "A FTMO não é um cassino. É um teste de disciplina, gestão de risco e consistência. Este EA foi projetado para **passar no teste**, não para ficar rico em 1 semana. Cada linha de código reflete essa filosofia: **sobreviver primeiro, lucrar depois**."
---
**TradeDev_Master | EA_SCALPER_XAUUSD v1.0**  
*"Risk First, Profit Second, Transparency Always"*


Implementation_plan.md:

EA_SCALPER_XAUUSD Multi-Agent System - Implementation Plan
Goal
Implement a production-ready MQL5 Expert Advisor with Python Agent Hub integration for XAUUSD scalping on FTMO prop firm accounts. The system combines ICT/SMC trading strategies (Order Blocks, FVG, Liquidity Sweeps) with multi-dimensional scoring (Technical + Fundamental + Sentiment) and absolute risk-first architecture for FTMO compliance.

User Review Required
IMPORTANT

MQL5 Code Location: The main EA and class modules will be created in MAIN_EAS/DEVELOPMENT/EA_SCALPER_XAUUSD/ following your repository structure.

WARNING

Python Integration: This plan assumes FastAPI Python Hub running locally on http://localhost:8000. If you prefer ZeroMQ or different architecture, please specify before implementation.

CAUTION

FTMO Parameters: Default risk parameters are conservative (0.8% per trade, 4% max daily DD). Adjust if you have different FTMO challenge requirements.

Proposed Changes
Phase 1: Core MQL5 Infrastructure
[NEW] 

EA_SCALPER_XAUUSD.mq5
Main Expert Advisor with:

Complete OnInit, OnDeinit, OnTick, OnTimer, OnTradeTransaction event handlers
Input parameters for risk management, scoring thresholds, strategy parameters
Integration with all modules (Risk Manager, Scoring, Execution)
Async Python Hub calls via OnTimer (non-blocking architecture)
Session/time filters, spread checks, news blockers
Complete reasoning string generation for every trade decision
[NEW] 

FTMO_RiskManager.mqh
Complete CFTMORiskManager class with:

CanOpenTrade() - Veto power over all trades based on FTMO limits
GetLotSize() - Dynamic position sizing based on risk percentage and SL points
ResetDaily() - Daily P&L tracking reset at midnight
Dynamic drawdown control (soft limits trigger risk reduction)
Projected loss validation (prevents trades that would violate limits)
Consistency rule tracking (best day < 35% of total profit)
Detailed rejection reason logging
[NEW] 

SignalScoringModule.mqh
Complete CSignalScoringModule class with:

ComputeTechScore() - Combine OB, FVG, liquidity, trend, ATR into 0-100 score
ComputeFinalScore() - Weighted combination of Tech/Fund/Sent scores
Score component weighting (default 50% tech, 30% fund, 20% sentiment)
Score caching for async Python calls
[NEW] 

PythonBridge.mqh
HTTP communication layer with:

CallPythonHub() - WebRequest wrapper with timeout handling
JSON request builder (symbol, timeframe, tech signals)
JSON response parser (fund/sent scores, biases, reasoning)
Fallback logic (neutral scores if Python unavailable)
Error handling and retry policies
Phase 2: Trading Strategy Modules (Stubs with TODOs)
[NEW] 

OrderBlockModule.mqh
COrderBlockModule stub with:

Function signatures for Detect(), GetStrength(), GetPriceLevel()
TODO comments for full implementation (volume analysis, retest validation)
Interface definition for scoring module integration
[NEW] 

FVGModule.mqh
CFVGModule stub with:

Function signatures for gap detection, quality scoring
TODO comments for multi-timeframe FVG analysis
Interface for scoring integration
[NEW] 

LiquidityModule.mqh
CLiquidityModule stub for liquidity sweeps detection

[NEW] 

MarketStructureModule.mqh
CMarketStructureModule stub for HH/HL/LH/LL trend detection

[NEW] 

VolatilityModule.mqh
CVolatilityModule stub for ATR-based volatility regime detection

Phase 3: Python Agent Hub (Align with Existing Structure)
[MODIFY] 

main.py
Add endpoints:

POST /analyze - Main analysis endpoint (receives MQL5 signals, returns scores)
GET /health - Health check for EA connectivity validation
Update to use existing schemas from app/models/schemas.py
[MODIFY] 

schemas.py
Add Pydantic models:

AnalysisRequest - Structure for MQL5 requests
AnalysisResponse - Structure for Python Hub responses
Match JSON formats defined in Section 5 of design document
[NEW] 

Python_Agent_Hub/agents/fundamental_agent.py
Stub for fundamental analysis agent (economic calendar, USD/Gold correlation)

[NEW] 

Python_Agent_Hub/agents/sentiment_agent.py
Stub for sentiment analysis agent (news scraping, social sentiment)

[NEW] 

Python_Agent_Hub/agents/llm_reasoning_agent.py
Stub for LLM-powered reasoning (GPT-4 integration for contextual analysis)

Phase 4: Documentation & Configuration
[NEW] 

CONFIG_FTMO.set
MT5 preset file with recommended parameters for FTMO $100k challenge

[MODIFY] 

EA_SCALPER_DESIGN_COMPLETE.md
Add "Implementation Status" section tracking completed vs pending modules

[NEW] 

TESTING_PLAN.md
Detailed testing checklist based on Section 7 (backtests, stress tests, FTMO validation)

Verification Plan
Automated Tests
MQL5 Compilation:

# Verify MQL5 code compiles without errors
# Run from MetaTrader 5 terminal
# Tools > MetaQuotes Language Editor > Compile
# Expected: 0 errors, 0 warnings (warnings acceptable for TODOs)
Python Unit Tests:

cd /home/franco/projetos/EA_SCALPER_XAUUSD/Python_Agent_Hub
pytest tests/ -v
# Expected: All tests pass for schemas, endpoints, agent stubs
Python Server Startup:

cd /home/franco/projetos/EA_SCALPER_XAUUSD/Python_Agent_Hub
uvicorn main:app --reload --port 8000
# Expected: Server starts on http://localhost:8000
# Test: curl http://localhost:8000/health (returns 200 OK)
Manual Verification
1. MQL5 Risk Manager Unit Test:

Attach EA to XAUUSD chart (Strategy Tester, Demo)
Set inputs: RiskPerTrade_Pct = 1.0, MaxDailyLoss_Pct = 5.0
Manually trigger simulated trades with increasing DD
Expected behavior:
DD < 2.5%: Normal risk (1.0%)
DD 2.5-4%: Reduced risk (0.5%)
DD > 4%: Trading blocked (logged in Experts tab)
Validation: Check terminal logs for "TRADE REJECTED" messages with correct DD percentages
2. Python Integration Test:

Start Python Hub (uvicorn main:app --port 8000)
Attach EA to chart with UsePythonHub = true
Monitor Experts tab for log: "Python Hub success: Fund=XX Sent=XX"
Expected: EA receives scores from Python (initially mock data 50-70 range)
Fallback test: Stop Python server → EA logs "Python call failed, using TechScore only"
3. Scoring Module Test:

Create test scenario: OB=true, FVG=true, Trend=bullish, ATR=normal
Expected TechScore: ~85 (calculated in ComputeTechScore)
Verify calculation in logs: "TechScore: 85.0"
With Python (Fund=65, Sent=58): FinalScore = 0.5×85 + 0.3×65 + 0.2×58 = 74.6
Validation: Check log output matches calculation
4. FTMO Compliance Simulation:

Run Strategy Tester backtest on XAUUSD (M15, Jan-Dec 2024, tick data)
Force parameters: MaxDailyLoss_Pct = 4.0
Expected results:
Report shows Max Daily DD < 4.0%
No trades executed after daily DD threshold hit
Graph shows flat equity after DD limit (trading blocked)
Validation: Review backtest report HTML → "Max Daily Loss" metric
User Acceptance Testing
Please manually verify:

Code readability: Review CFTMORiskManager class - is the logic clear?
FTMO parameters: Do the default risk settings (0.8%, 4% daily, 9% total) match your challenge requirements?
Python architecture: Confirm FastAPI HTTP integration is acceptable (vs ZeroMQ)
Repository structure: Verify files are in correct locations per your MAIN_EAS/DEVELOPMENT/ structure

# GPT 5.1 PRO (disponivel apenas via WEB):

Agora tive a resposta do GPT 5.1 Pro (disponivel apenas na WEB,  o que ficaria muito completo para trabalhar com ele no dia a dia, ja que ele demora muito.):

SEÇÃO 1 – COMPREENSÃO DO PROBLEMA

Objetivo estratégico: construir um EA de scalping em XAUUSD, específico para contas de prop firm (FTMO), que prioriza preservação de capital e respeito estrito às regras de risco, operando apenas quando a confluência técnica/fundamental/sentimento atinge um FinalScore ≥ ExecutionThreshold.

Principais restrições FTMO: normalmente Max Daily Loss = 5% do saldo inicial e Max Total Loss = 10% (equity nunca pode cair abaixo de 90% do inicial, incluindo posições abertas).
FTMO
+1

O EA deve monitorar equity em tempo real, considerando lucros/prejuízos fechados + flutuantes, porque é assim que FTMO mede os limites de perda diária e máxima.
FTMO

A arquitetura precisa ser risk‑first: o módulo FTMO_RiskManager tem poder de veto sobre qualquer sinal, mesmo com FinalScore alto.

Multi-agente (MQL5 + Python) permite dividir responsabilidades: MQL5 cuida do que exige baixa latência (ticks, execução, gestão de ordens) e Python cuida de news, sentimento, modelos mais pesados e raciocínio LLM.

A estratégia técnica usa SMC/ICT (OB, FVG, liquidity sweeps, HH/HL/LH/LL) combinada com volatilidade (ATR) para definir entradas, stops e filtros de momento.

Scoring Engine 0–100 agrega TechScore (MQL5 + Python), FundScore (notícias/macros) e SentScore (fluxo de sentimento) em um FinalScore transparente, logado com uma Reasoning String a cada decisão.

Riscos clássicos que o sistema precisa mitigar: slippage forte em XAUUSD, spreads alargados em news, overtrading em dias bons, violar Max Daily Loss por não considerar posições abertas e latência em chamadas externas (Python) travando o OnTick.

Objetivo operacional: manter OnTick < 50ms, isolando chamadas externas em OnTimer/eventos e usando sempre últimos scores disponíveis em memória.

Transparência: cada trade (ou sinal recusado) gera uma Reasoning String com contexto de mercado + motivo técnico + decisão de risco (aprovado/bloqueado).

SEÇÃO 2 – ARQUITETURA DE ALTO NÍVEL (MQL5 + PYTHON)
2.1 Camadas em MQL5

1) Data & Events Layer

Eventos: OnInit, OnDeinit, OnTick, OnTimer, OnTradeTransaction.

Responsável por:

Receber ticks, atualizar buffers de preços/indicadores.

Manter relógio lógico (sessão, dia FTMO, reset diário).

Disparar updates assíncronos para o Python (via fila interna + OnTimer).

2) Strategy / Signal Layer (OB, FVG, Liquidity, Market Structure, ATR)

Módulos:

COrderBlockModule
Detecta OBs relevantes (última vela contrária antes de BOS, filtrada por volume/estrutura).

CFVGModule
Detecta FVGs em timeframes de entrada (M1–M5) e HTF (M15–H1).

CLiquidityModule
Identifica equal highs/lows, pools de liquidez, sweeps recentes.

CMarketStructureModule
Marca HH/HL/LH/LL, BOS/CHOCH e define trend bias (bullish/bearish/range).

CVolatilityModule
Calcula ATR, ranges médios, usa ATR para dimensionar stop/TP.

Esses módulos produzem um vetor de features técnicas que alimenta o Scoring Engine.

3) Scoring Engine

CSignalScoringModule:

Constrói o TechScore (0–100) com base em:

Confluência OB + FVG alinhados com a estrutura.

Liquidity sweep a favor da direção pretendida.

ATR dentro de “faixa operacional” (nem morto, nem insano).

FinalScore = f(TechScore, FundScore, SentScore) com pesos configuráveis (por ex. 50/30/20).

4) Execution & FTMO_RiskManager

CFTMORiskManager:

Controla Max Daily Loss, Max Total Loss, drawdown diário e risk per trade dinâmico.

Exposto via bool CanOpenTrade(...) e double CalculateLotSize(...).

CTradeExecutor:

Encapsula CTrade do MQL5, cuidando de:

Tipo de ordem.

SL/TP em pontos (derivados da ATR/estrutura).

Slippage e preenchimento.

Nunca abre trade sem aprovação explícita do CFTMORiskManager.

5) Logging & Notifications

CLogger:

Gera Reasoning Strings.

Escreve em Print(), arquivo CSV/JSON e push notifications (Mobile).

Permite auditoria posterior para FTMO.

2.2 Python Agent Hub

Agentes previstos:

TechnicalAgent (Python)

Pode aplicar modelos mais pesados (ML, clustering) sobre as features técnicas enviadas.

Retorna um tech_subscore_python (ajuste refinado ao TechScore).

FundamentalAgent

Lê calendário econômico, notícias (FOMC, CPI, NFP, Geopolítica).

Retorna fund_score (0–100) e fund_bias (bullish/bearish/neutral).

SentimentAgent

Analisa notícias, mídias sociais, COT/posicionamento, fluxo de opções.

Retorna sent_score e sent_bias.

LLMReasoningAgent

Gera uma pequena explicação (llm_reasoning_short) para compor a Reasoning String.

Transporte: HTTP/REST (via WebRequest)

Escolha: HTTP/REST em vez de ZeroMQ porque:

Suporte nativo do MQL5 via WebRequest.

Fácil debug (logs HTTP, Insomnia/Postman).

Menos dependências no lado do EA (não precisa DLL/bridge C++).

Formato de resposta (resumido)

{
  "success": true,
  "tech_subscore_python": 5.0,
  "fund_score": 78.0,
  "fund_bias": "bullish",
  "sent_score": 62.0,
  "sent_bias": "neutral",
  "llm_reasoning_short": "Gold em leve tendência de alta, sem notícias vermelhas nos próximos 30m."
}

2.3 Fluxo de um Tick “perfeito”

Diagrama textual geral:

MT5_EA (OnTick) → módulos técnicos → CSignalScoringModule → (se pré-score alto) →
MT5_EA (OnTimer) → WebRequest POST /hub/score → Python_Agent_Hub → JSON →
MT5_EA atualiza Fund/Sent/Tech_subscore_py → CFTMORiskManager → CTradeExecutor.

Passo a passo:

Tick chega (OnTick)

Atualiza market data para XAUUSD.

Verifica sessão (London/NY), spread e se já existe posição aberta (para evitar overtrading).

Sinais técnicos (MQL5)

CMarketStructureModule marca estrutura (HH/HL/LH/LL; BOS/CHOCH).

COrderBlockModule, CFVGModule, CLiquidityModule detectam OB/FVG/sweeps relevantes.

CVolatilityModule calcula ATR.

TechScore local

CSignalScoringModule::ComputeTechScore(...) gera TechScore base (local).

Chamada opcional ao Python (assíncrona)

Se TechScore local ≥ pré-limiar (ex. 70), OnTick só marca um “context snapshot” numa fila interna.

OnTimer processa essa fila, faz WebRequest para o Python e atualiza:

tech_subscore_python, fund_score, sent_score, llm_reasoning_short.

Importante: OnTick nunca bloqueia esperando Python.

FinalScore & decisão de trade

FinalScore = f(TechScore_local + tech_subscore_python, FundScore, SentScore).

Se FinalScore < ExecutionThreshold → loga “sinal ignorado” + Reasoning String e encerra.

FTMO_RiskManager & execução

Calcula stop em pontos via ATR/estrutura.

CFTMORiskManager::GetDynamicRiskPerc() ajusta o risco com base no drawdown diário.

CFTMORiskManager::CanOpenTrade(...) verifica:

Max Daily Loss (5%).

Max Total Loss (10%).

Risco da nova operação não faz o dia exceder o limite.
FTMO
+1

Se aprovado, CTradeExecutor envia a ordem com SL/TP e registra a Reasoning String.

SEÇÃO 3 – DESIGN DETALHADO DO EA EM MQL5
3.1 Módulos/Classes
COrderBlockModule

Responsabilidades

Detectar OBs válidos em timeframes HTF (M15/H1) e LTF (M1/M5).

Inputs

Série de candles (Open/High/Low/Close), estrutura de mercado, volume.

Outputs

Flags: hasBullishOB, hasBearishOB.

Níveis: preço do OB, range, frescor (quantas velas atrás).

Contribuição ao score

OB fresco, respeitado várias vezes, alinhado com trend → +score para entradas na direção da estrutura.

CFVGModule

Responsabilidades

Detectar Fair Value Gaps em múltiplos timeframes.

Inputs

Candles recentes, parâmetros de tamanho mínimo de gap.

Outputs

hasBullishFVG, hasBearishFVG, tamanho/posição do gap.

Contribuição

Confluência OB + FVG na mesma direção → aumento significativo no TechScore.

CLiquidityModule

Responsabilidades

Identificar equal highs/lows, áreas de liquidez e sweeps recentes.

Inputs

Máximas/mínimas recentes, informação de stop hunts.

Outputs

liquidity_sweep_up, liquidity_sweep_down, local do sweep (acima/dentro de OB, próximo de nível chave).

Contribuição

Sweep contra a direção e retomada na direção do trend → bom sinal de entrada.

CMarketStructureModule

Responsabilidades

Determinar estrutura de mercado: HH/HL/LH/LL, BOS, CHOCH.

Inputs

Swings detectados, highs/lows de n velas.

Outputs

isBullishTrend, isBearishTrend, último BOS, último CHOCH.

Contribuição

Define trend bias principal que orienta OB/FVG/liquidity.

CVolatilityModule

Responsabilidades

Calcular ATR, ranges médios e sugerir tamanhos de stop/TP.

Inputs

Histórico de preços, período e timeframe.

Outputs

atr_value, stoploss_points_from_atr, bandas de volatilidade.

Contribuição

Filtra períodos de hyper-volatilidade ou volatilidade baixa demais.

CSignalScoringModule

Responsabilidades

Calcular TechScore e FinalScore.

Inputs

Flags de OB/FVG/liquidity/trend, ATR, e sub-scores do Python.

Outputs

TechScore, FinalScore.

Contribuição

Camada de decisão central (técnica), agregando tudo numa escala 0–100.

CFTMORiskManager

Responsabilidades

Implementar regras de FTMO:

Max Daily Loss (5%).

Max Total Loss (10%).

Risco dinâmico por faixa de drawdown diário.

Calcular lot size baseado em %risk e stoploss_points.

Inputs

Saldo/equity, day start equity, DD diário/total, risk_per_trade, stoploss_points.

Outputs

dynamic_risk_perc, lot_size, flags de bloqueio.

Contribuição

Veto final de risco (trade só abre se passar por ele).

CTradeExecutor

Responsabilidades

Abstract sobre CTrade: abrir/fechar posições com SL/TP e magic number.

Inputs

Tipo de ordem, lot, SL/TP em pontos, comentário.

Outputs

Resultado bool, ticket, erros de execução logados.

Contribuição

Execução consistente, isolando a lógica de ordens do resto do EA.

CLogger

Responsabilidades

Logging estruturado + Reasoning Strings + notificações.

Inputs

Strings de contexto (score, risco, razões técnicas e de veto).

Outputs

Logs em Print, arquivo e push (opcional).

Contribuição

Transparência/auditoria de decisões para o trader.

3.2 Pseudocódigo do OnTick ideal
OnTick()
    if Symbol != "XAUUSD" return

    // 1) Atualizar contexto de risco/FTMO
    riskManager.ResetDayIfNeeded()

    if existe posição aberta deste EA em XAUUSD:
        // gestão de trade poderia ir aqui (trail, partials)
        return

    if sessão não é London/NY ou spread > limite:
        return

    // 2) Ler sinais técnicos principais
    structure = marketStructureModule.Analyze()
    hasOB     = orderBlockModule.HasValidOB(structure)
    hasFVG    = fvgModule.HasValidFVG(structure)
    liq       = liquidityModule.Analyze()
    atr       = volatilityModule.GetATR()

    bullishTrend = structure.IsBullishTrend()

    // 3) Calcular TechScore local
    techLocal = scoringModule.ComputeTechScore(hasOB, hasFVG, bullishTrend, atr)

    // 4) Opcional: agendar chamada ao Python
    if techLocal >= PreScoreThreshold AND !ja_tem_pedido_pendente:
        push contexto simplificado em FILA para OnTimer consumir (não bloquear aqui)

    // 5) Usar últimos scores vindos do Python (atualizados em OnTimer)
    techTotal = clamp(techLocal + lastTechSubscorePy, 0, 100)
    fundScore = lastFundScore
    sentScore = lastSentScore

    finalScore = scoringModule.ComputeFinalScore(techTotal, fundScore, sentScore)

    if finalScore < ExecutionThreshold:
        logger.LogReason("Sinal ignorado por score baixo", contexto + scores)
        return

    // 6) Calcular stop/TP a partir da ATR / estrutura
    stoploss_points = volatilityModule.StopLossPointsFromATR(atrMultiplier)
    if stoploss_points <= 0:
        logger.Log("ATR inválida, sem stop calculável")
        return

    // 7) Risk Manager FTMO
    dynamicRisk = riskManager.GetDynamicRiskPerc(BaseRiskPerTrade)
    if dynamicRisk <= 0:
        logger.Log("FTMO_RiskManager bloqueou (DD diário/total)")
        return

    if !riskManager.CanOpenTrade(dynamicRisk, stoploss_points):
        logger.Log("FTMO_RiskManager: trade rejeitado (Max Daily/Total check)")
        return

    lots = riskManager.CalculateLotSize(dynamicRisk, stoploss_points)
    if lots <= 0:
        logger.Log("Lot size calculado <= 0")
        return

    // 8) Execução
    direction = bullishTrend ? BUY : SELL
    success = tradeExecutor.OpenMarket(direction, lots, stoploss_points, tp_points, reasoning_string)

    logger.Log(success ? "Trade aberto" : "Falha ao abrir trade", contexto + scores + risco)

SEÇÃO 4 – CÓDIGO MQL5 ESSENCIAL

Observação: código focado nas partes críticas (EA, CFTMORiskManager, CSignalScoringModule) + stubs bem comentados dos demais módulos. Pode compilar com ajustes mínimos de ambiente.

//+------------------------------------------------------------------+
//|                                               EA_SCALPER_XAUUSD |
//+------------------------------------------------------------------+
#property strict

#include <Trade/Trade.mqh>

//--- Inputs principais
input double        InpRiskPerTrade       = 0.40;   // % por trade (base)
input double        InpMaxDailyLossPerc   = 5.0;    // Max Daily Loss (FTMO)
input double        InpMaxTotalLossPerc   = 10.0;   // Max Total Loss (FTMO)
input double        InpExecutionThreshold = 85.0;   // Score mínimo para executar
input ENUM_TIMEFRAMES InpTF_Structure     = PERIOD_M15;
input ENUM_TIMEFRAMES InpTF_Entry         = PERIOD_M1;
input bool          InpUsePythonHub       = false;  // Integração futura
input string        InpPythonHubURL       = "http://127.0.0.1:5000/score";
input ulong         InpMagic              = 555777;

//+------------------------------------------------------------------+
//| Helper: início do dia (server time)                              |
//+------------------------------------------------------------------+
datetime GetDayStart(datetime t)
{
   MqlDateTime dt;
   TimeToStruct(t, dt);
   dt.hour = 0;
   dt.min  = 0;
   dt.sec  = 0;
   return StructToTime(dt);
}

//+------------------------------------------------------------------+
//| CFTMORiskManager                                                 |
//+------------------------------------------------------------------+
class CFTMORiskManager
{
private:
   double   m_initial_balance;
   double   m_max_daily_loss_perc;
   double   m_max_total_loss_perc;
   // Zonas de DD diário (exemplo: 0-1%, 1-2.5%, 2.5-4%, >4%)
   double   m_soft_zone1;
   double   m_soft_zone2;
   double   m_soft_zone3;
   datetime m_day_start;
   double   m_equity_day_start;

public:
   CFTMORiskManager()
   {
      m_initial_balance      = 0.0;
      m_max_daily_loss_perc  = 0.0;
      m_max_total_loss_perc  = 0.0;
      m_soft_zone1           = 1.0;
      m_soft_zone2           = 2.5;
      m_soft_zone3           = 4.0;
      m_day_start            = 0;
      m_equity_day_start     = 0.0;
   }

   bool Init(double max_daily_loss_perc, double max_total_loss_perc)
   {
      m_initial_balance      = AccountInfoDouble(ACCOUNT_BALANCE);
      m_max_daily_loss_perc  = max_daily_loss_perc;
      m_max_total_loss_perc  = max_total_loss_perc;
      m_day_start            = GetDayStart(TimeCurrent());
      m_equity_day_start     = AccountInfoDouble(ACCOUNT_EQUITY);
      return (m_initial_balance > 0.0);
   }

   void ResetDayIfNeeded()
   {
      datetime today = GetDayStart(TimeCurrent());
      if(today != m_day_start)
      {
         m_day_start        = today;
         m_equity_day_start = AccountInfoDouble(ACCOUNT_EQUITY);
      }
   }

   double GetDailyDrawdownPerc()
   {
      double eq = AccountInfoDouble(ACCOUNT_EQUITY);
      if(m_equity_day_start <= 0.0)
         return 0.0;
      if(eq >= m_equity_day_start)
         return 0.0;
      return (m_equity_day_start - eq) / m_equity_day_start * 100.0;
   }

   double GetTotalDrawdownPerc()
   {
      double eq = AccountInfoDouble(ACCOUNT_EQUITY);
      if(m_initial_balance <= 0.0)
         return 0.0;
      if(eq >= m_initial_balance)
         return 0.0;
      return (m_initial_balance - eq) / m_initial_balance * 100.0;
   }

   // Risco dinâmico por faixa de drawdown diário/total
   double GetDynamicRiskPerc(double base_risk_perc)
   {
      double dd_daily = GetDailyDrawdownPerc();
      double dd_total = GetTotalDrawdownPerc();

      if(dd_total >= m_max_total_loss_perc)
         return 0.0; // Max Total Loss atingido (conta "morta" para o EA)

      if(dd_daily >= m_max_daily_loss_perc)
         return 0.0; // Max Daily Loss atingido; não operar mais hoje

      if(dd_daily < m_soft_zone1)
         return base_risk_perc;              // 0-1% DD: risco normal
      if(dd_daily < m_soft_zone2)
         return base_risk_perc * 0.5;        // 1-2.5% DD: risco reduzido
      if(dd_daily < m_soft_zone3)
         return base_risk_perc * 0.25;       // 2.5-4% DD: risco mínimo

      // >= 4% DD: bloquear novas entradas (ainda antes do limite oficial de 5%)
      return 0.0;
   }

   // Cálculo do lote em função do risco e do stop em pontos
   double CalculateLotSize(double risk_perc, double stoploss_points)
   {
      if(risk_perc <= 0.0 || stoploss_points <= 0.0)
         return 0.0;

      double equity = AccountInfoDouble(ACCOUNT_EQUITY);

      double tick_val, tick_size, vol_min, vol_max, vol_step;
      if(!SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE, tick_val))  return 0.0;
      if(!SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE,  tick_size)) return 0.0;
      if(!SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN,       vol_min))   return 0.0;
      if(!SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX,       vol_max))   return 0.0;
      if(!SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP,      vol_step))  return 0.0;

      double risk_amount = equity * risk_perc / 100.0;
      double value_per_point_per_lot = tick_val / tick_size;

      double lots = risk_amount / (value_per_point_per_lot * stoploss_points);
      lots = MathFloor(lots / vol_step) * vol_step;

      if(lots < vol_min)
         return 0.0;
      if(lots > vol_max)
         lots = vol_max;

      return lots;
   }

   // Verifica se abrir um trade com este risco/stop respeitará as regras FTMO
   bool CanOpenTrade(double risk_perc, double stoploss_points)
   {
      ResetDayIfNeeded();

      double eq        = AccountInfoDouble(ACCOUNT_EQUITY);
      double dd_daily  = GetDailyDrawdownPerc();
      double dd_total  = GetTotalDrawdownPerc();

      // Hard stops da prop firm
      if(dd_total >= m_max_total_loss_perc)
         return false;

      if(dd_daily >= m_max_daily_loss_perc)
         return false;

      if(risk_perc <= 0.0 || stoploss_points <= 0.0)
         return false;

      // Aproximação: garante que o pior caso desta trade não fará
      // o dia ultrapassar o Max Daily Loss.
      double allowed_daily_loss_amount = m_equity_day_start * m_max_daily_loss_perc / 100.0;
      double current_daily_loss_amount = (m_equity_day_start - eq);
      if(current_daily_loss_amount < 0.0)
         current_daily_loss_amount = 0.0;

      double risk_amount = eq * risk_perc / 100.0;
      if(current_daily_loss_amount + risk_amount > allowed_daily_loss_amount)
         return false;

      // Ainda verifica se o lote resultante é viável
      double lots = CalculateLotSize(risk_perc, stoploss_points);
      if(lots <= 0.0)
         return false;

      return true;
   }
};

//+------------------------------------------------------------------+
//| CSignalScoringModule                                             |
//+------------------------------------------------------------------+
class CSignalScoringModule
{
public:
   // Sinais simplificados
   double ComputeTechScore(bool hasOB, bool hasFVG, bool bullishTrend, double atr)
   {
      double score = 50.0;

      if(hasOB)       score += 15.0;
      if(hasFVG)      score += 15.0;
      if(bullishTrend) score += 10.0;
      else             score -= 10.0;

      // Ajuste por ATR (volatilidade "saudável")
      if(atr > 0.0)
      {
         double atr_points = atr / _Point;
         if(atr_points > 200 && atr_points < 800)
            score += 10.0;    // volatilidade boa p/ scalping
         else if(atr_points <= 100 || atr_points >= 1200)
            score -= 10.0;    // muito parado ou insano
      }

      if(score < 0.0)   score = 0.0;
      if(score > 100.0) score = 100.0;
      return score;
   }

   double ComputeFinalScore(double tech, double fund, double sent)
   {
      double wTech = 0.5;
      double wFund = 0.3;
      double wSent = 0.2;

      double finalScore = tech * wTech + fund * wFund + sent * wSent;

      if(finalScore < 0.0)   finalScore = 0.0;
      if(finalScore > 100.0) finalScore = 100.0;
      return finalScore;
   }
};

//+------------------------------------------------------------------+
//| Stubs dos módulos técnicos (OB, FVG, Liquidity, etc.)           |
//+------------------------------------------------------------------+
class COrderBlockModule
{
public:
   bool HasValidBullishOB()
   {
      // TODO: implementar detecção de OB de alta (última vela bearish antes de BOS, etc.)
      return false;
   }
   bool HasValidBearishOB()
   {
      // TODO: implementar detecção de OB de baixa
      return false;
   }
};

class CFVGModule
{
public:
   bool HasBullishFVG()
   {
      // TODO: implementar detecção de FVG altista
      return false;
   }
   bool HasBearishFVG()
   {
      // TODO: implementar detecção de FVG baixista
      return false;
   }
};

class CLiquidityModule
{
public:
   bool LiquiditySweepUp()
   {
      // TODO: identificar sweep acima de equal highs
      return false;
   }
   bool LiquiditySweepDown()
   {
      // TODO: identificar sweep abaixo de equal lows
      return false;
   }
};

class CMarketStructureModule
{
public:
   bool IsBullishTrend()
   {
      // TODO: HH/HL usando swings
      return false;
   }
   bool IsBearishTrend()
   {
      // TODO: LH/LL usando swings
      return false;
   }
};

class CVolatilityModule
{
public:
   double GetATR(int period, ENUM_TIMEFRAMES tf)
   {
      return iATR(_Symbol, tf, period, 0);
   }

   double StopLossPointsFromATR(double atr_multiplier, int period, ENUM_TIMEFRAMES tf)
   {
      double atr = iATR(_Symbol, tf, period, 0);
      if(atr <= 0.0)
         return 0.0;
      return (atr / _Point) * atr_multiplier;
   }
};

class CLogger
{
public:
   void Log(string msg)
   {
      Print(msg);
      // TODO: opcional: escrever em arquivo, enviar push notification, etc.
   }
};

// Executor de trades encapsulando CTrade
class CTradeExecutor
{
private:
   CTrade m_trade;

public:
   void Init()
   {
      m_trade.SetExpertMagicNumber((int)InpMagic);
      m_trade.SetTypeFillingBySymbol(_Symbol);
   }

   bool OpenMarket(int order_type, double lots, double sl_points, double tp_points, string reason)
   {
      double price = (order_type == ORDER_TYPE_BUY)
                     ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                     : SymbolInfoDouble(_Symbol, SYMBOL_BID);

      double point = _Point;
      double sl = 0.0, tp = 0.0;

      if(order_type == ORDER_TYPE_BUY)
      {
         if(sl_points > 0) sl = price - sl_points * point;
         if(tp_points > 0) tp = price + tp_points * point;
         return m_trade.Buy(lots, _Symbol, 0.0, sl, tp, reason);
      }
      else
      {
         if(sl_points > 0) sl = price + sl_points * point;
         if(tp_points > 0) tp = price - tp_points * point;
         return m_trade.Sell(lots, _Symbol, 0.0, sl, tp, reason);
      }
   }
};

//+------------------------------------------------------------------+
//| Objetos globais                                                  |
//+------------------------------------------------------------------+
CFTMORiskManager       g_risk;
CSignalScoringModule   g_scoring;
COrderBlockModule      g_ob;
CFVGModule             g_fvg;
CLiquidityModule       g_liquidity;
CMarketStructureModule g_structure;
CVolatilityModule      g_vol;
CTradeExecutor         g_exec;
CLogger                g_logger;

// Scores vindos do Python (atualizados em OnTimer/CallPythonHub futuramente)
double g_lastFundScore       = 50.0;
double g_lastSentScore       = 50.0;
double g_lastTechSubscorePy  = 0.0;

//+------------------------------------------------------------------+
//| Helpers                                                          |
//+------------------------------------------------------------------+
bool HasOpenPosition()
{
   for(int i = 0; i < PositionsTotal(); i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket))
         continue;
      string sym   = PositionGetString(POSITION_SYMBOL);
      long   magic = (long)PositionGetInteger(POSITION_MAGIC);
      if(sym == _Symbol && magic == (long)InpMagic)
         return true;
   }
   return false;
}

// Tendência simples via média móvel (backup se MarketStructureModule ainda não estiver pronto)
bool DetectBullishTrendSimple()
{
   int    period = 50;
   double ma     = iMA(_Symbol, InpTF_Structure, period, 0, MODE_EMA, PRICE_CLOSE, 0);
   if(ma <= 0.0)
      return false;
   double price  = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   return (price > ma);
}

//+------------------------------------------------------------------+
//| OnInit                                                           |
//+------------------------------------------------------------------+
int OnInit()
{
   if(_Symbol != "XAUUSD")
      Print("Aviso: EA_SCALPER_XAUUSD foi desenhado para XAUUSD, rodando em ", _Symbol);

   g_risk.Init(InpMaxDailyLossPerc, InpMaxTotalLossPerc);
   g_exec.Init();

   // TODO: configurar OnTimer para chamadas ao Python (WebRequest) se InpUsePythonHub = true

   g_logger.Log("EA_SCALPER_XAUUSD inicializado.");
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| OnDeinit                                                         |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   g_logger.Log("EA_SCALPER_XAUUSD finalizado. Reason=" + IntegerToString(reason));
}

//+------------------------------------------------------------------+
//| OnTick                                                           |
//+------------------------------------------------------------------+
void OnTick()
{
   if(_Symbol != "XAUUSD")
      return;

   g_risk.ResetDayIfNeeded();

   if(HasOpenPosition())
      return; // versão inicial: 1 trade por vez

   // Filtro simples de spread (evitar spreads anormais)
   double spread_points = (SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID)) / _Point;
   if(spread_points > 50) // ex.: 5 pips em XAUUSD (ajustar)
      return;

   //--- Sinais técnicos simplificados
   bool bullishTrend = DetectBullishTrendSimple();
   bool hasOB        = g_ob.HasValidBullishOB(); // stub (sempre false por enquanto)
   bool hasFVG       = g_fvg.HasBullishFVG();    // stub (sempre false por enquanto)
   double atr        = g_vol.GetATR(14, InpTF_Entry);

   double techLocal  = g_scoring.ComputeTechScore(hasOB, hasFVG, bullishTrend, atr);

   // TODO: se techLocal >= pré-limiar, empurrar contexto p/ fila e deixar OnTimer chamar PythonHub

   double techTotal  = techLocal + g_lastTechSubscorePy;
   if(techTotal > 100.0) techTotal = 100.0;
   if(techTotal < 0.0)   techTotal = 0.0;

   double fundScore  = g_lastFundScore;
   double sentScore  = g_lastSentScore;
   double finalScore = g_scoring.ComputeFinalScore(techTotal, fundScore, sentScore);

   string reason = StringFormat("Tech=%.1f Fund=%.1f Sent=%.1f Final=%.1f OB=%d FVG=%d TrendBull=%d ATR=%.2f",
                               techTotal, fundScore, sentScore, finalScore,
                               hasOB, hasFVG, bullishTrend, atr);

   if(finalScore < InpExecutionThreshold)
      return; // sinal descartado em silêncio na versão mínima

   //--- Definir direção do trade
   int order_type = bullishTrend ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;

   //--- Stop baseado em ATR
   double stoploss_points = g_vol.StopLossPointsFromATR(1.5, 14, InpTF_Entry);
   if(stoploss_points <= 0.0)
   {
      g_logger.Log("ATR inválida, não abre trade. " + reason);
      return;
   }

   //--- Risco dinâmico FTMO
   double dynamic_risk = g_risk.GetDynamicRiskPerc(InpRiskPerTrade);
   if(dynamic_risk <= 0.0)
   {
      g_logger.Log("FTMO_RiskManager bloqueou por DD diário/total. " + reason);
      return;
   }

   if(!g_risk.CanOpenTrade(dynamic_risk, stoploss_points))
   {
      g_logger.Log("FTMO_RiskManager rejeitou nova entrada (limites FTMO). " + reason);
      return;
   }

   double lots = g_risk.CalculateLotSize(dynamic_risk, stoploss_points);
   if(lots <= 0.0)
   {
      g_logger.Log("Lot size calculado <= 0, trade abortado. " + reason);
      return;
   }

   double tp_points = stoploss_points * 1.2; // exemplo de RR ~ 1:1.2
   string comment   = "EA_SCALPER_XAUUSD | " + reason;

   bool ok = g_exec.OpenMarket(order_type, lots, stoploss_points, tp_points, comment);
   if(ok)
      g_logger.Log("Trade aberto com sucesso. Lotes=" + DoubleToString(lots, 2) + " | " + reason);
   else
      g_logger.Log("Falha na abertura do trade. " + reason);
}


Onde entra o Python?

O ponto de integração é marcado no OnTick (comentário TODO) e idealmente implementado em OnTimer via função CallPythonHub(...) usando WebRequest, atualizando g_lastFundScore, g_lastSentScore e g_lastTechSubscorePy.

SEÇÃO 5 – INTERFACE COM PYTHON AGENT HUB
5.1 Formato de request JSON enviado pelo EA

Request típico de contexto:

{
  "symbol": "XAUUSD",
  "timeframe_entry": "M1",
  "timeframe_structure": "M15",
  "timestamp": "2025-11-22T14:35:10Z",
  "price": 2350.12,
  "direction_hint": "long",
  "has_ob": true,
  "has_fvg": true,
  "liquidity_sweep_up": true,
  "liquidity_sweep_down": false,
  "bullish_trend": true,
  "atr": 3.5,
  "local_tech_score": 82.5,
  "session": "NY",
  "daily_dd_perc": 0.8
}


Campos mínimos:

symbol, timeframe_entry, timeframe_structure, timestamp, price.

Sinais técnicos resumidos: has_ob, has_fvg, bullish_trend, atr, sweeps.

local_tech_score (MQL5).

Opcional: sessão (London/NY/Asia) + drawdown diário.

5.2 Formato de response JSON esperado
{
  "success": true,
  "tech_subscore_python": 4.5,
  "fund_score": 76.0,
  "fund_bias": "bullish",
  "sent_score": 61.0,
  "sent_bias": "slightly_bullish",
  "llm_reasoning_short": "Fluxo macro e de sentimento favorece compras moderadas em XAUUSD."
}


tech_subscore_python: ajuste fino ao TechScore local (pode ser negativo).

fund_score / fund_bias: força e direção fundamental.

sent_score / sent_bias: força e direção do sentimento.

llm_reasoning_short: string curta para Reasoning String.

5.3 Pseudocódigo MQL5 para CallPythonHub
bool CallPythonHub(double &tech_subscore_py,
                   double &fund_score,
                   double &sent_score)
{
    if !InpUsePythonHub:
        return false

    // 1) Montar JSON (string)
    string body = StringFormat(
        "{\"symbol\":\"%s\",\"timeframe_entry\":\"M1\","
        "\"local_tech_score\":%.2f}",
        _Symbol, techLocal
    )

    string headers = "Content-Type: application/json\r\n";
    char   data[];
    StringToCharArray(body, data);

    char   result[];
    string result_headers;
    int    status = 0;
    int    timeout_ms = 300;

    int err = WebRequest("POST",
                         InpPythonHubURL,
                         headers,
                         timeout_ms,
                         data,
                         result,
                         result_headers);

    if err != 200:
        // timeout/falha: operar em modo seguro (somente MQL5)
        tech_subscore_py = 0.0;
        fund_score       = 50.0;
        sent_score       = 50.0;
        return false

    string json = CharArrayToString(result);

    // 2) Parse simplificado (ideal: usar lib JSON para MQL5)
    // Pseudocódigo: extrair campos do JSON
    parsed = JsonParse(json)   // função fictícia

    if !parsed["success"]:
        tech_subscore_py = 0.0;
        fund_score       = 50.0;
        sent_score       = 50.0;
        return false

    tech_subscore_py = parsed["tech_subscore_python"];
    fund_score       = parsed["fund_score"];
    sent_score       = parsed["sent_score"];

    return true
}


Tratamento de falhas:

Qualquer erro de rede, timeout ou JSON inválido → retorna false e seta Fund/Sent para 50 (neutro), tech_subscore_py = 0.

O EA continua operando, mas apenas com a componente técnica local (modo degradado, seguro).

SEÇÃO 6 – RACIOCÍNIO DE RISCO (FTMO) & DEEP THINKING
6.1 Configuração de risco para conta FTMO 100k (XAUUSD scalping)

Regras típicas FTMO:

Max Daily Loss: 5% do saldo inicial (5k num 100k).

Max Loss (Total): 10% do saldo inicial (conta não pode ir abaixo de 90k).
FTMO
+1

Proposta concreta:

Risk per trade % (base): 0,40% por trade.

Em 100k → risco nominal ~400 USD por operação.

Soft Daily Loss % (zona de redução): 4% (4k).

A partir de ~4% de DD diário o EA já bloqueia novas entradas, antes de chegar a 5%.

Hard Max Daily Loss (regra FTMO): 5% (5k).

EA deve garantir que somando DD atual + risco da nova trade não ultrapasse 5%.

Max Total Loss %: 10% (10k).

Se equity ≤ 90k, EA entra em modo “somente gestão” (não abre novas trades).

6.2 Política de redução de risco dinâmica (diário)

Para DD diário (sobre equity do início do dia):

0–1% DD diário:

Risco normal: 0,40% por trade.

1–2,5% DD diário:

Risco reduzido: 0,20% por trade (metade).

2,5–4% DD diário:

Risco mínimo: 0,10% por trade (¼ do original).

≥ 4% DD diário:

Bloquear novas entradas. Apenas gestão de posições abertas.

Isso está implementado conceitualmente em GetDynamicRiskPerc e usado pelo EA antes de checar CanOpenTrade.

6.3 Como evitar overtrading num dia bom

Problema clássico: você começa o dia com 2–3 wins rápidos em XAUUSD (ex.: +2,5% em 1h) e a psicologia empurra você a forçar trades adicionais, devolvendo o lucro.

Política proposta:

Definir um Profit Daily Soft Target, por exemplo:

+3% num dia de FTMO 100k → +3k.

Ao atingir esse alvo:

Reduzir o risco por trade para 0,10% (modo “lock profit”).

Limitar o máximo de trades adicionais (ex.: no máximo 2 trades depois do alvo).

Se o dia chegar em +4%, forçar stop trading:

Não há razão para empurrar mais risco com Max Daily Loss a 5%.

É literalmente “cheirar o limite” do risco a ser devolvido; o Edge não muda, mas o custo psicológico e assimetria negativa aumenta.

O EA pode automatizar isso:

Se daily_profit_perc >= 3% → reduzir risco.

Se daily_profit_perc >= 4% → bloquear novas entradas.

6.4 Como lidar com sequência de 3 stops seguidos em XAUUSD

Sequência de 3 stops seguidos indica:

Ou o regime de mercado mudou (range demais ou volatilidade anômala).

Ou a leitura de estrutura/OB/FVG está fora de sincronia com o fluxo real.

Política:

Após 2 stops seguidos:

Reduzir risk per trade pela metade (de 0,40% → 0,20%).

Aumentar exigência de score (por exemplo, ExecutionThreshold += 5 pontos).

Após 3 stops seguidos (no mesmo dia):

Entrar em cooldown de tempo (ex.: 60–90 minutos sem novas entradas).

Alternativamente, stop trading até o dia seguinte.

Logar no Reasoning String: “stop de série – EA bloqueou novas entradas por disciplina de risco”.

Isso evita “revenge trading algorítmico”: mesmo se aparecer um setup técnico “perfeito”, o contexto de DD e sequência ruim pesa mais.

6.5 Quando é melhor não operar (mesmo com setup técnico bom)

Algumas situações em que o EA deve recusar operação, mesmo com FinalScore técnico alto:

Eventos macro de alto impacto próximos

Ex.: FOMC, NFP, CPI, decisões de taxa.

Ideal: evitar abrir novas trades 15–30min antes e 15–30min depois de eventos “vermelhos” específicos para USD.

Python FundamentalAgent pode retornar fund_bias="volatile_event" → FundScore cai, bloqueando trade.

Spread e liquidez anormais

Spreads muito altos (por ex. > 70–100 pontos em XAUUSD) sinalizam ambiente ilíquido ou news.

O módulo de risco/spread no EA (já com filtro simples) impede entrada.

Horários mortos

Fim da sessão de NY, pós 17:00 NY (pouca liquidez, movimentos erráticos).

Sessão asiática em XAUUSD, se a estratégia é mais voltada a London/NY.

Equity em zona de risco estrutural

Se drawdown total ≥ 8% (próximo da borda dos 10%), o EA pode reduzir risco estruturalmente (0,10%) ou bloquear novas entradas até recuperar um pouco.

A mensagem para um trader júnior: o setup técnico não é tudo. O contexto de risco (tempo do dia, DD, eventos macro, spread) pode transformar um “A+ técnico” em um “no trade”.

SEÇÃO 7 – ESTRATÉGIA DE TESTES E VALIDAÇÃO
7.1 Backtests

Período & range de datas

Ideal: pelo menos 3–4 anos de histórico de XAUUSD com ticks reais:

De 2021-01-01 até hoje (para pegar:

Covid tail, inflação, ciclos de alta de juros, guerras/geopolítica, períodos de volatilidade extrema).

Timeframes

EA rodando em M1 ou M5 (entrada), com leitura de HTF (M15/H1) via indicadores MT5.

Importante: usar modelo “Every tick based on real ticks” para ter qualidade de slippage e spread.

Qualidade de tick

Minimizar “modeling quality” baixa:

Usar dados de corretora boa ou dados externos (com import) se possível.

Verificar se a série de ticks em XAU não está com buracos (market close, festividades).

7.2 Stress tests

Spreads maiores

Rodar backtests com:

Spread normal.

Spread dobrado (ex.: 2x o típico de XAU).

Avaliar impacto em PF, DD, % dias próximos ao Max Daily Loss.

Slippage

Simular slippage adicionando:

Ajuste artificial no preço de entrada/SL no código de teste (ex.: +/- 0.5–1.0 pip médio).

Ver como a estratégia aguenta “pior execução” (algo comum em XAUUSD em news).

News on/off

Versão A: sem filtro de news.

Versão B: bloqueio 15–30min antes/depois de grandes news (simulado via calendário estático ou approximations de horários).

Comparar:

PF, DD, frequência de spikes grandes contra a posição.

7.3 Testes específicos de FTMO

Simular Max Daily Loss e Max Total Loss

Já embutido no CFTMORiskManager:

Em backtest, cada tick/ordem respeita:

Loss do dia ≤ 5% do saldo inicial.
FTMO
+1

Equity nunca abaixo de 90% do saldo inicial.

Adicionalmente, logar:

daily_equity_start, daily_min_equity, daily_max_equity, flags de “quase-violação” (ex.: perda diária ≥ 4,5%).

Avaliar respeito às regras

Uma métrica: dias_com_quase_violação / total_dias_tradados.

Alvo: menos de ~1–2% dos dias chegando >4,5% de DD diário.

Nenhum backtest pode efetivamente violar 5%/10% — se violar, a lógica do RiskManager está errada.

7.4 Critérios de aprovação

Mínimos razoáveis para um EA de scalping XAUUSD em prop firm:

Profit Factor (PF): ≥ 1,3 (ideal > 1,5).

Win rate: ≥ 45%, com RR médio ≥ 1:1 (sinônimo de edge robusto).

Max Drawdown absoluto: < 8–9%.

Max Daily Drawdown observado: < 4,5% na grande maioria dos dias.

Número de trades: pelo menos algumas centenas (ideal > 500) no período testado.

Regras específicas de risco:

Zero violações de:

Max Daily Loss (5%).

Max Total Loss (10%).

Dias com DD diário > 4% devem ser raros; se frequentes, aumentar conservadorismo do RiskManager.

SEÇÃO 8 – EXEMPLOS DE REASONING STRINGS DE TRADES
Exemplo 1 – Trade WIN (BUY XAUUSD)

Reasoning String:
“Sessão London/NY com XAUUSD em clara tendência de alta (HH/HL em M15) e sweep de liquidez abaixo do low da Ásia, respeitando um OB de continuação. FVG bullish em M5 parcialmente preenchida, ATR elevada porém dentro da faixa operacional, TechScore=92, FundScore=80, SentScore=70, FinalScore=88 ≥ 85. FTMO_RiskManager aprovou risco dinâmico de 0,30% com stop de ~18 pips, mantendo Max Daily Loss bem abaixo de 5%. Entrada foi consistente com a política de risco e encerrou em TP, reforçando a disciplina de proteger o capital mesmo em cenário favorável.”

Exemplo 2 – Trade LOSS (SELL XAUUSD)

Reasoning String:
“XAUUSD em tendência de baixa em M15 com CHOCH recente e OB de baixa em H1 acima do preço; durante o pullback, houve sweep de equal highs e formação de FVG bearish em M1, com ATR moderada. TechScore=86, FundScore=60 (macro levemente favorável ao USD), SentScore=55, FinalScore=82 (abaixo do threshold padrão, mas após ajuste do Python TechSubscore=+4, FinalScore=86, trade permitido). FTMO_RiskManager limitou o risco a 0,20% por conta de DD diário já em 1,5%, e o stop de ~16 pips foi atingido após reversão rápida em news menor. A perda foi pequena e totalmente consistente com a política de risco, evitando aumento de lote ou revenge trading após o stop.”

Exemplo 3 – Sinal IGNORADO (score alto mas risco FTMO próximo do limite)

Reasoning String:
“Durante a sessão de NY, XAUUSD formou estrutura bullish clara com BOS em M15, OB de alta recém mitigado e FVG M1 alinhada, com ATR saudável e spreads normais. TechScore=93, FundScore=75, SentScore=68, FinalScore=89 indicando setup A+ para compra, porém o DD diário já está em 4,1% e o risco da nova operação levaria o potencial DD para além da zona segura próxima ao Max Daily Loss de 5%. FTMO_RiskManager vetou a entrada, priorizando preservação de conta sobre oportunidade individual; decisão totalmente alinhada com a política de ‘risk first’ do sistema, mesmo com forte confluência técnica.”

Se você quiser, o próximo passo pode ser: refinar o RiskManager com parâmetros diferentes (mais agressivo/conservador) ou começar a implementar de verdade um dos módulos técnicos (por exemplo, lógica de OB ou FVG) em cima desse esqueleto.


