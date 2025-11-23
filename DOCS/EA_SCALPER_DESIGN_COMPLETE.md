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
