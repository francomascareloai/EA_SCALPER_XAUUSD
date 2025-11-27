# EA_SCALPER_XAUUSD – Multi-Agent Hybrid System
## PARTE 3: Código MQL5 - EA Principal

---

# 💻 SEÇÃO 4 – CÓDIGO MQL5 ESSENCIAL

## 4.1 EA Principal (EA_SCALPER_XAUUSD.mq5)

```mql5
//+------------------------------------------------------------------+
//| filepath: c:\Users\Admin\Documents\EA_SCALPER_XAUUSD\EA_SCALPER_XAUUSD.mq5
//+------------------------------------------------------------------+
//|                                           EA_SCALPER_XAUUSD.mq5  |
//|                        TradeDev_Master Multi-Agent System        |
//+------------------------------------------------------------------+
#property copyright "TradeDev_Master"
#property link      "https://github.com/tradedev"
#property version   "1.00"
#property strict

//--- Include modules
#include "Modules\CFTMORiskManager.mqh"
#include "Modules\CSignalScoringModule.mqh"
#include "Modules\CDataStructures.mqh"
#include "Modules\COrderBlockModule.mqh"
#include "Modules\CFVGModule.mqh"
#include "Modules\CLiquidityModule.mqh"
#include "Modules\CMarketStructureModule.mqh"
#include "Modules\CVolatilityModule.mqh"
#include "Modules\CTradeExecutor.mqh"
#include "Modules\CLogger.mqh"

//+------------------------------------------------------------------+
//| INPUT PARAMETERS                                                  |
//+------------------------------------------------------------------+

//--- Risk Management Inputs
input group "══════ FTMO RISK MANAGEMENT ══════"
input double   InpRiskPerTrade        = 0.5;     // Risk per trade (%)
input double   InpMaxDailyLossPercent = 4.0;     // Max Daily Loss (%) - FTMO=5%
input double   InpMaxTotalLossPercent = 8.0;     // Max Total Loss (%) - FTMO=10%
input double   InpSoftDailyLoss       = 2.0;     // Soft Daily Loss (%) - Start reducing
input double   InpMinLotSize          = 0.01;    // Minimum lot size
input double   InpMaxLotSize          = 5.0;     // Maximum lot size

//--- Scoring Thresholds
input group "══════ SCORING ENGINE ══════"
input int      InpExecutionThreshold  = 85;      // Min FinalScore to execute (0-100)
input int      InpTechScorePreFilter  = 60;      // Min TechScore to continue
input double   InpTechWeight          = 0.50;    // Weight: Technical Score
input double   InpFundWeight          = 0.25;    // Weight: Fundamental Score
input double   InpSentWeight          = 0.25;    // Weight: Sentiment Score

//--- Strategy Parameters
input group "══════ STRATEGY SETTINGS ══════"
input ENUM_TIMEFRAMES InpHTF          = PERIOD_H4;  // Higher Timeframe
input ENUM_TIMEFRAMES InpMTF          = PERIOD_H1;  // Medium Timeframe
input ENUM_TIMEFRAMES InpLTF          = PERIOD_M15; // Lower Timeframe (entry)
input int      InpATRPeriod           = 14;         // ATR Period
input double   InpRiskRewardRatio     = 2.0;        // Risk:Reward Ratio
input int      InpMaxSpreadPoints     = 50;         // Max spread (points)

//--- Python Hub Integration
input group "══════ PYTHON HUB ══════"
input bool     InpUsePythonHub        = true;       // Enable Python Hub
input string   InpPythonHubURL        = "http://127.0.0.1:8000/analyze";
input int      InpPythonTimeout       = 5000;       // Timeout (ms)
input int      InpPythonUpdateSecs    = 30;         // Update interval (sec)

//--- Trade Management
input group "══════ TRADE MANAGEMENT ══════"
input int      InpMagicNumber         = 20240101;   // Magic Number
input int      InpMaxTradesPerDay     = 5;          // Max trades per day
input bool     InpUseTrailingStop     = true;       // Enable trailing stop
input double   InpTrailingATRMulti    = 1.5;        // Trailing ATR multiplier

//+------------------------------------------------------------------+
//| GLOBAL VARIABLES                                                  |
//+------------------------------------------------------------------+

// Module instances
CFTMORiskManager      *g_RiskManager;
CSignalScoringModule  *g_ScoringModule;
COrderBlockModule     *g_OBModule;
CFVGModule            *g_FVGModule;
CLiquidityModule      *g_LiqModule;
CMarketStructureModule *g_StructModule;
CVolatilityModule     *g_VolModule;
CTradeExecutor        *g_Executor;
CLogger               *g_Logger;

// Python Hub cache
SPythonHubData g_PythonData;

// Session tracking
int      g_TickCounter;
int      g_TradesToday;
datetime g_TodayStart;
double   g_DailyStartBalance;

//+------------------------------------------------------------------+
//| Expert initialization function                                    |
//+------------------------------------------------------------------+
int OnInit()
{
    //--- Validate symbol
    if(_Symbol != "XAUUSD" && _Symbol != "GOLD" && 
       StringFind(_Symbol, "XAU") < 0)
    {
        Print("ERROR: EA designed for XAUUSD. Current: ", _Symbol);
        return INIT_FAILED;
    }
    
    //--- Initialize modules
    g_RiskManager = new CFTMORiskManager(
        InpRiskPerTrade, InpMaxDailyLossPercent,
        InpMaxTotalLossPercent, InpSoftDailyLoss,
        InpMinLotSize, InpMaxLotSize
    );
    
    g_ScoringModule = new CSignalScoringModule(
        InpTechWeight, InpFundWeight, InpSentWeight
    );
    
    g_OBModule     = new COrderBlockModule(InpHTF, InpMTF, InpLTF);
    g_FVGModule    = new CFVGModule(InpMTF, InpLTF);
    g_LiqModule    = new CLiquidityModule(InpHTF, InpMTF);
    g_StructModule = new CMarketStructureModule(InpHTF, InpMTF);
    g_VolModule    = new CVolatilityModule(InpATRPeriod);
    g_Executor     = new CTradeExecutor(InpMagicNumber, InpMaxSpreadPoints);
    g_Logger       = new CLogger("EA_SCALPER_XAUUSD");
    
    //--- Initialize Python cache with defaults
    g_PythonData.isValid = false;
    g_PythonData.lastUpdate = 0;
    g_PythonData.fundScore = 50.0;
    g_PythonData.sentScore = 50.0;
    
    //--- Initialize session
    g_TickCounter = 0;
    g_TradesToday = 0;
    g_TodayStart = iTime(_Symbol, PERIOD_D1, 0);
    g_DailyStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    
    //--- Initialize Risk Manager
    g_RiskManager.Initialize(
        AccountInfoDouble(ACCOUNT_BALANCE),
        AccountInfoDouble(ACCOUNT_EQUITY)
    );
    
    //--- Set timer for Python Hub
    if(InpUsePythonHub)
        EventSetTimer(InpPythonUpdateSecs);
    
    //--- Log initialization
    g_Logger.Info("EA_SCALPER_XAUUSD initialized");
    g_Logger.Info(StringFormat("Balance: $%.2f | Risk: %.2f%% | Threshold: %d",
        AccountInfoDouble(ACCOUNT_BALANCE), InpRiskPerTrade, InpExecutionThreshold));
    
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    //--- Cleanup
    if(g_RiskManager  != NULL) { delete g_RiskManager;  g_RiskManager  = NULL; }
    if(g_ScoringModule!= NULL) { delete g_ScoringModule;g_ScoringModule= NULL; }
    if(g_OBModule     != NULL) { delete g_OBModule;     g_OBModule     = NULL; }
    if(g_FVGModule    != NULL) { delete g_FVGModule;    g_FVGModule    = NULL; }
    if(g_LiqModule    != NULL) { delete g_LiqModule;    g_LiqModule    = NULL; }
    if(g_StructModule != NULL) { delete g_StructModule; g_StructModule = NULL; }
    if(g_VolModule    != NULL) { delete g_VolModule;    g_VolModule    = NULL; }
    if(g_Executor     != NULL) { delete g_Executor;     g_Executor     = NULL; }
    if(g_Logger       != NULL) { delete g_Logger;       g_Logger       = NULL; }
    
    EventKillTimer();
    Print("EA_SCALPER_XAUUSD deinitialized. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
    ulong startTime = GetMicrosecondCount();
    
    //--- Daily reset check
    datetime currentDayStart = iTime(_Symbol, PERIOD_D1, 0);
    if(currentDayStart != g_TodayStart)
    {
        g_TodayStart = currentDayStart;
        g_TradesToday = 0;
        g_DailyStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
        g_RiskManager.ResetDaily(g_DailyStartBalance);
        g_Logger.Info("New day. Balance: $" + DoubleToString(g_DailyStartBalance, 2));
    }
    
    //--- Update Risk Manager
    g_RiskManager.Update(AccountInfoDouble(ACCOUNT_EQUITY));
    
    //--- Check if trading allowed
    if(!g_RiskManager.IsTradingAllowed())
        return;
    
    //--- Check spread
    long currentSpread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
    if(currentSpread > InpMaxSpreadPoints)
        return;
    
    //--- Manage existing position
    if(HasOpenPosition())
    {
        ManageOpenPosition();
        return;
    }
    
    //--- Rate limiting
    g_TickCounter++;
    if(g_TickCounter % 5 != 0)
        return;
    
    //--- Max trades check
    if(g_TradesToday >= InpMaxTradesPerDay)
        return;
    
    //--- Technical Analysis
    SOrderBlockData obData;
    g_OBModule.Analyze(obData);
    
    SFVGData fvgData;
    g_FVGModule.Analyze(fvgData);
    
    SLiquidityData liqData;
    g_LiqModule.Analyze(liqData);
    
    SMarketStructureData structData;
    g_StructModule.Analyze(structData);
    
    SVolatilityData volData;
    g_VolModule.Analyze(volData);
    
    //--- Compute Tech Score
    double techScore = g_ScoringModule.ComputeTechScore(
        obData.hasValidOB, obData.obStrength,
        fvgData.hasFVG, fvgData.fvgStrength,
        liqData.liquiditySwept,
        structData.structureStrength, structData.hasBOS,
        volData.currentATR, volData.volState
    );
    
    //--- Pre-filter
    if(techScore < InpTechScorePreFilter)
        return;
    
    //--- Get Python scores (from cache)
    double fundScore = g_PythonData.isValid ? g_PythonData.fundScore : 50.0;
    double sentScore = g_PythonData.isValid ? g_PythonData.sentScore : 50.0;
    
    //--- Compute Final Score
    double finalScore = g_ScoringModule.ComputeFinalScore(techScore, fundScore, sentScore);
    
    //--- Check threshold
    if(finalScore < InpExecutionThreshold)
    {
        g_Logger.Debug(StringFormat("Rejected: Score %.1f < %d", finalScore, InpExecutionThreshold));
        return;
    }
    
    //--- Determine direction
    ENUM_ORDER_TYPE direction = DetermineDirection(obData, structData, liqData);
    if(direction == WRONG_VALUE)
        return;
    
    //--- Calculate SL/TP
    double slPoints = volData.currentATR * 1.5;
    double tpPoints = slPoints * InpRiskRewardRatio;
    
    //--- RISK MANAGER VETO CHECK
    double lotSize;
    string vetoReason;
    if(!g_RiskManager.CanOpenTrade(InpRiskPerTrade, slPoints, lotSize, vetoReason))
    {
        g_Logger.Warn("VETOED: " + vetoReason);
        SendNotification("⚠️ " + vetoReason);
        return;
    }
    
    //--- Execute trade
    double entryPrice = (direction == ORDER_TYPE_BUY) ? 
        SymbolInfoDouble(_Symbol, SYMBOL_ASK) : 
        SymbolInfoDouble(_Symbol, SYMBOL_BID);
    
    double slPrice, tpPrice;
    if(direction == ORDER_TYPE_BUY)
    {
        slPrice = entryPrice - slPoints * _Point;
        tpPrice = entryPrice + tpPoints * _Point;
    }
    else
    {
        slPrice = entryPrice + slPoints * _Point;
        tpPrice = entryPrice - tpPoints * _Point;
    }
    
    ulong ticket = g_Executor.OpenPosition(direction, lotSize, slPrice, tpPrice);
    
    if(ticket > 0)
    {
        g_TradesToday++;
        
        string reasoning = GenerateReasoning(
            direction, lotSize, entryPrice, slPrice, tpPrice,
            techScore, fundScore, sentScore, finalScore,
            obData, structData, volData
        );
        
        g_Logger.LogTrade(ticket, reasoning);
        SendNotification(reasoning);
    }
    
    //--- Performance check
    ulong elapsed = GetMicrosecondCount() - startTime;
    if(elapsed > 50000)
        g_Logger.Warn("OnTick exceeded 50ms: " + IntegerToString(elapsed/1000) + "ms");
}

//+------------------------------------------------------------------+
//| Timer function                                                    |
//+------------------------------------------------------------------+
void OnTimer()
{
    if(!InpUsePythonHub)
        return;
    
    // Call Python Hub - implementation in Section 5
    CallPythonHub();
}

//+------------------------------------------------------------------+
//| Check for open position                                           |
//+------------------------------------------------------------------+
bool HasOpenPosition()
{
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        ulong ticket = PositionGetTicket(i);
        if(PositionSelectByTicket(ticket))
        {
            if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
               PositionGetInteger(POSITION_MAGIC) == InpMagicNumber)
                return true;
        }
    }
    return false;
}

//+------------------------------------------------------------------+
//| Manage open position                                              |
//+------------------------------------------------------------------+
void ManageOpenPosition()
{
    if(!InpUseTrailingStop)
        return;
    
    // Implementation: trailing stop based on ATR
    // ... (see full implementation in separate file)
}

//+------------------------------------------------------------------+
//| Determine trade direction                                         |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE DetermineDirection(const SOrderBlockData &ob,
                                   const SMarketStructureData &st,
                                   const SLiquidityData &liq)
{
    int bull = 0, bear = 0;
    
    // Structure weight (highest priority)
    if(st.currentStructure == STRUCTURE_BULLISH) bull += 3;
    else if(st.currentStructure == STRUCTURE_BEARISH) bear += 3;
    
    // Order Block
    if(ob.hasValidOB)
    {
        if(ob.obType == OB_BULLISH) bull += 2;
        else if(ob.obType == OB_BEARISH) bear += 2;
    }
    
    // Liquidity sweep
    if(liq.liquiditySwept)
    {
        if(liq.sweepType == SWEEP_BULLISH) bull += 2;
        else if(liq.sweepType == SWEEP_BEARISH) bear += 2;
    }
    
    // Need clear direction
    if(bull >= 5 && bull - bear >= 3) return ORDER_TYPE_BUY;
    if(bear >= 5 && bear - bull >= 3) return ORDER_TYPE_SELL;
    
    return WRONG_VALUE;
}

//+------------------------------------------------------------------+
//| Generate reasoning string                                         |
//+------------------------------------------------------------------+
string GenerateReasoning(ENUM_ORDER_TYPE dir, double lot, 
                         double entry, double sl, double tp,
                         double tech, double fund, double sent, double final,
                         const SOrderBlockData &ob,
                         const SMarketStructureData &st,
                         const SVolatilityData &vol)
{
    string d = (dir == ORDER_TYPE_BUY) ? "BUY" : "SELL";
    string structure = (st.currentStructure == STRUCTURE_BULLISH) ? "Bull" : 
                       (st.currentStructure == STRUCTURE_BEARISH) ? "Bear" : "Range";
    
    return StringFormat(
        "📊 %s XAUUSD @ %.2f | Lot: %.2f\n"
        "SL: %.2f | TP: %.2f | RR: %.1f\n"
        "Score: %.0f (T:%.0f F:%.0f S:%.0f)\n"
        "Structure: %s | ATR: %.0f | DD: %.1f%%",
        d, entry, lot, sl, tp, InpRiskRewardRatio,
        final, tech, fund, sent,
        structure, vol.currentATR, g_RiskManager.GetDailyDDPercent()
    );
}
```
