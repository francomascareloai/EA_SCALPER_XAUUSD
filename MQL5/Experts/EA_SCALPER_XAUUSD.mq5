//+------------------------------------------------------------------+
//|                                            EA_SCALPER_XAUUSD.mq5 |
//|                                                           Franco |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Franco"
#property link      "https://www.mql5.com"
#property version   "2.00"
#property strict

//--- Includes
#include <EA_Elite_Components/FTMO_RiskManager.mqh>
#include <EA_Elite_Components/SignalScoringModule.mqh>
#include <EA_Elite_Components/TradeExecutor.mqh>
//#include <EA_Elite_Components/PythonBridge.mqh> // Phase 2

//--- Input Parameters: Risk Management
input group "=== Risk Management (FTMO) ==="
input double   InpRiskPerTrade      = 0.5;      // Risk Per Trade (%)
input double   InpMaxDailyLoss      = 5.0;      // Max Daily Loss (%)
input double   InpSoftStop          = 3.5;      // Soft Stop Level (%)
input double   InpMaxTotalLoss      = 10.0;     // Max Total Loss (%)
input int      InpMaxTradesPerDay   = 20;       // Max Trades Per Day

//--- Input Parameters: Scoring Engine
input group "=== Scoring Engine ==="
input int      InpExecutionThreshold = 85;      // Execution Score Threshold (0-100)
input double   InpWeightTech        = 0.6;      // Weight: Technical
input double   InpWeightFund        = 0.25;     // Weight: Fundamental
input double   InpWeightSent        = 0.15;     // Weight: Sentiment

//--- Input Parameters: Execution
input group "=== Execution Settings ==="
input int      InpSlippage          = 50;       // Max Slippage (points)
input int      InpMagicNumber       = 123456;   // Magic Number
input string   InpTradeComment      = "EA_SCALPER_v2"; // Trade Comment

//--- Global Objects
CFTMO_RiskManager    g_RiskManager;
CSignalScoringModule g_ScoringEngine;
CTradeExecutor       g_Executor;

//--- Global State
bool g_IsEmergencyMode = false;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   // 1. Initialize Risk Manager
   if(!g_RiskManager.Init(InpRiskPerTrade, InpMaxDailyLoss, InpMaxTotalLoss, InpMaxTradesPerDay, InpSoftStop))
   {
      Print("Critical Error: Risk Manager Initialization Failed!");
      return(INIT_FAILED);
   }

   // 2. Initialize Scoring Engine
   if(!g_ScoringEngine.Init(InpWeightTech, InpWeightFund, InpWeightSent))
   {
      Print("Critical Error: Scoring Engine Initialization Failed!");
      return(INIT_FAILED);
   }

   // 3. Initialize Executor
   g_Executor.Init(InpMagicNumber, InpSlippage, InpTradeComment);

   // 4. Timer for Python Bridge (Phase 2)
   EventSetTimer(1); // 1 second heartbeat

   Print("EA_SCALPER_XAUUSD v2.0 Initialized Successfully.");
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
   Print("EA_SCALPER_XAUUSD v2.0 Deinitialized. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // 1. Critical Risk Check (Fastest)
   if(g_IsEmergencyMode) return;
   
   // Update Risk State (Floating P/L, Drawdown)
   g_RiskManager.OnTick();
   
   if(g_RiskManager.IsTradingHalted())
   {
      if(!g_IsEmergencyMode)
      {
         Print("TRADING HALTED: Risk Limits Breached.");
         g_IsEmergencyMode = true;
      }
      return;
   }

   // 2. Manage Open Positions (Trailing, BE, Partials)
   g_Executor.ManagePositions();

   // 3. Signal Generation (Scoring)
   // Only check for new trades if we haven't hit daily limits
   if(g_RiskManager.CanOpenNewTrade())
   {
      // Calculate Score (Fast Lane - MQL5 only for Phase 1)
      int score = g_ScoringEngine.CalculateScore();
      
      // 4. Execution Decision
      if(score >= InpExecutionThreshold)
      {
         // Determine Direction and Levels first
         ENUM_ORDER_TYPE direction = g_ScoringEngine.GetDirection();
         double slPrice = g_ScoringEngine.GetStopLossPrice();
         double tpPrice = g_ScoringEngine.GetTakeProfitPrice();
         
         // Calculate SL Distance in Points based on Current Price
         double currentPrice = (direction == ORDER_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);
         double slPoints = MathAbs(currentPrice - slPrice) / _Point;
         
         // Final Risk Check before execution (Lot Size calculation)
         double lotSize = g_RiskManager.CalculateLotSize(slPoints);
         
         if(lotSize > 0)
         {
             if(g_Executor.ExecuteTrade(direction, lotSize, slPrice, tpPrice, score))
             {
                Print("Trade Executed! Score: ", score, " Lot: ", lotSize);
                g_RiskManager.OnTradeExecuted(); // Update counters
             }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
{
   // Phase 2: Poll Python Hub for "Slow Lane" updates (Reasoning Strings, Fundamental Data)
   // g_PythonBridge.OnTimer();
}
//+------------------------------------------------------------------+
