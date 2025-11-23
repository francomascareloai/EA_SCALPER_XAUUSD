//+------------------------------------------------------------------+
//|                                                       CState.mqh |
//|                                                   MQL5 Architect |
//|                                      Copyright 2025, Elite Ops.  |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Elite Ops."
#property version   "1.00"

//--- Enums
enum EState
  {
   STATE_IDLE,       // Normal operation, waiting for signal
   STATE_BUSY,       // Managing open trade
   STATE_SURVIVAL,   // High Risk Mode (Market Driven)
   STATE_EMERGENCY   // System Failure (Heartbeat Lost)
  };

enum EMarketRegime
  {
   REGIME_NORMAL,
   REGIME_HIGH_VOL,
   REGIME_NEWS_WINDOW,
   REGIME_WAR
  };

//--- Structs
struct SRiskParams
  {
   double            risk_per_trade;      // % of Equity
   double            max_daily_loss;      // % of Balance
   double            max_total_loss;      // % of Balance
   double            soft_stop_threshold; // % of Daily Loss to trigger warning
   
   SRiskParams()
     {
      risk_per_trade = 0.5;
      max_daily_loss = 5.0;
      max_total_loss = 10.0;
      soft_stop_threshold = 3.5;
     }
  };

struct SNewsEvent
  {
   datetime          time;
   string            currency;
   string            impact; // HIGH, MEDIUM, LOW
   
   SNewsEvent()
     {
      time = 0;
      currency = "";
      impact = "";
     }
  };
//+------------------------------------------------------------------+
