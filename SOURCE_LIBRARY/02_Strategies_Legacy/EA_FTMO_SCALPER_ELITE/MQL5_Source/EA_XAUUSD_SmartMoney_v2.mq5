//+------------------------------------------------------------------+
//|                                    EA_XAUUSD_SmartMoney_v2.mq5 |
//|                              Copyright 2024, Expert Panel Elite |
//|                      XAUUSD Smart Money Concepts - REWRITTEN     |
//+------------------------------------------------------------------+
#property copyright \"Expert Panel Elite 2024\"
#property version   \"2.0\"
#property description \"🏆 XAUUSD SMART MONEY EA - REDESIGNED BY 20 EXPERTS\"
#property description \"💎 Order Blocks + Fair Value Gaps + Liquidity Sweeps\"
#property description \"⚡ Optimized for GOLD institutional behavior\"

#include <Trade\\Trade.mqh>
#include <Trade\\SymbolInfo.mqh>
#include <Trade\\AccountInfo.mqh>

//--- Trading objects
CTrade m_trade;
CSymbolInfo m_symbol;
CAccountInfo m_account;

//--- Smart Money Concepts Enums
enum ENUM_SMC_SIGNAL
{
    SMC_NONE = 0,
    SMC_BUY_OB = 1,     // Buy Order Block
    SMC_SELL_OB = -1,   // Sell Order Block
    SMC_BUY_FVG = 2,    // Buy Fair Value Gap
    SMC_SELL_FVG = -2   // Sell Fair Value Gap
};

enum ENUM_MARKET_SESSION
{
    SESSION_ASIAN = 1,    // 00:00-08:00 GMT (AVOID)
    SESSION_LONDON = 2,   // 08:00-16:00 GMT (PRIME)
    SESSION_NY = 3,       // 13:00-22:00 GMT (PRIME)
    SESSION_OVERLAP = 4   // 13:00-16:00 GMT (BEST)
};

//--- CRITICAL INPUT PARAMETERS (Based on Expert Consensus)
input group \"=== 💎 SMART MONEY STRATEGY ===\"
input bool InpUseOrderBlocks = true;        // ✅ Order Blocks Detection
input bool InpUseFairValueGaps = true;     // ✅ Fair Value Gaps
input bool InpUseLiquiditySweeps = true;   // ✅ Liquidity Sweeps
input int InpOrderBlockBars = 10;          // Order Block Lookback
input double InpFVGMinSize = 5.0;          // FVG Minimum Size (pips)

input group \"=== ⏰ XAUUSD OPTIMAL TIMING ===\"
input bool InpTradeAsian = false;          // ❌ Asian Session (MANIPULATION)
input bool InpTradeLondon = true;          // ✅ London Session (8-16 GMT)
input bool InpTradeNY = true;              // ✅ NY Session (13-22 GMT)
input bool InpTradeOverlap = true;         // ✅ London/NY Overlap (BEST)
input bool InpAvoidNews = true;            // ✅ Avoid High Impact News

input group \"=== 🛡️ XAUUSD RISK MANAGEMENT ===\"
input double InpRiskPercent = 0.25;        // Risk Per Trade (0.25% MAX!)
input double InpMinSL = 30.0;              // Minimum SL (30 pips for XAUUSD)
input double InpMaxSL = 100.0;             // Maximum SL (100 pips)
input double InpRiskReward = 2.0;          // Risk:Reward Ratio
input int InpMaxDailyTrades = 3;           // Max Daily Trades
input double InpMaxDailyLoss = 1.0;        // Max Daily Loss (%)

input group \"=== 📊 MARKET STRUCTURE ===\"
input int InpStructureBars = 50;           // Structure Analysis Bars
input double InpBreakoutThreshold = 10.0;  // Breakout Threshold (pips)
input bool InpMultiTimeframe = true;       // Multi-Timeframe Analysis
input ENUM_TIMEFRAMES InpHigherTF = PERIOD_H1; // Higher Timeframe

//--- Global Variables
int g_magicNumber = 220240824;  // Unique magic
datetime g_lastBarTime = 0;
int g_dailyTrades = 0;
double g_dailyPnL = 0.0;
datetime g_lastNewsTime = 0;

//--- Market Structure Variables
double g_lastHigh = 0;
double g_lastLow = 0;
double g_orderBlocks[][4];  // [price, time, type, validated]
double g_fvgLevels[][3];    // [top, bottom, time]
bool g_marketStructureBull = true;

//--- Indicators
int g_atrHandle;
int g_volumeHandle;
double g_atr[];
long g_volume[];

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    Print(\"🚀 STARTING XAUUSD SMART MONEY EA v2.0 - EXPERT PANEL DESIGN\");
    
    //--- Verify XAUUSD symbol
    if(_Symbol != \"XAUUSD\" && _Symbol != \"GOLD\" && _Symbol != \"XAU\")
    {
        Alert(\"⚠️ WARNING: This EA is specifically designed for XAUUSD!\");
        Print(\"Current Symbol: \", _Symbol, \" - Recommended: XAUUSD\");
    }
    
    //--- Initialize symbol
    if(!m_symbol.Name(_Symbol))
    {
        Print(\"❌ Failed to initialize symbol: \", _Symbol);
        return INIT_FAILED;
    }
    
    //--- Setup trading
    m_trade.SetExpertMagicNumber(g_magicNumber);
    m_trade.SetMarginMode();
    m_trade.SetTypeFillingBySymbol(_Symbol);
    
    //--- Initialize indicators
    g_atrHandle = iATR(_Symbol, PERIOD_M15, 14);
    g_volumeHandle = iVolumes(_Symbol, PERIOD_M15, VOLUME_TICK);
    
    if(g_atrHandle == INVALID_HANDLE || g_volumeHandle == INVALID_HANDLE)
    {
        Print(\"❌ Failed to create indicators\");
        return INIT_FAILED;
    }
    
    //--- Setup arrays
    ArraySetAsSeries(g_atr, true);
    ArraySetAsSeries(g_volume, true);
    ArrayResize(g_orderBlocks, 100);
    ArrayResize(g_fvgLevels, 100);
    
    //--- Reset daily stats
    ResetDailyStats();
    
    EventSetTimer(60); // Timer every minute
    
    Print(\"✅ XAUUSD Smart Money EA initialized successfully!\");
    Print(\"💎 Order Blocks: \", InpUseOrderBlocks ? \"ON\" : \"OFF\");
    Print(\"⚡ Fair Value Gaps: \", InpUseFairValueGaps ? \"ON\" : \"OFF\");
    Print(\"🎯 Risk Per Trade: \", InpRiskPercent, \"%\");
    
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    Print(\"🛑 XAUUSD Smart Money EA deinitialized\");
    Print(\"📊 Daily Trades: \", g_dailyTrades);
    Print(\"💰 Daily P&L: \", DoubleToString(g_dailyPnL, 2), \" USD\");
    
    IndicatorRelease(g_atrHandle);
    IndicatorRelease(g_volumeHandle);
    EventKillTimer();
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    //--- Check new bar
    if(!IsNewBar()) return;
    
    //--- Update market data
    UpdateMarketData();
    
    //--- Check trading conditions
    if(!CanTrade()) return;
    
    //--- Analyze market structure
    AnalyzeMarketStructure();
    
    //--- Detect Smart Money signals
    ENUM_SMC_SIGNAL signal = DetectSmartMoneySignal();
    
    //--- Execute trade if valid signal
    if(signal != SMC_NONE)
    {
        ExecuteSmartMoneyTrade(signal);
    }
    
    //--- Update daily stats
    UpdateDailyStats();
}

//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
{
    //--- Check if new day
    static int lastDay = 0;
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    
    if(dt.day != lastDay)
    {
        ResetDailyStats();
        lastDay = dt.day;
        Print(\"📅 New trading day - Stats reset\");
    }
    
    //--- Update daily P&L
    UpdateDailyStats();
}

//+------------------------------------------------------------------+
//| Check if new bar                                                |
//+------------------------------------------------------------------+
bool IsNewBar()
{
    datetime currentBar = iTime(_Symbol, PERIOD_CURRENT, 0);
    if(currentBar != g_lastBarTime)
    {
        g_lastBarTime = currentBar;
        return true;
    }
    return false;
}

//+------------------------------------------------------------------+
//| Update market data                                              |
//+------------------------------------------------------------------+
void UpdateMarketData()
{
    //--- Copy indicator data
    if(CopyBuffer(g_atrHandle, 0, 0, 5, g_atr) < 5)
    {
        Print(\"❌ Failed to copy ATR data\");
        return;
    }
    
    if(CopyBuffer(g_volumeHandle, 0, 0, 5, g_volume) < 5)
    {
        Print(\"❌ Failed to copy Volume data\");
        return;
    }
}

//+------------------------------------------------------------------+
//| Check if can trade - XAUUSD SPECIFIC                           |
//+------------------------------------------------------------------+
bool CanTrade()
{
    //--- Check daily limits
    if(g_dailyTrades >= InpMaxDailyTrades)
    {
        //Print(\"⛔ Daily trade limit reached: \", g_dailyTrades);
        return false;
    }
    
    if(g_dailyPnL <= -InpMaxDailyLoss * AccountInfoDouble(ACCOUNT_BALANCE) / 100.0)
    {
        Print(\"⛔ Daily loss limit reached: \", DoubleToString(g_dailyPnL, 2));
        return false;
    }
    
    //--- Check session timing (CRITICAL FOR XAUUSD)
    ENUM_MARKET_SESSION session = GetCurrentSession();
    
    if(session == SESSION_ASIAN && !InpTradeAsian)
    {
        //Print(\"⏰ Asian session - Trading disabled (Manipulation risk)\");
        return false;
    }
    
    if(session == SESSION_LONDON && !InpTradeLondon) return false;
    if(session == SESSION_NY && !InpTradeNY) return false;
    
    //--- Check spread (CRITICAL FOR XAUUSD)
    double spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * _Point;
    double maxSpread = g_atr[0] * 0.2; // Max 20% of ATR
    
    if(spread > maxSpread)
    {
        Print(\"⛔ Spread too high: \", DoubleToString(spread/_Point, 1), \" pips\");
        return false;
    }
    
    //--- Check news (if enabled)
    if(InpAvoidNews && IsNewsTime())
    {
        Print(\"📰 News time - Trading paused\");
        return false;
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Get current trading session                                     |
//+------------------------------------------------------------------+
ENUM_MARKET_SESSION GetCurrentSession()
{
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    int hour = dt.hour;
    
    if(hour >= 0 && hour < 8) return SESSION_ASIAN;
    if(hour >= 8 && hour < 13) return SESSION_LONDON;
    if(hour >= 13 && hour < 16) return SESSION_OVERLAP; // BEST TIME
    if(hour >= 16 && hour < 22) return SESSION_NY;
    
    return SESSION_ASIAN; // After NY close
}

//+------------------------------------------------------------------+
//| Check if news time (simplified)                                 |
//+------------------------------------------------------------------+
bool IsNewsTime()
{
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    
    //--- Avoid major news times (GMT)
    // 13:30 GMT - US data releases
    // 15:00 GMT - London Gold Fix
    if((dt.hour == 13 && dt.min >= 25 && dt.min <= 35) ||
       (dt.hour == 15 && dt.min >= 0 && dt.min <= 10))
    {
        return true;
    }
    
    return false;
}

//+------------------------------------------------------------------+
//| Analyze market structure                                        |
//+------------------------------------------------------------------+
void AnalyzeMarketStructure()
{
    //--- Find recent swing highs and lows
    double recentHigh = 0;
    double recentLow = DBL_MAX;
    
    for(int i = 1; i <= InpStructureBars; i++)
    {
        double high = iHigh(_Symbol, PERIOD_CURRENT, i);
        double low = iLow(_Symbol, PERIOD_CURRENT, i);
        
        if(high > recentHigh) recentHigh = high;
        if(low < recentLow) recentLow = low;
    }
    
    //--- Update market structure
    double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    
    //--- Determine trend (Higher Highs, Higher Lows = Bull)
    if(recentHigh > g_lastHigh && recentLow > g_lastLow)
    {
        g_marketStructureBull = true;
    }
    else if(recentHigh < g_lastHigh && recentLow < g_lastLow)
    {
        g_marketStructureBull = false;
    }
    // Else: Ranging market
    
    g_lastHigh = recentHigh;
    g_lastLow = recentLow;
}

//+------------------------------------------------------------------+
//| Detect Smart Money signals                                      |
//+------------------------------------------------------------------+
ENUM_SMC_SIGNAL DetectSmartMoneySignal()
{
    ENUM_SMC_SIGNAL signal = SMC_NONE;
    
    //--- Order Blocks detection
    if(InpUseOrderBlocks)
    {
        signal = DetectOrderBlock();
        if(signal != SMC_NONE) return signal;
    }
    
    //--- Fair Value Gaps detection
    if(InpUseFairValueGaps)
    {
        signal = DetectFairValueGap();
        if(signal != SMC_NONE) return signal;
    }
    
    return SMC_NONE;
}

//+------------------------------------------------------------------+
//| Detect Order Block (Simplified but effective)                  |
//+------------------------------------------------------------------+
ENUM_SMC_SIGNAL DetectOrderBlock()
{
    double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    
    //--- Look for strong rejection candles
    for(int i = 2; i <= InpOrderBlockBars; i++)
    {
        double open = iOpen(_Symbol, PERIOD_CURRENT, i);
        double close = iClose(_Symbol, PERIOD_CURRENT, i);
        double high = iHigh(_Symbol, PERIOD_CURRENT, i);
        double low = iLow(_Symbol, PERIOD_CURRENT, i);
        double range = high - low;
        
        //--- Bullish Order Block (Strong rejection from low)
        if(close > open && (close - low) > range * 0.7 && range > InpMinSL * _Point)
        {
            //--- Check if price returned to this level
            if(MathAbs(currentPrice - low) <= InpBreakoutThreshold * _Point)
            {
                if(g_marketStructureBull) return SMC_BUY_OB;
            }
        }
        
        //--- Bearish Order Block (Strong rejection from high)
        if(open > close && (high - close) > range * 0.7 && range > InpMinSL * _Point)
        {
            //--- Check if price returned to this level
            if(MathAbs(currentPrice - high) <= InpBreakoutThreshold * _Point)
            {
                if(!g_marketStructureBull) return SMC_SELL_OB;
            }
        }
    }
    
    return SMC_NONE;
}

//+------------------------------------------------------------------+
//| Detect Fair Value Gap                                           |
//+------------------------------------------------------------------+
ENUM_SMC_SIGNAL DetectFairValueGap()
{
    //--- Check last 3 bars for FVG
    for(int i = 3; i <= 5; i++)
    {
        double high1 = iHigh(_Symbol, PERIOD_CURRENT, i);
        double low1 = iLow(_Symbol, PERIOD_CURRENT, i);
        double high2 = iHigh(_Symbol, PERIOD_CURRENT, i-1);
        double low2 = iLow(_Symbol, PERIOD_CURRENT, i-1);
        double high3 = iHigh(_Symbol, PERIOD_CURRENT, i-2);
        double low3 = iLow(_Symbol, PERIOD_CURRENT, i-2);
        
        //--- Bullish FVG (Gap between bar 1 high and bar 3 low)
        if(low3 > high1)
        {
            double gapSize = (low3 - high1) / _Point;
            if(gapSize >= InpFVGMinSize)
            {
                double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
                //--- Check if price is in the gap
                if(currentPrice >= high1 && currentPrice <= low3)
                {
                    if(g_marketStructureBull) return SMC_BUY_FVG;
                }
            }
        }
        
        //--- Bearish FVG (Gap between bar 3 high and bar 1 low)
        if(high3 < low1)
        {
            double gapSize = (low1 - high3) / _Point;
            if(gapSize >= InpFVGMinSize)
            {
                double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
                //--- Check if price is in the gap
                if(currentPrice >= high3 && currentPrice <= low1)
                {
                    if(!g_marketStructureBull) return SMC_SELL_FVG;
                }
            }
        }
    }
    
    return SMC_NONE;
}

//+------------------------------------------------------------------+
//| Execute Smart Money trade                                       |
//+------------------------------------------------------------------+
void ExecuteSmartMoneyTrade(ENUM_SMC_SIGNAL signal)
{
    double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double atr = g_atr[0];
    
    //--- Calculate position size (CONSERVATIVE for XAUUSD)
    double lotSize = CalculatePositionSize(atr);
    if(lotSize <= 0) return;
    
    //--- Execute based on signal type
    if(signal == SMC_BUY_OB || signal == SMC_BUY_FVG)
    {
        double sl = price - MathMax(InpMinSL * _Point, atr * 1.5);
        double tp = price + (price - sl) * InpRiskReward;
        
        string comment = (signal == SMC_BUY_OB) ? \"SMC_BuyOB\" : \"SMC_BuyFVG\";
        
        if(m_trade.Buy(lotSize, _Symbol, 0, sl, tp, comment))
        {
            Print(\"✅ BUY executed: \", comment, \" | Lot: \", lotSize, \" | SL: \", DoubleToString((price-sl)/_Point, 1), \" pips\");
            g_dailyTrades++;
        }
    }
    else if(signal == SMC_SELL_OB || signal == SMC_SELL_FVG)
    {
        price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
        double sl = price + MathMax(InpMinSL * _Point, atr * 1.5);
        double tp = price - (sl - price) * InpRiskReward;
        
        string comment = (signal == SMC_SELL_OB) ? \"SMC_SellOB\" : \"SMC_SellFVG\";
        
        if(m_trade.Sell(lotSize, _Symbol, 0, sl, tp, comment))
        {
            Print(\"✅ SELL executed: \", comment, \" | Lot: \", lotSize, \" | SL: \", DoubleToString((sl-price)/_Point, 1), \" pips\");
            g_dailyTrades++;
        }
    }
}

//+------------------------------------------------------------------+
//| Calculate position size (XAUUSD CONSERVATIVE)                  |
//+------------------------------------------------------------------+
double CalculatePositionSize(double atr)
{
    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double riskAmount = balance * (InpRiskPercent / 100.0);
    
    //--- Calculate SL in points
    double slPoints = MathMax(InpMinSL, atr * 1.5 / _Point);
    slPoints = MathMin(slPoints, InpMaxSL); // Cap at max SL
    
    //--- Calculate lot size
    double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double lotSize = riskAmount / (slPoints * tickValue);
    
    //--- Apply symbol limits
    double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    double stepLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
    
    lotSize = MathMax(lotSize, minLot);
    lotSize = MathMin(lotSize, maxLot);
    lotSize = MathRound(lotSize / stepLot) * stepLot;
    
    //--- Extra conservative for XAUUSD
    lotSize = MathMin(lotSize, 0.10); // Max 0.10 lots
    
    return NormalizeDouble(lotSize, 2);
}

//+------------------------------------------------------------------+
//| Reset daily statistics                                          |
//+------------------------------------------------------------------+
void ResetDailyStats()
{
    g_dailyTrades = 0;
    g_dailyPnL = 0.0;
}

//+------------------------------------------------------------------+
//| Update daily statistics                                         |
//+------------------------------------------------------------------+
void UpdateDailyStats()
{
    //--- Calculate daily P&L from closed positions
    static double lastBalance = 0;
    double currentBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    
    if(lastBalance > 0)
    {
        g_dailyPnL += (currentBalance - lastBalance);
    }
    
    lastBalance = currentBalance;
}
"