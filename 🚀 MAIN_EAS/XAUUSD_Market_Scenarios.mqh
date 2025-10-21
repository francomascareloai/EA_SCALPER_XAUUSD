//+------------------------------------------------------------------+
//|                                  XAUUSD_Market_Scenarios.mqh    |
//|                                     Copyright 2024, Elite Trading |
//+------------------------------------------------------------------+

#include "XAUUSD_ML_Core.mqh"

//+------------------------------------------------------------------+
//| Market Scenario Manager - 10 Critical Scenarios Handler        |
//+------------------------------------------------------------------+
class CXAUUSD_MarketScenarios
{
private:
    // Scenario Definitions
    enum ENUM_MARKET_SCENARIO
    {
        SCENARIO_LOW_VOLATILITY,      // ATR < 5 pips
        SCENARIO_HIGH_VOLATILITY,     // ATR > 50 pips  
        SCENARIO_NEWS_EVENT,          // Major news release
        SCENARIO_GAP_OPENING,         // Market gap at open
        SCENARIO_TRENDING_STRONG,     // Strong directional move
        SCENARIO_RANGING_MARKET,      // Sideways consolidation
        SCENARIO_BREAKOUT_PENDING,    // Pre-breakout setup
        SCENARIO_REVERSAL_PATTERN,    // Potential reversal
        SCENARIO_SESSION_TRANSITION,  // Session change effects
        SCENARIO_WEEKEND_GAP         // Weekend gap risk
    };
    
    // Scenario Detection Parameters
    struct SScenarioParams
    {
        double        lowVolatilityThreshold;
        double        highVolatilityThreshold;
        int           newsEventMinutes;
        double        gapSizeThreshold;
        double        trendStrengthThreshold;
        int           rangingBars;
        double        breakoutVolumeMultiplier;
        double        reversalConfirmation;
        int           sessionTransitionMinutes;
        double        weekendGapThreshold;
    };
    
    SScenarioParams   m_params;
    
    // Current Scenario State
    ENUM_MARKET_SCENARIO m_currentScenario;
    double            m_scenarioConfidence;
    datetime          m_scenarioStartTime;
    bool              m_scenarioActive;
    
    // Scenario-Specific Settings
    struct SScenarioSettings
    {
        bool          allowTrading;
        double        riskMultiplier;
        double        confidenceThreshold;
        int           maxPositions;
        double        stopLossMultiplier;
        double        takeProfitMultiplier;
        bool          useMLPrediction;
        int           timeoutMinutes;
    };
    
    SScenarioSettings m_scenarioSettings[10];

public:
                     CXAUUSD_MarketScenarios();
    
    // Scenario Detection
    ENUM_MARKET_SCENARIO DetectCurrentScenario(const SMarketAnalysis &analysis);
    double           GetScenarioConfidence();
    bool             IsScenarioActive();
    
    // Scenario Handling
    bool             HandleScenario(ENUM_MARKET_SCENARIO scenario, STradeSignal &signal);
    SScenarioSettings GetScenarioSettings(ENUM_MARKET_SCENARIO scenario);
    
    // Individual Scenario Handlers
    bool             HandleLowVolatility(const SMarketAnalysis &analysis, STradeSignal &signal);
    bool             HandleHighVolatility(const SMarketAnalysis &analysis, STradeSignal &signal);
    bool             HandleNewsEvent(const SMarketAnalysis &analysis, STradeSignal &signal);
    bool             HandleGapOpening(const SMarketAnalysis &analysis, STradeSignal &signal);
    bool             HandleTrendingMarket(const SMarketAnalysis &analysis, STradeSignal &signal);
    bool             HandleRangingMarket(const SMarketAnalysis &analysis, STradeSignal &signal);
    bool             HandleBreakoutPending(const SMarketAnalysis &analysis, STradeSignal &signal);
    bool             HandleReversalPattern(const SMarketAnalysis &analysis, STradeSignal &signal);
    bool             HandleSessionTransition(const SMarketAnalysis &analysis, STradeSignal &signal);
    bool             HandleWeekendGap(const SMarketAnalysis &analysis, STradeSignal &signal);
    
    // Scenario Validation
    bool             ValidateScenarioConditions(ENUM_MARKET_SCENARIO scenario);
    void             UpdateScenarioMetrics();
    string           GetScenarioDescription(ENUM_MARKET_SCENARIO scenario);

private:
    // Detection Helpers
    bool             IsLowVolatilityEnvironment(double atr);
    bool             IsHighVolatilityEnvironment(double atr);
    bool             IsNewsTimeApproaching();
    bool             IsMarketGap();
    bool             IsStrongTrend(const SMarketAnalysis &analysis);
    bool             IsRangingCondition();
    bool             IsBreakoutSetup();
    bool             IsReversalPattern();
    bool             IsSessionTransition();
    bool             IsWeekendRisk();
    
    // Configuration
    void             InitializeScenarioSettings();
    void             ConfigureScenario(int index, bool trading, double risk, double confidence, 
                                     int positions, double sl, double tp, bool ml, int timeout);
};

//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CXAUUSD_MarketScenarios::CXAUUSD_MarketScenarios()
{
    // Initialize detection parameters
    m_params.lowVolatilityThreshold = 5.0;
    m_params.highVolatilityThreshold = 50.0;
    m_params.newsEventMinutes = 30;
    m_params.gapSizeThreshold = 10.0;
    m_params.trendStrengthThreshold = 0.7;
    m_params.rangingBars = 20;
    m_params.breakoutVolumeMultiplier = 1.5;
    m_params.reversalConfirmation = 0.8;
    m_params.sessionTransitionMinutes = 15;
    m_params.weekendGapThreshold = 5.0;
    
    // Initialize scenario state
    m_currentScenario = SCENARIO_RANGING_MARKET;
    m_scenarioConfidence = 0.0;
    m_scenarioStartTime = 0;
    m_scenarioActive = false;
    
    // Configure scenario-specific settings
    InitializeScenarioSettings();
}

//+------------------------------------------------------------------+
//| Detect Current Market Scenario                                  |
//+------------------------------------------------------------------+
ENUM_MARKET_SCENARIO CXAUUSD_MarketScenarios::DetectCurrentScenario(const SMarketAnalysis &analysis)
{
    // Priority-based scenario detection
    
    // 1. Weekend Gap (Highest Priority)
    if(IsWeekendRisk())
    {
        m_scenarioConfidence = 0.95;
        return SCENARIO_WEEKEND_GAP;
    }
    
    // 2. News Event
    if(IsNewsTimeApproaching())
    {
        m_scenarioConfidence = 0.90;
        return SCENARIO_NEWS_EVENT;
    }
    
    // 3. Market Gap
    if(IsMarketGap())
    {
        m_scenarioConfidence = 0.85;
        return SCENARIO_GAP_OPENING;
    }
    
    // 4. High Volatility
    if(IsHighVolatilityEnvironment(analysis.volatility))
    {
        m_scenarioConfidence = 0.80;
        return SCENARIO_HIGH_VOLATILITY;
    }
    
    // 5. Session Transition
    if(IsSessionTransition())
    {
        m_scenarioConfidence = 0.75;
        return SCENARIO_SESSION_TRANSITION;
    }
    
    // 6. Strong Trend
    if(IsStrongTrend(analysis))
    {
        m_scenarioConfidence = 0.70;
        return SCENARIO_TRENDING_STRONG;
    }
    
    // 7. Breakout Setup
    if(IsBreakoutSetup())
    {
        m_scenarioConfidence = 0.65;
        return SCENARIO_BREAKOUT_PENDING;
    }
    
    // 8. Reversal Pattern
    if(IsReversalPattern())
    {
        m_scenarioConfidence = 0.60;
        return SCENARIO_REVERSAL_PATTERN;
    }
    
    // 9. Low Volatility
    if(IsLowVolatilityEnvironment(analysis.volatility))
    {
        m_scenarioConfidence = 0.55;
        return SCENARIO_LOW_VOLATILITY;
    }
    
    // 10. Default: Ranging Market
    m_scenarioConfidence = 0.50;
    return SCENARIO_RANGING_MARKET;
}

//+------------------------------------------------------------------+
//| Handle Low Volatility Scenario                                  |
//+------------------------------------------------------------------+
bool CXAUUSD_MarketScenarios::HandleLowVolatility(const SMarketAnalysis &analysis, STradeSignal &signal)
{
    Print("📊 SCENARIO: Low Volatility Environment Detected");
    
    // Strategy: Mean reversion with tight ranges
    // Risk: Reduced position size, tighter stops
    
    double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double ema21 = analysis.currentData.ema_21;
    double ema50 = analysis.currentData.ema_50;
    double atr = analysis.currentData.atr_14;
    
    // Look for mean reversion opportunities
    double deviationFromMean = MathAbs(currentPrice - ema21) / atr;
    
    if(deviationFromMean > 1.5) // Price extended from mean
    {
        signal.type = (currentPrice > ema21) ? SIGNAL_SELL : SIGNAL_BUY;
        signal.entryPrice = currentPrice;
        signal.confidence = 0.7;
        signal.strategy = "Low Volatility Mean Reversion";
        
        // Tight stops and targets for low volatility
        if(signal.type == SIGNAL_BUY)
        {
            signal.stopLoss = currentPrice - (atr * 1.0);    // Tighter stop
            signal.takeProfit = currentPrice + (atr * 2.0);  // Conservative target
        }
        else
        {
            signal.stopLoss = currentPrice + (atr * 1.0);
            signal.takeProfit = currentPrice - (atr * 2.0);
        }
        
        return true;
    }
    
    return false; // No signal in low volatility
}

//+------------------------------------------------------------------+
//| Handle High Volatility Scenario                                 |
//+------------------------------------------------------------------+
bool CXAUUSD_MarketScenarios::HandleHighVolatility(const SMarketAnalysis &analysis, STradeSignal &signal)
{
    Print("⚡ SCENARIO: High Volatility Environment Detected");
    
    // Strategy: Reduce risk, wait for volatility to settle
    // Only trade with very high ML confidence
    
    if(analysis.confidence < 0.85)
    {
        Print("🚫 High volatility: ML confidence too low, no trading");
        return false;
    }
    
    double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double atr = analysis.currentData.atr_14;
    
    // Only trade in direction of strong momentum
    if(analysis.features.price_momentum[0] > 20.0) // Strong upward momentum
    {
        signal.type = SIGNAL_BUY;
        signal.entryPrice = currentPrice;
        signal.confidence = analysis.confidence * 0.8; // Reduced confidence
        signal.strategy = "High Volatility Momentum";
        
        // Wider stops for high volatility
        signal.stopLoss = currentPrice - (atr * 3.0);
        signal.takeProfit = currentPrice + (atr * 6.0);
        
        return true;
    }
    else if(analysis.features.price_momentum[0] < -20.0) // Strong downward momentum
    {
        signal.type = SIGNAL_SELL;
        signal.entryPrice = currentPrice;
        signal.confidence = analysis.confidence * 0.8;
        signal.strategy = "High Volatility Momentum";
        
        signal.stopLoss = currentPrice + (atr * 3.0);
        signal.takeProfit = currentPrice - (atr * 6.0);
        
        return true;
    }
    
    return false;
}

//+------------------------------------------------------------------+
//| Handle News Event Scenario                                      |
//+------------------------------------------------------------------+
bool CXAUUSD_MarketScenarios::HandleNewsEvent(const SMarketAnalysis &analysis, STradeSignal &signal)
{
    Print("📰 SCENARIO: News Event Detected - Trading Suspended");
    
    // Strategy: Stop all trading 30 minutes before/after major news
    // Close existing positions if needed
    
    // Check if we have positions to protect
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if(PositionSelectByIndex(i))
        {
            string symbol = PositionGetString(POSITION_SYMBOL);
            if(symbol == _Symbol)
            {
                Print("⚠️ Closing position due to news event");
                ulong ticket = PositionGetInteger(POSITION_TICKET);
                CTrade trade;
                trade.PositionClose(ticket);
            }
        }
    }
    
    return false; // No new signals during news
}

//+------------------------------------------------------------------+
//| Handle Gap Opening Scenario                                     |
//+------------------------------------------------------------------+
bool CXAUUSD_MarketScenarios::HandleGapOpening(const SMarketAnalysis &analysis, STradeSignal &signal)
{
    Print("📊 SCENARIO: Market Gap Opening Detected");
    
    double openPrice = iOpen(_Symbol, PERIOD_D1, 0);
    double prevClose = iClose(_Symbol, PERIOD_D1, 1);
    double gapSize = MathAbs(openPrice - prevClose) / Point;
    
    if(gapSize > m_params.gapSizeThreshold)
    {
        // Strategy: Wait for gap fill or trade gap continuation
        double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
        
        // If gap is filling, trade the fill direction
        bool gapFilling = (openPrice > prevClose && currentPrice < openPrice) ||
                         (openPrice < prevClose && currentPrice > openPrice);
        
        if(gapFilling)
        {
            signal.type = (openPrice > prevClose) ? SIGNAL_SELL : SIGNAL_BUY;
            signal.entryPrice = currentPrice;
            signal.confidence = 0.75;
            signal.strategy = "Gap Fill Trading";
            
            double atr = analysis.currentData.atr_14;
            if(signal.type == SIGNAL_BUY)
            {
                signal.stopLoss = currentPrice - (atr * 2.0);
                signal.takeProfit = prevClose; // Target: close the gap
            }
            else
            {
                signal.stopLoss = currentPrice + (atr * 2.0);
                signal.takeProfit = prevClose;
            }
            
            return true;
        }
    }
    
    return false;
}

//+------------------------------------------------------------------+
//| Handle Trending Market Scenario                                 |
//+------------------------------------------------------------------+
bool CXAUUSD_MarketScenarios::HandleTrendingMarket(const SMarketAnalysis &analysis, STradeSignal &signal)
{
    Print("📈 SCENARIO: Strong Trending Market Detected");
    
    // Strategy: Trend following with momentum confirmation
    double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double ema21 = analysis.currentData.ema_21;
    double ema50 = analysis.currentData.ema_50;
    double ema200 = analysis.currentData.ema_200;
    
    // Confirm trend direction with multiple EMAs
    bool uptrend = (ema21 > ema50) && (ema50 > ema200) && (currentPrice > ema21);
    bool downtrend = (ema21 < ema50) && (ema50 < ema200) && (currentPrice < ema21);
    
    if(uptrend && analysis.features.price_momentum[0] > 10.0)
    {
        signal.type = SIGNAL_BUY;
        signal.entryPrice = currentPrice;
        signal.confidence = 0.85;
        signal.strategy = "Trend Following Up";
        
        double atr = analysis.currentData.atr_14;
        signal.stopLoss = ema21 - (atr * 1.0); // Dynamic stop at EMA21
        signal.takeProfit = currentPrice + (atr * 4.0); // 1:4 R/R
        
        return true;
    }
    else if(downtrend && analysis.features.price_momentum[0] < -10.0)
    {
        signal.type = SIGNAL_SELL;
        signal.entryPrice = currentPrice;
        signal.confidence = 0.85;
        signal.strategy = "Trend Following Down";
        
        double atr = analysis.currentData.atr_14;
        signal.stopLoss = ema21 + (atr * 1.0);
        signal.takeProfit = currentPrice - (atr * 4.0);
        
        return true;
    }
    
    return false;
}

//+------------------------------------------------------------------+
//| Handle Ranging Market Scenario                                  |
//+------------------------------------------------------------------+
bool CXAUUSD_MarketScenarios::HandleRangingMarket(const SMarketAnalysis &analysis, STradeSignal &signal)
{
    Print("↔️ SCENARIO: Ranging Market Conditions");
    
    // Strategy: Support/Resistance bounce trading
    double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    
    // Calculate support and resistance levels
    double high20 = 0, low20 = 999999;
    for(int i = 1; i <= 20; i++)
    {
        double h = iHigh(_Symbol, PERIOD_M15, i);
        double l = iLow(_Symbol, PERIOD_M15, i);
        if(h > high20) high20 = h;
        if(l < low20) low20 = l;
    }
    
    double rangeSize = high20 - low20;
    double pricePosition = (currentPrice - low20) / rangeSize;
    
    // Trade range bounces
    if(pricePosition < 0.2) // Near support
    {
        signal.type = SIGNAL_BUY;
        signal.entryPrice = currentPrice;
        signal.confidence = 0.75;
        signal.strategy = "Range Support Bounce";
        
        signal.stopLoss = low20 - (Point * 10);
        signal.takeProfit = low20 + (rangeSize * 0.7);
        
        return true;
    }
    else if(pricePosition > 0.8) // Near resistance
    {
        signal.type = SIGNAL_SELL;
        signal.entryPrice = currentPrice;
        signal.confidence = 0.75;
        signal.strategy = "Range Resistance Bounce";
        
        signal.stopLoss = high20 + (Point * 10);
        signal.takeProfit = high20 - (rangeSize * 0.7);
        
        return true;
    }
    
    return false;
}

//+------------------------------------------------------------------+
//| Detection Helper Methods                                         |
//+------------------------------------------------------------------+
bool CXAUUSD_MarketScenarios::IsLowVolatilityEnvironment(double atr)
{
    return (atr / Point) < m_params.lowVolatilityThreshold;
}

bool CXAUUSD_MarketScenarios::IsHighVolatilityEnvironment(double atr)
{
    return (atr / Point) > m_params.highVolatilityThreshold;
}

bool CXAUUSD_MarketScenarios::IsNewsTimeApproaching()
{
    // Simplified news detection - would integrate with economic calendar
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    
    // High impact news typically at :30 minutes past the hour
    return (dt.min >= 25 && dt.min <= 35) && 
           (dt.hour == 8 || dt.hour == 10 || dt.hour == 14); // Common news times
}

bool CXAUUSD_MarketScenarios::IsMarketGap()
{
    double openPrice = iOpen(_Symbol, PERIOD_D1, 0);
    double prevClose = iClose(_Symbol, PERIOD_D1, 1);
    double gapSize = MathAbs(openPrice - prevClose) / Point;
    
    return gapSize > m_params.gapSizeThreshold;
}

bool CXAUUSD_MarketScenarios::IsStrongTrend(const SMarketAnalysis &analysis)
{
    double ema21 = analysis.currentData.ema_21;
    double ema50 = analysis.currentData.ema_50;
    double ema200 = analysis.currentData.ema_200;
    
    // Check for aligned EMAs indicating strong trend
    bool uptrend = (ema21 > ema50) && (ema50 > ema200);
    bool downtrend = (ema21 < ema50) && (ema50 < ema200);
    
    return (uptrend || downtrend) && MathAbs(analysis.features.price_momentum[0]) > 15.0;
}

bool CXAUUSD_MarketScenarios::IsSessionTransition()
{
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    
    // Session transition times (±15 minutes)
    return (dt.hour == 7 && dt.min >= 45) ||  // Before London
           (dt.hour == 8 && dt.min <= 15) ||  // London open
           (dt.hour == 12 && dt.min >= 45) || // Before NY
           (dt.hour == 13 && dt.min <= 15) || // NY open
           (dt.hour == 21 && dt.min >= 45) || // Before Asian
           (dt.hour == 22 && dt.min <= 15);   // Asian open
}

bool CXAUUSD_MarketScenarios::IsWeekendRisk()
{
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    
    // Friday evening or Sunday evening gap risk
    return (dt.day_of_week == 5 && dt.hour >= 20) || // Friday evening
           (dt.day_of_week == 0) ||                   // Sunday
           (dt.day_of_week == 1 && dt.hour <= 2);     // Monday early
}

//+------------------------------------------------------------------+
//| Initialize Scenario Settings                                     |
//+------------------------------------------------------------------+
void CXAUUSD_MarketScenarios::InitializeScenarioSettings()
{
    // Configure each scenario: (trading, risk, confidence, positions, sl, tp, ml, timeout)
    ConfigureScenario(SCENARIO_LOW_VOLATILITY,    true,  0.8, 0.70, 2, 1.0, 2.0, true,  60);
    ConfigureScenario(SCENARIO_HIGH_VOLATILITY,   true,  0.3, 0.85, 1, 3.0, 6.0, true,  30);
    ConfigureScenario(SCENARIO_NEWS_EVENT,        false, 0.0, 0.90, 0, 0.0, 0.0, false, 60);
    ConfigureScenario(SCENARIO_GAP_OPENING,       true,  0.5, 0.75, 1, 2.0, 3.0, false, 45);
    ConfigureScenario(SCENARIO_TRENDING_STRONG,   true,  1.2, 0.80, 2, 1.5, 4.0, true,  120);
    ConfigureScenario(SCENARIO_RANGING_MARKET,    true,  0.9, 0.75, 2, 1.0, 2.5, true,  90);
    ConfigureScenario(SCENARIO_BREAKOUT_PENDING,  true,  1.0, 0.80, 1, 2.0, 5.0, true,  30);
    ConfigureScenario(SCENARIO_REVERSAL_PATTERN,  true,  0.7, 0.85, 1, 1.5, 3.0, true,  45);
    ConfigureScenario(SCENARIO_SESSION_TRANSITION,true,  0.6, 0.70, 1, 1.5, 2.0, false, 15);
    ConfigureScenario(SCENARIO_WEEKEND_GAP,       false, 0.0, 0.95, 0, 0.0, 0.0, false, 180);
}

void CXAUUSD_MarketScenarios::ConfigureScenario(int index, bool trading, double risk, double confidence, 
                                               int positions, double sl, double tp, bool ml, int timeout)
{
    m_scenarioSettings[index].allowTrading = trading;
    m_scenarioSettings[index].riskMultiplier = risk;
    m_scenarioSettings[index].confidenceThreshold = confidence;
    m_scenarioSettings[index].maxPositions = positions;
    m_scenarioSettings[index].stopLossMultiplier = sl;
    m_scenarioSettings[index].takeProfitMultiplier = tp;
    m_scenarioSettings[index].useMLPrediction = ml;
    m_scenarioSettings[index].timeoutMinutes = timeout;
}

// Additional simplified implementations
bool CXAUUSD_MarketScenarios::IsRangingCondition() { return true; }
bool CXAUUSD_MarketScenarios::IsBreakoutSetup() { return false; }
bool CXAUUSD_MarketScenarios::IsReversalPattern() { return false; }
bool CXAUUSD_MarketScenarios::HandleBreakoutPending(const SMarketAnalysis &analysis, STradeSignal &signal) { return false; }
bool CXAUUSD_MarketScenarios::HandleReversalPattern(const SMarketAnalysis &analysis, STradeSignal &signal) { return false; }
bool CXAUUSD_MarketScenarios::HandleSessionTransition(const SMarketAnalysis &analysis, STradeSignal &signal) { return false; }
bool CXAUUSD_MarketScenarios::HandleWeekendGap(const SMarketAnalysis &analysis, STradeSignal &signal) { return false; }