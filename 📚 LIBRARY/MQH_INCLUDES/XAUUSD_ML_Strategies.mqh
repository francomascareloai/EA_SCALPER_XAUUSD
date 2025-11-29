//+------------------------------------------------------------------+
//|                                     XAUUSD_ML_Strategies.mqh     |
//|                                     Copyright 2024, Elite Trading |
//+------------------------------------------------------------------+

#include "XAUUSD_ML_Core.mqh"

//+------------------------------------------------------------------+
//| Advanced XAUUSD Trading Strategies Class                        |
//+------------------------------------------------------------------+
class CXAUUSD_MLStrategies
{
private:
    // Strategy Configuration
    bool              m_smartMoneyEnabled;
    bool              m_mlScalpingEnabled;
    bool              m_volatilityBreakoutEnabled;
    bool              m_multiTimeframeEnabled;
    
    // ICT Smart Money Variables
    double            m_orderBlockStrength;
    double            m_liquidityZoneDistance;
    double            m_fairValueGap;
    bool              m_institutionalBias;
    
    // ML Scalping Variables
    int               m_scalpingTimeframe;
    double            m_scalpingVolatilityMin;
    double            m_scalpingVolatilityMax;
    int               m_maxScalpingTrades;
    
    // Volatility Breakout Variables
    double            m_breakoutThreshold;
    int               m_consolidationPeriod;
    double            m_volumeMultiplier;
    
    // Multi-Timeframe Analysis
    ENUM_TIMEFRAMES   m_primaryTF;
    ENUM_TIMEFRAMES   m_confirmationTF;
    ENUM_TIMEFRAMES   m_trendTF;
    
    // Performance Tracking
    struct SStrategyPerformance
    {
        int           totalTrades;
        int           winningTrades;
        double        totalProfit;
        double        maxDrawdown;
        double        avgRRR;
        datetime      lastUpdate;
    };
    
    SStrategyPerformance m_smartMoneyPerf;
    SStrategyPerformance m_scalpingPerf;
    SStrategyPerformance m_breakoutPerf;

public:
                     CXAUUSD_MLStrategies();
                    ~CXAUUSD_MLStrategies();
    
    // Configuration
    void             EnableSmartMoney(bool enabled);
    void             EnableMLScalping(bool enabled);
    void             EnableVolatilityBreakout(bool enabled);
    void             EnableMultiTimeframe(bool enabled);
    
    // Strategy Selection
    ENUM_STRATEGY_TYPE SelectOptimalStrategy(const SMarketAnalysis &analysis);
    double           CalculateStrategyConfidence(ENUM_STRATEGY_TYPE strategy, const SMarketAnalysis &analysis);
    
    // Signal Generation
    bool             GenerateSmartMoneySignal(const SMarketAnalysis &analysis, STradeSignal &signal);
    bool             GenerateMLScalpingSignal(const SMarketAnalysis &analysis, STradeSignal &signal);
    bool             GenerateVolatilityBreakoutSignal(const SMarketAnalysis &analysis, STradeSignal &signal);
    
    // Position Management
    bool             ManagePosition(ulong ticket);
    bool             ShouldClosePosition(ulong ticket, const SMarketAnalysis &analysis);
    
    // Performance Analysis
    void             UpdateStrategyPerformance(ENUM_STRATEGY_TYPE strategy, bool isWin, double profit);
    double           GetStrategyWinRate(ENUM_STRATEGY_TYPE strategy);

private:
    // ICT Smart Money Concepts Implementation
    bool             DetectOrderBlocks(double &strength);
    bool             IdentifyLiquidityZones(double &distance);
    bool             ValidateInstitutionalFlow();
    double           CalculateFairValueGap();
    bool             DetectBreakOfStructure();
    bool             ValidateOrderBlockEntry(const SMarketAnalysis &analysis);
    
    // ML Scalping Implementation
    bool             ValidateScalpingConditions(const SMarketAnalysis &analysis);
    double           PredictShortTermMovement(const SMarketAnalysis &analysis);
    bool             IsOptimalScalpingTime();
    double           CalculateScalpingStopLoss(double entryPrice, ENUM_SIGNAL_TYPE signalType);
    double           CalculateScalpingTakeProfit(double entryPrice, ENUM_SIGNAL_TYPE signalType);
    
    // Volatility Breakout Implementation
    bool             DetectVolatilityExpansion(const SMarketAnalysis &analysis);
    bool             ValidateBreakoutStrength(const SMarketAnalysis &analysis);
    double           CalculateBreakoutTarget(double entryPrice, ENUM_SIGNAL_TYPE signalType);
    bool             IsConsolidationPhase();
    
    // Multi-Timeframe Analysis
    bool             ConfirmTrendDirection(ENUM_SIGNAL_TYPE signal);
    bool             ValidateMultiTimeframeAlignment(ENUM_SIGNAL_TYPE signal);
    double           GetHigherTimeframeTrend();
    
    // Market Structure Analysis
    double           AnalyzeSupportResistance(double price);
    bool             IsAtKeyLevel(double price);
    double           CalculateLevelStrength(double level);
    
    // Session Analysis
    bool             IsLondonSession();
    bool             IsNewYorkSession();
    bool             IsAsianSession();
    bool             IsSessionOverlap();
    double           GetSessionVolatilityMultiplier();
};

//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CXAUUSD_MLStrategies::CXAUUSD_MLStrategies()
{
    m_smartMoneyEnabled = true;
    m_mlScalpingEnabled = true;
    m_volatilityBreakoutEnabled = true;
    m_multiTimeframeEnabled = true;
    
    m_scalpingTimeframe = PERIOD_M15;
    m_scalpingVolatilityMin = 5.0;
    m_scalpingVolatilityMax = 25.0;
    m_maxScalpingTrades = 5;
    
    m_breakoutThreshold = 1.5;
    m_consolidationPeriod = 20;
    m_volumeMultiplier = 1.5;
    
    m_primaryTF = PERIOD_M15;
    m_confirmationTF = PERIOD_H1;
    m_trendTF = PERIOD_H4;
    
    // Initialize performance tracking
    ZeroMemory(m_smartMoneyPerf);
    ZeroMemory(m_scalpingPerf);
    ZeroMemory(m_breakoutPerf);
}

//+------------------------------------------------------------------+
//| Select Optimal Strategy                                          |
//+------------------------------------------------------------------+
ENUM_STRATEGY_TYPE CXAUUSD_MLStrategies::SelectOptimalStrategy(const SMarketAnalysis &analysis)
{
    double smartMoneyConf = 0.0;
    double scalpingConf = 0.0;
    double breakoutConf = 0.0;
    
    // Calculate confidence for each strategy
    if(m_smartMoneyEnabled)
    {
        smartMoneyConf = CalculateStrategyConfidence(STRATEGY_SMART_MONEY, analysis);
    }
    
    if(m_mlScalpingEnabled && ValidateScalpingConditions(analysis))
    {
        scalpingConf = CalculateStrategyConfidence(STRATEGY_ML_SCALPING, analysis);
    }
    
    if(m_volatilityBreakoutEnabled)
    {
        breakoutConf = CalculateStrategyConfidence(STRATEGY_VOLATILITY_BREAKOUT, analysis);
    }
    
    // Select strategy with highest confidence
    if(smartMoneyConf >= scalpingConf && smartMoneyConf >= breakoutConf && smartMoneyConf > 0.7)
    {
        return STRATEGY_SMART_MONEY;
    }
    else if(scalpingConf >= smartMoneyConf && scalpingConf >= breakoutConf && scalpingConf > 0.7)
    {
        return STRATEGY_ML_SCALPING;
    }
    else if(breakoutConf >= smartMoneyConf && breakoutConf >= scalpingConf && breakoutConf > 0.7)
    {
        return STRATEGY_VOLATILITY_BREAKOUT;
    }
    
    // Default to mean reversion in ranging markets
    if(analysis.regime == REGIME_RANGING)
    {
        return STRATEGY_MEAN_REVERSION;
    }
    
    // No strategy selected
    return STRATEGY_SMART_MONEY; // Default fallback
}

//+------------------------------------------------------------------+
//| Generate Smart Money Signal                                      |
//+------------------------------------------------------------------+
bool CXAUUSD_MLStrategies::GenerateSmartMoneySignal(const SMarketAnalysis &analysis, STradeSignal &signal)
{
    if(!m_smartMoneyEnabled) return false;
    
    // ICT Smart Money Concepts Analysis
    double orderBlockStrength;
    if(!DetectOrderBlocks(orderBlockStrength)) return false;
    
    double liquidityDistance;
    if(!IdentifyLiquidityZones(liquidityDistance)) return false;
    
    if(!ValidateInstitutionalFlow()) return false;
    
    // Check for Break of Structure (BOS)
    if(!DetectBreakOfStructure()) return false;
    
    // Validate entry conditions
    if(!ValidateOrderBlockEntry(analysis)) return false;
    
    // Multi-timeframe confirmation
    if(m_multiTimeframeEnabled)
    {
        double htfTrend = GetHigherTimeframeTrend();
        if(MathAbs(htfTrend) < 0.3) return false; // No clear trend
    }
    
    // Generate signal
    double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double atr = analysis.currentData.atr_14;
    
    // Determine signal direction based on institutional flow and order blocks
    ENUM_SIGNAL_TYPE signalType = SIGNAL_HOLD;
    
    if(m_institutionalBias && orderBlockStrength > 0.7)
    {
        signalType = SIGNAL_BUY;
    }
    else if(!m_institutionalBias && orderBlockStrength > 0.7)
    {
        signalType = SIGNAL_SELL;
    }
    else
    {
        return false;
    }
    
    // Calculate entry, stop loss, and take profit
    signal.type = signalType;
    signal.entryPrice = currentPrice;
    signal.confidence = orderBlockStrength * 0.4 + (1.0 - liquidityDistance) * 0.3 + analysis.confidence * 0.3;
    signal.strategy = "ICT Smart Money";
    signal.timestamp = TimeCurrent();
    
    if(signalType == SIGNAL_BUY)
    {
        signal.stopLoss = currentPrice - (atr * 2.0);
        signal.takeProfit = currentPrice + (atr * 3.0);
    }
    else
    {
        signal.stopLoss = currentPrice + (atr * 2.0);
        signal.takeProfit = currentPrice - (atr * 3.0);
    }
    
    return signal.confidence > 0.75;
}

//+------------------------------------------------------------------+
//| Generate ML Scalping Signal                                      |
//+------------------------------------------------------------------+
bool CXAUUSD_MLStrategies::GenerateMLScalpingSignal(const SMarketAnalysis &analysis, STradeSignal &signal)
{
    if(!m_mlScalpingEnabled) return false;
    
    // Validate scalping conditions
    if(!ValidateScalpingConditions(analysis)) return false;
    
    // Check optimal scalping time
    if(!IsOptimalScalpingTime()) return false;
    
    // ML prediction for short-term movement
    double shortTermPrediction = PredictShortTermMovement(analysis);
    if(MathAbs(shortTermPrediction) < 0.6) return false;
    
    // Current market state
    double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * Point;
    
    // Ensure spread is acceptable for scalping
    if(spread > analysis.currentData.atr_14 * 0.3) return false;
    
    // Generate signal
    ENUM_SIGNAL_TYPE signalType = (shortTermPrediction > 0.6) ? SIGNAL_BUY : SIGNAL_SELL;
    
    signal.type = signalType;
    signal.entryPrice = currentPrice;
    signal.confidence = MathAbs(shortTermPrediction) * 0.6 + analysis.confidence * 0.4;
    signal.strategy = "ML Adaptive Scalping";
    signal.timestamp = TimeCurrent();
    
    // Tight scalping stops and targets
    signal.stopLoss = CalculateScalpingStopLoss(currentPrice, signalType);
    signal.takeProfit = CalculateScalpingTakeProfit(currentPrice, signalType);
    
    return signal.confidence > 0.78; // Higher threshold for scalping
}

//+------------------------------------------------------------------+
//| Generate Volatility Breakout Signal                             |
//+------------------------------------------------------------------+
bool CXAUUSD_MLStrategies::GenerateVolatilityBreakoutSignal(const SMarketAnalysis &analysis, STradeSignal &signal)
{
    if(!m_volatilityBreakoutEnabled) return false;
    
    // Detect volatility expansion
    if(!DetectVolatilityExpansion(analysis)) return false;
    
    // Validate breakout strength
    if(!ValidateBreakoutStrength(analysis)) return false;
    
    // Check for consolidation phase before breakout
    if(!IsConsolidationPhase()) return false;
    
    // Current market state
    double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double atr = analysis.currentData.atr_14;
    
    // Determine breakout direction
    ENUM_SIGNAL_TYPE signalType = SIGNAL_HOLD;
    
    // Use price action and momentum to determine direction
    if(analysis.currentData.close > analysis.currentData.ema_50 && 
       analysis.features.price_momentum[0] > 10.0)
    {
        signalType = SIGNAL_BUY;
    }
    else if(analysis.currentData.close < analysis.currentData.ema_50 && 
            analysis.features.price_momentum[0] < -10.0)
    {
        signalType = SIGNAL_SELL;
    }
    else
    {
        return false;
    }
    
    // Generate signal
    signal.type = signalType;
    signal.entryPrice = currentPrice;
    signal.confidence = analysis.confidence * 0.5 + (analysis.volatility / 20.0) * 0.5;
    signal.strategy = "Volatility Breakout";
    signal.timestamp = TimeCurrent();
    
    // Wide stops for volatility breakout
    if(signalType == SIGNAL_BUY)
    {
        signal.stopLoss = currentPrice - (atr * 2.5);
        signal.takeProfit = CalculateBreakoutTarget(currentPrice, signalType);
    }
    else
    {
        signal.stopLoss = currentPrice + (atr * 2.5);
        signal.takeProfit = CalculateBreakoutTarget(currentPrice, signalType);
    }
    
    return signal.confidence > 0.70;
}

//+------------------------------------------------------------------+
//| Validate Scalping Conditions                                    |
//+------------------------------------------------------------------+
bool CXAUUSD_MLStrategies::ValidateScalpingConditions(const SMarketAnalysis &analysis)
{
    // Check volatility range
    double volatility = analysis.currentData.atr_14 / Point;
    if(volatility < m_scalpingVolatilityMin || volatility > m_scalpingVolatilityMax)
    {
        return false;
    }
    
    // Check session
    if(!IsLondonSession() && !IsNewYorkSession() && !IsSessionOverlap())
    {
        return false;
    }
    
    // Check spread
    double spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
    if(spread > 5) return false; // Max 5 point spread for scalping
    
    // Check existing positions
    int scalpingPositions = 0;
    for(int i = 0; i < PositionsTotal(); i++)
    {
        if(PositionSelectByIndex(i) && 
           PositionGetString(POSITION_COMMENT) == "ML Adaptive Scalping")
        {
            scalpingPositions++;
        }
    }
    
    return scalpingPositions < m_maxScalpingTrades;
}

//+------------------------------------------------------------------+
//| Detect Order Blocks                                             |
//+------------------------------------------------------------------+
bool CXAUUSD_MLStrategies::DetectOrderBlocks(double &strength)
{
    // Simplified order block detection
    // In real implementation, this would analyze volume and price action
    
    double highs[10], lows[10], volumes[10];
    
    for(int i = 0; i < 10; i++)
    {
        highs[i] = iHigh(_Symbol, m_primaryTF, i);
        lows[i] = iLow(_Symbol, m_primaryTF, i);
        volumes[i] = iVolume(_Symbol, m_primaryTF, i);
    }
    
    // Look for significant volume imbalances
    double avgVolume = 0;
    for(int i = 0; i < 10; i++)
    {
        avgVolume += volumes[i];
    }
    avgVolume /= 10.0;
    
    // Check recent bars for order block pattern
    bool hasOrderBlock = false;
    strength = 0.0;
    
    for(int i = 1; i < 5; i++)
    {
        if(volumes[i] > avgVolume * 1.5) // High volume bar
        {
            double priceRange = highs[i] - lows[i];
            double avgRange = 0;
            
            for(int j = 0; j < 10; j++)
            {
                avgRange += (highs[j] - lows[j]);
            }
            avgRange /= 10.0;
            
            if(priceRange > avgRange * 1.2) // Wide range bar
            {
                hasOrderBlock = true;
                strength = MathMin(1.0, (volumes[i] / avgVolume - 1.0) / 2.0);
                break;
            }
        }
    }
    
    m_orderBlockStrength = strength;
    return hasOrderBlock && strength > 0.5;
}

//+------------------------------------------------------------------+
//| Detect Volatility Expansion                                     |
//+------------------------------------------------------------------+
bool CXAUUSD_MLStrategies::DetectVolatilityExpansion(const SMarketAnalysis &analysis)
{
    double currentATR = analysis.currentData.atr_14;
    
    // Calculate average ATR over last 20 periods
    double atrSum = 0;
    double atrBuffer[20];
    int atrHandle = iATR(_Symbol, m_primaryTF, 14);
    
    if(CopyBuffer(atrHandle, 0, 1, 20, atrBuffer) <= 0)
    {
        IndicatorRelease(atrHandle);
        return false;
    }
    
    for(int i = 0; i < 20; i++)
    {
        atrSum += atrBuffer[i];
    }
    
    double avgATR = atrSum / 20.0;
    IndicatorRelease(atrHandle);
    
    // Check if current volatility is significantly higher
    return (currentATR > avgATR * m_breakoutThreshold);
}

//+------------------------------------------------------------------+
//| Additional helper methods...                                     |
//+------------------------------------------------------------------+
double CXAUUSD_MLStrategies::CalculateStrategyConfidence(ENUM_STRATEGY_TYPE strategy, const SMarketAnalysis &analysis)
{
    double baseConfidence = analysis.confidence;
    double sessionMultiplier = GetSessionVolatilityMultiplier();
    double performanceBonus = 0.0;
    
    // Add performance-based confidence boost
    switch(strategy)
    {
        case STRATEGY_SMART_MONEY:
            performanceBonus = (GetStrategyWinRate(strategy) - 0.5) * 0.2;
            break;
        case STRATEGY_ML_SCALPING:
            performanceBonus = (GetStrategyWinRate(strategy) - 0.5) * 0.15;
            break;
        case STRATEGY_VOLATILITY_BREAKOUT:
            performanceBonus = (GetStrategyWinRate(strategy) - 0.5) * 0.25;
            break;
    }
    
    return MathMin(1.0, baseConfidence * sessionMultiplier + performanceBonus);
}

double CXAUUSD_MLStrategies::GetStrategyWinRate(ENUM_STRATEGY_TYPE strategy)
{
    switch(strategy)
    {
        case STRATEGY_SMART_MONEY:
            return m_smartMoneyPerf.totalTrades > 0 ? 
                   (double)m_smartMoneyPerf.winningTrades / m_smartMoneyPerf.totalTrades : 0.5;
        case STRATEGY_ML_SCALPING:
            return m_scalpingPerf.totalTrades > 0 ? 
                   (double)m_scalpingPerf.winningTrades / m_scalpingPerf.totalTrades : 0.5;
        case STRATEGY_VOLATILITY_BREAKOUT:
            return m_breakoutPerf.totalTrades > 0 ? 
                   (double)m_breakoutPerf.winningTrades / m_breakoutPerf.totalTrades : 0.5;
    }
    return 0.5;
}

// Additional simplified implementations for demo
bool CXAUUSD_MLStrategies::IdentifyLiquidityZones(double &distance) { distance = 0.3; return true; }
bool CXAUUSD_MLStrategies::ValidateInstitutionalFlow() { m_institutionalBias = true; return true; }
double CXAUUSD_MLStrategies::CalculateFairValueGap() { return 0.0; }
bool CXAUUSD_MLStrategies::DetectBreakOfStructure() { return true; }
bool CXAUUSD_MLStrategies::ValidateOrderBlockEntry(const SMarketAnalysis &analysis) { return true; }
double CXAUUSD_MLStrategies::PredictShortTermMovement(const SMarketAnalysis &analysis) { return 0.7; }
bool CXAUUSD_MLStrategies::IsOptimalScalpingTime() { return IsLondonSession() || IsNewYorkSession(); }
double CXAUUSD_MLStrategies::CalculateScalpingStopLoss(double entryPrice, ENUM_SIGNAL_TYPE signalType) 
{ 
    double atr = iATR(_Symbol, PERIOD_M15, 14); 
    return (signalType == SIGNAL_BUY) ? entryPrice - atr : entryPrice + atr; 
}
double CXAUUSD_MLStrategies::CalculateScalpingTakeProfit(double entryPrice, ENUM_SIGNAL_TYPE signalType) 
{ 
    double atr = iATR(_Symbol, PERIOD_M15, 14); 
    return (signalType == SIGNAL_BUY) ? entryPrice + atr * 1.5 : entryPrice - atr * 1.5; 
}
bool CXAUUSD_MLStrategies::ValidateBreakoutStrength(const SMarketAnalysis &analysis) { return true; }
double CXAUUSD_MLStrategies::CalculateBreakoutTarget(double entryPrice, ENUM_SIGNAL_TYPE signalType) 
{ 
    double atr = iATR(_Symbol, PERIOD_M15, 14); 
    return (signalType == SIGNAL_BUY) ? entryPrice + atr * 4.0 : entryPrice - atr * 4.0; 
}
bool CXAUUSD_MLStrategies::IsConsolidationPhase() { return true; }
bool CXAUUSD_MLStrategies::IsLondonSession() 
{ 
    MqlDateTime dt; TimeToStruct(TimeCurrent(), dt); 
    return (dt.hour >= 8 && dt.hour <= 17); 
}
bool CXAUUSD_MLStrategies::IsNewYorkSession() 
{ 
    MqlDateTime dt; TimeToStruct(TimeCurrent(), dt); 
    return (dt.hour >= 13 && dt.hour <= 22); 
}
bool CXAUUSD_MLStrategies::IsAsianSession() 
{ 
    MqlDateTime dt; TimeToStruct(TimeCurrent(), dt); 
    return (dt.hour >= 22 || dt.hour <= 8); 
}
bool CXAUUSD_MLStrategies::IsSessionOverlap() { return IsLondonSession() && IsNewYorkSession(); }
double CXAUUSD_MLStrategies::GetSessionVolatilityMultiplier() 
{ 
    if(IsSessionOverlap()) return 1.2; 
    if(IsLondonSession() || IsNewYorkSession()) return 1.0; 
    return 0.7; 
}