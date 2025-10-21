//+------------------------------------------------------------------+
//|                                        XAUUSD_ML_Risk.mqh       |
//|                                     Copyright 2024, Elite Trading |
//+------------------------------------------------------------------+

#include "XAUUSD_ML_Core.mqh"

//+------------------------------------------------------------------+
//| Intelligent Risk Management Class for XAUUSD                    |
//+------------------------------------------------------------------+
class CXAUUSD_MLRisk
{
private:
    // Risk Configuration
    double            m_baseRiskPercent;
    double            m_maxDailyRisk;
    double            m_maxDrawdownPercent;
    bool              m_ftmoComplianceEnabled;
    
    // Account Monitoring
    double            m_initialBalance;
    double            m_currentEquity;
    double            m_dailyStartEquity;
    double            m_currentDrawdown;
    double            m_maxHistoricalDrawdown;
    
    // Daily Risk Tracking
    double            m_dailyPnL;
    double            m_dailyRiskTaken;
    int               m_dailyTrades;
    datetime          m_lastResetDate;
    
    // Position Sizing
    double            m_dynamicRiskMultiplier;
    double            m_volatilityAdjustment;
    double            m_correlationRiskReduction;
    
    // FTMO Specific
    double            m_ftmoDailyLossLimit;
    double            m_ftmoMaxDrawdown;
    int               m_ftmoMinTradingDays;
    int               m_ftmoTradingDaysCount;
    
    // Emergency Controls
    bool              m_emergencyStop;
    bool              m_tradingHalted;
    string            m_haltReason;

public:
                     CXAUUSD_MLRisk();
                    ~CXAUUSD_MLRisk();
    
    // Configuration
    void             SetBaseRisk(double riskPercent);
    void             SetMaxDailyRisk(double riskPercent);
    void             SetMaxDrawdown(double drawdownPercent);
    void             SetFTMOCompliance(bool enabled);
    
    // Risk Validation
    bool             ValidateRiskLimits();
    bool             ValidateSignal(const STradeSignal &signal);
    bool             CanTakeNewTrade();
    
    // Position Sizing
    double           CalculateOptimalLotSize(const STradeSignal &signal);
    double           CalculateDynamicRisk(const SMarketAnalysis &analysis);
    double           GetVolatilityAdjustedRisk(double baseRisk, double volatility);
    
    // Account Monitoring
    void             UpdateAccountMetrics();
    bool             CheckDrawdownLimits();
    bool             CheckDailyLossLimits();
    
    // FTMO Compliance
    bool             ValidateFTMOLimits();
    void             UpdateFTMOTracking();
    bool             IsFTMOCompliant();
    
    // Emergency Controls
    void             ActivateEmergencyStop(string reason);
    void             ResetEmergencyStop();
    bool             IsEmergencyStopActive();
    
    // Correlation Risk
    double           CalculatePortfolioCorrelation();
    double           GetCorrelationRiskReduction();
    
    // Performance Metrics
    double           GetCurrentDrawdown();
    double           GetDailyPnL();
    double           GetRiskAdjustedReturn();

private:
    // Internal Risk Calculations
    double           CalculatePositionRisk(double lotSize, double stopLossPips);
    double           GetAccountEquity();
    double           GetAccountBalance();
    void             ResetDailyMetrics();
    
    // Volatility Analysis
    double           GetMarketVolatilityMultiplier();
    double           CalculateVolatilityRiskAdjustment(double currentATR);
    
    // Correlation Analysis
    double           GetDXYCorrelation();
    double           GetEURUSDCorrelation();
    double           CalculatePortfolioHeat();
    
    // FTMO Specific Calculations
    bool             CheckFTMODailyLoss();
    bool             CheckFTMOMaxDrawdown();
    bool             CheckFTMOTradingDays();
};

//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CXAUUSD_MLRisk::CXAUUSD_MLRisk()
{
    m_baseRiskPercent = 0.01;           // 1% base risk
    m_maxDailyRisk = 0.02;              // 2% max daily risk
    m_maxDrawdownPercent = 0.03;        // 3% max drawdown
    m_ftmoComplianceEnabled = true;
    
    m_initialBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    m_currentEquity = AccountInfoDouble(ACCOUNT_EQUITY);
    m_dailyStartEquity = m_currentEquity;
    m_currentDrawdown = 0.0;
    m_maxHistoricalDrawdown = 0.0;
    
    m_dailyPnL = 0.0;
    m_dailyRiskTaken = 0.0;
    m_dailyTrades = 0;
    m_lastResetDate = 0;
    
    m_dynamicRiskMultiplier = 1.0;
    m_volatilityAdjustment = 1.0;
    m_correlationRiskReduction = 1.0;
    
    // FTMO Limits (Conservative with 50% safety margin)
    m_ftmoDailyLossLimit = 0.025;       // 2.5% vs FTMO's 5%
    m_ftmoMaxDrawdown = 0.05;           // 5% vs FTMO's 10%
    m_ftmoMinTradingDays = 5;
    m_ftmoTradingDaysCount = 0;
    
    m_emergencyStop = false;
    m_tradingHalted = false;
    m_haltReason = "";
}

//+------------------------------------------------------------------+
//| Validate Risk Limits                                            |
//+------------------------------------------------------------------+
bool CXAUUSD_MLRisk::ValidateRiskLimits()
{
    UpdateAccountMetrics();
    
    // Check emergency stop
    if(m_emergencyStop || m_tradingHalted)
    {
        return false;
    }
    
    // Check drawdown limits
    if(!CheckDrawdownLimits())
    {
        ActivateEmergencyStop("Maximum drawdown exceeded");
        return false;
    }
    
    // Check daily loss limits
    if(!CheckDailyLossLimits())
    {
        ActivateEmergencyStop("Daily loss limit exceeded");
        return false;
    }
    
    // FTMO compliance check
    if(m_ftmoComplianceEnabled && !ValidateFTMOLimits())
    {
        ActivateEmergencyStop("FTMO compliance violation");
        return false;
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Calculate Optimal Lot Size                                      |
//+------------------------------------------------------------------+
double CXAUUSD_MLRisk::CalculateOptimalLotSize(const STradeSignal &signal)
{
    // Get base risk amount
    double accountEquity = GetAccountEquity();
    double baseRiskAmount = accountEquity * m_baseRiskPercent;
    
    // Calculate stop loss in pips
    double stopLossPips = MathAbs(signal.entryPrice - signal.stopLoss) / Point;
    if(stopLossPips <= 0) return 0.0;
    
    // Calculate pip value for XAUUSD
    double pipValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    if(pipValue <= 0) pipValue = 1.0; // Fallback
    
    // Base lot size calculation
    double baseLotSize = baseRiskAmount / (stopLossPips * pipValue);
    
    // Apply dynamic risk adjustments
    double dynamicRisk = CalculateDynamicRisk(signal.analysis);
    double adjustedLotSize = baseLotSize * dynamicRisk;
    
    // Apply volatility adjustment
    double currentATR = iATR(_Symbol, PERIOD_M15, 14);
    double volatilityMultiplier = GetVolatilityAdjustedRisk(1.0, currentATR);
    adjustedLotSize *= volatilityMultiplier;
    
    // Apply correlation risk reduction
    double correlationReduction = GetCorrelationRiskReduction();
    adjustedLotSize *= correlationReduction;
    
    // Apply ML confidence scaling
    adjustedLotSize *= signal.confidence;
    
    // Apply session-based adjustments
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    if(dt.hour >= 22 || dt.hour <= 8) // Asian session
    {
        adjustedLotSize *= 0.5; // Reduce risk during Asian session
    }
    
    // Ensure minimum and maximum lot sizes
    double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    double stepLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
    
    adjustedLotSize = MathMax(minLot, adjustedLotSize);
    adjustedLotSize = MathMin(maxLot, adjustedLotSize);
    
    // Round to step size
    adjustedLotSize = MathRound(adjustedLotSize / stepLot) * stepLot;
    
    // Final validation - ensure position risk doesn't exceed limits
    double positionRisk = CalculatePositionRisk(adjustedLotSize, stopLossPips);
    double maxAllowedRisk = accountEquity * m_baseRiskPercent * 2.0; // 2x base risk max
    
    if(positionRisk > maxAllowedRisk)
    {
        adjustedLotSize = maxAllowedRisk / (stopLossPips * pipValue);
        adjustedLotSize = MathRound(adjustedLotSize / stepLot) * stepLot;
    }
    
    return adjustedLotSize;
}

//+------------------------------------------------------------------+
//| Calculate Dynamic Risk                                           |
//+------------------------------------------------------------------+
double CXAUUSD_MLRisk::CalculateDynamicRisk(const SMarketAnalysis &analysis)
{
    double riskMultiplier = 1.0;
    
    // ML Confidence boost
    if(analysis.confidence > 0.85)
    {
        riskMultiplier *= 1.3; // 30% increase for high confidence
    }
    else if(analysis.confidence < 0.70)
    {
        riskMultiplier *= 0.7; // 30% decrease for low confidence
    }
    
    // Market regime adjustment
    switch(analysis.regime)
    {
        case REGIME_TRENDING_UP:
        case REGIME_TRENDING_DOWN:
            riskMultiplier *= 1.1; // Slightly higher risk in trends
            break;
        case REGIME_RANGING:
            riskMultiplier *= 0.9; // Lower risk in ranging markets
            break;
        case REGIME_VOLATILE:
            riskMultiplier *= 0.8; // Reduced risk in volatile conditions
            break;
        case REGIME_LOW_VOLATILITY:
            riskMultiplier *= 1.2; // Higher risk in low volatility
            break;
    }
    
    // Performance-based adjustment
    if(m_dailyPnL > 0)
    {
        riskMultiplier *= 1.1; // Slight increase when profitable
    }
    else if(m_dailyPnL < -accountEquity * 0.01)
    {
        riskMultiplier *= 0.7; // Significant decrease when losing
    }
    
    // News time reduction
    if(analysis.isNewsTime)
    {
        riskMultiplier *= 0.5; // 50% risk reduction during news
    }
    
    // Ensure reasonable bounds
    riskMultiplier = MathMax(0.3, MathMin(1.5, riskMultiplier));
    
    return riskMultiplier;
}

//+------------------------------------------------------------------+
//| Update Account Metrics                                          |
//+------------------------------------------------------------------+
void CXAUUSD_MLRisk::UpdateAccountMetrics()
{
    // Reset daily metrics if new day
    MqlDateTime current, lastReset;
    TimeToStruct(TimeCurrent(), current);
    TimeToStruct(m_lastResetDate, lastReset);
    
    if(current.day != lastReset.day || m_lastResetDate == 0)
    {
        ResetDailyMetrics();
    }
    
    // Update current metrics
    m_currentEquity = GetAccountEquity();
    
    // Calculate current drawdown
    double peakEquity = MathMax(m_initialBalance, m_dailyStartEquity);
    m_currentDrawdown = (peakEquity - m_currentEquity) / peakEquity;
    m_maxHistoricalDrawdown = MathMax(m_maxHistoricalDrawdown, m_currentDrawdown);
    
    // Calculate daily P&L
    m_dailyPnL = m_currentEquity - m_dailyStartEquity;
    
    // Update FTMO tracking
    if(m_ftmoComplianceEnabled)
    {
        UpdateFTMOTracking();
    }
}

//+------------------------------------------------------------------+
//| Check Drawdown Limits                                           |
//+------------------------------------------------------------------+
bool CXAUUSD_MLRisk::CheckDrawdownLimits()
{
    if(m_currentDrawdown > m_maxDrawdownPercent)
    {
        Print("❌ Maximum drawdown exceeded: ", m_currentDrawdown * 100, "% vs limit ", m_maxDrawdownPercent * 100, "%");
        return false;
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Check Daily Loss Limits                                         |
//+------------------------------------------------------------------+
bool CXAUUSD_MLRisk::CheckDailyLossLimits()
{
    double dailyLossPercent = -m_dailyPnL / m_dailyStartEquity;
    
    if(dailyLossPercent > m_maxDailyRisk)
    {
        Print("❌ Daily loss limit exceeded: ", dailyLossPercent * 100, "% vs limit ", m_maxDailyRisk * 100, "%");
        return false;
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Validate FTMO Limits                                            |
//+------------------------------------------------------------------+
bool CXAUUSD_MLRisk::ValidateFTMOLimits()
{
    if(!m_ftmoComplianceEnabled) return true;
    
    // Check FTMO daily loss
    if(!CheckFTMODailyLoss())
    {
        Print("❌ FTMO daily loss limit violated");
        return false;
    }
    
    // Check FTMO max drawdown
    if(!CheckFTMOMaxDrawdown())
    {
        Print("❌ FTMO max drawdown limit violated");
        return false;
    }
    
    // Check minimum trading days
    if(!CheckFTMOTradingDays())
    {
        Print("❌ FTMO minimum trading days requirement not met");
        return false;
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Get Correlation Risk Reduction                                  |
//+------------------------------------------------------------------+
double CXAUUSD_MLRisk::GetCorrelationRiskReduction()
{
    double portfolioHeat = CalculatePortfolioHeat();
    
    // Reduce risk if portfolio heat is high
    if(portfolioHeat > 0.8)
    {
        return 0.5; // 50% risk reduction
    }
    else if(portfolioHeat > 0.6)
    {
        return 0.7; // 30% risk reduction
    }
    else if(portfolioHeat > 0.4)
    {
        return 0.85; // 15% risk reduction
    }
    
    return 1.0; // No reduction
}

//+------------------------------------------------------------------+
//| Calculate Portfolio Heat                                         |
//+------------------------------------------------------------------+
double CXAUUSD_MLRisk::CalculatePortfolioHeat()
{
    double totalRisk = 0.0;
    double accountEquity = GetAccountEquity();
    
    // Calculate risk from all open positions
    for(int i = 0; i < PositionsTotal(); i++)
    {
        if(PositionSelectByIndex(i))
        {
            double positionSize = PositionGetDouble(POSITION_VOLUME);
            double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
            double currentPrice = PositionGetDouble(POSITION_PRICE_CURRENT);
            double stopLoss = PositionGetDouble(POSITION_SL);
            
            if(stopLoss > 0)
            {
                double riskPips = MathAbs(openPrice - stopLoss) / Point;
                double pipValue = SymbolInfoDouble(PositionGetString(POSITION_SYMBOL), SYMBOL_TRADE_TICK_VALUE);
                double positionRisk = riskPips * pipValue * positionSize;
                totalRisk += positionRisk;
            }
        }
    }
    
    return totalRisk / accountEquity;
}

//+------------------------------------------------------------------+
//| Activate Emergency Stop                                         |
//+------------------------------------------------------------------+
void CXAUUSD_MLRisk::ActivateEmergencyStop(string reason)
{
    m_emergencyStop = true;
    m_tradingHalted = true;
    m_haltReason = reason;
    
    Print("🚨 EMERGENCY STOP ACTIVATED: ", reason);
    
    // Close all open positions if critical
    if(StringFind(reason, "drawdown") >= 0 || StringFind(reason, "loss") >= 0)
    {
        for(int i = PositionsTotal() - 1; i >= 0; i--)
        {
            if(PositionSelectByIndex(i))
            {
                ulong ticket = PositionGetInteger(POSITION_TICKET);
                CTrade trade;
                trade.PositionClose(ticket);
                Print("⚠️ Emergency closure of position: ", ticket);
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Reset Daily Metrics                                             |
//+------------------------------------------------------------------+
void CXAUUSD_MLRisk::ResetDailyMetrics()
{
    m_dailyStartEquity = GetAccountEquity();
    m_dailyPnL = 0.0;
    m_dailyRiskTaken = 0.0;
    m_dailyTrades = 0;
    m_lastResetDate = TimeCurrent();
    
    // Count trading day for FTMO
    if(m_ftmoComplianceEnabled && m_dailyTrades > 0)
    {
        m_ftmoTradingDaysCount++;
    }
    
    Print("📊 Daily metrics reset. New trading day started.");
}

//+------------------------------------------------------------------+
//| Helper Methods Implementation                                    |
//+------------------------------------------------------------------+
double CXAUUSD_MLRisk::GetAccountEquity() { return AccountInfoDouble(ACCOUNT_EQUITY); }
double CXAUUSD_MLRisk::GetAccountBalance() { return AccountInfoDouble(ACCOUNT_BALANCE); }
double CXAUUSD_MLRisk::GetCurrentDrawdown() { return m_currentDrawdown; }
double CXAUUSD_MLRisk::GetDailyPnL() { return m_dailyPnL; }
bool CXAUUSD_MLRisk::IsEmergencyStopActive() { return m_emergencyStop; }

bool CXAUUSD_MLRisk::CheckFTMODailyLoss()
{
    double dailyLossPercent = -m_dailyPnL / m_dailyStartEquity;
    return dailyLossPercent <= m_ftmoDailyLossLimit;
}

bool CXAUUSD_MLRisk::CheckFTMOMaxDrawdown()
{
    return m_currentDrawdown <= m_ftmoMaxDrawdown;
}

bool CXAUUSD_MLRisk::CheckFTMOTradingDays()
{
    return m_ftmoTradingDaysCount >= m_ftmoMinTradingDays;
}

double CXAUUSD_MLRisk::CalculatePositionRisk(double lotSize, double stopLossPips)
{
    double pipValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    return stopLossPips * pipValue * lotSize;
}

double CXAUUSD_MLRisk::GetVolatilityAdjustedRisk(double baseRisk, double volatility)
{
    double avgVolatility = 15.0; // Average XAUUSD volatility in pips
    double volatilityRatio = volatility / avgVolatility;
    
    // Inverse relationship - higher volatility = lower risk
    return baseRisk / MathSqrt(volatilityRatio);
}