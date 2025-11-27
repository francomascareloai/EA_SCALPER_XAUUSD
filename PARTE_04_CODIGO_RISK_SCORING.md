# EA_SCALPER_XAUUSD – Multi-Agent Hybrid System
## PARTE 4: Código MQL5 - Risk Manager e Scoring Module

---

## 4.2 CFTMORiskManager.mqh

```mql5
//+------------------------------------------------------------------+
//| filepath: c:\Users\Admin\Documents\EA_SCALPER_XAUUSD\Modules\CFTMORiskManager.mqh
//+------------------------------------------------------------------+
//|                                           CFTMORiskManager.mqh   |
//|                        FTMO-Compliant Risk Management Module     |
//+------------------------------------------------------------------+
#property copyright "TradeDev_Master"
#property strict

#include "CDataStructures.mqh"

//+------------------------------------------------------------------+
//| CFTMORiskManager Class                                            |
//+------------------------------------------------------------------+
class CFTMORiskManager
{
private:
    //--- Configuration
    double m_RiskPerTrade;
    double m_MaxDailyLossPercent;
    double m_MaxTotalLossPercent;
    double m_SoftDailyLossPercent;
    double m_MinLotSize;
    double m_MaxLotSize;
    
    //--- Account tracking
    double m_InitialBalance;
    double m_DailyStartBalance;
    double m_CurrentEquity;
    
    //--- State
    ENUM_RISK_STATE m_RiskState;
    bool   m_TradingBlocked;
    string m_BlockReason;
    
    //--- Dynamic risk multiplier
    double GetRiskMultiplier();
    
public:
    CFTMORiskManager(double riskPerTrade, double maxDailyLoss,
                     double maxTotalLoss, double softDailyLoss,
                     double minLot, double maxLot);
    
    void   Initialize(double balance, double equity);
    void   ResetDaily(double newDayBalance);
    void   Update(double currentEquity);
    
    //--- THE VETO FUNCTION
    bool   CanOpenTrade(double riskPercent, double slPoints, 
                        double &outLotSize, string &outReason);
    
    double CalculateLotSize(double riskPercent, double slPoints);
    
    //--- Getters
    bool   IsTradingAllowed()    { return !m_TradingBlocked; }
    ENUM_RISK_STATE GetRiskState() { return m_RiskState; }
    double GetCurrentDDPercent();
    double GetDailyDDPercent();
    string GetBlockReason()      { return m_BlockReason; }
};

//+------------------------------------------------------------------+
//| Constructor                                                       |
//+------------------------------------------------------------------+
CFTMORiskManager::CFTMORiskManager(double riskPerTrade, double maxDailyLoss,
                                   double maxTotalLoss, double softDailyLoss,
                                   double minLot, double maxLot)
{
    m_RiskPerTrade = riskPerTrade;
    m_MaxDailyLossPercent = maxDailyLoss;
    m_MaxTotalLossPercent = maxTotalLoss;
    m_SoftDailyLossPercent = softDailyLoss;
    m_MinLotSize = minLot;
    m_MaxLotSize = maxLot;
    
    m_RiskState = RISK_NORMAL;
    m_TradingBlocked = false;
    m_BlockReason = "";
}

//+------------------------------------------------------------------+
//| Initialize                                                        |
//+------------------------------------------------------------------+
void CFTMORiskManager::Initialize(double balance, double equity)
{
    m_InitialBalance = balance;
    m_DailyStartBalance = balance;
    m_CurrentEquity = equity;
    
    Print("RiskManager Init | Balance: $", m_InitialBalance);
    Print("Daily Loss Limit: ", m_MaxDailyLossPercent, "% = $", 
          m_DailyStartBalance * m_MaxDailyLossPercent / 100);
    Print("Total Loss Limit: ", m_MaxTotalLossPercent, "% = $",
          m_InitialBalance * m_MaxTotalLossPercent / 100);
}

//+------------------------------------------------------------------+
//| Reset for new day                                                 |
//+------------------------------------------------------------------+
void CFTMORiskManager::ResetDaily(double newDayBalance)
{
    m_DailyStartBalance = newDayBalance;
    
    double totalDD = GetCurrentDDPercent();
    if(totalDD < m_MaxTotalLossPercent - 1.0)
    {
        m_TradingBlocked = false;
        m_BlockReason = "";
        m_RiskState = RISK_NORMAL;
    }
    
    Print("Daily Reset | New balance: $", m_DailyStartBalance);
}

//+------------------------------------------------------------------+
//| Update - Call every tick                                          |
//+------------------------------------------------------------------+
void CFTMORiskManager::Update(double currentEquity)
{
    m_CurrentEquity = currentEquity;
    
    double dailyDD = GetDailyDDPercent();
    double totalDD = GetCurrentDDPercent();
    
    //--- CRITICAL: Max Total Loss
    if(totalDD >= m_MaxTotalLossPercent)
    {
        m_TradingBlocked = true;
        m_BlockReason = StringFormat("MAX TOTAL LOSS: %.2f%% >= %.2f%%",
                                     totalDD, m_MaxTotalLossPercent);
        m_RiskState = RISK_BLOCKED;
        return;
    }
    
    //--- CRITICAL: Max Daily Loss
    if(dailyDD >= m_MaxDailyLossPercent)
    {
        m_TradingBlocked = true;
        m_BlockReason = StringFormat("MAX DAILY LOSS: %.2f%% >= %.2f%%",
                                     dailyDD, m_MaxDailyLossPercent);
        m_RiskState = RISK_BLOCKED;
        return;
    }
    
    //--- Dynamic risk states
    if(dailyDD >= 4.0)
    {
        m_RiskState = RISK_BLOCKED;
        m_TradingBlocked = true;
        m_BlockReason = StringFormat("DD %.2f%% too close to limit", dailyDD);
    }
    else if(dailyDD >= 2.5)
    {
        m_RiskState = RISK_MINIMAL;
        m_TradingBlocked = false;
    }
    else if(dailyDD >= m_SoftDailyLossPercent)
    {
        m_RiskState = RISK_REDUCED;
        m_TradingBlocked = false;
    }
    else
    {
        m_RiskState = RISK_NORMAL;
        m_TradingBlocked = false;
    }
}

//+------------------------------------------------------------------+
//| Get risk multiplier based on state                                |
//+------------------------------------------------------------------+
double CFTMORiskManager::GetRiskMultiplier()
{
    switch(m_RiskState)
    {
        case RISK_NORMAL:   return 1.0;   // 100%
        case RISK_REDUCED:  return 0.5;   // 50%
        case RISK_MINIMAL:  return 0.25;  // 25%
        case RISK_BLOCKED:  return 0.0;   // 0%
        default:            return 1.0;
    }
}

//+------------------------------------------------------------------+
//| THE VETO FUNCTION                                                 |
//+------------------------------------------------------------------+
bool CFTMORiskManager::CanOpenTrade(double riskPercent, double slPoints,
                                    double &outLotSize, string &outReason)
{
    outLotSize = 0;
    outReason = "";
    
    //--- Check blocked
    if(m_TradingBlocked)
    {
        outReason = m_BlockReason;
        return false;
    }
    
    //--- Apply dynamic multiplier
    double adjustedRisk = riskPercent * GetRiskMultiplier();
    
    if(adjustedRisk <= 0)
    {
        outReason = "Risk multiplier zero - blocked";
        return false;
    }
    
    //--- Calculate lot
    double lotSize = CalculateLotSize(adjustedRisk, slPoints);
    
    if(lotSize < m_MinLotSize)
    {
        outReason = StringFormat("Lot %.4f < min %.2f", lotSize, m_MinLotSize);
        return false;
    }
    
    //--- Calculate potential loss
    double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
    double potentialLoss = (slPoints * _Point / tickSize) * tickValue * lotSize;
    
    //--- Check daily projection
    double currentDailyLoss = m_DailyStartBalance - m_CurrentEquity;
    if(currentDailyLoss < 0) currentDailyLoss = 0;
    double projectedDailyDD = ((currentDailyLoss + potentialLoss) / m_DailyStartBalance) * 100;
    
    if(projectedDailyDD >= m_MaxDailyLossPercent - 0.5)
    {
        outReason = StringFormat("Would breach daily: %.2f%% -> %.2f%%",
                                 GetDailyDDPercent(), projectedDailyDD);
        return false;
    }
    
    //--- Check total projection
    double currentTotalLoss = m_InitialBalance - m_CurrentEquity;
    if(currentTotalLoss < 0) currentTotalLoss = 0;
    double projectedTotalDD = ((currentTotalLoss + potentialLoss) / m_InitialBalance) * 100;
    
    if(projectedTotalDD >= m_MaxTotalLossPercent - 1.0)
    {
        outReason = StringFormat("Would breach total: %.2f%% -> %.2f%%",
                                 GetCurrentDDPercent(), projectedTotalDD);
        return false;
    }
    
    //--- APPROVED
    outLotSize = MathMin(lotSize, m_MaxLotSize);
    outReason = StringFormat("OK | Risk: %.2f%% | Lot: %.2f | State: %s",
                             adjustedRisk, outLotSize,
                             EnumToString(m_RiskState));
    
    return true;
}

//+------------------------------------------------------------------+
//| Calculate lot size                                                |
//+------------------------------------------------------------------+
double CFTMORiskManager::CalculateLotSize(double riskPercent, double slPoints)
{
    double accountEquity = m_CurrentEquity;
    double riskAmount = accountEquity * (riskPercent / 100.0);
    
    double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
    double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
    
    if(tickValue == 0 || tickSize == 0)
    {
        Print("ERROR: Cannot get tick value for ", _Symbol);
        return 0;
    }
    
    double slInTicks = (slPoints * _Point) / tickSize;
    double lotSize = riskAmount / (slInTicks * tickValue);
    
    lotSize = MathFloor(lotSize / lotStep) * lotStep;
    lotSize = MathMax(lotSize, m_MinLotSize);
    lotSize = MathMin(lotSize, m_MaxLotSize);
    
    return NormalizeDouble(lotSize, 2);
}

//+------------------------------------------------------------------+
//| Get total drawdown %                                              |
//+------------------------------------------------------------------+
double CFTMORiskManager::GetCurrentDDPercent()
{
    if(m_InitialBalance <= 0) return 0;
    double loss = m_InitialBalance - m_CurrentEquity;
    if(loss < 0) loss = 0;
    return (loss / m_InitialBalance) * 100.0;
}

//+------------------------------------------------------------------+
//| Get daily drawdown %                                              |
//+------------------------------------------------------------------+
double CFTMORiskManager::GetDailyDDPercent()
{
    if(m_DailyStartBalance <= 0) return 0;
    double loss = m_DailyStartBalance - m_CurrentEquity;
    if(loss < 0) loss = 0;
    return (loss / m_DailyStartBalance) * 100.0;
}
```

---

## 4.3 CSignalScoringModule.mqh

```mql5
//+------------------------------------------------------------------+
//| filepath: c:\Users\Admin\Documents\EA_SCALPER_XAUUSD\Modules\CSignalScoringModule.mqh
//+------------------------------------------------------------------+
//|                                       CSignalScoringModule.mqh   |
//|                              Scoring Engine for Trade Signals    |
//+------------------------------------------------------------------+
#property copyright "TradeDev_Master"
#property strict

#include "CDataStructures.mqh"

//+------------------------------------------------------------------+
//| CSignalScoringModule Class                                        |
//+------------------------------------------------------------------+
class CSignalScoringModule
{
private:
    double m_TechWeight;
    double m_FundWeight;
    double m_SentWeight;

public:
    CSignalScoringModule(double techW, double fundW, double sentW);
    
    double ComputeTechScore(bool hasOB, int obStrength,
                            bool hasFVG, int fvgStrength,
                            bool liquiditySwept,
                            int structStrength, bool hasBOS,
                            double atr, ENUM_VOLATILITY_STATE volState);
    
    double ComputeFinalScore(double tech, double fund, double sent);
    string GetScoreBreakdown(double t, double f, double s, double fin);
};

//+------------------------------------------------------------------+
//| Constructor                                                       |
//+------------------------------------------------------------------+
CSignalScoringModule::CSignalScoringModule(double techW, double fundW, double sentW)
{
    double total = techW + fundW + sentW;
    m_TechWeight = techW / total;
    m_FundWeight = fundW / total;
    m_SentWeight = sentW / total;
}

//+------------------------------------------------------------------+
//| Compute Technical Score (0-100)                                   |
//+------------------------------------------------------------------+
double CSignalScoringModule::ComputeTechScore(bool hasOB, int obStrength,
                                              bool hasFVG, int fvgStrength,
                                              bool liquiditySwept,
                                              int structStrength, bool hasBOS,
                                              double atr, ENUM_VOLATILITY_STATE volState)
{
    double score = 0;
    
    //--- ORDER BLOCK (max 25 pts)
    if(hasOB)
        score += (obStrength / 100.0) * 25.0;
    
    //--- FVG (max 20 pts)
    if(hasFVG)
        score += (fvgStrength / 100.0) * 20.0;
    
    //--- LIQUIDITY SWEEP (max 20 pts)
    if(liquiditySwept)
        score += 20.0;
    
    //--- MARKET STRUCTURE (max 25 pts)
    score += (structStrength / 100.0) * 20.0;
    if(hasBOS)
        score += 5.0;
    
    //--- VOLATILITY (max 10 pts)
    switch(volState)
    {
        case VOL_LOW:    score += 3.0;  break;
        case VOL_NORMAL: score += 10.0; break;
        case VOL_HIGH:   score += 5.0;  break;
    }
    
    return MathMax(0, MathMin(100, NormalizeDouble(score, 1)));
}

//+------------------------------------------------------------------+
//| Compute Final Score                                               |
//+------------------------------------------------------------------+
double CSignalScoringModule::ComputeFinalScore(double tech, double fund, double sent)
{
    double finalScore = (tech * m_TechWeight) + 
                        (fund * m_FundWeight) + 
                        (sent * m_SentWeight);
    
    //--- Confluence bonus
    if(tech >= 70 && fund >= 70 && sent >= 70)
        finalScore += 5.0;
    
    //--- Divergence penalty
    double maxS = MathMax(tech, MathMax(fund, sent));
    double minS = MathMin(tech, MathMin(fund, sent));
    double divergence = maxS - minS;
    
    if(divergence > 40)
        finalScore -= 10.0;
    else if(divergence > 25)
        finalScore -= 5.0;
    
    return MathMax(0, MathMin(100, NormalizeDouble(finalScore, 1)));
}

//+------------------------------------------------------------------+
//| Get breakdown string                                              |
//+------------------------------------------------------------------+
string CSignalScoringModule::GetScoreBreakdown(double t, double f, double s, double fin)
{
    return StringFormat("T:%.0f(%.0f%%) F:%.0f(%.0f%%) S:%.0f(%.0f%%) = %.0f",
                        t, m_TechWeight*100, f, m_FundWeight*100,
                        s, m_SentWeight*100, fin);
}
```
